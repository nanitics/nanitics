"""Observatory service + TraceCollector filters + response-model round-trip.

The observatory layer is a collection of components rather than a single
``Observatory`` class:

- ``TraceCollector`` (``nanitics.infrastructure.observability.collector``)
  listens to an ``EventEmitter`` and flushes to a ``PersistentTraceStore``.
- ``InMemoryPersistentTraceStore`` is the in-memory impl of the
  ``PersistentTraceStore`` protocol — exercised here in parallel to
  ``PostgresTraceStore`` (which has its own script,
  ``validation/observability/postgres_trace_store.py``).
- ``ObservatoryService`` composes store primitives into run/hierarchy/agent
  queries used by the router and streaming layers.
- ``nanitics.observatory.models`` — Pydantic response models; they are
  validated here via ``model_dump_json`` / ``model_validate_json`` round-trip
  on the service output.

This script drives a real ``ReActAgent`` end-to-end so the store receives
real trace shapes (``llm.request``, ``llm.response``, ``agent.start``,
``agent.end``, etc.), then pins the primary service and collector query
paths.

Acceptance criteria — ``TraceCollector`` filter APIs (no agent run):
  - Events of different types pushed through the collector are flushed
    to the store and can be retrieved via ``store.query_events``.
  - ``event_types=[...]`` filters match only the requested event types.
  - ``levels=[...]`` filters match only the requested levels
    (``classify_level`` assigns levels from event_type).
  - ``after_id=...`` returns only events with strictly greater id
    (cursor pagination).

Acceptance criteria — ``ObservatoryService`` over a real run:
  - ``register_run`` → ``update_run_status("completed")`` round-trips
    through ``get_run`` and ``list_runs(status="completed")``.
  - ``get_run_summary(run_id).llm_calls >= 1`` (the real Anthropic call
    registered in the trace).
  - ``get_span_tree(trace_id).root`` is populated and every event ends
    up attached somewhere in the tree.
  - ``list_agents(trace_id)`` returns at least one agent whose
    ``agent_name == "observatory-agent"``.
  - ``get_event(event_id)`` returns ``None`` for a non-existent id.
  - Every ``AgentStepEvent`` in the trace has at least one non-None
    among ``thought`` / ``action`` / ``observation`` / ``artifact``.
    Closes observability W2: a terminal no-tool ReAct step must carry
    its final content on ``observation`` so the Observatory UI has
    something legible to render.

Acceptance criteria — models round-trip:
  - ``TraceSummaryResponse`` and ``RunResponse`` survive a JSON
    serialize/deserialize round-trip with field-wise equality (pins
    the frontend-facing contract).
"""

from __future__ import annotations

import pytest

from nanitics import (
    InMemoryEmitter,
    InMemoryPersistentTraceStore,
    ReActAgent,
    TraceCollector,
)
from nanitics.infrastructure import LLMRequestEvent, LLMResponseEvent
from nanitics.infrastructure.observability.events import AgentStepEvent, Usage
from nanitics.observatory import ObservatoryService
from nanitics.observatory.models import (
    RunResponse,
    TraceSummaryResponse,
)
from validation.helpers import (
    make_llm_client,
    run_with_retry,
)

_RUN_ID_REAL = "obs-validation-run-real"
_RUN_ID_FILTERS = "obs-validation-run-filters"


@pytest.mark.quick
async def test_trace_collector_filter_apis() -> None:
    store = InMemoryPersistentTraceStore()
    collector = TraceCollector(store=store, parent_id=_RUN_ID_FILTERS)

    emitter = InMemoryEmitter(trace_id="obs-filters-trace")
    emitter.add_listener(collector.handle)

    # Push two event types through the emitter.
    emitter.emit(
        LLMRequestEvent(
            trace_id=emitter.trace_id,
            span_id=emitter.span_id,
            parent_span_id=emitter.parent_span_id,
            model_name="test-model",
            system_prompt="sp",
            messages=[{"role": "user", "content": "hi"}],
        )
    )
    emitter.emit(
        LLMResponseEvent(
            trace_id=emitter.trace_id,
            span_id=emitter.span_id,
            parent_span_id=emitter.parent_span_id,
            model_name="test-model",
            content="hi back",
            usage=Usage(input_tokens=1, output_tokens=1),
            duration_ms=1.0,
        )
    )

    await collector.flush()

    # Unfiltered retrieval sees both events.
    all_events = await store.query_events(_RUN_ID_FILTERS, limit=100)
    assert len(all_events) == 2, f"Expected 2 events, got {len(all_events)}: {all_events!r}"

    # event_types filter.
    request_only = await store.query_events(_RUN_ID_FILTERS, event_types=["llm.request"], limit=100)
    assert len(request_only) == 1, f"event_types filter should return exactly one event; got {request_only!r}"
    assert request_only[0].event_type == "llm.request", (
        f"event_types filter should select only llm.request; got {request_only!r}"
    )

    # levels filter — both events should classify identically so pass a
    # wrong level and assert zero matches, then the correct level.
    wrong_level = await store.query_events(_RUN_ID_FILTERS, levels=["error"], limit=100)
    assert wrong_level == [], f"levels=['error'] filter must exclude non-error events; got {wrong_level!r}"
    correct_level = await store.query_events(_RUN_ID_FILTERS, levels=[all_events[0].level], limit=100)
    assert correct_level, f"levels=[{all_events[0].level!r}] must match at least the first event."

    # after_id cursor: strictly greater than given id.
    first_id = all_events[0].id
    tail = await store.query_events(_RUN_ID_FILTERS, after_id=first_id, limit=100)
    assert all(e.id > first_id for e in tail), (
        f"after_id={first_id} must return only events with id > {first_id}; got {[e.id for e in tail]}."
    )

    await collector.close()


@pytest.mark.quick
async def test_observatory_service_over_real_run(
    traced_emitter: InMemoryEmitter,
) -> None:
    store = InMemoryPersistentTraceStore()
    collector = TraceCollector(store=store, parent_id=_RUN_ID_REAL)
    traced_emitter.add_listener(collector.handle)

    service = ObservatoryService(store)
    await service.register_run(
        _RUN_ID_REAL,
        trace_id=traced_emitter.trace_id,
        metadata={"purpose": "observatory-validation"},
    )

    agent = ReActAgent(
        name="observatory-agent",
        llm_client=make_llm_client("anthropic"),
        emitter=traced_emitter,
        system_prompt="Reply with a single short word.",
        tools=[],
        max_iterations=1,
    )
    await run_with_retry(lambda: agent.run("Say OK."), max_attempts=2)

    await service.update_run_status(_RUN_ID_REAL, "completed")
    await collector.flush()
    await collector.close()

    # --- Run lifecycle round-trip ---
    run_detail = await service.get_run(_RUN_ID_REAL)
    assert run_detail is not None, "Registered run must be retrievable."
    assert run_detail.run.status == "completed"
    assert run_detail.run.metadata == {"purpose": "observatory-validation"}

    run_listing = await service.list_runs(status="completed")
    listed_ids = {item.run.id for item in run_listing.runs}
    assert _RUN_ID_REAL in listed_ids, (
        f"list_runs(status='completed') must include {_RUN_ID_REAL!r}; got {listed_ids!r}"
    )

    # --- Summary: real LLM call recorded ---
    summary = await service.get_run_summary(_RUN_ID_REAL)
    assert summary.llm_calls >= 1, (
        f"Real Anthropic round-trip must register at least one llm_call; got {summary.llm_calls}."
    )
    assert summary.total_events >= summary.llm_calls * 2, (
        "Each llm_call produces at least a request+response pair; "
        f"got total_events={summary.total_events} llm_calls={summary.llm_calls}."
    )

    # --- Span tree: all events placed in the tree ---
    tree = await service.get_span_tree(traced_emitter.trace_id)
    assert tree.root is not None
    total_in_tree = _count_tree_events(tree.root)
    assert total_in_tree == summary.total_events, (
        f"Span tree must hold every recorded event; tree={total_in_tree} summary={summary.total_events}."
    )

    # --- Agent listing ---
    agent_list = await service.list_agents(traced_emitter.trace_id)
    agent_names = {a.agent_name for a in agent_list.agents}
    assert "observatory-agent" in agent_names, f"list_agents must surface 'observatory-agent'; got {agent_names!r}"

    # --- Negative: missing event id returns None, not a default payload ---
    missing = await service.get_event(event_id=10**9)
    assert missing is None, f"get_event on a non-existent id must return None (never mask failures); got {missing!r}"

    # --- Agent-step payload fidelity (observability W2 regression guard) ---
    # On every AgentStepEvent in the trace, at least one of thought, action,
    # observation, or artifact must be non-None. A terminal no-tool ReAct
    # step must populate observation from the model's final content so the
    # Observatory UI and trace analyzers have something legible to show.
    step_events = [e for e in traced_emitter.events if isinstance(e, AgentStepEvent)]
    assert step_events, "Expected at least one AgentStepEvent from the real run."
    for step in step_events:
        has_content = any(
            getattr(step, field) is not None for field in ("thought", "action", "observation", "artifact")
        )
        assert has_content, (
            f"AgentStepEvent id={step.event_id} step_number={step.step_number} agent={step.agent_name} "
            f"has all of thought/action/observation/artifact=None; regression of observability W2 "
            f"(terminal no-tool step must carry final content on observation)."
        )


def _count_tree_events(node) -> int:
    return len(node.events) + sum(_count_tree_events(c) for c in node.children)


@pytest.mark.quick
def test_observatory_models_roundtrip() -> None:
    run_payload = RunResponse(
        id="r1",
        trace_id="t1",
        status="completed",
        started_at="2026-04-14T00:00:00+00:00",
        completed_at="2026-04-14T00:00:05+00:00",
        metadata={"k": "v"},
        error=None,
        result="done",
    )
    restored_run = RunResponse.model_validate_json(run_payload.model_dump_json())
    assert restored_run == run_payload, (
        f"RunResponse must survive JSON round-trip; got {restored_run!r} != {run_payload!r}"
    )

    summary_payload = TraceSummaryResponse(
        total_events=7,
        events_by_level={"info": 5, "debug": 2},
        llm_calls=2,
        tool_calls=1,
        total_input_tokens=100,
        total_output_tokens=40,
        total_duration_ms=350,
        agent_names=["a", "b"],
        errors=0,
        cache_creation_tokens=0,
        cache_read_tokens=0,
    )
    restored_summary = TraceSummaryResponse.model_validate_json(summary_payload.model_dump_json())
    assert restored_summary == summary_payload
