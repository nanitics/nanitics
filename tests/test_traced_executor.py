"""Tests for TracedExecutor — run lifecycle management with event persistence."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from nanitics.composition.durability.models import SuspensionInfo
from nanitics.composition.durability.suspension import SuspendExecution
from nanitics.infrastructure import MockLLMClient
from nanitics.strategies import ReasoningAgent
from nanitics.tracing import (
    EventEmitter,
    InMemoryPersistentTraceStore,
    RunResult,
    TerminationReason,
    TracedExecutor,
)
from tests.testing_helpers import make_response


async def _successful_fn(emitter: EventEmitter, run_id: str) -> str:
    """Minimal callable that emits an event and returns."""
    del run_id  # unused in this factory
    agent = ReasoningAgent(
        name="test-agent",
        llm_client=MockLLMClient(responses=[make_response("done")]),
        emitter=emitter,
        system_prompt="test",
    )
    result = await agent.run("hello")
    return result.output


async def _failing_fn(emitter: EventEmitter, run_id: str) -> str:
    """Callable that emits events then raises."""
    del run_id  # unused in this factory
    agent = ReasoningAgent(
        name="test-agent",
        llm_client=MockLLMClient(responses=[]),
        emitter=emitter,
        system_prompt="test",
    )
    result = await agent.run("hello")
    return result.output


async def _suspending_fn(emitter: EventEmitter, run_id: str) -> str:
    """Callable that emits an event then suspends."""
    del run_id  # unused in this factory
    with emitter.span("before-suspend"):
        pass
    raise SuspendExecution(
        suspension_info=SuspensionInfo(
            suspension_id="sus-1",
            request_id="req-1",
            request_type="human_input",
            prompt="Need input",
        ),
    )


@pytest.fixture
def store() -> InMemoryPersistentTraceStore:
    return InMemoryPersistentTraceStore()


@pytest.fixture
def executor(store: InMemoryPersistentTraceStore) -> TracedExecutor:
    return TracedExecutor(store)


async def test_successful_execution(executor: TracedExecutor, store: InMemoryPersistentTraceStore) -> None:
    run_id, result = await executor.execute(_successful_fn)

    assert result == "done"
    assert isinstance(run_id, str)

    run = await store.get_run(run_id)
    assert run is not None
    assert run.status == "completed"
    assert run.result is not None
    assert run.result.output == "done"

    events = await store.query_events(run_id)
    assert len(events) > 0


async def test_failed_execution_persists_events(executor: TracedExecutor, store: InMemoryPersistentTraceStore) -> None:
    with pytest.raises(ValueError, match="no more scripted responses"):
        await executor.execute(_failing_fn)

    runs = await store.list_runs(status="failed")
    assert len(runs) == 1
    run = runs[0]
    assert run.status == "failed"
    assert run.error is not None

    # Events from before the failure should be persisted
    events = await store.query_events(run.id)
    assert len(events) > 0


async def test_suspended_execution(executor: TracedExecutor, store: InMemoryPersistentTraceStore) -> None:
    with pytest.raises(SuspendExecution):
        await executor.execute(_suspending_fn)

    runs = await store.list_runs(status="suspended")
    assert len(runs) == 1
    run = runs[0]
    assert run.status == "suspended"

    events = await store.query_events(run.id)
    assert len(events) > 0


async def test_metadata_passed_through(executor: TracedExecutor, store: InMemoryPersistentTraceStore) -> None:
    metadata = {"agent": "extraction", "doc_id": "abc-123"}
    run_id, _ = await executor.execute(_successful_fn, metadata=metadata)

    run = await store.get_run(run_id)
    assert run is not None
    assert run.metadata["agent"] == "extraction"
    assert run.metadata["doc_id"] == "abc-123"


async def test_queue_receives_events(executor: TracedExecutor, store: InMemoryPersistentTraceStore) -> None:
    queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    _run_id, _ = await executor.execute(_successful_fn, queue=queue)

    assert not queue.empty()
    event = queue.get_nowait()
    assert event["event_type"] == "trace"
    assert "payload" in event


async def test_default_metadata_is_empty(executor: TracedExecutor, store: InMemoryPersistentTraceStore) -> None:
    run_id, _ = await executor.execute(_successful_fn)

    run = await store.get_run(run_id)
    assert run is not None
    assert run.metadata == {}


async def test_string_result_maps_to_output(executor: TracedExecutor, store: InMemoryPersistentTraceStore) -> None:
    async def _string_fn(emitter: EventEmitter, run_id: str) -> str:
        del emitter, run_id
        return "plain-result"

    run_id, result = await executor.execute(_string_fn)

    assert result == "plain-result"
    run = await store.get_run(run_id)
    assert run is not None
    assert run.result == RunResult(output="plain-result")


async def test_structured_result_extracts_known_fields(
    executor: TracedExecutor, store: InMemoryPersistentTraceStore
) -> None:
    from pydantic import BaseModel

    class StrategyResult(BaseModel):
        output: str
        termination_reason: str
        total_steps: int
        extra_field: str

    async def _structured_fn(emitter: EventEmitter, run_id: str) -> StrategyResult:
        del emitter, run_id
        return StrategyResult(
            output="answer", termination_reason="iteration_limit", total_steps=7, extra_field="ignored"
        )

    run_id, _ = await executor.execute(_structured_fn)

    run = await store.get_run(run_id)
    assert run is not None
    assert run.result == RunResult(
        output="answer",
        termination_reason=TerminationReason.ITERATION_LIMIT,
        termination_reason_raw=None,
        total_steps=7,
    )


async def test_unknown_termination_lands_in_other(
    executor: TracedExecutor, store: InMemoryPersistentTraceStore
) -> None:
    async def _custom_fn(emitter: EventEmitter, run_id: str) -> dict[str, object]:
        del emitter, run_id
        return {"output": "x", "termination_reason": "novel_reason", "total_steps": 2}

    run_id, _ = await executor.execute(_custom_fn)

    run = await store.get_run(run_id)
    assert run is not None
    assert run.result is not None
    assert run.result.termination_reason is TerminationReason.OTHER
    assert run.result.termination_reason_raw == "novel_reason"


async def test_dict_without_known_fields_yields_empty_result(
    executor: TracedExecutor, store: InMemoryPersistentTraceStore
) -> None:
    async def _opaque_dict_fn(emitter: EventEmitter, run_id: str) -> dict[str, int]:
        del emitter, run_id
        return {"a": 1, "b": 2}

    run_id, _ = await executor.execute(_opaque_dict_fn)

    run = await store.get_run(run_id)
    assert run is not None
    assert run.result == RunResult()


async def test_object_without_known_fields_yields_empty_result(
    executor: TracedExecutor, store: InMemoryPersistentTraceStore
) -> None:
    class Opaque:
        pass

    async def _opaque_fn(emitter: EventEmitter, run_id: str) -> Opaque:
        del emitter, run_id
        return Opaque()

    run_id, _ = await executor.execute(_opaque_fn)

    run = await store.get_run(run_id)
    assert run is not None
    assert run.result == RunResult()


async def test_none_result_yields_empty_result(executor: TracedExecutor, store: InMemoryPersistentTraceStore) -> None:
    async def _none_fn(emitter: EventEmitter, run_id: str) -> None:
        del emitter, run_id

    run_id, _ = await executor.execute(_none_fn)

    run = await store.get_run(run_id)
    assert run is not None
    assert run.result == RunResult()


async def test_run_result_passthrough(executor: TracedExecutor, store: InMemoryPersistentTraceStore) -> None:
    """A factory that already returns a ``RunResult`` is stored verbatim."""

    async def _runresult_fn(emitter: EventEmitter, run_id: str) -> RunResult:
        del emitter, run_id
        return RunResult(output="explicit", termination_reason=TerminationReason.COMPLETED, total_steps=3)

    run_id, _ = await executor.execute(_runresult_fn)

    run = await store.get_run(run_id)
    assert run is not None
    assert run.result == RunResult(output="explicit", termination_reason=TerminationReason.COMPLETED, total_steps=3)


async def test_non_string_field_values_are_ignored(
    executor: TracedExecutor, store: InMemoryPersistentTraceStore
) -> None:
    """Non-string ``output`` / non-int ``total_steps`` are not coerced; they become ``None``."""

    async def _bad_types_fn(emitter: EventEmitter, run_id: str) -> dict[str, object]:
        del emitter, run_id
        return {"output": 123, "total_steps": True, "termination_reason": None}

    run_id, _ = await executor.execute(_bad_types_fn)

    run = await store.get_run(run_id)
    assert run is not None
    assert run.result == RunResult()


async def test_factory_receives_same_run_id_as_returned(
    executor: TracedExecutor, store: InMemoryPersistentTraceStore
) -> None:
    """The ``run_id`` the factory sees equals the ``run_id`` returned.

    This invariant is the whole point of passing ``run_id`` into the
    factory — callers that key external durable state on the Observatory
    ``run_id`` before ``execute`` returns must be able to trust the
    in-factory value without inferring it from the emitter's
    ``trace_id``.
    """
    captured: dict[str, str] = {}

    async def _factory(emitter: EventEmitter, run_id: str) -> str:
        del emitter
        captured["run_id"] = run_id
        return "ok"

    returned_run_id, result = await executor.execute(_factory)

    assert result == "ok"
    assert captured["run_id"] == returned_run_id
    # Surface a second signal: the run record the executor wrote is
    # keyed on that same identifier.
    run = await store.get_run(returned_run_id)
    assert run is not None
    assert run.id == returned_run_id


async def test_caller_supplied_run_id_is_honored(executor: TracedExecutor, store: InMemoryPersistentTraceStore) -> None:
    """A caller-supplied ``run_id`` is honored end-to-end.

    The factory sees the supplied id, the return tuple matches, the run
    record is keyed on it, and at least one persisted event is parented
    under it. Pinning all four signals together is what makes the
    HTTP-boundary fire-and-stream pattern (route returns ``202 {run_id}``
    before scheduling the executor) safe to build on.
    """
    captured: dict[str, str] = {}

    async def _factory(emitter: EventEmitter, run_id: str) -> str:
        captured["run_id"] = run_id
        with emitter.span("caller-supplied-id-marker"):
            pass
        return "ok"

    returned_run_id, result = await executor.execute(_factory, run_id="caller-id")

    assert result == "ok"
    assert captured["run_id"] == "caller-id"
    assert returned_run_id == "caller-id"

    run = await store.get_run("caller-id")
    assert run is not None
    assert run.id == "caller-id"

    events = await store.query_events("caller-id")
    assert len(events) > 0


async def test_omitted_run_id_falls_back_to_generated(
    executor: TracedExecutor, store: InMemoryPersistentTraceStore
) -> None:
    """Omitting ``run_id`` preserves the existing UUID-generated behavior.

    Pins the absence-of-kwarg path with a direct test so a future
    regression that, e.g., introduces a non-UUID default is caught here
    rather than only by indirect signal from the existing tests.
    """

    async def _factory(emitter: EventEmitter, run_id: str) -> str:
        del emitter, run_id
        return "ok"

    returned_run_id, _ = await executor.execute(_factory)

    assert isinstance(returned_run_id, str)
    # ``uuid.UUID`` raises ``ValueError`` on a non-UUID string; letting it
    # propagate is the assertion.
    uuid.UUID(returned_run_id)
