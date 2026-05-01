"""Tests for :func:`trace_events_from_stored` — SDK helper that turns
``list[StoredTraceEvent]`` rows (what :class:`PersistentTraceStore` returns)
back into validated ``list[TraceEvent]`` (what external trace-consumer tools
ingest). The retrospective self-improver runner under
``docker/full-stack/self_improver/`` motivates this helper; every adopter
building a trace-as-data tool would otherwise re-discover the same
conversion.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import pytest

from nanitics import trace_events_from_stored
from nanitics.infrastructure.observability.events import (
    AgentCompleteEvent,
    AgentStartEvent,
    AgentStepEvent,
    LLMRequestEvent,
    LLMResponseEvent,
    RunCompleteEvent,
    RunStartEvent,
    SpanEndEvent,
    SpanStartEvent,
    ToolInfo,
    ToolInvokeEvent,
    ToolResultEvent,
    TraceEvent,
    Usage,
)
from nanitics.infrastructure.observability.storage import (
    MalformedStoredEventError,
    StoredTraceEvent,
)

_TRACE_ID = "trace-round-trip"
_SPAN_ID = "span-root"
_TS = datetime(2026, 4, 16, 12, 0, 0, tzinfo=UTC)


def _wrap(event: TraceEvent, *, row_id: int = 1) -> StoredTraceEvent:
    """Wrap a ``TraceEvent`` as a ``StoredTraceEvent`` the way the
    collector's write path serialises events.
    """
    payload = event.model_dump(mode="json")
    return StoredTraceEvent(
        id=row_id,
        event_type=event.event_type,
        level="info",
        trace_id=event.trace_id,
        span_id=event.span_id,
        parent_span_id=event.parent_span_id,
        payload=payload,
        sdk_timestamp=event.timestamp,
    )


def _agent_start() -> AgentStartEvent:
    return AgentStartEvent(
        trace_id=_TRACE_ID,
        span_id=_SPAN_ID,
        timestamp=_TS,
        agent_name="task-agent",
        task_input="read the docs",
        model_name="claude-haiku",
        tools_available=["list_bundled_docs", "read_bundled_doc"],
        tool_schemas=[
            ToolInfo(name="list_bundled_docs", description="Lists available documents."),
        ],
    )


def _agent_step() -> AgentStepEvent:
    return AgentStepEvent(
        trace_id=_TRACE_ID,
        span_id=_SPAN_ID,
        timestamp=_TS,
        agent_name="task-agent",
        step_number=1,
        thought="list the documents",
        action="list_bundled_docs",
        observation="[01-overview.md, 02-events-and-storage.md]",
    )


def _agent_complete() -> AgentCompleteEvent:
    return AgentCompleteEvent(
        trace_id=_TRACE_ID,
        span_id=_SPAN_ID,
        timestamp=_TS,
        agent_name="task-agent",
        output="Summary: see 01-overview.md",
        total_steps=3,
        termination_reason="final_answer",
    )


def _tool_invoke() -> ToolInvokeEvent:
    return ToolInvokeEvent(
        trace_id=_TRACE_ID,
        span_id=_SPAN_ID,
        timestamp=_TS,
        tool_call_id="call-1",
        tool_name="read_bundled_doc",
        parameters={"filename": "01-overview.md"},
    )


def _tool_result() -> ToolResultEvent:
    return ToolResultEvent(
        trace_id=_TRACE_ID,
        span_id=_SPAN_ID,
        timestamp=_TS,
        tool_call_id="call-1",
        tool_name="read_bundled_doc",
        result="# Overview\n...",
        success=True,
        duration_ms=12.5,
    )


def _llm_request() -> LLMRequestEvent:
    return LLMRequestEvent(
        trace_id=_TRACE_ID,
        span_id=_SPAN_ID,
        timestamp=_TS,
        model_name="claude-haiku",
        system_prompt="You are a task agent.",
        messages=[{"role": "user", "content": "summarise the docs"}],
    )


def _llm_response() -> LLMResponseEvent:
    return LLMResponseEvent(
        trace_id=_TRACE_ID,
        span_id=_SPAN_ID,
        timestamp=_TS,
        model_name="claude-haiku",
        content="here is the summary",
        tool_calls=None,
        usage=Usage(input_tokens=120, output_tokens=50),
        duration_ms=200.5,
    )


def _span_start() -> SpanStartEvent:
    return SpanStartEvent(
        trace_id=_TRACE_ID,
        span_id="span-child",
        parent_span_id=_SPAN_ID,
        timestamp=_TS,
        name="critic.analyze",
    )


def _span_end() -> SpanEndEvent:
    return SpanEndEvent(
        trace_id=_TRACE_ID,
        span_id="span-child",
        parent_span_id=_SPAN_ID,
        timestamp=_TS,
        name="critic.analyze",
        duration_ms=42.0,
    )


def _run_start() -> RunStartEvent:
    return RunStartEvent(
        trace_id=_TRACE_ID,
        span_id=_SPAN_ID,
        timestamp=_TS,
        run_id="run-1",
        metadata={"runner": "self-improver"},
    )


def _run_complete() -> RunCompleteEvent:
    return RunCompleteEvent(
        trace_id=_TRACE_ID,
        span_id=_SPAN_ID,
        timestamp=_TS,
        run_id="run-1",
        duration_ms=150,
    )


class TestRoundtrip:
    """Round-trip equality across every trace-event variant external consumers read."""

    def test_roundtrip_agent_start_event(self) -> None:
        event = _agent_start()
        [result] = trace_events_from_stored([_wrap(event)])
        assert isinstance(result, AgentStartEvent)
        assert result == event

    def test_roundtrip_tool_invoke_event(self) -> None:
        event = _tool_invoke()
        [result] = trace_events_from_stored([_wrap(event)])
        assert isinstance(result, ToolInvokeEvent)
        assert result == event

    def test_roundtrip_llm_response_event(self) -> None:
        event = _llm_response()
        [result] = trace_events_from_stored([_wrap(event)])
        assert isinstance(result, LLMResponseEvent)
        assert result == event


class TestFullVariantSample:
    """Parametrised coverage of every variant."""

    @pytest.mark.parametrize(
        "factory",
        [
            _agent_start,
            _agent_step,
            _agent_complete,
            _tool_invoke,
            _tool_result,
            _llm_request,
            _llm_response,
            _span_start,
            _span_end,
            _run_start,
            _run_complete,
        ],
    )
    def test_roundtrip_full_variant_sample(self, factory: Any) -> None:
        event = factory()
        [result] = trace_events_from_stored([_wrap(event)])
        assert result == event


class TestOrderingAndIterables:
    def test_empty_iterable_returns_empty_list(self) -> None:
        assert trace_events_from_stored([]) == []

    def test_generator_input(self) -> None:
        events = [_agent_start(), _tool_invoke()]
        stored = [_wrap(e, row_id=idx + 1) for idx, e in enumerate(events)]

        def _gen() -> Iterable[StoredTraceEvent]:
            yield from stored

        result = trace_events_from_stored(_gen())
        assert result == events

    def test_preserves_input_order(self) -> None:
        # Deliberately out-of-order: tool.result first, agent.start second.
        events = [_tool_result(), _agent_start(), _llm_response()]
        stored = [_wrap(e, row_id=idx + 1) for idx, e in enumerate(events)]
        result = trace_events_from_stored(stored)
        assert [e.event_type for e in result] == [
            "tool.result",
            "agent.start",
            "llm.response",
        ]


class TestMixedVariantInput:
    def test_mixed_variant_input_preserves_types(self) -> None:
        events = [_agent_start(), _tool_invoke(), _llm_response(), _agent_complete()]
        stored = [_wrap(e, row_id=idx + 1) for idx, e in enumerate(events)]
        result = trace_events_from_stored(stored)
        assert isinstance(result[0], AgentStartEvent)
        assert isinstance(result[1], ToolInvokeEvent)
        assert isinstance(result[2], LLMResponseEvent)
        assert isinstance(result[3], AgentCompleteEvent)


class TestMalformedPayload:
    def test_malformed_payload_raises_with_row_id(self) -> None:
        # Missing required field: AgentStartEvent lacks ``task_input``.
        bad = StoredTraceEvent(
            id=7,
            event_type="agent.start",
            level="info",
            trace_id=_TRACE_ID,
            span_id=_SPAN_ID,
            parent_span_id=None,
            payload={
                "event_type": "agent.start",
                "trace_id": _TRACE_ID,
                "span_id": _SPAN_ID,
                # ``task_input`` and ``agent_name`` deliberately absent.
                "tools_available": [],
            },
            sdk_timestamp=_TS,
        )
        with pytest.raises(MalformedStoredEventError) as excinfo:
            trace_events_from_stored([bad])

        err = excinfo.value
        assert err.row_id == 7
        assert err.event_type == "agent.start"
        assert "7" in str(err)
        assert "agent.start" in str(err)
        assert err.reason  # non-empty validator message

    def test_malformed_payload_is_value_error(self) -> None:
        bad = StoredTraceEvent(
            id=99,
            event_type="agent.start",
            level="info",
            trace_id=_TRACE_ID,
            span_id=_SPAN_ID,
            parent_span_id=None,
            payload={"event_type": "agent.start"},
            sdk_timestamp=_TS,
        )
        # Broad ``except ValueError`` should also catch external parsers'
        # ``MalformedTraceError`` — both are ``ValueError`` subclasses.
        with pytest.raises(ValueError):
            trace_events_from_stored([bad])

    def test_malformed_payload_never_silently_skipped(self) -> None:
        good = _agent_start()
        bad = StoredTraceEvent(
            id=2,
            event_type="agent.start",
            level="info",
            trace_id=_TRACE_ID,
            span_id=_SPAN_ID,
            parent_span_id=None,
            payload={"event_type": "agent.start"},
            sdk_timestamp=_TS,
        )
        # A malformed row in the middle aborts the whole call — the
        # helper must not swallow failures.
        with pytest.raises(MalformedStoredEventError):
            trace_events_from_stored([_wrap(good, row_id=1), bad])


class TestTopLevelImport:
    def test_trace_events_from_stored_is_top_level_exported(self) -> None:
        # Smoke check — the helper must be reachable from the top-level
        # package as part of the public re-export contract.
        import nanitics

        assert hasattr(nanitics, "trace_events_from_stored")
        assert "trace_events_from_stored" in nanitics.__all__
        assert "MalformedStoredEventError" in nanitics.__all__
