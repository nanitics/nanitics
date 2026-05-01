"""Tests for PersistentTraceStore data models."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from nanitics.infrastructure.observability.storage import (
    StoredTraceEvent,
    TraceEventRecord,
    TraceSummaryStats,
)


class TestTraceEventRecord:
    def test_construction(self) -> None:
        record = TraceEventRecord(
            event_type="agent.start",
            level="info",
            trace_id="trace-1",
            span_id="span-1",
            parent_span_id=None,
            payload={"agent_name": "test"},
            sdk_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )
        assert record.event_type == "agent.start"
        assert record.level == "info"
        assert record.parent_span_id is None

    def test_frozen(self) -> None:
        record = TraceEventRecord(
            event_type="agent.start",
            level="info",
            trace_id="t",
            span_id="s",
            parent_span_id=None,
            payload={},
            sdk_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )
        with pytest.raises(ValidationError):
            record.event_type = "agent.complete"

    def test_with_parent_span(self) -> None:
        record = TraceEventRecord(
            event_type="tool.invoke",
            level="debug",
            trace_id="t",
            span_id="s2",
            parent_span_id="s1",
            payload={"tool_name": "search"},
            sdk_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )
        assert record.parent_span_id == "s1"


class TestStoredTraceEvent:
    def test_extends_record_with_id(self) -> None:
        stored = StoredTraceEvent(
            id=42,
            event_type="llm.response",
            level="debug",
            trace_id="t",
            span_id="s",
            parent_span_id=None,
            payload={"model": "claude"},
            sdk_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )
        assert stored.id == 42
        assert stored.event_type == "llm.response"

    def test_inherits_all_record_fields(self) -> None:
        stored = StoredTraceEvent(
            id=1,
            event_type="agent.step",
            level="debug",
            trace_id="t",
            span_id="s",
            parent_span_id="p",
            payload={},
            sdk_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )
        assert isinstance(stored, TraceEventRecord)


class TestTraceSummaryStats:
    def test_construction(self) -> None:
        stats = TraceSummaryStats(
            total_events=100,
            events_by_level={"info": 20, "debug": 50, "verbose": 30},
            llm_calls=10,
            tool_calls=15,
            total_input_tokens=5000,
            total_output_tokens=3000,
            total_duration_ms=12500,
            agent_names=["analyzer", "planner"],
            errors=2,
        )
        assert stats.total_events == 100
        assert stats.events_by_level["info"] == 20
        assert stats.agent_names == ["analyzer", "planner"]

    def test_nullable_duration(self) -> None:
        stats = TraceSummaryStats(
            total_events=0,
            events_by_level={},
            llm_calls=0,
            tool_calls=0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_duration_ms=None,
            agent_names=[],
            errors=0,
        )
        assert stats.total_duration_ms is None

    def test_frozen(self) -> None:
        stats = TraceSummaryStats(
            total_events=1,
            events_by_level={"info": 1},
            llm_calls=0,
            tool_calls=0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_duration_ms=None,
            agent_names=[],
            errors=0,
        )
        with pytest.raises(ValidationError):
            stats.total_events = 2
