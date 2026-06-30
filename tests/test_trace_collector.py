"""Tests for TraceCollector — buffering, flushing, level filtering, queue integration."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from nanitics.infrastructure.observability.collector import (
    MAX_FLUSH_ATTEMPTS,
    TraceCollector,
)
from nanitics.infrastructure.observability.events import (
    AgentStartEvent,
    AgentStepEvent,
    LLMRequestEvent,
    SpanStartEvent,
    WorkingMemoryUpdateEvent,
)
from nanitics.infrastructure.observability.storage import (
    PersistentTraceStore,
    TraceEventRecord,
)


def _make_mock_store() -> AsyncMock:
    """Create a mock PersistentTraceStore."""
    store = AsyncMock(spec=PersistentTraceStore)
    store.save_events_batch = AsyncMock()
    return store


def _make_event(
    event_type: str = "agent.start",
    trace_id: str = "trace-1",
    span_id: str = "span-1",
) -> AgentStartEvent | AgentStepEvent | LLMRequestEvent | SpanStartEvent | WorkingMemoryUpdateEvent:
    """Create a trace event of the given type for testing."""
    ts = datetime.now(UTC)
    if event_type == "agent.start":
        return AgentStartEvent(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=None,
            timestamp=ts,
            agent_name="test-agent",
            task_input="do things",
            tools_available=["tool_a"],
        )
    if event_type == "agent.step":
        return AgentStepEvent(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=None,
            timestamp=ts,
            agent_name="test-agent",
            step_number=1,
        )
    if event_type == "llm.request":
        return LLMRequestEvent(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=None,
            timestamp=ts,
            model_name="claude-3",
            messages=[],
            system_prompt="test",
        )
    if event_type == "span.start":
        return SpanStartEvent(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=None,
            timestamp=ts,
            name="test-span",
        )
    # verbose event
    return WorkingMemoryUpdateEvent(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=None,
        timestamp=ts,
        previous_content=None,
        new_content="updated",
        source="test",
    )


class TestBuffering:
    """Events are buffered and flushed to the store."""

    async def test_handle_buffers_events(self) -> None:
        store = _make_mock_store()
        collector = TraceCollector(store=store, parent_id="run-1", flush_interval=10)

        collector.handle(_make_event())
        collector.handle(_make_event(event_type="llm.request"))

        # Events should be buffered, not yet flushed
        store.save_events_batch.assert_not_called()
        assert len(collector._buffer) == 2
        await collector.close()

    async def test_flush_persists_buffer(self) -> None:
        store = _make_mock_store()
        collector = TraceCollector(store=store, parent_id="run-1", flush_interval=10)

        collector.handle(_make_event())
        collector.handle(_make_event(event_type="llm.request"))

        await collector.flush()

        store.save_events_batch.assert_called_once()
        call_args = store.save_events_batch.call_args
        assert call_args[0][0] == "run-1"
        records = call_args[0][1]
        assert len(records) == 2
        assert all(isinstance(r, TraceEventRecord) for r in records)
        assert collector._buffer == []
        await collector.close()

    async def test_flush_empty_buffer_is_noop(self) -> None:
        store = _make_mock_store()
        collector = TraceCollector(store=store, parent_id="run-1", flush_interval=10)

        await collector.flush()

        store.save_events_batch.assert_not_called()
        await collector.close()


class TestFlushOnError:
    """Events are restored to buffer if flush fails."""

    async def test_flush_failure_restores_buffer(self) -> None:
        store = _make_mock_store()
        store.save_events_batch.side_effect = RuntimeError("db down")
        collector = TraceCollector(store=store, parent_id="run-1", flush_interval=10)

        collector.handle(_make_event())
        collector.handle(_make_event(event_type="llm.request"))

        with pytest.warns(UserWarning, match="flush failed"):
            await collector.flush()

        # Events should be back in the buffer
        assert len(collector._buffer) == 2
        with pytest.warns(UserWarning, match="flush failed"):
            await collector.close()

    async def test_flush_failure_emits_warning(self) -> None:
        store = _make_mock_store()
        store.save_events_batch.side_effect = RuntimeError("db down")
        collector = TraceCollector(store=store, parent_id="run-1", flush_interval=10)

        collector.handle(_make_event())

        with pytest.warns(UserWarning, match="flush failed.*1 consecutive.*db down"):
            await collector.flush()
        with pytest.warns(UserWarning, match="flush failed"):
            await collector.close()

    async def test_consecutive_failure_count_increments_and_resets(self) -> None:
        store = _make_mock_store()
        store.save_events_batch.side_effect = RuntimeError("fail")
        collector = TraceCollector(store=store, parent_id="run-1", flush_interval=10)

        collector.handle(_make_event())
        with pytest.warns(UserWarning, match="1 consecutive"):
            await collector.flush()
        assert collector._consecutive_failures == 1

        collector.handle(_make_event())
        with pytest.warns(UserWarning, match="2 consecutive"):
            await collector.flush()
        assert collector._consecutive_failures == 2

        # Successful flush resets counter
        store.save_events_batch.side_effect = None
        collector._buffer.clear()
        collector.handle(_make_event())
        await collector.flush()
        assert collector._consecutive_failures == 0
        await collector.close()

    async def test_buffer_capped_at_max_buffer_size(self) -> None:
        store = _make_mock_store()
        store.save_events_batch.side_effect = RuntimeError("fail")
        collector = TraceCollector(store=store, parent_id="run-1", flush_interval=10, max_buffer_size=5)

        for _ in range(10):
            collector.handle(_make_event())

        with pytest.warns(UserWarning, match="failed"):
            await collector.flush()

        # Buffer should be capped at max_buffer_size
        assert len(collector._buffer) <= 5
        with pytest.warns(UserWarning, match="failed"):
            await collector.close()


class TestPoisonBatchDrop:
    """A permanently-failing batch is dropped after a bounded number of attempts."""

    async def test_batch_dropped_after_max_consecutive_failures(self) -> None:
        store = _make_mock_store()
        store.save_events_batch.side_effect = RuntimeError("nul byte rejected")
        collector = TraceCollector(store=store, parent_id="run-1", flush_interval=10)

        collector.handle(_make_event())

        # Failures below the threshold re-buffer and retry — nothing is lost yet.
        for _ in range(MAX_FLUSH_ATTEMPTS - 1):
            with pytest.warns(UserWarning, match="flush failed"):
                await collector.flush()
        assert len(collector._buffer) == 1

        # The threshold-th consecutive failure drops the un-storable batch and
        # resets the counter so later events are no longer blocked behind it.
        with pytest.warns(UserWarning, match="dropping 1 un-storable events"):
            await collector.flush()
        assert collector._buffer == []
        assert collector._consecutive_failures == 0

        # Buffer is empty, so close() flushes nothing and emits no warning.
        await collector.close()

    async def test_tracing_resumes_after_poison_batch_dropped(self) -> None:
        store = _make_mock_store()
        store.save_events_batch.side_effect = RuntimeError("nul byte rejected")
        collector = TraceCollector(store=store, parent_id="run-1", flush_interval=10)

        collector.handle(_make_event())
        for _ in range(MAX_FLUSH_ATTEMPTS - 1):
            with pytest.warns(UserWarning, match="flush failed"):
                await collector.flush()
        with pytest.warns(UserWarning, match="dropping"):
            await collector.flush()
        assert collector._buffer == []

        # After the poison batch is gone, a healthy store persists new events.
        store.save_events_batch.side_effect = None
        store.save_events_batch.reset_mock()
        collector.handle(_make_event(event_type="llm.request"))
        await collector.flush()
        store.save_events_batch.assert_called_once()
        assert collector._buffer == []
        await collector.close()


class TestPeriodicFlush:
    """The background flush loop triggers flushes at the configured interval."""

    async def test_periodic_flush_triggers(self) -> None:
        store = _make_mock_store()
        collector = TraceCollector(store=store, parent_id="run-1", flush_interval=0.05)

        collector.handle(_make_event())

        # Wait for at least one flush cycle
        await asyncio.sleep(0.15)

        store.save_events_batch.assert_called()
        assert collector._buffer == []
        await collector.close()


class TestClose:
    """Close performs a final flush and cancels the flush task."""

    async def test_close_flushes_remaining(self) -> None:
        store = _make_mock_store()
        collector = TraceCollector(store=store, parent_id="run-1", flush_interval=10)

        collector.handle(_make_event())
        collector.handle(_make_event(event_type="llm.request"))

        await collector.close()

        store.save_events_batch.assert_called_once()
        assert collector._flush_task is None

    async def test_close_cancels_flush_task(self) -> None:
        store = _make_mock_store()
        collector = TraceCollector(store=store, parent_id="run-1", flush_interval=10)

        collector.handle(_make_event())
        assert collector._flush_task is not None

        await collector.close()
        assert collector._flush_task is None


class TestLevelClassification:
    """Events are classified by level in their TraceEventRecord."""

    async def test_info_event_classified(self) -> None:
        store = _make_mock_store()
        collector = TraceCollector(store=store, parent_id="run-1", flush_interval=10)

        collector.handle(_make_event(event_type="agent.start"))
        await collector.flush()

        records = store.save_events_batch.call_args[0][1]
        assert records[0].level == "info"
        await collector.close()

    async def test_debug_event_classified(self) -> None:
        store = _make_mock_store()
        collector = TraceCollector(store=store, parent_id="run-1", flush_interval=10)

        collector.handle(_make_event(event_type="llm.request"))
        await collector.flush()

        records = store.save_events_batch.call_args[0][1]
        assert records[0].level == "debug"
        await collector.close()

    async def test_verbose_event_classified(self) -> None:
        store = _make_mock_store()
        collector = TraceCollector(store=store, parent_id="run-1", flush_interval=10)

        collector.handle(_make_event(event_type="memory.working.update"))
        await collector.flush()

        records = store.save_events_batch.call_args[0][1]
        assert records[0].level == "verbose"
        await collector.close()


class TestRecordFields:
    """TraceEventRecords carry the correct field values."""

    async def test_record_has_correct_fields(self) -> None:
        store = _make_mock_store()
        collector = TraceCollector(store=store, parent_id="run-1", flush_interval=10)

        event = _make_event(event_type="agent.start", trace_id="t-42", span_id="s-99")
        collector.handle(event)
        await collector.flush()

        record = store.save_events_batch.call_args[0][1][0]
        assert record.event_type == "agent.start"
        assert record.trace_id == "t-42"
        assert record.span_id == "s-99"
        assert record.parent_span_id is None
        assert record.sdk_timestamp == event.timestamp
        assert isinstance(record.payload, dict)
        assert record.payload["event_type"] == "agent.start"
        await collector.close()


class TestQueueIntegration:
    """Events are pushed to the SSE queue when they meet the level threshold."""

    async def test_info_events_pushed_at_info_level(self) -> None:
        store = _make_mock_store()
        queue: asyncio.Queue[dict] = asyncio.Queue()
        collector = TraceCollector(store=store, parent_id="run-1", queue=queue, min_level="info", flush_interval=10)

        collector.handle(_make_event(event_type="agent.start"))

        assert not queue.empty()
        msg = queue.get_nowait()
        assert msg["event_type"] == "trace"
        assert msg["payload"]["sdk_event_type"] == "agent.start"
        assert msg["payload"]["level"] == "info"
        await collector.close()

    async def test_debug_events_excluded_at_info_level(self) -> None:
        store = _make_mock_store()
        queue: asyncio.Queue[dict] = asyncio.Queue()
        collector = TraceCollector(store=store, parent_id="run-1", queue=queue, min_level="info", flush_interval=10)

        collector.handle(_make_event(event_type="llm.request"))

        assert queue.empty()
        await collector.close()

    async def test_debug_events_included_at_debug_level(self) -> None:
        store = _make_mock_store()
        queue: asyncio.Queue[dict] = asyncio.Queue()
        collector = TraceCollector(store=store, parent_id="run-1", queue=queue, min_level="debug", flush_interval=10)

        collector.handle(_make_event(event_type="llm.request"))
        collector.handle(_make_event(event_type="agent.start"))

        assert queue.qsize() == 2
        await collector.close()

    async def test_verbose_events_included_at_verbose_level(self) -> None:
        store = _make_mock_store()
        queue: asyncio.Queue[dict] = asyncio.Queue()
        collector = TraceCollector(store=store, parent_id="run-1", queue=queue, min_level="verbose", flush_interval=10)

        collector.handle(_make_event(event_type="memory.working.update"))
        collector.handle(_make_event(event_type="llm.request"))
        collector.handle(_make_event(event_type="agent.start"))

        assert queue.qsize() == 3
        await collector.close()

    async def test_no_queue_means_no_push(self) -> None:
        store = _make_mock_store()
        collector = TraceCollector(store=store, parent_id="run-1", queue=None, min_level="verbose", flush_interval=10)

        # Should not raise even though no queue is set
        collector.handle(_make_event(event_type="agent.start"))
        await collector.close()

    async def test_queue_payload_contains_event_data(self) -> None:
        store = _make_mock_store()
        queue: asyncio.Queue[dict] = asyncio.Queue()
        collector = TraceCollector(store=store, parent_id="run-1", queue=queue, min_level="info", flush_interval=10)

        event = _make_event(event_type="agent.start", trace_id="t-5", span_id="s-7")
        collector.handle(event)

        msg = queue.get_nowait()
        payload = msg["payload"]
        assert payload["trace_id"] == "t-5"
        assert payload["span_id"] == "s-7"
        # Pydantic JSON mode serializes UTC as "Z", Python isoformat uses "+00:00"
        assert payload["timestamp"] is not None
        await collector.close()
