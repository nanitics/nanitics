from datetime import UTC, datetime, timedelta

from nanitics import (
    InMemoryTraceStore,
    Trace,
    TraceQuery,
    TraceStore,
    TraceSummary,
    Usage,
)
from nanitics.infrastructure.observability.events import AgentStartEvent, LLMResponseEvent

# --- Helpers ---


def _make_trace(
    trace_id: str,
    start: datetime,
    *,
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> Trace:
    return Trace(
        trace_id=trace_id,
        events=[
            AgentStartEvent(
                trace_id=trace_id,
                span_id="span-1",
                timestamp=start,
                agent_name="test-agent",
                task_input="test",
                tools_available=["tool1"],
            ),
            LLMResponseEvent(
                trace_id=trace_id,
                span_id="span-1",
                timestamp=start + timedelta(seconds=1),
                model_name="test-model",
                content="response",
                usage=Usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
                duration_ms=100.0,
            ),
        ],
    )


# --- Data Model Tests ---


class TestTrace:
    def test_construction(self) -> None:
        t = Trace(trace_id="t-1", events=[])
        assert t.trace_id == "t-1"
        assert t.events == []


class TestTraceSummary:
    def test_construction(self) -> None:
        now = datetime.now(UTC)
        ts = TraceSummary(
            trace_id="t-1",
            start_time=now,
            end_time=now + timedelta(seconds=5),
            event_count=3,
            total_input_tokens=100,
            total_output_tokens=50,
        )
        assert ts.trace_id == "t-1"
        assert ts.event_count == 3


class TestTraceQuery:
    def test_defaults(self) -> None:
        q = TraceQuery()
        assert q.start_time is None
        assert q.end_time is None
        assert q.limit == 100
        assert q.offset == 0


# --- InMemoryTraceStore Tests ---


class TestInMemoryTraceStore:
    async def test_save_and_get(self) -> None:
        store = InMemoryTraceStore()
        trace = _make_trace("t-1", datetime.now(UTC))
        await store.save_trace(trace)
        result = await store.get_trace("t-1")
        assert result is not None
        assert result.trace_id == "t-1"
        assert len(result.events) == 2

    async def test_get_not_found(self) -> None:
        store = InMemoryTraceStore()
        result = await store.get_trace("nonexistent")
        assert result is None

    async def test_overwrite(self) -> None:
        store = InMemoryTraceStore()
        now = datetime.now(UTC)
        await store.save_trace(_make_trace("t-1", now, input_tokens=100))
        await store.save_trace(_make_trace("t-1", now, input_tokens=200))
        result = await store.get_trace("t-1")
        assert result is not None
        llm_event = result.events[1]
        assert isinstance(llm_event, LLMResponseEvent)
        assert llm_event.usage.input_tokens == 200

    async def test_query_returns_all(self) -> None:
        store = InMemoryTraceStore()
        now = datetime.now(UTC)
        await store.save_trace(_make_trace("t-1", now))
        await store.save_trace(_make_trace("t-2", now + timedelta(seconds=10)))
        summaries = await store.query_traces(TraceQuery())
        assert len(summaries) == 2

    async def test_query_descending_order(self) -> None:
        store = InMemoryTraceStore()
        now = datetime.now(UTC)
        await store.save_trace(_make_trace("t-old", now))
        await store.save_trace(_make_trace("t-new", now + timedelta(minutes=5)))
        summaries = await store.query_traces(TraceQuery())
        assert summaries[0].trace_id == "t-new"
        assert summaries[1].trace_id == "t-old"

    async def test_query_time_range_filter(self) -> None:
        store = InMemoryTraceStore()
        base = datetime(2025, 1, 1, tzinfo=UTC)
        await store.save_trace(_make_trace("t-before", base))
        await store.save_trace(_make_trace("t-during", base + timedelta(hours=1)))
        await store.save_trace(_make_trace("t-after", base + timedelta(hours=3)))

        summaries = await store.query_traces(
            TraceQuery(
                start_time=base + timedelta(minutes=30),
                end_time=base + timedelta(hours=2),
            )
        )
        assert len(summaries) == 1
        assert summaries[0].trace_id == "t-during"

    async def test_query_limit_and_offset(self) -> None:
        store = InMemoryTraceStore()
        now = datetime.now(UTC)
        for i in range(5):
            await store.save_trace(_make_trace(f"t-{i}", now + timedelta(seconds=i)))

        page1 = await store.query_traces(TraceQuery(limit=2, offset=0))
        assert len(page1) == 2
        assert page1[0].trace_id == "t-4"
        assert page1[1].trace_id == "t-3"

        page2 = await store.query_traces(TraceQuery(limit=2, offset=2))
        assert len(page2) == 2
        assert page2[0].trace_id == "t-2"

    async def test_query_token_aggregation(self) -> None:
        store = InMemoryTraceStore()
        trace = _make_trace("t-1", datetime.now(UTC), input_tokens=150, output_tokens=75)
        await store.save_trace(trace)
        summaries = await store.query_traces(TraceQuery())
        assert summaries[0].total_input_tokens == 150
        assert summaries[0].total_output_tokens == 75

    async def test_query_summary_timestamps(self) -> None:
        store = InMemoryTraceStore()
        start = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)
        trace = _make_trace("t-1", start)
        await store.save_trace(trace)
        summaries = await store.query_traces(TraceQuery())
        assert summaries[0].start_time == start
        assert summaries[0].end_time == start + timedelta(seconds=1)
        assert summaries[0].event_count == 2

    async def test_protocol_conformance(self) -> None:
        store = InMemoryTraceStore()
        assert isinstance(store, TraceStore)

    async def test_query_skips_empty_traces(self) -> None:
        store = InMemoryTraceStore()
        now = datetime.now(UTC)
        await store.save_trace(Trace(trace_id="t-empty", events=[]))
        await store.save_trace(_make_trace("t-normal", now))
        summaries = await store.query_traces(TraceQuery())
        assert len(summaries) == 1
        assert summaries[0].trace_id == "t-normal"

    async def test_query_start_time_filter_only(self) -> None:
        store = InMemoryTraceStore()
        base = datetime(2025, 1, 1, tzinfo=UTC)
        await store.save_trace(_make_trace("t-old", base))
        await store.save_trace(_make_trace("t-new", base + timedelta(hours=2)))
        summaries = await store.query_traces(TraceQuery(start_time=base + timedelta(hours=1)))
        assert len(summaries) == 1
        assert summaries[0].trace_id == "t-new"
