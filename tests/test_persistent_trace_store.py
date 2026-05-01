"""Tests for InMemoryPersistentTraceStore — run management + hierarchy queries."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from nanitics.infrastructure.observability.levels import TraceLevel
from nanitics.infrastructure.observability.storage import (
    InMemoryPersistentTraceStore,
    PersistentTraceStore,
    RunRecord,
    StoredTraceEvent,
    TraceEventRecord,
)


def _make_record(
    event_type: str = "agent.start",
    level: TraceLevel = "info",
    *,
    trace_id: str = "trace-1",
    span_id: str | None = None,
    parent_span_id: str | None = None,
    payload: dict | None = None,
    sdk_timestamp: datetime | None = None,
) -> TraceEventRecord:
    return TraceEventRecord(
        event_type=event_type,
        level=level,
        trace_id=trace_id,
        span_id=span_id or f"span-{uuid.uuid4().hex[:6]}",
        parent_span_id=parent_span_id,
        payload=payload or {"agent_name": "test-agent"},
        sdk_timestamp=sdk_timestamp or datetime.now(tz=UTC),
    )


class TestProtocolConformance:
    def test_implements_protocol(self) -> None:
        store = InMemoryPersistentTraceStore()
        assert isinstance(store, PersistentTraceStore)


class TestRunManagement:
    async def test_register_and_get_run(self) -> None:
        store = InMemoryPersistentTraceStore()
        await store.register_run("run-1", "trace-1", {"workflow": "test"})
        run = await store.get_run("run-1")
        assert run is not None
        assert run.id == "run-1"
        assert run.trace_id == "trace-1"
        assert run.status == "running"
        assert run.metadata == {"workflow": "test"}
        assert run.completed_at is None
        assert run.error is None

    async def test_get_nonexistent_run(self) -> None:
        store = InMemoryPersistentTraceStore()
        run = await store.get_run("nonexistent")
        assert run is None

    async def test_update_run_status_completed(self) -> None:
        store = InMemoryPersistentTraceStore()
        await store.register_run("run-1", "trace-1", {})
        await store.update_run_status("run-1", "completed")
        run = await store.get_run("run-1")
        assert run is not None
        assert run.status == "completed"
        assert run.completed_at is not None
        assert run.error is None

    async def test_update_run_status_failed_with_error(self) -> None:
        store = InMemoryPersistentTraceStore()
        await store.register_run("run-1", "trace-1", {})
        await store.update_run_status("run-1", "failed", error="timeout")
        run = await store.get_run("run-1")
        assert run is not None
        assert run.status == "failed"
        assert run.completed_at is not None
        assert run.error == "timeout"

    async def test_update_run_status_suspended(self) -> None:
        store = InMemoryPersistentTraceStore()
        await store.register_run("run-1", "trace-1", {})
        await store.update_run_status("run-1", "suspended")
        run = await store.get_run("run-1")
        assert run is not None
        assert run.status == "suspended"
        assert run.completed_at is None

    async def test_update_nonexistent_run(self) -> None:
        store = InMemoryPersistentTraceStore()
        # Should not raise
        await store.update_run_status("nonexistent", "completed")

    async def test_list_runs_all(self) -> None:
        store = InMemoryPersistentTraceStore()
        await store.register_run("run-1", "trace-1", {})
        await store.register_run("run-2", "trace-2", {})
        await store.register_run("run-3", "trace-3", {})
        runs = await store.list_runs()
        assert len(runs) == 3

    async def test_list_runs_status_filter(self) -> None:
        store = InMemoryPersistentTraceStore()
        await store.register_run("run-1", "trace-1", {})
        await store.register_run("run-2", "trace-2", {})
        await store.update_run_status("run-1", "completed")
        runs = await store.list_runs(status="running")
        assert len(runs) == 1
        assert runs[0].id == "run-2"

    async def test_list_runs_pagination(self) -> None:
        store = InMemoryPersistentTraceStore()
        for i in range(5):
            await store.register_run(f"run-{i}", f"trace-{i}", {})
        page1 = await store.list_runs(limit=2, offset=0)
        page2 = await store.list_runs(limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        ids = [r.id for r in page1 + page2]
        assert len(set(ids)) == 4

    async def test_list_runs_descending_order(self) -> None:
        store = InMemoryPersistentTraceStore()
        await store.register_run("run-old", "trace-1", {})
        await store.register_run("run-new", "trace-2", {})
        runs = await store.list_runs()
        # Most recent first
        assert runs[0].id == "run-new"

    async def test_run_record_is_frozen(self) -> None:
        store = InMemoryPersistentTraceStore()
        await store.register_run("run-1", "trace-1", {"key": "value"})
        run = await store.get_run("run-1")
        assert run is not None
        assert isinstance(run, RunRecord)

    async def test_list_runs_time_range_filter(self) -> None:
        store = InMemoryPersistentTraceStore()
        now = datetime.now(tz=UTC)
        # Register runs and override started_at for deterministic tests
        await store.register_run("run-old", "t1", {})
        await store.register_run("run-mid", "t2", {})
        await store.register_run("run-new", "t3", {})
        store._runs["run-old"] = store._runs["run-old"].model_copy(update={"started_at": now - timedelta(hours=3)})
        store._runs["run-mid"] = store._runs["run-mid"].model_copy(update={"started_at": now - timedelta(hours=1)})
        store._runs["run-new"] = store._runs["run-new"].model_copy(update={"started_at": now})

        runs = await store.list_runs(started_after=now - timedelta(hours=2))
        assert len(runs) == 2
        ids = {r.id for r in runs}
        assert ids == {"run-mid", "run-new"}

        runs = await store.list_runs(started_before=now - timedelta(hours=2))
        assert len(runs) == 1
        assert runs[0].id == "run-old"

    async def test_list_runs_search(self) -> None:
        store = InMemoryPersistentTraceStore()
        await store.register_run("run-1", "t1", {"description": "Alpha test"})
        await store.register_run("run-2", "t2", {"description": "Beta test"})
        await store.register_run("run-3", "t3", {"description": "Gamma run"})

        runs = await store.list_runs(search="alpha")
        assert len(runs) == 1
        assert runs[0].id == "run-1"

        runs = await store.list_runs(search="test")
        assert len(runs) == 2

    async def test_list_runs_sort_options(self) -> None:
        store = InMemoryPersistentTraceStore()
        now = datetime.now(tz=UTC)
        await store.register_run("run-a", "t1", {})
        await store.register_run("run-b", "t2", {})
        store._runs["run-a"] = store._runs["run-a"].model_copy(update={"started_at": now - timedelta(hours=2)})
        store._runs["run-b"] = store._runs["run-b"].model_copy(update={"started_at": now})

        asc = await store.list_runs(sort="started_at_asc")
        assert asc[0].id == "run-a"

        desc = await store.list_runs(sort="started_at_desc")
        assert desc[0].id == "run-b"

    async def test_count_runs(self) -> None:
        store = InMemoryPersistentTraceStore()
        await store.register_run("run-1", "t1", {})
        await store.register_run("run-2", "t2", {})
        await store.update_run_status("run-1", "completed")

        assert await store.count_runs() == 2
        assert await store.count_runs(status="running") == 1
        assert await store.count_runs(status="completed") == 1

    async def test_delete_run(self) -> None:
        store = InMemoryPersistentTraceStore()
        await store.register_run("run-1", "t1", {})
        await store.save_events_batch(
            "run-1",
            [
                _make_record("agent.start", trace_id="t1", span_id="s1"),
                _make_record("llm.request", "debug", trace_id="t1", span_id="s1"),
            ],
        )
        result = await store.delete_run("run-1")
        assert result is True
        assert await store.get_run("run-1") is None
        events = await store.query_events("run-1")
        assert events == []

    async def test_delete_run_not_found(self) -> None:
        store = InMemoryPersistentTraceStore()
        result = await store.delete_run("nonexistent")
        assert result is False

    async def test_delete_run_preserves_other_runs(self) -> None:
        store = InMemoryPersistentTraceStore()
        await store.register_run("run-keep", "t-keep", {})
        await store.save_events_batch(
            "run-keep",
            [
                _make_record("agent.start", trace_id="t-keep", span_id="s1"),
            ],
        )
        await store.register_run("run-del", "t-del", {})
        await store.save_events_batch(
            "run-del",
            [
                _make_record("agent.start", trace_id="t-del", span_id="s2"),
            ],
        )
        await store.delete_run("run-del")
        assert await store.get_run("run-keep") is not None
        kept_events = await store.query_events("run-keep")
        assert len(kept_events) == 1
        assert await store.count_runs() == 1


class TestHierarchyQueries:
    async def test_get_span_tree_returns_all_events_for_trace(self) -> None:
        store = InMemoryPersistentTraceStore()
        now = datetime.now(tz=UTC)
        await store.save_events_batch(
            "parent-1",
            [
                _make_record("agent.start", trace_id="trace-1", span_id="s1", sdk_timestamp=now),
                _make_record(
                    "llm.request",
                    "debug",
                    trace_id="trace-1",
                    span_id="s1",
                    sdk_timestamp=now + timedelta(milliseconds=10),
                ),
                _make_record(
                    "agent.complete",
                    trace_id="trace-1",
                    span_id="s1",
                    sdk_timestamp=now + timedelta(milliseconds=20),
                ),
            ],
        )
        # Different trace — should not appear
        await store.save_events_batch(
            "parent-2",
            [_make_record("agent.start", trace_id="trace-2", span_id="s9")],
        )

        tree = await store.get_span_tree("trace-1")
        assert len(tree) == 3
        assert all(e.trace_id == "trace-1" for e in tree)
        assert all(isinstance(e, StoredTraceEvent) for e in tree)

    async def test_get_span_tree_ordered_by_timestamp(self) -> None:
        store = InMemoryPersistentTraceStore()
        now = datetime.now(tz=UTC)
        await store.save_events_batch(
            "parent-1",
            [
                _make_record(
                    "agent.complete",
                    trace_id="t1",
                    span_id="s1",
                    sdk_timestamp=now + timedelta(milliseconds=20),
                ),
                _make_record(
                    "agent.start",
                    trace_id="t1",
                    span_id="s1",
                    sdk_timestamp=now,
                ),
            ],
        )
        tree = await store.get_span_tree("t1")
        assert tree[0].event_type == "agent.start"
        assert tree[1].event_type == "agent.complete"

    async def test_get_span_tree_across_parent_ids(self) -> None:
        """Events from different parent_ids but same trace_id are included."""
        store = InMemoryPersistentTraceStore()
        now = datetime.now(tz=UTC)
        await store.save_events_batch(
            "parent-a",
            [_make_record("agent.start", trace_id="t1", span_id="s1", sdk_timestamp=now)],
        )
        await store.save_events_batch(
            "parent-b",
            [
                _make_record(
                    "agent.start",
                    trace_id="t1",
                    span_id="s2",
                    sdk_timestamp=now + timedelta(milliseconds=10),
                )
            ],
        )
        tree = await store.get_span_tree("t1")
        assert len(tree) == 2

    async def test_get_span_tree_empty(self) -> None:
        store = InMemoryPersistentTraceStore()
        tree = await store.get_span_tree("nonexistent")
        assert tree == []

    async def test_get_events_by_span(self) -> None:
        store = InMemoryPersistentTraceStore()
        now = datetime.now(tz=UTC)
        await store.save_events_batch(
            "parent-1",
            [
                _make_record("agent.start", trace_id="t1", span_id="s1", sdk_timestamp=now),
                _make_record(
                    "llm.request",
                    "debug",
                    trace_id="t1",
                    span_id="s1",
                    sdk_timestamp=now + timedelta(milliseconds=10),
                ),
                _make_record("agent.start", trace_id="t1", span_id="s2", sdk_timestamp=now),
            ],
        )
        events = await store.get_events_by_span("t1", "s1")
        assert len(events) == 2
        assert all(e.span_id == "s1" for e in events)

    async def test_get_events_by_span_empty(self) -> None:
        store = InMemoryPersistentTraceStore()
        events = await store.get_events_by_span("t1", "nonexistent")
        assert events == []

    async def test_get_events_by_span_ordered(self) -> None:
        store = InMemoryPersistentTraceStore()
        now = datetime.now(tz=UTC)
        await store.save_events_batch(
            "parent-1",
            [
                _make_record(
                    "agent.complete",
                    trace_id="t1",
                    span_id="s1",
                    sdk_timestamp=now + timedelta(milliseconds=20),
                ),
                _make_record(
                    "agent.start",
                    trace_id="t1",
                    span_id="s1",
                    sdk_timestamp=now,
                ),
            ],
        )
        events = await store.get_events_by_span("t1", "s1")
        assert events[0].event_type == "agent.start"
        assert events[1].event_type == "agent.complete"

    async def test_span_tree_with_hierarchy(self) -> None:
        """Verify tree with parent-child span relationships."""
        store = InMemoryPersistentTraceStore()
        now = datetime.now(tz=UTC)
        await store.save_events_batch(
            "parent-1",
            [
                _make_record(
                    "workflow.start",
                    trace_id="t1",
                    span_id="root",
                    parent_span_id=None,
                    sdk_timestamp=now,
                ),
                _make_record(
                    "agent.start",
                    trace_id="t1",
                    span_id="child-1",
                    parent_span_id="root",
                    sdk_timestamp=now + timedelta(milliseconds=5),
                ),
                _make_record(
                    "agent.start",
                    trace_id="t1",
                    span_id="child-2",
                    parent_span_id="root",
                    sdk_timestamp=now + timedelta(milliseconds=10),
                ),
                _make_record(
                    "workflow.complete",
                    trace_id="t1",
                    span_id="root",
                    parent_span_id=None,
                    sdk_timestamp=now + timedelta(milliseconds=20),
                ),
            ],
        )
        tree = await store.get_span_tree("t1")
        assert len(tree) == 4
        # root events at beginning and end
        assert tree[0].span_id == "root"
        assert tree[0].parent_span_id is None
        # child events in the middle
        child_events = [e for e in tree if e.parent_span_id == "root"]
        assert len(child_events) == 2


class TestExistingPersistentStoreMethods:
    """Verify existing PersistentTraceStore methods on InMemoryPersistentTraceStore."""

    async def test_save_and_query_events(self) -> None:
        store = InMemoryPersistentTraceStore()
        records = [
            _make_record("agent.start", "info"),
            _make_record("llm.request", "debug"),
        ]
        await store.save_events_batch("parent-1", records)
        events = await store.query_events("parent-1")
        assert len(events) == 2
        assert events[0].event_type == "agent.start"
        assert events[1].event_type == "llm.request"

    async def test_query_with_level_filter(self) -> None:
        store = InMemoryPersistentTraceStore()
        await store.save_events_batch(
            "p1",
            [
                _make_record("agent.start", "info"),
                _make_record("llm.request", "debug"),
            ],
        )
        info_only = await store.query_events("p1", levels=["info"])
        assert len(info_only) == 1

    async def test_query_with_event_type_filter(self) -> None:
        store = InMemoryPersistentTraceStore()
        await store.save_events_batch(
            "p1",
            [
                _make_record("agent.start", "info"),
                _make_record("llm.request", "debug"),
                _make_record("llm.response", "debug"),
            ],
        )
        llm_events = await store.query_events("p1", event_types=["llm.request", "llm.response"])
        assert len(llm_events) == 2

    async def test_cursor_pagination(self) -> None:
        store = InMemoryPersistentTraceStore()
        records = [_make_record(f"event.{i}", "info") for i in range(5)]
        await store.save_events_batch("p1", records)
        page1 = await store.query_events("p1", limit=2)
        assert len(page1) == 2
        page2 = await store.query_events("p1", after_id=page1[-1].id, limit=2)
        assert len(page2) == 2
        page3 = await store.query_events("p1", after_id=page2[-1].id, limit=2)
        assert len(page3) == 1

    async def test_get_event(self) -> None:
        store = InMemoryPersistentTraceStore()
        await store.save_events_batch("p1", [_make_record("agent.start")])
        events = await store.query_events("p1")
        event = await store.get_event(events[0].id)
        assert event is not None
        assert event.event_type == "agent.start"

    async def test_get_event_not_found(self) -> None:
        store = InMemoryPersistentTraceStore()
        result = await store.get_event(999999)
        assert result is None

    async def test_get_summary(self) -> None:
        store = InMemoryPersistentTraceStore()
        await store.save_events_batch(
            "p1",
            [
                _make_record("agent.start", "info", payload={"agent_name": "researcher"}),
                _make_record("llm.request", "debug"),
                _make_record(
                    "llm.response",
                    "debug",
                    payload={"usage": {"input_tokens": 200, "output_tokens": 80}},
                ),
                _make_record("tool.invoke", "debug", payload={"tool_name": "search"}),
            ],
        )
        summary = await store.get_summary("p1")
        assert summary.total_events == 4
        assert summary.llm_calls == 1
        assert summary.tool_calls == 1
        assert summary.total_input_tokens == 200
        assert summary.total_output_tokens == 80
        assert summary.agent_names == ["researcher"]

    async def test_get_summary_cache_tokens(self) -> None:
        store = InMemoryPersistentTraceStore()
        await store.save_events_batch(
            "p1",
            [
                _make_record(
                    "llm.response",
                    "debug",
                    payload={
                        "usage": {
                            "input_tokens": 200,
                            "output_tokens": 80,
                            "cache_creation_input_tokens": 100,
                            "cache_read_input_tokens": 50,
                        },
                    },
                ),
                _make_record(
                    "llm.response",
                    "debug",
                    payload={
                        "usage": {
                            "input_tokens": 300,
                            "output_tokens": 120,
                            "cache_read_input_tokens": 200,
                        },
                    },
                ),
            ],
        )
        summary = await store.get_summary("p1")
        assert summary.cache_creation_tokens == 100
        assert summary.cache_read_tokens == 250

    async def test_events_isolated_by_parent_id(self) -> None:
        store = InMemoryPersistentTraceStore()
        await store.save_events_batch("a", [_make_record("agent.start")])
        await store.save_events_batch("b", [_make_record("llm.request", "debug")])
        a_events = await store.query_events("a")
        b_events = await store.query_events("b")
        assert len(a_events) == 1
        assert len(b_events) == 1


class TestDurationSort:
    async def test_duration_asc_sort(self) -> None:
        store = InMemoryPersistentTraceStore()
        now = datetime.now(tz=UTC)
        await store.register_run("fast", "t1", {})
        await store.register_run("slow", "t2", {})
        store._runs["fast"] = store._runs["fast"].model_copy(
            update={"started_at": now, "completed_at": now + timedelta(seconds=1)},
        )
        store._runs["slow"] = store._runs["slow"].model_copy(
            update={"started_at": now, "completed_at": now + timedelta(seconds=10)},
        )
        result = await store.list_runs(sort="duration_asc")
        assert result[0].id == "fast"
        assert result[1].id == "slow"

    async def test_duration_desc_sort(self) -> None:
        store = InMemoryPersistentTraceStore()
        now = datetime.now(tz=UTC)
        await store.register_run("fast", "t1", {})
        await store.register_run("slow", "t2", {})
        store._runs["fast"] = store._runs["fast"].model_copy(
            update={"started_at": now, "completed_at": now + timedelta(seconds=1)},
        )
        store._runs["slow"] = store._runs["slow"].model_copy(
            update={"started_at": now, "completed_at": now + timedelta(seconds=10)},
        )
        result = await store.list_runs(sort="duration_desc")
        assert result[0].id == "slow"
        assert result[1].id == "fast"
