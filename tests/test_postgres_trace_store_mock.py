"""Mock-based tests for PostgresTraceStore — no real Postgres needed."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanitics.infrastructure.observability.postgres_store import (
    PostgresTraceStore,
    _build_migrations,
    _row_to_event,
    _row_to_run,
    get_schema_sql,
)
from nanitics.infrastructure.observability.storage import (
    RunResult,
    StoredTraceEvent,
    TerminationReason,
    TraceEventRecord,
    TraceSummaryStats,
)


def _make_transaction_cm() -> AsyncMock:
    tx = AsyncMock()
    tx.__aenter__ = AsyncMock(return_value=tx)
    tx.__aexit__ = AsyncMock(return_value=False)
    return tx


def _make_pool() -> tuple[MagicMock, AsyncMock]:
    pool = MagicMock()
    conn = AsyncMock()
    # Replace `transaction()` with a plain callable that returns an async CM.
    conn.transaction = MagicMock(side_effect=lambda: _make_transaction_cm())
    acquire_cm = AsyncMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = acquire_cm
    # Pool-direct methods are AsyncMocks too; tests override per-case.
    pool.fetch = AsyncMock(return_value=[])
    pool.fetchrow = AsyncMock(return_value=None)
    return pool, conn


def _ts(seconds: int = 0) -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds)


def _event_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": 1,
        "event_type": "agent.start",
        "level": "info",
        "trace_id": "trace-1",
        "span_id": "span-1",
        "parent_span_id": None,
        "payload": {"k": "v"},
        "sdk_timestamp": _ts(),
    }
    row.update(overrides)
    return row


def _run_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": "run-1",
        "trace_id": "trace-1",
        "status": "running",
        "started_at": _ts(),
        "completed_at": None,
        "metadata": {"k": "v"},
        "error": None,
        "result": None,
        "parent_run_id": None,
    }
    row.update(overrides)
    return row


# ── get_schema_sql ─────────────────────────────────────────


class TestGetSchemaSql:
    def test_default_returns_baseline_sql(self) -> None:
        sql = get_schema_sql()
        assert "trace_events" in sql
        assert " runs (" in sql

    def test_custom_table_name_substituted(self) -> None:
        sql = get_schema_sql(table_name="my_events")
        assert "my_events" in sql
        assert "trace_events" not in sql
        # Runs table untouched.
        assert " runs (" in sql

    def test_custom_runs_table_name_substituted(self) -> None:
        sql = get_schema_sql(runs_table_name="my_runs")
        assert " my_runs (" in sql
        assert " runs (" not in sql
        assert "idx_my_runs_" in sql
        assert "idx_runs_" not in sql


# ── _build_migrations ──────────────────────────────────────


class TestBuildMigrations:
    def test_default_table_names(self) -> None:
        migrations = _build_migrations("trace_events", "runs")
        assert len(migrations) == 3
        baseline = migrations[0].up_sql[0]
        assert "trace_events" in baseline
        assert " runs (" in baseline
        assert migrations[0].version == 1
        assert migrations[1].version == 2
        assert migrations[2].version == 3

    def test_custom_events_table(self) -> None:
        migrations = _build_migrations("my_events", "runs")
        baseline = migrations[0].up_sql[0]
        assert "my_events" in baseline
        assert "trace_events" not in baseline
        v2_sql = migrations[1].up_sql[0]
        assert "ALTER TABLE my_events" in v2_sql
        assert "table_name = 'my_events'" in v2_sql

    def test_custom_runs_table(self) -> None:
        migrations = _build_migrations("trace_events", "my_runs")
        baseline = migrations[0].up_sql[0]
        assert " my_runs (" in baseline
        assert "idx_my_runs_" in baseline
        assert " runs (" not in baseline

    def test_v3_adds_parent_run_id_with_cascade_and_partial_index(self) -> None:
        migrations = _build_migrations("trace_events", "runs")
        v3 = migrations[2]
        assert v3.version == 3
        joined = "\n".join(v3.up_sql)
        assert "ALTER TABLE runs" in joined
        assert "ADD COLUMN parent_run_id TEXT NULL" in joined
        assert "REFERENCES runs(id) ON DELETE CASCADE" in joined
        assert "CREATE INDEX idx_runs_parent" in joined
        assert "WHERE parent_run_id IS NOT NULL" in joined

    def test_v3_substitutes_custom_runs_table_in_alter_index_and_fk(self) -> None:
        migrations = _build_migrations("evt_x", "runs_y")
        v3 = migrations[2]
        joined = "\n".join(v3.up_sql)
        assert "ALTER TABLE runs_y" in joined
        assert "REFERENCES runs_y(id)" in joined
        assert "idx_runs_y_parent" in joined
        assert "ON runs_y (parent_run_id)" in joined
        # No bare ``runs`` token leaks through.
        assert " runs " not in joined
        assert " runs(" not in joined


# ── ensure_schema ──────────────────────────────────────────


def _sql_calls(conn: AsyncMock) -> list[str]:
    return [call.args[0] for call in conn.execute.await_args_list]


class TestEnsureSchema:
    async def test_fresh_database_creates_version_table_and_runs_all_migrations(self) -> None:
        pool, conn = _make_pool()
        # fetchval sequence:
        #   1) version table exists? -> False
        #   2) events table exists? -> False
        #   3) current version after inserting 0 -> 0
        conn.fetchval = AsyncMock(side_effect=[False, False, 0])

        store = PostgresTraceStore(pool)
        await store.ensure_schema()

        sqls = _sql_calls(conn)
        joined = "\n".join(sqls)
        assert any("pg_advisory_lock" in s for s in sqls)
        assert any("CREATE TABLE _trace_events_schema_version" in s for s in sqls)
        assert any("INSERT INTO _trace_events_schema_version (version) VALUES (0)" in s for s in sqls)
        # All three migrations applied (baseline + v2 alter + v3 parent_run_id).
        assert "CREATE TABLE IF NOT EXISTS trace_events" in joined
        assert "ALTER TABLE trace_events" in joined
        assert "ADD COLUMN parent_run_id" in joined
        assert "CREATE INDEX idx_runs_parent" in joined
        # Version updated to 1, then 2, then 3.
        version_updates = [s for s in sqls if "SET version = $1" in s]
        assert len(version_updates) == 3
        assert any("pg_advisory_unlock" in s for s in sqls)

    async def test_legacy_database_marks_v1_then_runs_v2_and_v3(self) -> None:
        pool, conn = _make_pool()
        # Version table missing, events table present, current version = 1,
        # sibling-guard check: runs table exists.
        conn.fetchval = AsyncMock(side_effect=[False, True, 1, True])

        store = PostgresTraceStore(pool)
        await store.ensure_schema()

        sqls = _sql_calls(conn)
        # Marked as v1 baseline already applied.
        assert any("INSERT INTO _trace_events_schema_version (version) VALUES (1)" in s for s in sqls)
        # Baseline (v1) is skipped — no CREATE TABLE IF NOT EXISTS trace_events issued.
        assert not any("CREATE TABLE IF NOT EXISTS trace_events" in s for s in sqls)
        # v2 migration applied.
        assert any("ALTER TABLE trace_events" in s for s in sqls)
        # v3 migration applied.
        assert any("ADD COLUMN parent_run_id" in s for s in sqls)
        # Version updated twice (to 2, then 3).
        version_updates = [s for s in sqls if "SET version = $1" in s]
        assert len(version_updates) == 2

    async def test_v2_to_v3_records_alter_and_partial_index(self) -> None:
        """Specifically verify the v2→v3 transition emits both DDL statements."""
        pool, conn = _make_pool()
        # Version table exists, current version 2, runs table exists.
        conn.fetchval = AsyncMock(side_effect=[True, 2, True])

        store = PostgresTraceStore(pool)
        await store.ensure_schema()

        sqls = _sql_calls(conn)
        joined = "\n".join(sqls)
        # No v0/v1/v2 DDL appears.
        assert "CREATE TABLE IF NOT EXISTS trace_events" not in joined
        assert "ALTER COLUMN parent_id" not in joined
        # v3 ALTER + partial index do appear.
        assert "ADD COLUMN parent_run_id" in joined
        assert "REFERENCES runs(id) ON DELETE CASCADE" in joined
        assert "CREATE INDEX idx_runs_parent" in joined
        assert "WHERE parent_run_id IS NOT NULL" in joined
        # Single version bump to 3.
        version_updates = [s for s in sqls if "SET version = $1" in s]
        assert len(version_updates) == 1

    async def test_already_at_latest_version_is_noop(self) -> None:
        pool, conn = _make_pool()
        # Version table exists, current version is already 3, sibling-guard check: runs table exists.
        conn.fetchval = AsyncMock(side_effect=[True, 3, True])

        store = PostgresTraceStore(pool)
        await store.ensure_schema()

        sqls = _sql_calls(conn)
        # No schema_version INSERT, no migration up_sql, no UPDATE ... version.
        assert not any("INSERT INTO _trace_events_schema_version" in s for s in sqls)
        assert not any("CREATE TABLE IF NOT EXISTS trace_events" in s for s in sqls)
        assert not any("ALTER TABLE trace_events" in s for s in sqls)
        assert not any("ADD COLUMN parent_run_id" in s for s in sqls)
        assert not any("SET version = $1" in s for s in sqls)
        # Advisory lock acquired and released.
        assert any("pg_advisory_lock" in s for s in sqls)
        assert any("pg_advisory_unlock" in s for s in sqls)

    async def test_advisory_lock_released_on_migration_failure(self) -> None:
        pool, conn = _make_pool()
        # Version table exists, current_version=1, sibling-guard check passes (runs exists),
        # transaction then fails.
        conn.fetchval = AsyncMock(side_effect=[True, 1, True])

        # Make `conn.transaction()` raise inside the try block so the
        # advisory_unlock in the finally is exercised.
        failing_tx = AsyncMock()
        failing_tx.__aenter__ = AsyncMock(side_effect=RuntimeError("boom"))
        failing_tx.__aexit__ = AsyncMock(return_value=False)
        conn.transaction = MagicMock(return_value=failing_tx)

        store = PostgresTraceStore(pool)
        with pytest.raises(RuntimeError, match="boom"):
            await store.ensure_schema()

        sqls = _sql_calls(conn)
        assert any("pg_advisory_lock" in s for s in sqls)
        assert any("pg_advisory_unlock" in s for s in sqls)

    async def test_sibling_conflict_guard_raises_when_runs_table_missing(self) -> None:
        """If the version row claims baseline ran but configured runs_table is absent,
        ``ensure_schema`` raises ``RuntimeError`` naming both tables — the SDK will not
        silently re-run migrations under a different runs_table configuration.
        """
        pool, conn = _make_pool()
        # Version table exists, current_version=2, sibling-guard check: runs_b missing.
        conn.fetchval = AsyncMock(side_effect=[True, 2, False])

        store = PostgresTraceStore(pool, table_name="evt_shared", runs_table="runs_b")
        with pytest.raises(RuntimeError) as excinfo:
            await store.ensure_schema()
        message = str(excinfo.value)
        assert "runs_b" in message
        assert "evt_shared" in message
        assert "schema version row" in message

        # Advisory lock still released.
        sqls = _sql_calls(conn)
        assert any("pg_advisory_unlock" in s for s in sqls)

    async def test_sibling_guard_does_not_fire_on_fresh_database(self) -> None:
        """``current_version == 0`` path: runs table is legitimately absent and about
        to be created by the baseline migration. Guard must not fire.
        """
        pool, conn = _make_pool()
        # Version table missing, events table missing, current version=0 (fresh).
        # If the guard fired here, fetchval would be exhausted earlier — we provide
        # no fourth value because the guard branch must not be taken.
        conn.fetchval = AsyncMock(side_effect=[False, False, 0])

        store = PostgresTraceStore(pool)
        await store.ensure_schema()  # must not raise


# ── save_events_batch ──────────────────────────────────────


def _make_event_record(event_type: str = "agent.start", payload: dict[str, Any] | None = None) -> TraceEventRecord:
    return TraceEventRecord(
        event_type=event_type,
        level="info",
        trace_id="trace-1",
        span_id="span-1",
        parent_span_id=None,
        payload=payload if payload is not None else {"k": "v"},
        sdk_timestamp=_ts(),
    )


class TestSaveEventsBatch:
    async def test_empty_events_short_circuits(self) -> None:
        pool, conn = _make_pool()
        store = PostgresTraceStore(pool)
        await store.save_events_batch("parent-1", [])
        pool.acquire.assert_not_called()
        conn.executemany.assert_not_called()

    async def test_non_empty_events_executes_executemany_with_correct_tuples(self) -> None:
        pool, conn = _make_pool()
        store = PostgresTraceStore(pool)
        events = [
            _make_event_record("agent.start", {"a": 1}),
            _make_event_record("agent.complete", {"b": 2}),
        ]
        await store.save_events_batch("parent-1", events)
        conn.executemany.assert_awaited_once()
        sql, values = conn.executemany.await_args[0]
        assert "INSERT INTO trace_events" in sql
        assert isinstance(values, list)
        assert len(values) == 2
        # payload is the 7th element (index 6) and JSON-serialised.
        assert json.loads(values[0][6]) == {"a": 1}
        assert json.loads(values[1][6]) == {"b": 2}
        # parent_id is the first element.
        assert values[0][0] == "parent-1"


# ── query_events ───────────────────────────────────────────


class TestQueryEvents:
    async def test_no_filters_returns_all_rows(self) -> None:
        pool, _ = _make_pool()
        pool.fetch = AsyncMock(return_value=[_event_row(id=42)])

        store = PostgresTraceStore(pool)
        results = await store.query_events("parent-1")

        assert len(results) == 1
        assert isinstance(results[0], StoredTraceEvent)
        assert results[0].id == 42
        sql = pool.fetch.await_args[0][0]
        assert "WHERE parent_id = $1" in sql
        assert "ANY(" not in sql

    async def test_with_after_id(self) -> None:
        pool, _ = _make_pool()
        pool.fetch = AsyncMock(return_value=[])

        store = PostgresTraceStore(pool)
        await store.query_events("parent-1", after_id=100)

        args = pool.fetch.await_args[0]
        sql = args[0]
        assert "id > $2" in sql
        # Params: parent_id, after_id, ..., limit
        assert 100 in args

    async def test_with_levels(self) -> None:
        pool, _ = _make_pool()
        pool.fetch = AsyncMock(return_value=[])

        store = PostgresTraceStore(pool)
        await store.query_events("parent-1", levels=["info", "debug"])

        sql = pool.fetch.await_args[0][0]
        assert "level = ANY($" in sql

    async def test_with_event_types(self) -> None:
        pool, _ = _make_pool()
        pool.fetch = AsyncMock(return_value=[])

        store = PostgresTraceStore(pool)
        await store.query_events("parent-1", event_types=["agent.start"])

        sql = pool.fetch.await_args[0][0]
        assert "event_type = ANY($" in sql

    async def test_with_all_filters_combined(self) -> None:
        pool, _ = _make_pool()
        pool.fetch = AsyncMock(return_value=[])

        store = PostgresTraceStore(pool)
        await store.query_events(
            "parent-1",
            after_id=5,
            levels=["info"],
            event_types=["agent.start"],
            limit=25,
        )

        args = pool.fetch.await_args[0]
        sql = args[0]
        assert "id > $2" in sql
        assert "level = ANY($3)" in sql
        assert "event_type = ANY($4)" in sql
        assert "LIMIT $5" in sql
        assert args[5] == 25  # limit is positional param 5


# ── get_event ──────────────────────────────────────────────


class TestGetEvent:
    async def test_found(self) -> None:
        pool, _ = _make_pool()
        pool.fetchrow = AsyncMock(return_value=_event_row(id=7))

        store = PostgresTraceStore(pool)
        result = await store.get_event(7)

        assert result is not None
        assert result.id == 7
        assert isinstance(result, StoredTraceEvent)

    async def test_not_found(self) -> None:
        pool, _ = _make_pool()
        pool.fetchrow = AsyncMock(return_value=None)

        store = PostgresTraceStore(pool)
        result = await store.get_event(999)
        assert result is None


# ── get_summary ────────────────────────────────────────────


class TestGetSummary:
    async def test_with_populated_aggregates(self) -> None:
        pool, _ = _make_pool()
        first_ts = _ts()
        last_ts = _ts(seconds=2)
        agg_row = {
            "total_events": 10,
            "info_count": 6,
            "debug_count": 3,
            "verbose_count": 1,
            "tool_calls": 2,
            "errors": 0,
            "first_ts": first_ts,
            "last_ts": last_ts,
        }
        tokens_row = {
            "llm_calls": 4,
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_tokens": 20,
            "cache_read_tokens": 10,
        }
        pool.fetchrow = AsyncMock(side_effect=[agg_row, tokens_row])
        pool.fetch = AsyncMock(return_value=[{"agent": "researcher"}, {"agent": "writer"}])

        store = PostgresTraceStore(pool)
        result = await store.get_summary("parent-1")

        assert isinstance(result, TraceSummaryStats)
        assert result.total_events == 10
        assert result.events_by_level == {"info": 6, "debug": 3, "verbose": 1}
        assert result.llm_calls == 4
        assert result.tool_calls == 2
        assert result.total_input_tokens == 100
        assert result.total_output_tokens == 50
        assert result.cache_creation_tokens == 20
        assert result.cache_read_tokens == 10
        assert result.total_duration_ms == 2000
        assert result.agent_names == ["researcher", "writer"]
        assert result.errors == 0

    async def test_with_empty_aggregates(self) -> None:
        pool, _ = _make_pool()
        pool.fetchrow = AsyncMock(side_effect=[None, None])
        pool.fetch = AsyncMock(return_value=[])

        store = PostgresTraceStore(pool)
        result = await store.get_summary("empty")

        assert result.total_events == 0
        assert result.events_by_level == {"info": 0, "debug": 0, "verbose": 0}
        assert result.llm_calls == 0
        assert result.tool_calls == 0
        assert result.total_input_tokens == 0
        assert result.total_output_tokens == 0
        assert result.total_duration_ms is None
        assert result.agent_names == []
        assert result.errors == 0
        assert result.cache_creation_tokens == 0
        assert result.cache_read_tokens == 0

    async def test_duration_ms_none_when_only_first_ts_set(self) -> None:
        pool, _ = _make_pool()
        agg_row = {
            "total_events": 1,
            "info_count": 1,
            "debug_count": 0,
            "verbose_count": 0,
            "tool_calls": 0,
            "errors": 0,
            "first_ts": _ts(),
            "last_ts": None,
        }
        tokens_row = {
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
        }
        pool.fetchrow = AsyncMock(side_effect=[agg_row, tokens_row])
        pool.fetch = AsyncMock(return_value=[])

        store = PostgresTraceStore(pool)
        result = await store.get_summary("parent")
        assert result.total_duration_ms is None


# ── register_run ───────────────────────────────────────────


class TestRegisterRun:
    async def test_inserts_with_metadata_json(self) -> None:
        pool, conn = _make_pool()
        store = PostgresTraceStore(pool)
        await store.register_run("run-1", "trace-1", {"k": "v"})

        conn.execute.assert_awaited_once()
        args = conn.execute.await_args[0]
        assert "INSERT INTO runs" in args[0]
        assert "parent_run_id" in args[0]
        assert args[1] == "run-1"
        assert args[2] == "trace-1"
        assert args[3] == json.dumps({"k": "v"})
        # parent_run_id defaults to None.
        assert args[4] is None

    async def test_inserts_with_explicit_parent_run_id(self) -> None:
        pool, conn = _make_pool()
        store = PostgresTraceStore(pool)
        await store.register_run("child-1", "trace-1", {}, parent_run_id="run-parent")

        args = conn.execute.await_args[0]
        assert "INSERT INTO runs" in args[0]
        assert args[1] == "child-1"
        assert args[4] == "run-parent"


# ── update_run_status ──────────────────────────────────────


class TestUpdateRunStatus:
    async def test_completed_uses_completed_at_branch(self) -> None:
        pool, conn = _make_pool()
        store = PostgresTraceStore(pool)
        await store.update_run_status("run-1", "completed")
        sql = conn.execute.await_args[0][0]
        assert "completed_at = NOW()" in sql

    async def test_failed_uses_completed_at_branch(self) -> None:
        pool, conn = _make_pool()
        store = PostgresTraceStore(pool)
        await store.update_run_status("run-1", "failed", error="oops")
        sql = conn.execute.await_args[0][0]
        assert "completed_at = NOW()" in sql

    async def test_running_uses_running_branch(self) -> None:
        pool, conn = _make_pool()
        store = PostgresTraceStore(pool)
        await store.update_run_status("run-1", "running")
        sql = conn.execute.await_args[0][0]
        assert "completed_at = NOW()" not in sql


# ── get_run ────────────────────────────────────────────────


class TestGetRun:
    async def test_found(self) -> None:
        pool, _ = _make_pool()
        pool.fetchrow = AsyncMock(return_value=_run_row(id="run-1"))

        store = PostgresTraceStore(pool)
        result = await store.get_run("run-1")
        assert result is not None
        assert result.id == "run-1"

    async def test_not_found(self) -> None:
        pool, _ = _make_pool()
        pool.fetchrow = AsyncMock(return_value=None)

        store = PostgresTraceStore(pool)
        result = await store.get_run("missing")
        assert result is None


# ── list_runs ──────────────────────────────────────────────


class TestListRuns:
    async def test_no_filters_default_sort(self) -> None:
        pool, _ = _make_pool()
        pool.fetch = AsyncMock(return_value=[])

        store = PostgresTraceStore(pool)
        await store.list_runs()

        sql = pool.fetch.await_args[0][0]
        assert "ORDER BY started_at DESC" in sql
        assert "WHERE" not in sql

    async def test_with_status_filter(self) -> None:
        pool, _ = _make_pool()
        pool.fetch = AsyncMock(return_value=[])

        store = PostgresTraceStore(pool)
        await store.list_runs(status="running")

        sql = pool.fetch.await_args[0][0]
        assert "WHERE status = $1" in sql

    async def test_with_started_after_filter(self) -> None:
        pool, _ = _make_pool()
        pool.fetch = AsyncMock(return_value=[])

        store = PostgresTraceStore(pool)
        await store.list_runs(started_after=_ts())

        sql = pool.fetch.await_args[0][0]
        assert "started_at >= $" in sql

    async def test_with_started_before_filter(self) -> None:
        pool, _ = _make_pool()
        pool.fetch = AsyncMock(return_value=[])

        store = PostgresTraceStore(pool)
        await store.list_runs(started_before=_ts())

        sql = pool.fetch.await_args[0][0]
        assert "started_at < $" in sql

    async def test_with_search_filter(self) -> None:
        pool, _ = _make_pool()
        pool.fetch = AsyncMock(return_value=[])

        store = PostgresTraceStore(pool)
        await store.list_runs(search="needle")

        args = pool.fetch.await_args[0]
        sql = args[0]
        assert "metadata::text ILIKE $" in sql
        assert "%needle%" in args

    async def test_with_all_filters_combined(self) -> None:
        pool, _ = _make_pool()
        pool.fetch = AsyncMock(return_value=[])

        store = PostgresTraceStore(pool)
        await store.list_runs(
            status="running",
            started_after=_ts(),
            started_before=_ts(seconds=60),
            search="x",
        )

        sql = pool.fetch.await_args[0][0]
        assert sql.count("AND") >= 3  # three ANDs joining four predicates
        assert "status = $" in sql
        assert "started_at >= $" in sql
        assert "started_at < $" in sql
        assert "metadata::text ILIKE $" in sql

    async def test_sort_started_at_asc(self) -> None:
        pool, _ = _make_pool()
        pool.fetch = AsyncMock(return_value=[])

        store = PostgresTraceStore(pool)
        await store.list_runs(sort="started_at_asc")

        sql = pool.fetch.await_args[0][0]
        assert "ORDER BY started_at ASC" in sql

    async def test_sort_duration_desc(self) -> None:
        pool, _ = _make_pool()
        pool.fetch = AsyncMock(return_value=[])

        store = PostgresTraceStore(pool)
        await store.list_runs(sort="duration_desc")

        sql = pool.fetch.await_args[0][0]
        assert "(COALESCE(completed_at, started_at) - started_at) DESC" in sql

    async def test_sort_duration_asc(self) -> None:
        pool, _ = _make_pool()
        pool.fetch = AsyncMock(return_value=[])

        store = PostgresTraceStore(pool)
        await store.list_runs(sort="duration_asc")

        sql = pool.fetch.await_args[0][0]
        assert "(COALESCE(completed_at, started_at) - started_at) ASC" in sql

    async def test_unknown_sort_falls_back_to_default(self) -> None:
        pool, _ = _make_pool()
        pool.fetch = AsyncMock(return_value=[])

        store = PostgresTraceStore(pool)
        await store.list_runs(sort="garbage")

        sql = pool.fetch.await_args[0][0]
        assert "ORDER BY started_at DESC" in sql

    async def test_returns_list_of_run_records(self) -> None:
        pool, _ = _make_pool()
        pool.fetch = AsyncMock(return_value=[_run_row(id="run-xyz")])

        store = PostgresTraceStore(pool)
        results = await store.list_runs()
        assert len(results) == 1
        assert results[0].id == "run-xyz"

    async def test_parent_run_id_default_unset_omits_predicate(self) -> None:
        pool, _ = _make_pool()
        pool.fetch = AsyncMock(return_value=[])

        store = PostgresTraceStore(pool)
        await store.list_runs()

        sql = pool.fetch.await_args[0][0]
        assert "parent_run_id" not in sql.split("FROM")[1]

    async def test_parent_run_id_none_filters_top_level_only(self) -> None:
        pool, _ = _make_pool()
        pool.fetch = AsyncMock(return_value=[])

        store = PostgresTraceStore(pool)
        await store.list_runs(parent_run_id=None)

        args = pool.fetch.await_args[0]
        sql = args[0]
        assert "parent_run_id IS NULL" in sql
        # No bound parameter for the IS NULL predicate.
        assert "run-parent" not in args

    async def test_parent_run_id_string_filters_children(self) -> None:
        pool, _ = _make_pool()
        pool.fetch = AsyncMock(return_value=[])

        store = PostgresTraceStore(pool)
        await store.list_runs(parent_run_id="run-parent")

        args = pool.fetch.await_args[0]
        sql = args[0]
        assert "parent_run_id = $" in sql
        assert "run-parent" in args

    async def test_select_list_includes_parent_run_id(self) -> None:
        pool, _ = _make_pool()
        pool.fetch = AsyncMock(return_value=[])

        store = PostgresTraceStore(pool)
        await store.list_runs()
        sql = pool.fetch.await_args[0][0]
        assert "parent_run_id" in sql.split("FROM")[0]


# ── count_runs ─────────────────────────────────────────────


class TestCountRuns:
    async def test_no_filters_returns_count(self) -> None:
        pool, _ = _make_pool()
        pool.fetchrow = AsyncMock(return_value={"cnt": 7})

        store = PostgresTraceStore(pool)
        assert await store.count_runs() == 7

    async def test_with_filters_applies_where_clause(self) -> None:
        pool, _ = _make_pool()
        pool.fetchrow = AsyncMock(return_value={"cnt": 3})

        store = PostgresTraceStore(pool)
        result = await store.count_runs(status="completed")
        assert result == 3
        sql = pool.fetchrow.await_args[0][0]
        assert "WHERE" in sql
        assert "status = $1" in sql

    async def test_returns_zero_when_row_is_none(self) -> None:
        pool, _ = _make_pool()
        pool.fetchrow = AsyncMock(return_value=None)

        store = PostgresTraceStore(pool)
        assert await store.count_runs() == 0

    async def test_parent_run_id_default_unset_omits_predicate(self) -> None:
        pool, _ = _make_pool()
        pool.fetchrow = AsyncMock(return_value={"cnt": 0})

        store = PostgresTraceStore(pool)
        await store.count_runs()

        sql = pool.fetchrow.await_args[0][0]
        assert "parent_run_id" not in sql

    async def test_parent_run_id_none_filters_top_level_only(self) -> None:
        pool, _ = _make_pool()
        pool.fetchrow = AsyncMock(return_value={"cnt": 0})

        store = PostgresTraceStore(pool)
        await store.count_runs(parent_run_id=None)

        sql = pool.fetchrow.await_args[0][0]
        assert "parent_run_id IS NULL" in sql

    async def test_parent_run_id_string_filters_children(self) -> None:
        pool, _ = _make_pool()
        pool.fetchrow = AsyncMock(return_value={"cnt": 0})

        store = PostgresTraceStore(pool)
        await store.count_runs(parent_run_id="run-parent")

        args = pool.fetchrow.await_args[0]
        assert "parent_run_id = $" in args[0]
        assert "run-parent" in args


# ── delete_run ─────────────────────────────────────────────


class TestDeleteRun:
    async def test_found_deletes_events_and_run_returns_true(self) -> None:
        pool, conn = _make_pool()
        conn.fetchrow = AsyncMock(return_value={"id": "run-1"})

        store = PostgresTraceStore(pool)
        result = await store.delete_run("run-1")
        assert result is True

        sqls = _sql_calls(conn)
        assert any("DELETE FROM trace_events WHERE parent_id = $1" in s for s in sqls)
        assert any("DELETE FROM runs WHERE id = $1" in s for s in sqls)

    async def test_not_found_returns_false_without_deletes(self) -> None:
        pool, conn = _make_pool()
        conn.fetchrow = AsyncMock(return_value=None)

        store = PostgresTraceStore(pool)
        result = await store.delete_run("missing")
        assert result is False
        # No DELETE issued.
        sqls = _sql_calls(conn)
        assert not any("DELETE FROM" in s for s in sqls)


# ── get_span_tree ──────────────────────────────────────────


class TestGetSpanTree:
    async def test_returns_events_ordered(self) -> None:
        pool, _ = _make_pool()
        pool.fetch = AsyncMock(return_value=[_event_row(id=1), _event_row(id=2)])

        store = PostgresTraceStore(pool)
        results = await store.get_span_tree("trace-1")
        assert len(results) == 2
        sql = pool.fetch.await_args[0][0]
        assert "WHERE trace_id = $1" in sql
        assert "ORDER BY sdk_timestamp ASC, id ASC" in sql


# ── get_events_by_span ─────────────────────────────────────


class TestGetEventsBySpan:
    async def test_returns_events_for_span(self) -> None:
        pool, _ = _make_pool()
        pool.fetch = AsyncMock(return_value=[_event_row(id=1)])

        store = PostgresTraceStore(pool)
        results = await store.get_events_by_span("trace-1", "span-1")
        assert len(results) == 1
        sql = pool.fetch.await_args[0][0]
        assert "trace_id = $1 AND span_id = $2" in sql


# ── _row_to_event ──────────────────────────────────────────


class TestRowToEvent:
    def test_payload_as_dict_passthrough(self) -> None:
        row = _event_row(payload={"k": "v"})
        result = _row_to_event(row)
        assert result.payload == {"k": "v"}

    def test_payload_as_string_is_json_decoded(self) -> None:
        row = _event_row(payload='{"k": "v"}')
        result = _row_to_event(row)
        assert result.payload == {"k": "v"}


# ── _row_to_run ────────────────────────────────────────────


class TestRowToRun:
    def test_metadata_as_dict_passthrough(self) -> None:
        row = _run_row(metadata={"k": "v"})
        result = _row_to_run(row)
        assert result.metadata == {"k": "v"}

    def test_metadata_as_string_is_json_decoded(self) -> None:
        row = _run_row(metadata='{"k": "v"}')
        result = _row_to_run(row)
        assert result.metadata == {"k": "v"}

    def test_result_none_yields_none(self) -> None:
        row = _run_row(result=None)
        run = _row_to_run(row)
        assert run.result is None

    def test_result_structured_json_round_trips(self) -> None:
        payload = RunResult(
            output="answer",
            termination_reason=TerminationReason.ITERATION_LIMIT,
            total_steps=4,
        ).model_dump_json()
        row = _run_row(result=payload)
        run = _row_to_run(row)
        assert run.result == RunResult(
            output="answer",
            termination_reason=TerminationReason.ITERATION_LIMIT,
            total_steps=4,
        )

    def test_legacy_string_falls_back_to_output(self) -> None:
        """Pre-migration rows contain arbitrary strings; expose them as ``output``."""
        row = _run_row(result="not-valid-json-shape")
        run = _row_to_run(row)
        assert run.result == RunResult(output="not-valid-json-shape")

    def test_parent_run_id_none_passthrough(self) -> None:
        row = _run_row(parent_run_id=None)
        run = _row_to_run(row)
        assert run.parent_run_id is None

    def test_parent_run_id_string_passthrough(self) -> None:
        row = _run_row(parent_run_id="run-parent")
        run = _row_to_run(row)
        assert run.parent_run_id == "run-parent"
