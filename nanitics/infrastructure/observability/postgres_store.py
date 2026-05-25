"""PostgresTraceStore — persistent trace storage backed by PostgreSQL.

Implements the :class:`PersistentTraceStore` protocol using ``asyncpg``
for bulk insertion, filtered querying, and aggregation of trace events.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import ValidationError

from nanitics.infrastructure.observability.storage import (
    _UNSET,
    DEFAULT_EVENTS_LIMIT,
    DEFAULT_RUNS_LIMIT,
    RunRecord,
    RunResult,
    RunStatus,
    StoredTraceEvent,
    TraceEventRecord,
    TraceSummaryStats,
    _UnsetType,
)

if TYPE_CHECKING:
    import asyncpg

    from nanitics.infrastructure.observability.levels import TraceLevel

SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS trace_events (
    id BIGSERIAL PRIMARY KEY,
    parent_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    level TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    span_id TEXT NOT NULL,
    parent_span_id TEXT,
    payload JSONB NOT NULL,
    sdk_timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trace_events_parent
    ON trace_events (parent_id, id);
CREATE INDEX IF NOT EXISTS idx_trace_events_level
    ON trace_events (parent_id, level, id);
CREATE INDEX IF NOT EXISTS idx_trace_events_trace
    ON trace_events (trace_id, id);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}',
    error TEXT,
    result TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_status
    ON runs (status, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_trace
    ON runs (trace_id);
"""


# ---------------------------------------------------------------------------
# Versioned schema migrations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Migration:
    version: int
    description: str
    up_sql: list[str]


def _build_migrations(events_table: str, runs_table: str) -> list[_Migration]:
    """Build the ordered migration list, parameterised by table names."""
    baseline_sql = SCHEMA_SQL
    if events_table != "trace_events":
        baseline_sql = baseline_sql.replace("trace_events", events_table)
    if runs_table != "runs":
        baseline_sql = baseline_sql.replace(" runs (", f" {runs_table} (")
        baseline_sql = baseline_sql.replace("idx_runs_", f"idx_{runs_table}_")

    return [
        _Migration(
            version=1,
            description="Baseline schema — create trace_events and runs tables",
            up_sql=[baseline_sql],
        ),
        _Migration(
            version=2,
            description="Fix column types — ensure parent_id is TEXT with NOT NULL",
            up_sql=[
                # Idempotent: only alter if the column is not already TEXT.
                f"""
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = '{events_table}'
                          AND column_name = 'parent_id'
                          AND data_type <> 'text'
                    ) THEN
                        ALTER TABLE {events_table}
                            ALTER COLUMN parent_id TYPE TEXT USING parent_id::TEXT;
                    END IF;

                    -- Ensure NOT NULL on columns that require it
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = '{events_table}'
                          AND column_name = 'parent_id'
                          AND is_nullable = 'YES'
                    ) THEN
                        ALTER TABLE {events_table}
                            ALTER COLUMN parent_id SET NOT NULL;
                    END IF;
                END
                $$;
                """,
            ],
        ),
        _Migration(
            version=3,
            description="Add parent_run_id to runs for hierarchical specialist runs",
            up_sql=[
                f"""
                ALTER TABLE {runs_table}
                    ADD COLUMN parent_run_id TEXT NULL
                        REFERENCES {runs_table}(id) ON DELETE CASCADE
                """,
                f"""
                CREATE INDEX idx_{runs_table}_parent
                    ON {runs_table} (parent_run_id)
                    WHERE parent_run_id IS NOT NULL
                """,
            ],
        ),
    ]


LATEST_VERSION = 3


def get_schema_sql(*, table_name: str = "trace_events", runs_table_name: str = "runs") -> str:
    """Return the CREATE TABLE + INDEX statements.

    Applications can execute this during database setup to ensure the
    tables exist.  If custom names are used, they must also be
    passed to the :class:`PostgresTraceStore` constructor.

    .. note::

        For production use, prefer :meth:`PostgresTraceStore.ensure_schema`
        which handles both initial creation and schema evolution via
        versioned migrations.
    """
    sql = SCHEMA_SQL
    if table_name != "trace_events":
        sql = sql.replace("trace_events", table_name)
    if runs_table_name != "runs":
        sql = sql.replace(" runs (", f" {runs_table_name} (")
        sql = sql.replace("idx_runs_", f"idx_{runs_table_name}_")
    return sql


class PostgresTraceStore:
    """Persistent trace store backed by PostgreSQL via ``asyncpg``.

    Args:
        pool: An initialised ``asyncpg.Pool``.
        table_name: Name of the trace events table (default ``"trace_events"``).
    """

    def __init__(self, pool: asyncpg.Pool, *, table_name: str = "trace_events", runs_table: str = "runs") -> None:
        self._pool = pool
        self._table = table_name
        self._runs_table = runs_table
        self._version_table = f"_{table_name}_schema_version"

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    async def ensure_schema(self) -> None:
        """Create or migrate trace tables to the latest schema version.

        Handles three cases:

        - **Fresh database:** Creates all tables from scratch.
        - **Legacy database:** Tables exist from ``CREATE TABLE IF NOT EXISTS``
          but no version tracking — applies corrective migrations.
        - **Current database:** Already at latest version — no-op.

        v3 adds ``parent_run_id`` to the runs table along with a
        self-referencing foreign key (``ON DELETE CASCADE``) and a partial
        index on non-null values. Idempotency guarantees apply: re-running
        ``ensure_schema`` on a database already at v3 is a no-op.

        Raises:
            RuntimeError: The schema version row reports a baseline that
                must have created the configured ``runs_table``, but the
                table is missing from ``information_schema``. The usual
                cause is another store instance sharing ``table_name`` and
                using a different ``runs_table`` configuration; the SDK
                refuses to silently re-run migrations under a different
                table name.

        Uses a PostgreSQL advisory lock to prevent concurrent migration races
        when multiple application instances start simultaneously.
        """
        # Derive a stable advisory lock key from the table name.
        lock_key = zlib.crc32(self._table.encode()) & 0x7FFFFFFF

        async with self._pool.acquire() as conn:
            # Advisory lock scoped to the connection (released on connection return).
            await conn.execute("SELECT pg_advisory_lock($1)", lock_key)
            try:
                await self._apply_migrations(conn)
            finally:
                await conn.execute("SELECT pg_advisory_unlock($1)", lock_key)

    async def _apply_migrations(self, conn: asyncpg.Connection) -> None:
        """Detect current state and apply pending migrations."""
        version_exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = $1
            )
            """,
            self._version_table,
        )

        if not version_exists:
            events_exist = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = $1
                )
                """,
                self._table,
            )

            # Create the version tracking table
            await conn.execute(
                f"""
                CREATE TABLE {self._version_table} (
                    version INTEGER NOT NULL
                )
                """
            )

            if events_exist:
                # Legacy database: tables exist but no version tracking.
                # Mark as v1 (baseline already applied), then run v2+.
                await conn.execute(f"INSERT INTO {self._version_table} (version) VALUES (1)")
            else:
                # Fresh database: no tables at all.
                await conn.execute(f"INSERT INTO {self._version_table} (version) VALUES (0)")

        current_version = await conn.fetchval(f"SELECT version FROM {self._version_table}")

        # Schema-sibling guard. The version row reports the schema lineage
        # for ``self._table`` (events table). If another store instance
        # shares ``self._table`` but was constructed with a different
        # ``runs_table`` name, the version row will report a baseline that
        # claims to have created ``self._runs_table`` while that table does
        # not actually exist. Surface this loud rather than silently
        # re-running migrations under a fresh ``runs_table`` name (which
        # would be the per-pair version-keying redesign the SDK declined).
        if current_version >= 1:
            runs_table_exists = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = $1
                )
                """,
                self._runs_table,
            )
            if not runs_table_exists:
                raise RuntimeError(
                    f"PostgresTraceStore.ensure_schema: schema version row reports "
                    f"v{current_version} but configured runs_table {self._runs_table!r} "
                    f"does not exist. This usually means another PostgresTraceStore "
                    f"instance sharing table_name={self._table!r} created the version "
                    f"row under a different runs_table configuration. The current store "
                    f"will not silently re-run migrations under a different table name — "
                    f"reconcile the configuration or use a distinct table_name."
                )

        migrations = _build_migrations(self._table, self._runs_table)

        for migration in migrations:
            if migration.version <= current_version:
                continue

            async with conn.transaction():
                for sql in migration.up_sql:
                    await conn.execute(sql)
                await conn.execute(
                    f"UPDATE {self._version_table} SET version = $1",
                    migration.version,
                )

    # ------------------------------------------------------------------
    # PersistentTraceStore protocol
    # ------------------------------------------------------------------

    async def save_events_batch(self, parent_id: str, events: list[TraceEventRecord]) -> None:
        """Bulk-insert trace event records."""
        if not events:
            return
        async with self._pool.acquire() as conn:
            await conn.executemany(
                f"""
                INSERT INTO {self._table}
                    (parent_id, event_type, level, trace_id, span_id,
                     parent_span_id, payload, sdk_timestamp)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                [
                    (
                        parent_id,
                        e.event_type,
                        e.level,
                        e.trace_id,
                        e.span_id,
                        e.parent_span_id,
                        json.dumps(e.payload),
                        e.sdk_timestamp,
                    )
                    for e in events
                ],
            )

    async def query_events(
        self,
        parent_id: str,
        *,
        levels: list[TraceLevel] | None = None,
        event_types: list[str] | None = None,
        after_id: int | None = None,
        limit: int = DEFAULT_EVENTS_LIMIT,
    ) -> list[StoredTraceEvent]:
        """Filtered, cursor-paginated retrieval of trace events."""
        conditions = ["parent_id = $1"]
        params: list[object] = [parent_id]

        if after_id is not None:
            params.append(after_id)
            conditions.append(f"id > ${len(params)}")

        if levels:
            params.append(levels)
            conditions.append(f"level = ANY(${len(params)})")

        if event_types:
            params.append(event_types)
            conditions.append(f"event_type = ANY(${len(params)})")

        params.append(limit)
        where = " AND ".join(conditions)

        rows = await self._pool.fetch(
            f"""
            SELECT id, event_type, level, trace_id, span_id, parent_span_id,
                   payload, sdk_timestamp
            FROM {self._table}
            WHERE {where}
            ORDER BY id ASC
            LIMIT ${len(params)}
            """,
            *params,
        )
        return [_row_to_event(row) for row in rows]

    async def get_event(self, event_id: int) -> StoredTraceEvent | None:
        """Retrieve a single event by database ID."""
        row = await self._pool.fetchrow(
            f"""
            SELECT id, event_type, level, trace_id, span_id, parent_span_id,
                   payload, sdk_timestamp
            FROM {self._table}
            WHERE id = $1
            """,
            event_id,
        )
        if row is None:
            return None
        return _row_to_event(row)

    async def get_summary(self, parent_id: str) -> TraceSummaryStats:
        """Return aggregated statistics for all events under *parent_id*."""
        pool = self._pool
        table = self._table

        # Basic counts and time range
        agg = await pool.fetchrow(
            f"""
            SELECT
                count(*) AS total_events,
                count(*) FILTER (WHERE level = 'info') AS info_count,
                count(*) FILTER (WHERE level = 'debug') AS debug_count,
                count(*) FILTER (WHERE level = 'verbose') AS verbose_count,
                count(*) FILTER (WHERE event_type = 'tool.invoke') AS tool_calls,
                count(*) FILTER (WHERE event_type LIKE 'agent.error%%'
                                    OR event_type LIKE 'workflow.error%%') AS errors,
                min(sdk_timestamp) AS first_ts,
                max(sdk_timestamp) AS last_ts
            FROM {table}
            WHERE parent_id = $1
            """,
            parent_id,
        )

        # Token aggregation from llm.response payloads
        # Payload stores usage as a nested object via model_dump
        tokens = await pool.fetchrow(
            f"""
            SELECT
                count(*) AS llm_calls,
                coalesce(sum(
                    CASE
                        WHEN payload->'usage'->>'input_tokens' IS NOT NULL
                        THEN (payload->'usage'->>'input_tokens')::int
                        WHEN payload->>'input_tokens' IS NOT NULL
                        THEN (payload->>'input_tokens')::int
                        ELSE 0
                    END
                ), 0) AS input_tokens,
                coalesce(sum(
                    CASE
                        WHEN payload->'usage'->>'output_tokens' IS NOT NULL
                        THEN (payload->'usage'->>'output_tokens')::int
                        WHEN payload->>'output_tokens' IS NOT NULL
                        THEN (payload->>'output_tokens')::int
                        ELSE 0
                    END
                ), 0) AS output_tokens,
                coalesce(sum(
                    CASE
                        WHEN payload->'usage'->>'cache_creation_input_tokens' IS NOT NULL
                        THEN (payload->'usage'->>'cache_creation_input_tokens')::int
                        WHEN payload->>'cache_creation_input_tokens' IS NOT NULL
                        THEN (payload->>'cache_creation_input_tokens')::int
                        ELSE 0
                    END
                ), 0) AS cache_creation_tokens,
                coalesce(sum(
                    CASE
                        WHEN payload->'usage'->>'cache_read_input_tokens' IS NOT NULL
                        THEN (payload->'usage'->>'cache_read_input_tokens')::int
                        WHEN payload->>'cache_read_input_tokens' IS NOT NULL
                        THEN (payload->>'cache_read_input_tokens')::int
                        ELSE 0
                    END
                ), 0) AS cache_read_tokens
            FROM {table}
            WHERE parent_id = $1 AND event_type = 'llm.response'
            """,
            parent_id,
        )

        # Unique agent names
        agent_rows = await pool.fetch(
            f"""
            SELECT DISTINCT payload->>'agent_name' AS agent
            FROM {table}
            WHERE parent_id = $1
              AND event_type = 'agent.start'
              AND payload->>'agent_name' IS NOT NULL
            """,
            parent_id,
        )

        duration_ms: int | None = None
        if agg and agg["first_ts"] and agg["last_ts"]:
            delta = agg["last_ts"] - agg["first_ts"]
            duration_ms = int(delta.total_seconds() * 1000)

        return TraceSummaryStats(
            total_events=agg["total_events"] if agg else 0,
            events_by_level={
                "info": agg["info_count"] if agg else 0,
                "debug": agg["debug_count"] if agg else 0,
                "verbose": agg["verbose_count"] if agg else 0,
            },
            llm_calls=tokens["llm_calls"] if tokens else 0,
            tool_calls=agg["tool_calls"] if agg else 0,
            total_input_tokens=tokens["input_tokens"] if tokens else 0,
            total_output_tokens=tokens["output_tokens"] if tokens else 0,
            total_duration_ms=duration_ms,
            agent_names=[row["agent"] for row in agent_rows],
            errors=agg["errors"] if agg else 0,
            cache_creation_tokens=tokens["cache_creation_tokens"] if tokens else 0,
            cache_read_tokens=tokens["cache_read_tokens"] if tokens else 0,
        )

    # ------------------------------------------------------------------
    # Run management
    # ------------------------------------------------------------------

    async def register_run(
        self,
        run_id: str,
        trace_id: str,
        metadata: dict[str, Any],
        *,
        parent_run_id: str | None = None,
    ) -> None:
        """Register a new run with initial 'running' status.

        ``parent_run_id`` links a specialist (child) run to the run that
        dispatched it. Pass ``None`` (the default) for top-level runs. The
        column carries a self-referencing foreign key with ``ON DELETE
        CASCADE``: deleting a parent run also deletes its children (and,
        transitively, their descendants). No pre-insert FK existence check
        is performed — if the referenced ``parent_run_id`` does not exist,
        Postgres surfaces the FK violation directly.

        The SDK does **not** enforce that the child's ``trace_id`` matches
        the parent's; the caller decides whether parent and child share a
        trace or each carry their own.
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self._runs_table}
                    (id, trace_id, status, started_at, metadata, parent_run_id)
                VALUES ($1, $2, 'running', NOW(), $3, $4)
                """,
                run_id,
                trace_id,
                json.dumps(metadata),
                parent_run_id,
            )

    async def update_run_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        error: str | None = None,
        result: RunResult | None = None,
    ) -> None:
        """Update run status and optional error message / result."""
        result_json = result.model_dump_json() if result is not None else None
        async with self._pool.acquire() as conn:
            if status in ("completed", "failed"):
                await conn.execute(
                    f"""
                    UPDATE {self._runs_table}
                    SET status = $1, completed_at = NOW(), error = $2, result = $3
                    WHERE id = $4
                    """,
                    status,
                    error,
                    result_json,
                    run_id,
                )
            else:
                await conn.execute(
                    f"""
                    UPDATE {self._runs_table}
                    SET status = $1, error = $2, result = $3
                    WHERE id = $4
                    """,
                    status,
                    error,
                    result_json,
                    run_id,
                )

    async def get_run(self, run_id: str) -> RunRecord | None:
        """Retrieve a run by ID."""
        row = await self._pool.fetchrow(
            f"""
            SELECT id, trace_id, status, started_at, completed_at, metadata, error, result,
                   parent_run_id
            FROM {self._runs_table}
            WHERE id = $1
            """,
            run_id,
        )
        if row is None:
            return None
        return _row_to_run(row)

    def _build_run_conditions(
        self,
        params: list[object],
        *,
        status: str | None = None,
        started_after: datetime | None = None,
        started_before: datetime | None = None,
        search: str | None = None,
        parent_run_id: str | None | _UnsetType = _UNSET,
    ) -> list[str]:
        """Build WHERE conditions and append to *params*. Returns condition list."""
        conditions: list[str] = []
        if status is not None:
            params.append(status)
            conditions.append(f"status = ${len(params)}")
        if started_after is not None:
            params.append(started_after)
            conditions.append(f"started_at >= ${len(params)}")
        if started_before is not None:
            params.append(started_before)
            conditions.append(f"started_at < ${len(params)}")
        if search is not None:
            params.append(f"%{search}%")
            conditions.append(f"metadata::text ILIKE ${len(params)}")
        if not isinstance(parent_run_id, _UnsetType):
            if parent_run_id is None:
                conditions.append("parent_run_id IS NULL")
            else:
                params.append(parent_run_id)
                conditions.append(f"parent_run_id = ${len(params)}")
        return conditions

    _SORT_COLUMNS: ClassVar[dict[str, str]] = {
        "started_at_desc": "started_at DESC",
        "started_at_asc": "started_at ASC",
        "duration_desc": "(COALESCE(completed_at, started_at) - started_at) DESC",
        "duration_asc": "(COALESCE(completed_at, started_at) - started_at) ASC",
    }

    async def list_runs(
        self,
        *,
        status: str | None = None,
        started_after: datetime | None = None,
        started_before: datetime | None = None,
        sort: str = "started_at_desc",
        search: str | None = None,
        parent_run_id: str | None | _UnsetType = _UNSET,
        limit: int = DEFAULT_RUNS_LIMIT,
        offset: int = 0,
    ) -> list[RunRecord]:
        """List runs with optional filters, ordered by *sort*.

        ``parent_run_id`` is a three-state filter: omitted (the default
        ``_UNSET`` sentinel) applies no filter and returns runs at every
        level; ``None`` restricts to top-level runs only
        (``parent_run_id IS NULL``); a ``str`` restricts to children of
        that parent. The sentinel default preserves the prior behavior of
        every existing caller.
        """
        params: list[object] = []
        conditions = self._build_run_conditions(
            params,
            status=status,
            started_after=started_after,
            started_before=started_before,
            search=search,
            parent_run_id=parent_run_id,
        )

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        order_by = self._SORT_COLUMNS.get(sort, "started_at DESC")

        params.append(limit)
        limit_idx = len(params)
        params.append(offset)
        offset_idx = len(params)

        rows = await self._pool.fetch(
            f"""
            SELECT id, trace_id, status, started_at, completed_at, metadata, error, result,
                   parent_run_id
            FROM {self._runs_table}
            {where}
            ORDER BY {order_by}
            LIMIT ${limit_idx} OFFSET ${offset_idx}
            """,
            *params,
        )
        return [_row_to_run(row) for row in rows]

    async def count_runs(
        self,
        *,
        status: str | None = None,
        started_after: datetime | None = None,
        started_before: datetime | None = None,
        search: str | None = None,
        parent_run_id: str | None | _UnsetType = _UNSET,
    ) -> int:
        """Return total count of runs matching the given filters.

        ``parent_run_id`` follows the same three-state semantics as
        :meth:`list_runs`.
        """
        params: list[object] = []
        conditions = self._build_run_conditions(
            params,
            status=status,
            started_after=started_after,
            started_before=started_before,
            search=search,
            parent_run_id=parent_run_id,
        )

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        row = await self._pool.fetchrow(
            f"""
            SELECT count(*) AS cnt
            FROM {self._runs_table}
            {where}
            """,
            *params,
        )
        return row["cnt"] if row else 0

    async def delete_run(self, run_id: str) -> bool:
        """Delete a run and all associated trace events in a transaction.

        The runs table carries a self-referencing FK with ``ON DELETE
        CASCADE`` on ``parent_run_id``: deleting a parent silently deletes
        its children (and, transitively, descendants). The explicit DELETE
        on the events table inside this method covers the events of the
        targeted run only; children's events are left to be cleaned up
        separately because the events table has no FK driving its own
        cascade today.
        """
        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                f"SELECT id FROM {self._runs_table} WHERE id = $1",
                run_id,
            )
            if row is None:
                return False
            await conn.execute(
                f"DELETE FROM {self._table} WHERE parent_id = $1",
                run_id,
            )
            await conn.execute(
                f"DELETE FROM {self._runs_table} WHERE id = $1",
                run_id,
            )
            return True

    # ------------------------------------------------------------------
    # Hierarchy queries
    # ------------------------------------------------------------------

    async def get_span_tree(self, trace_id: str) -> list[StoredTraceEvent]:
        """Return all events for a trace ordered for tree reconstruction."""
        rows = await self._pool.fetch(
            f"""
            SELECT id, event_type, level, trace_id, span_id, parent_span_id,
                   payload, sdk_timestamp
            FROM {self._table}
            WHERE trace_id = $1
            ORDER BY sdk_timestamp ASC, id ASC
            """,
            trace_id,
        )
        return [_row_to_event(row) for row in rows]

    async def get_events_by_span(self, trace_id: str, span_id: str) -> list[StoredTraceEvent]:
        """Return events within a specific span."""
        rows = await self._pool.fetch(
            f"""
            SELECT id, event_type, level, trace_id, span_id, parent_span_id,
                   payload, sdk_timestamp
            FROM {self._table}
            WHERE trace_id = $1 AND span_id = $2
            ORDER BY sdk_timestamp ASC, id ASC
            """,
            trace_id,
            span_id,
        )
        return [_row_to_event(row) for row in rows]


def _row_to_event(row: asyncpg.Record) -> StoredTraceEvent:
    """Convert an asyncpg row to a :class:`StoredTraceEvent`."""
    payload = row["payload"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    return StoredTraceEvent(
        id=row["id"],
        event_type=row["event_type"],
        level=row["level"],
        trace_id=row["trace_id"],
        span_id=row["span_id"],
        parent_span_id=row["parent_span_id"],
        payload=payload,
        sdk_timestamp=row["sdk_timestamp"],
    )


def _row_to_run(row: asyncpg.Record) -> RunRecord:
    """Convert an asyncpg row to a :class:`RunRecord`."""
    metadata = row["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return RunRecord(
        id=row["id"],
        trace_id=row["trace_id"],
        status=row["status"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        metadata=metadata,
        error=row["error"],
        result=_parse_result(row["result"]),
        parent_run_id=row["parent_run_id"],
    )


def _parse_result(raw: str | None) -> RunResult | None:
    """Parse a stored result column into :class:`RunResult`.

    Legacy rows written before the structured-result migration contain
    arbitrary strings (the old ``_serialize_result`` output). Those are
    surfaced as ``RunResult(output=<raw>)`` so consumers never see a
    typed-validation failure on historical data.
    """
    if raw is None:
        return None
    try:
        return RunResult.model_validate_json(raw)
    except ValidationError:
        return RunResult(output=raw)
