"""PostgresCheckpointStore — persistent checkpoint storage backed by PostgreSQL.

Implements the :class:`CheckpointStore` protocol using ``asyncpg``
for persisting execution checkpoints across process restarts.
"""

from __future__ import annotations

import asyncpg

from nanitics.composition.durability.models import RunCheckpoint, StepRecord

CHECKPOINT_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    data          JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_run_created
    ON checkpoints (run_id, created_at DESC, checkpoint_id DESC);

CREATE TABLE IF NOT EXISTS step_journal (
    run_id     TEXT NOT NULL,
    step_path  TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    seq        BIGSERIAL,
    data       JSONB NOT NULL,
    PRIMARY KEY (run_id, step_path)
);

CREATE INDEX IF NOT EXISTS idx_step_journal_run_seq
    ON step_journal (run_id, seq);
"""


def get_checkpoint_schema_sql() -> str:
    """Return the CREATE TABLE + INDEX statements for the checkpoint and journal tables.

    Covers the ``checkpoints`` cursor table and the append-only
    ``step_journal`` table, the latter keyed ``(run_id, step_path)`` so
    ``append_step`` is idempotent on the step key and ``load_journal`` can
    return records in append order via the monotonic ``seq`` column.
    """
    return CHECKPOINT_SCHEMA_SQL


class PostgresCheckpointStore:
    """Persistent checkpoint store backed by PostgreSQL via ``asyncpg``.

    Stores each :class:`RunCheckpoint` as a single JSONB blob keyed by
    ``checkpoint_id``. ``load(run_id)`` returns the most recent
    checkpoint for a run, ordered by ``created_at DESC`` with
    ``checkpoint_id DESC`` as a deterministic tie-break.

    Args:
        pool: An initialised ``asyncpg.Pool``.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def save(self, checkpoint: RunCheckpoint) -> None:
        """Persist a checkpoint.

        The full :class:`RunCheckpoint` payload is serialised via
        ``model_dump_json()`` into the ``data`` JSONB column; the
        model's existing ``checkpoint_id``, ``run_id`` and ``created_at``
        are also bound to their own columns for indexed querying.
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO checkpoints (checkpoint_id, run_id, created_at, data)
                VALUES ($1, $2, $3, $4)
                """,
                checkpoint.checkpoint_id,
                checkpoint.run_id,
                checkpoint.created_at,
                checkpoint.model_dump_json(),
            )

    async def load(self, run_id: str) -> RunCheckpoint | None:
        """Load the most recent checkpoint for a run, or ``None`` if none exists.

        Ordering is ``created_at DESC, checkpoint_id DESC`` so that
        same-microsecond writes resolve deterministically by id.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT data FROM checkpoints
                WHERE run_id = $1
                ORDER BY created_at DESC, checkpoint_id DESC
                LIMIT 1
                """,
                run_id,
            )
        if row is None:
            return None
        raw = row["data"]
        if isinstance(raw, str):
            return RunCheckpoint.model_validate_json(raw)
        return RunCheckpoint.model_validate(raw)

    async def delete(self, checkpoint_id: str) -> None:
        """Delete a specific checkpoint. Silent if the checkpoint does not exist."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM checkpoints WHERE checkpoint_id = $1",
                checkpoint_id,
            )

    async def delete_for_run(self, run_id: str) -> None:
        """Delete all checkpoints and journal entries for a run.

        Silent if nothing exists for the run.
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM checkpoints WHERE run_id = $1",
                run_id,
            )
            await conn.execute(
                "DELETE FROM step_journal WHERE run_id = $1",
                run_id,
            )

    async def append_step(self, record: StepRecord) -> None:
        """Append a completed-step result to the journal.

        Idempotent on ``(run_id, step_path)`` via ``ON CONFLICT DO UPDATE``:
        re-appending the same step key overwrites ``data`` (last-write-wins)
        while retaining the row's original ``seq``, so append order is
        preserved. ``data`` carries the full :class:`StepRecord` JSON.
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO step_journal (run_id, step_path, created_at, data)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (run_id, step_path)
                DO UPDATE SET data = EXCLUDED.data, created_at = EXCLUDED.created_at
                """,
                record.run_id,
                record.step_path,
                record.created_at,
                record.model_dump_json(),
            )

    async def load_journal(self, run_id: str) -> list[StepRecord]:
        """Return all step records for a run, in append order (by ``seq``).

        Returns an empty list when the run has no journal entries.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data FROM step_journal
                WHERE run_id = $1
                ORDER BY seq ASC
                """,
                run_id,
            )
        records: list[StepRecord] = []
        for row in rows:
            raw = row["data"]
            if isinstance(raw, str):
                records.append(StepRecord.model_validate_json(raw))
            else:
                records.append(StepRecord.model_validate(raw))
        return records
