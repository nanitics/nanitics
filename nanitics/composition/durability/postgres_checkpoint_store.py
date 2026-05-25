"""PostgresCheckpointStore — persistent checkpoint storage backed by PostgreSQL.

Implements the :class:`CheckpointStore` protocol using ``asyncpg``
for persisting execution checkpoints across process restarts.
"""

from __future__ import annotations

import asyncpg

from nanitics.composition.durability.models import RunCheckpoint

CHECKPOINT_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    data          JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_run_created
    ON checkpoints (run_id, created_at DESC, checkpoint_id DESC);
"""


def get_checkpoint_schema_sql() -> str:
    """Return the CREATE TABLE + INDEX statements for the checkpoints table."""
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
        """Delete all checkpoints for a run. Silent if no checkpoints exist."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM checkpoints WHERE run_id = $1",
                run_id,
            )
