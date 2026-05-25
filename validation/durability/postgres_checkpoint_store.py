"""PostgresCheckpointStore round-trip against real Postgres.

Pins the defining persistence properties of
:class:`PostgresCheckpointStore`: full save/load round-trip equality,
``load()`` returning the most recent checkpoint for a run, the
deterministic composite-index tie-break under same-``created_at``
collisions, and silent ``delete`` / ``delete_for_run`` semantics
matching :class:`InMemoryCheckpointStore`.

The PostgreSQL ``checkpoints`` table is dropped in the ``finally``
block so the script is re-runnable on a shared database.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from nanitics.composition.durability.models import RunCheckpoint, SuspensionInfo
from nanitics.composition.durability.postgres_checkpoint_store import (
    PostgresCheckpointStore,
    get_checkpoint_schema_sql,
)
from validation.helpers import make_postgres_pool, requires_postgres


async def _drop_table(pool: Any) -> None:
    async with pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS checkpoints")


async def _ensure_schema(pool: Any) -> None:
    async with pool.acquire() as conn:
        await conn.execute(get_checkpoint_schema_sql())


def _info() -> SuspensionInfo:
    return SuspensionInfo(
        suspension_id="sus-1",
        request_id="req-1",
        request_type="approval",
        prompt="Approve?",
    )


@requires_postgres
@pytest.mark.quick
async def test_postgres_checkpoint_store_roundtrip() -> None:
    run_id = f"checkpoint-{uuid.uuid4().hex[:8]}"
    async with make_postgres_pool() as pool:
        try:
            await _drop_table(pool)
            await _ensure_schema(pool)

            store = PostgresCheckpointStore(pool)

            base_ts = datetime.now(UTC)
            cp1 = RunCheckpoint(
                checkpoint_id="cp-1",
                run_id=run_id,
                checkpoint_type="orchestration",
                state={"step": 0},
                suspension_info=_info(),
                created_at=base_ts,
            )
            cp2 = RunCheckpoint(
                checkpoint_id="cp-2",
                run_id=run_id,
                checkpoint_type="orchestration",
                state={"step": 1},
                suspension_info=_info(),
                created_at=base_ts + timedelta(milliseconds=10),
            )
            # Tie-break: same created_at as cp2, lexically larger checkpoint_id.
            cp3 = RunCheckpoint(
                checkpoint_id="cp-3",
                run_id=run_id,
                checkpoint_type="orchestration",
                state={"step": 2},
                suspension_info=_info(),
                created_at=cp2.created_at,
            )

            # Round-trip equality.
            await store.save(cp1)
            loaded = await store.load(run_id)
            assert loaded is not None
            assert loaded.model_dump() == cp1.model_dump()

            # Most-recent wins.
            await store.save(cp2)
            loaded = await store.load(run_id)
            assert loaded is not None
            assert loaded.checkpoint_id == "cp-2"

            # Tie-break: cp-3 wins by checkpoint_id DESC.
            await store.save(cp3)
            loaded = await store.load(run_id)
            assert loaded is not None
            assert loaded.checkpoint_id == "cp-3"

            # delete(known) silently removes; cp-2 is the next-most-recent.
            await store.delete("cp-3")
            loaded = await store.load(run_id)
            assert loaded is not None
            assert loaded.checkpoint_id == "cp-2"

            # delete(unknown) silent.
            await store.delete("does-not-exist")

            # delete_for_run removes all.
            await store.delete_for_run(run_id)
            assert await store.load(run_id) is None

            # delete_for_run(unknown) silent.
            await store.delete_for_run("does-not-exist")
        finally:
            await _drop_table(pool)
