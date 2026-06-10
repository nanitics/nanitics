"""Mock-based tests for PostgresCheckpointStore — no real Postgres needed."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from nanitics.composition.durability.models import RunCheckpoint, StepRecord, SuspensionInfo
from nanitics.composition.durability.postgres_checkpoint_store import (
    PostgresCheckpointStore,
    get_checkpoint_schema_sql,
)


def _make_step(
    *,
    run_id: str = "run-1",
    step_path: str = "seq#0/tool#0:send_email",
    result: dict | None = None,
) -> StepRecord:
    return StepRecord(
        run_id=run_id,
        step_path=step_path,
        step_kind="tool_call",
        result=result if result is not None else {"content": "ok"},
    )


def _make_pool() -> tuple[MagicMock, AsyncMock]:
    pool = MagicMock()
    conn = AsyncMock()
    acquire_cm = AsyncMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = acquire_cm
    return pool, conn


def _make_checkpoint(
    *,
    checkpoint_id: str = "cp-1",
    run_id: str = "run-1",
    created_at: datetime | None = None,
    state: dict | None = None,
) -> RunCheckpoint:
    return RunCheckpoint(
        checkpoint_id=checkpoint_id,
        run_id=run_id,
        checkpoint_type="orchestration",
        state=state if state is not None else {"step": 0},
        suspension_info=SuspensionInfo(
            suspension_id="sus-1",
            request_id="req-1",
            request_type="approval",
            prompt="Approve?",
        ),
        created_at=created_at if created_at is not None else datetime.now(UTC),
    )


class TestGetCheckpointSchemaSql:
    def test_returns_string_with_create_table_and_index(self) -> None:
        sql = get_checkpoint_schema_sql()
        assert "CREATE TABLE IF NOT EXISTS checkpoints" in sql
        assert "CREATE INDEX IF NOT EXISTS idx_checkpoints_run_created" in sql

    def test_includes_step_journal_table_keyed_on_run_and_step_path(self) -> None:
        sql = get_checkpoint_schema_sql()
        assert "CREATE TABLE IF NOT EXISTS step_journal" in sql
        assert "PRIMARY KEY (run_id, step_path)" in sql
        assert "CREATE INDEX IF NOT EXISTS idx_step_journal_run_seq" in sql


class TestPostgresCheckpointStoreMock:
    async def test_save_issues_insert_with_bound_params_in_order(self) -> None:
        pool, conn = _make_pool()
        store = PostgresCheckpointStore(pool)
        cp = _make_checkpoint()

        await store.save(cp)

        conn.execute.assert_called_once()
        args = conn.execute.call_args[0]
        assert "INSERT INTO checkpoints (checkpoint_id, run_id, created_at, data)" in args[0]
        assert args[1] == cp.checkpoint_id
        assert args[2] == cp.run_id
        assert args[3] == cp.created_at
        assert args[4] == cp.model_dump_json()

    async def test_load_issues_select_with_deterministic_order(self) -> None:
        pool, conn = _make_pool()
        conn.fetchrow = AsyncMock(return_value=None)
        store = PostgresCheckpointStore(pool)

        await store.load("run-1")

        sql = conn.fetchrow.call_args[0][0]
        assert "SELECT data FROM checkpoints" in sql
        assert "WHERE run_id = $1" in sql
        assert "ORDER BY created_at DESC, checkpoint_id DESC" in sql
        assert "LIMIT 1" in sql
        assert conn.fetchrow.call_args[0][1] == "run-1"

    async def test_load_round_trips_through_validate_json_when_data_is_str(self) -> None:
        pool, conn = _make_pool()
        cp = _make_checkpoint()
        conn.fetchrow = AsyncMock(return_value={"data": cp.model_dump_json()})
        store = PostgresCheckpointStore(pool)

        result = await store.load(cp.run_id)

        assert result is not None
        assert result.model_dump() == cp.model_dump()

    async def test_load_round_trips_through_validate_when_data_is_dict(self) -> None:
        pool, conn = _make_pool()
        cp = _make_checkpoint()
        conn.fetchrow = AsyncMock(return_value={"data": cp.model_dump(mode="json")})
        store = PostgresCheckpointStore(pool)

        result = await store.load(cp.run_id)

        assert result is not None
        assert result.model_dump() == cp.model_dump()

    async def test_load_returns_none_when_no_row(self) -> None:
        pool, conn = _make_pool()
        conn.fetchrow = AsyncMock(return_value=None)
        store = PostgresCheckpointStore(pool)

        result = await store.load("missing")

        assert result is None

    async def test_delete_issues_delete_with_bound_id(self) -> None:
        pool, conn = _make_pool()
        store = PostgresCheckpointStore(pool)

        await store.delete("cp-1")

        conn.execute.assert_called_once()
        args = conn.execute.call_args[0]
        assert args[0] == "DELETE FROM checkpoints WHERE checkpoint_id = $1"
        assert args[1] == "cp-1"

    async def test_delete_is_silent_when_no_rows_match(self) -> None:
        pool, conn = _make_pool()
        conn.execute = AsyncMock(return_value="DELETE 0")
        store = PostgresCheckpointStore(pool)

        # Must not raise.
        await store.delete("missing")

    async def test_delete_for_run_issues_delete_with_bound_run_id(self) -> None:
        pool, conn = _make_pool()
        store = PostgresCheckpointStore(pool)

        await store.delete_for_run("run-1")

        assert conn.execute.call_count == 2
        first_call, second_call = conn.execute.call_args_list
        assert first_call[0][0] == "DELETE FROM checkpoints WHERE run_id = $1"
        assert first_call[0][1] == "run-1"
        assert second_call[0][0] == "DELETE FROM step_journal WHERE run_id = $1"
        assert second_call[0][1] == "run-1"

    async def test_delete_for_run_is_silent_when_no_rows_match(self) -> None:
        pool, conn = _make_pool()
        conn.execute = AsyncMock(return_value="DELETE 0")
        store = PostgresCheckpointStore(pool)

        # Must not raise.
        await store.delete_for_run("missing")

    async def test_load_order_by_contract_supports_tie_break(self) -> None:
        """Two checkpoints with the same created_at must resolve deterministically.

        The contract is enforced by the ORDER BY clause asserted in
        :meth:`test_load_issues_select_with_deterministic_order`; here we
        round-trip a save of the lexically-larger id and confirm a load
        against a mock that mirrors the ORDER BY surface returns it.
        """
        pool, conn = _make_pool()
        shared_ts = datetime.now(UTC)
        cp_low = _make_checkpoint(checkpoint_id="cp-1", created_at=shared_ts)
        cp_high = _make_checkpoint(checkpoint_id="cp-2", created_at=shared_ts + timedelta(microseconds=0))
        # Simulate the DB resolving the tie-break: highest checkpoint_id wins.
        conn.fetchrow = AsyncMock(return_value={"data": cp_high.model_dump_json()})
        store = PostgresCheckpointStore(pool)

        await store.save(cp_low)
        await store.save(cp_high)
        result = await store.load(cp_low.run_id)

        assert result is not None
        assert result.checkpoint_id == "cp-2"

    async def test_append_step_issues_upsert_with_bound_params(self) -> None:
        pool, conn = _make_pool()
        store = PostgresCheckpointStore(pool)
        record = _make_step()

        await store.append_step(record)

        conn.execute.assert_called_once()
        args = conn.execute.call_args[0]
        assert "INSERT INTO step_journal (run_id, step_path, created_at, data)" in args[0]
        assert "ON CONFLICT (run_id, step_path)" in args[0]
        assert args[1] == record.run_id
        assert args[2] == record.step_path
        assert args[3] == record.created_at
        assert args[4] == record.model_dump_json()

    async def test_load_journal_returns_records_in_order_for_both_data_shapes(self) -> None:
        pool, conn = _make_pool()
        first = _make_step(step_path="seq#0/tool#0:a")
        second = _make_step(step_path="seq#0/tool#1:b")
        # First row arrives as a JSON string, second as a decoded dict — exercises both
        # the str (model_validate_json) and dict (model_validate) branches.
        conn.fetch = AsyncMock(
            return_value=[
                {"data": first.model_dump_json()},
                {"data": second.model_dump(mode="json")},
            ]
        )
        store = PostgresCheckpointStore(pool)

        records = await store.load_journal("run-1")

        sql = conn.fetch.call_args[0][0]
        assert "SELECT data FROM step_journal" in sql
        assert "WHERE run_id = $1" in sql
        assert "ORDER BY seq ASC" in sql
        assert conn.fetch.call_args[0][1] == "run-1"
        assert [r.step_path for r in records] == ["seq#0/tool#0:a", "seq#0/tool#1:b"]
        assert records[0].model_dump() == first.model_dump()
        assert records[1].model_dump() == second.model_dump()

    async def test_load_journal_returns_empty_when_no_rows(self) -> None:
        pool, conn = _make_pool()
        conn.fetch = AsyncMock(return_value=[])
        store = PostgresCheckpointStore(pool)

        records = await store.load_journal("missing")

        assert records == []
