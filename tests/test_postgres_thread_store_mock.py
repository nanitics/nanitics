"""Mock-based tests for PostgresThreadStore — no real Postgres needed."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from nanitics.composition.threads.postgres_thread_store import (
    PostgresThreadStore,
    get_thread_schema_sql,
)
from nanitics.infrastructure.llm.protocol import Message


def _make_pool() -> tuple[MagicMock, AsyncMock]:
    pool = MagicMock()
    conn = AsyncMock()
    acquire_cm = AsyncMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = acquire_cm
    return pool, conn


class TestGetThreadSchemaSql:
    def test_returns_string_with_create_table_and_primary_key(self) -> None:
        sql = get_thread_schema_sql()
        assert "CREATE TABLE IF NOT EXISTS thread_messages" in sql
        assert "PRIMARY KEY (thread_key, seq)" in sql
        assert "BIGSERIAL" in sql


class TestPostgresThreadStoreMock:
    async def test_load_issues_select_ordered_by_seq_asc(self) -> None:
        pool, conn = _make_pool()
        conn.fetch = AsyncMock(return_value=[])
        store = PostgresThreadStore(pool)

        await store.load("t1")

        sql = conn.fetch.call_args[0][0]
        assert "SELECT data FROM thread_messages" in sql
        assert "WHERE thread_key = $1" in sql
        assert "ORDER BY seq ASC" in sql
        assert conn.fetch.call_args[0][1] == "t1"

    async def test_load_returns_empty_list_when_no_rows(self) -> None:
        pool, conn = _make_pool()
        conn.fetch = AsyncMock(return_value=[])
        store = PostgresThreadStore(pool)

        result = await store.load("missing")

        assert result == []

    async def test_load_round_trips_messages_in_order_when_data_is_str(self) -> None:
        pool, conn = _make_pool()
        m1 = Message(role="user", content="hello")
        m2 = Message(role="assistant", content="hi back")
        conn.fetch = AsyncMock(
            return_value=[
                {"data": m1.model_dump_json()},
                {"data": m2.model_dump_json()},
            ]
        )
        store = PostgresThreadStore(pool)

        result = await store.load("t1")

        assert [msg.model_dump() for msg in result] == [m1.model_dump(), m2.model_dump()]

    async def test_load_round_trips_when_data_is_dict(self) -> None:
        pool, conn = _make_pool()
        m1 = Message(role="user", content="hello")
        conn.fetch = AsyncMock(return_value=[{"data": m1.model_dump(mode="json")}])
        store = PostgresThreadStore(pool)

        result = await store.load("t1")

        assert len(result) == 1
        assert result[0].model_dump() == m1.model_dump()

    async def test_append_is_noop_for_empty_messages(self) -> None:
        pool, _conn = _make_pool()
        store = PostgresThreadStore(pool)

        await store.append("t1", [])

        pool.acquire.assert_not_called()

    async def test_append_issues_executemany_with_bound_params(self) -> None:
        pool, conn = _make_pool()
        store = PostgresThreadStore(pool)
        messages = [
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi back"),
        ]

        await store.append("t1", messages)

        conn.executemany.assert_called_once()
        sql, rows = conn.executemany.call_args[0]
        assert "INSERT INTO thread_messages (thread_key, data)" in sql
        assert rows == [
            ("t1", messages[0].model_dump_json()),
            ("t1", messages[1].model_dump_json()),
        ]

    async def test_clear_issues_delete_with_bound_thread_key(self) -> None:
        pool, conn = _make_pool()
        store = PostgresThreadStore(pool)

        await store.clear("t1")

        conn.execute.assert_called_once()
        args = conn.execute.call_args[0]
        assert args[0] == "DELETE FROM thread_messages WHERE thread_key = $1"
        assert args[1] == "t1"

    async def test_clear_is_silent_when_no_rows_match(self) -> None:
        pool, conn = _make_pool()
        conn.execute = AsyncMock(return_value="DELETE 0")
        store = PostgresThreadStore(pool)

        # Must not raise.
        await store.clear("missing")
