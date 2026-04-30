"""Mock-based tests for PostgresSemanticStore — no real Postgres needed."""

from __future__ import annotations

import contextlib
import zlib
from unittest.mock import AsyncMock, MagicMock, call

from nanitics.capabilities.memory.postgres_semantic import (
    PostgresSemanticStore,
    _vector_to_text,
    get_semantic_store_schema_sql,
)
from nanitics.capabilities.memory.semantic import SearchResult, SemanticStore


def _make_pool() -> tuple[MagicMock, AsyncMock]:
    pool = MagicMock()
    conn = AsyncMock()
    acquire_cm = AsyncMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = acquire_cm
    return pool, conn


def _make_embedding_client(vectors: list[list[float]] | None = None) -> AsyncMock:
    client = AsyncMock()
    if vectors is not None:
        client.embed = AsyncMock(return_value=vectors)
    else:
        client.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    return client


class TestVectorToText:
    def test_encodes_floats(self) -> None:
        result = _vector_to_text([0.1, 0.2, 0.3])
        assert result == "[0.1,0.2,0.3]"

    def test_empty_vector(self) -> None:
        result = _vector_to_text([])
        assert result == "[]"

    def test_single_element(self) -> None:
        result = _vector_to_text([1.0])
        assert result == "[1.0]"


class TestGetSemanticStoreSchemaSql:
    def test_returns_string_with_expected_elements(self) -> None:
        sql = get_semantic_store_schema_sql()
        assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
        assert "CREATE TABLE" in sql
        assert "semantic_entries" in sql
        assert "hnsw" in sql.lower()
        assert "vector_cosine_ops" in sql
        assert "id UUID" in sql
        assert "content TEXT" in sql
        assert "metadata JSONB" in sql
        assert "embedding vector(1024)" in sql
        assert "namespace TEXT" in sql
        assert "created_at TIMESTAMPTZ" in sql

    def test_custom_table_name(self) -> None:
        sql = get_semantic_store_schema_sql(table_name="my_entries")
        assert "my_entries" in sql
        assert "semantic_entries" not in sql

    def test_custom_dimension(self) -> None:
        sql = get_semantic_store_schema_sql(dimension=768)
        assert "vector(768)" in sql
        assert "vector(1024)" not in sql

    def test_custom_table_and_dimension(self) -> None:
        sql = get_semantic_store_schema_sql(table_name="kb_entries", dimension=512)
        assert "kb_entries" in sql
        assert "vector(512)" in sql


class TestPostgresSemanticStoreProtocol:
    def test_satisfies_semantic_store_protocol(self) -> None:
        pool = MagicMock()
        client = AsyncMock()
        store = PostgresSemanticStore(pool, client)
        assert isinstance(store, SemanticStore)


class TestEnsureSchemaFreshDatabase:
    """Fresh DB: no version table, no data table."""

    async def test_creates_version_table_and_applies_migrations(self) -> None:
        pool, conn = _make_pool()
        client = _make_embedding_client()
        store = PostgresSemanticStore(pool, client)

        lock_key = zlib.crc32(b"semantic_entries") & 0x7FFFFFFF

        # version table does not exist, data table does not exist
        conn.fetchval = AsyncMock(side_effect=[False, False, 0])
        # transaction context manager
        conn.transaction = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(),
                __aexit__=AsyncMock(return_value=False),
            )
        )

        await store.ensure_schema()

        # Advisory lock acquired and released
        execute_calls = conn.execute.call_args_list
        assert execute_calls[0] == call("SELECT pg_advisory_lock($1)", lock_key)
        assert execute_calls[-1] == call("SELECT pg_advisory_unlock($1)", lock_key)

        # Version table created
        create_version_sql = execute_calls[1][0][0]
        assert "_semantic_entries_schema_version" in create_version_sql
        assert "CREATE TABLE" in create_version_sql

        # Fresh path: version set to 0
        assert execute_calls[2] == call("INSERT INTO _semantic_entries_schema_version (version) VALUES (0)")

        # Migration applied (baseline SQL)
        migration_sql = execute_calls[3][0][0]
        assert "CREATE EXTENSION" in migration_sql
        assert "semantic_entries" in migration_sql

        # Version updated to 1
        assert execute_calls[4] == call("UPDATE _semantic_entries_schema_version SET version = $1", 1)


class TestEnsureSchemaLegacyDatabase:
    """Legacy DB: data table exists, but no version tracking."""

    async def test_marks_as_version_1_and_skips_baseline(self) -> None:
        pool, conn = _make_pool()
        client = _make_embedding_client()
        store = PostgresSemanticStore(pool, client)

        # version table does not exist, data table exists, current version = 1
        conn.fetchval = AsyncMock(side_effect=[False, True, 1])
        conn.transaction = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(),
                __aexit__=AsyncMock(return_value=False),
            )
        )

        await store.ensure_schema()

        execute_calls = conn.execute.call_args_list
        # Version table created
        create_version_sql = execute_calls[1][0][0]
        assert "CREATE TABLE" in create_version_sql

        # Legacy path: version set to 1
        assert execute_calls[2] == call("INSERT INTO _semantic_entries_schema_version (version) VALUES (1)")

        # No migration applied (already at version 1)
        # Only: advisory_lock, create_version_table, insert(1), advisory_unlock
        assert len(execute_calls) == 4


class TestEnsureSchemaCurrentDatabase:
    """Current DB: version table exists, at latest version."""

    async def test_noop_when_at_latest_version(self) -> None:
        pool, conn = _make_pool()
        client = _make_embedding_client()
        store = PostgresSemanticStore(pool, client)

        # version table exists, current version = 1
        conn.fetchval = AsyncMock(side_effect=[True, 1])
        conn.transaction = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(),
                __aexit__=AsyncMock(return_value=False),
            )
        )

        await store.ensure_schema()

        execute_calls = conn.execute.call_args_list
        # Only: advisory_lock, advisory_unlock
        assert len(execute_calls) == 2


class TestEnsureSchemaAdvisoryLockReleasedOnError:
    async def test_unlock_called_even_on_error(self) -> None:
        pool, conn = _make_pool()
        client = _make_embedding_client()
        store = PostgresSemanticStore(pool, client)

        conn.fetchval = AsyncMock(side_effect=RuntimeError("db error"))

        with contextlib.suppress(RuntimeError):
            await store.ensure_schema()

        execute_calls = conn.execute.call_args_list
        lock_key = zlib.crc32(b"semantic_entries") & 0x7FFFFFFF
        assert execute_calls[0] == call("SELECT pg_advisory_lock($1)", lock_key)
        assert execute_calls[-1] == call("SELECT pg_advisory_unlock($1)", lock_key)


class TestAdd:
    async def test_embeds_content_and_inserts(self) -> None:
        pool, conn = _make_pool()
        client = _make_embedding_client([[0.5, 0.6, 0.7]])
        store = PostgresSemanticStore(pool, client, dimension=3)

        entry_id = await store.add("test content", metadata={"key": "val"})

        # Embedding client called with content
        client.embed.assert_called_once_with(["test content"])

        # Returns a UUID string
        assert isinstance(entry_id, str)
        assert len(entry_id) == 36  # UUID format

        # INSERT executed
        conn.execute.assert_called_once()
        args = conn.execute.call_args[0]
        assert "INSERT INTO semantic_entries" in args[0]
        assert args[1] == entry_id
        assert args[2] == "test content"
        assert args[3] == '{"key": "val"}'
        assert args[4] == "[0.5,0.6,0.7]"
        assert args[5] is None  # namespace

    async def test_none_metadata_stored_as_null(self) -> None:
        pool, conn = _make_pool()
        client = _make_embedding_client([[0.1, 0.2]])
        store = PostgresSemanticStore(pool, client, dimension=2)

        await store.add("content", metadata=None)

        args = conn.execute.call_args[0]
        assert args[3] is None  # metadata is SQL NULL

    async def test_default_metadata_is_null(self) -> None:
        pool, conn = _make_pool()
        client = _make_embedding_client([[0.1, 0.2]])
        store = PostgresSemanticStore(pool, client, dimension=2)

        await store.add("content")

        args = conn.execute.call_args[0]
        assert args[3] is None

    async def test_namespace_included_in_insert(self) -> None:
        pool, conn = _make_pool()
        client = _make_embedding_client([[0.1, 0.2]])
        store = PostgresSemanticStore(pool, client, namespace="research", dimension=2)

        await store.add("content")

        args = conn.execute.call_args[0]
        assert args[5] == "research"


class TestSearch:
    async def test_searches_with_cosine_distance(self) -> None:
        pool, conn = _make_pool()
        client = _make_embedding_client([[0.1, 0.2, 0.3]])
        store = PostgresSemanticStore(pool, client, dimension=3)

        conn.fetch = AsyncMock(
            return_value=[
                {"id": "abc-123", "content": "result text", "score": 0.95, "metadata": {"k": "v"}},
            ]
        )

        results = await store.search("test query", limit=3)

        # Embedding client called with query
        client.embed.assert_called_once_with(["test query"])

        # Fetch called with SELECT SQL
        fetch_args = conn.fetch.call_args[0]
        assert "<=>" in fetch_args[0]
        assert "ORDER BY" in fetch_args[0]
        assert "LIMIT" in fetch_args[0]
        assert fetch_args[1] == "[0.1,0.2,0.3]"
        assert fetch_args[2] == 3  # limit

        # Results converted to SearchResult
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].id == "abc-123"
        assert results[0].content == "result text"
        assert results[0].score == 0.95
        assert results[0].metadata == {"k": "v"}

    async def test_search_with_namespace_filter(self) -> None:
        pool, conn = _make_pool()
        client = _make_embedding_client([[0.1, 0.2]])
        store = PostgresSemanticStore(pool, client, namespace="research", dimension=2)

        conn.fetch = AsyncMock(return_value=[])

        await store.search("query")

        fetch_args = conn.fetch.call_args[0]
        assert "namespace = $2" in fetch_args[0]
        assert fetch_args[1] == "[0.1,0.2]"
        assert fetch_args[2] == "research"
        assert fetch_args[3] == 5  # default limit

    async def test_search_without_namespace_no_filter(self) -> None:
        pool, conn = _make_pool()
        client = _make_embedding_client([[0.1, 0.2]])
        store = PostgresSemanticStore(pool, client, namespace=None, dimension=2)

        conn.fetch = AsyncMock(return_value=[])

        await store.search("query")

        fetch_args = conn.fetch.call_args[0]
        assert "namespace" not in fetch_args[0]

    async def test_search_empty_results(self) -> None:
        pool, conn = _make_pool()
        client = _make_embedding_client([[0.1, 0.2]])
        store = PostgresSemanticStore(pool, client, dimension=2)

        conn.fetch = AsyncMock(return_value=[])

        results = await store.search("no matches")
        assert results == []

    async def test_search_metadata_string_deserialized(self) -> None:
        pool, conn = _make_pool()
        client = _make_embedding_client([[0.1, 0.2]])
        store = PostgresSemanticStore(pool, client, dimension=2)

        conn.fetch = AsyncMock(
            return_value=[
                {"id": "x", "content": "text", "score": 0.8, "metadata": '{"k": "v"}'},
            ]
        )

        results = await store.search("query")
        assert results[0].metadata == {"k": "v"}

    async def test_search_metadata_dict_passthrough(self) -> None:
        pool, conn = _make_pool()
        client = _make_embedding_client([[0.1, 0.2]])
        store = PostgresSemanticStore(pool, client, dimension=2)

        conn.fetch = AsyncMock(
            return_value=[
                {"id": "x", "content": "text", "score": 0.8, "metadata": {"k": "v"}},
            ]
        )

        results = await store.search("query")
        assert results[0].metadata == {"k": "v"}

    async def test_search_metadata_none(self) -> None:
        pool, conn = _make_pool()
        client = _make_embedding_client([[0.1, 0.2]])
        store = PostgresSemanticStore(pool, client, dimension=2)

        conn.fetch = AsyncMock(
            return_value=[
                {"id": "x", "content": "text", "score": 0.8, "metadata": None},
            ]
        )

        results = await store.search("query")
        assert results[0].metadata is None

    async def test_search_multiple_results(self) -> None:
        pool, conn = _make_pool()
        client = _make_embedding_client([[0.1, 0.2]])
        store = PostgresSemanticStore(pool, client, dimension=2)

        conn.fetch = AsyncMock(
            return_value=[
                {"id": "a", "content": "first", "score": 0.95, "metadata": None},
                {"id": "b", "content": "second", "score": 0.80, "metadata": {"src": "test"}},
            ]
        )

        results = await store.search("query", limit=2)
        assert len(results) == 2
        assert results[0].id == "a"
        assert results[1].id == "b"


class TestDelete:
    async def test_deletes_by_id(self) -> None:
        pool, conn = _make_pool()
        client = _make_embedding_client()
        store = PostgresSemanticStore(pool, client)

        await store.delete("abc-123")

        conn.execute.assert_called_once()
        args = conn.execute.call_args[0]
        assert "DELETE FROM semantic_entries" in args[0]
        assert "$1::uuid" in args[0]
        assert args[1] == "abc-123"


class TestCustomTableName:
    async def test_add_uses_custom_table(self) -> None:
        pool, conn = _make_pool()
        client = _make_embedding_client([[0.1]])
        store = PostgresSemanticStore(pool, client, table_name="kb_entries", dimension=1)

        await store.add("content")

        args = conn.execute.call_args[0]
        assert "INSERT INTO kb_entries" in args[0]

    async def test_search_uses_custom_table(self) -> None:
        pool, conn = _make_pool()
        client = _make_embedding_client([[0.1]])
        store = PostgresSemanticStore(pool, client, table_name="kb_entries", dimension=1)
        conn.fetch = AsyncMock(return_value=[])

        await store.search("query")

        fetch_args = conn.fetch.call_args[0]
        assert "kb_entries" in fetch_args[0]

    async def test_delete_uses_custom_table(self) -> None:
        pool, conn = _make_pool()
        client = _make_embedding_client()
        store = PostgresSemanticStore(pool, client, table_name="kb_entries")

        await store.delete("x")

        args = conn.execute.call_args[0]
        assert "DELETE FROM kb_entries" in args[0]

    async def test_ensure_schema_uses_custom_table(self) -> None:
        pool, conn = _make_pool()
        client = _make_embedding_client()
        store = PostgresSemanticStore(pool, client, table_name="kb_entries")

        lock_key = zlib.crc32(b"kb_entries") & 0x7FFFFFFF

        conn.fetchval = AsyncMock(side_effect=[True, 1])
        conn.transaction = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(),
                __aexit__=AsyncMock(return_value=False),
            )
        )

        await store.ensure_schema()

        execute_calls = conn.execute.call_args_list
        assert execute_calls[0] == call("SELECT pg_advisory_lock($1)", lock_key)
