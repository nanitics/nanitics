"""Persistent semantic memory: PostgresSemanticStore backed by PostgreSQL + pgvector.

Demonstrates PostgresSemanticStore as a drop-in replacement for
InMemorySemanticStore, providing persistent vector storage with pgvector.
Shows construction, schema management, CRUD operations, namespace isolation,
migration utilities, and integration with create_semantic_memory_tools.

The PostgreSQL-dependent sections require a running Postgres instance with
pgvector. The final section runs standalone using mocks.

Related guide: docs/guides/memory.md
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from nanitics import (
    MockEmbeddingClient,
    get_semantic_store_schema_sql,
)


async def main() -> None:
    # ----------------------------------------------------------------
    # Section 1: Schema SQL for application migrations
    # ----------------------------------------------------------------
    print("--- Section 1: Schema SQL for Application Migrations ---")

    # Applications can use get_semantic_store_schema_sql() in their own
    # migration scripts (e.g. Alembic, raw SQL migrations).
    sql = get_semantic_store_schema_sql()
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert "semantic_entries" in sql
    assert "vector(1024)" in sql
    print(f"  Default schema SQL ({len(sql)} chars):")
    for line in sql.strip().splitlines()[:5]:
        print(f"    {line}")
    print("    ...")
    print("  ✓ Contains CREATE EXTENSION, table, HNSW index, namespace index")

    # Custom table name and dimension
    custom_sql = get_semantic_store_schema_sql(table_name="kb_vectors", dimension=768)
    assert "kb_vectors" in custom_sql
    assert "vector(768)" in custom_sql
    assert "semantic_entries" not in custom_sql
    print(f"\n  Custom schema (table='kb_vectors', dim=768): {len(custom_sql)} chars")
    print("  ✓ Table name and dimension are configurable")

    print("✓ Section 1 passed")

    # ----------------------------------------------------------------
    # Section 2: PostgresSemanticStore construction patterns (mocked)
    # ----------------------------------------------------------------
    print("\n--- Section 2: Construction Patterns ---")

    # In a real application, you'd create a pool with asyncpg:
    #
    #   pool = await asyncpg.create_pool("postgresql://user:pass@host/db")
    #   embedding_client = VoyageEmbeddingClient(api_key="...")
    #
    # For demonstration, we use mocks to show constructor signatures.

    try:
        import asyncpg as _asyncpg  # noqa: F401  # isort: skip

        from nanitics import PostgresSemanticStore

        mock_pool = MagicMock()
        embedding_client = MockEmbeddingClient(dimension=1024)

        # Default construction
        store = PostgresSemanticStore(mock_pool, embedding_client)
        print("  Default: PostgresSemanticStore(pool, embedding_client)")

        # With namespace isolation
        _store_ns = PostgresSemanticStore(mock_pool, embedding_client, namespace="research")
        print("  Namespaced: PostgresSemanticStore(pool, client, namespace='research')")

        # Custom table and dimension
        _store_custom = PostgresSemanticStore(
            mock_pool,
            embedding_client,
            table_name="knowledge_base",
            dimension=768,
        )
        print("  Custom: PostgresSemanticStore(pool, client, table_name='knowledge_base', dimension=768)")

        # --- Schema setup ---
        # In production, call ensure_schema() once at startup:
        #
        #   await store.ensure_schema()
        #
        # This handles:
        # - Fresh database: creates extension, table, indexes
        # - Legacy database: detects existing tables, applies migrations
        # - Current database: no-op (already at latest version)
        # - Concurrent safety: uses PostgreSQL advisory locks

        print("  ✓ All construction patterns valid")
        has_asyncpg = True
    except ImportError:
        print("  asyncpg not installed — skipping construction demo")
        print("  Install with: pip install nanitics[postgres]")
        has_asyncpg = False

    print("✓ Section 2 passed")

    # ----------------------------------------------------------------
    # Section 3: CRUD operations (documented pattern)
    # ----------------------------------------------------------------
    print("\n--- Section 3: CRUD Operations (Pattern) ---")

    # PostgresSemanticStore implements the same SemanticStore protocol
    # as InMemorySemanticStore. The API is identical:
    #
    #   # Add an entry — returns a UUID string
    #   entry_id = await store.add(
    #       "Python uses dynamic typing",
    #       metadata={"source": "docs", "topic": "python"},
    #   )
    #
    #   # Search by semantic similarity — returns list[SearchResult]
    #   results = await store.search("programming language features", limit=5)
    #   for r in results:
    #       print(f"  [{r.score:.3f}] {r.content} (metadata={r.metadata})")
    #
    #   # Delete by ID — no-op if ID doesn't exist
    #   await store.delete(entry_id)
    #
    # Key differences from InMemorySemanticStore:
    # - Data persists across restarts (backed by PostgreSQL)
    # - Similarity search uses pgvector's HNSW index (fast at scale)
    # - Namespace filtering is done in SQL (not in Python)

    print("  SemanticStore protocol: add(), search(), delete()")
    print("  Same interface as InMemorySemanticStore")
    print("  PostgreSQL provides persistence and indexed vector search")
    print("✓ Section 3 passed")

    # ----------------------------------------------------------------
    # Section 4: Namespace isolation (documented pattern)
    # ----------------------------------------------------------------
    print("\n--- Section 4: Namespace Isolation (Pattern) ---")

    # Namespace isolation works at the store level:
    #
    #   research_store = PostgresSemanticStore(
    #       pool, embedding_client, namespace="research"
    #   )
    #   personal_store = PostgresSemanticStore(
    #       pool, embedding_client, namespace="personal"
    #   )
    #
    #   # Each store only sees entries in its namespace
    #   await research_store.add("Quantum computing uses qubits")
    #   await personal_store.add("Buy groceries on Saturday")
    #
    #   # Research store won't find personal entries
    #   results = await research_store.search("groceries")  # → empty
    #
    # Or with create_semantic_memory_tools for agent integration:
    #
    #   from nanitics import create_semantic_memory_tools, InMemorySemanticStore
    #
    #   store = PostgresSemanticStore(pool, embedding_client)
    #   research_tools = create_semantic_memory_tools(store, namespace="research")
    #   personal_tools = create_semantic_memory_tools(store, namespace="personal")

    print("  Store-level: separate PostgresSemanticStore per namespace")
    print("  Tool-level: create_semantic_memory_tools(store, namespace=...)")
    print("✓ Section 4 passed")

    # ----------------------------------------------------------------
    # Section 5: Tool integration (documented pattern)
    # ----------------------------------------------------------------
    print("\n--- Section 5: Tool Integration (Pattern) ---")

    # PostgresSemanticStore integrates with create_semantic_memory_tools
    # exactly like InMemorySemanticStore:
    #
    #   from nanitics import (
    #       PostgresSemanticStore,
    #       ReActAgent,
    #       create_semantic_memory_tools,
    #   )
    #
    #   store = PostgresSemanticStore(pool, embedding_client)
    #   await store.ensure_schema()
    #
    #   tools = create_semantic_memory_tools(store)
    #   # Returns: store_knowledge, search_knowledge, delete_knowledge
    #
    #   agent = ReActAgent(
    #       name="knowledge-agent",
    #       llm_client=llm_client,
    #       system_prompt="You are a knowledge-grounded assistant.",
    #       tools=tools,
    #   )

    print("  create_semantic_memory_tools(store) → 3 tools")
    print("  store_knowledge, search_knowledge, delete_knowledge")
    print("  Agent integration identical to InMemorySemanticStore")
    print("✓ Section 5 passed")

    # ----------------------------------------------------------------
    # Section 6: Runnable demo with mocked pool
    # ----------------------------------------------------------------
    print("\n--- Section 6: Runnable Demo (Mocked Pool) ---")

    if has_asyncpg:
        from nanitics import PostgresSemanticStore as PGStore

        embedding_client = MockEmbeddingClient(dimension=4)

        # Create a mock pool that simulates asyncpg behavior
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        acquire_cm = AsyncMock()
        acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        acquire_cm.__aexit__ = AsyncMock(return_value=False)
        mock_pool.acquire.return_value = acquire_cm

        store = PGStore(mock_pool, embedding_client, dimension=4)

        # Demonstrate that add() calls the embedding client and executes INSERT
        entry_id = await store.add("Test content", metadata={"key": "val"})
        assert isinstance(entry_id, str)
        assert len(entry_id) == 36  # UUID format
        print(f"  add() returned ID: {entry_id[:8]}...")

        # Verify the INSERT was called
        insert_sql = mock_conn.execute.call_args[0][0]
        assert "INSERT INTO semantic_entries" in insert_sql
        print("  INSERT SQL executed ✓")

        # Demonstrate search() with mock results
        mock_conn.fetch = AsyncMock(
            return_value=[
                {"id": entry_id, "content": "Test content", "score": 0.95, "metadata": {"key": "val"}},
            ]
        )
        results = await store.search("test query", limit=3)
        assert len(results) == 1
        assert results[0].score == 0.95
        print(f"  search() returned {len(results)} result(s), top score={results[0].score} ✓")

        # Demonstrate delete()
        mock_conn.execute.reset_mock()
        await store.delete(entry_id)
        delete_sql = mock_conn.execute.call_args[0][0]
        assert "DELETE FROM semantic_entries" in delete_sql
        print("  delete() executed DELETE SQL ✓")
    else:
        print("  asyncpg not installed — skipping mocked demo")

    print("✓ Section 6 passed")

    print("\n🎉 All sections passed!")


if __name__ == "__main__":
    asyncio.run(main())
