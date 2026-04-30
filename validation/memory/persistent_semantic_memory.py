"""Persistent semantic memory backed by PostgreSQL + pgvector.

Validates that :class:`PostgresSemanticStore` with a real Voyage
embedding client round-trips content + metadata, ranks semantically
closest entries first under pgvector's cosine-distance operator, isolates
namespaces at the SQL layer, and emits the expected observability events
when driven through :func:`create_semantic_memory_tools` on a
:class:`ReActAgent`.

This script requires a reachable PostgreSQL with ``pgvector`` installed
(``POSTGRES_URL`` env var, ``asyncpg`` extra) and a live Voyage API key
(``VOYAGE_API_KEY``). Each test uses a unique table name so parallel
runs do not collide, and drops its table on teardown so reruns are
deterministic.

Acceptance criteria — persistence and ranking:
  - ``ensure_schema()`` succeeds on a fresh, uniquely-named table.
  - Four entries with diverse metadata are inserted; a ``top_k=3``
    search for a Paris-related query returns the Paris entry first.
  - Top similarity score clears a degeneracy floor (> 0.2) and the
    relevant entry scores measurably above the irrelevant ones
    (separation > 0.05) — catches mis-wired embeddings.
  - ``limit=3`` returns exactly three results when the store holds four.
  - Metadata round-trips intact on the top hit (pins JSONB decode).

Acceptance criteria — namespace filtering:
  - A namespaced store (``namespace="ns_a"``) seeing a mixed population
    (``ns_a`` + ``ns_b`` under the same table) returns only ``ns_a``
    rows — pins the SQL-side ``WHERE namespace = $2`` path.

Acceptance criteria — agent trace:
  - A :class:`ReActAgent` driving :func:`create_semantic_memory_tools`
    emits a ``SemanticSearchEvent`` with ``results_count >= 1`` and
    ``top_score > 0.2`` (non-degenerate retrieval through the tool).

Teardown:
  - The unique test table and its ``_<table>_schema_version`` companion
    are dropped whether the test passes or fails.
"""

from __future__ import annotations

import uuid

from nanitics import (
    InMemoryEmitter,
    PostgresSemanticStore,
    ReActAgent,
    create_semantic_memory_tools,
)
from nanitics.infrastructure import SemanticSearchEvent
from validation.helpers import (
    assert_trace_contains,
    make_embedding_client,
    make_llm_client,
    make_postgres_pool,
    requires_postgres,
    requires_voyage,
    run_with_retry,
)

# voyage-3-lite produces 512-dimensional embeddings; pin the store
# dimension to match.
_VOYAGE_DIM = 512


def _unique_table_name(prefix: str) -> str:
    """Return a table name with a UUID suffix for parallel-run isolation."""
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


async def _drop_test_tables(pool: object, table_name: str) -> None:
    """Drop the entry table and version-tracking companion table.

    Runs as teardown. Uses ``DROP ... IF EXISTS`` so it is idempotent
    even when the body errored before ``ensure_schema()`` completed.
    """
    async with pool.acquire() as conn:  # type: ignore[attr-defined]
        await conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        await conn.execute(f"DROP TABLE IF EXISTS _{table_name}_schema_version")


@requires_postgres
@requires_voyage
async def test_postgres_semantic_store_search_and_metadata(traced_emitter: InMemoryEmitter) -> None:
    table_name = _unique_table_name("val_semantic")
    embedding_client = make_embedding_client("voyage")

    async with make_postgres_pool() as pool:
        store = PostgresSemanticStore(
            pool,
            embedding_client,
            table_name=table_name,
            dimension=_VOYAGE_DIM,
        )
        try:
            await run_with_retry(lambda: store.ensure_schema(), max_attempts=2)

            # --- Seed four diverse entries ---
            paris_entry = "Paris is the capital of France and sits on the river Seine."
            python_entry = "Python is a high-level programming language with dynamic typing."
            cooking_entry = "Al dente pasta is cooked one minute short of the package time."
            music_entry = "Counterpoint is the relationship between voices in classical music."

            await run_with_retry(
                lambda: store.add(paris_entry, metadata={"domain": "geography", "country": "France"}),
                max_attempts=2,
            )
            await run_with_retry(
                lambda: store.add(python_entry, metadata={"domain": "programming"}),
                max_attempts=2,
            )
            await run_with_retry(
                lambda: store.add(cooking_entry, metadata={"domain": "cooking"}),
                max_attempts=2,
            )
            await run_with_retry(
                lambda: store.add(music_entry, metadata={"domain": "music"}),
                max_attempts=2,
            )

            # --- Direct search: Paris-related query must surface Paris first ---
            results = await run_with_retry(
                lambda: store.search("What is the capital of France?", limit=3),
                max_attempts=2,
            )
            assert len(results) == 3, (
                f"Expected limit=3 against four entries to return exactly 3 results; got: {len(results)}"
            )
            assert results[0].content == paris_entry, (
                f"Expected Paris entry on top for a capital-of-France query; got: {results[0].content!r}"
            )
            assert results[0].score > 0.2, f"Expected top score above degeneracy floor; got: {results[0].score}"
            separation = results[0].score - results[-1].score
            assert separation > 0.05, (
                f"Expected relevant entry to score measurably above irrelevant ones; "
                f"got scores: {[r.score for r in results]}"
            )
            # Metadata round-trip on the top hit (pins JSONB decode).
            assert results[0].metadata == {"domain": "geography", "country": "France"}, (
                f"Expected metadata round-trip for Paris entry; got: {results[0].metadata!r}"
            )

            # --- Agent path: emits SemanticSearchEvent through the tool ---
            agent = ReActAgent(
                name="pg-semantic-agent",
                llm_client=make_llm_client("anthropic"),
                emitter=traced_emitter,
                system_prompt=(
                    "You maintain a persistent knowledge base. Use search_knowledge "
                    "to retrieve the most relevant fact and answer the user concisely."
                ),
                tools=create_semantic_memory_tools(store),
                max_iterations=4,
            )

            await run_with_retry(
                lambda: agent.run("Search the knowledge base for the capital of France and report what you find."),
                max_attempts=2,
            )

            assert_trace_contains(
                traced_emitter,
                SemanticSearchEvent,
                predicate=lambda e: e.results_count >= 1 and e.top_score is not None and e.top_score > 0.2,
            )
        finally:
            await _drop_test_tables(pool, table_name)


@requires_postgres
@requires_voyage
async def test_postgres_semantic_store_namespace_filter(traced_emitter: InMemoryEmitter) -> None:
    """Namespaced store must only return rows matching its namespace.

    Uses a single table shared by two :class:`PostgresSemanticStore`
    instances, one per namespace. Searching through the ``ns_a`` store
    must never surface ``ns_b`` rows — this pins the SQL-side
    ``WHERE namespace = $2`` path, independent of any Python-side
    post-filtering.
    """
    table_name = _unique_table_name("val_semantic_ns")
    embedding_client = make_embedding_client("voyage")

    async with make_postgres_pool() as pool:
        store_ns_a = PostgresSemanticStore(
            pool,
            embedding_client,
            namespace="ns_a",
            table_name=table_name,
            dimension=_VOYAGE_DIM,
        )
        store_ns_b = PostgresSemanticStore(
            pool,
            embedding_client,
            namespace="ns_b",
            table_name=table_name,
            dimension=_VOYAGE_DIM,
        )
        try:
            # ensure_schema on either one is enough — they share the table.
            await run_with_retry(lambda: store_ns_a.ensure_schema(), max_attempts=2)

            ns_a_entry = "Paris is the capital of France."
            ns_b_entry = "Berlin is the capital of Germany."
            await run_with_retry(lambda: store_ns_a.add(ns_a_entry, metadata={"ns": "a"}), max_attempts=2)
            await run_with_retry(lambda: store_ns_b.add(ns_b_entry, metadata={"ns": "b"}), max_attempts=2)

            # Query is deliberately broad — both entries are structurally similar
            # (``X is the capital of Y``). Only the namespace filter separates them.
            results_a = await run_with_retry(
                lambda: store_ns_a.search("capital city of a European country", limit=10),
                max_attempts=2,
            )
            assert results_a, "Expected at least one ns_a result."
            assert all(r.content == ns_a_entry for r in results_a), (
                f"Expected ns_a search to return only ns_a content; got: {[r.content for r in results_a]}"
            )
            # The ns_b row must not leak through.
            assert all(ns_b_entry not in r.content for r in results_a), (
                f"Namespace leak: ns_b entry appeared in ns_a search results: {[r.content for r in results_a]}"
            )

            # Symmetric check for ns_b.
            results_b = await run_with_retry(
                lambda: store_ns_b.search("capital city of a European country", limit=10),
                max_attempts=2,
            )
            assert results_b, "Expected at least one ns_b result."
            assert all(r.content == ns_b_entry for r in results_b), (
                f"Expected ns_b search to return only ns_b content; got: {[r.content for r in results_b]}"
            )
        finally:
            await _drop_test_tables(pool, table_name)
