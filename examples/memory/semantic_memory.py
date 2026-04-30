"""Semantic memory: similarity-based knowledge storage and retrieval.

Covers MockEmbeddingClient (deterministic embeddings, call tracking),
InMemorySemanticStore (add, search, delete, precomputed vectors),
similarity ranking with hand-crafted vectors, create_semantic_memory_tools
factory, agent integration with event verification, namespace isolation,
and SemanticMemoryProvider / SemanticMemoryContributor for automatic
context injection.

Related guide: docs/guides/memory.md
"""

import asyncio
import math

from examples.helpers import make_emitter, make_response
from nanitics import (
    ContextContent,
    InMemoryEmitter,
    InMemorySemanticStore,
    Message,
    MockEmbeddingClient,
    MockLLMClient,
    ReActAgent,
    SearchResult,
    SemanticMemoryContributor,
    SemanticMemoryProvider,
    SystemPromptContributor,
    ToolCall,
    create_semantic_memory_tools,
)
from nanitics.infrastructure import (
    SemanticSearchEvent,
    SemanticStoreEvent,
)


def _normalize(v: list[float]) -> list[float]:
    """Normalize a vector to unit length."""
    mag = math.sqrt(sum(x * x for x in v))
    return [x / mag for x in v]


async def main() -> None:
    # --- Section 1: EmbeddingClient and MockEmbeddingClient ---
    print("--- Section 1: EmbeddingClient and MockEmbeddingClient ---")

    client = MockEmbeddingClient(dimension=8)

    # Embed a single text — returns a list of vectors (one per input text)
    vectors = await client.embed(["Python is a programming language"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 8
    print(f"  Embedded 1 text → vector of dimension {len(vectors[0])} ✓")

    # Vectors are normalized to unit length
    magnitude = math.sqrt(sum(x * x for x in vectors[0]))
    assert abs(magnitude - 1.0) < 1e-6, "Vector should be unit length"
    print(f"  Vector magnitude: {magnitude:.6f} (unit length) ✓")

    # Deterministic: same input always produces the same vector
    vectors_again = await client.embed(["Python is a programming language"])
    assert vectors[0] == vectors_again[0], "Same text should produce identical vector"
    print("  Deterministic: same input → same vector ✓")

    # Different input produces a different vector
    vectors_other = await client.embed(["JavaScript is used for web development"])
    assert vectors[0] != vectors_other[0], "Different text should produce different vector"
    print("  Different input → different vector ✓")

    # Call tracking — every embed() call is recorded
    assert len(client.calls) == 3
    assert client.calls[0] == ["Python is a programming language"]
    print(f"  Call tracking: {len(client.calls)} calls recorded ✓")

    print("✓ Section 1 passed")

    # --- Section 2: InMemorySemanticStore — Add, Search, Delete ---
    print("\n--- Section 2: InMemorySemanticStore — Add, Search, Delete ---")

    embedding_client = MockEmbeddingClient(dimension=8)
    store = InMemorySemanticStore(embedding_client)

    # Add entries — returns a unique ID for each
    id1 = await store.add("Python is a versatile programming language", metadata={"source": "docs"})
    id2 = await store.add("The weather forecast shows rain tomorrow")
    id3 = await store.add("JavaScript runs in the browser", metadata={"source": "tutorial"})
    assert id1 != id2 != id3, "Each entry gets a unique ID"
    print(f"  Added 3 entries with IDs: {id1[:8]}…, {id2[:8]}…, {id3[:8]}… ✓")

    # Search returns SearchResult objects sorted by score (descending)
    results = await store.search("programming languages", limit=3)
    assert len(results) == 3
    assert all(isinstance(r, SearchResult) for r in results)
    assert results[0].score >= results[1].score >= results[2].score, "Results sorted by score descending"
    print(f"  Search returned {len(results)} results, sorted by score ✓")

    # Inspect SearchResult fields
    top = results[0]
    assert isinstance(top.id, str)
    assert isinstance(top.content, str)
    assert isinstance(top.score, float)
    print(f"  Top result: score={top.score:.3f}, content={top.content!r} ✓")

    # SearchResult is frozen (immutable)
    try:
        top.score = 0.99  # type: ignore[misc]
        raise AssertionError("Should not be able to mutate frozen model")
    except Exception:
        pass
    print("  SearchResult is frozen ✓")

    # Metadata is preserved
    results_with_meta = [r for r in results if r.metadata and "source" in r.metadata]
    assert len(results_with_meta) >= 1, "At least one result should have metadata"
    print(f"  Metadata preserved on {len(results_with_meta)} result(s) ✓")

    # Delete an entry
    await store.delete(id1)
    results_after = await store.search("programming", limit=3)
    result_ids = {r.id for r in results_after}
    assert id1 not in result_ids, "Deleted entry should not appear in search results"
    print(f"  Deleted {id1[:8]}…, now {len(results_after)} results ✓")

    # Note: MockEmbeddingClient scores are NOT semantically meaningful.
    # Similar texts do not produce similar vectors — the mock uses SHA-256 hashing.
    # The next section demonstrates what real similarity ranking looks like.

    print("✓ Section 2 passed")

    # --- Section 3: Similarity Ranking with Precomputed Vectors ---
    print("\n--- Section 3: Similarity Ranking with Precomputed Vectors ---")

    # MockEmbeddingClient doesn't capture semantic meaning. To demonstrate
    # what similarity-based retrieval actually looks like, we use
    # load_precomputed() with hand-crafted vectors in 4D space.

    embedding_client = MockEmbeddingClient(dimension=4)
    store = InMemorySemanticStore(embedding_client)

    # Hand-crafted vectors — programming entries point in a similar direction,
    # weather and cooking point in orthogonal directions.
    programming_1 = _normalize([0.9, 0.1, 0.0, 0.0])  # "Python is a programming language"
    programming_2 = _normalize([0.8, 0.2, 0.1, 0.0])  # "JavaScript is used for web development"
    weather = _normalize([0.0, 0.1, 0.9, 0.0])  # "The weather in Paris is sunny"
    cooking = _normalize([0.0, 0.0, 0.1, 0.9])  # "Cooking pasta requires boiling water"

    store.load_precomputed(
        [
            {"id": "prog-1", "content": "Python is a programming language", "vector": programming_1},
            {"id": "prog-2", "content": "JavaScript is used for web development", "vector": programming_2},
            {"id": "weather", "content": "The weather in Paris is sunny", "vector": weather},
            {"id": "cooking", "content": "Cooking pasta requires boiling water", "vector": cooking},
        ]
    )

    # Query vector points toward the programming cluster
    query_vector = _normalize([0.85, 0.15, 0.05, 0.0])

    # Temporarily override embed to return our known query vector
    original_embed = embedding_client.embed

    async def mock_embed(texts: list[str]) -> list[list[float]]:
        return [query_vector for _ in texts]

    embedding_client.embed = mock_embed  # type: ignore[assignment]

    results = await store.search("programming languages", limit=4)

    # Restore original embed
    embedding_client.embed = original_embed  # type: ignore[assignment]

    # Programming entries should rank above weather and cooking
    assert results[0].id in ("prog-1", "prog-2"), f"Top result should be programming, got {results[0].id}"
    assert results[1].id in ("prog-1", "prog-2"), f"Second result should be programming, got {results[1].id}"
    assert results[0].score > results[2].score, "Programming should score higher than weather/cooking"
    assert results[0].score > results[3].score, "Programming should score higher than weather/cooking"

    print("  Similarity ranking with precomputed vectors:")
    for r in results:
        print(f"    [{r.score:.3f}] {r.content}")
    print("  Programming entries ranked above weather/cooking ✓")

    # This simulates what VoyageEmbeddingClient would produce naturally —
    # semantically related content scores higher than unrelated content.

    print("✓ Section 3 passed")

    # --- Section 4: Tool Factory — create_semantic_memory_tools ---
    print("\n--- Section 4: Tool Factory — create_semantic_memory_tools ---")

    embedding_client = MockEmbeddingClient()
    store = InMemorySemanticStore(embedding_client)
    tools = create_semantic_memory_tools(store)

    assert len(tools) == 3
    tool_names = {t.schema.name for t in tools}
    assert tool_names == {"store_knowledge", "search_knowledge", "delete_knowledge"}
    print(f"  Factory returned {len(tools)} tools: {tool_names} ✓")

    # Each tool has a description
    for t in tools:
        assert t.schema.description, f"Tool {t.schema.name} should have a description"
        print(f"    {t.schema.name}: {t.schema.description[:60]}…")

    print("✓ Section 4 passed")

    # --- Section 5: Agent Integration with Semantic Memory Tools ---
    print("\n--- Section 5: Agent Integration with Semantic Memory Tools ---")

    embedding_client = MockEmbeddingClient()
    store = InMemorySemanticStore(embedding_client)
    tools = create_semantic_memory_tools(store)
    emitter = make_emitter()

    client = MockLLMClient(
        [
            # Step 1: Store a knowledge entry
            make_response(
                content="I'll store this information for later.",
                tool_calls=[
                    ToolCall(
                        id="tc-1",
                        name="store_knowledge",
                        arguments={
                            "content": "The capital of France is Paris",
                            "metadata": "geography fact",
                        },
                    )
                ],
                stop_reason="tool_use",
            ),
            # Step 2: Search for the stored knowledge
            make_response(
                content="Let me search for that information.",
                tool_calls=[
                    ToolCall(
                        id="tc-2",
                        name="search_knowledge",
                        arguments={"query": "capital of France"},
                    )
                ],
                stop_reason="tool_use",
            ),
            # Step 3: Final answer citing search results
            make_response(
                content="The capital of France is Paris.",
            ),
        ]
    )

    agent = ReActAgent(
        name="semantic-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a knowledge-grounded assistant with semantic memory.",
        tools=tools,
    )

    result = await agent.run("What is the capital of France?")
    assert result.output == "The capital of France is Paris."
    assert result.total_steps == 3
    print(f"  Agent completed in {result.total_steps} steps ✓")
    print(f"  Output: {result.output}")

    # Verify the store contains the entry
    stored = await store.search("capital", limit=1)
    assert len(stored) == 1
    assert "Paris" in stored[0].content
    print("  Knowledge persisted in store ✓")

    # Verify events were emitted
    assert isinstance(emitter, InMemoryEmitter)
    store_events = [e for e in emitter.events if isinstance(e, SemanticStoreEvent)]
    search_events = [e for e in emitter.events if isinstance(e, SemanticSearchEvent)]
    assert len(store_events) == 1
    assert store_events[0].content == "The capital of France is Paris"
    assert store_events[0].entry_id == stored[0].id
    assert store_events[0].namespace is None
    print(f"  SemanticStoreEvent: content={store_events[0].content!r} ✓")

    assert len(search_events) == 1
    assert search_events[0].query == "capital of France"
    assert search_events[0].results_count == 1
    assert search_events[0].top_score is not None
    assert search_events[0].namespace is None
    print(f"  SemanticSearchEvent: query={search_events[0].query!r}, results={search_events[0].results_count} ✓")

    print("✓ Section 5 passed")

    # --- Section 6: Namespace Isolation ---
    print("\n--- Section 6: Namespace Isolation ---")

    # A single store can serve multiple agents with isolated views.
    # Namespace filtering happens at the tool layer via metadata.
    embedding_client = MockEmbeddingClient()
    store = InMemorySemanticStore(embedding_client)

    tools_research = create_semantic_memory_tools(store, namespace="research")
    tools_personal = create_semantic_memory_tools(store, namespace="personal")

    # Helper to find a tool by name
    def get_tool(tool_list: list, name: str):
        return next(t for t in tool_list if t.schema.name == name)

    store_research = get_tool(tools_research, "store_knowledge")
    store_personal = get_tool(tools_personal, "store_knowledge")
    search_research = get_tool(tools_research, "search_knowledge")
    search_personal = get_tool(tools_personal, "search_knowledge")

    # Store entries via each namespace's tools
    await store_research.execute(content="Quantum computing uses qubits")
    await store_research.execute(content="Machine learning requires training data")
    await store_personal.execute(content="Buy groceries on Saturday")
    await store_personal.execute(content="Call dentist for appointment")

    # Search through research tools — only research entries visible
    research_results = await search_research.execute(query="computing")
    assert "qubits" in research_results.content or "machine learning" in research_results.content.lower()
    assert "groceries" not in research_results.content
    assert "dentist" not in research_results.content
    print("  Research namespace: only research entries visible ✓")

    # Search through personal tools — only personal entries visible
    personal_results = await search_personal.execute(query="schedule")
    assert "qubits" not in personal_results.content.lower()
    assert "machine learning" not in personal_results.content.lower()
    print("  Personal namespace: only personal entries visible ✓")

    # Direct store.search() bypasses namespace filtering — all entries visible
    all_results = await store.search("anything", limit=10)
    assert len(all_results) == 4, f"Store should contain all 4 entries, got {len(all_results)}"
    print(f"  Direct store.search(): all {len(all_results)} entries visible (no namespace filter) ✓")

    print("✓ Section 6 passed")

    # --- Section 7: SemanticMemoryProvider — Automatic Context Injection ---
    print("\n--- Section 7: SemanticMemoryProvider — Automatic Context Injection ---")

    # Pre-load knowledge entries. Use MockEmbeddingClient: identical text
    # produces similarity 1.0, unrelated text near-zero similarity.
    embedding_client = MockEmbeddingClient()
    store = InMemorySemanticStore(embedding_client)
    await store.add("Python is a high-level programming language", metadata={"source": "docs"})

    # SemanticMemoryContributor teaches the agent about [Semantic Knowledge] blocks.
    contributor = SemanticMemoryContributor()
    assert isinstance(contributor, SystemPromptContributor)
    key, instructions = contributor.system_prompt_section()
    assert key == "semantic_memory"
    assert "[Semantic Knowledge]" in instructions
    print(f"  Contributor section: {key!r} ✓")
    print(f"  Instructions: {instructions[:70]}...")

    # SemanticMemoryProvider retrieves similar entries as context.
    # min_score filters out low-similarity results — essential with
    # MockEmbeddingClient since unrelated texts produce random similarity.
    provider = SemanticMemoryProvider(store=store, min_score=0.5)

    # Provide with a matching user message.
    messages = [Message(role="user", content="Python is a high-level programming language")]
    context = await provider.provide(messages)

    assert context is not None
    assert isinstance(context, ContextContent)
    assert "[Semantic Knowledge]" in context.content
    assert "Python is a high-level programming language" in context.content
    assert "similarity:" in context.content
    assert context.priority == 10
    assert context.protected is False
    assert context.provider_name == "semantic_memory"
    print("  Provider returned ContextContent ✓")
    print(f"  priority={context.priority}, protected={context.protected} ✓")
    print(f"  provider_name={context.provider_name!r} ✓")
    print("  Content preview:")
    for line in context.content.split("\n")[:5]:
        print(f"    {line}")

    # No user messages → returns None.
    empty_result = await provider.provide([])
    assert empty_result is None
    print("  Provider with no messages: None ✓")

    # Non-matching query → returns None (min_score filters out low similarity).
    non_matching = await provider.provide([Message(role="user", content="Completely unrelated topic")])
    assert non_matching is None
    print("  Provider with non-matching query: None ✓")

    print("✓ Section 7 passed")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
