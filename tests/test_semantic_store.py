import pytest
from pydantic import ValidationError

from nanitics.capabilities.memory.semantic import (
    InMemorySemanticStore,
    SearchResult,
    SemanticStore,
)
from nanitics.infrastructure.embeddings import MockEmbeddingClient


class TestInMemorySemanticStore:
    async def test_add_returns_id(self) -> None:
        store = InMemorySemanticStore(MockEmbeddingClient(dimension=32))
        entry_id = await store.add("test content")
        assert isinstance(entry_id, str)
        assert len(entry_id) > 0

    async def test_search_empty_store(self) -> None:
        store = InMemorySemanticStore(MockEmbeddingClient(dimension=32))
        results = await store.search("anything")
        assert results == []

    async def test_search_returns_added_content(self) -> None:
        store = InMemorySemanticStore(MockEmbeddingClient(dimension=32))
        await store.add("hello world")
        results = await store.search("hello world")
        assert len(results) == 1
        assert results[0].content == "hello world"

    async def test_search_identical_query_has_score_1(self) -> None:
        store = InMemorySemanticStore(MockEmbeddingClient(dimension=32))
        await store.add("hello world")
        results = await store.search("hello world")
        assert abs(results[0].score - 1.0) < 1e-6

    async def test_search_respects_limit(self) -> None:
        store = InMemorySemanticStore(MockEmbeddingClient(dimension=32))
        for i in range(10):
            await store.add(f"entry {i}")
        results = await store.search("entry", limit=3)
        assert len(results) == 3

    async def test_search_results_sorted_by_score(self) -> None:
        store = InMemorySemanticStore(MockEmbeddingClient(dimension=32))
        await store.add("alpha")
        await store.add("beta")
        await store.add("gamma")
        results = await store.search("alpha")
        # The exact match should be first with score 1.0
        assert results[0].content == "alpha"
        assert abs(results[0].score - 1.0) < 1e-6
        # All results should be in descending score order
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    async def test_delete_removes_entry(self) -> None:
        store = InMemorySemanticStore(MockEmbeddingClient(dimension=32))
        entry_id = await store.add("to be deleted")
        await store.delete(entry_id)
        results = await store.search("to be deleted")
        assert results == []

    async def test_delete_nonexistent_id_is_noop(self) -> None:
        store = InMemorySemanticStore(MockEmbeddingClient(dimension=32))
        await store.add("keep me")
        await store.delete("nonexistent-id")
        results = await store.search("keep me")
        assert len(results) == 1

    async def test_metadata_preserved(self) -> None:
        store = InMemorySemanticStore(MockEmbeddingClient(dimension=32))
        await store.add("test", metadata={"source": "unit_test", "priority": 1})
        results = await store.search("test")
        assert results[0].metadata == {"source": "unit_test", "priority": 1}

    async def test_metadata_none_by_default(self) -> None:
        store = InMemorySemanticStore(MockEmbeddingClient(dimension=32))
        await store.add("test")
        results = await store.search("test")
        assert results[0].metadata is None

    async def test_search_result_has_id(self) -> None:
        store = InMemorySemanticStore(MockEmbeddingClient(dimension=32))
        entry_id = await store.add("test")
        results = await store.search("test")
        assert results[0].id == entry_id

    def test_satisfies_protocol(self) -> None:
        store = InMemorySemanticStore(MockEmbeddingClient(dimension=32))
        assert isinstance(store, SemanticStore)

    def test_search_result_is_frozen(self) -> None:
        result = SearchResult(id="x", content="y", score=0.5)
        with pytest.raises(ValidationError):
            result.content = "z"

    def test_embedding_client_is_public(self) -> None:
        client = MockEmbeddingClient(dimension=32)
        store = InMemorySemanticStore(client)
        assert store.embedding_client is client


class TestLoadPrecomputed:
    async def test_loaded_entries_are_searchable(self) -> None:
        client = MockEmbeddingClient(dimension=4)
        store = InMemorySemanticStore(client)
        store.load_precomputed(
            [
                {
                    "id": "pre-1",
                    "content": "alpha content",
                    "vector": [1.0, 0.0, 0.0, 0.0],
                    "metadata": {"category": "test"},
                },
                {
                    "id": "pre-2",
                    "content": "beta content",
                    "vector": [0.0, 1.0, 0.0, 0.0],
                    "metadata": None,
                },
            ]
        )
        results = await store.search("anything")
        assert len(results) == 2
        contents = {r.content for r in results}
        assert contents == {"alpha content", "beta content"}

    async def test_ids_preserved(self) -> None:
        client = MockEmbeddingClient(dimension=4)
        store = InMemorySemanticStore(client)
        store.load_precomputed(
            [
                {"id": "my-id", "content": "test", "vector": [1.0, 0.0, 0.0, 0.0]},
            ]
        )
        results = await store.search("test")
        assert results[0].id == "my-id"

    async def test_metadata_preserved(self) -> None:
        client = MockEmbeddingClient(dimension=4)
        store = InMemorySemanticStore(client)
        store.load_precomputed(
            [
                {
                    "id": "m-1",
                    "content": "test",
                    "vector": [1.0, 0.0, 0.0, 0.0],
                    "metadata": {"key": "value"},
                },
            ]
        )
        results = await store.search("test")
        assert results[0].metadata == {"key": "value"}

    async def test_search_uses_similarity(self) -> None:
        client = MockEmbeddingClient(dimension=4)
        store = InMemorySemanticStore(client)
        store.load_precomputed(
            [
                {"id": "a", "content": "alpha", "vector": [1.0, 0.0, 0.0, 0.0]},
                {"id": "b", "content": "beta", "vector": [0.0, 0.0, 0.0, 1.0]},
            ]
        )
        # Embed query will produce a hash-based vector; both entries should appear
        # with different scores since their vectors differ
        results = await store.search("anything", limit=2)
        assert len(results) == 2
        assert results[0].score >= results[1].score

    def test_does_not_call_embed(self) -> None:
        client = MockEmbeddingClient(dimension=4)
        store = InMemorySemanticStore(client)
        store.load_precomputed(
            [
                {"id": "x", "content": "test", "vector": [1.0, 0.0, 0.0, 0.0]},
            ]
        )
        assert client.calls == []


class TestClear:
    async def test_clear_removes_all_entries(self) -> None:
        store = InMemorySemanticStore(MockEmbeddingClient(dimension=32))
        await store.add("one")
        await store.add("two")
        store.clear()
        results = await store.search("one")
        assert results == []

    async def test_clear_empty_store_is_noop(self) -> None:
        store = InMemorySemanticStore(MockEmbeddingClient(dimension=32))
        store.clear()
        results = await store.search("anything")
        assert results == []

    async def test_add_works_after_clear(self) -> None:
        store = InMemorySemanticStore(MockEmbeddingClient(dimension=32))
        await store.add("before clear")
        store.clear()
        await store.add("after clear")
        results = await store.search("after clear")
        assert len(results) == 1
        assert results[0].content == "after clear"
