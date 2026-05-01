"""Tests for semantic memory context provider and system prompt contributor."""

from nanitics.capabilities.memory.semantic import (
    InMemorySemanticStore,
    SemanticMemoryContributor,
    SemanticMemoryProvider,
)
from nanitics.infrastructure.embeddings import MockEmbeddingClient
from nanitics.infrastructure.llm.protocol import Message
from nanitics.infrastructure.observability.events import SemanticSearchEvent
from tests.testing_helpers import make_emitter


def make_store() -> InMemorySemanticStore:
    return InMemorySemanticStore(MockEmbeddingClient(dimension=32))


# ──────────────────────────────────────────────────────────
# SemanticMemoryProvider
# ──────────────────────────────────────────────────────────


class TestSemanticMemoryProvider:
    async def test_returns_none_on_empty_store(self) -> None:
        store = make_store()
        provider = SemanticMemoryProvider(store)
        messages = [Message(role="user", content="solve 2+2")]
        result = await provider.provide(messages)
        assert result is None

    async def test_retrieves_and_formats_entries(self) -> None:
        store = make_store()
        await store.add("Python is a dynamically typed language")
        provider = SemanticMemoryProvider(store)
        messages = [Message(role="user", content="Python is a dynamically typed language")]
        result = await provider.provide(messages)
        assert result is not None
        assert "[Semantic Knowledge]" in result.content
        assert "Python is a dynamically typed language" in result.content
        # Similarity score rendered to two decimals on the entry header.
        assert "similarity: 1.00" in result.content or "similarity: " in result.content
        assert result.priority == 10
        assert result.protected is False
        assert result.provider_name == "semantic_memory"

    async def test_uses_latest_user_message(self) -> None:
        store = make_store()
        await store.add("deploy app")
        provider = SemanticMemoryProvider(store)
        messages = [
            Message(role="user", content="old question"),
            Message(role="assistant", content="old answer"),
            Message(role="user", content="deploy app"),
        ]
        result = await provider.provide(messages)
        assert result is not None
        assert "deploy app" in result.content

    async def test_respects_limit(self) -> None:
        store = make_store()
        for i in range(10):
            await store.add(f"task {i}")
        provider = SemanticMemoryProvider(store, limit=2)
        messages = [Message(role="user", content="task")]
        result = await provider.provide(messages)
        assert result is not None
        assert result.content.count("## Entry ") == 2

    async def test_respects_min_score(self) -> None:
        store = make_store()
        # Matching content → similarity 1.0 under MockEmbeddingClient.
        await store.add("deploy app")
        # Non-matching content → near-random low similarity.
        await store.add("cooking pasta technique")
        provider = SemanticMemoryProvider(store, min_score=0.95)
        messages = [Message(role="user", content="deploy app")]
        result = await provider.provide(messages)
        assert result is not None
        # Only the high-scoring match survives min_score=0.95.
        assert "deploy app" in result.content
        assert "cooking pasta" not in result.content

    async def test_returns_none_when_no_user_messages(self) -> None:
        store = make_store()
        await store.add("some knowledge")
        provider = SemanticMemoryProvider(store)
        messages = [Message(role="assistant", content="hello")]
        result = await provider.provide(messages)
        assert result is None

    async def test_namespace_filter_excludes_other_scopes(self) -> None:
        store = make_store()
        await store.add("ns_a entry content", metadata={"_namespace": "ns_a"})
        await store.add("ns_b entry content", metadata={"_namespace": "ns_b"})
        provider = SemanticMemoryProvider(store, namespace="ns_a")
        messages = [Message(role="user", content="entry content")]
        result = await provider.provide(messages)
        assert result is not None
        assert "ns_a entry content" in result.content
        assert "ns_b entry content" not in result.content

    async def test_emits_search_event(self) -> None:
        store = make_store()
        await store.add("knowledge item")
        emitter = make_emitter()
        provider = SemanticMemoryProvider(store, emitter=emitter)
        messages = [Message(role="user", content="knowledge item")]
        await provider.provide(messages)
        events = [e for e in emitter.events if isinstance(e, SemanticSearchEvent)]
        assert len(events) == 1
        assert events[0].query == "knowledge item"
        assert events[0].results_count == 1
        assert events[0].top_score is not None
        assert events[0].namespace is None

    async def test_emits_search_event_with_namespace(self) -> None:
        store = make_store()
        await store.add("scoped knowledge", metadata={"_namespace": "ns_a"})
        emitter = make_emitter()
        provider = SemanticMemoryProvider(store, emitter=emitter, namespace="ns_a")
        messages = [Message(role="user", content="scoped knowledge")]
        await provider.provide(messages)
        events = [e for e in emitter.events if isinstance(e, SemanticSearchEvent)]
        assert len(events) == 1
        assert events[0].namespace == "ns_a"

    async def test_metadata_line_omitted_when_only_namespace_present(self) -> None:
        store = make_store()
        await store.add("only-namespace entry", metadata={"_namespace": "ns_a"})
        provider = SemanticMemoryProvider(store, namespace="ns_a")
        messages = [Message(role="user", content="only-namespace entry")]
        result = await provider.provide(messages)
        assert result is not None
        assert "Metadata:" not in result.content

    async def test_metadata_line_renders_other_keys_without_namespace(self) -> None:
        store = make_store()
        await store.add(
            "entry with extra metadata",
            metadata={"_namespace": "ns_a", "source": "docs"},
        )
        provider = SemanticMemoryProvider(store, namespace="ns_a")
        messages = [Message(role="user", content="entry with extra metadata")]
        result = await provider.provide(messages)
        assert result is not None
        assert "Metadata:" in result.content
        assert "source" in result.content
        assert "docs" in result.content
        assert "_namespace" not in result.content

    async def test_no_results_after_filter_returns_none(self) -> None:
        store = make_store()
        # Content with near-zero similarity to the query under MockEmbeddingClient.
        await store.add("completely unrelated cooking technique")
        provider = SemanticMemoryProvider(store, min_score=0.95)
        messages = [Message(role="user", content="Python programming")]
        result = await provider.provide(messages)
        assert result is None

    async def test_emitter_provider_resolves_emitter_dynamically(self) -> None:
        store = make_store()
        await store.add("resolvable knowledge")
        emitter = make_emitter()
        provider = SemanticMemoryProvider(store, emitter_provider=lambda: emitter)
        messages = [Message(role="user", content="resolvable knowledge")]
        await provider.provide(messages)
        events = [e for e in emitter.events if isinstance(e, SemanticSearchEvent)]
        assert len(events) == 1

    async def test_namespace_filter_keeps_entries_without_metadata(self) -> None:
        """Entries with ``metadata is None`` are assumed to be scoped by the
        underlying store layer (e.g., ``PostgresSemanticStore`` with a
        namespace column) and pass the post-filter unchanged.
        """
        store = make_store()
        await store.add("store-layer scoped entry")  # metadata is None
        provider = SemanticMemoryProvider(store, namespace="ns_a")
        messages = [Message(role="user", content="store-layer scoped entry")]
        result = await provider.provide(messages)
        assert result is not None
        assert "store-layer scoped entry" in result.content

    async def test_namespace_filter_keeps_entries_with_metadata_missing_namespace_key(self) -> None:
        """Entries whose metadata exists but has no ``_namespace`` key are
        assumed to be scoped by the underlying store layer and pass the
        post-filter unchanged.
        """
        store = make_store()
        await store.add(
            "metadata-but-no-namespace entry",
            metadata={"source": "external"},
        )
        provider = SemanticMemoryProvider(store, namespace="ns_a")
        messages = [Message(role="user", content="metadata-but-no-namespace entry")]
        result = await provider.provide(messages)
        assert result is not None
        assert "metadata-but-no-namespace entry" in result.content


# ──────────────────────────────────────────────────────────
# SemanticMemoryContributor
# ──────────────────────────────────────────────────────────


class TestSemanticMemoryContributor:
    def test_returns_section(self) -> None:
        contributor = SemanticMemoryContributor()
        section = contributor.system_prompt_section()
        assert section is not None
        name, content = section
        assert name == "semantic_memory"
        assert "[Semantic Knowledge]" in content
