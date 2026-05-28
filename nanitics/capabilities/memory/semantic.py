from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from nanitics.capabilities.memory._similarity import cosine_similarity
from nanitics.capabilities.memory.context_provider import ContextContent
from nanitics.infrastructure.embeddings.protocol import EmbeddingClient
from nanitics.infrastructure.llm.protocol import Message
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import SemanticSearchEvent


class SearchResult(BaseModel):
    """A single result from a semantic similarity search.

    Attributes:
        id: Unique identifier for the stored entry.
        content: The stored text content.
        score: Cosine similarity score (0.0 to 1.0).
        metadata: Optional metadata dictionary associated with the entry.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    content: str
    score: float
    metadata: dict[str, Any] | None = None


@runtime_checkable
class SemanticStore(Protocol):
    """Protocol for a similarity-based knowledge store.

    Content is embedded into vectors and retrieved by semantic similarity
    rather than exact key match. Requires an ``EmbeddingClient`` for
    vector conversion.

    **For:** retrieval-augmented generation, document search, finding
    relevant prior knowledge by meaning rather than identifier.

    **Not for:** exact-key lookup of known facts (use ``LongTermStore``),
    in-run scratchpad (use ``WorkingMemory``), recall of full task
    experiences with outcomes (use ``EpisodeStore``), or multi-agent
    coordination (use ``SharedMemory``).
    """

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """Add content to the store.

        Args:
            content: Text content to store and make searchable.
            metadata: Optional metadata to associate with the entry.

        Returns:
            Unique identifier for the stored entry.
        """
        ...

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Search for content by semantic similarity.

        Args:
            query: Natural language query to search for.
            limit: Maximum number of results to return.

        Returns:
            Results ranked by descending similarity score.
        """
        ...

    async def delete(self, id: str) -> None:
        """Remove an entry by ID.

        Args:
            id: The entry identifier returned by ``add()``.
        """
        ...


class InMemorySemanticStore:
    """In-memory implementation of the ``SemanticStore`` protocol.

    Embeds content on ``add()`` and computes cosine similarity on ``search()``.
    Supports pre-loading entries with precomputed vectors via ``load_precomputed()``.
    Useful for testing and prototyping — for production, implement
    ``SemanticStore`` with a vector database.

    Args:
        embedding_client: Client used to convert text into embedding vectors.
    """

    def __init__(self, embedding_client: EmbeddingClient) -> None:
        self.embedding_client = embedding_client
        self._entries: list[tuple[str, str, dict[str, Any] | None, list[float]]] = []

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        vectors = await self.embedding_client.embed([content])
        entry_id = str(uuid4())
        self._entries.append((entry_id, content, metadata, vectors[0]))
        return entry_id

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        if not self._entries:
            return []
        query_vector = (await self.embedding_client.embed([query]))[0]
        scored = [
            SearchResult(
                id=entry_id,
                content=content,
                score=cosine_similarity(query_vector, vector),
                metadata=metadata,
            )
            for entry_id, content, metadata, vector in self._entries
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:limit]

    async def delete(self, id: str) -> None:
        self._entries = [e for e in self._entries if e[0] != id]

    def load_precomputed(
        self,
        entries: Sequence[dict[str, Any]],
    ) -> None:
        for entry in entries:
            self._entries.append(
                (
                    entry["id"],
                    entry["content"],
                    entry.get("metadata"),
                    entry["vector"],
                )
            )

    def clear(self) -> None:
        self._entries.clear()


class SemanticMemoryProvider:
    """Context provider that automatically injects relevant stored knowledge.

    Before each LLM call, extracts the most recent user message as a query
    and searches the semantic store. Injects top results as a
    ``[Semantic Knowledge]`` context block.

    Args:
        store: The semantic store to search.
        emitter: Optional event emitter for observability.
        limit: Maximum number of entries to inject (default: 3).
        min_score: Minimum similarity score threshold.
        namespace: Optional namespace to scope retrieval. When set against
            an in-memory store, results are filtered client-side using the
            ``_namespace`` metadata key (mirrors ``semantic_tools.py``).
            Against a store with native namespace support (e.g. a
            namespace-scoped ``PostgresSemanticStore``), the filter is
            expected at the store layer and this value is used only to
            label the emitted ``SemanticSearchEvent``.
    """

    def __init__(
        self,
        store: SemanticStore,
        emitter: EventEmitter | None = None,
        limit: int = 3,
        min_score: float | None = None,
        namespace: str | None = None,
        *,
        emitter_provider: Callable[[], EventEmitter | None] | None = None,
    ) -> None:
        self._store = store
        self._static_emitter = emitter
        self._emitter_provider: Callable[[], EventEmitter | None] | None = emitter_provider
        self._limit = limit
        self._min_score = min_score
        self._namespace = namespace

    @property
    def _emitter(self) -> EventEmitter | None:
        """Emitter used for trace events.

        Resolves through ``emitter_provider`` when set (so the provider
        follows its owning agent's per-task bound emitter); otherwise
        the static emitter passed at construction.
        """
        if self._emitter_provider is not None:
            return self._emitter_provider()
        return self._static_emitter

    async def provide(self, messages: list[Message]) -> ContextContent | None:
        query: str | None = None
        for msg in reversed(messages):
            if msg.role == "user" and isinstance(msg.content, str):
                query = msg.content
                break
        if query is None:
            return None
        # Namespace fetch strategy mirrors ``semantic_tools.py``: when
        # namespace is set, fetch a large batch and post-filter client-side
        # so namespace filtering doesn't truncate below limit.
        fetch_limit = self._limit if self._namespace is None else 10000
        results = await self._store.search(query, limit=fetch_limit)
        if self._namespace is not None:
            results = [r for r in results if _keep_namespace(r, self._namespace)]
        if self._min_score is not None:
            results = [r for r in results if r.score >= self._min_score]
        results = results[: self._limit]
        if not results:
            return None
        lines = ["[Semantic Knowledge]", ""]
        for i, r in enumerate(results, 1):
            lines.append(f"## Entry {i} (similarity: {r.score:.2f})")
            lines.append(r.content)
            if r.metadata:
                display_meta = {k: v for k, v in r.metadata.items() if k != "_namespace"}
                if display_meta:
                    lines.append(f"Metadata: {display_meta}")
            lines.append("")
        formatted = "\n".join(lines).rstrip()
        if self._emitter is not None:
            self._emitter.emit(
                SemanticSearchEvent(
                    trace_id=self._emitter.trace_id,
                    span_id=self._emitter.span_id,
                    parent_span_id=self._emitter.parent_span_id,
                    query=query,
                    results_count=len(results),
                    top_score=results[0].score if results else None,
                    namespace=self._namespace,
                )
            )
        return ContextContent(content=formatted, priority=10, protected=False, provider_name="semantic_memory")


def _keep_namespace(result: SearchResult, ns: str) -> bool:
    """Namespace post-filter predicate.

    Mirrors ``semantic_tools.py`` semantics for the in-memory case (where
    every entry has a ``_namespace`` key because the tool layer injected
    it) and avoids a surprising zero-result outcome against
    ``PostgresSemanticStore`` (where namespace is a first-class column,
    not metadata): results without a ``_namespace`` metadata key are
    assumed to have been scoped at the store layer and kept.
    """
    if result.metadata is None:
        return True
    meta_ns = result.metadata.get("_namespace")
    if meta_ns is None:
        return True
    return bool(meta_ns == ns)


_SEMANTIC_MEMORY_INSTRUCTIONS = (
    "Relevant stored knowledge may appear in [Semantic Knowledge] blocks in the "
    "conversation. These are entries retrieved by similarity to the current "
    "user turn, ranked by relevance score.\n\n"
    "Use these entries as grounding when they are applicable. Do not assume "
    "the block is complete — the store may contain other relevant entries "
    "that did not rank high enough for this turn. If an entry contradicts "
    "the user's request or appears irrelevant, ignore it."
)


class SemanticMemoryContributor:
    """System prompt contributor that teaches the agent how to use semantic memory.

    Adds instructions explaining the ``[Semantic Knowledge]`` context blocks —
    how to interpret similarity-ranked entries and how to treat them as
    grounding without assuming the block is complete.
    """

    def system_prompt_section(self) -> tuple[str, str]:
        return ("semantic_memory", _SEMANTIC_MEMORY_INSTRUCTIONS)
