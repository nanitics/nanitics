from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from nanitics.capabilities.memory._similarity import cosine_similarity
from nanitics.capabilities.memory.context_provider import ContextContent
from nanitics.infrastructure.embeddings.protocol import EmbeddingClient
from nanitics.infrastructure.llm.protocol import ContentBlock, Message
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import EpisodeRecallEvent

if TYPE_CHECKING:
    from nanitics.strategies.agents.base import AgentInput, AgentResult


class OutcomeType(StrEnum):
    """The outcome of an agent experience.

    Attributes:
        SUCCESS: The task was completed successfully.
        FAILURE: The task failed (iteration limit, cancellation, etc.).
        PARTIAL: The task was partially completed.
    """

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


class Episode(BaseModel):
    """A recorded agent experience.

    Captures the situation the agent faced, the action it took, and the
    outcome. Optionally includes a reflection on why the approach worked
    or didn't, plus arbitrary metadata.

    Attributes:
        id: Auto-generated UUID.
        situation: Description of what the agent faced.
        action: Summary of what the agent did.
        outcome: Whether the experience was a success, failure, or partial.
        outcome_detail: Detailed description of the outcome.
        reflection: Analysis of why the approach worked or didn't.
        evaluator_feedback: Verbatim evaluator feedback that drove the
            rejection (when applicable). Distinct from ``reflection``,
            which is an LLM analysis. ``None`` when no rejection feedback
            applies (e.g., success episodes, evaluator-error episodes,
            episodes constructed outside any evaluator loop).
        metadata: Arbitrary metadata (e.g., task type, namespace).
        timestamp: When the episode was recorded.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    situation: str | list[ContentBlock]
    action: str
    outcome: OutcomeType
    outcome_detail: str | None = None
    reflection: str | None = None
    evaluator_feedback: str | None = None
    metadata: dict[str, Any] | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RecallFilters(BaseModel):
    """Filters for episode recall queries.

    All filters are optional. When multiple filters are set, they are
    combined with AND semantics.

    Attributes:
        outcome: Filter by outcome type.
        metadata_filters: Require matching metadata key-value pairs.
        min_score: Minimum similarity score threshold.
        after: Only episodes after this timestamp.
        before: Only episodes before this timestamp.
    """

    model_config = ConfigDict(frozen=True)

    outcome: OutcomeType | None = None
    metadata_filters: dict[str, Any] | None = None
    min_score: float | None = None
    after: datetime | None = None
    before: datetime | None = None


class RecallResult(BaseModel):
    """A recalled episode with its similarity score.

    Attributes:
        episode: The recalled episode.
        similarity_score: Cosine similarity between the query and the
            episode's situation (0.0 to 1.0).
    """

    model_config = ConfigDict(frozen=True)

    episode: Episode
    similarity_score: float


@runtime_checkable
class EpisodeStore(Protocol):
    """Protocol for storing and recalling agent experiences.

    Episodes are recorded with situation descriptions (embedded into vectors)
    and recalled by semantic similarity to a query.

    **For:** learning from past task outcomes — what worked, what failed,
    which approach succeeded under which conditions. Episodes carry an
    outcome label and supersede older attempts on the same situation.

    **Not for:** general document retrieval (use ``SemanticStore`` —
    episodes are task-scoped, not corpus-scoped), exact-key fact lookup
    (use ``LongTermStore``), in-run scratchpad (use ``WorkingMemory``),
    or multi-agent coordination (use ``SharedMemory``).
    """

    async def record(self, episode: Episode) -> str:
        """Record an episode.

        Args:
            episode: The episode to store.

        Returns:
            The episode ID.
        """
        ...

    async def recall(self, query: str, filters: RecallFilters | None = None, limit: int = 5) -> list[RecallResult]:
        """Recall episodes similar to the query.

        Args:
            query: Natural language description of the situation to match.
            filters: Optional filters to narrow results.
            limit: Maximum number of episodes to return.

        Returns:
            Episodes ranked by descending similarity score.
        """
        ...

    async def forget(self, episode_id: str) -> None:
        """Remove an episode by ID.

        Args:
            episode_id: The episode to remove.
        """
        ...

    async def count(self) -> int:
        """Return the total number of stored episodes."""
        ...


class InMemoryEpisodeStore:
    """In-memory implementation of the ``EpisodeStore`` protocol.

    Embeds episode situations on ``record()`` and retrieves by cosine
    similarity on ``recall()``. Supports a maximum episode cap with
    oldest-first eviction, and pruning of failure episodes superseded
    by a success for the same situation.

    Args:
        embedding_client: Client for embedding episode situations.
        max_episodes: Optional cap on stored episodes. When exceeded,
            the oldest episode is evicted.
    """

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        max_episodes: int | None = None,
    ) -> None:
        self.embedding_client = embedding_client
        self._max_episodes = max_episodes
        self._entries: list[tuple[str, Episode, list[float]]] = []

    async def record(self, episode: Episode) -> str:
        if self._max_episodes is not None and len(self._entries) >= self._max_episodes:
            oldest_idx = min(range(len(self._entries)), key=lambda i: self._entries[i][1].timestamp)
            self._entries.pop(oldest_idx)
        if isinstance(episode.situation, str):
            situation_text = episode.situation
        else:
            situation_text = " ".join(b.text for b in episode.situation if hasattr(b, "text"))
        vectors = await self.embedding_client.embed([situation_text])
        self._entries.append((episode.id, episode, vectors[0]))
        return episode.id

    async def recall(self, query: str, filters: RecallFilters | None = None, limit: int = 5) -> list[RecallResult]:
        if not self._entries:
            return []
        query_vector = (await self.embedding_client.embed([query]))[0]
        results: list[RecallResult] = []
        for _entry_id, episode, vector in self._entries:
            score = cosine_similarity(query_vector, vector)
            if filters:
                if filters.outcome is not None and episode.outcome != filters.outcome:
                    continue
                if filters.metadata_filters is not None and episode.metadata is not None:
                    if not all(episode.metadata.get(k) == v for k, v in filters.metadata_filters.items()):
                        continue
                if filters.metadata_filters is not None and episode.metadata is None:
                    continue
                if filters.min_score is not None and score < filters.min_score:
                    continue
                if filters.after is not None and episode.timestamp <= filters.after:
                    continue
                if filters.before is not None and episode.timestamp >= filters.before:
                    continue
            results.append(RecallResult(episode=episode, similarity_score=score))
        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results[:limit]

    async def forget(self, episode_id: str) -> None:
        self._entries = [e for e in self._entries if e[0] != episode_id]

    async def count(self) -> int:
        return len(self._entries)

    async def prune_superseded(
        self,
        situation: str,
        similarity_threshold: float = 0.9,
    ) -> list[str]:
        """Remove failure episodes superseded by a success for a similar situation.

        Finds episodes similar to the given situation above the threshold.
        If any of those episodes are successes, removes all failure episodes
        in the same group. This prevents the agent from seeing obsolete
        failure patterns.

        Args:
            situation: The situation to match against.
            similarity_threshold: Minimum similarity to consider episodes
                as related (default: 0.9).

        Returns:
            List of removed episode IDs.
        """
        if not self._entries:
            return []
        query_vector = (await self.embedding_client.embed([situation]))[0]
        similar: list[tuple[str, Episode, float]] = []
        for entry_id, episode, vector in self._entries:
            score = cosine_similarity(query_vector, vector)
            if score >= similarity_threshold:
                similar.append((entry_id, episode, score))
        has_success = any(ep.outcome == OutcomeType.SUCCESS for _, ep, _ in similar)
        if not has_success:
            return []
        to_remove = [eid for eid, ep, _ in similar if ep.outcome == OutcomeType.FAILURE]
        self._entries = [e for e in self._entries if e[0] not in to_remove]
        return to_remove


_TERMINATION_TO_OUTCOME: dict[str, OutcomeType] = {
    "complete": OutcomeType.SUCCESS,
    "iteration_limit": OutcomeType.FAILURE,
    "cancelled": OutcomeType.FAILURE,
    "evaluation_failed": OutcomeType.FAILURE,
}


def _summarize_trajectory(messages: list[Any]) -> str:
    tool_names: list[str] = []
    first_assistant: str | None = None
    for msg in messages:
        if msg.role == "assistant":
            if first_assistant is None and msg.content:
                first_assistant = msg.content
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.name not in tool_names:
                        tool_names.append(tc.name)
    parts: list[str] = []
    if first_assistant:
        truncated = first_assistant[:200]
        if len(first_assistant) > 200:
            truncated += "..."
        parts.append(truncated)
    if tool_names:
        parts.append(f"Tools used: {', '.join(tool_names)}")
    return "; ".join(parts) if parts else "No actions recorded"


def extract_episode(
    task_input: AgentInput,
    result: AgentResult,
    *,
    outcome: OutcomeType | None = None,
    outcome_detail: str | None = None,
    reflection: str | None = None,
    evaluator_feedback: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Episode:
    """Create an episode from an agent run result.

    Builds an ``Episode`` from the task input and agent result. The action
    is automatically summarized from the agent's message trajectory
    (first assistant message + tools used). The outcome is inferred from
    the termination reason unless explicitly provided.

    Outcome inference from termination reason:
        - ``"complete"`` → ``SUCCESS``
        - ``"iteration_limit"``, ``"cancelled"``, ``"evaluation_failed"`` → ``FAILURE``
        - Anything else → ``PARTIAL``

    Args:
        task_input: The original task given to the agent.
        result: The ``AgentResult`` from the completed run.
        outcome: Explicit outcome override.
        outcome_detail: Detailed description of the outcome.
        reflection: Analysis of why the approach worked or didn't.
        evaluator_feedback: Verbatim evaluator feedback that drove the
            rejection (when applicable). Forwarded directly to the
            constructed ``Episode``.
        metadata: Arbitrary metadata to attach to the episode.

    Returns:
        A new ``Episode`` ready to be recorded in an ``EpisodeStore``.
    """
    if outcome is not None:
        resolved_outcome = outcome
    else:
        resolved_outcome = _TERMINATION_TO_OUTCOME.get(result.termination_reason, OutcomeType.PARTIAL)
    action = _summarize_trajectory(result.messages)
    return Episode(
        situation=task_input,
        action=action,
        outcome=resolved_outcome,
        outcome_detail=outcome_detail,
        reflection=reflection,
        evaluator_feedback=evaluator_feedback,
        metadata=metadata,
    )


class EpisodicMemoryProvider:
    """Context provider that automatically injects relevant past experiences.

    Before each LLM call, extracts the most recent user message as a query
    and recalls similar episodes from the store. Injects them as a
    ``[Past Experiences]`` context block.

    Args:
        store: The episode store to recall from.
        emitter: Optional event emitter for observability.
        limit: Maximum number of episodes to inject (default: 3).
        outcome_filter: Only recall episodes with this outcome type.
        min_score: Minimum similarity score threshold.
    """

    def __init__(
        self,
        store: EpisodeStore,
        emitter: EventEmitter | None = None,
        limit: int = 3,
        outcome_filter: OutcomeType | None = None,
        min_score: float | None = None,
        *,
        emitter_provider: Callable[[], EventEmitter | None] | None = None,
    ) -> None:
        self._store = store
        self._static_emitter = emitter
        self._emitter_provider: Callable[[], EventEmitter | None] | None = emitter_provider
        self._limit = limit
        self._outcome_filter = outcome_filter
        self._min_score = min_score

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
        filters = RecallFilters(
            outcome=self._outcome_filter,
            min_score=self._min_score,
        )
        results = await self._store.recall(query, filters=filters, limit=self._limit)
        if not results:
            return None
        lines = ["[Past Experiences]", ""]
        for i, r in enumerate(results, 1):
            ep = r.episode
            lines.append(f"## Experience {i} ({ep.outcome.value}, similarity: {r.similarity_score:.2f})")
            lines.append(f"Situation: {ep.situation}")
            lines.append(f"Action: {ep.action}")
            if ep.outcome_detail:
                lines.append(f"Outcome: {ep.outcome_detail}")
            else:
                lines.append(f"Outcome: {ep.outcome.value}")
            if ep.evaluator_feedback:
                lines.append(f"Evaluator feedback: {ep.evaluator_feedback}")
            if ep.reflection:
                lines.append(f"Reflection: {ep.reflection}")
            lines.append("")
        formatted = "\n".join(lines).rstrip()
        if self._emitter is not None:
            self._emitter.emit(
                EpisodeRecallEvent(
                    trace_id=self._emitter.trace_id,
                    span_id=self._emitter.span_id,
                    parent_span_id=self._emitter.parent_span_id,
                    query=query,
                    results_count=len(results),
                    top_score=results[0].similarity_score if results else None,
                    namespace=None,
                )
            )
        return ContextContent(content=formatted, priority=10, protected=False, provider_name="episodic_memory")


_EPISODIC_MEMORY_INSTRUCTIONS = (
    "Past experiences from similar tasks may appear in [Past Experiences] blocks "
    "in the conversation. These are records of what was tried before and what "
    "happened.\n\n"
    "Successful experiences suggest proven approaches. Failed experiences with "
    "reflections indicate what to avoid and why. Use these to inform your "
    "strategy, but do not blindly repeat past approaches if the current "
    "situation differs meaningfully."
)


class EpisodicMemoryContributor:
    """System prompt contributor that teaches the agent how to use episodic memory.

    Adds instructions explaining the ``[Past Experiences]`` context blocks —
    how to interpret past successes and failures, and how to use them to
    inform strategy without blindly repeating past approaches.
    """

    def system_prompt_section(self) -> tuple[str, str]:
        return ("episodic_memory", _EPISODIC_MEMORY_INSTRUCTIONS)
