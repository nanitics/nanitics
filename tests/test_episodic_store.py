"""Tests for episodic memory: data model, store operations, filters, eviction, pruning."""

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from nanitics.capabilities.memory.episodic import (
    Episode,
    EpisodeStore,
    InMemoryEpisodeStore,
    OutcomeType,
    RecallFilters,
    RecallResult,
    extract_episode,
)
from nanitics.core.agents.base import AgentResult
from nanitics.infrastructure.embeddings import MockEmbeddingClient
from nanitics.infrastructure.llm.protocol import Message, TextContentBlock, ToolCall
from nanitics.infrastructure.observability.events import Usage


def make_store(max_episodes: int | None = None) -> InMemoryEpisodeStore:
    return InMemoryEpisodeStore(MockEmbeddingClient(dimension=32), max_episodes=max_episodes)


def make_episode(**kwargs: Any) -> Episode:
    defaults: dict[str, Any] = {
        "situation": "test situation",
        "action": "test action",
        "outcome": OutcomeType.SUCCESS,
    }
    defaults.update(kwargs)
    return Episode(**defaults)


# ──────────────────────────────────────────────────────────
# Episode Model
# ──────────────────────────────────────────────────────────


class TestEpisodeModel:
    def test_construction_with_defaults(self) -> None:
        ep = make_episode()
        assert ep.situation == "test situation"
        assert ep.action == "test action"
        assert ep.outcome == OutcomeType.SUCCESS
        assert ep.id  # auto-generated UUID
        assert ep.timestamp  # auto-generated
        assert ep.outcome_detail is None
        assert ep.reflection is None
        assert ep.metadata is None

    def test_construction_with_all_fields(self) -> None:
        ts = datetime.now(UTC)
        ep = Episode(
            id="custom-id",
            situation="debug failing test",
            action="used debugger",
            outcome=OutcomeType.FAILURE,
            outcome_detail="test still fails",
            reflection="should check logs first",
            evaluator_feedback="Output must mention the failing assertion id.",
            metadata={"task_type": "debugging"},
            timestamp=ts,
        )
        assert ep.id == "custom-id"
        assert ep.outcome == OutcomeType.FAILURE
        assert ep.outcome_detail == "test still fails"
        assert ep.reflection == "should check logs first"
        assert ep.evaluator_feedback == "Output must mention the failing assertion id."
        assert ep.metadata == {"task_type": "debugging"}
        assert ep.timestamp == ts

    def test_evaluator_feedback_defaults_to_none(self) -> None:
        ep = make_episode()
        assert ep.evaluator_feedback is None

    def test_evaluator_feedback_round_trip(self) -> None:
        feedback = "The haiku must include the literal word 'jellyfish'."
        ep = make_episode(evaluator_feedback=feedback)
        assert ep.evaluator_feedback == feedback

    def test_frozen(self) -> None:
        ep = make_episode()
        with pytest.raises(ValidationError):
            ep.situation = "changed"

    def test_outcome_types(self) -> None:
        assert OutcomeType.SUCCESS.value == "success"
        assert OutcomeType.FAILURE.value == "failure"
        assert OutcomeType.PARTIAL.value == "partial"


# ──────────────────────────────────────────────────────────
# Record / Recall / Forget / Count
# ──────────────────────────────────────────────────────────


class TestInMemoryEpisodeStore:
    async def test_record_returns_id(self) -> None:
        store = make_store()
        ep = make_episode()
        episode_id = await store.record(ep)
        assert episode_id == ep.id

    async def test_count_empty(self) -> None:
        store = make_store()
        assert await store.count() == 0

    async def test_count_after_record(self) -> None:
        store = make_store()
        await store.record(make_episode())
        await store.record(make_episode())
        assert await store.count() == 2

    async def test_recall_empty_store(self) -> None:
        store = make_store()
        results = await store.recall("anything")
        assert results == []

    async def test_recall_returns_recorded_episode(self) -> None:
        store = make_store()
        ep = make_episode(situation="deploy to production")
        await store.record(ep)
        results = await store.recall("deploy to production")
        assert len(results) == 1
        assert results[0].episode.situation == "deploy to production"

    async def test_recall_exact_match_has_high_score(self) -> None:
        store = make_store()
        await store.record(make_episode(situation="deploy to production"))
        results = await store.recall("deploy to production")
        assert abs(results[0].similarity_score - 1.0) < 1e-6

    async def test_recall_respects_limit(self) -> None:
        store = make_store()
        for i in range(10):
            await store.record(make_episode(situation=f"task {i}"))
        results = await store.recall("task", limit=3)
        assert len(results) == 3

    async def test_recall_sorted_by_similarity(self) -> None:
        store = make_store()
        await store.record(make_episode(situation="alpha"))
        await store.record(make_episode(situation="beta"))
        await store.record(make_episode(situation="gamma"))
        results = await store.recall("alpha")
        assert results[0].episode.situation == "alpha"
        for i in range(len(results) - 1):
            assert results[i].similarity_score >= results[i + 1].similarity_score

    async def test_forget_removes_episode(self) -> None:
        store = make_store()
        ep = make_episode(situation="to forget")
        await store.record(ep)
        await store.forget(ep.id)
        assert await store.count() == 0
        results = await store.recall("to forget")
        assert results == []

    async def test_forget_nonexistent_is_noop(self) -> None:
        store = make_store()
        await store.record(make_episode())
        await store.forget("nonexistent-id")
        assert await store.count() == 1

    async def test_recall_result_structure(self) -> None:
        store = make_store()
        ep = make_episode()
        await store.record(ep)
        results = await store.recall("test situation")
        assert isinstance(results[0], RecallResult)
        assert isinstance(results[0].episode, Episode)
        assert isinstance(results[0].similarity_score, float)

    def test_satisfies_protocol(self) -> None:
        store = make_store()
        assert isinstance(store, EpisodeStore)


# ──────────────────────────────────────────────────────────
# Recall Filters
# ──────────────────────────────────────────────────────────


class TestRecallFilters:
    async def test_filter_by_outcome(self) -> None:
        store = make_store()
        await store.record(make_episode(situation="task A", outcome=OutcomeType.SUCCESS))
        await store.record(make_episode(situation="task A", outcome=OutcomeType.FAILURE))
        results = await store.recall("task A", filters=RecallFilters(outcome=OutcomeType.SUCCESS))
        assert len(results) == 1
        assert results[0].episode.outcome == OutcomeType.SUCCESS

    async def test_filter_by_metadata(self) -> None:
        store = make_store()
        await store.record(make_episode(situation="task", metadata={"task_type": "coding"}))
        await store.record(make_episode(situation="task", metadata={"task_type": "research"}))
        results = await store.recall(
            "task",
            filters=RecallFilters(metadata_filters={"task_type": "coding"}),
        )
        assert len(results) == 1
        assert results[0].episode.metadata is not None
        assert results[0].episode.metadata["task_type"] == "coding"

    async def test_filter_by_metadata_excludes_none_metadata(self) -> None:
        store = make_store()
        await store.record(make_episode(situation="task"))  # no metadata
        results = await store.recall(
            "task",
            filters=RecallFilters(metadata_filters={"task_type": "coding"}),
        )
        assert len(results) == 0

    async def test_filter_by_min_score(self) -> None:
        store = make_store()
        await store.record(make_episode(situation="exact match query"))
        await store.record(make_episode(situation="something completely different"))
        results = await store.recall("exact match query", filters=RecallFilters(min_score=0.99))
        # Only the exact match should survive the high threshold
        assert len(results) == 1
        assert results[0].episode.situation == "exact match query"

    async def test_filter_by_time_range(self) -> None:
        store = make_store()
        old_time = datetime(2024, 1, 1, tzinfo=UTC)
        new_time = datetime(2025, 1, 1, tzinfo=UTC)
        await store.record(make_episode(situation="old task", timestamp=old_time))
        await store.record(make_episode(situation="new task", timestamp=new_time))
        cutoff = datetime(2024, 6, 1, tzinfo=UTC)
        results = await store.recall("task", filters=RecallFilters(after=cutoff))
        assert len(results) == 1
        assert results[0].episode.situation == "new task"

    async def test_filter_before(self) -> None:
        store = make_store()
        old_time = datetime(2024, 1, 1, tzinfo=UTC)
        new_time = datetime(2025, 1, 1, tzinfo=UTC)
        await store.record(make_episode(situation="old task", timestamp=old_time))
        await store.record(make_episode(situation="new task", timestamp=new_time))
        cutoff = datetime(2024, 6, 1, tzinfo=UTC)
        results = await store.recall("task", filters=RecallFilters(before=cutoff))
        assert len(results) == 1
        assert results[0].episode.situation == "old task"


# ──────────────────────────────────────────────────────────
# Capacity Eviction
# ──────────────────────────────────────────────────────────


class TestCapacityEviction:
    async def test_evicts_oldest_when_full(self) -> None:
        store = make_store(max_episodes=2)
        t1 = datetime(2024, 1, 1, tzinfo=UTC)
        t2 = datetime(2024, 2, 1, tzinfo=UTC)
        t3 = datetime(2024, 3, 1, tzinfo=UTC)
        await store.record(make_episode(situation="first", timestamp=t1))
        await store.record(make_episode(situation="second", timestamp=t2))
        await store.record(make_episode(situation="third", timestamp=t3))
        assert await store.count() == 2
        results = await store.recall("first")
        # "first" should have been evicted
        situations = [r.episode.situation for r in results]
        assert "first" not in situations
        assert "second" in situations
        assert "third" in situations

    async def test_no_eviction_when_under_capacity(self) -> None:
        store = make_store(max_episodes=5)
        await store.record(make_episode(situation="one"))
        await store.record(make_episode(situation="two"))
        assert await store.count() == 2

    async def test_no_eviction_when_unlimited(self) -> None:
        store = make_store(max_episodes=None)
        for i in range(20):
            await store.record(make_episode(situation=f"ep {i}"))
        assert await store.count() == 20


# ──────────────────────────────────────────────────────────
# Prune Superseded
# ──────────────────────────────────────────────────────────


class TestPruneSuperseded:
    async def test_removes_failures_when_success_exists(self) -> None:
        store = make_store()
        await store.record(make_episode(situation="deploy app", outcome=OutcomeType.FAILURE))
        await store.record(make_episode(situation="deploy app", outcome=OutcomeType.SUCCESS))
        removed = await store.prune_superseded("deploy app", similarity_threshold=0.99)
        assert len(removed) == 1
        assert await store.count() == 1
        results = await store.recall("deploy app")
        assert results[0].episode.outcome == OutcomeType.SUCCESS

    async def test_noop_when_no_success(self) -> None:
        store = make_store()
        await store.record(make_episode(situation="deploy app", outcome=OutcomeType.FAILURE))
        await store.record(make_episode(situation="deploy app", outcome=OutcomeType.FAILURE))
        removed = await store.prune_superseded("deploy app", similarity_threshold=0.99)
        assert len(removed) == 0
        assert await store.count() == 2

    async def test_noop_on_empty_store(self) -> None:
        store = make_store()
        removed = await store.prune_superseded("anything")
        assert removed == []

    async def test_only_prunes_similar_episodes(self) -> None:
        store = make_store()
        await store.record(make_episode(situation="deploy app", outcome=OutcomeType.FAILURE))
        await store.record(make_episode(situation="deploy app", outcome=OutcomeType.SUCCESS))
        await store.record(
            make_episode(
                situation="something completely unrelated and different",
                outcome=OutcomeType.FAILURE,
            )
        )
        removed = await store.prune_superseded("deploy app", similarity_threshold=0.99)
        assert len(removed) == 1
        assert await store.count() == 2  # success + unrelated failure remain


# ──────────────────────────────────────────────────────────
# Episode Formation (extract_episode)
# ──────────────────────────────────────────────────────────


def _make_usage() -> Usage:
    return Usage(input_tokens=10, output_tokens=5)


def _make_result(
    termination_reason: str = "complete",
    messages: list[Message] | None = None,
) -> AgentResult:
    if messages is None:
        messages = [
            Message(role="user", content="solve 2+2"),
            Message(
                role="assistant",
                content="I'll calculate this.",
                tool_calls=[ToolCall(id="1", name="calculator", arguments={"expr": "2+2"})],
            ),
            Message(role="tool_result", content="4", tool_call_id="1", name="calculator"),
            Message(role="assistant", content="The answer is 4."),
        ]
    return AgentResult(
        output="The answer is 4.",
        total_steps=2,
        termination_reason=termination_reason,
        messages=messages,
        usage=_make_usage(),
    )


class TestExtractEpisode:
    def test_basic_extraction(self) -> None:
        result = _make_result()
        ep = extract_episode("solve 2+2", result)
        assert ep.situation == "solve 2+2"
        assert ep.outcome == OutcomeType.SUCCESS
        assert "calculator" in ep.action
        assert ep.id  # auto-generated
        assert ep.timestamp

    def test_explicit_outcome(self) -> None:
        result = _make_result()
        ep = extract_episode("solve 2+2", result, outcome=OutcomeType.PARTIAL)
        assert ep.outcome == OutcomeType.PARTIAL

    def test_inferred_outcome_complete(self) -> None:
        result = _make_result(termination_reason="complete")
        ep = extract_episode("task", result)
        assert ep.outcome == OutcomeType.SUCCESS

    def test_inferred_outcome_iteration_limit(self) -> None:
        result = _make_result(termination_reason="iteration_limit")
        ep = extract_episode("task", result)
        assert ep.outcome == OutcomeType.FAILURE

    def test_inferred_outcome_cancelled(self) -> None:
        result = _make_result(termination_reason="cancelled")
        ep = extract_episode("task", result)
        assert ep.outcome == OutcomeType.FAILURE

    def test_inferred_outcome_evaluation_failed(self) -> None:
        result = _make_result(termination_reason="evaluation_failed")
        ep = extract_episode("task", result)
        assert ep.outcome == OutcomeType.FAILURE

    def test_inferred_outcome_unknown_defaults_to_partial(self) -> None:
        result = _make_result(termination_reason="unknown_reason")
        ep = extract_episode("task", result)
        assert ep.outcome == OutcomeType.PARTIAL

    def test_trajectory_summarization_includes_tools(self) -> None:
        result = _make_result()
        ep = extract_episode("task", result)
        assert "calculator" in ep.action
        assert "Tools used:" in ep.action

    def test_trajectory_summarization_includes_first_assistant_message(self) -> None:
        result = _make_result()
        ep = extract_episode("task", result)
        assert "calculate" in ep.action

    def test_trajectory_summarization_truncates_long_content(self) -> None:
        messages = [
            Message(role="assistant", content="A" * 300),
        ]
        result = _make_result(messages=messages)
        ep = extract_episode("task", result)
        assert len(ep.action) < 300
        assert "..." in ep.action

    def test_optional_fields_pass_through(self) -> None:
        result = _make_result()
        ep = extract_episode(
            "task",
            result,
            outcome_detail="evaluation said it was good",
            reflection="the approach worked well",
            metadata={"task_type": "math"},
        )
        assert ep.outcome_detail == "evaluation said it was good"
        assert ep.reflection == "the approach worked well"
        assert ep.metadata == {"task_type": "math"}

    def test_evaluator_feedback_default_is_none(self) -> None:
        result = _make_result()
        ep = extract_episode("task", result)
        assert ep.evaluator_feedback is None

    def test_evaluator_feedback_pass_through(self) -> None:
        result = _make_result()
        ep = extract_episode(
            "task",
            result,
            outcome=OutcomeType.FAILURE,
            evaluator_feedback="Output must include 'jellyfish' and 'lighthouse'.",
        )
        assert ep.evaluator_feedback == "Output must include 'jellyfish' and 'lighthouse'."

    def test_empty_messages(self) -> None:
        result = _make_result(messages=[])
        ep = extract_episode("task", result)
        assert ep.action == "No actions recorded"

    def test_deduplicates_tool_names(self) -> None:
        messages = [
            Message(
                role="assistant",
                content="step 1",
                tool_calls=[ToolCall(id="1", name="search", arguments={})],
            ),
            Message(role="tool_result", content="result", tool_call_id="1", name="search"),
            Message(
                role="assistant",
                content="step 2",
                tool_calls=[ToolCall(id="2", name="search", arguments={})],
            ),
        ]
        result = _make_result(messages=messages)
        ep = extract_episode("task", result)
        # "search" should appear only once in the tools list
        assert ep.action.count("search") == 1


# ──────────────────────────────────────────────────────────
# Cosine Similarity
# ──────────────────────────────────────────────────────────


class TestCosineSimilarity:
    def test_zero_vector_returns_zero(self) -> None:
        from nanitics.capabilities.memory._similarity import cosine_similarity

        assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0


# ──────────────────────────────────────────────────────────
# Multimodal Situation
# ──────────────────────────────────────────────────────────


class TestMultimodalSituation:
    async def test_record_with_content_block_situation(self) -> None:
        store = make_store()
        ep = make_episode(
            situation=[
                TextContentBlock(text="visual context"),
                TextContentBlock(text="more context"),
            ]
        )
        await store.record(ep)
        results = await store.recall("visual context")
        assert len(results) == 1
