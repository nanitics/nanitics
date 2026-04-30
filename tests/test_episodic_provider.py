"""Tests for episodic memory context provider and system prompt contributor."""

from nanitics.capabilities.memory.episodic import (
    Episode,
    EpisodicMemoryContributor,
    EpisodicMemoryProvider,
    InMemoryEpisodeStore,
    OutcomeType,
)
from nanitics.infrastructure.embeddings import MockEmbeddingClient
from nanitics.infrastructure.llm.protocol import Message
from nanitics.infrastructure.observability.events import EpisodeRecallEvent
from tests.testing_helpers import make_emitter


def make_store() -> InMemoryEpisodeStore:
    return InMemoryEpisodeStore(MockEmbeddingClient(dimension=32))


# ──────────────────────────────────────────────────────────
# EpisodicMemoryProvider
# ──────────────────────────────────────────────────────────


class TestEpisodicMemoryProvider:
    async def test_returns_none_on_empty_store(self) -> None:
        store = make_store()
        provider = EpisodicMemoryProvider(store)
        messages = [Message(role="user", content="solve 2+2")]
        result = await provider.provide(messages)
        assert result is None

    async def test_retrieves_and_formats_episodes(self) -> None:
        store = make_store()
        await store.record(
            Episode(
                situation="solve math problem",
                action="used calculator tool",
                outcome=OutcomeType.SUCCESS,
                outcome_detail="got correct answer",
                reflection="calculator is reliable for arithmetic",
            )
        )
        provider = EpisodicMemoryProvider(store)
        messages = [Message(role="user", content="solve math problem")]
        result = await provider.provide(messages)
        assert result is not None
        assert "[Past Experiences]" in result.content
        assert "solve math problem" in result.content
        assert "used calculator tool" in result.content
        assert "got correct answer" in result.content
        assert "calculator is reliable" in result.content
        assert result.priority == 10
        assert result.protected is False

    async def test_uses_latest_user_message(self) -> None:
        store = make_store()
        await store.record(
            Episode(
                situation="deploy app",
                action="used CI",
                outcome=OutcomeType.SUCCESS,
            )
        )
        provider = EpisodicMemoryProvider(store)
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
            await store.record(
                Episode(
                    situation=f"task {i}",
                    action=f"action {i}",
                    outcome=OutcomeType.SUCCESS,
                )
            )
        provider = EpisodicMemoryProvider(store, limit=2)
        messages = [Message(role="user", content="task")]
        result = await provider.provide(messages)
        assert result is not None
        assert result.content.count("## Experience") == 2

    async def test_respects_outcome_filter(self) -> None:
        store = make_store()
        await store.record(
            Episode(
                situation="deploy app",
                action="used CI",
                outcome=OutcomeType.SUCCESS,
            )
        )
        await store.record(
            Episode(
                situation="deploy app",
                action="manual deploy",
                outcome=OutcomeType.FAILURE,
            )
        )
        provider = EpisodicMemoryProvider(store, outcome_filter=OutcomeType.SUCCESS)
        messages = [Message(role="user", content="deploy app")]
        result = await provider.provide(messages)
        assert result is not None
        assert "success" in result.content
        assert "manual deploy" not in result.content

    async def test_returns_none_when_no_user_messages(self) -> None:
        store = make_store()
        await store.record(
            Episode(
                situation="task",
                action="action",
                outcome=OutcomeType.SUCCESS,
            )
        )
        provider = EpisodicMemoryProvider(store)
        messages = [Message(role="assistant", content="hello")]
        result = await provider.provide(messages)
        assert result is None

    async def test_emits_recall_event(self) -> None:
        store = make_store()
        await store.record(
            Episode(
                situation="task",
                action="action",
                outcome=OutcomeType.SUCCESS,
            )
        )
        emitter = make_emitter()
        provider = EpisodicMemoryProvider(store, emitter=emitter)
        messages = [Message(role="user", content="task")]
        await provider.provide(messages)
        events = [e for e in emitter.events if isinstance(e, EpisodeRecallEvent)]
        assert len(events) == 1
        assert events[0].query == "task"
        assert events[0].results_count == 1

    async def test_renders_evaluator_feedback_when_set(self) -> None:
        store = make_store()
        await store.record(
            Episode(
                situation="write a haiku",
                action="produced first attempt",
                outcome=OutcomeType.FAILURE,
                evaluator_feedback="The haiku must include the literal word 'jellyfish'.",
            )
        )
        provider = EpisodicMemoryProvider(store)
        messages = [Message(role="user", content="write a haiku")]
        result = await provider.provide(messages)
        assert result is not None
        assert "Evaluator feedback: The haiku must include the literal word 'jellyfish'." in result.content

    async def test_omits_evaluator_feedback_when_none(self) -> None:
        store = make_store()
        await store.record(
            Episode(
                situation="write a haiku",
                action="produced first attempt",
                outcome=OutcomeType.SUCCESS,
            )
        )
        provider = EpisodicMemoryProvider(store)
        messages = [Message(role="user", content="write a haiku")]
        result = await provider.provide(messages)
        assert result is not None
        assert "Evaluator feedback:" not in result.content

    async def test_evaluator_feedback_renders_above_reflection(self) -> None:
        store = make_store()
        await store.record(
            Episode(
                situation="write a haiku",
                action="produced first attempt",
                outcome=OutcomeType.FAILURE,
                evaluator_feedback="Must include 'jellyfish'.",
                reflection="Next time, mention sea creatures explicitly.",
            )
        )
        provider = EpisodicMemoryProvider(store)
        messages = [Message(role="user", content="write a haiku")]
        result = await provider.provide(messages)
        assert result is not None
        feedback_idx = result.content.index("Evaluator feedback:")
        reflection_idx = result.content.index("Reflection:")
        assert feedback_idx < reflection_idx, (
            "Verbatim evaluator feedback must appear above the LLM-narrative reflection — "
            "the imperative belongs ahead of the paraphrase."
        )


# ──────────────────────────────────────────────────────────
# EpisodicMemoryContributor
# ──────────────────────────────────────────────────────────


class TestEpisodicMemoryContributor:
    def test_returns_section(self) -> None:
        contributor = EpisodicMemoryContributor()
        section = contributor.system_prompt_section()
        assert section is not None
        name, content = section
        assert name == "episodic_memory"
        assert "Past Experiences" in content
        assert "proven approaches" in content
