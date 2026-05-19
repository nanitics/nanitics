"""Tests for episodic memory tools: factory, execution, event emission, namespace isolation."""

from nanitics.capabilities.memory.episodic import Episode, InMemoryEpisodeStore, OutcomeType
from nanitics.capabilities.memory.episodic_tools import create_episodic_memory_tools
from nanitics.infrastructure.embeddings import MockEmbeddingClient
from nanitics.infrastructure.observability.emitter import InMemoryEmitter
from nanitics.infrastructure.observability.events import (
    EpisodeForgetEvent,
    EpisodeRecallEvent,
    EpisodeRecordEvent,
)
from nanitics.strategies import (
    Tool,
    ToolRegistry,
)
from nanitics.strategies.tools.function_tool import FunctionTool
from nanitics.tracing import ToolCall
from tests.testing_helpers import make_emitter


def make_store() -> InMemoryEpisodeStore:
    return InMemoryEpisodeStore(MockEmbeddingClient(dimension=32))


# ──────────────────────────────────────────────────────────
# Tool Factory
# ──────────────────────────────────────────────────────────


class TestCreateEpisodicMemoryTools:
    def test_returns_three_tools(self) -> None:
        tools = create_episodic_memory_tools(make_store())
        assert len(tools) == 3

    def test_tools_are_function_tools(self) -> None:
        tools = create_episodic_memory_tools(make_store())
        for t in tools:
            assert isinstance(t, FunctionTool)
            assert isinstance(t, Tool)

    def test_tool_names(self) -> None:
        tools = create_episodic_memory_tools(make_store())
        names = {t.schema.name for t in tools}
        assert names == {"recall_episodes", "record_episode", "forget_episode"}

    def test_tool_schemas_have_descriptions(self) -> None:
        tools = create_episodic_memory_tools(make_store())
        for t in tools:
            assert t.schema.description


# ──────────────────────────────────────────────────────────
# Tool Execution
# ──────────────────────────────────────────────────────────


class TestEpisodicMemoryToolExecution:
    def _get_tool(self, tools: list[FunctionTool], name: str) -> FunctionTool:
        return next(t for t in tools if t.schema.name == name)

    async def test_record_episode_tool(self) -> None:
        store = make_store()
        tools = create_episodic_memory_tools(store)
        record_tool = self._get_tool(tools, "record_episode")
        result = await record_tool.execute(
            situation="debug failing test",
            action="used print statements",
            outcome="success",
        )
        assert "Recorded" in result.content

    async def test_recall_episodes_tool(self) -> None:
        store = make_store()
        tools = create_episodic_memory_tools(store)
        record_tool = self._get_tool(tools, "record_episode")
        recall_tool = self._get_tool(tools, "recall_episodes")
        await record_tool.execute(
            situation="debug failing test",
            action="used print statements",
            outcome="success",
        )
        result = await recall_tool.execute(query="debug failing test")
        assert "debug failing test" in result.content

    async def test_recall_no_results(self) -> None:
        store = make_store()
        tools = create_episodic_memory_tools(store)
        recall_tool = self._get_tool(tools, "recall_episodes")
        result = await recall_tool.execute(query="anything")
        assert "No matching" in result.content

    async def test_recall_with_outcome_detail(self) -> None:
        store = make_store()
        ep = Episode(
            situation="debug failing test",
            action="used debugger",
            outcome=OutcomeType.SUCCESS,
            outcome_detail="fixed the off-by-one error",
        )
        await store.record(ep)
        tools = create_episodic_memory_tools(store)
        recall_tool = self._get_tool(tools, "recall_episodes")
        result = await recall_tool.execute(query="debug failing test")
        assert "Outcome: fixed the off-by-one error" in result.content

    async def test_forget_episode_tool(self) -> None:
        store = make_store()
        tools = create_episodic_memory_tools(store)
        record_tool = self._get_tool(tools, "record_episode")
        forget_tool = self._get_tool(tools, "forget_episode")
        record_result = await record_tool.execute(
            situation="test",
            action="test",
            outcome="success",
        )
        episode_id = record_result.content.split("id: ")[1].rstrip(").")
        result = await forget_tool.execute(id=episode_id)
        assert "Forgot" in result.content
        assert await store.count() == 0

    async def test_record_with_optional_fields(self) -> None:
        store = make_store()
        tools = create_episodic_memory_tools(store)
        record_tool = self._get_tool(tools, "record_episode")
        result = await record_tool.execute(
            situation="deploy app",
            action="used CI pipeline",
            outcome="failure",
            outcome_detail="timeout in health check",
            reflection="should increase timeout",
        )
        assert "Recorded" in result.content


# ──────────────────────────────────────────────────────────
# Event Emission
# ──────────────────────────────────────────────────────────


class TestEpisodicMemoryToolEvents:
    def _get_tool(self, tools: list[FunctionTool], name: str) -> FunctionTool:
        return next(t for t in tools if t.schema.name == name)

    def _make_registry(self, tools: list[FunctionTool], emitter: InMemoryEmitter) -> ToolRegistry:
        registry = ToolRegistry(emitter=emitter)
        for t in tools:
            registry.register(t)
        return registry

    async def test_record_emits_event(self) -> None:
        store = make_store()
        emitter = make_emitter()
        tools = create_episodic_memory_tools(store)
        registry = self._make_registry(tools, emitter)
        await registry.dispatch(
            ToolCall(
                id="1",
                name="record_episode",
                arguments={
                    "situation": "test task",
                    "action": "tried approach A",
                    "outcome": "success",
                },
            )
        )
        events = [e for e in emitter.events if isinstance(e, EpisodeRecordEvent)]
        assert len(events) == 1
        assert events[0].situation == "test task"
        assert events[0].outcome == "success"
        assert events[0].has_reflection is False
        assert events[0].trace_id == "test-trace"
        assert events[0].namespace is None

    async def test_recall_emits_event(self) -> None:
        store = make_store()
        await store.record(
            Episode(
                situation="past task",
                action="past action",
                outcome=OutcomeType.SUCCESS,
            )
        )
        emitter = make_emitter()
        tools = create_episodic_memory_tools(store)
        registry = self._make_registry(tools, emitter)
        await registry.dispatch(
            ToolCall(
                id="1",
                name="recall_episodes",
                arguments={"query": "past task"},
            )
        )
        events = [e for e in emitter.events if isinstance(e, EpisodeRecallEvent)]
        assert len(events) == 1
        assert events[0].query == "past task"
        assert events[0].results_count == 1
        assert events[0].top_score is not None

    async def test_recall_emits_event_no_results(self) -> None:
        store = make_store()
        emitter = make_emitter()
        tools = create_episodic_memory_tools(store)
        registry = self._make_registry(tools, emitter)
        await registry.dispatch(
            ToolCall(
                id="1",
                name="recall_episodes",
                arguments={"query": "anything"},
            )
        )
        events = [e for e in emitter.events if isinstance(e, EpisodeRecallEvent)]
        assert len(events) == 1
        assert events[0].results_count == 0
        assert events[0].top_score is None

    async def test_forget_emits_event(self) -> None:
        store = make_store()
        ep = Episode(
            situation="to forget",
            action="some action",
            outcome=OutcomeType.SUCCESS,
        )
        await store.record(ep)
        emitter = make_emitter()
        tools = create_episodic_memory_tools(store)
        registry = self._make_registry(tools, emitter)
        await registry.dispatch(
            ToolCall(
                id="1",
                name="forget_episode",
                arguments={"id": ep.id},
            )
        )
        events = [e for e in emitter.events if isinstance(e, EpisodeForgetEvent)]
        assert len(events) == 1
        assert events[0].episode_id == ep.id

    async def test_record_event_includes_namespace(self) -> None:
        store = make_store()
        emitter = make_emitter()
        tools = create_episodic_memory_tools(store, namespace="agent1")
        registry = self._make_registry(tools, emitter)
        await registry.dispatch(
            ToolCall(
                id="1",
                name="record_episode",
                arguments={
                    "situation": "test",
                    "action": "test",
                    "outcome": "success",
                },
            )
        )
        events = [e for e in emitter.events if isinstance(e, EpisodeRecordEvent)]
        assert len(events) == 1
        assert events[0].namespace == "agent1"


# ──────────────────────────────────────────────────────────
# Namespace Isolation
# ──────────────────────────────────────────────────────────


class TestEpisodicMemoryNamespaceIsolation:
    def _get_tool(self, tools: list[FunctionTool], name: str) -> FunctionTool:
        return next(t for t in tools if t.schema.name == name)

    async def test_namespace_isolates_recall_results(self) -> None:
        store = make_store()
        tools_a = create_episodic_memory_tools(store, namespace="agent_a")
        tools_b = create_episodic_memory_tools(store, namespace="agent_b")

        record_a = self._get_tool(tools_a, "record_episode")
        record_b = self._get_tool(tools_b, "record_episode")
        recall_a = self._get_tool(tools_a, "recall_episodes")
        recall_b = self._get_tool(tools_b, "recall_episodes")

        await record_a.execute(
            situation="task for agent A",
            action="approach A",
            outcome="success",
        )
        await record_b.execute(
            situation="task for agent B",
            action="approach B",
            outcome="success",
        )

        result_a = await recall_a.execute(query="task for agent A")
        assert "agent A" in result_a.content
        assert "agent B" not in result_a.content

        result_b = await recall_b.execute(query="task for agent B")
        assert "agent B" in result_b.content
        assert "agent A" not in result_b.content


# ──────────────────────────────────────────────────────────
# Invalid Outcome Handling
# ──────────────────────────────────────────────────────────


class TestEpisodicMemoryInvalidOutcome:
    def _get_tool(self, tools: list[FunctionTool], name: str) -> FunctionTool:
        return next(t for t in tools if t.schema.name == name)

    async def test_record_invalid_outcome_returns_error(self) -> None:
        store = make_store()
        tools = create_episodic_memory_tools(store)
        record_tool = self._get_tool(tools, "record_episode")
        result = await record_tool.execute(
            situation="test",
            action="test",
            outcome="succeeded",
        )
        assert "Invalid outcome" in result.content
        assert "'succeeded'" in result.content
        assert "'success'" in result.content
        assert await store.count() == 0

    async def test_recall_invalid_outcome_filter_returns_error(self) -> None:
        store = make_store()
        tools = create_episodic_memory_tools(store)
        recall_tool = self._get_tool(tools, "recall_episodes")
        result = await recall_tool.execute(query="test", outcome_filter="won")
        assert "Invalid outcome_filter" in result.content
        assert "'won'" in result.content
        assert "'success'" in result.content
