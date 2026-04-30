"""Tests for semantic memory tools: factory, execution, event emission, namespace isolation."""

from nanitics import (
    MockLLMClient,
    ReActAgent,
    Tool,
    ToolCall,
    ToolRegistry,
)
from nanitics.capabilities.memory.semantic import InMemorySemanticStore
from nanitics.capabilities.memory.semantic_tools import create_semantic_memory_tools
from nanitics.core.tools.function_tool import FunctionTool
from nanitics.infrastructure.embeddings import MockEmbeddingClient
from nanitics.infrastructure.observability.emitter import InMemoryEmitter
from nanitics.infrastructure.observability.events import (
    SemanticDeleteEvent,
    SemanticSearchEvent,
    SemanticStoreEvent,
)
from tests.testing_helpers import make_emitter, make_response


def make_store() -> InMemorySemanticStore:
    return InMemorySemanticStore(MockEmbeddingClient(dimension=32))


# ──────────────────────────────────────────────────────────
# Tool Factory
# ──────────────────────────────────────────────────────────


class TestCreateSemanticMemoryTools:
    def test_returns_three_tools(self) -> None:
        tools = create_semantic_memory_tools(make_store())
        assert len(tools) == 3

    def test_tools_are_function_tools(self) -> None:
        tools = create_semantic_memory_tools(make_store())
        for t in tools:
            assert isinstance(t, FunctionTool)
            assert isinstance(t, Tool)

    def test_tool_names(self) -> None:
        tools = create_semantic_memory_tools(make_store())
        names = {t.schema.name for t in tools}
        assert names == {"store_knowledge", "search_knowledge", "delete_knowledge"}

    def test_tool_schemas_have_descriptions(self) -> None:
        tools = create_semantic_memory_tools(make_store())
        for t in tools:
            assert t.schema.description


# ──────────────────────────────────────────────────────────
# Tool Execution
# ──────────────────────────────────────────────────────────


class TestSemanticMemoryToolExecution:
    def _get_tool(self, tools: list[FunctionTool], name: str) -> FunctionTool:
        return next(t for t in tools if t.schema.name == name)

    async def test_store_knowledge_tool(self) -> None:
        store = make_store()
        tools = create_semantic_memory_tools(store)
        store_tool = self._get_tool(tools, "store_knowledge")
        result = await store_tool.execute(content="Python is a programming language")
        assert "Stored" in result.content

    async def test_store_knowledge_with_metadata(self) -> None:
        store = make_store()
        tools = create_semantic_memory_tools(store)
        store_tool = self._get_tool(tools, "store_knowledge")
        result = await store_tool.execute(content="test content", metadata="source document")
        assert "Stored" in result.content

    async def test_search_knowledge_tool(self) -> None:
        store = make_store()
        tools = create_semantic_memory_tools(store)
        store_tool = self._get_tool(tools, "store_knowledge")
        search_tool = self._get_tool(tools, "search_knowledge")
        await store_tool.execute(content="Python is a programming language")
        result = await search_tool.execute(query="Python is a programming language")
        assert "Python" in result.content

    async def test_search_knowledge_no_results(self) -> None:
        store = make_store()
        tools = create_semantic_memory_tools(store)
        search_tool = self._get_tool(tools, "search_knowledge")
        result = await search_tool.execute(query="anything")
        assert "No matching" in result.content

    async def test_delete_knowledge_tool(self) -> None:
        store = make_store()
        tools = create_semantic_memory_tools(store)
        store_tool = self._get_tool(tools, "store_knowledge")
        search_tool = self._get_tool(tools, "search_knowledge")
        delete_tool = self._get_tool(tools, "delete_knowledge")
        store_result = await store_tool.execute(content="to delete")
        # Extract the ID from the result
        entry_id = store_result.content.split("id: ")[1].rstrip(").")
        result = await delete_tool.execute(id=entry_id)
        assert "Deleted" in result.content
        # Verify it's gone
        search_result = await search_tool.execute(query="to delete")
        assert "No matching" in search_result.content


# ──────────────────────────────────────────────────────────
# Tool Event Emission
# ──────────────────────────────────────────────────────────


class TestSemanticMemoryToolEvents:
    def _get_tool(self, tools: list[FunctionTool], name: str) -> FunctionTool:
        return next(t for t in tools if t.schema.name == name)

    def _make_registry(self, tools: list[FunctionTool], emitter: InMemoryEmitter) -> ToolRegistry:
        registry = ToolRegistry(emitter=emitter)
        for t in tools:
            registry.register(t)
        return registry

    async def test_store_emits_event(self) -> None:
        store = make_store()
        emitter = make_emitter()
        tools = create_semantic_memory_tools(store)
        registry = self._make_registry(tools, emitter)
        await registry.dispatch(
            ToolCall(
                id="1",
                name="store_knowledge",
                arguments={"content": "test fact"},
            )
        )
        events = [e for e in emitter.events if isinstance(e, SemanticStoreEvent)]
        assert len(events) == 1
        assert events[0].content == "test fact"
        assert events[0].trace_id == "test-trace"
        assert events[0].namespace is None

    async def test_search_emits_event(self) -> None:
        store = make_store()
        await store.add("some knowledge")
        emitter = make_emitter()
        tools = create_semantic_memory_tools(store)
        registry = self._make_registry(tools, emitter)
        await registry.dispatch(
            ToolCall(
                id="1",
                name="search_knowledge",
                arguments={"query": "knowledge"},
            )
        )
        events = [e for e in emitter.events if isinstance(e, SemanticSearchEvent)]
        assert len(events) == 1
        assert events[0].query == "knowledge"
        assert events[0].results_count == 1
        assert events[0].top_score is not None

    async def test_search_emits_event_no_results(self) -> None:
        store = make_store()
        emitter = make_emitter()
        tools = create_semantic_memory_tools(store)
        registry = self._make_registry(tools, emitter)
        await registry.dispatch(
            ToolCall(
                id="1",
                name="search_knowledge",
                arguments={"query": "anything"},
            )
        )
        events = [e for e in emitter.events if isinstance(e, SemanticSearchEvent)]
        assert len(events) == 1
        assert events[0].results_count == 0
        assert events[0].top_score is None

    async def test_delete_emits_event(self) -> None:
        store = make_store()
        entry_id = await store.add("to delete")
        emitter = make_emitter()
        tools = create_semantic_memory_tools(store)
        registry = self._make_registry(tools, emitter)
        await registry.dispatch(
            ToolCall(
                id="1",
                name="delete_knowledge",
                arguments={"id": entry_id},
            )
        )
        events = [e for e in emitter.events if isinstance(e, SemanticDeleteEvent)]
        assert len(events) == 1
        assert events[0].entry_id == entry_id

    async def test_store_event_includes_namespace(self) -> None:
        store = make_store()
        emitter = make_emitter()
        tools = create_semantic_memory_tools(store, namespace="agent1")
        registry = self._make_registry(tools, emitter)
        await registry.dispatch(
            ToolCall(
                id="1",
                name="store_knowledge",
                arguments={"content": "namespaced fact"},
            )
        )
        events = [e for e in emitter.events if isinstance(e, SemanticStoreEvent)]
        assert len(events) == 1
        assert events[0].namespace == "agent1"


# ──────────────────────────────────────────────────────────
# Namespace Isolation
# ──────────────────────────────────────────────────────────


class TestSemanticMemoryNamespaceIsolation:
    def _get_tool(self, tools: list[FunctionTool], name: str) -> FunctionTool:
        return next(t for t in tools if t.schema.name == name)

    async def test_namespace_isolates_search_results(self) -> None:
        store = make_store()
        tools_a = create_semantic_memory_tools(store, namespace="agent_a")
        tools_b = create_semantic_memory_tools(store, namespace="agent_b")

        store_a = self._get_tool(tools_a, "store_knowledge")
        store_b = self._get_tool(tools_b, "store_knowledge")
        search_a = self._get_tool(tools_a, "search_knowledge")
        search_b = self._get_tool(tools_b, "search_knowledge")

        await store_a.execute(content="fact for agent A")
        await store_b.execute(content="fact for agent B")

        result_a = await search_a.execute(query="fact for agent A")
        assert "agent A" in result_a.content
        assert "agent B" not in result_a.content

        result_b = await search_b.execute(query="fact for agent B")
        assert "agent B" in result_b.content
        assert "agent A" not in result_b.content

    async def test_namespace_filtering_does_not_truncate_results(self) -> None:
        """When namespace is active, results should not be truncated by other namespaces."""
        store = make_store()
        tools_a = create_semantic_memory_tools(store, namespace="target")
        tools_other = create_semantic_memory_tools(store, namespace="other")

        store_a = self._get_tool(tools_a, "store_knowledge")
        store_other = self._get_tool(tools_other, "store_knowledge")
        search_a = self._get_tool(tools_a, "search_knowledge")

        # Store 3 entries for the target namespace
        for i in range(3):
            await store_a.execute(content=f"target fact {i}")
        # Store 10 entries for another namespace (would crowd out target with old post-filter)
        for i in range(10):
            await store_other.execute(content=f"other fact {i}")

        result = await search_a.execute(query="target fact", limit=3)
        # All 3 target entries should be returned despite 10 other-namespace entries
        assert result.content.count("target fact") == 3


# ──────────────────────────────────────────────────────────
# Integration: Full Workflow via Registry
# ──────────────────────────────────────────────────────────


class TestSemanticMemoryRegistryIntegration:
    async def test_store_search_delete_via_registry(self) -> None:
        """Full cycle dispatched through ToolRegistry."""
        store = make_store()
        tools = create_semantic_memory_tools(store)
        registry = ToolRegistry()
        for t in tools:
            registry.register(t)

        # Store
        store_result = await registry.dispatch(
            ToolCall(
                id="1",
                name="store_knowledge",
                arguments={"content": "Python was created by Guido van Rossum"},
            )
        )
        assert "Stored" in store_result.content

        # Search
        search_result = await registry.dispatch(
            ToolCall(
                id="2",
                name="search_knowledge",
                arguments={"query": "Python was created by Guido van Rossum"},
            )
        )
        assert "Guido" in search_result.content

        # Extract ID from search result
        entry_id = search_result.content.split("id: ")[1].split(")")[0]

        # Delete
        delete_result = await registry.dispatch(
            ToolCall(
                id="3",
                name="delete_knowledge",
                arguments={"id": entry_id},
            )
        )
        assert "Deleted" in delete_result.content

        # Search after delete returns no results
        gone_result = await registry.dispatch(
            ToolCall(
                id="4",
                name="search_knowledge",
                arguments={"query": "Python creator"},
            )
        )
        assert "No matching" in gone_result.content


# ──────────────────────────────────────────────────────────
# Integration: ReActAgent with Semantic Memory Tools
# ──────────────────────────────────────────────────────────


class TestReActAgentSemanticMemory:
    async def test_agent_stores_and_searches(self) -> None:
        """ReActAgent with MockLLMClient stores knowledge then searches it."""
        store = make_store()
        emitter = make_emitter()
        tools = create_semantic_memory_tools(store)

        store_call = ToolCall(
            id="tc1",
            name="store_knowledge",
            arguments={"content": "The capital of France is Paris"},
        )
        search_call = ToolCall(
            id="tc2",
            name="search_knowledge",
            arguments={"query": "The capital of France is Paris"},
        )
        responses = [
            make_response(content="I'll store this fact.", tool_calls=[store_call]),
            make_response(content="Now I'll search for it.", tool_calls=[search_call]),
            make_response(content="The capital of France is Paris."),
        ]
        client = MockLLMClient(responses)
        agent = ReActAgent(
            name="semantic-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="You have semantic memory.",
            tools=tools,
        )

        result = await agent.run("What is the capital of France?")
        assert result.output == "The capital of France is Paris."

        # Verify events were emitted
        store_events = [e for e in emitter.events if isinstance(e, SemanticStoreEvent)]
        search_events = [e for e in emitter.events if isinstance(e, SemanticSearchEvent)]
        assert len(store_events) == 1
        assert len(search_events) == 1
        assert store_events[0].content == "The capital of France is Paris"
        assert search_events[0].results_count == 1
