"""Tests for long-term memory: protocol, in-memory store, events, tools, and integration."""

import pytest
from pydantic import ValidationError

from nanitics import (
    FunctionTool,
    InMemoryEmitter,
    InMemoryLongTermStore,
    LongTermStore,
    MockLLMClient,
    ReActAgent,
    Tool,
    ToolCall,
    ToolRegistry,
    create_long_term_memory_tools,
)
from nanitics.infrastructure import (
    LongTermDeleteEvent,
    LongTermListEvent,
    LongTermRetrieveEvent,
    LongTermStoreEvent,
)
from tests.testing_helpers import make_emitter, make_response

# ──────────────────────────────────────────────────────────
# Protocol Conformance
# ──────────────────────────────────────────────────────────


class TestLongTermStoreProtocol:
    def test_in_memory_store_satisfies_protocol(self) -> None:
        store = InMemoryLongTermStore()
        assert isinstance(store, LongTermStore)


# ──────────────────────────────────────────────────────────
# InMemoryLongTermStore CRUD
# ──────────────────────────────────────────────────────────


class TestInMemoryLongTermStore:
    async def test_store_and_retrieve(self) -> None:
        store = InMemoryLongTermStore()
        await store.store("key1", "value1")
        assert await store.retrieve("key1") == "value1"

    async def test_retrieve_missing_returns_none(self) -> None:
        store = InMemoryLongTermStore()
        assert await store.retrieve("nonexistent") is None

    async def test_store_overwrites_existing(self) -> None:
        store = InMemoryLongTermStore()
        await store.store("key1", "original")
        await store.store("key1", "updated")
        assert await store.retrieve("key1") == "updated"

    async def test_delete_removes_key(self) -> None:
        store = InMemoryLongTermStore()
        await store.store("key1", "value1")
        await store.delete("key1")
        assert await store.retrieve("key1") is None

    async def test_delete_nonexistent_is_idempotent(self) -> None:
        store = InMemoryLongTermStore()
        await store.delete("nonexistent")  # Should not raise

    async def test_list_keys_empty(self) -> None:
        store = InMemoryLongTermStore()
        assert await store.list_keys() == []

    async def test_list_keys_returns_stored_keys(self) -> None:
        store = InMemoryLongTermStore()
        await store.store("alpha", "a")
        await store.store("beta", "b")
        keys = await store.list_keys()
        assert sorted(keys) == ["alpha", "beta"]

    async def test_namespace_isolation(self) -> None:
        store = InMemoryLongTermStore()
        await store.store("key1", "global_val")
        await store.store("key1", "ns_val", namespace="ns1")
        assert await store.retrieve("key1") == "global_val"
        assert await store.retrieve("key1", namespace="ns1") == "ns_val"

    async def test_list_keys_namespace_scoped(self) -> None:
        store = InMemoryLongTermStore()
        await store.store("global_key", "v")
        await store.store("ns_key", "v", namespace="ns1")
        assert await store.list_keys() == ["global_key"]
        assert await store.list_keys(namespace="ns1") == ["ns_key"]

    async def test_delete_in_namespace_does_not_affect_other(self) -> None:
        store = InMemoryLongTermStore()
        await store.store("key1", "global", namespace=None)
        await store.store("key1", "namespaced", namespace="ns1")
        await store.delete("key1", namespace="ns1")
        assert await store.retrieve("key1") == "global"
        assert await store.retrieve("key1", namespace="ns1") is None


# ──────────────────────────────────────────────────────────
# Event Construction and Serialization
# ──────────────────────────────────────────────────────────


class TestLongTermMemoryEvents:
    def test_store_event_construction(self) -> None:
        event = LongTermStoreEvent(trace_id="t1", span_id="s1", key="k", value="v", namespace=None)
        assert event.event_type == "memory.longterm.store"
        assert event.key == "k"
        assert event.value == "v"
        assert event.namespace is None

    def test_store_event_frozen(self) -> None:
        event = LongTermStoreEvent(trace_id="t1", span_id="s1", key="k", value="v", namespace=None)
        with pytest.raises(ValidationError):
            event.key = "new"

    def test_retrieve_event_construction(self) -> None:
        event = LongTermRetrieveEvent(trace_id="t1", span_id="s1", key="k", namespace="ns", found=True, value="v")
        assert event.event_type == "memory.longterm.retrieve"
        assert event.found is True
        assert event.value == "v"

    def test_retrieve_event_not_found(self) -> None:
        event = LongTermRetrieveEvent(trace_id="t1", span_id="s1", key="k", namespace=None, found=False, value=None)
        assert event.found is False
        assert event.value is None

    def test_delete_event_construction(self) -> None:
        event = LongTermDeleteEvent(trace_id="t1", span_id="s1", key="k", namespace="ns")
        assert event.event_type == "memory.longterm.delete"

    def test_list_event_construction(self) -> None:
        event = LongTermListEvent(trace_id="t1", span_id="s1", namespace=None, keys=["a", "b"])
        assert event.event_type == "memory.longterm.list"
        assert event.keys == ["a", "b"]

    def test_all_events_serializable(self) -> None:
        events = [
            LongTermStoreEvent(trace_id="t", span_id="s", key="k", value="v", namespace=None),
            LongTermRetrieveEvent(trace_id="t", span_id="s", key="k", namespace=None, found=True, value="v"),
            LongTermDeleteEvent(trace_id="t", span_id="s", key="k", namespace=None),
            LongTermListEvent(trace_id="t", span_id="s", namespace=None, keys=[]),
        ]
        for event in events:
            data = event.model_dump()
            assert "event_type" in data
            assert data["event_type"].startswith("memory.longterm.")


# ──────────────────────────────────────────────────────────
# Tool Factory
# ──────────────────────────────────────────────────────────


class TestCreateLongTermMemoryTools:
    def test_returns_four_tools(self) -> None:
        store = InMemoryLongTermStore()
        tools = create_long_term_memory_tools(store)
        assert len(tools) == 4

    def test_tools_are_function_tools(self) -> None:
        store = InMemoryLongTermStore()
        tools = create_long_term_memory_tools(store)
        for t in tools:
            assert isinstance(t, FunctionTool)
            assert isinstance(t, Tool)

    def test_tool_names(self) -> None:
        store = InMemoryLongTermStore()
        tools = create_long_term_memory_tools(store)
        names = {t.schema.name for t in tools}
        assert names == {"store_memory", "recall_memory", "delete_memory", "list_memory_keys"}

    def test_tool_schemas_have_descriptions(self) -> None:
        store = InMemoryLongTermStore()
        tools = create_long_term_memory_tools(store)
        for t in tools:
            assert t.schema.description


# ──────────────────────────────────────────────────────────
# Tool Execution
# ──────────────────────────────────────────────────────────


class TestLongTermMemoryToolExecution:
    def _get_tool(self, tools: list[FunctionTool], name: str) -> FunctionTool:
        return next(t for t in tools if t.schema.name == name)

    async def test_store_memory_tool(self) -> None:
        store = InMemoryLongTermStore()
        tools = create_long_term_memory_tools(store)
        store_tool = self._get_tool(tools, "store_memory")
        result = await store_tool.execute(key="pref", value="dark_mode")
        assert "pref" in result.content
        assert await store.retrieve("pref") == "dark_mode"

    async def test_recall_memory_tool_found(self) -> None:
        store = InMemoryLongTermStore()
        await store.store("pref", "dark_mode")
        tools = create_long_term_memory_tools(store)
        recall_tool = self._get_tool(tools, "recall_memory")
        result = await recall_tool.execute(key="pref")
        assert result.content == "dark_mode"

    async def test_recall_memory_tool_not_found(self) -> None:
        store = InMemoryLongTermStore()
        tools = create_long_term_memory_tools(store)
        recall_tool = self._get_tool(tools, "recall_memory")
        result = await recall_tool.execute(key="missing")
        assert "No value found" in result.content

    async def test_delete_memory_tool(self) -> None:
        store = InMemoryLongTermStore()
        await store.store("pref", "dark_mode")
        tools = create_long_term_memory_tools(store)
        delete_tool = self._get_tool(tools, "delete_memory")
        result = await delete_tool.execute(key="pref")
        assert "Deleted" in result.content
        assert await store.retrieve("pref") is None

    async def test_list_memory_keys_tool_with_keys(self) -> None:
        store = InMemoryLongTermStore()
        await store.store("alpha", "a")
        await store.store("beta", "b")
        tools = create_long_term_memory_tools(store)
        list_tool = self._get_tool(tools, "list_memory_keys")
        result = await list_tool.execute()
        assert "alpha" in result.content
        assert "beta" in result.content

    async def test_list_memory_keys_tool_empty(self) -> None:
        store = InMemoryLongTermStore()
        tools = create_long_term_memory_tools(store)
        list_tool = self._get_tool(tools, "list_memory_keys")
        result = await list_tool.execute()
        assert result.content == "No keys stored."

    async def test_tools_use_bound_namespace(self) -> None:
        store = InMemoryLongTermStore()
        tools = create_long_term_memory_tools(store, namespace="agent1")
        store_tool = self._get_tool(tools, "store_memory")
        await store_tool.execute(key="k", value="v")
        # Directly check store: value is in namespace, not global
        assert await store.retrieve("k", namespace="agent1") == "v"
        assert await store.retrieve("k") is None


# ──────────────────────────────────────────────────────────
# Tool Event Emission
# ──────────────────────────────────────────────────────────


class TestLongTermMemoryToolEvents:
    def _get_tool(self, tools: list[FunctionTool], name: str) -> FunctionTool:
        return next(t for t in tools if t.schema.name == name)

    def _make_registry(self, tools: list[FunctionTool], emitter: InMemoryEmitter) -> ToolRegistry:
        registry = ToolRegistry(emitter=emitter)
        for t in tools:
            registry.register(t)
        return registry

    async def test_store_emits_event(self) -> None:
        store = InMemoryLongTermStore()
        emitter = make_emitter()
        tools = create_long_term_memory_tools(store)
        registry = self._make_registry(tools, emitter)
        await registry.dispatch(ToolCall(id="1", name="store_memory", arguments={"key": "k", "value": "v"}))
        events = [e for e in emitter.events if isinstance(e, LongTermStoreEvent)]
        assert len(events) == 1
        assert events[0].key == "k"
        assert events[0].value == "v"
        assert events[0].trace_id == "test-trace"

    async def test_recall_emits_event_found(self) -> None:
        store = InMemoryLongTermStore()
        await store.store("k", "v")
        emitter = make_emitter()
        tools = create_long_term_memory_tools(store)
        registry = self._make_registry(tools, emitter)
        await registry.dispatch(ToolCall(id="1", name="recall_memory", arguments={"key": "k"}))
        events = [e for e in emitter.events if isinstance(e, LongTermRetrieveEvent)]
        assert len(events) == 1
        assert events[0].found is True
        assert events[0].value == "v"

    async def test_recall_emits_event_not_found(self) -> None:
        store = InMemoryLongTermStore()
        emitter = make_emitter()
        tools = create_long_term_memory_tools(store)
        registry = self._make_registry(tools, emitter)
        await registry.dispatch(ToolCall(id="1", name="recall_memory", arguments={"key": "missing"}))
        events = [e for e in emitter.events if isinstance(e, LongTermRetrieveEvent)]
        assert len(events) == 1
        assert events[0].found is False
        assert events[0].value is None

    async def test_delete_emits_event(self) -> None:
        store = InMemoryLongTermStore()
        emitter = make_emitter()
        tools = create_long_term_memory_tools(store)
        registry = self._make_registry(tools, emitter)
        await registry.dispatch(ToolCall(id="1", name="delete_memory", arguments={"key": "k"}))
        events = [e for e in emitter.events if isinstance(e, LongTermDeleteEvent)]
        assert len(events) == 1
        assert events[0].key == "k"

    async def test_list_emits_event(self) -> None:
        store = InMemoryLongTermStore()
        await store.store("a", "1")
        emitter = make_emitter()
        tools = create_long_term_memory_tools(store)
        registry = self._make_registry(tools, emitter)
        await registry.dispatch(ToolCall(id="1", name="list_memory_keys", arguments={}))
        events = [e for e in emitter.events if isinstance(e, LongTermListEvent)]
        assert len(events) == 1
        assert events[0].keys == ["a"]

    async def test_no_events_without_emitter(self) -> None:
        store = InMemoryLongTermStore()
        tools = create_long_term_memory_tools(store)
        registry = ToolRegistry()  # no emitter
        for t in tools:
            registry.register(t)
        await registry.dispatch(ToolCall(id="1", name="store_memory", arguments={"key": "k", "value": "v"}))

    async def test_event_namespace_propagated(self) -> None:
        store = InMemoryLongTermStore()
        emitter = make_emitter()
        tools = create_long_term_memory_tools(store, namespace="agent1")
        registry = self._make_registry(tools, emitter)
        await registry.dispatch(ToolCall(id="1", name="store_memory", arguments={"key": "k", "value": "v"}))
        events = [e for e in emitter.events if isinstance(e, LongTermStoreEvent)]
        assert events[0].namespace == "agent1"


# ──────────────────────────────────────────────────────────
# Integration: ToolRegistry Dispatch
# ──────────────────────────────────────────────────────────


class TestLongTermMemoryRegistryIntegration:
    async def test_store_recall_delete_via_registry(self) -> None:
        """Full CRUD cycle dispatched through ToolRegistry."""
        store = InMemoryLongTermStore()
        tools = create_long_term_memory_tools(store)
        registry = ToolRegistry()
        for t in tools:
            registry.register(t)

        # Store
        store_result = await registry.dispatch(
            ToolCall(id="1", name="store_memory", arguments={"key": "pref", "value": "dark"})
        )
        assert "pref" in store_result.content

        # Recall
        recall_result = await registry.dispatch(ToolCall(id="2", name="recall_memory", arguments={"key": "pref"}))
        assert recall_result.content == "dark"

        # Delete
        delete_result = await registry.dispatch(ToolCall(id="3", name="delete_memory", arguments={"key": "pref"}))
        assert "Deleted" in delete_result.content

        # Recall after delete
        gone_result = await registry.dispatch(ToolCall(id="4", name="recall_memory", arguments={"key": "pref"}))
        assert "No value found" in gone_result.content


# ──────────────────────────────────────────────────────────
# Integration: Namespace Isolation with Shared Store
# ──────────────────────────────────────────────────────────


class TestNamespaceIsolationIntegration:
    async def test_two_namespaces_isolated(self) -> None:
        """Two tool sets with different namespaces sharing the same store don't see each other's keys."""
        store = InMemoryLongTermStore()
        tools_a = create_long_term_memory_tools(store, namespace="agent_a")
        tools_b = create_long_term_memory_tools(store, namespace="agent_b")

        store_a = next(t for t in tools_a if t.schema.name == "store_memory")
        store_b = next(t for t in tools_b if t.schema.name == "store_memory")
        recall_a = next(t for t in tools_a if t.schema.name == "recall_memory")
        recall_b = next(t for t in tools_b if t.schema.name == "recall_memory")
        list_a = next(t for t in tools_a if t.schema.name == "list_memory_keys")
        list_b = next(t for t in tools_b if t.schema.name == "list_memory_keys")

        await store_a.execute(key="secret", value="agent_a_data")
        await store_b.execute(key="secret", value="agent_b_data")

        result_a = await recall_a.execute(key="secret")
        result_b = await recall_b.execute(key="secret")
        assert result_a.content == "agent_a_data"
        assert result_b.content == "agent_b_data"

        keys_a = await list_a.execute()
        keys_b = await list_b.execute()
        assert keys_a.content == "secret"
        assert keys_b.content == "secret"


# ──────────────────────────────────────────────────────────
# Integration: ReActAgent with Long-Term Memory Tools
# ──────────────────────────────────────────────────────────


class TestReActAgentLongTermMemory:
    async def test_agent_stores_and_recalls(self) -> None:
        """ReActAgent with MockLLMClient stores a value then recalls it."""
        store = InMemoryLongTermStore()
        emitter = make_emitter()
        tools = create_long_term_memory_tools(store)

        store_call = ToolCall(
            id="tc1",
            name="store_memory",
            arguments={"key": "user_name", "value": "Alice"},
        )
        recall_call = ToolCall(
            id="tc2",
            name="recall_memory",
            arguments={"key": "user_name"},
        )
        responses = [
            make_response(content="I'll store the name.", tool_calls=[store_call]),
            make_response(content="Now I'll recall it.", tool_calls=[recall_call]),
            make_response(content="The user's name is Alice."),
        ]
        client = MockLLMClient(responses)
        agent = ReActAgent(
            name="memory-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="You have long-term memory.",
            tools=tools,
        )

        result = await agent.run("What's my name?")
        assert result.output == "The user's name is Alice."
        assert await store.retrieve("user_name") == "Alice"

    async def test_cross_run_persistence(self) -> None:
        """Store persists data across two agent runs."""
        store = InMemoryLongTermStore()

        # Run 1: store data
        emitter1 = make_emitter()
        tools1 = create_long_term_memory_tools(store)
        store_call = ToolCall(
            id="tc1",
            name="store_memory",
            arguments={"key": "project", "value": "nanitics"},
        )
        client1 = MockLLMClient(
            [
                make_response(content="Storing.", tool_calls=[store_call]),
                make_response(content="Done."),
            ]
        )
        agent1 = ReActAgent(
            name="agent1",
            llm_client=client1,
            emitter=emitter1,
            system_prompt="prompt",
            tools=tools1,
        )
        await agent1.run("Store the project name.")

        # Run 2: recall data
        emitter2 = make_emitter()
        tools2 = create_long_term_memory_tools(store)
        recall_call = ToolCall(
            id="tc2",
            name="recall_memory",
            arguments={"key": "project"},
        )
        client2 = MockLLMClient(
            [
                make_response(content="Recalling.", tool_calls=[recall_call]),
                make_response(content="The project is nanitics."),
            ]
        )
        agent2 = ReActAgent(
            name="agent2",
            llm_client=client2,
            emitter=emitter2,
            system_prompt="prompt",
            tools=tools2,
        )
        result = await agent2.run("What's the project?")
        assert result.output == "The project is nanitics."
