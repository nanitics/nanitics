"""Long-term memory: a key-value store that persists across agent runs.

Covers InMemoryLongTermStore direct API (store, retrieve, overwrite, delete, list_keys),
namespace isolation (direct API and via tool factory), agent integration with
create_long_term_memory_tools where memory persists across separate agent.run() calls,
and event verification for all four event types (store, retrieve, list, delete).

Related guide: docs/guides/memory.md
"""

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics.infrastructure import (
    LongTermDeleteEvent,
    LongTermListEvent,
    LongTermRetrieveEvent,
    LongTermStoreEvent,
    MockLLMClient,
)
from nanitics.memory import (
    InMemoryLongTermStore,
    create_long_term_memory_tools,
)
from nanitics.strategies import ReActAgent
from nanitics.tracing import (
    InMemoryEmitter,
    ToolCall,
)


async def main() -> None:
    # --- Section 1: InMemoryLongTermStore — Direct API ---
    print("--- Section 1: InMemoryLongTermStore — Direct API ---")

    store = InMemoryLongTermStore()

    # Basic store and retrieve
    await store.store("greeting", "Hello, world!")
    value = await store.retrieve("greeting")
    assert value == "Hello, world!"
    print(f"  store + retrieve: {value!r} ✓")

    # Non-existent key returns None
    value = await store.retrieve("nonexistent")
    assert value is None
    print("  retrieve non-existent key: None ✓")

    # list_keys returns all stored keys
    await store.store("language", "Python")
    keys = await store.list_keys()
    assert "greeting" in keys
    assert "language" in keys
    print(f"  list_keys: {keys} ✓")

    # Overwrite (upsert semantics)
    await store.store("greeting", "Hi there!")
    value = await store.retrieve("greeting")
    assert value == "Hi there!", "Overwrite should replace existing value"
    print(f"  overwrite: {value!r} ✓")

    # Delete
    await store.delete("greeting")
    value = await store.retrieve("greeting")
    assert value is None, "Deleted key should return None"
    keys = await store.list_keys()
    assert "greeting" not in keys
    print("  delete: key removed ✓")

    print("✓ Section 1 passed")

    # --- Section 2: Namespace Isolation ---
    print("\n--- Section 2: Namespace Isolation ---")

    store = InMemoryLongTermStore()

    # Same key in different namespaces holds different values
    await store.store("status", "active", namespace="agent_a")
    await store.store("status", "idle", namespace="agent_b")

    value_a = await store.retrieve("status", namespace="agent_a")
    value_b = await store.retrieve("status", namespace="agent_b")
    assert value_a == "active"
    assert value_b == "idle"
    print(f"  agent_a status: {value_a!r}, agent_b status: {value_b!r} ✓")

    # list_keys scoped to namespace
    await store.store("extra", "data", namespace="agent_a")
    keys_a = await store.list_keys(namespace="agent_a")
    keys_b = await store.list_keys(namespace="agent_b")
    assert "extra" in keys_a
    assert "extra" not in keys_b, "Keys should not leak across namespaces"
    print(f"  agent_a keys: {keys_a}, agent_b keys: {keys_b} ✓")

    # Namespace via tool factory: each agent gets an isolated view
    store = InMemoryLongTermStore()
    researcher_tools = create_long_term_memory_tools(store, namespace="researcher")
    writer_tools = create_long_term_memory_tools(store, namespace="writer")

    # Helper to find and execute a tool by name
    def get_tool(tool_list: list, name: str):
        return next(t for t in tool_list if t.schema.name == name)

    await get_tool(researcher_tools, "store_memory").execute(key="topic", value="AI safety")
    await get_tool(writer_tools, "store_memory").execute(key="topic", value="Climate change")

    researcher_result = await get_tool(researcher_tools, "recall_memory").execute(key="topic")
    writer_result = await get_tool(writer_tools, "recall_memory").execute(key="topic")
    assert "AI safety" in researcher_result.content
    assert "Climate change" in writer_result.content
    print("  Researcher topic: AI safety ✓")
    print("  Writer topic: Climate change ✓")

    # Direct API confirms both live in same store, different namespaces
    assert await store.retrieve("topic", namespace="researcher") == "AI safety"
    assert await store.retrieve("topic", namespace="writer") == "Climate change"
    print("  Same store, isolated namespaces ✓")

    print("✓ Section 2 passed")

    # --- Section 3: Agent Integration — Memory Across Runs ---
    print("\n--- Section 3: Agent Integration — Memory Across Runs ---")

    # The key insight: the same store outlives any single agent.run() call.
    # Run 1 stores preferences, run 2 recalls and uses them.
    store = InMemoryLongTermStore()
    memory_tools = create_long_term_memory_tools(store)
    emitter = make_emitter()

    client = MockLLMClient(
        [
            # --- Run 1: "Remember my preferences" ---
            # Step 1: Store user name
            make_response(
                content="I'll save your name.",
                tool_calls=[ToolCall(id="tc-1", name="store_memory", arguments={"key": "user_name", "value": "Alice"})],
                stop_reason="tool_use",
            ),
            # Step 2: Store preferred format
            make_response(
                content="And your format preference.",
                tool_calls=[
                    ToolCall(
                        id="tc-2",
                        name="store_memory",
                        arguments={"key": "preferred_format", "value": "concise bullet points"},
                    )
                ],
                stop_reason="tool_use",
            ),
            # Step 3: Final response
            make_response(
                content="I've saved your preferences. "
                "I'll remember your name and format preference for future conversations.",
            ),
            # --- Run 2: "What are the benefits of Python?" ---
            # Step 1: Recall user name
            make_response(
                content="Let me check who I'm talking to.",
                tool_calls=[ToolCall(id="tc-3", name="recall_memory", arguments={"key": "user_name"})],
                stop_reason="tool_use",
            ),
            # Step 2: Recall preferred format
            make_response(
                content="And check your format preference.",
                tool_calls=[ToolCall(id="tc-4", name="recall_memory", arguments={"key": "preferred_format"})],
                stop_reason="tool_use",
            ),
            # Step 3: Final response using recalled preferences
            make_response(
                content="Here you go, Alice:\n- Easy to read and write\n"
                "- Large ecosystem of libraries\n- Strong community support",
            ),
        ]
    )

    agent = ReActAgent(
        name="assistant",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a helpful assistant with long-term memory.",
        tools=memory_tools,
    )

    # Run 1 — store preferences
    result1 = await agent.run("My name is Alice and I prefer concise bullet-point answers.")
    assert result1.total_steps == 3
    print(f"  Run 1 steps: {result1.total_steps} ✓")
    print(f"  Run 1 output: {result1.output}")

    # Verify store contains expected keys after run 1
    keys = await store.list_keys()
    assert "user_name" in keys
    assert "preferred_format" in keys
    print(f"  Store keys after run 1: {keys} ✓")

    # Run 2 — recall preferences (same agent, same store)
    result2 = await agent.run("What are the benefits of Python?")
    assert result2.total_steps == 3
    print(f"  Run 2 steps: {result2.total_steps} ✓")
    print(f"  Run 2 output: {result2.output}")

    # Verify recall worked — output references the stored name and uses bullet points
    assert result2.output is not None
    assert "Alice" in result2.output, "Output should contain recalled user name"
    assert "- " in result2.output, "Output should contain bullet points"
    print("  Recalled name and format in output ✓")

    print("✓ Section 3 passed")

    # --- Section 4: Observability — Memory Events ---
    print("\n--- Section 4: Observability — Memory Events ---")

    assert isinstance(emitter, InMemoryEmitter)
    events = emitter.events

    # Store events from run 1
    store_events = [e for e in events if isinstance(e, LongTermStoreEvent)]
    assert len(store_events) >= 2, f"Expected at least 2 store events, got {len(store_events)}"
    print(f"  LongTermStoreEvent count: {len(store_events)} ✓")

    # Verify store event data
    store_keys = {e.key for e in store_events}
    assert "user_name" in store_keys
    assert "preferred_format" in store_keys
    print(f"  Store event keys: {store_keys} ✓")

    # Retrieve events from run 2
    retrieve_events = [e for e in events if isinstance(e, LongTermRetrieveEvent)]
    assert len(retrieve_events) >= 2, f"Expected at least 2 retrieve events, got {len(retrieve_events)}"
    print(f"  LongTermRetrieveEvent count: {len(retrieve_events)} ✓")

    # Verify retrieve events carried found=True
    for e in retrieve_events:
        assert e.found is True, f"Expected found=True for key {e.key!r}"
    print("  All retrieve events: found=True ✓")

    # Delete and list events are also emitted by the tools.
    # Run a short sequence that exercises list + delete to verify.
    store2 = InMemoryLongTermStore()
    emitter2 = make_emitter()
    client2 = MockLLMClient(
        [
            make_response(
                content="Storing a value.",
                tool_calls=[ToolCall(id="tc-d1", name="store_memory", arguments={"key": "temp", "value": "data"})],
                stop_reason="tool_use",
            ),
            make_response(
                content="Listing keys.",
                tool_calls=[ToolCall(id="tc-d2", name="list_memory_keys", arguments={})],
                stop_reason="tool_use",
            ),
            make_response(
                content="Deleting the key.",
                tool_calls=[ToolCall(id="tc-d3", name="delete_memory", arguments={"key": "temp"})],
                stop_reason="tool_use",
            ),
            make_response(content="Done."),
        ]
    )
    agent2 = ReActAgent(
        name="cleanup",
        llm_client=client2,
        emitter=emitter2,
        system_prompt="You manage stored data.",
        tools=create_long_term_memory_tools(store2),
    )
    await agent2.run("Store temp data, list keys, then delete it.")

    assert isinstance(emitter2, InMemoryEmitter)
    list_events = [e for e in emitter2.events if isinstance(e, LongTermListEvent)]
    assert len(list_events) == 1
    assert "temp" in list_events[0].keys
    print(f"  LongTermListEvent: keys={list_events[0].keys} ✓")

    delete_events = [e for e in emitter2.events if isinstance(e, LongTermDeleteEvent)]
    assert len(delete_events) == 1
    assert delete_events[0].key == "temp"
    print(f"  LongTermDeleteEvent: key={delete_events[0].key!r} ✓")

    print("✓ Section 4 passed")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
