"""Shared memory: multi-agent coordination through shared state.

Covers the InMemorySharedMemory store, SharedEntry lifecycle (active → superseded /
retracted), create_shared_memory_tools for agent integration, two agents coordinating
through shared state, SharedMemoryProvider for automatic context injection, and
SharedMemoryContributor for system prompt guidance.

Related guide: docs/guides/memory.md
"""

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics import (
    ContextContent,
    InMemorySharedMemory,
    MockLLMClient,
    ReActAgent,
    SharedEntry,
    SharedMemory,
    SharedMemoryContributor,
    SharedMemoryProvider,
    SystemPromptBuilder,
    SystemPromptContributor,
    ToolCall,
    create_shared_memory_tools,
)
from nanitics.infrastructure import (
    SharedMemoryWriteEvent,
)


async def main() -> None:
    # --- Section 1: Store and Entry Lifecycle ---
    print("--- Section 1: Store and Entry Lifecycle ---")

    store = InMemorySharedMemory()
    assert isinstance(store, SharedMemory)

    # Write entries with author attribution and scope
    id_1 = await store.write("Revenue grew 15% YoY", author="analyst", scope="findings")
    id_2 = await store.write("Customer churn decreased to 5%", author="analyst", scope="findings")
    id_3 = await store.write("Recommend expanding to EU market", author="strategist", scope="decisions")
    assert await store.count() == 3
    print(f"  Written 3 entries: count={await store.count()} ✓")

    # Read entries (default: active only, newest first)
    entries = await store.read()
    assert len(entries) == 3
    assert entries[0].content == "Recommend expanding to EU market"  # newest first
    print(f"  Read all: {len(entries)} entries, newest first ✓")

    # Filter by scope
    findings = await store.read(scope="findings")
    assert len(findings) == 2
    assert all(e.scope == "findings" for e in findings)
    print(f"  Scope filter (findings): {len(findings)} entries ✓")

    decisions = await store.read(scope="decisions")
    assert len(decisions) == 1
    assert decisions[0].author == "strategist"
    print(f"  Scope filter (decisions): {len(decisions)} entry ✓")

    # Inspect SharedEntry fields
    entry = await store.read_by_id(id_1)
    assert entry is not None
    assert isinstance(entry, SharedEntry)
    assert entry.id == id_1
    assert entry.content == "Revenue grew 15% YoY"
    assert entry.author == "analyst"
    assert entry.scope == "findings"
    assert entry.status == "active"
    assert entry.timestamp is not None
    assert entry.superseded_by is None
    assert entry.retracted_reason is None
    print(f"  SharedEntry fields: id={entry.id[:8]}..., author={entry.author}, status={entry.status} ✓")

    # Supersede an entry: original marked "superseded", new entry created
    new_id = await store.supersede(id_1, "Revenue grew 18% YoY (revised)", author="analyst")
    original = await store.read_by_id(id_1)
    assert original is not None
    assert original.status == "superseded"
    assert original.superseded_by == new_id
    revised = await store.read_by_id(new_id)
    assert revised is not None
    assert revised.content == "Revenue grew 18% YoY (revised)"
    assert revised.status == "active"
    print(f"  Superseded: original status={original.status}, superseded_by={new_id[:8]}... ✓")

    # Default reads exclude superseded entries
    active_findings = await store.read(scope="findings")
    assert len(active_findings) == 2  # revised + churn (original hidden)
    assert all(e.status == "active" for e in active_findings)
    print(f"  Active findings after supersede: {len(active_findings)} ✓")

    # Retract an entry: marks "retracted" with reason, hidden from default reads
    await store.retract(id_2, "Data was from wrong quarter", author="analyst")
    retracted = await store.read_by_id(id_2)
    assert retracted is not None
    assert retracted.status == "retracted"
    assert retracted.retracted_reason == "Data was from wrong quarter"
    print(f"  Retracted: status={retracted.status}, reason={retracted.retracted_reason!r} ✓")

    # Default reads now exclude both superseded and retracted
    active_findings = await store.read(scope="findings")
    assert len(active_findings) == 1  # only the revised entry remains
    print(f"  Active findings after retract: {len(active_findings)} ✓")

    # include_inactive=True returns all entries
    all_entries = await store.read(include_inactive=True)
    assert len(all_entries) == 4  # 3 original + 1 supersede replacement
    print(f"  All entries (include_inactive): {len(all_entries)} ✓")

    # Count with inactive
    active_count = await store.count()
    total_count = await store.count(include_inactive=True)
    assert active_count == 2  # revised finding + decision
    assert total_count == 4
    print(f"  Count: active={active_count}, total={total_count} ✓")

    # Author enforcement: only original author can supersede/retract
    try:
        await store.supersede(new_id, "Hijacked content", author="imposter")
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  Author enforcement (supersede): ValueError raised ✓")

    try:
        await store.retract(id_3, "Hijacked retraction", author="imposter")
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  Author enforcement (retract): ValueError raised ✓")

    print("✓ Section 1 passed")

    # --- Section 2: Shared Memory Tools ---
    print("\n--- Section 2: Shared Memory Tools ---")

    store = InMemorySharedMemory()
    emitter = make_emitter("shared-s2")

    # create_shared_memory_tools returns 4 tools with baked-in agent name
    tools = create_shared_memory_tools(store, "analyst")
    assert len(tools) == 4
    tool_names = {t.schema.name for t in tools}
    assert tool_names == {"write_to_shared", "read_shared", "supersede_shared", "retract_shared"}
    print(f"  Tools created: {sorted(tool_names)} ✓")

    # Agent uses write_to_shared and read_shared
    client = MockLLMClient(
        [
            # Turn 1: Write a finding
            make_response(
                content="Let me record my finding.",
                tool_calls=[
                    ToolCall(
                        id="tc-1",
                        name="write_to_shared",
                        arguments={"content": "Revenue grew 15% YoY", "scope": "findings"},
                    )
                ],
                stop_reason="tool_use",
            ),
            # Turn 2: Read back the findings
            make_response(
                content="Let me check what's on the board.",
                tool_calls=[
                    ToolCall(
                        id="tc-2",
                        name="read_shared",
                        arguments={"scope": "findings"},
                    )
                ],
                stop_reason="tool_use",
            ),
            # Turn 3: Final response
            make_response("Revenue growth of 15% YoY has been recorded and verified."),
        ]
    )

    agent = ReActAgent(
        name="analyst",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a financial analyst. Record findings to shared memory.",
        tools=tools,
    )

    result = await agent.run("Analyze the revenue data")
    assert result.termination_reason == "complete"
    print(f"  Agent completed: {result.termination_reason} ✓")

    # Verify store has the entry with correct author (baked into tool closure)
    entries = await store.read(scope="findings")
    assert len(entries) == 1
    assert entries[0].content == "Revenue grew 15% YoY"
    assert entries[0].author == "analyst"
    print(f"  Store entry: author={entries[0].author!r}, content={entries[0].content!r} ✓")

    # Verify SharedMemoryWriteEvent was emitted
    write_events = [e for e in emitter.events if isinstance(e, SharedMemoryWriteEvent)]
    assert len(write_events) == 1
    assert write_events[0].author == "analyst"
    assert write_events[0].content == "Revenue grew 15% YoY"
    assert write_events[0].scope == "findings"
    print(f"  SharedMemoryWriteEvent: author={write_events[0].author!r}, scope={write_events[0].scope!r} ✓")

    print("✓ Section 2 passed")

    # --- Section 3: Two Agents Sharing State ---
    print("\n--- Section 3: Two Agents Sharing State ---")

    # One shared store, two agents with separate clients and emitters
    store = InMemorySharedMemory()

    # Agent 1: Researcher — writes findings
    researcher_client = MockLLMClient(
        [
            make_response(
                content="Recording market size finding.",
                tool_calls=[
                    ToolCall(
                        id="tc-r1",
                        name="write_to_shared",
                        arguments={"content": "Market size is $50B", "scope": "findings"},
                    )
                ],
                stop_reason="tool_use",
            ),
            make_response(
                content="Recording competitor finding.",
                tool_calls=[
                    ToolCall(
                        id="tc-r2",
                        name="write_to_shared",
                        arguments={"content": "Top competitor has 30% share", "scope": "findings"},
                    )
                ],
                stop_reason="tool_use",
            ),
            make_response("Research complete."),
        ]
    )
    researcher_emitter = make_emitter("researcher-trace")
    researcher_tools = create_shared_memory_tools(store, "researcher")

    researcher = ReActAgent(
        name="researcher",
        llm_client=researcher_client,
        emitter=researcher_emitter,
        system_prompt="You are a market researcher. Write findings to shared memory.",
        tools=researcher_tools,
    )

    # Agent 2: Analyst — reads findings, writes analysis
    analyst_client = MockLLMClient(
        [
            make_response(
                content="Let me review the research findings.",
                tool_calls=[
                    ToolCall(
                        id="tc-a1",
                        name="read_shared",
                        arguments={"scope": "findings"},
                    )
                ],
                stop_reason="tool_use",
            ),
            make_response(
                content="Based on the findings, I'll write my analysis.",
                tool_calls=[
                    ToolCall(
                        id="tc-a2",
                        name="write_to_shared",
                        arguments={
                            "content": "Addressable market is $35B after excluding segments",
                            "scope": "analysis",
                        },
                    )
                ],
                stop_reason="tool_use",
            ),
            make_response("Analysis complete."),
        ]
    )
    analyst_emitter = make_emitter("analyst-trace")
    analyst_tools = create_shared_memory_tools(store, "analyst")

    analyst = ReActAgent(
        name="analyst",
        llm_client=analyst_client,
        emitter=analyst_emitter,
        system_prompt="You are a market analyst. Read findings and write analysis.",
        tools=analyst_tools,
    )

    # Run sequentially: researcher first, analyst second
    researcher_result = await researcher.run("Research the market opportunity")
    assert researcher_result.termination_reason == "complete"
    print(f"  Researcher completed: {researcher_result.termination_reason} ✓")

    # After researcher: 2 entries, both authored "researcher"
    assert await store.count() == 2
    researcher_entries = await store.read(author="researcher")
    assert len(researcher_entries) == 2
    assert all(e.author == "researcher" for e in researcher_entries)
    print(f"  Store after researcher: {await store.count()} entries, all by 'researcher' ✓")

    analyst_result = await analyst.run("Analyze the research findings")
    assert analyst_result.termination_reason == "complete"
    print(f"  Analyst completed: {analyst_result.termination_reason} ✓")

    # After analyst: 3 entries total
    assert await store.count() == 3

    # Scope isolation: each scope has the right entries
    findings = await store.read(scope="findings")
    assert len(findings) == 2
    assert all(e.author == "researcher" for e in findings)
    print(f"  Scope 'findings': {len(findings)} entries, all by 'researcher' ✓")

    analysis = await store.read(scope="analysis")
    assert len(analysis) == 1
    assert analysis[0].author == "analyst"
    assert analysis[0].content == "Addressable market is $35B after excluding segments"
    print(f"  Scope 'analysis': {len(analysis)} entry, by 'analyst' ✓")

    # Both agents completed successfully
    assert researcher_result.termination_reason == "complete"
    assert analyst_result.termination_reason == "complete"
    print(f"  Total entries: {await store.count()} (2 researcher + 1 analyst) ✓")

    print("✓ Section 3 passed")

    # --- Section 4: Context Provider and Contributor ---
    print("\n--- Section 4: Context Provider and Contributor ---")

    # Pre-populate the store with entries from two authors
    store = InMemorySharedMemory()
    await store.write("Revenue grew 15% YoY", author="analyst", scope="findings")
    await store.write("Market size is $50B", author="researcher", scope="findings")
    await store.write("Recommend EU expansion", author="strategist", scope="decisions")

    # SharedMemoryProvider injects board state into agent context
    emitter = make_emitter("shared-s4")
    provider = SharedMemoryProvider(store, emitter=emitter, scopes=["findings"], max_entries=50)

    # Provider returns ContextContent with formatted board state
    from nanitics import Message

    messages = [Message(role="user", content="Summarize the findings")]
    context = await provider.provide(messages)

    assert context is not None
    assert isinstance(context, ContextContent)
    assert "[Shared Memory Board]" in context.content
    assert "analyst" in context.content
    assert "researcher" in context.content
    assert "Revenue grew 15% YoY" in context.content
    assert "Market size is $50B" in context.content
    # Scoped provider excludes entries from other scopes
    assert "Recommend EU expansion" not in context.content
    assert context.provider_name == "shared_memory"
    print("  Provider returned ContextContent ✓")
    print(f"  provider_name={context.provider_name!r} ✓")
    print("  Content preview:")
    for line in context.content.split("\n")[:6]:
        print(f"    {line}")

    # Provider returns None for empty board
    empty_store = InMemorySharedMemory()
    empty_provider = SharedMemoryProvider(empty_store)
    empty_result = await empty_provider.provide(messages)
    assert empty_result is None
    print("  Provider with empty store: None ✓")

    # SharedMemoryContributor teaches the agent the shared memory protocol
    contributor = SharedMemoryContributor()
    assert isinstance(contributor, SystemPromptContributor)
    key, instructions = contributor.system_prompt_section()
    assert key == "shared_memory"
    assert "supersede" in instructions.lower()
    assert "retract" in instructions.lower()
    assert "attribution" in instructions.lower() or "attributed" in instructions.lower()
    print(f"  Contributor section: {key!r} ✓")
    print(f"  Instructions: {instructions[:80]}...")

    # SystemPromptBuilder integration
    builder = SystemPromptBuilder()
    builder.add_section("role", "You are a collaborative analyst.")
    section = contributor.system_prompt_section()
    builder.add_section(section[0], section[1])
    prompt = builder.build()
    assert "collaborative analyst" in prompt
    assert "supersede" in prompt.lower()
    print(f"  SystemPromptBuilder with contributor: prompt built ({len(prompt)} chars) ✓")

    print("✓ Section 4 passed")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
