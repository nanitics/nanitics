"""Blackboard: shared-memory agent coordination.

Demonstrates ``Blackboard`` — agents coordinate through a shared memory space
instead of communicating directly. Covers scheduled control, convergence via
``NoNewContributions``, prioritized ordering, parallel execution with
``OpportunisticControl``, and event trace inspection.

Related guide: docs/guides/multi-agent-coordination.md
"""

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics import (
    Blackboard,
    BlackboardResult,
    BlackboardRoundEntry,
    InMemorySharedMemory,
    MaxRoundsTermination,
    MockLLMClient,
    NoNewContributions,
    OpportunisticControl,
    PrioritizedControl,
    ReActAgent,
    ScheduledControl,
    ToolCall,
)
from nanitics.infrastructure import (
    BlackboardCompleteEvent,
    BlackboardRoundEvent,
    BlackboardStartEvent,
)


async def main() -> None:
    # --- Section 1: Basic Blackboard (Scheduled Control) ---
    # Two agents write to shared memory in a single round.
    # ScheduledControl runs them sequentially. MaxRoundsTermination(1) stops after one round.

    print("--- Section 1: Basic Blackboard (Scheduled Control) ---")

    emitter = make_emitter("bb-s1")
    shared = InMemorySharedMemory()

    analyst = ReActAgent(
        name="analyst",
        llm_client=MockLLMClient(
            [
                make_response(
                    "Writing my analysis.",
                    tool_calls=[
                        ToolCall(
                            id="tc-1",
                            name="write_to_shared",
                            arguments={"content": "Revenue grew 15% in Q4.", "scope": "analysis"},
                        )
                    ],
                    stop_reason="tool_use",
                ),
                make_response("Analysis complete."),
            ]
        ),
        emitter=emitter,
        system_prompt="You are a financial analyst.",
        tools=[],  # Blackboard injects shared memory tools
    )

    reviewer = ReActAgent(
        name="reviewer",
        llm_client=MockLLMClient(
            [
                make_response(
                    "Adding my review.",
                    tool_calls=[
                        ToolCall(
                            id="tc-2",
                            name="write_to_shared",
                            arguments={
                                "content": "Analysis is sound. Recommend deeper margin breakdown.",
                                "scope": "review",
                            },
                        )
                    ],
                    stop_reason="tool_use",
                ),
                make_response("Review complete."),
            ]
        ),
        emitter=emitter,
        system_prompt="You review financial analyses.",
        tools=[],
    )

    blackboard = Blackboard(
        shared_memory=shared,
        agents=[analyst, reviewer],
        emitter=emitter,
        control=ScheduledControl(),
        termination=MaxRoundsTermination(max_rounds=1),
        max_rounds=5,
    )

    result: BlackboardResult = await blackboard.run("Analyze Q4 financial performance")

    assert result.rounds_completed == 1
    assert result.termination_reason == "MaxRoundsTermination"
    assert result.agent_contributions["analyst"] == 1
    assert result.agent_contributions["reviewer"] == 1

    # Shared memory contains both entries
    entries = await shared.read()
    assert len(entries) == 2
    authors = {e.author for e in entries}
    assert authors == {"analyst", "reviewer"}

    print(f"  Rounds: {result.rounds_completed}")
    print(f"  Termination: {result.termination_reason}")
    print(f"  Contributions: {dict(result.agent_contributions)}")
    print(f"  Entries: {len(entries)} (authors: {sorted(authors)})")
    print("✓ Two agents wrote to shared memory in one round")

    # --- Section 2: Convergence (NoNewContributions) ---
    # Agent writes in round 1, does nothing in round 2.
    # NoNewContributions fires when a round produces zero contributions.

    print("\n--- Section 2: Convergence (NoNewContributions) ---")

    emitter = make_emitter("bb-s2")
    shared = InMemorySharedMemory()

    # Round 1: agent writes one entry. Round 2: agent responds without writing.
    writer = ReActAgent(
        name="writer",
        llm_client=MockLLMClient(
            [
                # Round 1: write to shared memory
                make_response(
                    "Adding my finding.",
                    tool_calls=[
                        ToolCall(
                            id="tc-w1",
                            name="write_to_shared",
                            arguments={"content": "Key insight: market share increased.", "scope": "findings"},
                        )
                    ],
                    stop_reason="tool_use",
                ),
                make_response("Finding recorded."),
                # Round 2: nothing to add
                make_response("Nothing more to contribute."),
            ]
        ),
        emitter=emitter,
        system_prompt="You contribute findings to the board.",
        tools=[],
    )

    blackboard = Blackboard(
        shared_memory=shared,
        agents=[writer],
        emitter=emitter,
        termination=NoNewContributions(),
        max_rounds=5,
    )

    result = await blackboard.run("Identify key market trends")

    assert result.rounds_completed == 2
    assert result.termination_reason == "NoNewContributions"
    assert result.agent_contributions["writer"] == 1

    print(f"  Rounds: {result.rounds_completed} (wrote in round 1, silent in round 2)")
    print(f"  Termination: {result.termination_reason}")
    print(f"  Contributions: {result.agent_contributions['writer']}")
    print("✓ Blackboard stopped when no new contributions were made")

    # --- Section 3: Prioritized Control ---
    # Three agents with different priorities. PrioritizedControl runs them
    # highest-priority first. All contribute in a single round.

    print("\n--- Section 3: Prioritized Control ---")

    emitter = make_emitter("bb-s3")
    shared = InMemorySharedMemory()

    def make_writing_agent(name: str, content: str, scope: str) -> ReActAgent:
        return ReActAgent(
            name=name,
            llm_client=MockLLMClient(
                [
                    make_response(
                        f"Writing {scope}.",
                        tool_calls=[
                            ToolCall(
                                id=f"tc-{name}",
                                name="write_to_shared",
                                arguments={"content": content, "scope": scope},
                            )
                        ],
                        stop_reason="tool_use",
                    ),
                    make_response(f"{scope.capitalize()} complete."),
                ]
            ),
            emitter=emitter,
            system_prompt=f"You contribute {scope}.",
            tools=[],
        )

    lead = make_writing_agent("lead", "Strategic direction: expand into Asia-Pacific.", "strategy")
    analyst = make_writing_agent("analyst", "Market data supports expansion.", "analysis")
    junior = make_writing_agent("junior", "Compiled reference materials.", "references")

    blackboard = Blackboard(
        shared_memory=shared,
        agents=[junior, analyst, lead],  # passed in wrong order — priority fixes it
        emitter=emitter,
        control=PrioritizedControl(priorities={"lead": 10, "analyst": 5, "junior": 1}),
        termination=MaxRoundsTermination(max_rounds=1),
    )

    result = await blackboard.run("Develop market expansion strategy")

    assert result.agent_contributions["lead"] == 1
    assert result.agent_contributions["analyst"] == 1
    assert result.agent_contributions["junior"] == 1

    # Verify execution order via round event
    round_events = [e for e in emitter.events if isinstance(e, BlackboardRoundEvent)]
    assert len(round_events) == 1
    assert round_events[0].agents_activated == ["lead", "analyst", "junior"]

    print(f"  Execution order: {round_events[0].agents_activated}")
    print(f"  Contributions: {dict(result.agent_contributions)}")
    print("✓ PrioritizedControl ran agents in priority order (lead=10, analyst=5, junior=1)")

    # --- Section 4: Parallel Execution (OpportunisticControl) ---
    # Two agents run concurrently in a single round.

    print("\n--- Section 4: Parallel Execution (OpportunisticControl) ---")

    emitter = make_emitter("bb-s4")
    shared = InMemorySharedMemory()

    researcher_a = ReActAgent(
        name="researcher-a",
        llm_client=MockLLMClient(
            [
                make_response(
                    "Writing research finding A.",
                    tool_calls=[
                        ToolCall(
                            id="tc-a",
                            name="write_to_shared",
                            arguments={"content": "Finding A: patent landscape is favorable.", "scope": "research"},
                        )
                    ],
                    stop_reason="tool_use",
                ),
                make_response("Done."),
            ]
        ),
        emitter=emitter,
        system_prompt="You research patents.",
        tools=[],
    )

    researcher_b = ReActAgent(
        name="researcher-b",
        llm_client=MockLLMClient(
            [
                make_response(
                    "Writing research finding B.",
                    tool_calls=[
                        ToolCall(
                            id="tc-b",
                            name="write_to_shared",
                            arguments={"content": "Finding B: competitor filed 3 new patents.", "scope": "research"},
                        )
                    ],
                    stop_reason="tool_use",
                ),
                make_response("Done."),
            ]
        ),
        emitter=emitter,
        system_prompt="You research competitors.",
        tools=[],
    )

    blackboard = Blackboard(
        shared_memory=shared,
        agents=[researcher_a, researcher_b],
        emitter=emitter,
        control=OpportunisticControl(),
        termination=MaxRoundsTermination(max_rounds=1),
    )

    result = await blackboard.run("Research patent landscape")

    assert result.rounds_completed == 1
    assert result.agent_contributions["researcher-a"] == 1
    assert result.agent_contributions["researcher-b"] == 1

    entries = await shared.read()
    assert len(entries) == 2
    entry_contents = {e.content for e in entries}
    assert "Finding A: patent landscape is favorable." in entry_contents
    assert "Finding B: competitor filed 3 new patents." in entry_contents

    print(f"  Rounds: {result.rounds_completed}")
    print(f"  Contributions: {dict(result.agent_contributions)}")
    print(f"  Entries: {len(entries)} (both agents wrote concurrently)")
    print("✓ OpportunisticControl ran both agents in parallel")

    # --- Section 5: Event Verification ---
    # Inspect the full event lifecycle: start, round (with round_entries), complete.

    print("\n--- Section 5: Event Verification ---")

    emitter = make_emitter("bb-s5")
    shared = InMemorySharedMemory()

    agent = ReActAgent(
        name="contributor",
        llm_client=MockLLMClient(
            [
                make_response(
                    "Writing to the board.",
                    tool_calls=[
                        ToolCall(
                            id="tc-ev",
                            name="write_to_shared",
                            arguments={"content": "Event test contribution.", "scope": "test"},
                        )
                    ],
                    stop_reason="tool_use",
                ),
                make_response("Done."),
            ]
        ),
        emitter=emitter,
        system_prompt="You contribute test data.",
        tools=[],
    )

    blackboard = Blackboard(
        shared_memory=shared,
        agents=[agent],
        emitter=emitter,
        control=ScheduledControl(),
        termination=MaxRoundsTermination(max_rounds=1),
        max_rounds=1,
    )

    result = await blackboard.run("Write test data")

    # BlackboardStartEvent
    start_events = [e for e in emitter.events if isinstance(e, BlackboardStartEvent)]
    assert len(start_events) == 1
    start = start_events[0]
    assert start.task == "Write test data"
    assert start.agent_names == ["contributor"]
    assert start.control_strategy == "ScheduledControl"
    assert start.max_rounds == 1
    print(f"  BlackboardStartEvent: task={start.task!r}, agents={start.agent_names}, strategy={start.control_strategy}")

    # BlackboardRoundEvent with round_entries
    round_events = [e for e in emitter.events if isinstance(e, BlackboardRoundEvent)]
    assert len(round_events) == 1
    rnd = round_events[0]
    assert rnd.round_number == 1
    assert rnd.agents_activated == ["contributor"]
    assert rnd.contributions == 1
    assert len(rnd.round_entries) == 1

    entry: BlackboardRoundEntry = rnd.round_entries[0]
    assert entry.operation == "write"
    assert entry.author == "contributor"
    assert entry.content == "Event test contribution."
    assert entry.scope == "test"
    assert entry.entry_id  # non-empty UUID
    print(f"  BlackboardRoundEvent: round={rnd.round_number}, contributions={rnd.contributions}")
    print(f"    RoundEntry: op={entry.operation}, author={entry.author}, scope={entry.scope}")

    # BlackboardCompleteEvent
    complete_events = [e for e in emitter.events if isinstance(e, BlackboardCompleteEvent)]
    assert len(complete_events) == 1
    complete = complete_events[0]
    assert complete.rounds_completed == 1
    assert complete.termination_reason == "MaxRoundsTermination"
    assert complete.total_contributions == 1
    assert complete.agent_contributions == {"contributor": 1}
    print(f"  BlackboardCompleteEvent: rounds={complete.rounds_completed}, reason={complete.termination_reason}")
    print(f"    total_contributions={complete.total_contributions}, agent_contributions={complete.agent_contributions}")

    print("✓ Full event lifecycle: start → round (with entries) → complete")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
