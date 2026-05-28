"""Behavioral continuity: repeated ``AgentTool`` invocations in one outer run.

Demonstrates ``AgentTool(thread_key=...)``. A coordinator wraps a single
specialist via one ``AgentTool`` carrying a ``thread_key``. In one outer
``coordinator.run`` the coordinator dispatches the specialist twice; the
specialist's second turn sees its first turn as its own prior assistant
message.

The wrapping ``AgentTool`` forwards the thread key on every ``execute``
call. The specialist must be configured with a ``ThreadStore`` for the
prefix to be persisted.

Related guide: docs/guides/multi-agent-foundations.md#behavioral-continuity-in-multi-agent-patterns
"""

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics.composition import AgentTool, InMemoryThreadStore
from nanitics.infrastructure import (
    AgentStartEvent,
    DelegationEvent,
    MockLLMClient,
)
from nanitics.strategies import ReActAgent
from nanitics.tracing import ToolCall


async def main() -> None:
    # --- Section 1: Specialist with a thread store ---
    print("--- Section 1: Specialist with a thread store ---")

    emitter = make_emitter("threads-repeated-agent-tool")
    store = InMemoryThreadStore()

    specialist_client = MockLLMClient(
        [
            # Specialist turn 1 — answers the coordinator's first question.
            make_response(
                "Estimate v1: based on the schema, the migration touches three tables and roughly 120k rows."
            ),
            # Specialist turn 2 — refines using the prior turn as context.
            make_response(
                "Estimate v2: extending the earlier 120k-row estimate, the "
                "indexed columns add about 8 minutes of downtime under the "
                "current write load."
            ),
        ]
    )

    specialist = ReActAgent(
        name="db-specialist",
        llm_client=specialist_client,
        emitter=emitter,
        system_prompt=(
            "You estimate database migrations. Treat prior assistant turns as your own work and build on them directly."
        ),
        tools=[],
        thread_store=store,
    )

    specialist_tool = AgentTool(
        agent=specialist,
        emitter=emitter,
        description="Estimate database migration cost and impact.",
        caller_name="coordinator",
        thread_key="db-thread",
    )

    print("  Specialist wrapped as AgentTool with thread_key='db-thread'.")
    print("  Single specialist instance + single ThreadStore — both calls share state.")

    print("✓ Section 1 passed")

    # --- Section 2: Coordinator dispatches the specialist twice ---
    print("\n--- Section 2: Coordinator dispatches the specialist twice ---")

    coordinator_client = MockLLMClient(
        [
            # Step 1: delegate the initial estimate.
            make_response(
                content="Let me ask the specialist for an initial estimate.",
                tool_calls=[
                    ToolCall(
                        id="tc-1",
                        name="db-specialist",
                        arguments={"task": "Estimate the migration scope."},
                    )
                ],
                stop_reason="tool_use",
            ),
            # Step 2: delegate the refinement using the same tool.
            make_response(
                content="Now ask for the downtime estimate, given that scope.",
                tool_calls=[
                    ToolCall(
                        id="tc-2",
                        name="db-specialist",
                        arguments={"task": "Estimate downtime given that scope."},
                    )
                ],
                stop_reason="tool_use",
            ),
            # Step 3: synthesize the final answer.
            make_response(
                "Final: the migration touches three tables (120k rows) and will incur roughly 8 minutes of downtime."
            ),
        ]
    )

    coordinator = ReActAgent(
        name="coordinator",
        llm_client=coordinator_client,
        emitter=emitter,
        system_prompt="You coordinate database migrations by delegating to specialists.",
        tools=[specialist_tool],
    )

    result = await coordinator.run("Plan the schema migration.")

    assert result.total_steps == 3
    assert "120k" in (result.output or "")
    assert "8 minutes" in (result.output or "")
    print(f"  Coordinator output: {result.output!r}")
    print(f"  Coordinator steps: {result.total_steps}")

    print("✓ Section 2 passed")

    # --- Section 3: DelegationEvents and replay assertions ---
    print("\n--- Section 3: DelegationEvents and replay assertions ---")

    delegations = [e for e in emitter.events if isinstance(e, DelegationEvent)]
    assert len(delegations) == 2, "Two delegations expected — one per specialist call."
    assert delegations[0].caller_agent == "coordinator"
    assert delegations[0].delegate_agent == "db-specialist"
    assert delegations[1].caller_agent == "coordinator"
    assert delegations[1].delegate_agent == "db-specialist"
    print(f"  DelegationEvents: {len(delegations)} (coordinator → db-specialist x2)")

    # Specialist AgentStartEvents — exactly two; the second must show replay.
    specialist_starts = [
        e for e in emitter.events if isinstance(e, AgentStartEvent) and e.agent_name == "db-specialist"
    ]
    assert len(specialist_starts) == 2

    assert specialist_starts[0].thread_key == "db-thread"
    assert specialist_starts[0].replayed_message_count == 0
    print(
        f"  Specialist call 1: thread_key={specialist_starts[0].thread_key!r}, "
        f"replayed_message_count={specialist_starts[0].replayed_message_count}"
    )

    assert specialist_starts[1].thread_key == "db-thread"
    assert specialist_starts[1].replayed_message_count >= 1, (
        "Second specialist call must observe at least one replayed message."
    )
    print(
        f"  Specialist call 2: thread_key={specialist_starts[1].thread_key!r}, "
        f"replayed_message_count={specialist_starts[1].replayed_message_count}"
    )

    # Exactly one specialist AgentStartEvent has replay > 0.
    replayed = [e for e in specialist_starts if e.replayed_message_count >= 1]
    assert len(replayed) == 1
    print(f"  Specialist AgentStartEvents with replay >= 1: {len(replayed)} (the second call)")

    # Confirm the thread persists both user+assistant pairs.
    db_thread = await store.load("db-thread")
    assert len(db_thread) >= 4
    print(f"  Thread 'db-thread' message count: {len(db_thread)}")

    print("✓ Section 3 passed")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
