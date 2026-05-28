"""Behavioral continuity: drafter→critic→drafter via ``create_handoff_chain``.

Demonstrates the ``thread_keys`` parameter on ``create_handoff_chain``. The
same ``drafter`` agent instance appears at positions 0 and 2 of the chain,
both keyed to the same thread. Between those positions the ``critic``
reviews the draft on a separate (null) key. When the drafter runs the
second time it sees its own first turn as its own prior assistant message
— not as injected context — and revises it directly.

Behavioral continuity is opt-in: it requires a ``ThreadStore`` wired into
the agent plus matching ``thread_keys`` entries on the chain.

Related guide: docs/guides/multi-agent-foundations.md#behavioral-continuity-in-multi-agent-patterns
"""

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics.composition import InMemoryThreadStore
from nanitics.infrastructure import (
    AgentStartEvent,
    HandoffEvent,
    MockLLMClient,
)
from nanitics.patterns import create_handoff_chain
from nanitics.strategies import ReActAgent


async def main() -> None:
    # --- Section 1: Shared thread store and scripted clients ---
    print("--- Section 1: Shared thread store and scripted clients ---")

    emitter = make_emitter("threads-drafter-critic")

    # One thread store is shared across every agent in the chain. The
    # drafter is keyed to "draft-1" at positions 0 and 2; the critic runs
    # at position 1 with no thread key.
    store = InMemoryThreadStore()

    drafter_client = MockLLMClient(
        [
            # Position 0: initial draft, no replay.
            make_response(
                "Draft v1: A small coffee shop, Bean There, brews single-origin espresso for the morning commute."
            ),
            # Position 2: revision, sees its v1 as a prior assistant turn.
            make_response(
                "Draft v2: Building on my earlier draft, Bean There now leans "
                "into community — single-origin espresso, brewed for the "
                "neighbours who keep coming back."
            ),
        ]
    )

    critic_client = MockLLMClient(
        [
            make_response(
                "Critique: the draft is concrete but reads as transactional. Lean into community to give it warmth."
            ),
        ]
    )

    drafter = ReActAgent(
        name="drafter",
        llm_client=drafter_client,
        emitter=emitter,
        system_prompt=(
            "You are a copywriter. When revising, treat your prior assistant "
            "turn as your own work and rewrite it in place."
        ),
        tools=[],
        thread_store=store,
    )
    critic = ReActAgent(
        name="critic",
        llm_client=critic_client,
        emitter=emitter,
        system_prompt="You are an editor. Critique the draft.",
        tools=[],
        thread_store=store,
    )

    print("  Drafter and critic share one InMemoryThreadStore.")
    print("  Critic runs without a thread key (position 1, stateless).")

    print("✓ Section 1 passed")

    # --- Section 2: Chain with positional thread_keys ---
    print("\n--- Section 2: Chain with positional thread_keys ---")

    # thread_keys is parallel to agents. Same key at positions 0 and 2 means
    # the drafter's second invocation replays its first turn.
    chain = create_handoff_chain(
        name="drafter-critic-drafter",
        agents=[drafter, critic, drafter],
        emitter=emitter,
        thread_keys=["draft-1", None, "draft-1"],
    )

    result = await chain.execute("Write a one-sentence pitch for a coffee shop named 'Bean There'.")

    # The chain returns the final drafter's output (RawOutputTransfer at the tail).
    assert "Draft v2" in (result.output or "")
    assert "earlier draft" in (result.output or "")
    print(f"  Final output: {result.output!r}")

    # Intermediate results expose each step's output.
    intermediate = result.metadata.get("intermediate_results", {})
    assert "drafter" in intermediate
    assert "critic" in intermediate
    print(f"  Steps executed: {list(intermediate.keys())}")

    print("✓ Section 2 passed")

    # --- Section 3: Behavioral-continuity assertions ---
    print("\n--- Section 3: Behavioral-continuity assertions ---")

    # AgentStartEvents report replay state per run. Order: drafter (pos 0),
    # critic (pos 1), drafter (pos 2).
    start_events = [e for e in emitter.events if isinstance(e, AgentStartEvent)]
    drafter_starts = [e for e in start_events if e.agent_name == "drafter"]
    critic_starts = [e for e in start_events if e.agent_name == "critic"]

    assert len(drafter_starts) == 2
    assert len(critic_starts) == 1

    # First drafter invocation: empty store, no replay.
    assert drafter_starts[0].thread_key == "draft-1"
    assert drafter_starts[0].replayed_message_count == 0
    print(
        f"  Drafter run 1: thread_key={drafter_starts[0].thread_key!r}, "
        f"replayed_message_count={drafter_starts[0].replayed_message_count}"
    )

    # Critic runs stateless.
    assert critic_starts[0].thread_key is None
    assert critic_starts[0].replayed_message_count == 0
    print(
        f"  Critic run:    thread_key={critic_starts[0].thread_key!r}, "
        f"replayed_message_count={critic_starts[0].replayed_message_count}"
    )

    # Second drafter invocation: sees its prior user+assistant pair.
    assert drafter_starts[1].thread_key == "draft-1"
    assert drafter_starts[1].replayed_message_count >= 1, (
        "Second drafter invocation must observe at least one replayed message."
    )
    print(
        f"  Drafter run 2: thread_key={drafter_starts[1].thread_key!r}, "
        f"replayed_message_count={drafter_starts[1].replayed_message_count}"
    )

    # Two handoff events in the chain (drafter→critic, critic→drafter,
    # drafter→output is three actually since each step emits one).
    handoff_events = [e for e in emitter.events if isinstance(e, HandoffEvent)]
    assert len(handoff_events) == 3
    print(f"  HandoffEvents: {[(e.from_agent, e.to_agent) for e in handoff_events]}")

    # The store advanced: the drafter's thread now contains v1's user+assistant
    # pair plus v2's user+assistant pair (the input the chain passes to the
    # second drafter invocation is the critique).
    final_thread = await store.load("draft-1")
    assert len(final_thread) >= 4
    roles = [m.role for m in final_thread]
    assert roles.count("assistant") >= 2, "Both drafter turns should be persisted."
    print(f"  Thread 'draft-1' message roles: {roles}")

    print("✓ Section 3 passed")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
