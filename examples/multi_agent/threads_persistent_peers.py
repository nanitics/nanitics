"""Behavioral continuity: per-peer threads across repeated ``PeerNetwork`` runs.

Demonstrates ``PeerSpec.thread_key`` + ``PeerNetwork(thread_store=...)``.
Two peers — ``planner`` and ``executor`` — each carry their own
``thread_key``. Two sequential ``network.run("planner", …)`` calls
exercise per-peer identity: the planner's second invocation sees its
first plan as its own prior turn; the executor never runs in this
example, so its thread stays empty.

Per-peer-identity is the default scoping. Each peer accumulates its own
history across ``network.run`` calls regardless of who called it.

Related guide: docs/guides/multi-agent-foundations.md#behavioral-continuity-in-multi-agent-patterns
"""

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics.composition import InMemoryThreadStore
from nanitics.infrastructure import (
    AgentStartEvent,
    MockLLMClient,
)
from nanitics.specialized import PeerNetwork, PeerSpec


async def main() -> None:
    # --- Section 1: Per-peer thread keys ---
    print("--- Section 1: Per-peer thread keys ---")

    emitter = make_emitter("threads-persistent-peers")

    # Scripted responses: the planner answers twice, the executor never runs.
    planner_client = MockLLMClient(
        [
            make_response(
                "Plan v1: (1) audit existing schema, (2) draft a migration script, (3) dry-run against staging."
            ),
            make_response(
                "Plan v2: refining yesterday's three-step plan — the staging dry-run "
                "passed, so today (4) widen the rollout to one production shard and "
                "(5) capture metrics for the team review."
            ),
        ]
    )
    executor_client = MockLLMClient(
        # Never reached in this scenario; only present to satisfy the spec.
        [make_response("(unused)")]
    )

    store = InMemoryThreadStore()

    network = PeerNetwork(
        peers=[
            PeerSpec(
                name="planner",
                description="Drafts and refines migration plans.",
                llm_client=planner_client,
                system_prompt=(
                    "You are a migration planner. When refining, treat your "
                    "prior plan as your own work and revise it directly."
                ),
                tools=[],
                allowed_peers=[],  # leaf for this scenario; planner doesn't consult.
                thread_key="planner-thread",
            ),
            PeerSpec(
                name="executor",
                description="Executes plan steps.",
                llm_client=executor_client,
                system_prompt="You execute migration steps.",
                tools=[],
                allowed_peers=[],
                thread_key="executor-thread",
            ),
        ],
        emitter=emitter,
        thread_store=store,
        max_invocations=10,
    )

    print("  PeerNetwork wired with two peers; each carries its own thread_key.")
    print("  thread_store is shared across both peers.")

    print("✓ Section 1 passed")

    # --- Section 2: First network.run — empty store, no replay ---
    print("\n--- Section 2: First network.run — empty store, no replay ---")

    result_1 = await network.run("planner", "Draft the migration plan.")
    assert "Plan v1" in (result_1.output or "")
    print(f"  Run 1 output: {result_1.output!r}")

    print("✓ Section 2 passed")

    # --- Section 3: Second network.run — planner sees its prior plan ---
    print("\n--- Section 3: Second network.run — planner sees its prior plan ---")

    result_2 = await network.run("planner", "Refine based on yesterday's progress.")
    assert "Plan v2" in (result_2.output or "")
    # Concrete cross-reference: the second response explicitly references
    # the first plan's content ("three-step plan").
    assert "three-step plan" in (result_2.output or "").lower()
    print(f"  Run 2 output: {result_2.output!r}")
    print("  Run 2 references run 1 explicitly ('three-step plan') ✓")

    print("✓ Section 3 passed")

    # --- Section 4: Behavioral-continuity assertions ---
    print("\n--- Section 4: Behavioral-continuity assertions ---")

    start_events = [e for e in emitter.events if isinstance(e, AgentStartEvent)]
    planner_starts = [e for e in start_events if e.agent_name == "planner"]
    executor_starts = [e for e in start_events if e.agent_name == "executor"]

    assert len(planner_starts) == 2, "Planner ran twice."
    assert len(executor_starts) == 0, "Executor never ran in this scenario."

    # First planner run: empty store.
    assert planner_starts[0].thread_key == "planner-thread"
    assert planner_starts[0].replayed_message_count == 0
    print(
        f"  Planner run 1: thread_key={planner_starts[0].thread_key!r}, "
        f"replayed_message_count={planner_starts[0].replayed_message_count}"
    )

    # Second planner run: replay length >= 1.
    assert planner_starts[1].thread_key == "planner-thread"
    assert planner_starts[1].replayed_message_count >= 1, (
        "Second planner invocation must observe at least one replayed message."
    )
    print(
        f"  Planner run 2: thread_key={planner_starts[1].thread_key!r}, "
        f"replayed_message_count={planner_starts[1].replayed_message_count}"
    )

    # Executor's thread is empty — per-peer identity, not per-network.
    executor_thread = await store.load("executor-thread")
    assert executor_thread == []
    print(f"  Executor thread length: {len(executor_thread)} (never ran)")

    planner_thread = await store.load("planner-thread")
    assert len(planner_thread) >= 4  # two user inputs + two assistant turns
    print(f"  Planner thread length: {len(planner_thread)} messages")

    print("✓ Section 4 passed")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
