"""Behavioral continuity: orchestrator with a stateful specialist.

Demonstrates ``create_orchestrator`` with a specialist whose ``AgentTool``
carries a ``thread_key``. The orchestrator dispatches the same specialist
twice in one outer run; the specialist's second turn references its
prior turn.

The orchestrator pattern dispatches specialists through ``AgentTool``
instances. Threading is configured per specialist by wiring a
``thread_key`` on the ``AgentTool`` and a ``ThreadStore`` on the wrapped
agent — the orchestrator itself does not need to know about threads.

Related guide: docs/guides/multi-agent-coordination.md
"""

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics.composition import AgentTool, InMemoryThreadStore
from nanitics.infrastructure import (
    AgentStartEvent,
    DelegationEvent,
    MockLLMClient,
)
from nanitics.patterns import create_orchestrator
from nanitics.strategies import ReActAgent
from nanitics.tracing import ToolCall


async def main() -> None:
    # --- Section 1: Stateful specialist behind an AgentTool ---
    print("--- Section 1: Stateful specialist behind an AgentTool ---")

    emitter = make_emitter("threads-orchestrator-stateful-specialist")
    store = InMemoryThreadStore()

    # Researcher: stateful via thread_key.
    researcher_client = MockLLMClient(
        [
            make_response(
                "Finding v1: latency on the checkout path spiked 18% last "
                "Tuesday; the regression coincides with the cache rollout."
            ),
            make_response(
                "Finding v2: continuing from the Tuesday spike, the cache "
                "hit-rate dropped from 92% to 71% — likely the cause of "
                "the latency regression."
            ),
        ]
    )
    researcher = ReActAgent(
        name="researcher",
        llm_client=researcher_client,
        emitter=emitter,
        system_prompt=(
            "You investigate production incidents. Treat prior assistant "
            "turns as your own findings and extend them directly."
        ),
        tools=[],
        thread_store=store,
    )

    # Writer: stateless — runs once, no thread key.
    writer_client = MockLLMClient(
        [
            make_response(
                "Report: a Tuesday checkout-latency spike (+18%) tracks to a "
                "cache hit-rate drop from 92% to 71% following the rollout."
            ),
        ]
    )
    writer = ReActAgent(
        name="writer",
        llm_client=writer_client,
        emitter=emitter,
        system_prompt="You write incident reports.",
        tools=[],
    )

    specialists = [
        AgentTool(
            agent=researcher,
            emitter=emitter,
            description="Investigate production incidents and build evidence.",
            thread_key="researcher-thread",
        ),
        AgentTool(
            agent=writer,
            emitter=emitter,
            description="Write incident reports from prior findings.",
        ),
    ]

    print("  Researcher AgentTool carries thread_key='researcher-thread'.")
    print("  Writer AgentTool is stateless.")

    print("✓ Section 1 passed")

    # --- Section 2: Orchestrator dispatches the researcher twice, then writer ---
    print("\n--- Section 2: Orchestrator dispatches the researcher twice, then writer ---")

    orchestrator_client = MockLLMClient(
        [
            # Step 1: first researcher dispatch.
            make_response(
                content="Start with the latency regression.",
                tool_calls=[
                    ToolCall(
                        id="tc-r1",
                        name="researcher",
                        arguments={"task": "What changed last Tuesday?"},
                    )
                ],
                stop_reason="tool_use",
            ),
            # Step 2: second researcher dispatch — same specialist, continues thread.
            make_response(
                content="Drill into the cache as the likely cause.",
                tool_calls=[
                    ToolCall(
                        id="tc-r2",
                        name="researcher",
                        arguments={"task": "Pin the cause to the cache rollout."},
                    )
                ],
                stop_reason="tool_use",
            ),
            # Step 3: hand off to the writer.
            make_response(
                content="Hand the findings to the writer.",
                tool_calls=[
                    ToolCall(
                        id="tc-w1",
                        name="writer",
                        arguments={"task": "Write up the incident."},
                    )
                ],
                stop_reason="tool_use",
            ),
            # Step 4: synthesize the final response.
            make_response(
                "Incident summary: Tuesday checkout-latency spike (+18%) caused "
                "by a cache hit-rate drop from 92% to 71% after the rollout."
            ),
        ]
    )

    orchestrator = create_orchestrator(
        name="coordinator",
        llm_client=orchestrator_client,
        emitter=emitter,
        specialists=specialists,
    )

    result = await orchestrator.run("Investigate the checkout-latency regression.")

    assert "18%" in (result.output or "")
    assert "cache hit-rate" in (result.output or "")
    print(f"  Coordinator output: {result.output!r}")
    print(f"  Coordinator steps: {result.total_steps}")

    print("✓ Section 2 passed")

    # --- Section 3: Behavioral-continuity assertions ---
    print("\n--- Section 3: Behavioral-continuity assertions ---")

    delegations = [e for e in emitter.events if isinstance(e, DelegationEvent)]
    assert len(delegations) == 3
    delegate_names = [e.delegate_agent for e in delegations]
    assert delegate_names == ["researcher", "researcher", "writer"]
    print(f"  Delegations: {delegate_names}")

    researcher_starts = [e for e in emitter.events if isinstance(e, AgentStartEvent) and e.agent_name == "researcher"]
    assert len(researcher_starts) == 2

    assert researcher_starts[0].thread_key == "researcher-thread"
    assert researcher_starts[0].replayed_message_count == 0
    print(
        f"  Researcher call 1: thread_key={researcher_starts[0].thread_key!r}, "
        f"replayed_message_count={researcher_starts[0].replayed_message_count}"
    )

    assert researcher_starts[1].thread_key == "researcher-thread"
    assert researcher_starts[1].replayed_message_count >= 1, (
        "Second researcher dispatch must observe at least one replayed message."
    )
    print(
        f"  Researcher call 2: thread_key={researcher_starts[1].thread_key!r}, "
        f"replayed_message_count={researcher_starts[1].replayed_message_count}"
    )

    # Writer runs stateless — no replay, no thread key.
    writer_starts = [e for e in emitter.events if isinstance(e, AgentStartEvent) and e.agent_name == "writer"]
    assert len(writer_starts) == 1
    assert writer_starts[0].thread_key is None
    assert writer_starts[0].replayed_message_count == 0
    print(
        f"  Writer call:       thread_key={writer_starts[0].thread_key!r}, "
        f"replayed_message_count={writer_starts[0].replayed_message_count}"
    )

    print("✓ Section 3 passed")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
