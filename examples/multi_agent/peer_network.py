"""Peer Network: decentralized agent consultation with shared budget.

Demonstrates PeerNetwork — a peer-to-peer pattern where agents consult each
other directly via auto-generated consult_<name> tools. A shared invocation
budget prevents runaway recursion.

Related guide: docs/guides/multi-agent-foundations.md
"""

import asyncio

from examples.helpers import make_emitter, make_response, make_usage
from nanitics import (
    LLMResponse,
    MockLLMClient,
    ToolCall,
)
from nanitics.experimental import (
    PeerNetwork,
    PeerSpec,
)
from nanitics.infrastructure import (
    PeerConsultationEvent,
    PeerNetworkCompleteEvent,
    PeerNetworkStartEvent,
)


async def main() -> None:
    # --- Section 1: Basic Peer Consultation ---
    # Two peers: researcher (entry) consults analyst, then produces a synthesis.

    print("--- Section 1: Basic Peer Consultation ---")

    emitter = make_emitter("peer-s1")

    researcher_client = MockLLMClient(
        responses=[
            # Step 1: researcher decides to consult the analyst
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="tc-1",
                        name="consult_analyst",
                        arguments={"message": "What are the key risks of entering the Asian market?"},
                    )
                ],
                usage=make_usage(),
                model="mock-model",
                stop_reason="tool_use",
            ),
            # Step 2: researcher synthesizes final answer after receiving analyst's input
            make_response(
                "Based on the analyst's assessment, the main risks are regulatory "
                "complexity and currency volatility, but the growth opportunity outweighs them."
            ),
        ]
    )

    analyst_client = MockLLMClient(
        responses=[
            make_response(
                "Key risks: (1) regulatory fragmentation across countries, "
                "(2) currency volatility, (3) established local competitors."
            ),
        ]
    )

    network = PeerNetwork(
        peers=[
            PeerSpec(
                name="researcher",
                description="Researches market opportunities and synthesizes findings.",
                llm_client=researcher_client,
                system_prompt="You are a market researcher.",
                tools=[],
            ),
            PeerSpec(
                name="analyst",
                description="Analyzes risks and provides quantitative assessments.",
                llm_client=analyst_client,
                system_prompt="You are a risk analyst.",
                tools=[],
            ),
        ],
        emitter=emitter,
        max_invocations=10,
    )

    result = await network.run("researcher", "Should we expand into the Asian market?")

    # Researcher's final synthesis incorporates analyst's input
    assert "regulatory" in result.output.lower()
    assert result.total_steps == 2
    print("  Task: Should we expand into the Asian market?")

    # Verify events
    start_events = [e for e in emitter.events if isinstance(e, PeerNetworkStartEvent)]
    assert len(start_events) == 1
    assert start_events[0].entry_agent == "researcher"
    assert set(start_events[0].peer_names) == {"researcher", "analyst"}
    assert start_events[0].max_invocations == 10

    consultation_events = [e for e in emitter.events if isinstance(e, PeerConsultationEvent)]
    assert len(consultation_events) == 1
    assert consultation_events[0].from_agent == "researcher"
    assert consultation_events[0].to_agent == "analyst"
    assert consultation_events[0].consultation_number == 1
    print(f"  researcher consulted analyst: {consultation_events[0].message!r}")

    complete_events = [e for e in emitter.events if isinstance(e, PeerNetworkCompleteEvent)]
    assert len(complete_events) == 1
    assert complete_events[0].total_consultations == 1
    assert complete_events[0].agents_consulted == ["analyst"]
    print(f"  Final output: {result.output}")
    print(f"  Total consultations: {complete_events[0].total_consultations}")
    print(f"  Agents consulted: {complete_events[0].agents_consulted}")

    print("✓ Researcher consulted analyst and produced synthesis")

    # --- Section 2: Transitive Consultation Chain ---
    # Three peers: lead → strategist → data_analyst.
    # The consultation graph is declared structurally via `allowed_peers`:
    # lead consults only strategist; strategist consults only data_analyst;
    # data_analyst is a leaf consultant with no downstream peers.

    print("\n--- Section 2: Transitive Consultation Chain ---")

    emitter = make_emitter("peer-s2")

    lead_client = MockLLMClient(
        responses=[
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="tc-1",
                        name="consult_strategist",
                        arguments={"message": "Develop a go-to-market strategy for product launch."},
                    )
                ],
                usage=make_usage(),
                model="mock-model",
                stop_reason="tool_use",
            ),
            make_response(
                "Strategy finalized: phased rollout starting with enterprise segment, "
                "leveraging strong unit economics confirmed by data analysis."
            ),
        ]
    )

    strategist_client = MockLLMClient(
        responses=[
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="tc-2",
                        name="consult_data_analyst",
                        arguments={"message": "What do the unit economics look like for enterprise vs SMB segments?"},
                    )
                ],
                usage=make_usage(),
                model="mock-model",
                stop_reason="tool_use",
            ),
            make_response(
                "Recommend enterprise-first approach. Unit economics are 3x stronger "
                "in enterprise (confirmed by data team) with lower churn."
            ),
        ]
    )

    data_analyst_client = MockLLMClient(
        responses=[
            make_response(
                "Enterprise segment: CAC $2k, LTV $18k, ratio 9:1. "
                "SMB segment: CAC $500, LTV $1.5k, ratio 3:1. Enterprise clearly stronger."
            ),
        ]
    )

    network = PeerNetwork(
        peers=[
            PeerSpec(
                name="lead",
                description="Leads product strategy and makes final decisions.",
                llm_client=lead_client,
                system_prompt="You are a product lead.",
                tools=[],
                allowed_peers=["strategist"],
            ),
            PeerSpec(
                name="strategist",
                description="Develops go-to-market strategies based on data.",
                llm_client=strategist_client,
                system_prompt="You are a go-to-market strategist.",
                tools=[],
                allowed_peers=["data_analyst"],
            ),
            PeerSpec(
                name="data_analyst",
                description="Analyzes unit economics and market data.",
                llm_client=data_analyst_client,
                system_prompt="You are a data analyst specializing in SaaS metrics.",
                tools=[],
                allowed_peers=[],  # Leaf consultant — no downstream peers.
            ),
        ],
        emitter=emitter,
        max_invocations=10,
    )

    result = await network.run("lead", "Plan our product launch strategy.")

    assert "enterprise" in result.output.lower()

    # Two consultation events: lead→strategist, then strategist→data_analyst
    consultation_events = [e for e in emitter.events if isinstance(e, PeerConsultationEvent)]
    assert len(consultation_events) == 2
    assert consultation_events[0].from_agent == "lead"
    assert consultation_events[0].to_agent == "strategist"
    assert consultation_events[0].consultation_number == 1
    assert consultation_events[1].from_agent == "strategist"
    assert consultation_events[1].to_agent == "data_analyst"
    assert consultation_events[1].consultation_number == 2
    print("  Chain: lead → strategist → data_analyst")
    print(f"  Consultation 1: {consultation_events[0].from_agent} → {consultation_events[0].to_agent}")
    print(f"  Consultation 2: {consultation_events[1].from_agent} → {consultation_events[1].to_agent}")

    complete_events = [e for e in emitter.events if isinstance(e, PeerNetworkCompleteEvent)]
    assert complete_events[0].total_consultations == 2
    assert complete_events[0].agents_consulted == ["data_analyst", "strategist"]
    print(f"  Final output: {result.output}")
    print(f"  Total consultations: {complete_events[0].total_consultations}")
    print(f"  Agents consulted: {complete_events[0].agents_consulted}")

    print("✓ Transitive chain lead→strategist→data_analyst completed within budget")

    # --- Section 3: Budget Exhaustion ---
    # Tight budget (max_invocations=1). Coordinator consults expert_a (succeeds),
    # then tries expert_b (fails — budget exhausted). Agent must produce a
    # partial answer acknowledging the limitation.

    print("\n--- Section 3: Budget Exhaustion ---")

    emitter = make_emitter("peer-s3")

    coordinator_client = MockLLMClient(
        responses=[
            # Step 1: consult expert_a (succeeds)
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="tc-1",
                        name="consult_expert_a",
                        arguments={"message": "Analyze technical feasibility."},
                    )
                ],
                usage=make_usage(),
                model="mock-model",
                stop_reason="tool_use",
            ),
            # Step 2: try to consult expert_b (will fail — budget exhausted)
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="tc-2",
                        name="consult_expert_b",
                        arguments={"message": "Analyze market demand."},
                    )
                ],
                usage=make_usage(),
                model="mock-model",
                stop_reason="tool_use",
            ),
            # Step 3: final answer after budget error
            make_response(
                "Based on available input: technically feasible per expert_a. "
                "Market analysis unavailable due to consultation budget limits."
            ),
        ]
    )

    expert_a_client = MockLLMClient(
        responses=[
            make_response("Technically feasible. Estimated 3 months development time with existing infrastructure."),
        ]
    )

    expert_b_client = MockLLMClient(
        responses=[
            # Won't be reached — budget is exhausted before this agent runs
            make_response("Strong market demand — never reached."),
        ]
    )

    network = PeerNetwork(
        peers=[
            PeerSpec(
                name="coordinator",
                description="Coordinates expert input for project evaluation.",
                llm_client=coordinator_client,
                system_prompt="You coordinate expert assessments.",
                tools=[],
            ),
            PeerSpec(
                name="expert_a",
                description="Technical feasibility expert.",
                llm_client=expert_a_client,
                system_prompt="You assess technical feasibility.",
                tools=[],
            ),
            PeerSpec(
                name="expert_b",
                description="Market demand analyst.",
                llm_client=expert_b_client,
                system_prompt="You analyze market demand.",
                tools=[],
            ),
        ],
        emitter=emitter,
        max_invocations=1,  # only one consultation allowed
    )

    result = await network.run("coordinator", "Evaluate this project proposal.")

    # Coordinator produced a partial answer acknowledging the budget limit
    assert "budget" in result.output.lower()
    assert result.total_steps == 3  # consult_a success, consult_b failure, final text

    # Only one consultation succeeded
    consultation_events = [e for e in emitter.events if isinstance(e, PeerConsultationEvent)]
    assert len(consultation_events) == 1
    assert consultation_events[0].to_agent == "expert_a"

    complete_events = [e for e in emitter.events if isinstance(e, PeerNetworkCompleteEvent)]
    assert complete_events[0].total_consultations == 1
    print("  Budget: max_invocations=1")
    print("  Consultation 1 (expert_a): succeeded ✓")
    print("  Consultation 2 (expert_b): budget exhausted ✗")
    print(f"  Final output: {result.output}")
    print(f"  Total consultations: {complete_events[0].total_consultations}")
    print(f"  Total steps: {result.total_steps}")

    print("✓ Agent handled budget exhaustion and produced partial answer")


if __name__ == "__main__":
    asyncio.run(main())
