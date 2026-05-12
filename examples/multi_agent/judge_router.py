"""JudgeRouter: comparative-judgment task allocation.

Demonstrates ``JudgeRouter`` — a centralised comparative-judgment routing
primitive that asks a single judge LLM to rank all candidate agents in
one call, counter-balancing the self-overclaim bias inherent to
independent self-rated bidding (see ``Bidding``). Covers the basic flow,
the calibration-anchor template, the optional confidence threshold, the
full event family, and a side-by-side comparison with ``Bidding``.

Related guide: docs/guides/multi-agent-coordination.md
"""

import asyncio
import json

from examples.helpers import make_emitter, make_response
from nanitics import (
    DEFAULT_CALIBRATED_JUDGE_PROMPT_TEMPLATE,
    BiddableAgent,
    JudgeRouter,
    MockLLMClient,
    ReActAgent,
)
from nanitics.experimental import (
    Bidding,
    FixedBidGenerator,
    HighestConfidence,
)
from nanitics.infrastructure import (
    JudgeAllocatedEvent,
    JudgeRankingEvent,
    JudgeRoutingCompleteEvent,
    JudgeRoutingStartEvent,
)


def _ranking_response(*candidates: dict[str, object]) -> str:
    """Encode a judge-ranking JSON payload for the structured-output schema."""
    return json.dumps({"ranking": list(candidates)})


async def main() -> None:
    # --- Section 1: Basic Comparative Judgment ---
    print("--- Section 1: Basic Comparative Judgment ---")

    emitter = make_emitter("judge-router-s1")

    billing_agent = ReActAgent(
        name="billing-specialist",
        llm_client=MockLLMClient(
            responses=[
                make_response(
                    "I checked your invoice — the discrepancy is a duplicate line item; refund issued.",
                ),
            ]
        ),
        emitter=emitter,
        system_prompt="You are a billing specialist.",
        tools=[],
    )

    technical_agent = ReActAgent(
        name="technical-support",
        llm_client=MockLLMClient(responses=[make_response("Not invoked — ranked below billing.")]),
        emitter=emitter,
        system_prompt="You are technical support.",
        tools=[],
    )

    account_agent = ReActAgent(
        name="account-manager",
        llm_client=MockLLMClient(responses=[make_response("Not invoked — ranked below billing.")]),
        emitter=emitter,
        system_prompt="You are an account manager.",
        tools=[],
    )

    # FixedBidGenerator placeholders — JudgeRouter ignores bid generators,
    # but BiddableAgent is reused so adopters can swap Bidding ↔ JudgeRouter
    # at the call site without rebuilding agents.
    participants = [
        BiddableAgent(agent=billing_agent, bid_generator=FixedBidGenerator(confidence=0.0)),
        BiddableAgent(agent=technical_agent, bid_generator=FixedBidGenerator(confidence=0.0)),
        BiddableAgent(agent=account_agent, bid_generator=FixedBidGenerator(confidence=0.0)),
    ]

    judge_client = MockLLMClient(
        responses=[
            make_response(
                _ranking_response(
                    {
                        "agent_name": "billing-specialist",
                        "confidence": 0.9,
                        "capabilities": ["invoices", "refunds"],
                        "estimated_cost": 0.04,
                        "reasoning": "Invoice-amount disputes fall squarely in this specialist's scope.",
                    },
                    {
                        "agent_name": "account-manager",
                        "confidence": 0.4,
                        "capabilities": ["account-status"],
                        "estimated_cost": 0.03,
                        "reasoning": "Adjacent — account context helps but billing is the primary concern.",
                    },
                    {
                        "agent_name": "technical-support",
                        "confidence": 0.0,
                        "capabilities": ["bugs"],
                        "estimated_cost": 0.02,
                        "reasoning": "Out of scope — no technical fault implied by the request.",
                    },
                )
            ),
        ]
    )

    router = JudgeRouter(
        participants=participants,
        judge_llm=judge_client,
        emitter=emitter,
    )

    result = await router.run("My invoice shows the wrong amount.")

    assert result.allocated is True
    assert result.winner is not None
    assert result.winner.agent_name == "billing-specialist"
    assert result.winner.confidence == 0.9
    assert len(result.ranking) == 3
    # Winning agent actually executed.
    assert "refund issued" in result.execution_result
    print(f"  Winner: {result.winner.agent_name} (confidence={result.winner.confidence})")
    print(f"  Ranking: {[c.agent_name for c in result.ranking]}")
    print(f"  Execution result: {result.execution_result}")
    print("✓ Single LLM call ranked all candidates and the top match executed")

    # --- Section 2: Calibration-Anchor Template ---
    print("\n--- Section 2: Calibration-Anchor Template ---")

    # The default template embeds four-tier calibration anchors so the
    # judge picks a band rather than rating in isolation. Pin the anchors
    # are present in the canonical template body.
    for anchor in (
        "0.9 = uniquely positioned",
        "0.7 = capable",
        "0.4 = adjacent",
        "0.0 = out of scope",
    ):
        assert anchor in DEFAULT_CALIBRATED_JUDGE_PROMPT_TEMPLATE, (
            f"Calibration anchor missing from default template: {anchor!r}"
        )
    assert "{participants}" in DEFAULT_CALIBRATED_JUDGE_PROMPT_TEMPLATE
    assert "{task}" in DEFAULT_CALIBRATED_JUDGE_PROMPT_TEMPLATE
    print("  Default template carries all four calibration anchors")

    # Verify the rendered prompt actually reaches the judge LLM.
    emitter = make_emitter("judge-router-s2")
    judge_client = MockLLMClient(
        responses=[
            make_response(
                _ranking_response(
                    {
                        "agent_name": "billing-specialist",
                        "confidence": 0.9,
                        "capabilities": ["invoices"],
                        "reasoning": "Calibrated 0.9 — uniquely positioned.",
                    },
                )
            ),
        ]
    )
    router = JudgeRouter(
        participants=[
            BiddableAgent(
                agent=ReActAgent(
                    name="billing-specialist",
                    llm_client=MockLLMClient(responses=[make_response("done")]),
                    emitter=emitter,
                    system_prompt="Billing.",
                    tools=[],
                ),
                bid_generator=FixedBidGenerator(confidence=0.0),
            ),
        ],
        judge_llm=judge_client,
        emitter=emitter,
    )
    await router.run("Invoice amount wrong.")
    judge_prompt = judge_client.calls[0]["messages"][0].content
    for anchor in ("0.9 = uniquely positioned", "0.7 = capable", "0.4 = adjacent", "0.0 = out of scope"):
        assert anchor in judge_prompt, f"Anchor not rendered into judge prompt: {anchor!r}"
    assert "billing-specialist" in judge_prompt  # participants substituted
    assert "Invoice amount wrong." in judge_prompt  # task substituted
    print("  Calibration anchors rendered into the judge LLM call")
    print("✓ Calibration template counters self-overclaim by anchoring against bands")

    # --- Section 3: Below-Threshold Rejection ---
    print("\n--- Section 3: Below-Threshold Rejection ---")

    emitter = make_emitter("judge-router-s3")
    weak_agent = ReActAgent(
        name="generalist",
        llm_client=MockLLMClient(responses=[make_response("not invoked")]),
        emitter=emitter,
        system_prompt="Generalist.",
        tools=[],
    )
    judge_client = MockLLMClient(
        responses=[
            make_response(
                _ranking_response(
                    {
                        "agent_name": "generalist",
                        "confidence": 0.3,
                        "capabilities": ["misc"],
                        "reasoning": "Adjacent at best — no strong fit.",
                    },
                )
            ),
        ]
    )
    router = JudgeRouter(
        participants=[
            BiddableAgent(agent=weak_agent, bid_generator=FixedBidGenerator(confidence=0.0)),
        ],
        judge_llm=judge_client,
        emitter=emitter,
        min_confidence_threshold=0.7,
    )
    result = await router.run("Specialised cryptography question.")
    assert result.allocated is False
    assert result.winner is None
    assert result.execution_result is None
    # The ranking is still surfaced — the caller can inspect rejected candidates.
    assert len(result.ranking) == 1
    assert result.ranking[0].confidence == 0.3
    alloc_events = [e for e in emitter.events if isinstance(e, JudgeAllocatedEvent)]
    assert len(alloc_events) == 1
    assert alloc_events[0].rejection_reason == "below_threshold"
    print(f"  Top candidate confidence: {result.ranking[0].confidence} (threshold: 0.7)")
    print(f"  Allocated: {result.allocated}, rejection_reason: {alloc_events[0].rejection_reason!r}")
    print("✓ Threshold rejects weak top match — caller sees the ranking but no execution")

    # --- Section 4: Event Trace ---
    print("\n--- Section 4: Event Trace ---")

    emitter = make_emitter("judge-router-s4")
    agents = [
        ReActAgent(
            name=name,
            llm_client=MockLLMClient(responses=[make_response(f"{name} executed")]),
            emitter=emitter,
            system_prompt=f"{name} prompt.",
            tools=[],
        )
        for name in ("billing-specialist", "technical-support", "account-manager")
    ]
    participants = [BiddableAgent(agent=a, bid_generator=FixedBidGenerator(confidence=0.0)) for a in agents]
    judge_client = MockLLMClient(
        responses=[
            make_response(
                _ranking_response(
                    {
                        "agent_name": "billing-specialist",
                        "confidence": 0.9,
                        "capabilities": ["invoices"],
                        "reasoning": "0.9 — uniquely positioned.",
                    },
                    {
                        "agent_name": "account-manager",
                        "confidence": 0.4,
                        "capabilities": ["accounts"],
                        "reasoning": "Adjacent.",
                    },
                    {
                        "agent_name": "technical-support",
                        "confidence": 0.0,
                        "capabilities": ["bugs"],
                        "reasoning": "Out of scope.",
                    },
                )
            ),
        ]
    )
    router = JudgeRouter(participants=participants, judge_llm=judge_client, emitter=emitter)
    await router.run("Invoice dispute.")

    start = [e for e in emitter.events if isinstance(e, JudgeRoutingStartEvent)]
    assert len(start) == 1
    assert set(start[0].participant_names) == {
        "billing-specialist",
        "technical-support",
        "account-manager",
    }
    print(f"  JudgeRoutingStartEvent: participants={sorted(start[0].participant_names)}")

    rankings = [e for e in emitter.events if isinstance(e, JudgeRankingEvent)]
    assert len(rankings) == 3
    assert [e.rank for e in rankings] == [0, 1, 2]
    assert rankings[0].agent_name == "billing-specialist"
    print(f"  JudgeRankingEvent×{len(rankings)}: ranks={[e.rank for e in rankings]}")

    alloc = [e for e in emitter.events if isinstance(e, JudgeAllocatedEvent)]
    assert len(alloc) == 1
    assert alloc[0].winner == "billing-specialist"
    assert alloc[0].total_candidates == 3
    print(f"  JudgeAllocatedEvent: winner={alloc[0].winner}, total_candidates={alloc[0].total_candidates}")

    complete = [e for e in emitter.events if isinstance(e, JudgeRoutingCompleteEvent)]
    assert len(complete) == 1
    assert complete[0].winner == "billing-specialist"
    assert complete[0].allocated is True
    assert complete[0].total_participants == 3
    print(f"  JudgeRoutingCompleteEvent: winner={complete[0].winner}, allocated={complete[0].allocated}")
    print("✓ Full event trace: start → N rankings → allocated → complete")

    # --- Section 5: Bidding vs JudgeRouter ---
    print("\n--- Section 5: Bidding vs JudgeRouter (same task, two primitives) ---")

    # Same task, same agents — see how the two routing primitives differ in
    # the trace they produce. Bidding emits one bid event per participant
    # (decentralised self-assessment); JudgeRouter emits one ranking event
    # per candidate from a single judge call.
    task = "Refund my invoice."

    bidding_emitter = make_emitter("compare-bidding")
    bidding_agent = ReActAgent(
        name="billing-specialist",
        llm_client=MockLLMClient(responses=[make_response("Refund issued.")]),
        emitter=bidding_emitter,
        system_prompt="Billing.",
        tools=[],
    )
    bidding = Bidding(
        participants=[
            BiddableAgent(
                agent=bidding_agent,
                bid_generator=FixedBidGenerator(confidence=0.9, capabilities=["invoices"]),
            ),
        ],
        emitter=bidding_emitter,
        allocation_strategy=HighestConfidence(),
    )
    bidding_result = await bidding.run(task)
    bidding_event_types = {type(e).__name__ for e in bidding_emitter.events}

    judge_emitter = make_emitter("compare-judge")
    judge_agent = ReActAgent(
        name="billing-specialist",
        llm_client=MockLLMClient(responses=[make_response("Refund issued.")]),
        emitter=judge_emitter,
        system_prompt="Billing.",
        tools=[],
    )
    judge_client = MockLLMClient(
        responses=[
            make_response(
                _ranking_response(
                    {
                        "agent_name": "billing-specialist",
                        "confidence": 0.9,
                        "capabilities": ["invoices"],
                        "reasoning": "0.9 — uniquely positioned.",
                    },
                )
            ),
        ]
    )
    router = JudgeRouter(
        participants=[
            BiddableAgent(agent=judge_agent, bid_generator=FixedBidGenerator(confidence=0.0)),
        ],
        judge_llm=judge_client,
        emitter=judge_emitter,
    )
    judge_result = await router.run(task)
    judge_event_types = {type(e).__name__ for e in judge_emitter.events}

    # Both primitives reach the same winner — but their trace shapes differ.
    assert bidding_result.winning_bid is not None
    assert judge_result.winner is not None
    assert bidding_result.winning_bid.agent_name == judge_result.winner.agent_name == "billing-specialist"

    # Bidding emits Bid* events; JudgeRouter emits Judge* events.
    assert "BiddingStartEvent" in bidding_event_types
    assert "BidReceivedEvent" in bidding_event_types
    assert "BidAllocatedEvent" in bidding_event_types
    assert "JudgeRoutingStartEvent" in judge_event_types
    assert "JudgeRankingEvent" in judge_event_types
    assert "JudgeAllocatedEvent" in judge_event_types
    assert "BiddingStartEvent" not in judge_event_types
    assert "JudgeRoutingStartEvent" not in bidding_event_types

    print(f"  Bidding trace event types include: {sorted(t for t in bidding_event_types if 'Bid' in t)}")
    print(f"  JudgeRouter trace event types include: {sorted(t for t in judge_event_types if 'Judge' in t)}")
    print("✓ Same winner via different mechanisms — Bidding fans out, JudgeRouter compares")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
