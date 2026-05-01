"""Bidding: competitive auction-based task allocation.

Demonstrates ``Bidding`` — a competitive auction where agents bid on tasks based
on self-assessed capability. Covers allocation strategies (``HighestConfidence``,
``LowestCost``, ``WeightedScore``), basic auction with ``FixedBidGenerator``,
LLM-driven bidding with ``LLMBidGenerator`` (including the calibration-anchor
template that counters self-overclaim), the
``HighestConfidence(tiebreaker=...)`` chain that breaks strict-tie wins
deterministically, minimum bid threshold rejection, and event trace inspection.

Related guide: docs/guides/multi-agent-coordination.md
"""

import asyncio
import json

from examples.helpers import make_emitter, make_response
from nanitics import (
    DEFAULT_CALIBRATED_BID_PROMPT_TEMPLATE,
    Bid,
    BiddableAgent,
    Bidding,
    FixedBidGenerator,
    HighestConfidence,
    LLMBidGenerator,
    LowestCost,
    MockLLMClient,
    ReActAgent,
    WeightedScore,
)
from nanitics.infrastructure import (
    BidAllocatedEvent,
    BiddingCompleteEvent,
    BiddingStartEvent,
    BidReceivedEvent,
)


async def main() -> None:
    # --- Section 1: Allocation Strategies ---
    print("--- Section 1: Allocation Strategies ---")

    # HighestConfidence — selects the bid with the highest confidence score
    bids = [
        Bid(agent_name="agent-a", confidence=0.6, capabilities=["search"], reasoning="Decent match"),
        Bid(agent_name="agent-b", confidence=0.9, capabilities=["analysis"], reasoning="Strong match"),
        Bid(agent_name="agent-c", confidence=0.4, capabilities=["writing"], reasoning="Weak match"),
    ]

    winner = HighestConfidence().select(bids)
    assert winner is not None
    assert winner.agent_name == "agent-b"
    assert winner.confidence == 0.9
    print(f"  HighestConfidence winner: {winner.agent_name} (confidence={winner.confidence})")

    # LowestCost — selects the bid with the lowest estimated cost
    # Bids without estimated_cost are excluded
    bids_with_cost = [
        Bid(agent_name="expensive", confidence=0.9, capabilities=["all"], estimated_cost=5.0, reasoning="Full suite"),
        Bid(agent_name="cheap", confidence=0.7, capabilities=["basic"], estimated_cost=0.5, reasoning="Budget option"),
        Bid(agent_name="no-cost", confidence=0.8, capabilities=["unknown"], reasoning="No cost info"),
    ]

    winner = LowestCost().select(bids_with_cost)
    assert winner is not None
    assert winner.agent_name == "cheap"
    assert winner.estimated_cost == 0.5
    print(f"  LowestCost winner: {winner.agent_name} (cost={winner.estimated_cost})")

    # Verify no-cost bids are excluded
    no_cost_bids = [
        Bid(agent_name="a", confidence=0.9, capabilities=[], reasoning="No cost"),
    ]
    assert LowestCost().select(no_cost_bids) is None
    print("  LowestCost with no costed bids: None (excluded)")

    # WeightedScore — multi-dimensional scoring across confidence, cost, capabilities
    # Agent-a: lower confidence but cheaper and more capabilities
    # Agent-b: higher confidence but expensive and fewer capabilities
    weighted_bids = [
        Bid(
            agent_name="generalist", confidence=0.6, capabilities=["a", "b", "c"], estimated_cost=1.0, reasoning="Broad"
        ),
        Bid(agent_name="specialist", confidence=0.9, capabilities=["a"], estimated_cost=5.0, reasoning="Narrow"),
    ]

    # Weight cost and capabilities heavily — generalist should win
    strategy = WeightedScore(weights={"confidence": 0.2, "cost": 0.4, "capabilities": 0.4})
    winner = strategy.select(weighted_bids)
    assert winner is not None
    assert winner.agent_name == "generalist"
    print(f"  WeightedScore winner: {winner.agent_name} (cost+capabilities weighted)")

    # Flip weights to favor confidence — specialist should win
    strategy = WeightedScore(weights={"confidence": 0.8, "cost": 0.1, "capabilities": 0.1})
    winner = strategy.select(weighted_bids)
    assert winner is not None
    assert winner.agent_name == "specialist"
    print(f"  WeightedScore winner: {winner.agent_name} (confidence weighted)")

    print("✓ Three allocation strategies select winners by different criteria")

    # --- Section 2: Basic Auction ---
    print("\n--- Section 2: Basic Auction ---")

    emitter = make_emitter("bidding-s2")

    # Three specialist agents with FixedBidGenerators
    math_agent = ReActAgent(
        name="math-expert",
        llm_client=MockLLMClient(
            responses=[
                make_response("The integral of x² is x³/3 + C."),
            ]
        ),
        emitter=emitter,
        system_prompt="You are a mathematics expert.",
        tools=[],
    )

    writing_agent = ReActAgent(
        name="writing-expert",
        llm_client=MockLLMClient(
            responses=[
                make_response("A creative essay about mathematics."),
            ]
        ),
        emitter=emitter,
        system_prompt="You are a creative writer.",
        tools=[],
    )

    research_agent = ReActAgent(
        name="research-expert",
        llm_client=MockLLMClient(
            responses=[
                make_response("Research findings on calculus history."),
            ]
        ),
        emitter=emitter,
        system_prompt="You are a research analyst.",
        tools=[],
    )

    participants = [
        BiddableAgent(
            agent=math_agent,
            bid_generator=FixedBidGenerator(
                confidence=0.9,
                capabilities=["algebra", "calculus"],
                estimated_cost=0.01,
            ),
        ),
        BiddableAgent(
            agent=writing_agent,
            bid_generator=FixedBidGenerator(
                confidence=0.3,
                capabilities=["creative-writing"],
                estimated_cost=0.02,
            ),
        ),
        BiddableAgent(
            agent=research_agent,
            bid_generator=FixedBidGenerator(
                confidence=0.7,
                capabilities=["analysis", "search"],
                estimated_cost=0.015,
            ),
        ),
    ]

    bidding = Bidding(
        participants=participants,
        emitter=emitter,
        allocation_strategy=HighestConfidence(),
    )

    result = await bidding.run("Solve this integral: ∫x²dx")

    # Math expert wins — highest confidence (0.9)
    assert result.allocated is True
    assert result.winning_bid is not None
    assert result.winning_bid.agent_name == "math-expert"
    assert result.winning_bid.confidence == 0.9
    assert len(result.all_bids) == 3

    # Winning agent executed the task
    assert result.execution_result == "The integral of x² is x³/3 + C."

    print(f"  Winner: {result.winning_bid.agent_name} (confidence={result.winning_bid.confidence})")
    print(f"  Total bids: {len(result.all_bids)}")
    print(f"  Execution result: {result.execution_result}")
    print(f"  Allocated: {result.allocated}")
    print("✓ Highest-confidence agent won the auction and executed the task")

    # --- Section 3: Minimum Bid Threshold ---
    print("\n--- Section 3: Minimum Bid Threshold ---")

    emitter = make_emitter("bidding-s3")

    # Two agents with low confidence — both below the threshold
    low_conf_a = ReActAgent(
        name="uncertain-a",
        llm_client=MockLLMClient(
            responses=[
                make_response("Maybe something about integrals?"),
            ]
        ),
        emitter=emitter,
        system_prompt="Generalist agent.",
        tools=[],
    )

    low_conf_b = ReActAgent(
        name="uncertain-b",
        llm_client=MockLLMClient(
            responses=[
                make_response("I think it involves calculus."),
            ]
        ),
        emitter=emitter,
        system_prompt="Another generalist.",
        tools=[],
    )

    participants = [
        BiddableAgent(
            agent=low_conf_a,
            bid_generator=FixedBidGenerator(confidence=0.3, capabilities=["general"]),
        ),
        BiddableAgent(
            agent=low_conf_b,
            bid_generator=FixedBidGenerator(confidence=0.4, capabilities=["general"]),
        ),
    ]

    bidding = Bidding(
        participants=participants,
        emitter=emitter,
        allocation_strategy=HighestConfidence(),
        min_bid_threshold=0.5,
    )

    result = await bidding.run("Prove the Riemann hypothesis")

    # No winner — all bids below threshold
    assert result.allocated is False
    assert result.winning_bid is None
    assert result.execution_result is None
    assert len(result.all_bids) == 2

    print(f"  Allocated: {result.allocated}")
    print(f"  Winning bid: {result.winning_bid}")
    print(f"  Execution result: {result.execution_result}")
    print(f"  Bids received: {len(result.all_bids)} (both below threshold 0.5)")
    print("✓ No agent allocated when all bids fall below min_bid_threshold")

    # --- Section 4: Event Trace ---
    print("\n--- Section 4: Event Trace ---")

    emitter = make_emitter("bidding-s4")

    # Reuse the Section 2 scenario for event inspection
    math_agent = ReActAgent(
        name="math-expert",
        llm_client=MockLLMClient(
            responses=[
                make_response("x³/3 + C"),
            ]
        ),
        emitter=emitter,
        system_prompt="Math expert.",
        tools=[],
    )

    writing_agent = ReActAgent(
        name="writing-expert",
        llm_client=MockLLMClient(
            responses=[
                make_response("An essay about math."),
            ]
        ),
        emitter=emitter,
        system_prompt="Writer.",
        tools=[],
    )

    research_agent = ReActAgent(
        name="research-expert",
        llm_client=MockLLMClient(
            responses=[
                make_response("Research on calculus."),
            ]
        ),
        emitter=emitter,
        system_prompt="Researcher.",
        tools=[],
    )

    participants = [
        BiddableAgent(
            agent=math_agent,
            bid_generator=FixedBidGenerator(confidence=0.9, capabilities=["calculus"]),
        ),
        BiddableAgent(
            agent=writing_agent,
            bid_generator=FixedBidGenerator(confidence=0.3, capabilities=["writing"]),
        ),
        BiddableAgent(
            agent=research_agent,
            bid_generator=FixedBidGenerator(confidence=0.7, capabilities=["analysis"]),
        ),
    ]

    bidding = Bidding(
        participants=participants,
        emitter=emitter,
        allocation_strategy=HighestConfidence(),
    )

    result = await bidding.run("Solve ∫x²dx")

    # BiddingStartEvent — auction began
    start_events = [e for e in emitter.events if isinstance(e, BiddingStartEvent)]
    assert len(start_events) == 1
    assert start_events[0].task == "Solve ∫x²dx"
    assert set(start_events[0].participant_names) == {"math-expert", "writing-expert", "research-expert"}
    print(f"  BiddingStartEvent: task={start_events[0].task!r}, participants={start_events[0].participant_names}")

    # BidReceivedEvent — one per participant
    bid_events = [e for e in emitter.events if isinstance(e, BidReceivedEvent)]
    assert len(bid_events) == 3
    bid_names = {e.agent_name for e in bid_events}
    assert bid_names == {"math-expert", "writing-expert", "research-expert"}
    print(f"  BidReceivedEvent: {len(bid_events)} bids from {sorted(bid_names)}")

    # BidAllocatedEvent — winner selected
    alloc_events = [e for e in emitter.events if isinstance(e, BidAllocatedEvent)]
    assert len(alloc_events) == 1
    assert alloc_events[0].winner == "math-expert"
    assert alloc_events[0].total_bids == 3
    assert alloc_events[0].rejection_reason is None
    print(f"  BidAllocatedEvent: winner={alloc_events[0].winner}, total_bids={alloc_events[0].total_bids}")

    # BiddingCompleteEvent — execution finished
    complete_events = [e for e in emitter.events if isinstance(e, BiddingCompleteEvent)]
    assert len(complete_events) == 1
    assert complete_events[0].winner == "math-expert"
    assert complete_events[0].allocated is True
    assert complete_events[0].total_participants == 3
    print(f"  BiddingCompleteEvent: winner={complete_events[0].winner}, allocated={complete_events[0].allocated}")

    print("✓ Full event trace: start → bids → allocation → complete")

    # --- Section 5: LLM-Driven Bidding ---
    print("\n--- Section 5: LLM-Driven Bidding ---")

    # LLMBidGenerator uses an LLM to dynamically assess agent suitability.
    # Each agent's LLM receives the task + agent description and returns a
    # structured bid (confidence, capabilities, cost, reasoning).
    emitter = make_emitter("bidding-s5")

    data_scientist = ReActAgent(
        name="data-scientist",
        llm_client=MockLLMClient(
            responses=[
                make_response("Churn analysis: 23% of users churn within 90 days."),
            ]
        ),
        emitter=emitter,
        system_prompt="You are a data scientist.",
        tools=[],
    )

    copywriter = ReActAgent(
        name="copywriter",
        llm_client=MockLLMClient(
            responses=[
                make_response("Here's a catchy headline about churn."),
            ]
        ),
        emitter=emitter,
        system_prompt="You are a copywriter.",
        tools=[],
    )

    # Each LLMBidGenerator gets its own MockLLMClient that returns a JSON bid.
    # The data scientist's LLM bids high confidence for an analytical task.
    ds_bid_client = MockLLMClient(
        responses=[
            make_response(
                json.dumps(
                    {
                        "confidence": 0.92,
                        "capabilities": ["statistics", "machine-learning", "data-analysis"],
                        "estimated_cost": 0.05,
                        "reasoning": "This is a data analysis task that matches my expertise.",
                    }
                )
            ),
        ]
    )

    # The copywriter's LLM bids low confidence — wrong domain.
    cw_bid_client = MockLLMClient(
        responses=[
            make_response(
                json.dumps(
                    {
                        "confidence": 0.15,
                        "capabilities": ["writing"],
                        "estimated_cost": 0.03,
                        "reasoning": "This requires data skills I don't have.",
                    }
                )
            ),
        ]
    )

    participants = [
        BiddableAgent(
            agent=data_scientist,
            bid_generator=LLMBidGenerator(
                llm_client=ds_bid_client,
                agent_description="Data scientist skilled in statistics, ML, and churn analysis.",
            ),
        ),
        BiddableAgent(
            agent=copywriter,
            bid_generator=LLMBidGenerator(
                llm_client=cw_bid_client,
                agent_description="Copywriter for marketing headlines and blog posts.",
            ),
        ),
    ]

    bidding = Bidding(
        participants=participants,
        emitter=emitter,
        allocation_strategy=HighestConfidence(),
    )

    result = await bidding.run("Analyze customer churn patterns")

    # Data scientist wins — LLM assessed high confidence for this task
    assert result.allocated is True
    assert result.winning_bid is not None
    assert result.winning_bid.agent_name == "data-scientist"
    assert result.winning_bid.confidence == 0.92
    assert "statistics" in result.winning_bid.capabilities
    assert result.winning_bid.reasoning == "This is a data analysis task that matches my expertise."
    assert result.execution_result == "Churn analysis: 23% of users churn within 90 days."

    # Copywriter's bid was recorded but lost
    cw_bid = next(b for b in result.all_bids if b.agent_name == "copywriter")
    assert cw_bid.confidence == 0.15
    assert cw_bid.reasoning == "This requires data skills I don't have."

    # Verify LLM received the agent description and task in its prompt
    assert len(ds_bid_client.calls) == 1
    bid_prompt = ds_bid_client.calls[0]["messages"][0].content
    assert "data-scientist" in bid_prompt
    assert "Analyze customer churn patterns" in bid_prompt
    assert "churn analysis" in bid_prompt  # agent_description passed through

    print(
        f"  Data scientist bid: confidence={result.winning_bid.confidence}, "
        f"capabilities={result.winning_bid.capabilities}"
    )
    print(f"  Copywriter bid: confidence={cw_bid.confidence}, reasoning={cw_bid.reasoning!r}")
    print(f"  Winner: {result.winning_bid.agent_name}")
    print(f"  Result: {result.execution_result}")
    print("✓ LLMBidGenerator produced dynamic bids based on task + agent description")

    # --- Section 6: Calibration Anchors + Tiebreaker Chain ---
    print("\n--- Section 6: Calibration Anchors + Tiebreaker Chain ---")

    # 6a. The default calibration template counters self-overclaim by
    # anchoring confidence against four bands (0.9 unique / 0.7 capable /
    # 0.4 adjacent / 0.0 out of scope) — independent self-rating without
    # anchors tends to converge on uniformly high scores.
    for anchor in (
        "0.9 = uniquely positioned",
        "0.7 = capable",
        "0.4 = adjacent",
        "0.0 = out of scope",
    ):
        assert anchor in DEFAULT_CALIBRATED_BID_PROMPT_TEMPLATE, (
            f"Anchor missing from default calibrated bid template: {anchor!r}"
        )

    # Verify the anchors actually reach the LLM call.
    emitter = make_emitter("bidding-s6-anchors")
    calibrated_agent = ReActAgent(
        name="billing-specialist",
        llm_client=MockLLMClient(responses=[make_response("Refund issued.")]),
        emitter=emitter,
        system_prompt="Billing.",
        tools=[],
    )
    calibrated_bid_client = MockLLMClient(
        responses=[
            make_response(
                json.dumps(
                    {
                        "confidence": 0.9,
                        "capabilities": ["invoices"],
                        "estimated_cost": 0.04,
                        "reasoning": "0.9 — uniquely positioned for invoice disputes.",
                    }
                )
            ),
        ]
    )
    bidding = Bidding(
        participants=[
            BiddableAgent(
                agent=calibrated_agent,
                bid_generator=LLMBidGenerator(
                    llm_client=calibrated_bid_client,
                    agent_description="Billing specialist for invoices and refunds.",
                    bid_prompt_template=DEFAULT_CALIBRATED_BID_PROMPT_TEMPLATE,
                ),
            ),
        ],
        emitter=emitter,
        allocation_strategy=HighestConfidence(),
    )
    await bidding.run("My invoice shows the wrong amount.")

    rendered_prompt = calibrated_bid_client.calls[0]["messages"][0].content
    for anchor in ("0.9 = uniquely positioned", "0.7 = capable", "0.4 = adjacent", "0.0 = out of scope"):
        assert anchor in rendered_prompt, f"Anchor not rendered into bid prompt: {anchor!r}"
    print("  Calibration anchors rendered into the bid LLM call")

    # 6b. HighestConfidence(tiebreaker=...) breaks strict-tie wins
    # deterministically. Without a tiebreaker, first-listed wins on tie —
    # a footgun when ordering is incidental. Chain LowestCost so equal
    # confidence resolves to the cheaper bid.
    tied_bids = [
        Bid(agent_name="first-listed", confidence=0.9, capabilities=["a"], estimated_cost=0.05, reasoning="Tie."),
        Bid(agent_name="cheapest", confidence=0.9, capabilities=["a"], estimated_cost=0.01, reasoning="Tie."),
        Bid(agent_name="third", confidence=0.9, capabilities=["a"], estimated_cost=0.03, reasoning="Tie."),
    ]

    # Default behaviour — first-listed wins on tie. Pinned for regression.
    legacy_winner = HighestConfidence().select(tied_bids)
    assert legacy_winner is not None
    assert legacy_winner.agent_name == "first-listed"
    print(f"  No tiebreaker: winner={legacy_winner.agent_name} (first-listed wins on strict tie)")

    # With LowestCost as tiebreaker, the cheapest tied bid wins.
    cheap_winner = HighestConfidence(tiebreaker=LowestCost()).select(tied_bids)
    assert cheap_winner is not None
    assert cheap_winner.agent_name == "cheapest"
    assert cheap_winner.estimated_cost == 0.01
    print(f"  LowestCost tiebreaker: winner={cheap_winner.agent_name} (cost={cheap_winner.estimated_cost})")
    print("✓ Calibration anchors counter self-overclaim; tiebreaker chain replaces first-listed-wins")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
