"""Bidding auction covering three allocation strategies and both bid generators.

Two parametrised tests exercise ``Bidding`` against real agents:

* ``test_bidding_fixed_generator`` — parametrised over
  ``HighestConfidence``, ``LowestCost``, and ``WeightedScore``. Uses
  ``FixedBidGenerator`` so bid values are known and the test can pin the
  expected winner deterministically for each strategy. A real
  ``ReActAgent`` wins and executes the task, proving that the winning
  agent is actually run (not just selected).

* ``test_bidding_llm_generator`` — uses ``LLMBidGenerator`` with a real
  Anthropic client for each participant. The domain mismatch between the
  participants (``data_scientist`` vs ``copywriter``) is strong enough
  that any reasonable judge should bid the data scientist higher for a
  churn-analysis task. Pins that LLM-driven bids feed the same
  allocation path.

Acceptance criteria (fixed — parametrised over strategy):
  - Three ``BidReceivedEvent`` instances with the expected per-participant
    confidence/cost tuples (pins that the SDK emits one event per bid).
  - ``BidAllocatedEvent.winner`` equals the deterministic winner for the
    chosen strategy (catches regressions that e.g. ignore
    ``estimated_cost`` or reverse the comparator).
  - ``BiddingResult.winning_bid.agent_name`` matches
    ``BidAllocatedEvent.winner``.
  - ``BiddingResult.allocated is True`` and ``execution_result`` is
    non-empty (winning agent actually executed).
  - ``all_bids`` has length 3 and agent names match the participants.

Acceptance criteria (LLM generator):
  - ``BiddingStartEvent.participant_names`` lists both participants.
  - Both participants emit ``BidReceivedEvent`` with ``confidence`` in
    ``[0.0, 1.0]`` (proves LLM output flowed through ``_BidSchema``
    clamping).
  - Winner is ``data_scientist`` (fuzzy expectation — any reasonable
    judgment on a churn-analysis task should prefer the data scientist
    over the copywriter; real-provider flakes are handled by
    ``run_with_retry``).
  - ``execution_result`` is non-empty.
"""

from __future__ import annotations

import pytest

from nanitics import (
    BiddableAgent,
    Bidding,
    FixedBidGenerator,
    HighestConfidence,
    InMemoryEmitter,
    LLMBidGenerator,
    LowestCost,
    ReActAgent,
    WeightedScore,
)
from nanitics.infrastructure import (
    BidAllocatedEvent,
    BiddingCompleteEvent,
    BiddingStartEvent,
    BidReceivedEvent,
)
from validation.helpers import (
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)

# Deterministic bid table: keyed on agent name, used to reason about the
# expected winner per allocation strategy.
#   math-expert: confidence 0.9, cost 0.05, 2 capabilities
#   writer:      confidence 0.3, cost 0.02, 1 capability
#   researcher:  confidence 0.7, cost 0.01, 2 capabilities
# HighestConfidence → math-expert
# LowestCost        → researcher (cost 0.01)
# WeightedScore     → researcher (high confidence + lowest cost + tied
#                     max capability count; beats math-expert on cost
#                     normalization and beats writer on confidence)
_BID_TABLE = {
    "math-expert": {"confidence": 0.9, "cost": 0.05, "capabilities": ["algebra", "calculus"]},
    "writer": {"confidence": 0.3, "cost": 0.02, "capabilities": ["prose"]},
    "researcher": {"confidence": 0.7, "cost": 0.01, "capabilities": ["search", "analysis"]},
}

_STRATEGY_CASES = [
    ("HighestConfidence", "math-expert"),
    ("LowestCost", "researcher"),
    ("WeightedScore", "researcher"),
]


def _make_participant(
    name: str,
    system_prompt: str,
    client,
    emitter: InMemoryEmitter,
) -> BiddableAgent:
    bid = _BID_TABLE[name]
    agent = ReActAgent(
        name=name,
        llm_client=client,
        emitter=emitter,
        system_prompt=system_prompt,
        tools=[],
        max_iterations=2,
    )
    return BiddableAgent(
        agent=agent,
        bid_generator=FixedBidGenerator(
            confidence=bid["confidence"],
            capabilities=bid["capabilities"],
            estimated_cost=bid["cost"],
        ),
    )


@pytest.mark.quick
@pytest.mark.parametrize(("strategy_name", "expected_winner"), _STRATEGY_CASES)
async def test_bidding_fixed_generator(
    traced_emitter: InMemoryEmitter,
    strategy_name: str,
    expected_winner: str,
) -> None:
    client = make_llm_client("anthropic")

    participants = [
        _make_participant(
            "math-expert",
            "You are a mathematics expert. Answer concisely in one sentence.",
            client,
            traced_emitter,
        ),
        _make_participant(
            "writer",
            "You are a creative writer. Answer in one sentence.",
            client,
            traced_emitter,
        ),
        _make_participant(
            "researcher",
            "You are a research analyst. Answer concisely in one sentence.",
            client,
            traced_emitter,
        ),
    ]

    strategies = {
        "HighestConfidence": HighestConfidence(),
        "LowestCost": LowestCost(),
        # Balanced weights — with the fixed bid table this picks researcher.
        "WeightedScore": WeightedScore(
            weights={"confidence": 0.4, "cost": 0.4, "capabilities": 0.2},
        ),
    }

    bidding = Bidding(
        participants=participants,
        emitter=traced_emitter,
        allocation_strategy=strategies[strategy_name],
    )

    task = "Summarise the definition of a Riemann integral in one sentence."
    result = await run_with_retry(lambda: bidding.run(task), max_attempts=2)

    # --- Pin BiddingResult fields ---
    assert result.allocated is True, f"Expected allocation to succeed, got: {result.allocated}"
    assert result.winning_bid is not None
    assert result.winning_bid.agent_name == expected_winner, (
        f"Strategy {strategy_name}: expected winner {expected_winner!r}, got: {result.winning_bid.agent_name!r}"
    )
    assert len(result.all_bids) == 3
    assert {b.agent_name for b in result.all_bids} == {"math-expert", "writer", "researcher"}
    assert result.execution_result, f"Winning agent must execute; got execution_result={result.execution_result!r}"
    assert result.execution_error is None

    # --- Trace invariants ---
    assert_trace_contains(
        traced_emitter,
        BiddingStartEvent,
        predicate=lambda e: set(e.participant_names) == {"math-expert", "writer", "researcher"},
    )

    bid_events = [e for e in traced_emitter.events if isinstance(e, BidReceivedEvent)]
    assert len(bid_events) == 3, f"Expected 3 BidReceivedEvent instances, got: {len(bid_events)}"
    for name in ("math-expert", "writer", "researcher"):
        entry = _BID_TABLE[name]
        matching = [e for e in bid_events if e.agent_name == name]
        assert len(matching) == 1, f"Expected one BidReceivedEvent for {name!r}, got: {len(matching)}"
        assert matching[0].confidence == entry["confidence"]
        assert matching[0].estimated_cost == entry["cost"]

    assert_trace_contains(
        traced_emitter,
        BidAllocatedEvent,
        predicate=lambda e: e.winner == expected_winner and e.total_bids == 3,
    )
    assert_trace_contains(
        traced_emitter,
        BiddingCompleteEvent,
        predicate=lambda e: e.winner == expected_winner and e.allocated is True,
    )


async def test_bidding_llm_generator(traced_emitter: InMemoryEmitter) -> None:
    """LLMBidGenerator feeds a real structured bid through the allocation path."""
    client = make_llm_client("anthropic")

    data_scientist = ReActAgent(
        name="data-scientist",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You are a data scientist. Given a churn question, produce a single-sentence analytical answer."
        ),
        tools=[],
        max_iterations=2,
    )
    copywriter = ReActAgent(
        name="copywriter",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=("You are a marketing copywriter. Produce a single-sentence answer in your voice."),
        tools=[],
        max_iterations=2,
    )

    participants = [
        BiddableAgent(
            agent=data_scientist,
            bid_generator=LLMBidGenerator(
                llm_client=client,
                agent_description=(
                    "Data scientist specialised in statistics, machine learning, and customer churn analytics."
                ),
            ),
        ),
        BiddableAgent(
            agent=copywriter,
            bid_generator=LLMBidGenerator(
                llm_client=client,
                agent_description=(
                    "Marketing copywriter focused on headlines and brand "
                    "voice. No analytical or statistical background."
                ),
            ),
        ),
    ]

    bidding = Bidding(
        participants=participants,
        emitter=traced_emitter,
        allocation_strategy=HighestConfidence(),
    )

    task = "Analyse customer churn patterns for a B2B SaaS product."
    result = await run_with_retry(lambda: bidding.run(task), max_attempts=2)

    # --- Start event lists both participants ---
    assert_trace_contains(
        traced_emitter,
        BiddingStartEvent,
        predicate=lambda e: set(e.participant_names) == {"data-scientist", "copywriter"},
    )

    # --- Bids are clamped to [0,1] by the schema path ---
    bid_events = [e for e in traced_emitter.events if isinstance(e, BidReceivedEvent)]
    assert len(bid_events) == 2, f"Expected 2 BidReceivedEvent instances, got: {len(bid_events)}"
    for e in bid_events:
        assert 0.0 <= e.confidence <= 1.0, f"Confidence out of range for {e.agent_name!r}: {e.confidence}"

    # --- Winner is the domain expert ---
    assert result.allocated is True
    assert result.winning_bid is not None
    assert result.winning_bid.agent_name == "data-scientist", (
        f"Expected data-scientist to win churn-analysis task; got: "
        f"{result.winning_bid.agent_name!r} with bids: "
        f"{[(b.agent_name, b.confidence) for b in result.all_bids]}"
    )
    assert result.execution_result, f"Winning agent must execute; got: {result.execution_result!r}"
