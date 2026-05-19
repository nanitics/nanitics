"""Consensus with three real voters and MajorityVoting.

Three ``ReActAgent`` instances independently answer a fact-recall
question (``"What is the capital of France?"``). Each is prompted to
answer with a single word. ``MajorityVoting`` aggregates the responses.
A one-round ``DeliberationConfig`` with a 2/3 agreement threshold is
used so the aggregation path emits a ``ConsensusAgreementEvent`` when
the voters converge — exercising both the vote collection and the
agreement-detection code paths in a single run.

Acceptance criteria:
  - Aggregation result contains the substring ``"Paris"`` (fuzzy —
    real providers may return ``"Paris"``, ``"Paris."``, etc.).
  - Aggregation ``agreement_level >= 0.66`` (at least 2/3 majority).
  - Aggregation ``strategy == "MajorityVoting"``.
  - ``vote_distribution`` sums to 3 and its max group size is ``>= 2``
    (proving grouping by equality produced a real majority).
  - ``result.termination_reason == "agreement_reached"`` (proves the
    agreement-detection short-circuit fired, not the max-rounds fallback).
  - ``result.rounds_completed == 1`` and ``result.agents_participated == 3``.
  - Exactly three ``ConsensusVoteEvent`` instances, one per distinct
    voter (``{"expert_1","expert_2","expert_3"}``), all at ``round == 1``.
  - ``ConsensusAgreementEvent`` with ``converged=True`` and ``round == 1``
    emitted by deliberation.
  - ``ConsensusCompleteEvent`` with ``agents_participated == 3`` and
    ``termination_reason == "agreement_reached"``.
"""

from __future__ import annotations

import pytest

from nanitics.infrastructure import (
    ConsensusAgreementEvent,
    ConsensusCompleteEvent,
    ConsensusVoteEvent,
)
from nanitics.specialized import (
    Consensus,
    DeliberationConfig,
    MajorityVoting,
)
from nanitics.strategies import ReActAgent
from nanitics.tracing import InMemoryEmitter
from validation.helpers import (
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)


@pytest.mark.quick
async def test_consensus_majority_voting(traced_emitter: InMemoryEmitter) -> None:
    client = make_llm_client("anthropic")

    system_prompt = (
        "Answer the question with exactly one word — the name of the entity. "
        "No punctuation, no explanation, no trailing period."
    )

    agents = [
        ReActAgent(
            name=f"expert_{i}",
            llm_client=client,
            emitter=traced_emitter,
            system_prompt=system_prompt,
            tools=[],
            max_iterations=2,
        )
        for i in (1, 2, 3)
    ]

    consensus = Consensus(
        agents=agents,
        emitter=traced_emitter,
        aggregation_strategy=MajorityVoting(),
        deliberation=DeliberationConfig(
            max_rounds=1,
            agreement_threshold=0.66,
        ),
    )

    result = await run_with_retry(
        lambda: consensus.run("What is the capital of France?"),
        max_attempts=2,
    )

    # --- Aggregation invariants (fuzzy result check, not exact pinning) ---
    assert "Paris" in str(result.aggregation.result), (
        f"Expected aggregation.result to contain 'Paris', got: {result.aggregation.result!r}"
    )
    assert result.aggregation.agreement_level >= 0.66, (
        f"Expected agreement_level >= 0.66, got: {result.aggregation.agreement_level}"
    )
    assert result.aggregation.strategy == "MajorityVoting", (
        f"Expected strategy='MajorityVoting', got: {result.aggregation.strategy!r}"
    )

    # --- vote_distribution proves grouping-by-equality yielded a real majority ---
    vote_distribution = result.aggregation.vote_distribution
    assert sum(vote_distribution.values()) == 3, f"Expected vote_distribution to sum to 3, got: {vote_distribution}"
    assert max(vote_distribution.values()) >= 2, (
        f"Expected max group size >= 2 (a real majority), got: {vote_distribution}"
    )

    # --- Termination path: agreement short-circuit, not max-rounds fallback ---
    assert result.termination_reason == "agreement_reached", (
        f"Expected termination_reason='agreement_reached' (proves the agreement "
        f"short-circuit fired), got: {result.termination_reason!r}"
    )
    assert result.rounds_completed == 1, f"Expected rounds_completed == 1, got: {result.rounds_completed}"
    assert result.agents_participated == 3, f"Expected agents_participated == 3, got: {result.agents_participated}"

    # --- Vote events: exactly 3, one per named voter, all round 1 ---
    vote_events = [e for e in traced_emitter.events if isinstance(e, ConsensusVoteEvent)]
    assert len(vote_events) == 3, f"Expected exactly 3 ConsensusVoteEvent instances, got: {len(vote_events)}"
    assert {e.agent_name for e in vote_events} == {"expert_1", "expert_2", "expert_3"}, (
        f"Expected one vote per distinct voter, got: {[e.agent_name for e in vote_events]}"
    )
    assert all(e.round == 1 for e in vote_events), (
        f"Expected all vote events at round 1, got: {[e.round for e in vote_events]}"
    )

    # --- Agreement event proves convergence was detected on round 1 ---
    assert_trace_contains(
        traced_emitter,
        ConsensusAgreementEvent,
        predicate=lambda e: e.converged and e.round == 1,
    )

    # --- Complete event ties config to outcome ---
    assert_trace_contains(
        traced_emitter,
        ConsensusCompleteEvent,
        predicate=lambda e: (
            e.agents_participated == 3 and e.termination_reason == "agreement_reached" and e.rounds_completed == 1
        ),
    )
