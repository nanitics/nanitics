"""Consensus: redundant execution with aggregation and deliberation.

Demonstrates ``Consensus`` — the coordination pattern that gathers independent
responses from multiple agents and aggregates them into a collective decision.
Covers all three built-in aggregation strategies (``MajorityVoting``,
``WeightedVoting``, ``BestOfN``), single-round execution, and multi-round
deliberation with both convergence and fallback outcomes.

Related guide: docs/guides/multi-agent-coordination.md
"""

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics.infrastructure import (
    ConsensusAgreementEvent,
    ConsensusCompleteEvent,
    ConsensusStartEvent,
    ConsensusVoteEvent,
    MockLLMClient,
)
from nanitics.specialized import (
    BestOfN,
    Consensus,
    DeliberationConfig,
    MajorityVoting,
    WeightedVoting,
)
from nanitics.strategies import ReActAgent


async def main() -> None:
    # --- Section 1: MajorityVoting — Single Round ---
    print("--- Section 1: MajorityVoting — Single Round ---")

    # Three agents vote independently. Two say "Paris", one says "Lyon".
    # MajorityVoting selects the response held by the largest group.
    emitter = make_emitter("consensus-s1")

    agents = [
        ReActAgent(
            name="expert_1",
            llm_client=MockLLMClient(responses=[make_response("Paris")]),
            emitter=emitter,
            system_prompt="Answer the question.",
            tools=[],
        ),
        ReActAgent(
            name="expert_2",
            llm_client=MockLLMClient(responses=[make_response("Lyon")]),
            emitter=emitter,
            system_prompt="Answer the question.",
            tools=[],
        ),
        ReActAgent(
            name="expert_3",
            llm_client=MockLLMClient(responses=[make_response("Paris")]),
            emitter=emitter,
            system_prompt="Answer the question.",
            tools=[],
        ),
    ]

    consensus = Consensus(
        agents=agents,
        emitter=emitter,
        aggregation_strategy=MajorityVoting(),
    )

    result = await consensus.run("What is the capital of France?")

    # Result: majority wins
    assert result.aggregation.result == "Paris"
    assert abs(result.aggregation.agreement_level - 2 / 3) < 0.01
    assert result.aggregation.vote_distribution == {"Paris": 2, "Lyon": 1}
    assert result.aggregation.strategy == "MajorityVoting"
    assert result.rounds_completed == 1
    assert result.termination_reason == "single_round"
    assert result.agents_participated == 3
    assert len(result.responses) == 3

    print(f"  Result: {result.aggregation.result}")
    print(f"  Agreement: {result.aggregation.agreement_level:.3f}")
    print(f"  Votes: {result.aggregation.vote_distribution}")
    print("✓ MajorityVoting selected the majority response")

    # Events: start → 3 votes → complete
    start_events = [e for e in emitter.events if isinstance(e, ConsensusStartEvent)]
    assert len(start_events) == 1
    assert set(start_events[0].agent_names) == {"expert_1", "expert_2", "expert_3"}
    assert start_events[0].strategy == "MajorityVoting"
    assert start_events[0].deliberation_enabled is False

    vote_events = [e for e in emitter.events if isinstance(e, ConsensusVoteEvent)]
    assert len(vote_events) == 3
    vote_outputs = {e.output for e in vote_events}
    assert vote_outputs == {"Paris", "Lyon"}

    complete_events = [e for e in emitter.events if isinstance(e, ConsensusCompleteEvent)]
    assert len(complete_events) == 1
    assert abs(complete_events[0].final_agreement - 2 / 3) < 0.01
    assert complete_events[0].termination_reason == "single_round"

    # No agreement events in single-round mode
    agreement_events = [e for e in emitter.events if isinstance(e, ConsensusAgreementEvent)]
    assert len(agreement_events) == 0

    print(f"  Events: {len(start_events)} start, {len(vote_events)} votes, {len(complete_events)} complete")
    print("✓ All event types emitted and verified")

    # --- Section 2: WeightedVoting — Minority Wins via Weight ---
    print("\n--- Section 2: WeightedVoting — Minority Wins via Weight ---")

    # Two agents give short answers ("A"), one gives a detailed answer.
    # WeightedVoting with weight_fn=len(output) lets the detailed response
    # outweigh the numerical majority.
    emitter = make_emitter("consensus-s2")

    agents = [
        ReActAgent(
            name="quick_1",
            llm_client=MockLLMClient(responses=[make_response("A")]),
            emitter=emitter,
            system_prompt="Answer briefly.",
            tools=[],
        ),
        ReActAgent(
            name="quick_2",
            llm_client=MockLLMClient(responses=[make_response("A")]),
            emitter=emitter,
            system_prompt="Answer briefly.",
            tools=[],
        ),
        ReActAgent(
            name="thorough",
            llm_client=MockLLMClient(responses=[make_response("Comprehensive analysis: B")]),
            emitter=emitter,
            system_prompt="Provide detailed analysis.",
            tools=[],
        ),
    ]

    consensus = Consensus(
        agents=agents,
        emitter=emitter,
        aggregation_strategy=WeightedVoting(
            weight_fn=lambda resp: len(resp.output),
        ),
    )

    result = await consensus.run("Choose A or B")

    # Minority by count (1 vs 2), but majority by weight (25 vs 2)
    assert result.aggregation.result == "Comprehensive analysis: B"
    assert result.aggregation.strategy == "WeightedVoting"
    assert result.aggregation.vote_distribution == {"A": 2, "Comprehensive analysis: B": 1}
    assert result.termination_reason == "single_round"

    # Agreement level = best_group_weight / total_weight = 25 / 27
    assert abs(result.aggregation.agreement_level - 25 / 27) < 0.01

    print(f"  Result: {result.aggregation.result}")
    print(f"  Vote counts: {result.aggregation.vote_distribution}")
    print(f"  Agreement (by weight): {result.aggregation.agreement_level:.3f}")
    print("✓ WeightedVoting: minority response won via higher weight")

    # --- Section 3: BestOfN — Scorer Picks Winner ---
    print("\n--- Section 3: BestOfN — Scorer Picks Winner ---")

    # BestOfN scores each response individually and picks the highest.
    # Scorer counts digit characters — the most data-rich answer wins.
    emitter = make_emitter("consensus-s3")

    agents = [
        ReActAgent(
            name="analyst_1",
            llm_client=MockLLMClient(responses=[make_response("Answer with 3 points")]),
            emitter=emitter,
            system_prompt="Analyze.",
            tools=[],
        ),
        ReActAgent(
            name="analyst_2",
            llm_client=MockLLMClient(responses=[make_response("Brief")]),
            emitter=emitter,
            system_prompt="Analyze.",
            tools=[],
        ),
        ReActAgent(
            name="analyst_3",
            llm_client=MockLLMClient(
                responses=[
                    make_response("Analysis: 7 categories, 42 items"),
                ]
            ),
            emitter=emitter,
            system_prompt="Analyze.",
            tools=[],
        ),
    ]

    consensus = Consensus(
        agents=agents,
        emitter=emitter,
        aggregation_strategy=BestOfN(
            scorer=lambda resp: sum(c.isdigit() for c in str(resp.output)),
        ),
    )

    result = await consensus.run("Provide analysis")

    # Agent 3 wins: 3 digits (7, 4, 2) vs 1 digit (3) vs 0 digits
    assert result.aggregation.result == "Analysis: 7 categories, 42 items"
    assert result.aggregation.strategy == "BestOfN"
    assert result.aggregation.agreement_level == 1.0  # BestOfN always returns 1.0
    assert result.aggregation.vote_distribution == {"analyst_3": 1}
    assert result.termination_reason == "single_round"

    print(f"  Result: {result.aggregation.result}")
    print(f"  Strategy: {result.aggregation.strategy}")
    print("✓ BestOfN selected the highest-scoring response")

    # --- Section 4: Deliberation — Agreement Reached ---
    print("\n--- Section 4: Deliberation — Agreement Reached ---")

    # Three agents deliberate across rounds. In round 1 they diverge
    # (2 say Python, 1 says Rust). In round 2, after seeing peers,
    # all converge on Python — unanimous agreement stops deliberation early.
    emitter = make_emitter("consensus-s4")

    agents = [
        ReActAgent(
            name="dev_1",
            llm_client=MockLLMClient(
                responses=[
                    make_response("Python"),  # Round 1
                    make_response("Python"),  # Round 2
                ]
            ),
            emitter=emitter,
            system_prompt="Recommend a language.",
            tools=[],
        ),
        ReActAgent(
            name="dev_2",
            llm_client=MockLLMClient(
                responses=[
                    make_response("Rust"),  # Round 1
                    make_response("Python"),  # Round 2 — revised after seeing peers
                ]
            ),
            emitter=emitter,
            system_prompt="Recommend a language.",
            tools=[],
        ),
        ReActAgent(
            name="dev_3",
            llm_client=MockLLMClient(
                responses=[
                    make_response("Python"),  # Round 1
                    make_response("Python"),  # Round 2
                ]
            ),
            emitter=emitter,
            system_prompt="Recommend a language.",
            tools=[],
        ),
    ]

    consensus = Consensus(
        agents=agents,
        emitter=emitter,
        aggregation_strategy=MajorityVoting(),
        deliberation=DeliberationConfig(
            max_rounds=3,
            agreement_threshold=1.0,  # Require unanimous agreement
        ),
    )

    result = await consensus.run("Best language for data science?")

    # Unanimous in round 2 — stopped early (didn't need round 3)
    assert result.aggregation.result == "Python"
    assert result.aggregation.agreement_level == 1.0
    assert result.rounds_completed == 2
    assert result.termination_reason == "agreement_reached"
    assert len(result.responses) == 6  # 3 agents × 2 rounds

    print(f"  Result: {result.aggregation.result}")
    print(f"  Rounds: {result.rounds_completed} of 3 max")
    print(f"  Termination: {result.termination_reason}")
    print("✓ Agents converged in round 2 — round 3 skipped")

    # Agreement events track convergence progression
    agreement_events = [e for e in emitter.events if isinstance(e, ConsensusAgreementEvent)]
    assert len(agreement_events) == 2
    assert agreement_events[0].round == 1
    assert agreement_events[0].converged is False  # 2/3 < 1.0 threshold
    assert agreement_events[1].round == 2
    assert agreement_events[1].converged is True  # 3/3 = 1.0 ≥ threshold

    print(f"  Round 1 agreement: {agreement_events[0].agreement_level:.3f} (converged={agreement_events[0].converged})")
    print(f"  Round 2 agreement: {agreement_events[1].agreement_level:.3f} (converged={agreement_events[1].converged})")
    print("✓ ConsensusAgreementEvents show convergence progression")

    # --- Section 5: Deliberation — Max Rounds with Fallback ---
    print("\n--- Section 5: Deliberation — Max Rounds with Fallback ---")

    # Three agents who never agree. After exhausting max_rounds,
    # the fallback_strategy (BestOfN) picks the winner instead of
    # the primary strategy (MajorityVoting).
    emitter = make_emitter("consensus-s5")

    agents = [
        ReActAgent(
            name="designer_1",
            llm_client=MockLLMClient(
                responses=[
                    make_response("Red"),  # Round 1
                    make_response("Red"),  # Round 2 — unchanged
                ]
            ),
            emitter=emitter,
            system_prompt="Pick a color.",
            tools=[],
        ),
        ReActAgent(
            name="designer_2",
            llm_client=MockLLMClient(
                responses=[
                    make_response("Blue"),  # Round 1
                    make_response("Blue"),  # Round 2 — unchanged
                ]
            ),
            emitter=emitter,
            system_prompt="Pick a color.",
            tools=[],
        ),
        ReActAgent(
            name="designer_3",
            llm_client=MockLLMClient(
                responses=[
                    make_response("Green"),  # Round 1
                    make_response("Green"),  # Round 2 — unchanged
                ]
            ),
            emitter=emitter,
            system_prompt="Pick a color.",
            tools=[],
        ),
    ]

    consensus = Consensus(
        agents=agents,
        emitter=emitter,
        aggregation_strategy=MajorityVoting(),
        deliberation=DeliberationConfig(
            max_rounds=2,
            agreement_threshold=1.0,
            fallback_strategy=BestOfN(
                scorer=lambda resp: len(resp.output),  # Longest name wins
            ),
        ),
    )

    result = await consensus.run("Choose a brand color")

    # No agreement after 2 rounds — fallback BestOfN picks "Green" (5 chars > 4 > 3)
    assert result.rounds_completed == 2
    assert result.termination_reason == "max_rounds"
    assert result.aggregation.strategy == "BestOfN"  # Fallback was used, not primary
    assert result.aggregation.result == "Green"

    print(f"  Result: {result.aggregation.result}")
    print(f"  Rounds: {result.rounds_completed}")
    print(f"  Termination: {result.termination_reason}")
    print(f"  Strategy used: {result.aggregation.strategy} (fallback, not primary MajorityVoting)")
    print("✓ Max rounds exhausted — fallback strategy selected winner")

    # Both rounds show no convergence
    agreement_events = [e for e in emitter.events if isinstance(e, ConsensusAgreementEvent)]
    assert len(agreement_events) == 2
    assert all(not e.converged for e in agreement_events)

    complete_events = [e for e in emitter.events if isinstance(e, ConsensusCompleteEvent)]
    assert len(complete_events) == 1
    assert complete_events[0].termination_reason == "max_rounds"

    print("  Agreement events: both rounds converged=False")
    print(f"  Complete event: termination_reason={complete_events[0].termination_reason}")
    print("✓ Events confirm deliberation exhausted without convergence")


if __name__ == "__main__":
    asyncio.run(main())
