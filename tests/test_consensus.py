"""Tests for Consensus: models, aggregation strategies, controller, events."""

import pytest
from pydantic import ValidationError

from nanitics.composition.multi_agent.consensus import (
    AggregationStrategy,
    BestOfN,
    Consensus,
    ConsensusAggregation,
    ConsensusResponse,
    ConsensusResult,
    DeliberationConfig,
    MajorityVoting,
    WeightedVoting,
    _default_agreement,
    _format_peer_responses,
)
from nanitics.infrastructure import MockLLMClient
from nanitics.infrastructure.observability.events import (
    ConsensusAgreementEvent,
    ConsensusCompleteEvent,
    ConsensusStartEvent,
    ConsensusVoteEvent,
)
from nanitics.strategies import ReActAgent
from nanitics.tracing import InMemoryEmitter
from tests.testing_helpers import make_emitter, make_response


def make_agent(
    name: str,
    emitter: InMemoryEmitter,
    response_content: str = "done",
    num_responses: int = 1,
) -> ReActAgent:
    return ReActAgent(
        name=name,
        llm_client=MockLLMClient([make_response(response_content)] * num_responses),
        emitter=emitter,
        system_prompt=f"You are {name}.",
        tools=[],
    )


def make_failing_agent(name: str, emitter: InMemoryEmitter) -> ReActAgent:
    return ReActAgent(
        name=name,
        llm_client=MockLLMClient([]),
        emitter=emitter,
        system_prompt=f"You are {name}.",
        tools=[],
    )


# ──────────────────────────────────────────────────────────
# Model Tests
# ──────────────────────────────────────────────────────────


class TestModels:
    def test_consensus_response_frozen(self) -> None:
        resp = ConsensusResponse(agent_name="a", output="out", round=1, steps=1, termination_reason="done")
        with pytest.raises(ValidationError):
            resp.output = "new"

    def test_consensus_aggregation_frozen(self) -> None:
        agg = ConsensusAggregation(result="r", agreement_level=1.0, vote_distribution={"r": 1}, strategy="s")
        with pytest.raises(ValidationError):
            agg.result = "new"

    def test_consensus_result_frozen(self) -> None:
        result = ConsensusResult(
            aggregation=ConsensusAggregation(
                result="r",
                agreement_level=1.0,
                vote_distribution={"r": 1},
                strategy="s",
            ),
            responses=[],
            rounds_completed=1,
            termination_reason="single_round",
            agents_participated=0,
        )
        with pytest.raises(ValidationError):
            result.rounds_completed = 2

    def test_deliberation_config_frozen(self) -> None:
        cfg = DeliberationConfig(max_rounds=3, agreement_threshold=0.8)
        with pytest.raises(ValidationError):
            cfg.max_rounds = 5

    def test_consensus_response_fields(self) -> None:
        resp = ConsensusResponse(agent_name="bot", output="answer", round=2, steps=3, termination_reason="end")
        assert resp.agent_name == "bot"
        assert resp.output == "answer"
        assert resp.round == 2
        assert resp.steps == 3
        assert resp.termination_reason == "end"

    def test_consensus_aggregation_fields(self) -> None:
        agg = ConsensusAggregation(
            result="winner",
            agreement_level=0.67,
            vote_distribution={"winner": 2, "other": 1},
            strategy="MajorityVoting",
        )
        assert agg.result == "winner"
        assert agg.agreement_level == 0.67
        assert agg.vote_distribution == {"winner": 2, "other": 1}
        assert agg.strategy == "MajorityVoting"

    def test_deliberation_config_defaults(self) -> None:
        cfg = DeliberationConfig()
        assert cfg.max_rounds == 3
        assert cfg.agreement_threshold == 1.0
        assert cfg.agreement_fn is None
        assert cfg.fallback_strategy is None


# ──────────────────────────────────────────────────────────
# Protocol Conformance
# ──────────────────────────────────────────────────────────


class TestProtocols:
    def test_majority_voting_satisfies_protocol(self) -> None:
        assert isinstance(MajorityVoting(), AggregationStrategy)

    def test_weighted_voting_satisfies_protocol(self) -> None:
        assert isinstance(WeightedVoting(weight_fn=lambda r: 1.0), AggregationStrategy)

    def test_best_of_n_satisfies_protocol(self) -> None:
        assert isinstance(BestOfN(scorer=lambda r: 1.0), AggregationStrategy)

    def test_custom_class_satisfies_protocol(self) -> None:
        class CustomAggregation:
            async def aggregate(self, responses: list[ConsensusResponse]) -> ConsensusAggregation:
                return ConsensusAggregation(result="", agreement_level=0.0, vote_distribution={}, strategy="custom")

        assert isinstance(CustomAggregation(), AggregationStrategy)


# ──────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────


class TestHelpers:
    def test_format_peer_responses_excludes_self(self) -> None:
        responses = [
            ConsensusResponse(agent_name="A", output="ans-a", round=1, steps=1, termination_reason="done"),
            ConsensusResponse(agent_name="B", output="ans-b", round=1, steps=1, termination_reason="done"),
            ConsensusResponse(agent_name="C", output="ans-c", round=1, steps=1, termination_reason="done"),
        ]
        result = _format_peer_responses(responses, "B")
        assert "[A]: ans-a" in result
        assert "[C]: ans-c" in result
        assert "[B]" not in result

    def test_format_peer_responses_empty_when_only_self(self) -> None:
        responses = [
            ConsensusResponse(agent_name="A", output="ans-a", round=1, steps=1, termination_reason="done"),
        ]
        result = _format_peer_responses(responses, "A")
        assert result == ""

    def test_default_agreement_all_same(self) -> None:
        responses = [
            ConsensusResponse(agent_name="A", output="same", round=1, steps=1, termination_reason="done"),
            ConsensusResponse(agent_name="B", output="same", round=1, steps=1, termination_reason="done"),
            ConsensusResponse(agent_name="C", output="same", round=1, steps=1, termination_reason="done"),
        ]
        assert _default_agreement(responses) == 1.0

    def test_default_agreement_majority(self) -> None:
        responses = [
            ConsensusResponse(agent_name="A", output="yes", round=1, steps=1, termination_reason="done"),
            ConsensusResponse(agent_name="B", output="yes", round=1, steps=1, termination_reason="done"),
            ConsensusResponse(agent_name="C", output="no", round=1, steps=1, termination_reason="done"),
        ]
        assert abs(_default_agreement(responses) - 2 / 3) < 1e-9

    def test_default_agreement_all_different(self) -> None:
        responses = [
            ConsensusResponse(agent_name="A", output="a", round=1, steps=1, termination_reason="done"),
            ConsensusResponse(agent_name="B", output="b", round=1, steps=1, termination_reason="done"),
            ConsensusResponse(agent_name="C", output="c", round=1, steps=1, termination_reason="done"),
        ]
        assert abs(_default_agreement(responses) - 1 / 3) < 1e-9

    def test_default_agreement_empty(self) -> None:
        assert _default_agreement([]) == 0.0


# ──────────────────────────────────────────────────────────
# MajorityVoting
# ──────────────────────────────────────────────────────────


class TestMajorityVoting:
    async def test_two_agree_out_of_three(self) -> None:
        strategy = MajorityVoting()
        responses = [
            ConsensusResponse(agent_name="A", output="yes", round=1, steps=1, termination_reason="done"),
            ConsensusResponse(agent_name="B", output="yes", round=1, steps=1, termination_reason="done"),
            ConsensusResponse(agent_name="C", output="no", round=1, steps=1, termination_reason="done"),
        ]
        agg = await strategy.aggregate(responses)
        assert agg.result == "yes"
        assert abs(agg.agreement_level - 2 / 3) < 1e-9
        assert agg.vote_distribution == {"yes": 2, "no": 1}
        assert agg.strategy == "MajorityVoting"

    async def test_all_agree(self) -> None:
        strategy = MajorityVoting()
        responses = [
            ConsensusResponse(agent_name="A", output="same", round=1, steps=1, termination_reason="done"),
            ConsensusResponse(agent_name="B", output="same", round=1, steps=1, termination_reason="done"),
            ConsensusResponse(agent_name="C", output="same", round=1, steps=1, termination_reason="done"),
        ]
        agg = await strategy.aggregate(responses)
        assert agg.result == "same"
        assert agg.agreement_level == 1.0

    async def test_all_different(self) -> None:
        strategy = MajorityVoting()
        responses = [
            ConsensusResponse(agent_name="A", output="a", round=1, steps=1, termination_reason="done"),
            ConsensusResponse(agent_name="B", output="b", round=1, steps=1, termination_reason="done"),
            ConsensusResponse(agent_name="C", output="c", round=1, steps=1, termination_reason="done"),
        ]
        agg = await strategy.aggregate(responses)
        # First encountered group wins
        assert agg.result == "a"
        assert abs(agg.agreement_level - 1 / 3) < 1e-9

    async def test_custom_eq_fn(self) -> None:
        # Case-insensitive comparison
        strategy = MajorityVoting(eq_fn=lambda a, b: str(a).lower() == str(b).lower())
        responses = [
            ConsensusResponse(agent_name="A", output="Yes", round=1, steps=1, termination_reason="done"),
            ConsensusResponse(agent_name="B", output="yes", round=1, steps=1, termination_reason="done"),
            ConsensusResponse(agent_name="C", output="No", round=1, steps=1, termination_reason="done"),
        ]
        agg = await strategy.aggregate(responses)
        assert agg.result == "Yes"  # first encountered representative
        assert abs(agg.agreement_level - 2 / 3) < 1e-9


# ──────────────────────────────────────────────────────────
# WeightedVoting
# ──────────────────────────────────────────────────────────


class TestWeightedVoting:
    async def test_highest_weight_wins_despite_fewer_votes(self) -> None:
        # Agent C has weight 10, A and B have weight 1 each
        def weight_fn(r: ConsensusResponse) -> float:
            return 10.0 if r.agent_name == "C" else 1.0

        strategy = WeightedVoting(weight_fn=weight_fn)
        responses = [
            ConsensusResponse(agent_name="A", output="yes", round=1, steps=1, termination_reason="done"),
            ConsensusResponse(agent_name="B", output="yes", round=1, steps=1, termination_reason="done"),
            ConsensusResponse(agent_name="C", output="no", round=1, steps=1, termination_reason="done"),
        ]
        agg = await strategy.aggregate(responses)
        assert agg.result == "no"
        assert agg.agreement_level == 10.0 / 12.0
        assert agg.strategy == "WeightedVoting"

    async def test_equal_weights_degrades_to_majority(self) -> None:
        strategy = WeightedVoting(weight_fn=lambda r: 1.0)
        responses = [
            ConsensusResponse(agent_name="A", output="yes", round=1, steps=1, termination_reason="done"),
            ConsensusResponse(agent_name="B", output="yes", round=1, steps=1, termination_reason="done"),
            ConsensusResponse(agent_name="C", output="no", round=1, steps=1, termination_reason="done"),
        ]
        agg = await strategy.aggregate(responses)
        assert agg.result == "yes"
        assert abs(agg.agreement_level - 2 / 3) < 1e-9


# ──────────────────────────────────────────────────────────
# BestOfN
# ──────────────────────────────────────────────────────────


class TestBestOfN:
    async def test_highest_scored_wins(self) -> None:
        def scorer(r: ConsensusResponse) -> float:
            scores = {"A": 0.5, "B": 0.9, "C": 0.3}
            return scores[r.agent_name]

        strategy = BestOfN(scorer=scorer)
        responses = [
            ConsensusResponse(agent_name="A", output="ans-a", round=1, steps=1, termination_reason="done"),
            ConsensusResponse(agent_name="B", output="ans-b", round=1, steps=1, termination_reason="done"),
            ConsensusResponse(agent_name="C", output="ans-c", round=1, steps=1, termination_reason="done"),
        ]
        agg = await strategy.aggregate(responses)
        assert agg.result == "ans-b"
        assert agg.agreement_level == 1.0
        assert agg.vote_distribution == {"B": 1}
        assert agg.strategy == "BestOfN"

    async def test_async_scorer(self) -> None:
        async def async_scorer(r: ConsensusResponse) -> float:
            return 1.0 if r.agent_name == "C" else 0.0

        strategy = BestOfN(scorer=async_scorer)
        responses = [
            ConsensusResponse(agent_name="A", output="a", round=1, steps=1, termination_reason="done"),
            ConsensusResponse(agent_name="B", output="b", round=1, steps=1, termination_reason="done"),
            ConsensusResponse(agent_name="C", output="c", round=1, steps=1, termination_reason="done"),
        ]
        agg = await strategy.aggregate(responses)
        assert agg.result == "c"


# ──────────────────────────────────────────────────────────
# Consensus Controller — Single Round
# ──────────────────────────────────────────────────────────


class TestConsensusSingleRound:
    def test_requires_at_least_two_agents(self) -> None:
        emitter = make_emitter()
        agent = make_agent("a", emitter)
        with pytest.raises(ValueError, match="at least 2"):
            Consensus(agents=[agent], emitter=emitter)

    async def test_three_agents_majority_voting(self) -> None:
        emitter = make_emitter()
        agent_a = make_agent("A", emitter, "yes")
        agent_b = make_agent("B", emitter, "yes")
        agent_c = make_agent("C", emitter, "no")

        consensus = Consensus(agents=[agent_a, agent_b, agent_c], emitter=emitter)
        result = await consensus.run("What is the answer?")

        assert result.rounds_completed == 1
        assert result.termination_reason == "single_round"
        assert result.aggregation.result == "yes"
        assert abs(result.aggregation.agreement_level - 2 / 3) < 1e-9
        assert result.agents_participated == 3
        assert len(result.responses) == 3

    async def test_agent_failure_still_produces_result(self) -> None:
        emitter = make_emitter()
        agent_a = make_agent("A", emitter, "yes")
        agent_b = make_agent("B", emitter, "yes")
        agent_c = make_failing_agent("C", emitter)

        consensus = Consensus(agents=[agent_a, agent_b, agent_c], emitter=emitter)
        result = await consensus.run("task")

        # C failed but A and B succeeded
        assert result.agents_participated == 2
        assert len(result.responses) == 2
        assert result.aggregation.result == "yes"

    async def test_all_agents_fail(self) -> None:
        emitter = make_emitter()
        agent_a = make_failing_agent("A", emitter)
        agent_b = make_failing_agent("B", emitter)

        consensus = Consensus(agents=[agent_a, agent_b], emitter=emitter)

        # No responses to aggregate — max() on empty groups raises ValueError
        with pytest.raises(ValueError):
            await consensus.run("task")

    async def test_two_agents_minimum(self) -> None:
        emitter = make_emitter()
        agent_a = make_agent("A", emitter, "answer")
        agent_b = make_agent("B", emitter, "answer")

        consensus = Consensus(agents=[agent_a, agent_b], emitter=emitter)
        result = await consensus.run("task")

        assert result.agents_participated == 2
        assert result.aggregation.agreement_level == 1.0


# ──────────────────────────────────────────────────────────
# Consensus Controller — Deliberation
# ──────────────────────────────────────────────────────────


class TestConsensusDeliberation:
    async def test_agreement_reached_in_round_1(self) -> None:
        emitter = make_emitter()
        # All agents agree immediately
        agent_a = make_agent("A", emitter, "same")
        agent_b = make_agent("B", emitter, "same")
        agent_c = make_agent("C", emitter, "same")

        consensus = Consensus(
            agents=[agent_a, agent_b, agent_c],
            emitter=emitter,
            deliberation=DeliberationConfig(max_rounds=3, agreement_threshold=1.0),
        )
        result = await consensus.run("task")

        assert result.rounds_completed == 1
        assert result.termination_reason == "agreement_reached"
        assert result.aggregation.agreement_level == 1.0

    async def test_agreement_reached_in_round_2(self) -> None:
        emitter = make_emitter()
        # Round 1: A="yes", B="no" (no agreement)
        # Round 2: both say "yes" (agreement)
        client_a = MockLLMClient([make_response("yes"), make_response("yes")])
        client_b = MockLLMClient([make_response("no"), make_response("yes")])
        agent_a = ReActAgent(
            name="A",
            llm_client=client_a,
            emitter=emitter,
            system_prompt="You are A.",
            tools=[],
        )
        agent_b = ReActAgent(
            name="B",
            llm_client=client_b,
            emitter=emitter,
            system_prompt="You are B.",
            tools=[],
        )

        consensus = Consensus(
            agents=[agent_a, agent_b],
            emitter=emitter,
            deliberation=DeliberationConfig(max_rounds=3, agreement_threshold=1.0),
        )
        result = await consensus.run("task")

        assert result.rounds_completed == 2
        assert result.termination_reason == "agreement_reached"

    async def test_max_rounds_hit_without_agreement(self) -> None:
        emitter = make_emitter()
        # Agents never agree
        client_a = MockLLMClient([make_response("yes")] * 3)
        client_b = MockLLMClient([make_response("no")] * 3)
        agent_a = ReActAgent(
            name="A",
            llm_client=client_a,
            emitter=emitter,
            system_prompt="You are A.",
            tools=[],
        )
        agent_b = ReActAgent(
            name="B",
            llm_client=client_b,
            emitter=emitter,
            system_prompt="You are B.",
            tools=[],
        )

        consensus = Consensus(
            agents=[agent_a, agent_b],
            emitter=emitter,
            deliberation=DeliberationConfig(max_rounds=3, agreement_threshold=1.0),
        )
        result = await consensus.run("task")

        assert result.rounds_completed == 3
        assert result.termination_reason == "max_rounds"

    async def test_fallback_strategy_used_on_max_rounds(self) -> None:
        emitter = make_emitter()
        client_a = MockLLMClient([make_response("yes")] * 2)
        client_b = MockLLMClient([make_response("no")] * 2)
        agent_a = ReActAgent(
            name="A",
            llm_client=client_a,
            emitter=emitter,
            system_prompt="You are A.",
            tools=[],
        )
        agent_b = ReActAgent(
            name="B",
            llm_client=client_b,
            emitter=emitter,
            system_prompt="You are B.",
            tools=[],
        )

        # Use BestOfN as fallback — always picks first (A)
        def scorer(r: ConsensusResponse) -> float:
            return 1.0 if r.agent_name == "A" else 0.0

        consensus = Consensus(
            agents=[agent_a, agent_b],
            emitter=emitter,
            deliberation=DeliberationConfig(
                max_rounds=2,
                agreement_threshold=1.0,
                fallback_strategy=BestOfN(scorer=scorer),
            ),
        )
        result = await consensus.run("task")

        assert result.termination_reason == "max_rounds"
        assert result.aggregation.result == "yes"
        assert result.aggregation.strategy == "BestOfN"

    async def test_peer_responses_formatted_in_deliberation(self) -> None:
        emitter = make_emitter()
        client_a = MockLLMClient([make_response("ans-a"), make_response("revised-a")])
        client_b = MockLLMClient([make_response("ans-b"), make_response("revised-b")])
        agent_a = ReActAgent(
            name="A",
            llm_client=client_a,
            emitter=emitter,
            system_prompt="You are A.",
            tools=[],
        )
        agent_b = ReActAgent(
            name="B",
            llm_client=client_b,
            emitter=emitter,
            system_prompt="You are B.",
            tools=[],
        )

        consensus = Consensus(
            agents=[agent_a, agent_b],
            emitter=emitter,
            deliberation=DeliberationConfig(max_rounds=2, agreement_threshold=1.0),
        )
        await consensus.run("my task")

        # Round 2: agent A should see B's response, agent B should see A's response
        round2_msg_a = client_a.calls[1]["messages"][0]
        assert "my task" in round2_msg_a.content
        assert "[B]: ans-b" in round2_msg_a.content
        assert "Considering all perspectives" in round2_msg_a.content

        round2_msg_b = client_b.calls[1]["messages"][0]
        assert "[A]: ans-a" in round2_msg_b.content

    async def test_custom_agreement_fn(self) -> None:
        emitter = make_emitter()
        agent_a = make_agent("A", emitter, "close-enough-a")
        agent_b = make_agent("B", emitter, "close-enough-b")

        # Custom agreement: always returns 1.0 (instant convergence)
        def always_agree(responses: list[ConsensusResponse]) -> float:
            return 1.0

        consensus = Consensus(
            agents=[agent_a, agent_b],
            emitter=emitter,
            deliberation=DeliberationConfig(
                max_rounds=3,
                agreement_threshold=1.0,
                agreement_fn=always_agree,
            ),
        )
        result = await consensus.run("task")

        assert result.rounds_completed == 1
        assert result.termination_reason == "agreement_reached"


# ──────────────────────────────────────────────────────────
# Event Emission — Single Round
# ──────────────────────────────────────────────────────────


class TestConsensusEventsSingleRound:
    async def test_events_emitted_in_correct_order(self) -> None:
        emitter = make_emitter()
        agent_a = make_agent("A", emitter, "yes")
        agent_b = make_agent("B", emitter, "no")

        consensus = Consensus(agents=[agent_a, agent_b], emitter=emitter)
        await consensus.run("task")

        consensus_events = [
            e
            for e in emitter.events
            if isinstance(
                e,
                (
                    ConsensusStartEvent,
                    ConsensusVoteEvent,
                    ConsensusCompleteEvent,
                ),
            )
        ]

        # start, 2 votes, complete
        assert len(consensus_events) == 4
        assert isinstance(consensus_events[0], ConsensusStartEvent)
        assert isinstance(consensus_events[1], ConsensusVoteEvent)
        assert isinstance(consensus_events[2], ConsensusVoteEvent)
        assert isinstance(consensus_events[3], ConsensusCompleteEvent)

    async def test_start_event_data(self) -> None:
        emitter = make_emitter()
        agent_a = make_agent("A", emitter, "yes")
        agent_b = make_agent("B", emitter, "no")

        consensus = Consensus(agents=[agent_a, agent_b], emitter=emitter)
        await consensus.run("test task")

        start_events = [e for e in emitter.events if isinstance(e, ConsensusStartEvent)]
        assert len(start_events) == 1
        start = start_events[0]
        assert start.task == "test task"
        assert set(start.agent_names) == {"A", "B"}
        assert start.strategy == "MajorityVoting"
        assert start.deliberation_enabled is False

    async def test_vote_event_data(self) -> None:
        emitter = make_emitter()
        agent_a = make_agent("A", emitter, "my answer")
        agent_b = make_agent("B", emitter, "other answer")

        consensus = Consensus(agents=[agent_a, agent_b], emitter=emitter)
        await consensus.run("task")

        vote_events = [e for e in emitter.events if isinstance(e, ConsensusVoteEvent)]
        assert len(vote_events) == 2
        names = {v.agent_name for v in vote_events}
        assert names == {"A", "B"}
        for v in vote_events:
            assert v.round == 1
            assert v.error is None

    async def test_complete_event_data(self) -> None:
        emitter = make_emitter()
        agent_a = make_agent("A", emitter, "yes")
        agent_b = make_agent("B", emitter, "yes")

        consensus = Consensus(agents=[agent_a, agent_b], emitter=emitter)
        await consensus.run("task")

        complete_events = [e for e in emitter.events if isinstance(e, ConsensusCompleteEvent)]
        assert len(complete_events) == 1
        c = complete_events[0]
        assert c.strategy == "MajorityVoting"
        assert c.rounds_completed == 1
        assert c.final_agreement == 1.0
        assert c.agents_participated == 2
        assert c.termination_reason == "single_round"

    async def test_vote_event_on_failure(self) -> None:
        emitter = make_emitter()
        agent_a = make_agent("A", emitter, "yes")
        agent_b = make_failing_agent("B", emitter)

        consensus = Consensus(agents=[agent_a, agent_b], emitter=emitter)
        await consensus.run("task")

        vote_events = [e for e in emitter.events if isinstance(e, ConsensusVoteEvent)]
        # One success vote, one error vote
        success_votes = [v for v in vote_events if v.error is None]
        error_votes = [v for v in vote_events if v.error is not None]
        assert len(success_votes) >= 1
        assert len(error_votes) >= 1
        assert error_votes[0].agent_name == "B"


# ──────────────────────────────────────────────────────────
# Event Emission — Deliberation
# ──────────────────────────────────────────────────────────


class TestConsensusEventsDeliberation:
    async def test_deliberation_events_include_agreement(self) -> None:
        emitter = make_emitter()
        agent_a = make_agent("A", emitter, "same")
        agent_b = make_agent("B", emitter, "same")

        consensus = Consensus(
            agents=[agent_a, agent_b],
            emitter=emitter,
            deliberation=DeliberationConfig(max_rounds=3, agreement_threshold=1.0),
        )
        await consensus.run("task")

        agreement_events = [e for e in emitter.events if isinstance(e, ConsensusAgreementEvent)]
        assert len(agreement_events) >= 1
        assert agreement_events[0].round == 1
        assert agreement_events[0].agreement_level == 1.0
        assert agreement_events[0].converged is True

    async def test_deliberation_start_event_marks_enabled(self) -> None:
        emitter = make_emitter()
        agent_a = make_agent("A", emitter, "a")
        agent_b = make_agent("B", emitter, "a")

        consensus = Consensus(
            agents=[agent_a, agent_b],
            emitter=emitter,
            deliberation=DeliberationConfig(max_rounds=2, agreement_threshold=1.0),
        )
        await consensus.run("task")

        start_events = [e for e in emitter.events if isinstance(e, ConsensusStartEvent)]
        assert len(start_events) == 1
        assert start_events[0].deliberation_enabled is True

    async def test_multi_round_deliberation_events(self) -> None:
        emitter = make_emitter()
        client_a = MockLLMClient([make_response("yes")] * 3)
        client_b = MockLLMClient([make_response("no")] * 3)
        agent_a = ReActAgent(
            name="A",
            llm_client=client_a,
            emitter=emitter,
            system_prompt="You are A.",
            tools=[],
        )
        agent_b = ReActAgent(
            name="B",
            llm_client=client_b,
            emitter=emitter,
            system_prompt="You are B.",
            tools=[],
        )

        consensus = Consensus(
            agents=[agent_a, agent_b],
            emitter=emitter,
            deliberation=DeliberationConfig(max_rounds=3, agreement_threshold=1.0),
        )
        await consensus.run("task")

        agreement_events = [e for e in emitter.events if isinstance(e, ConsensusAgreementEvent)]
        # 3 rounds → 3 agreement checks
        assert len(agreement_events) == 3
        for i, ae in enumerate(agreement_events):
            assert ae.round == i + 1
            assert ae.converged is False  # never converged
