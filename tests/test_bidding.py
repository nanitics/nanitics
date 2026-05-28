"""Tests for Bidding: models, generators, allocation strategies, controller, events."""

import json

import pytest

from nanitics.composition.multi_agent.bidding import (
    DEFAULT_CALIBRATED_BID_PROMPT_TEMPLATE,
    AllocationStrategy,
    Bid,
    BiddableAgent,
    Bidding,
    BiddingResult,
    BidGenerator,
    FixedBidGenerator,
    HighestConfidence,
    LLMBidGenerator,
    LowestCost,
    WeightedScore,
)
from nanitics.infrastructure import MockLLMClient
from nanitics.infrastructure.observability.events import (
    BidAllocatedEvent,
    BiddingCompleteEvent,
    BiddingStartEvent,
    BidReceivedEvent,
    LLMRequestEvent,
    LLMResponseEvent,
)
from nanitics.safety.cancellation import CancellationToken
from nanitics.strategies import ReActAgent
from nanitics.tracing import InMemoryEmitter
from tests.testing_helpers import make_emitter, make_response


def make_agent(
    name: str,
    emitter: InMemoryEmitter,
    response_content: str = "done",
) -> ReActAgent:
    return ReActAgent(
        name=name,
        llm_client=MockLLMClient([make_response(response_content)]),
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


def make_bid(
    agent_name: str,
    confidence: float = 0.8,
    capabilities: list[str] | None = None,
    estimated_cost: float | None = None,
) -> Bid:
    return Bid(
        agent_name=agent_name,
        confidence=confidence,
        capabilities=capabilities or [],
        estimated_cost=estimated_cost,
        reasoning="test reasoning",
    )


# ──────────────────────────────────────────────────────────
# Protocol Conformance
# ──────────────────────────────────────────────────────────


class TestProtocols:
    def test_fixed_bid_generator_satisfies_protocol(self) -> None:
        assert isinstance(FixedBidGenerator(confidence=0.5), BidGenerator)

    def test_highest_confidence_satisfies_protocol(self) -> None:
        assert isinstance(HighestConfidence(), AllocationStrategy)

    def test_lowest_cost_satisfies_protocol(self) -> None:
        assert isinstance(LowestCost(), AllocationStrategy)

    def test_weighted_score_satisfies_protocol(self) -> None:
        assert isinstance(WeightedScore(weights={}), AllocationStrategy)


# ──────────────────────────────────────────────────────────
# Allocation Strategies
# ──────────────────────────────────────────────────────────


class TestHighestConfidence:
    def test_selects_highest(self) -> None:
        bids = [make_bid("a", 0.3), make_bid("b", 0.9), make_bid("c", 0.6)]
        result = HighestConfidence().select(bids)
        assert result is not None
        assert result.agent_name == "b"

    def test_ties_broken_by_order(self) -> None:
        bids = [make_bid("first", 0.8), make_bid("second", 0.8)]
        result = HighestConfidence().select(bids)
        assert result is not None
        assert result.agent_name == "first"

    def test_empty_bids_returns_none(self) -> None:
        assert HighestConfidence().select([]) is None

    def test_tiebreaker_lowest_cost_picks_cheaper(self) -> None:
        bids = [
            make_bid("expensive", 0.9, estimated_cost=0.05),
            make_bid("cheap", 0.9, estimated_cost=0.01),
            make_bid("loser", 0.5, estimated_cost=0.001),
        ]
        result = HighestConfidence(tiebreaker=LowestCost()).select(bids)
        assert result is not None
        assert result.agent_name == "cheap"

    def test_tiebreaker_lowest_cost_falls_through_when_no_costs(self) -> None:
        # LowestCost on tied bids that all lack estimated_cost returns None;
        # HighestConfidence should fall through to the first tied bid.
        bids = [make_bid("first", 0.9), make_bid("second", 0.9)]
        result = HighestConfidence(tiebreaker=LowestCost()).select(bids)
        assert result is not None
        assert result.agent_name == "first"

    def test_tiebreaker_chain_lowest_cost_then_capability_count(self) -> None:
        # Three bids tied at 0.9 confidence, two of those tied at the cheapest
        # cost; the inner LowestCost tiebreaker resolves the cost tie via a
        # capability-count strategy.
        class _MostCapabilities:
            def select(self, bids: list[Bid]) -> Bid | None:
                if not bids:
                    return None
                return max(bids, key=lambda b: len(b.capabilities))

        bids = [
            make_bid("tied_cheap_few", 0.9, capabilities=["a"], estimated_cost=0.01),
            make_bid("tied_cheap_many", 0.9, capabilities=["a", "b", "c"], estimated_cost=0.01),
            make_bid("expensive", 0.9, capabilities=["a", "b", "c", "d"], estimated_cost=0.10),
        ]
        chained = HighestConfidence(
            tiebreaker=LowestCost(tiebreaker=_MostCapabilities()),
        )
        result = chained.select(bids)
        assert result is not None
        assert result.agent_name == "tied_cheap_many"

    def test_no_tiebreaker_preserves_first_listed_wins_on_tie(self) -> None:
        # Regression: with tiebreaker=None, behaviour matches the legacy
        # first-listed-wins-on-tie semantics so existing adopters see no
        # behavioural change.
        bids = [
            make_bid("first", 0.9, estimated_cost=0.10),
            make_bid("second", 0.9, estimated_cost=0.01),
        ]
        result = HighestConfidence().select(bids)
        assert result is not None
        assert result.agent_name == "first"


class TestLowestCost:
    def test_inner_tiebreaker_returning_none_falls_through(self) -> None:
        # When LowestCost's inner tiebreaker returns None for the cost-tied
        # subset, LowestCost falls through to the first tied bid so callers
        # always get a winner when costed bids exist.
        class _ReturnsNone:
            def select(self, bids: list[Bid]) -> Bid | None:
                del bids
                return None

        bids = [
            make_bid("first", 0.5, estimated_cost=0.01),
            make_bid("second", 0.5, estimated_cost=0.01),
        ]
        result = LowestCost(tiebreaker=_ReturnsNone()).select(bids)
        assert result is not None
        assert result.agent_name == "first"

    def test_selects_lowest_cost(self) -> None:
        bids = [
            make_bid("a", estimated_cost=100.0),
            make_bid("b", estimated_cost=50.0),
            make_bid("c", estimated_cost=75.0),
        ]
        result = LowestCost().select(bids)
        assert result is not None
        assert result.agent_name == "b"

    def test_excludes_bids_without_cost(self) -> None:
        bids = [
            make_bid("no_cost"),  # no estimated_cost
            make_bid("has_cost", estimated_cost=100.0),
        ]
        result = LowestCost().select(bids)
        assert result is not None
        assert result.agent_name == "has_cost"

    def test_all_without_cost_returns_none(self) -> None:
        bids = [make_bid("a"), make_bid("b")]
        assert LowestCost().select(bids) is None

    def test_empty_bids_returns_none(self) -> None:
        assert LowestCost().select([]) is None


class TestWeightedScore:
    def test_confidence_weighted(self) -> None:
        bids = [make_bid("low", 0.2), make_bid("high", 0.9)]
        result = WeightedScore(weights={"confidence": 1.0}).select(bids)
        assert result is not None
        assert result.agent_name == "high"

    def test_cost_weighted_lower_is_better(self) -> None:
        bids = [
            make_bid("expensive", 0.5, estimated_cost=100.0),
            make_bid("cheap", 0.5, estimated_cost=10.0),
        ]
        result = WeightedScore(weights={"cost": 1.0}).select(bids)
        assert result is not None
        assert result.agent_name == "cheap"

    def test_capabilities_weighted(self) -> None:
        bids = [
            make_bid("few", 0.5, capabilities=["a"]),
            make_bid("many", 0.5, capabilities=["a", "b", "c"]),
        ]
        result = WeightedScore(weights={"capabilities": 1.0}).select(bids)
        assert result is not None
        assert result.agent_name == "many"

    def test_composite_scoring(self) -> None:
        bids = [
            make_bid("a", 0.9, capabilities=["x"], estimated_cost=100.0),
            make_bid("b", 0.7, capabilities=["x", "y", "z"], estimated_cost=10.0),
        ]
        # Heavily weight capabilities and cost over confidence
        result = WeightedScore(weights={"confidence": 0.1, "cost": 0.5, "capabilities": 0.5}).select(bids)
        assert result is not None
        assert result.agent_name == "b"

    def test_empty_bids_returns_none(self) -> None:
        assert WeightedScore(weights={"confidence": 1.0}).select([]) is None

    def test_equal_confidence_normalizes_correctly(self) -> None:
        # When all bids have equal confidence, _normalize returns 1.0 (max_v == min_v branch)
        bids = [make_bid("a", 0.5), make_bid("b", 0.5)]
        result = WeightedScore(weights={"confidence": 1.0}).select(bids)
        # Both score equally — first one wins as the initial best
        assert result is not None


# ──────────────────────────────────────────────────────────
# Bid Generators
# ──────────────────────────────────────────────────────────


class TestFixedBidGenerator:
    async def test_returns_static_bid(self) -> None:
        emitter = make_emitter()
        gen = FixedBidGenerator(confidence=0.7, capabilities=["code"], estimated_cost=50.0)
        bid = await gen.generate("agent-1", "some task", emitter=emitter)

        assert bid.agent_name == "agent-1"
        assert bid.confidence == 0.7
        assert bid.capabilities == ["code"]
        assert bid.estimated_cost == 50.0
        assert bid.reasoning == "Fixed bid"
        # FixedBidGenerator does no work — no events emitted.
        assert emitter.events == []

    async def test_defaults(self) -> None:
        emitter = make_emitter()
        gen = FixedBidGenerator(confidence=0.5)
        bid = await gen.generate("agent-1", "task", emitter=emitter)
        assert bid.capabilities == []
        assert bid.estimated_cost is None


class TestLLMBidGenerator:
    async def test_produces_valid_bid(self) -> None:
        bid_json = json.dumps(
            {
                "confidence": 0.85,
                "capabilities": ["analysis", "coding"],
                "estimated_cost": 25.0,
                "reasoning": "Agent is well-suited for this task.",
            }
        )
        client = MockLLMClient([make_response(bid_json)])
        gen = LLMBidGenerator(llm_client=client, agent_description="A coding agent")
        emitter = make_emitter()
        bid = await gen.generate("coder", "build a feature", emitter=emitter)

        assert bid.agent_name == "coder"
        assert bid.confidence == 0.85
        assert bid.capabilities == ["analysis", "coding"]
        assert bid.estimated_cost == 25.0
        assert len(client.calls) == 1

    async def test_clamps_confidence_above_one(self) -> None:
        bid_json = json.dumps({"confidence": 1.5, "capabilities": [], "estimated_cost": None, "reasoning": "over"})
        client = MockLLMClient([make_response(bid_json)])
        gen = LLMBidGenerator(llm_client=client, agent_description="test")
        bid = await gen.generate("agent", "task", emitter=make_emitter())
        assert bid.confidence == 1.0

    async def test_clamps_confidence_below_zero(self) -> None:
        bid_json = json.dumps({"confidence": -0.3, "capabilities": [], "estimated_cost": None, "reasoning": "under"})
        client = MockLLMClient([make_response(bid_json)])
        gen = LLMBidGenerator(llm_client=client, agent_description="test")
        bid = await gen.generate("agent", "task", emitter=make_emitter())
        assert bid.confidence == 0.0

    async def test_default_prompt_unchanged_for_backward_compatibility(self) -> None:
        """Regression: with no ``bid_prompt_template``, the prompt body is
        identical to the legacy uncalibrated wording, so existing adopters
        see no behavioural change.
        """
        bid_json = json.dumps(
            {
                "confidence": 0.5,
                "capabilities": [],
                "estimated_cost": None,
                "reasoning": "r",
            }
        )
        client = MockLLMClient([make_response(bid_json)])
        gen = LLMBidGenerator(llm_client=client, agent_description="x")
        await gen.generate("a1", "task-text", emitter=make_emitter())

        # Pin the exact prompt body to catch accidental drift.
        expected = (
            "You are evaluating whether agent 'a1' is suitable for a task.\n\n"
            "Agent description: x\n\n"
            "Task: task-text\n\n"
            "Rate your confidence (0.0-1.0), list relevant capabilities, "
            "estimate cost if possible, and explain your reasoning."
        )
        assert client.calls[0]["messages"][0].content == expected

    async def test_calibrated_template_renders_anchors_and_substitutions(self) -> None:
        bid_json = json.dumps(
            {
                "confidence": 0.7,
                "capabilities": ["billing"],
                "estimated_cost": None,
                "reasoning": "r",
            }
        )
        client = MockLLMClient([make_response(bid_json)])
        gen = LLMBidGenerator(
            llm_client=client,
            agent_description="A billing specialist",
            bid_prompt_template=DEFAULT_CALIBRATED_BID_PROMPT_TEMPLATE,
        )
        await gen.generate("billing-specialist", "Refund my invoice", emitter=make_emitter())

        prompt: str = client.calls[0]["messages"][0].content
        # Substitutions land.
        assert "billing-specialist" in prompt
        assert "A billing specialist" in prompt
        assert "Refund my invoice" in prompt
        # Verbatim anchors.
        assert "0.9 = uniquely positioned" in prompt
        assert "0.7 = capable" in prompt
        assert "0.4 = adjacent" in prompt
        assert "0.0 = out of scope" in prompt

    def test_template_missing_placeholder_raises_value_error_at_construction(self) -> None:
        client = MockLLMClient([])
        # Missing {task} placeholder — must fail at __init__, not at generate().
        with pytest.raises(ValueError, match="task"):
            LLMBidGenerator(
                llm_client=client,
                agent_description="x",
                bid_prompt_template="agent {agent_name} desc {agent_description}",
            )

    def test_template_missing_agent_name_placeholder_raises(self) -> None:
        client = MockLLMClient([])
        with pytest.raises(ValueError, match="agent_name"):
            LLMBidGenerator(
                llm_client=client,
                agent_description="x",
                bid_prompt_template="desc {agent_description} task {task}",
            )

    async def test_emits_bid_labelled_llm_events(self) -> None:
        """Every ``LLMBidGenerator.generate`` call emits one ``LLMRequestEvent``
        and one ``LLMResponseEvent`` — both labelled ``"bid"`` — through the
        caller-supplied emitter. This is the telemetry contract that makes
        bid-phase LLM spend visible in a ``Bidding`` run's summary.
        """
        bid_json = json.dumps(
            {
                "confidence": 0.7,
                "capabilities": ["analysis"],
                "estimated_cost": None,
                "reasoning": "Telemetry test",
            }
        )
        client = MockLLMClient([make_response(bid_json)])
        gen = LLMBidGenerator(llm_client=client, agent_description="An analyst")
        emitter = make_emitter()

        await gen.generate("analyst", "assess this", emitter=emitter)

        requests = [e for e in emitter.events if isinstance(e, LLMRequestEvent)]
        responses = [e for e in emitter.events if isinstance(e, LLMResponseEvent)]
        assert len(requests) == 1
        assert len(responses) == 1
        assert requests[0].label == "bid"
        assert responses[0].label == "bid"


# ──────────────────────────────────────────────────────────
# Bidding Controller
# ──────────────────────────────────────────────────────────


class TestBiddingController:
    async def test_basic_bidding_highest_confidence_wins(self) -> None:
        emitter = make_emitter()
        participants = [
            BiddableAgent(
                agent=make_agent("a1", emitter, "result-1"),
                bid_generator=FixedBidGenerator(confidence=0.5),
            ),
            BiddableAgent(
                agent=make_agent("a2", emitter, "result-2"),
                bid_generator=FixedBidGenerator(confidence=0.9),
            ),
        ]
        bidding = Bidding(participants=participants, emitter=emitter)
        result = await bidding.run("solve this")

        assert isinstance(result, BiddingResult)
        assert result.allocated is True
        assert result.winning_bid is not None
        assert result.winning_bid.agent_name == "a2"
        assert result.execution_result == "result-2"
        assert len(result.all_bids) == 2

    async def test_lowest_cost_allocation(self) -> None:
        emitter = make_emitter()
        participants = [
            BiddableAgent(
                agent=make_agent("expensive", emitter, "exp-result"),
                bid_generator=FixedBidGenerator(confidence=0.9, estimated_cost=100.0),
            ),
            BiddableAgent(
                agent=make_agent("cheap", emitter, "cheap-result"),
                bid_generator=FixedBidGenerator(confidence=0.8, estimated_cost=10.0),
            ),
        ]
        bidding = Bidding(
            participants=participants,
            emitter=emitter,
            allocation_strategy=LowestCost(),
        )
        result = await bidding.run("task")

        assert result.winning_bid is not None
        assert result.winning_bid.agent_name == "cheap"
        assert result.execution_result == "cheap-result"

    async def test_weighted_score_allocation(self) -> None:
        emitter = make_emitter()
        participants = [
            BiddableAgent(
                agent=make_agent("a1", emitter, "r1"),
                bid_generator=FixedBidGenerator(confidence=0.9, capabilities=["x"], estimated_cost=100.0),
            ),
            BiddableAgent(
                agent=make_agent("a2", emitter, "r2"),
                bid_generator=FixedBidGenerator(confidence=0.7, capabilities=["x", "y", "z"], estimated_cost=10.0),
            ),
        ]
        bidding = Bidding(
            participants=participants,
            emitter=emitter,
            allocation_strategy=WeightedScore(weights={"confidence": 0.1, "cost": 0.5, "capabilities": 0.5}),
        )
        result = await bidding.run("task")

        assert result.winning_bid is not None
        assert result.winning_bid.agent_name == "a2"

    async def test_min_bid_threshold_rejection(self) -> None:
        emitter = make_emitter()
        participants = [
            BiddableAgent(
                agent=make_agent("low", emitter, "result"),
                bid_generator=FixedBidGenerator(confidence=0.3),
            ),
        ]
        bidding = Bidding(
            participants=participants,
            emitter=emitter,
            min_bid_threshold=0.5,
        )
        result = await bidding.run("task")

        assert result.allocated is False
        assert result.winning_bid is None
        assert result.execution_result is None

    async def test_no_bids_all_generators_fail(self) -> None:
        emitter = make_emitter()

        from nanitics.infrastructure.observability.emitter import EventEmitter

        class FailingGenerator:
            async def generate(self, agent_name: str, task: str, *, emitter: EventEmitter) -> Bid:
                raise RuntimeError("generation failed")

        participants = [
            BiddableAgent(
                agent=make_agent("a1", emitter),
                bid_generator=FailingGenerator(),
            ),
        ]
        bidding = Bidding(participants=participants, emitter=emitter)
        result = await bidding.run("task")

        assert result.allocated is False
        assert result.winning_bid is None
        assert len(result.all_bids) == 0
        assert len(result.bid_failures) == 1
        assert result.bid_failures[0].agent_name == "a1"
        assert result.bid_failures[0].error_type == "RuntimeError"
        assert result.bid_failures[0].error_message == "generation failed"

    async def test_single_participant(self) -> None:
        emitter = make_emitter()
        participants = [
            BiddableAgent(
                agent=make_agent("solo", emitter, "solo-result"),
                bid_generator=FixedBidGenerator(confidence=0.7),
            ),
        ]
        bidding = Bidding(participants=participants, emitter=emitter)
        result = await bidding.run("task")

        assert result.allocated is True
        assert result.winning_bid is not None
        assert result.winning_bid.agent_name == "solo"
        assert result.execution_result == "solo-result"

    async def test_bid_generation_failure_excluded(self) -> None:
        emitter = make_emitter()

        from nanitics.infrastructure.observability.emitter import EventEmitter

        class FailingGenerator:
            async def generate(self, agent_name: str, task: str, *, emitter: EventEmitter) -> Bid:
                raise RuntimeError("fail")

        participants = [
            BiddableAgent(
                agent=make_agent("good", emitter, "good-result"),
                bid_generator=FixedBidGenerator(confidence=0.8),
            ),
            BiddableAgent(
                agent=make_agent("bad", emitter, "bad-result"),
                bid_generator=FailingGenerator(),
            ),
        ]
        bidding = Bidding(participants=participants, emitter=emitter)
        result = await bidding.run("task")

        assert result.allocated is True
        assert result.winning_bid is not None
        assert result.winning_bid.agent_name == "good"
        assert len(result.all_bids) == 1
        assert len(result.bid_failures) == 1
        assert result.bid_failures[0].agent_name == "bad"
        assert result.bid_failures[0].error_type == "RuntimeError"
        assert result.bid_failures[0].error_message == "fail"

    async def test_winner_execution_failure(self) -> None:
        emitter = make_emitter()
        participants = [
            BiddableAgent(
                agent=make_failing_agent("winner", emitter),
                bid_generator=FixedBidGenerator(confidence=0.9),
            ),
        ]
        bidding = Bidding(participants=participants, emitter=emitter)
        result = await bidding.run("task")

        assert result.allocated is True
        assert result.winning_bid is not None
        assert result.execution_result is None
        assert result.execution_error is not None
        # Backward compat: metadata still has execution_error
        assert "execution_error" in result.winning_bid.metadata

    async def test_event_emission(self) -> None:
        emitter = make_emitter()
        participants = [
            BiddableAgent(
                agent=make_agent("a1", emitter, "r1"),
                bid_generator=FixedBidGenerator(confidence=0.5),
            ),
            BiddableAgent(
                agent=make_agent("a2", emitter, "r2"),
                bid_generator=FixedBidGenerator(confidence=0.9),
            ),
        ]
        bidding = Bidding(participants=participants, emitter=emitter)
        await bidding.run("task")

        start_events = [e for e in emitter.events if isinstance(e, BiddingStartEvent)]
        bid_events = [e for e in emitter.events if isinstance(e, BidReceivedEvent)]
        allocated_events = [e for e in emitter.events if isinstance(e, BidAllocatedEvent)]
        complete_events = [e for e in emitter.events if isinstance(e, BiddingCompleteEvent)]

        assert len(start_events) == 1
        assert start_events[0].task == "task"
        assert set(start_events[0].participant_names) == {"a1", "a2"}

        assert len(bid_events) == 2
        bid_names = {e.agent_name for e in bid_events}
        assert bid_names == {"a1", "a2"}

        assert len(allocated_events) == 1
        assert allocated_events[0].winner == "a2"
        assert allocated_events[0].total_bids == 2

        assert len(complete_events) == 1
        assert complete_events[0].allocated is True
        assert complete_events[0].winner == "a2"

    async def test_bid_generation_failure_emits_event_with_error(self) -> None:
        emitter = make_emitter()

        from nanitics.infrastructure.observability.emitter import EventEmitter

        class FailingGenerator:
            async def generate(self, agent_name: str, task: str, *, emitter: EventEmitter) -> Bid:
                raise RuntimeError("bid generation exploded")

        participants = [
            BiddableAgent(
                agent=make_agent("failing", emitter),
                bid_generator=FailingGenerator(),
            ),
        ]
        bidding = Bidding(participants=participants, emitter=emitter)
        await bidding.run("task")

        bid_events = [e for e in emitter.events if isinstance(e, BidReceivedEvent)]
        assert len(bid_events) == 1
        assert bid_events[0].agent_name == "failing"
        assert bid_events[0].confidence == 0.0
        assert bid_events[0].reasoning == ""
        assert bid_events[0].error == "bid generation exploded"

    async def test_bidding_run_emits_one_llm_event_pair_per_llm_bid_generator(self) -> None:
        """With two ``LLMBidGenerator`` participants, ``Bidding.run`` emits
        exactly two ``LLMRequestEvent`` + two ``LLMResponseEvent`` instances
        — all labelled ``"bid"`` — through the same emitter that carries
        the bidding-primitive events. This is the telemetry contract that
        makes bid-phase spend roll up into the run's summary.
        """
        emitter = make_emitter()

        bid_json_a = json.dumps(
            {
                "confidence": 0.8,
                "capabilities": ["analysis"],
                "estimated_cost": None,
                "reasoning": "a reasoning",
            }
        )
        bid_json_b = json.dumps(
            {
                "confidence": 0.6,
                "capabilities": ["writing"],
                "estimated_cost": None,
                "reasoning": "b reasoning",
            }
        )

        participants = [
            BiddableAgent(
                agent=make_agent("a1", emitter, "result-a1"),
                bid_generator=LLMBidGenerator(
                    llm_client=MockLLMClient([make_response(bid_json_a)]),
                    agent_description="Analyst agent",
                ),
            ),
            BiddableAgent(
                agent=make_agent("a2", emitter, "result-a2"),
                bid_generator=LLMBidGenerator(
                    llm_client=MockLLMClient([make_response(bid_json_b)]),
                    agent_description="Writer agent",
                ),
            ),
        ]
        bidding = Bidding(participants=participants, emitter=emitter)
        result = await bidding.run("assess this")

        # Winner is the higher-confidence participant; its agent executed.
        assert result.allocated is True
        assert result.winning_bid is not None
        assert result.winning_bid.agent_name == "a1"

        bid_requests = [e for e in emitter.events if isinstance(e, LLMRequestEvent) and e.label == "bid"]
        bid_responses = [e for e in emitter.events if isinstance(e, LLMResponseEvent) and e.label == "bid"]
        assert len(bid_requests) == 2
        assert len(bid_responses) == 2

        # The existing bidding events still fire — one start, two bids, one
        # allocation, one complete — plus whatever the winning agent emits.
        assert sum(1 for e in emitter.events if isinstance(e, BiddingStartEvent)) == 1
        assert sum(1 for e in emitter.events if isinstance(e, BidReceivedEvent)) == 2
        assert sum(1 for e in emitter.events if isinstance(e, BidAllocatedEvent)) == 1
        assert sum(1 for e in emitter.events if isinstance(e, BiddingCompleteEvent)) == 1

    async def test_cancellation_token_passed(self) -> None:
        token = CancellationToken()
        emitter = make_emitter()
        agent = ReActAgent(
            name="a1",
            llm_client=MockLLMClient([make_response("out")]),
            emitter=emitter,
            system_prompt="You are a1.",
            tools=[],
            cancellation_token=token,
        )
        participants = [
            BiddableAgent(
                agent=agent,
                bid_generator=FixedBidGenerator(confidence=0.8),
            ),
        ]
        bidding = Bidding(
            participants=participants,
            emitter=emitter,
            cancellation_token=token,
        )
        result = await bidding.run("task")
        assert result.allocated is True
        assert result.execution_result == "out"


# ──────────────────────────────────────────────────────────
# thread_key propagation
# ──────────────────────────────────────────────────────────


class TestBiddingThreadKeys:
    def test_rejects_unknown_agent_name(self) -> None:
        emitter = make_emitter()
        agent = make_agent("a", emitter)
        with pytest.raises(ValueError, match="thread_keys references agents"):
            Bidding(
                participants=[BiddableAgent(agent=agent, bid_generator=FixedBidGenerator(confidence=0.9))],
                emitter=emitter,
                thread_keys={"missing": "k"},
            )

    def test_thread_keys_default_empty(self) -> None:
        emitter = make_emitter()
        agent = make_agent("a", emitter)
        bidding = Bidding(
            participants=[BiddableAgent(agent=agent, bid_generator=FixedBidGenerator(confidence=0.9))],
            emitter=emitter,
        )
        assert bidding._thread_keys == {}

    async def test_winner_thread_accumulates_across_runs(self) -> None:
        from nanitics.composition import InMemoryThreadStore

        emitter = make_emitter()
        thread_store = InMemoryThreadStore()
        winner = ReActAgent(
            name="winner",
            llm_client=MockLLMClient([make_response("win1"), make_response("win2")]),
            emitter=emitter,
            system_prompt="answer.",
            tools=[],
            thread_store=thread_store,
        )
        loser = make_agent("loser", emitter)

        bidding = Bidding(
            participants=[
                BiddableAgent(agent=winner, bid_generator=FixedBidGenerator(confidence=0.9)),
                BiddableAgent(agent=loser, bid_generator=FixedBidGenerator(confidence=0.1)),
            ],
            emitter=emitter,
            thread_keys={"winner": "winner-thread"},
        )

        # Two auctions, both won by `winner` (higher confidence).
        await bidding.run("task1")
        await bidding.run("task2")

        loaded = await thread_store.load("winner-thread")
        assert sum(1 for m in loaded if m.role == "assistant") >= 2
