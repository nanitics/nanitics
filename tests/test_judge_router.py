"""Tests for JudgeRouter: comparative-judgment routing primitive."""

from __future__ import annotations

import json

import pytest

from nanitics.composition.multi_agent.bidding import (
    DEFAULT_CALIBRATED_JUDGE_PROMPT_TEMPLATE,
    BiddableAgent,
    FixedBidGenerator,
)
from nanitics.composition.multi_agent.judge_router import (
    JudgeRouter,
    JudgeRouterResult,
    RankedCandidate,
)
from nanitics.infrastructure import MockLLMClient
from nanitics.infrastructure.observability.events import (
    JudgeAllocatedEvent,
    JudgeRankingEvent,
    JudgeRoutingCompleteEvent,
    JudgeRoutingStartEvent,
    LLMRequestEvent,
    LLMResponseEvent,
)
from nanitics.strategies import ReActAgent
from nanitics.tracing import InMemoryEmitter
from tests.testing_helpers import make_emitter, make_response


def _make_agent(name: str, emitter: InMemoryEmitter, response: str = "answered") -> ReActAgent:
    return ReActAgent(
        name=name,
        llm_client=MockLLMClient([make_response(response)]),
        emitter=emitter,
        system_prompt=f"You are {name}.",
        tools=[],
    )


def _participant(name: str, emitter: InMemoryEmitter, response: str = "answered") -> BiddableAgent:
    return BiddableAgent(
        agent=_make_agent(name, emitter, response=response),
        bid_generator=FixedBidGenerator(confidence=0.0),
    )


def _ranking_response(entries: list[dict[str, object]]) -> str:
    return json.dumps({"ranking": entries})


class TestRankedCandidateModel:
    def test_frozen_with_defaults(self) -> None:
        candidate = RankedCandidate(
            agent_name="a",
            confidence=0.8,
            capabilities=["x"],
            reasoning="r",
        )
        assert candidate.estimated_cost is None
        with pytest.raises(ValueError):
            candidate.confidence = 0.1  # type: ignore[misc]


class TestJudgeRouterTemplate:
    def test_invalid_template_raises_value_error(self) -> None:
        emitter = make_emitter()
        with pytest.raises(ValueError, match="participants"):
            JudgeRouter(
                participants=[_participant("a", emitter)],
                judge_llm=MockLLMClient([]),
                emitter=emitter,
                prompt_template="No placeholders here",
            )

    async def test_default_template_includes_anchors(self) -> None:
        emitter = make_emitter()
        client = MockLLMClient(
            [
                make_response(
                    _ranking_response(
                        [
                            {
                                "agent_name": "a",
                                "confidence": 0.9,
                                "capabilities": ["x"],
                                "estimated_cost": None,
                                "reasoning": "r",
                            }
                        ]
                    )
                )
            ]
        )
        router = JudgeRouter(
            participants=[_participant("a", emitter)],
            judge_llm=client,
            emitter=emitter,
        )
        await router.run("a task")
        prompt = client.calls[0]["messages"][0].content
        # Anchor lines from DEFAULT_CALIBRATED_JUDGE_PROMPT_TEMPLATE.
        assert "0.9 = uniquely positioned" in prompt
        assert "0.7 = capable" in prompt
        assert "0.4 = adjacent" in prompt
        assert "0.0 = out of scope" in prompt
        assert "- a" in prompt
        assert "Task: a task" in prompt
        # Sanity: same template constant is publicly exported.
        assert "Calibration anchors" in DEFAULT_CALIBRATED_JUDGE_PROMPT_TEMPLATE


class TestJudgeRouterHappyPath:
    async def test_single_winner_full_event_sequence(self) -> None:
        emitter = make_emitter()
        ranking_json = _ranking_response(
            [
                {
                    "agent_name": "billing",
                    "confidence": 0.9,
                    "capabilities": ["invoices"],
                    "estimated_cost": 0.02,
                    "reasoning": "Closest fit",
                },
                {
                    "agent_name": "support",
                    "confidence": 0.4,
                    "capabilities": ["faq"],
                    "estimated_cost": 0.01,
                    "reasoning": "Adjacent only",
                },
            ]
        )
        judge_client = MockLLMClient([make_response(ranking_json)])
        billing = _participant("billing", emitter, response="Refund issued")
        support = _participant("support", emitter)
        router = JudgeRouter(
            participants=[billing, support],
            judge_llm=judge_client,
            emitter=emitter,
        )

        result = await router.run("invoice issue")

        assert isinstance(result, JudgeRouterResult)
        assert result.allocated is True
        assert result.winner is not None
        assert result.winner.agent_name == "billing"
        assert result.winner.confidence == 0.9
        assert len(result.ranking) == 2
        assert result.ranking[0].agent_name == "billing"
        assert result.execution_result == "Refund issued"
        assert result.judge_error is None
        assert result.execution_error is None

        # Event sequence: 1 start, 1 LLM request, 1 LLM response (label="judge"),
        # 2 ranking events, 1 allocated, then the winner agent's own subtree,
        # ending with 1 complete.
        events = emitter.events
        start = [e for e in events if isinstance(e, JudgeRoutingStartEvent)]
        rankings = [e for e in events if isinstance(e, JudgeRankingEvent)]
        allocated = [e for e in events if isinstance(e, JudgeAllocatedEvent)]
        complete = [e for e in events if isinstance(e, JudgeRoutingCompleteEvent)]
        assert len(start) == 1
        assert len(rankings) == 2
        assert [r.rank for r in rankings] == [0, 1]
        assert len(allocated) == 1
        assert allocated[0].winner == "billing"
        assert allocated[0].rejection_reason is None
        assert allocated[0].total_candidates == 2
        assert len(complete) == 1
        assert complete[0].allocated is True

        # Judge LLM events labelled "judge".
        judge_requests = [e for e in events if isinstance(e, LLMRequestEvent) and e.label == "judge"]
        judge_responses = [e for e in events if isinstance(e, LLMResponseEvent) and e.label == "judge"]
        assert len(judge_requests) == 1
        assert len(judge_responses) == 1

    async def test_confidence_is_clamped(self) -> None:
        emitter = make_emitter()
        ranking_json = _ranking_response(
            [
                {
                    "agent_name": "a",
                    "confidence": 1.5,
                    "capabilities": [],
                    "estimated_cost": None,
                    "reasoning": "over",
                },
                {
                    "agent_name": "b",
                    "confidence": -0.5,
                    "capabilities": [],
                    "estimated_cost": None,
                    "reasoning": "under",
                },
            ]
        )
        router = JudgeRouter(
            participants=[_participant("a", emitter), _participant("b", emitter)],
            judge_llm=MockLLMClient([make_response(ranking_json)]),
            emitter=emitter,
        )
        result = await router.run("task")
        assert result.ranking[0].confidence == 1.0
        assert result.ranking[1].confidence == 0.0


class TestJudgeRouterFailureModes:
    async def test_unknown_agent_returns_no_winner(self) -> None:
        emitter = make_emitter()
        ranking_json = _ranking_response(
            [
                {
                    "agent_name": "ghost",
                    "confidence": 0.9,
                    "capabilities": [],
                    "estimated_cost": None,
                    "reasoning": "r",
                }
            ]
        )
        router = JudgeRouter(
            participants=[_participant("a", emitter)],
            judge_llm=MockLLMClient([make_response(ranking_json)]),
            emitter=emitter,
        )
        result = await router.run("task")
        assert result.allocated is False
        assert result.winner is None
        assert result.judge_error == "unknown_agent: ghost"
        allocated_events = [e for e in emitter.events if isinstance(e, JudgeAllocatedEvent)]
        assert allocated_events[0].rejection_reason == "unknown_agent"

    async def test_empty_ranking_returns_no_winner(self) -> None:
        emitter = make_emitter()
        ranking_json = _ranking_response([])
        router = JudgeRouter(
            participants=[_participant("a", emitter)],
            judge_llm=MockLLMClient([make_response(ranking_json)]),
            emitter=emitter,
        )
        result = await router.run("task")
        assert result.allocated is False
        assert result.winner is None
        assert result.judge_error == "empty_ranking"
        # Exactly: 1 start, 0 ranking, 1 allocated, 1 complete.
        rankings = [e for e in emitter.events if isinstance(e, JudgeRankingEvent)]
        allocated = [e for e in emitter.events if isinstance(e, JudgeAllocatedEvent)]
        complete = [e for e in emitter.events if isinstance(e, JudgeRoutingCompleteEvent)]
        assert rankings == []
        assert len(allocated) == 1
        assert allocated[0].winner is None
        assert allocated[0].rejection_reason == "empty_ranking"
        assert len(complete) == 1
        assert complete[0].judge_error == "empty_ranking"

    async def test_below_threshold_returns_no_winner(self) -> None:
        emitter = make_emitter()
        ranking_json = _ranking_response(
            [
                {
                    "agent_name": "a",
                    "confidence": 0.3,
                    "capabilities": [],
                    "estimated_cost": None,
                    "reasoning": "weak",
                }
            ]
        )
        router = JudgeRouter(
            participants=[_participant("a", emitter)],
            judge_llm=MockLLMClient([make_response(ranking_json)]),
            emitter=emitter,
            min_confidence_threshold=0.5,
        )
        result = await router.run("task")
        assert result.allocated is False
        assert result.winner is None
        assert result.judge_error is None
        allocated_events = [e for e in emitter.events if isinstance(e, JudgeAllocatedEvent)]
        assert allocated_events[0].rejection_reason == "below_threshold"

    async def test_judge_llm_exception_propagates(self) -> None:
        emitter = make_emitter()
        # MockLLMClient with empty response queue raises on call — surfaces failure.
        router = JudgeRouter(
            participants=[_participant("a", emitter)],
            judge_llm=MockLLMClient([]),
            emitter=emitter,
        )
        with pytest.raises(ValueError, match="no more scripted responses"):
            await router.run("task")

    async def test_winning_agent_execution_error_is_captured(self) -> None:
        emitter = make_emitter()
        ranking_json = _ranking_response(
            [
                {
                    "agent_name": "a",
                    "confidence": 0.9,
                    "capabilities": [],
                    "estimated_cost": None,
                    "reasoning": "r",
                }
            ]
        )
        # Empty MockLLMClient on the winning agent raises during run.
        failing_agent = ReActAgent(
            name="a",
            llm_client=MockLLMClient([]),
            emitter=emitter,
            system_prompt="x",
            tools=[],
        )
        participant = BiddableAgent(agent=failing_agent, bid_generator=FixedBidGenerator(confidence=0.0))
        router = JudgeRouter(
            participants=[participant],
            judge_llm=MockLLMClient([make_response(ranking_json)]),
            emitter=emitter,
        )
        result = await router.run("task")
        assert result.allocated is True  # winner was selected
        assert result.execution_error is not None
        assert result.execution_result is None


class TestJudgeRoutingEventRoundTrip:
    """Pin tagged-union round-trip for the four new event types."""

    def test_round_trip_each_event(self) -> None:
        from pydantic import TypeAdapter

        from nanitics.infrastructure.observability.events import TraceEvent

        adapter: TypeAdapter[TraceEvent] = TypeAdapter(TraceEvent)
        events: list[TraceEvent] = [
            JudgeRoutingStartEvent(
                trace_id="t",
                span_id="s",
                parent_span_id=None,
                task="task",
                participant_names=["a", "b"],
            ),
            JudgeRankingEvent(
                trace_id="t",
                span_id="s",
                parent_span_id=None,
                agent_name="a",
                rank=0,
                confidence=0.9,
                reasoning="r",
                estimated_cost=0.01,
            ),
            JudgeAllocatedEvent(
                trace_id="t",
                span_id="s",
                parent_span_id=None,
                winner="a",
                confidence=0.9,
                total_candidates=2,
                rejection_reason=None,
            ),
            JudgeRoutingCompleteEvent(
                trace_id="t",
                span_id="s",
                parent_span_id=None,
                winner="a",
                total_participants=2,
                allocated=True,
                judge_error=None,
            ),
        ]
        for original in events:
            data = original.model_dump()
            recovered = adapter.validate_python(data)
            assert recovered == original


class TestJudgeRouterThreadKeys:
    def test_rejects_unknown_agent_name(self) -> None:
        emitter = make_emitter()
        with pytest.raises(ValueError, match="thread_keys references agents"):
            JudgeRouter(
                participants=[_participant("a", emitter)],
                judge_llm=MockLLMClient([]),
                emitter=emitter,
                thread_keys={"missing": "k"},
            )

    def test_thread_keys_default_empty(self) -> None:
        emitter = make_emitter()
        router = JudgeRouter(
            participants=[_participant("a", emitter)],
            judge_llm=MockLLMClient([]),
            emitter=emitter,
        )
        assert router._thread_keys == {}

    async def test_winner_thread_accumulates_across_runs(self) -> None:
        from nanitics.composition import InMemoryThreadStore

        emitter = make_emitter()
        thread_store = InMemoryThreadStore()

        winner = ReActAgent(
            name="winner",
            llm_client=MockLLMClient([make_response("ans1"), make_response("ans2")]),
            emitter=emitter,
            system_prompt="answer.",
            tools=[],
            thread_store=thread_store,
        )
        loser = _make_agent("loser", emitter)

        judge_responses = [
            make_response(
                _ranking_response(
                    [
                        {
                            "agent_name": "winner",
                            "confidence": 0.9,
                            "capabilities": [],
                            "estimated_cost": None,
                            "reasoning": "r",
                        },
                        {
                            "agent_name": "loser",
                            "confidence": 0.1,
                            "capabilities": [],
                            "estimated_cost": None,
                            "reasoning": "r",
                        },
                    ]
                )
            )
            for _ in range(2)
        ]

        router = JudgeRouter(
            participants=[
                BiddableAgent(agent=winner, bid_generator=FixedBidGenerator(confidence=0.0)),
                BiddableAgent(agent=loser, bid_generator=FixedBidGenerator(confidence=0.0)),
            ],
            judge_llm=MockLLMClient(judge_responses),
            emitter=emitter,
            thread_keys={"winner": "winner-thread"},
        )
        await router.run("first")
        await router.run("second")

        loaded = await thread_store.load("winner-thread")
        assert sum(1 for m in loaded if m.role == "assistant") >= 2
