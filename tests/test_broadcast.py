"""Tests for Broadcast: models, strategies, filters, controller, events."""

from nanitics import (
    InMemoryEmitter,
    MockLLMClient,
    ReActAgent,
)
from nanitics.composition.multi_agent.broadcast import (
    AllEligible,
    Broadcast,
    BroadcastResponse,
    BroadcastResult,
    CapabilityFilter,
    CollectAll,
    EligibilityFilter,
    FilterResponses,
    MergeResponses,
    ResponseStrategy,
    SelectBest,
)
from nanitics.core.agents.base import Agent
from nanitics.infrastructure.observability.events import (
    BroadcastCompleteEvent,
    BroadcastResponseEvent,
    BroadcastStartEvent,
)
from nanitics.safety.cancellation import CancellationToken
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
    """Agent whose LLM client will raise on generate."""

    client = MockLLMClient([])  # No responses → will raise

    return ReActAgent(
        name=name,
        llm_client=client,
        emitter=emitter,
        system_prompt=f"You are {name}.",
        tools=[],
    )


# ──────────────────────────────────────────────────────────
# Protocol Conformance
# ──────────────────────────────────────────────────────────


class TestProtocols:
    def test_collect_all_satisfies_response_strategy(self) -> None:
        assert isinstance(CollectAll(), ResponseStrategy)

    def test_all_eligible_satisfies_eligibility_filter(self) -> None:
        assert isinstance(AllEligible(), EligibilityFilter)


# ──────────────────────────────────────────────────────────
# Response Strategies
# ──────────────────────────────────────────────────────────


class TestCollectAll:
    async def test_returns_list_of_outputs(self) -> None:
        responses = [
            BroadcastResponse(agent_name="a", output="out-a", steps=1, termination_reason="end"),
            BroadcastResponse(agent_name="b", output="out-b", steps=2, termination_reason="end"),
        ]
        result = await CollectAll().aggregate(responses)
        assert result == ["out-a", "out-b"]

    async def test_empty_responses(self) -> None:
        result = await CollectAll().aggregate([])
        assert result == []


class TestSelectBest:
    async def test_selects_highest_scored(self) -> None:
        responses = [
            BroadcastResponse(agent_name="a", output="low", steps=1, termination_reason="end"),
            BroadcastResponse(agent_name="b", output="high", steps=2, termination_reason="end"),
            BroadcastResponse(agent_name="c", output="mid", steps=1, termination_reason="end"),
        ]
        strategy = SelectBest(scorer=lambda r: float(r.steps))
        result = await strategy.aggregate(responses)
        assert result == "high"

    async def test_ties_broken_by_order(self) -> None:
        responses = [
            BroadcastResponse(agent_name="a", output="first", steps=1, termination_reason="end"),
            BroadcastResponse(agent_name="b", output="second", steps=1, termination_reason="end"),
        ]
        strategy = SelectBest(scorer=lambda r: 1.0)
        result = await strategy.aggregate(responses)
        assert result == "first"

    async def test_empty_responses_returns_none(self) -> None:
        strategy = SelectBest(scorer=lambda r: 1.0)
        result = await strategy.aggregate([])
        assert result is None

    async def test_async_scorer(self) -> None:
        responses = [
            BroadcastResponse(agent_name="a", output="low", steps=1, termination_reason="end"),
            BroadcastResponse(agent_name="b", output="high", steps=5, termination_reason="end"),
        ]

        async def async_scorer(r: BroadcastResponse) -> float:
            return float(r.steps)

        strategy = SelectBest(scorer=async_scorer)
        result = await strategy.aggregate(responses)
        assert result == "high"


class TestMergeResponses:
    async def test_calls_llm_with_responses(self) -> None:
        client = MockLLMClient([make_response("synthesized answer")])
        strategy = MergeResponses(llm_client=client)
        responses = [
            BroadcastResponse(agent_name="a", output="ans-a", steps=1, termination_reason="end"),
            BroadcastResponse(agent_name="b", output="ans-b", steps=1, termination_reason="end"),
        ]
        result = await strategy.aggregate(responses)
        assert result == "synthesized answer"
        assert len(client.calls) == 1
        # Verify the prompt contains both responses
        user_msg = client.calls[0]["messages"][0]
        assert "ans-a" in user_msg.content
        assert "ans-b" in user_msg.content

    async def test_empty_responses_returns_none(self) -> None:
        client = MockLLMClient([])
        strategy = MergeResponses(llm_client=client)
        result = await strategy.aggregate([])
        assert result is None

    async def test_custom_merge_prompt(self) -> None:
        client = MockLLMClient([make_response("merged")])
        strategy = MergeResponses(llm_client=client, merge_prompt="Custom: {responses}")
        responses = [
            BroadcastResponse(agent_name="a", output="ans-a", steps=1, termination_reason="end"),
        ]
        result = await strategy.aggregate(responses)
        assert result == "merged"
        user_msg = client.calls[0]["messages"][0]
        assert user_msg.content.startswith("Custom:")


class TestFilterResponses:
    async def test_keeps_matching_responses(self) -> None:
        responses = [
            BroadcastResponse(agent_name="a", output="keep", steps=1, termination_reason="end"),
            BroadcastResponse(agent_name="b", output="drop", steps=1, termination_reason="end"),
            BroadcastResponse(agent_name="c", output="keep", steps=1, termination_reason="end"),
        ]
        strategy = FilterResponses(predicate=lambda r: r.output == "keep")
        result = await strategy.aggregate(responses)
        assert result == ["keep", "keep"]

    async def test_async_predicate(self) -> None:
        responses = [
            BroadcastResponse(agent_name="a", output="yes", steps=1, termination_reason="end"),
            BroadcastResponse(agent_name="b", output="no", steps=1, termination_reason="end"),
        ]

        async def async_pred(r: BroadcastResponse) -> bool:
            return bool(r.output == "yes")

        strategy = FilterResponses(predicate=async_pred)
        result = await strategy.aggregate(responses)
        assert result == ["yes"]


# ──────────────────────────────────────────────────────────
# Eligibility Filters
# ──────────────────────────────────────────────────────────


class TestAllEligible:
    async def test_returns_all_agents(self) -> None:
        emitter = make_emitter()
        agents: list[Agent] = [make_agent(f"a{i}", emitter) for i in range(3)]
        result = await AllEligible().filter(agents, "task")
        assert len(result) == 3


class TestCapabilityFilter:
    async def test_filters_by_capability_overlap(self) -> None:
        emitter = make_emitter()
        agents: list[Agent] = [
            make_agent("coder", emitter),
            make_agent("writer", emitter),
            make_agent("reviewer", emitter),
        ]
        filt = CapabilityFilter(
            capabilities={
                "coder": ["code", "debug"],
                "writer": ["write", "edit"],
                "reviewer": ["review", "code"],
            },
            required=["code"],
        )
        result = await filt.filter(agents, "task")
        names = [a.name for a in result]
        assert names == ["coder", "reviewer"]

    async def test_agents_not_in_mapping_excluded(self) -> None:
        emitter = make_emitter()
        agents: list[Agent] = [make_agent("known", emitter), make_agent("unknown", emitter)]
        filt = CapabilityFilter(
            capabilities={"known": ["code"]},
            required=["code"],
        )
        result = await filt.filter(agents, "task")
        assert [a.name for a in result] == ["known"]


# ──────────────────────────────────────────────────────────
# Broadcast Controller
# ──────────────────────────────────────────────────────────


class TestBroadcastController:
    async def test_basic_broadcast_collect_all(self) -> None:
        emitter = make_emitter()
        agents = [
            make_agent("a1", emitter, "output-1"),
            make_agent("a2", emitter, "output-2"),
        ]
        broadcast = Broadcast(agents=agents, emitter=emitter)
        result = await broadcast.run("solve this")

        assert isinstance(result, BroadcastResult)
        assert result.agents_participated == 2
        assert len(result.responses) == 2
        assert result.response_strategy == "CollectAll"
        assert set(result.aggregated_output) == {"output-1", "output-2"}

    async def test_select_best_strategy(self) -> None:
        emitter = make_emitter()
        agents = [
            make_agent("short", emitter, "ab"),
            make_agent("long", emitter, "abcdef"),
        ]
        strategy = SelectBest(scorer=lambda r: len(str(r.output)))
        broadcast = Broadcast(agents=agents, emitter=emitter, response_strategy=strategy)
        result = await broadcast.run("task")

        assert result.aggregated_output == "abcdef"
        assert result.response_strategy == "SelectBest"

    async def test_merge_responses_strategy(self) -> None:
        merge_client = MockLLMClient([make_response("synthesized")])
        emitter = make_emitter()
        agents = [
            make_agent("a1", emitter, "resp-1"),
            make_agent("a2", emitter, "resp-2"),
        ]
        strategy = MergeResponses(llm_client=merge_client)
        broadcast = Broadcast(agents=agents, emitter=emitter, response_strategy=strategy)
        result = await broadcast.run("task")

        assert result.aggregated_output == "synthesized"

    async def test_filter_responses_strategy(self) -> None:
        emitter = make_emitter()
        agents = [
            make_agent("a1", emitter, "keep-this"),
            make_agent("a2", emitter, "drop-this"),
        ]
        strategy = FilterResponses(predicate=lambda r: "keep" in str(r.output))
        broadcast = Broadcast(agents=agents, emitter=emitter, response_strategy=strategy)
        result = await broadcast.run("task")

        assert result.aggregated_output == ["keep-this"]

    async def test_eligibility_filter(self) -> None:
        emitter = make_emitter()
        agents = [
            make_agent("coder", emitter, "code-output"),
            make_agent("writer", emitter, "write-output"),
        ]
        filt = CapabilityFilter(
            capabilities={"coder": ["code"], "writer": ["write"]},
            required=["code"],
        )
        broadcast = Broadcast(agents=agents, emitter=emitter, eligibility_filter=filt)
        result = await broadcast.run("task")

        assert result.agents_participated == 1
        assert len(result.responses) == 1
        assert result.responses[0].agent_name == "coder"

    async def test_empty_agents_list(self) -> None:
        emitter = make_emitter()
        broadcast = Broadcast(agents=[], emitter=emitter)
        result = await broadcast.run("task")

        assert result.agents_participated == 0
        assert result.responses == []
        assert result.aggregated_output is None

    async def test_no_eligible_agents(self) -> None:
        emitter = make_emitter()
        agents = [make_agent("a1", emitter)]
        filt = CapabilityFilter(
            capabilities={"a1": ["write"]},
            required=["code"],
        )
        broadcast = Broadcast(agents=agents, emitter=emitter, eligibility_filter=filt)
        result = await broadcast.run("task")

        assert result.agents_participated == 0
        assert result.responses == []
        assert result.aggregated_output is None

    async def test_agent_failure_excluded(self) -> None:
        emitter = make_emitter()
        agents = [
            make_agent("good", emitter, "success"),
            make_failing_agent("bad", emitter),
        ]
        broadcast = Broadcast(agents=agents, emitter=emitter)
        result = await broadcast.run("task")

        assert result.agents_participated == 2
        assert len(result.responses) == 1
        assert result.responses[0].agent_name == "good"
        assert len(result.failures) == 1
        assert result.failures[0].agent_name == "bad"
        assert result.failures[0].error_type != ""
        assert result.failures[0].error_message != ""

    async def test_all_agents_fail(self) -> None:
        emitter = make_emitter()
        agents = [
            make_failing_agent("bad1", emitter),
            make_failing_agent("bad2", emitter),
        ]
        broadcast = Broadcast(agents=agents, emitter=emitter)
        result = await broadcast.run("task")

        assert result.agents_participated == 2
        assert len(result.responses) == 0
        assert result.aggregated_output == []  # CollectAll on empty → []
        assert len(result.failures) == 2
        failure_names = {f.agent_name for f in result.failures}
        assert failure_names == {"bad1", "bad2"}

    async def test_event_emission(self) -> None:
        emitter = make_emitter()
        agents = [
            make_agent("a1", emitter, "out-1"),
            make_agent("a2", emitter, "out-2"),
        ]
        broadcast = Broadcast(agents=agents, emitter=emitter)
        await broadcast.run("task")

        start_events = [e for e in emitter.events if isinstance(e, BroadcastStartEvent)]
        response_events = [e for e in emitter.events if isinstance(e, BroadcastResponseEvent)]
        complete_events = [e for e in emitter.events if isinstance(e, BroadcastCompleteEvent)]

        assert len(start_events) == 1
        assert start_events[0].task == "task"
        assert set(start_events[0].agent_names) == {"a1", "a2"}
        assert start_events[0].response_strategy == "CollectAll"

        assert len(response_events) == 2
        agent_names = {e.agent_name for e in response_events}
        assert agent_names == {"a1", "a2"}
        for e in response_events:
            assert e.error is None

        assert len(complete_events) == 1
        assert complete_events[0].total_agents == 2
        assert complete_events[0].responses_collected == 2

    async def test_error_event_on_failure(self) -> None:
        emitter = make_emitter()
        agents = [
            make_agent("good", emitter, "ok"),
            make_failing_agent("bad", emitter),
        ]
        broadcast = Broadcast(agents=agents, emitter=emitter)
        result = await broadcast.run("task")

        response_events = [e for e in emitter.events if isinstance(e, BroadcastResponseEvent)]
        error_events = [e for e in response_events if e.error is not None]
        assert len(error_events) == 1
        assert error_events[0].agent_name == "bad"

        # Verify result also contains failure details
        assert len(result.failures) == 1
        assert result.failures[0].agent_name == "bad"

        # Verify complete event has failure count
        complete_events = [e for e in emitter.events if isinstance(e, BroadcastCompleteEvent)]
        assert complete_events[0].failures == 1

    async def test_cancellation_token_passed(self) -> None:
        token = CancellationToken()
        emitter = make_emitter()
        # Provide a cancellation token to the agent
        agent_with_token = ReActAgent(
            name="a1",
            llm_client=MockLLMClient([make_response("out")]),
            emitter=emitter,
            system_prompt="You are a1.",
            tools=[],
            cancellation_token=token,
        )
        broadcast = Broadcast(
            agents=[agent_with_token],
            emitter=emitter,
            cancellation_token=token,
        )
        result = await broadcast.run("task")
        assert len(result.responses) == 1
