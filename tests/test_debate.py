"""Tests for Debate: models, resolution strategies, controller, events."""

import pytest
from pydantic import ValidationError

from nanitics.composition.multi_agent.debate import (
    Argument,
    Debate,
    Debater,
    DebateResolution,
    DebateResult,
    JudgeResolution,
    LLMJudgeResolution,
    ResolutionStrategy,
    _format_transcript,
)
from nanitics.infrastructure import MockLLMClient
from nanitics.infrastructure.observability.events import (
    DebateArgumentEvent,
    DebateCompleteEvent,
    DebateResolutionEvent,
    DebateStartEvent,
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
    def test_argument_frozen(self) -> None:
        arg = Argument(round=1, agent_name="a", position="pro", content="text")
        with pytest.raises(ValidationError):
            arg.content = "new"

    def test_debate_resolution_frozen(self) -> None:
        res = DebateResolution(winner="pro", reasoning="good", synthesis="synth")
        with pytest.raises(ValidationError):
            res.winner = "con"

    def test_debater_frozen(self) -> None:
        emitter = make_emitter()
        agent = make_agent("a", emitter)
        debater = Debater(agent=agent, position="pro")
        with pytest.raises(ValidationError):
            debater.position = "con"

    def test_debate_result_frozen(self) -> None:
        result = DebateResult(
            resolution=DebateResolution(winner=None, reasoning="r", synthesis="s"),
            transcript=[],
            rounds_completed=1,
            termination_reason="max_rounds",
        )
        with pytest.raises(ValidationError):
            result.rounds_completed = 2

    def test_argument_fields(self) -> None:
        arg = Argument(round=2, agent_name="bot", position="con", content="my arg")
        assert arg.round == 2
        assert arg.agent_name == "bot"
        assert arg.position == "con"
        assert arg.content == "my arg"

    def test_debate_resolution_fields(self) -> None:
        res = DebateResolution(winner="pro", reasoning="because", synthesis="final")
        assert res.winner == "pro"
        assert res.reasoning == "because"
        assert res.synthesis == "final"

    def test_debate_resolution_winner_none(self) -> None:
        res = DebateResolution(winner=None, reasoning="tie", synthesis="both")
        assert res.winner is None


# ──────────────────────────────────────────────────────────
# Protocol Conformance
# ──────────────────────────────────────────────────────────


class TestProtocols:
    def test_judge_resolution_satisfies_protocol(self) -> None:
        emitter = make_emitter()
        agent = make_agent("judge", emitter)
        assert isinstance(JudgeResolution(judge=agent), ResolutionStrategy)

    def test_llm_judge_resolution_satisfies_protocol(self) -> None:
        client = MockLLMClient([])
        assert isinstance(LLMJudgeResolution(llm_client=client), ResolutionStrategy)

    def test_custom_class_satisfies_protocol(self) -> None:
        class CustomResolution:
            async def resolve(self, transcript: list[Argument], task: str) -> DebateResolution:
                return DebateResolution(winner=None, reasoning="", synthesis="")

        assert isinstance(CustomResolution(), ResolutionStrategy)


# ──────────────────────────────────────────────────────────
# Transcript Formatting
# ──────────────────────────────────────────────────────────


class TestTranscriptFormatting:
    def test_empty_transcript(self) -> None:
        assert _format_transcript([]) == ""

    def test_single_round(self) -> None:
        transcript = [
            Argument(round=1, agent_name="A", position="pro", content="arg-a"),
            Argument(round=1, agent_name="B", position="con", content="arg-b"),
        ]
        result = _format_transcript(transcript)
        assert "Round 1:" in result
        assert "[pro - A]: arg-a" in result
        assert "[con - B]: arg-b" in result

    def test_multiple_rounds(self) -> None:
        transcript = [
            Argument(round=1, agent_name="A", position="pro", content="r1a"),
            Argument(round=1, agent_name="B", position="con", content="r1b"),
            Argument(round=2, agent_name="A", position="pro", content="r2a"),
            Argument(round=2, agent_name="B", position="con", content="r2b"),
        ]
        result = _format_transcript(transcript)
        assert "Round 1:" in result
        assert "Round 2:" in result
        # Round 1 appears before Round 2
        assert result.index("Round 1:") < result.index("Round 2:")


# ──────────────────────────────────────────────────────────
# Resolution Strategies
# ──────────────────────────────────────────────────────────


class TestJudgeResolution:
    async def test_judge_receives_formatted_transcript(self) -> None:
        emitter = make_emitter()
        judge = make_agent("judge", emitter, response_content="Pro wins clearly")
        strategy = JudgeResolution(judge=judge)

        transcript = [
            Argument(round=1, agent_name="A", position="pro", content="pro arg"),
            Argument(round=1, agent_name="B", position="con", content="con arg"),
        ]
        resolution = await strategy.resolve(transcript, "test topic")

        assert resolution.winner is None
        assert resolution.reasoning == "Pro wins clearly"
        assert resolution.synthesis == "Pro wins clearly"


class TestLLMJudgeResolution:
    async def test_structured_output_parsing(self) -> None:
        import json

        verdict = json.dumps({"winner": "pro", "reasoning": "stronger args", "synthesis": "final answer"})
        client = MockLLMClient([make_response(verdict)])
        strategy = LLMJudgeResolution(llm_client=client)

        transcript = [
            Argument(round=1, agent_name="A", position="pro", content="pro arg"),
            Argument(round=1, agent_name="B", position="con", content="con arg"),
        ]
        resolution = await strategy.resolve(transcript, "test topic")

        assert resolution.winner == "pro"
        assert resolution.reasoning == "stronger args"
        assert resolution.synthesis == "final answer"

    async def test_custom_criteria_in_prompt(self) -> None:
        import json

        verdict = json.dumps({"winner": "con", "reasoning": "better evidence", "synthesis": "synthesis"})
        client = MockLLMClient([make_response(verdict)])
        strategy = LLMJudgeResolution(llm_client=client, criteria="Focus on scientific rigor")

        transcript = [
            Argument(round=1, agent_name="A", position="pro", content="pro arg"),
        ]
        await strategy.resolve(transcript, "topic")

        assert len(client.calls) == 1
        user_msg = client.calls[0]["messages"][0]
        assert "Focus on scientific rigor" in user_msg.content

    async def test_schema_violation_propagates(self) -> None:
        # Non-JSON response — LLMSchemaViolationError propagates to caller
        from nanitics.infrastructure.errors import LLMSchemaViolationError

        client = MockLLMClient([make_response("unstructured judgment")])
        strategy = LLMJudgeResolution(llm_client=client)

        transcript = [
            Argument(round=1, agent_name="A", position="pro", content="arg"),
        ]
        with pytest.raises(LLMSchemaViolationError):
            await strategy.resolve(transcript, "topic")

    async def test_fallback_when_parsed_is_none(self) -> None:
        from nanitics.infrastructure import LLMResponse
        from tests.testing_helpers import make_usage

        # Response with content=None → parsed stays None → fallback path
        response = LLMResponse(
            content=None,
            tool_calls=[],
            usage=make_usage(),
            model="test-model",
            stop_reason="end_turn",
        )
        client = MockLLMClient([response])
        strategy = LLMJudgeResolution(llm_client=client)

        transcript = [
            Argument(round=1, agent_name="A", position="pro", content="arg"),
        ]
        resolution = await strategy.resolve(transcript, "topic")

        assert resolution.winner is None
        assert resolution.reasoning == ""
        assert resolution.synthesis == ""


# ──────────────────────────────────────────────────────────
# Debate Controller
# ──────────────────────────────────────────────────────────


class _StubResolution:
    """Resolution strategy that records what it receives."""

    def __init__(self, winner: str | None = "pro") -> None:
        self.received_transcript: list[Argument] = []
        self.received_task: str = ""
        self._winner = winner

    async def resolve(self, transcript: list[Argument], task: str) -> DebateResolution:
        self.received_transcript = transcript
        self.received_task = task
        return DebateResolution(
            winner=self._winner,
            reasoning="stub reasoning",
            synthesis="stub synthesis",
        )


class TestDebateController:
    def test_requires_at_least_two_debaters(self) -> None:
        emitter = make_emitter()
        agent = make_agent("a", emitter)
        with pytest.raises(ValueError, match="at least 2"):
            Debate(
                debaters=[Debater(agent=agent, position="pro")],
                emitter=emitter,
                resolution=_StubResolution(),
            )

    async def test_two_debaters_two_rounds(self) -> None:
        emitter = make_emitter()
        agent_a = make_agent("AgentA", emitter, "pro argument", num_responses=2)
        agent_b = make_agent("AgentB", emitter, "con argument", num_responses=2)
        resolution = _StubResolution()

        debate = Debate(
            debaters=[
                Debater(agent=agent_a, position="pro"),
                Debater(agent=agent_b, position="con"),
            ],
            emitter=emitter,
            resolution=resolution,
            max_rounds=2,
        )
        result = await debate.run("Should AI be regulated?")

        assert result.rounds_completed == 2
        assert result.termination_reason == "max_rounds"
        assert len(result.transcript) == 4  # 2 debaters × 2 rounds

        # Verify transcript order
        assert result.transcript[0].round == 1
        assert result.transcript[0].agent_name == "AgentA"
        assert result.transcript[0].position == "pro"
        assert result.transcript[1].round == 1
        assert result.transcript[1].agent_name == "AgentB"
        assert result.transcript[1].position == "con"
        assert result.transcript[2].round == 2
        assert result.transcript[3].round == 2

        # Resolution received complete transcript
        assert len(resolution.received_transcript) == 4
        assert resolution.received_task == "Should AI be regulated?"

    async def test_three_debaters_multi_party(self) -> None:
        emitter = make_emitter()
        agents = [make_agent(f"Agent{i}", emitter, f"arg-{i}", num_responses=2) for i in range(3)]
        resolution = _StubResolution()

        debate = Debate(
            debaters=[
                Debater(agent=agents[0], position="pro"),
                Debater(agent=agents[1], position="con"),
                Debater(agent=agents[2], position="neutral"),
            ],
            emitter=emitter,
            resolution=resolution,
            max_rounds=2,
        )
        result = await debate.run("topic")

        assert len(result.transcript) == 6  # 3 debaters × 2 rounds
        # All three participate in each round
        round1 = [a for a in result.transcript if a.round == 1]
        assert len(round1) == 3
        round2 = [a for a in result.transcript if a.round == 2]
        assert len(round2) == 3

    async def test_round_1_contains_positional_instruction(self) -> None:
        emitter = make_emitter()
        client_a = MockLLMClient([make_response("pro arg")])
        client_b = MockLLMClient([make_response("con arg")])
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

        debate = Debate(
            debaters=[
                Debater(agent=agent_a, position="pro"),
                Debater(agent=agent_b, position="con"),
            ],
            emitter=emitter,
            resolution=_StubResolution(),
            max_rounds=1,
        )
        await debate.run("test topic")

        # The agent's run() input should include positional instruction
        # We check via the LLM client calls - the user message contains the task
        assert len(client_a.calls) == 1
        user_msg_a = client_a.calls[0]["messages"][0]
        assert "**pro**" in user_msg_a.content
        assert "test topic" in user_msg_a.content

        user_msg_b = client_b.calls[0]["messages"][0]
        assert "**con**" in user_msg_b.content

    async def test_later_rounds_include_transcript(self) -> None:
        emitter = make_emitter()
        client_a = MockLLMClient([make_response("r1"), make_response("r2")])
        client_b = MockLLMClient([make_response("r1b"), make_response("r2b")])
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

        debate = Debate(
            debaters=[
                Debater(agent=agent_a, position="pro"),
                Debater(agent=agent_b, position="con"),
            ],
            emitter=emitter,
            resolution=_StubResolution(),
            max_rounds=2,
        )
        await debate.run("topic")

        # Round 2 calls should include transcript from round 1
        # Agent A's round 2 call is calls[1]
        round2_msg_a = client_a.calls[1]["messages"][0]
        assert "Round 1:" in round2_msg_a.content
        assert "Respond to the opposing arguments" in round2_msg_a.content

    async def test_resolution_strategy_called_with_complete_transcript(self) -> None:
        emitter = make_emitter()
        agent_a = make_agent("A", emitter, "arg-a", num_responses=2)
        agent_b = make_agent("B", emitter, "arg-b", num_responses=2)
        resolution = _StubResolution("con")

        debate = Debate(
            debaters=[
                Debater(agent=agent_a, position="pro"),
                Debater(agent=agent_b, position="con"),
            ],
            emitter=emitter,
            resolution=resolution,
            max_rounds=2,
        )
        result = await debate.run("task")

        assert result.resolution.winner == "con"
        assert result.resolution.reasoning == "stub reasoning"
        assert len(resolution.received_transcript) == 4

    async def test_max_rounds_1_opening_arguments_only(self) -> None:
        emitter = make_emitter()
        agent_a = make_agent("A", emitter, "opening-a")
        agent_b = make_agent("B", emitter, "opening-b")

        debate = Debate(
            debaters=[
                Debater(agent=agent_a, position="pro"),
                Debater(agent=agent_b, position="con"),
            ],
            emitter=emitter,
            resolution=_StubResolution(),
            max_rounds=1,
        )
        result = await debate.run("topic")

        assert result.rounds_completed == 1
        assert len(result.transcript) == 2

    async def test_agent_failure_propagates(self) -> None:
        emitter = make_emitter()
        agent_a = make_failing_agent("A", emitter)
        agent_b = make_agent("B", emitter, "arg-b")

        debate = Debate(
            debaters=[
                Debater(agent=agent_a, position="pro"),
                Debater(agent=agent_b, position="con"),
            ],
            emitter=emitter,
            resolution=_StubResolution(),
            max_rounds=1,
        )
        with pytest.raises(ValueError):
            await debate.run("topic")


# ──────────────────────────────────────────────────────────
# Event Emission
# ──────────────────────────────────────────────────────────


class TestDebateEvents:
    async def test_events_emitted_in_correct_order(self) -> None:
        emitter = make_emitter()
        agent_a = make_agent("A", emitter, "arg-a", num_responses=2)
        agent_b = make_agent("B", emitter, "arg-b", num_responses=2)

        debate = Debate(
            debaters=[
                Debater(agent=agent_a, position="pro"),
                Debater(agent=agent_b, position="con"),
            ],
            emitter=emitter,
            resolution=_StubResolution(),
            max_rounds=2,
        )
        await debate.run("topic")

        # Filter debate-specific events
        debate_events = [
            e
            for e in emitter.events
            if isinstance(
                e,
                (
                    DebateStartEvent,
                    DebateArgumentEvent,
                    DebateResolutionEvent,
                    DebateCompleteEvent,
                ),
            )
        ]

        # Expected: start, 4 arguments (2 rounds × 2 debaters), resolution, complete
        assert len(debate_events) == 7
        assert isinstance(debate_events[0], DebateStartEvent)
        assert isinstance(debate_events[1], DebateArgumentEvent)
        assert isinstance(debate_events[2], DebateArgumentEvent)
        assert isinstance(debate_events[3], DebateArgumentEvent)
        assert isinstance(debate_events[4], DebateArgumentEvent)
        assert isinstance(debate_events[5], DebateResolutionEvent)
        assert isinstance(debate_events[6], DebateCompleteEvent)

    async def test_start_event_data(self) -> None:
        emitter = make_emitter()
        agent_a = make_agent("A", emitter, "arg")
        agent_b = make_agent("B", emitter, "arg")

        debate = Debate(
            debaters=[
                Debater(agent=agent_a, position="pro"),
                Debater(agent=agent_b, position="con"),
            ],
            emitter=emitter,
            resolution=_StubResolution(),
            max_rounds=1,
        )
        await debate.run("test task")

        start_events = [e for e in emitter.events if isinstance(e, DebateStartEvent)]
        assert len(start_events) == 1
        start = start_events[0]
        assert start.task == "test task"
        assert start.debater_names == ["A", "B"]
        assert start.positions == {"A": "pro", "B": "con"}
        assert start.max_rounds == 1
        assert start.resolution_strategy == "_StubResolution"

    async def test_argument_event_data(self) -> None:
        emitter = make_emitter()
        agent_a = make_agent("A", emitter, "pro argument text")
        agent_b = make_agent("B", emitter, "con argument text")

        debate = Debate(
            debaters=[
                Debater(agent=agent_a, position="pro"),
                Debater(agent=agent_b, position="con"),
            ],
            emitter=emitter,
            resolution=_StubResolution(),
            max_rounds=1,
        )
        await debate.run("topic")

        arg_events = [e for e in emitter.events if isinstance(e, DebateArgumentEvent)]
        assert len(arg_events) == 2
        assert arg_events[0].round == 1
        assert arg_events[0].agent_name == "A"
        assert arg_events[0].position == "pro"
        assert "pro argument text" in arg_events[0].argument

    async def test_resolution_event_data(self) -> None:
        emitter = make_emitter()
        agent_a = make_agent("A", emitter, "arg")
        agent_b = make_agent("B", emitter, "arg")

        debate = Debate(
            debaters=[
                Debater(agent=agent_a, position="pro"),
                Debater(agent=agent_b, position="con"),
            ],
            emitter=emitter,
            resolution=_StubResolution("pro"),
            max_rounds=1,
        )
        await debate.run("topic")

        res_events = [e for e in emitter.events if isinstance(e, DebateResolutionEvent)]
        assert len(res_events) == 1
        assert res_events[0].winner == "pro"
        assert res_events[0].rounds_completed == 1

    async def test_complete_event_data(self) -> None:
        emitter = make_emitter()
        agent_a = make_agent("A", emitter, "arg")
        agent_b = make_agent("B", emitter, "arg")

        debate = Debate(
            debaters=[
                Debater(agent=agent_a, position="pro"),
                Debater(agent=agent_b, position="con"),
            ],
            emitter=emitter,
            resolution=_StubResolution("pro"),
            max_rounds=1,
        )
        await debate.run("topic")

        complete_events = [e for e in emitter.events if isinstance(e, DebateCompleteEvent)]
        assert len(complete_events) == 1
        assert complete_events[0].winner == "pro"
        assert complete_events[0].rounds_completed == 1
        assert complete_events[0].total_arguments == 2
        assert complete_events[0].termination_reason == "max_rounds"
