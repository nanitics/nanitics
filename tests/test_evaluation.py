import pytest
from pydantic import ValidationError

from nanitics.capabilities.evaluation import (
    EvaluationCheck,
    EvaluationContext,
    EvaluationResult,
    EvaluationVerdict,
    OutputEvaluator,
    ProgrammaticEvaluator,
)
from nanitics.infrastructure.llm.protocol import Message
from nanitics.infrastructure.observability.events import (
    EvaluationEvent,
    EvaluationExhaustedEvent,
    EvaluationRevisionEvent,
)


def _make_context() -> EvaluationContext:
    return EvaluationContext(
        messages=[Message(role="user", content="test input")],
        task_input="test input",
    )


# --- EvaluationResult ---


class TestEvaluationResult:
    def test_construction(self) -> None:
        result = EvaluationResult(
            verdict=EvaluationVerdict.ACCEPT,
            score=0.9,
            feedback=None,
            evaluator_name="test",
        )
        assert result.verdict == EvaluationVerdict.ACCEPT
        assert result.score == 0.9
        assert result.feedback is None
        assert result.evaluator_name == "test"

    def test_optional_fields(self) -> None:
        result = EvaluationResult(
            verdict=EvaluationVerdict.REVISE,
            evaluator_name="test",
        )
        assert result.score is None
        assert result.feedback is None

    def test_immutability(self) -> None:
        result = EvaluationResult(
            verdict=EvaluationVerdict.ACCEPT,
            evaluator_name="test",
        )
        with pytest.raises(ValidationError):
            result.verdict = EvaluationVerdict.REJECT


# --- EvaluationContext ---


class TestEvaluationContext:
    def test_construction(self) -> None:
        ctx = _make_context()
        assert ctx.task_input == "test input"
        assert len(ctx.messages) == 1

    def test_immutability(self) -> None:
        ctx = _make_context()
        with pytest.raises(ValidationError):
            ctx.task_input = "changed"


# --- EvaluationCheck ---


class TestEvaluationCheck:
    def test_construction(self) -> None:
        check = EvaluationCheck(
            name="length",
            check=lambda output: len(output) > 10,
            feedback="Output too short",
        )
        assert check.name == "length"
        assert check.check("hello world") is True
        assert check.check("short") is False

    def test_immutability(self) -> None:
        check = EvaluationCheck(
            name="test",
            check=lambda _: True,
            feedback="msg",
        )
        with pytest.raises(ValidationError):
            check.name = "changed"


# --- EvaluationVerdict ---


class TestEvaluationVerdict:
    def test_values(self) -> None:
        assert EvaluationVerdict.ACCEPT.value == "accept"
        assert EvaluationVerdict.REVISE.value == "revise"
        assert EvaluationVerdict.REJECT.value == "reject"
        assert EvaluationVerdict.EVALUATOR_ERROR.value == "evaluator_error"


# --- ProgrammaticEvaluator ---


class TestProgrammaticEvaluator:
    async def test_all_checks_pass(self) -> None:
        evaluator = ProgrammaticEvaluator(
            checks=[
                EvaluationCheck(
                    name="non_empty",
                    check=lambda o: len(o) > 0,
                    feedback="Output is empty",
                ),
                EvaluationCheck(
                    name="has_content",
                    check=lambda o: "hello" in o,
                    feedback="Missing hello",
                ),
            ]
        )
        result = await evaluator.evaluate("hello world", _make_context())
        assert result.verdict == EvaluationVerdict.ACCEPT
        assert result.score == 1.0

    async def test_some_checks_fail(self) -> None:
        evaluator = ProgrammaticEvaluator(
            checks=[
                EvaluationCheck(
                    name="non_empty",
                    check=lambda o: len(o) > 0,
                    feedback="Output is empty",
                ),
                EvaluationCheck(
                    name="has_greeting",
                    check=lambda o: "hello" in o,
                    feedback="Missing hello",
                ),
            ]
        )
        result = await evaluator.evaluate("goodbye world", _make_context())
        assert result.verdict == EvaluationVerdict.REVISE
        assert result.score == 0.0
        assert result.feedback is not None
        assert "has_greeting" in result.feedback
        assert "Missing hello" in result.feedback

    async def test_all_checks_fail_aggregates_feedback(self) -> None:
        evaluator = ProgrammaticEvaluator(
            checks=[
                EvaluationCheck(
                    name="check_a",
                    check=lambda _: False,
                    feedback="A failed",
                ),
                EvaluationCheck(
                    name="check_b",
                    check=lambda _: False,
                    feedback="B failed",
                ),
            ]
        )
        result = await evaluator.evaluate("anything", _make_context())
        assert result.verdict == EvaluationVerdict.REVISE
        assert result.feedback is not None
        assert "check_a" in result.feedback
        assert "check_b" in result.feedback
        assert "A failed" in result.feedback
        assert "B failed" in result.feedback

    async def test_never_produces_reject(self) -> None:
        evaluator = ProgrammaticEvaluator(
            checks=[
                EvaluationCheck(
                    name="always_fails",
                    check=lambda _: False,
                    feedback="fail",
                ),
            ]
        )
        result = await evaluator.evaluate("anything", _make_context())
        assert result.verdict == EvaluationVerdict.REVISE

    def test_max_revisions_default(self) -> None:
        evaluator = ProgrammaticEvaluator(checks=[])
        assert evaluator.max_revisions == 1

    def test_max_revisions_custom(self) -> None:
        evaluator = ProgrammaticEvaluator(checks=[], max_revisions=3)
        assert evaluator.max_revisions == 3

    async def test_empty_checks_accepts(self) -> None:
        evaluator = ProgrammaticEvaluator(checks=[])
        result = await evaluator.evaluate("anything", _make_context())
        assert result.verdict == EvaluationVerdict.ACCEPT

    def test_protocol_conformance(self) -> None:
        evaluator = ProgrammaticEvaluator(checks=[])
        assert isinstance(evaluator, OutputEvaluator)


# --- Event Types ---


class TestEvaluationEvent:
    def test_construction(self) -> None:
        event = EvaluationEvent(
            trace_id="t1",
            span_id="s1",
            evaluator_name="programmatic",
            verdict="accept",
            score=1.0,
            feedback=None,
            revision_attempt=0,
        )
        assert event.event_type == "evaluation.result"
        assert event.evaluator_name == "programmatic"
        assert event.verdict == "accept"
        assert event.revision_attempt == 0


class TestEvaluationRevisionEvent:
    def test_construction(self) -> None:
        event = EvaluationRevisionEvent(
            trace_id="t1",
            span_id="s1",
            feedback="fix this",
            revision_attempt=1,
            max_revisions=3,
        )
        assert event.event_type == "evaluation.revision"
        assert event.feedback == "fix this"
        assert event.revision_attempt == 1
        assert event.max_revisions == 3


# ──────────────────────────────────────────────────────────
# Agent Integration Tests
# ──────────────────────────────────────────────────────────

from nanitics.infrastructure import (
    LLMResponse,
    MockLLMClient,
)
from nanitics.strategies import (
    ReActAgent,
    ReasoningAgent,
    tool,
)
from nanitics.tracing import (
    InMemoryEmitter,
    ToolCall,
    Usage,
)


def _make_usage(input_tokens: int = 10, output_tokens: int = 5) -> Usage:
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _make_response(
    content: str | None = "response",
    tool_calls: list[ToolCall] | None = None,
    usage: Usage | None = None,
) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        usage=usage or _make_usage(),
        model="test-model",
        stop_reason="end_turn",
    )


def _make_emitter() -> InMemoryEmitter:
    return InMemoryEmitter(trace_id="test-trace")


@tool(name="add", description="Add two numbers")
async def _add_tool(a: int, b: int) -> str:
    return str(a + b)


def _always_accept_evaluator() -> ProgrammaticEvaluator:
    return ProgrammaticEvaluator(checks=[])


def _reject_then_accept_evaluator(
    fail_keyword: str = "bad",
) -> ProgrammaticEvaluator:
    """Rejects if output contains fail_keyword, accepts otherwise."""
    return ProgrammaticEvaluator(
        checks=[
            EvaluationCheck(
                name="no_bad",
                check=lambda o, kw=fail_keyword: kw not in o,  # type: ignore[misc]
                feedback=f"Output must not contain '{fail_keyword}'",
            ),
        ],
        max_revisions=2,
    )


def _always_reject_evaluator() -> ProgrammaticEvaluator:
    return ProgrammaticEvaluator(
        checks=[
            EvaluationCheck(
                name="impossible",
                check=lambda _: False,
                feedback="Always fails",
            ),
        ],
        max_revisions=2,
    )


class TestReActAgentEvaluation:
    async def test_accepts_on_first_try(self) -> None:
        client = MockLLMClient([_make_response(content="good answer")])
        emitter = _make_emitter()
        agent = ReActAgent(
            name="eval-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[_add_tool],
            output_evaluator=_always_accept_evaluator(),
        )

        result = await agent.run("Hi")

        assert result.output == "good answer"
        assert result.termination_reason == "complete"
        assert result.total_steps == 1

    async def test_revise_then_accept(self) -> None:
        """Evaluator rejects first output (contains 'bad'), agent retries, second output accepted."""
        responses = [
            _make_response(content="bad answer"),
            _make_response(content="good answer"),
        ]
        client = MockLLMClient(responses)
        emitter = _make_emitter()
        agent = ReActAgent(
            name="eval-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[_add_tool],
            output_evaluator=_reject_then_accept_evaluator(),
        )

        result = await agent.run("Hi")

        assert result.output == "good answer"
        assert result.termination_reason == "complete"

    async def test_always_rejects_budget_exhaustion(self) -> None:
        """Evaluator always rejects. Budget exhausted → evaluation_failed."""
        responses = [
            _make_response(content="attempt 1"),
            _make_response(content="attempt 2"),
            _make_response(content="attempt 3"),
        ]
        client = MockLLMClient(responses)
        emitter = _make_emitter()
        agent = ReActAgent(
            name="eval-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[_add_tool],
            output_evaluator=_always_reject_evaluator(),
        )

        result = await agent.run("Hi")

        assert result.termination_reason == "evaluation_failed"
        # Output is the last attempt
        assert result.output == "attempt 3"

        exhausted = [e for e in emitter.events if isinstance(e, EvaluationExhaustedEvent)]
        assert len(exhausted) == 1
        assert exhausted[0].evaluator_name == "programmatic"
        assert exhausted[0].verdict == "revise"
        assert exhausted[0].revision_count == 2
        assert exhausted[0].max_revisions == 2
        assert exhausted[0].feedback is not None

    async def test_evaluation_events_emitted(self) -> None:
        """Evaluation events are emitted on accept."""
        client = MockLLMClient([_make_response(content="good answer")])
        emitter = _make_emitter()
        agent = ReActAgent(
            name="eval-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[_add_tool],
            output_evaluator=_always_accept_evaluator(),
        )

        await agent.run("Hi")

        eval_events = [e for e in emitter.events if isinstance(e, EvaluationEvent)]
        assert len(eval_events) == 1
        assert eval_events[0].verdict == "accept"
        assert eval_events[0].revision_attempt == 0

    async def test_revision_events_emitted(self) -> None:
        """EvaluationRevisionEvent emitted on each retry."""
        responses = [
            _make_response(content="bad answer"),
            _make_response(content="good answer"),
        ]
        client = MockLLMClient(responses)
        emitter = _make_emitter()
        agent = ReActAgent(
            name="eval-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[_add_tool],
            output_evaluator=_reject_then_accept_evaluator(),
        )

        await agent.run("Hi")

        revision_events = [e for e in emitter.events if isinstance(e, EvaluationRevisionEvent)]
        assert len(revision_events) == 1
        assert revision_events[0].revision_attempt == 0

        eval_events = [e for e in emitter.events if isinstance(e, EvaluationEvent)]
        # Two evaluation events: first reject (attempt 0), then accept (attempt 1)
        assert len(eval_events) == 2
        assert eval_events[0].verdict == "revise"
        assert eval_events[1].verdict == "accept"

    async def test_evaluation_retry_consumes_iteration(self) -> None:
        """Evaluation retry uses a loop iteration. If budget runs out, iteration_limit applies."""
        responses = [
            _make_response(content="bad answer"),
            _make_response(content="bad answer"),
            _make_response(content="bad answer"),
        ]
        client = MockLLMClient(responses)
        emitter = _make_emitter()
        agent = ReActAgent(
            name="eval-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[_add_tool],
            max_iterations=2,
            output_evaluator=_always_reject_evaluator(),
        )

        result = await agent.run("Hi")

        # The agent should hit the iteration limit before exhausting evaluation budget
        assert result.termination_reason == "iteration_limit"

    async def test_no_evaluator_unchanged_behavior(self) -> None:
        """Agent without evaluator behaves exactly as before."""
        client = MockLLMClient([_make_response(content="answer")])
        emitter = _make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[_add_tool],
        )

        result = await agent.run("Hi")

        assert result.output == "answer"
        assert result.termination_reason == "complete"
        eval_events = [e for e in emitter.events if isinstance(e, EvaluationEvent)]
        assert len(eval_events) == 0

    async def test_evaluation_with_tool_use(self) -> None:
        """Agent uses tools, then evaluates final output."""
        tool_call = ToolCall(id="tc1", name="add", arguments={"a": 2, "b": 3})
        responses = [
            _make_response(content="Let me add", tool_calls=[tool_call]),
            _make_response(content="The answer is 5"),
        ]
        client = MockLLMClient(responses)
        emitter = _make_emitter()
        agent = ReActAgent(
            name="eval-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[_add_tool],
            output_evaluator=_always_accept_evaluator(),
        )

        result = await agent.run("Add 2 and 3")

        assert result.output == "The answer is 5"
        assert result.termination_reason == "complete"
        eval_events = [e for e in emitter.events if isinstance(e, EvaluationEvent)]
        assert len(eval_events) == 1


class TestReasoningAgentEvaluation:
    async def test_accepts_on_first_try(self) -> None:
        client = MockLLMClient([_make_response(content="good answer")])
        emitter = _make_emitter()
        agent = ReasoningAgent(
            name="eval-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            output_evaluator=_always_accept_evaluator(),
        )

        result = await agent.run("Hi")

        assert result.output == "good answer"
        assert result.termination_reason == "complete"
        assert result.total_steps == 1

    async def test_revise_then_accept(self) -> None:
        responses = [
            _make_response(content="bad answer"),
            _make_response(content="good answer"),
        ]
        client = MockLLMClient(responses)
        emitter = _make_emitter()
        agent = ReasoningAgent(
            name="eval-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            output_evaluator=_reject_then_accept_evaluator(),
        )

        result = await agent.run("Hi")

        assert result.output == "good answer"
        assert result.termination_reason == "complete"
        assert result.total_steps == 2

    async def test_always_rejects_budget_exhaustion(self) -> None:
        responses = [
            _make_response(content="attempt 1"),
            _make_response(content="attempt 2"),
            _make_response(content="attempt 3"),
        ]
        client = MockLLMClient(responses)
        emitter = _make_emitter()
        agent = ReasoningAgent(
            name="eval-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            output_evaluator=_always_reject_evaluator(),
        )

        result = await agent.run("Hi")

        assert result.termination_reason == "evaluation_failed"
        assert result.output == "attempt 3"
        assert result.total_steps == 3

        exhausted = [e for e in emitter.events if isinstance(e, EvaluationExhaustedEvent)]
        assert len(exhausted) == 1
        assert exhausted[0].evaluator_name == "programmatic"
        assert exhausted[0].verdict == "revise"
        assert exhausted[0].revision_count == 2
        assert exhausted[0].max_revisions == 2

    async def test_evaluation_events_emitted(self) -> None:
        responses = [
            _make_response(content="bad answer"),
            _make_response(content="good answer"),
        ]
        client = MockLLMClient(responses)
        emitter = _make_emitter()
        agent = ReasoningAgent(
            name="eval-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            output_evaluator=_reject_then_accept_evaluator(),
        )

        await agent.run("Hi")

        eval_events = [e for e in emitter.events if isinstance(e, EvaluationEvent)]
        assert len(eval_events) == 2
        assert eval_events[0].verdict == "revise"
        assert eval_events[1].verdict == "accept"

        revision_events = [e for e in emitter.events if isinstance(e, EvaluationRevisionEvent)]
        assert len(revision_events) == 1

    async def test_no_evaluator_unchanged_behavior(self) -> None:
        client = MockLLMClient([_make_response(content="answer")])
        emitter = _make_emitter()
        agent = ReasoningAgent(
            name="reasoning-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
        )

        result = await agent.run("Hi")

        assert result.output == "answer"
        assert result.termination_reason == "complete"
        assert result.total_steps == 1


# ──────────────────────────────────────────────────────────
# LLMEvaluator & CompositeEvaluator Tests
# ──────────────────────────────────────────────────────────

import json

from nanitics.capabilities.evaluation import CompositeEvaluator, LLMEvaluator


def _make_llm_eval_response(score: float, reasoning: str, issues: list[str]) -> LLMResponse:
    content = json.dumps({"score": score, "reasoning": reasoning, "issues": issues})
    return _make_response(content=content)


class TestLLMEvaluator:
    async def test_accept_above_threshold(self) -> None:
        client = MockLLMClient([_make_llm_eval_response(0.9, "Good output", [])])
        evaluator = LLMEvaluator(llm_client=client, criteria="Be accurate.", score_threshold=0.7)

        result = await evaluator.evaluate("good answer", _make_context())

        assert result.verdict == EvaluationVerdict.ACCEPT
        assert result.score == 0.9
        assert result.feedback is None
        assert result.evaluator_name == "llm"

    async def test_revise_below_threshold(self) -> None:
        client = MockLLMClient(
            [_make_llm_eval_response(0.3, "Missing key details", ["No sources cited", "Incomplete"])]
        )
        evaluator = LLMEvaluator(llm_client=client, criteria="Be thorough.", score_threshold=0.7)

        result = await evaluator.evaluate("weak answer", _make_context())

        assert result.verdict == EvaluationVerdict.REVISE
        assert result.score == 0.3
        assert result.feedback is not None
        assert "Missing key details" in result.feedback
        assert "No sources cited" in result.feedback
        assert "Incomplete" in result.feedback

    async def test_exact_threshold_accepts(self) -> None:
        client = MockLLMClient([_make_llm_eval_response(0.7, "Meets minimum", [])])
        evaluator = LLMEvaluator(llm_client=client, criteria="Be accurate.", score_threshold=0.7)

        result = await evaluator.evaluate("ok answer", _make_context())

        assert result.verdict == EvaluationVerdict.ACCEPT
        assert result.score == 0.7

    async def test_unparseable_response_returns_evaluator_error(self) -> None:
        bad_response = _make_response(content="not valid json")
        client = MockLLMClient([bad_response])
        evaluator = LLMEvaluator(llm_client=client, criteria="Be accurate.", score_threshold=0.7)

        result = await evaluator.evaluate("answer", _make_context())

        assert result.verdict == EvaluationVerdict.EVALUATOR_ERROR
        assert result.score is None
        assert result.feedback is not None
        assert "failed" in result.feedback.lower()

    async def test_passes_correct_schema_to_client(self) -> None:
        client = MockLLMClient([_make_llm_eval_response(0.9, "Good", [])])
        evaluator = LLMEvaluator(llm_client=client, criteria="Be accurate.")

        await evaluator.evaluate("answer", _make_context())

        assert len(client.calls) == 1
        assert client.calls[0]["output_schema"] is not None

    async def test_prompt_contains_task_input_and_criteria(self) -> None:
        client = MockLLMClient([_make_llm_eval_response(0.9, "Good", [])])
        evaluator = LLMEvaluator(llm_client=client, criteria="Must cite sources.")
        ctx = EvaluationContext(
            messages=[Message(role="user", content="What is AI?")],
            task_input="What is AI?",
        )

        await evaluator.evaluate("AI is cool", ctx)

        user_message = client.calls[0]["messages"][0].content
        assert "What is AI?" in user_message
        assert "Must cite sources." in user_message
        assert "AI is cool" in user_message

    def test_max_revisions_default(self) -> None:
        client = MockLLMClient([])
        evaluator = LLMEvaluator(llm_client=client, criteria="test")
        assert evaluator.max_revisions == 1

    def test_max_revisions_custom(self) -> None:
        client = MockLLMClient([])
        evaluator = LLMEvaluator(llm_client=client, criteria="test", max_revisions=3)
        assert evaluator.max_revisions == 3

    def test_protocol_conformance(self) -> None:
        client = MockLLMClient([])
        evaluator = LLMEvaluator(llm_client=client, criteria="test")
        assert isinstance(evaluator, OutputEvaluator)

    async def test_no_issues_feedback_is_reasoning_only(self) -> None:
        client = MockLLMClient([_make_llm_eval_response(0.4, "Generally weak", [])])
        evaluator = LLMEvaluator(llm_client=client, criteria="Be good.", score_threshold=0.7)

        result = await evaluator.evaluate("meh", _make_context())

        assert result.verdict == EvaluationVerdict.REVISE
        assert result.feedback == "Generally weak"


class TestCompositeEvaluator:
    async def test_all_accept(self) -> None:
        prog = ProgrammaticEvaluator(checks=[])
        llm_client = MockLLMClient([_make_llm_eval_response(0.9, "Good", [])])
        llm_eval = LLMEvaluator(llm_client=llm_client, criteria="Be accurate.")
        composite = CompositeEvaluator(evaluators=[prog, llm_eval])

        result = await composite.evaluate("good answer", _make_context())

        assert result.verdict == EvaluationVerdict.ACCEPT
        # LLM evaluator ran (last evaluator)
        assert result.evaluator_name == "llm"
        assert result.score == 0.9

    async def test_programmatic_rejects_llm_not_called(self) -> None:
        prog = ProgrammaticEvaluator(
            checks=[
                EvaluationCheck(
                    name="has_hello",
                    check=lambda o: "hello" in o,
                    feedback="Must contain hello",
                ),
            ]
        )
        llm_client = MockLLMClient([])  # No responses — should not be called
        llm_eval = LLMEvaluator(llm_client=llm_client, criteria="Be accurate.")
        composite = CompositeEvaluator(evaluators=[prog, llm_eval])

        result = await composite.evaluate("goodbye", _make_context())

        assert result.verdict == EvaluationVerdict.REVISE
        assert result.evaluator_name == "programmatic"
        # LLM client was never called
        assert len(llm_client.calls) == 0

    async def test_short_circuits_on_revise(self) -> None:
        """Second evaluator revises; third evaluator never runs."""
        prog_pass = ProgrammaticEvaluator(checks=[])
        prog_fail = ProgrammaticEvaluator(
            checks=[
                EvaluationCheck(
                    name="always_fail",
                    check=lambda _: False,
                    feedback="Fail",
                ),
            ]
        )
        llm_client = MockLLMClient([])
        llm_eval = LLMEvaluator(llm_client=llm_client, criteria="test")
        composite = CompositeEvaluator(evaluators=[prog_pass, prog_fail, llm_eval])

        result = await composite.evaluate("anything", _make_context())

        assert result.verdict == EvaluationVerdict.REVISE
        assert len(llm_client.calls) == 0

    async def test_empty_evaluators_accepts(self) -> None:
        composite = CompositeEvaluator(evaluators=[])

        result = await composite.evaluate("anything", _make_context())

        assert result.verdict == EvaluationVerdict.ACCEPT

    def test_max_revisions_default(self) -> None:
        composite = CompositeEvaluator(evaluators=[])
        assert composite.max_revisions == 1

    def test_max_revisions_custom(self) -> None:
        composite = CompositeEvaluator(evaluators=[], max_revisions=5)
        assert composite.max_revisions == 5

    def test_protocol_conformance(self) -> None:
        composite = CompositeEvaluator(evaluators=[])
        assert isinstance(composite, OutputEvaluator)


class TestCompositeEvaluatorAgentIntegration:
    async def test_react_agent_with_composite(self) -> None:
        """End-to-end: ReActAgent with composite evaluator (programmatic + LLM)."""
        prog = ProgrammaticEvaluator(
            checks=[
                EvaluationCheck(
                    name="non_empty",
                    check=lambda o: len(o) > 0,
                    feedback="Output is empty",
                ),
            ]
        )
        llm_eval_client = MockLLMClient([_make_llm_eval_response(0.9, "Good output", [])])
        llm_eval = LLMEvaluator(llm_client=llm_eval_client, criteria="Be accurate.")
        composite = CompositeEvaluator(evaluators=[prog, llm_eval])

        agent_client = MockLLMClient([_make_response(content="great answer")])
        emitter = _make_emitter()
        agent = ReActAgent(
            name="eval-agent",
            llm_client=agent_client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[_add_tool],
            output_evaluator=composite,
        )

        result = await agent.run("Hi")

        assert result.output == "great answer"
        assert result.termination_reason == "complete"

    async def test_composite_reject_then_accept(self) -> None:
        """Composite rejects first output (programmatic), agent retries, second accepted."""
        prog = ProgrammaticEvaluator(
            checks=[
                EvaluationCheck(
                    name="no_bad",
                    check=lambda o: "bad" not in o,
                    feedback="Must not contain bad",
                ),
            ]
        )
        llm_eval_client = MockLLMClient([_make_llm_eval_response(0.9, "Good", [])])
        llm_eval = LLMEvaluator(llm_client=llm_eval_client, criteria="Be accurate.")
        composite = CompositeEvaluator(evaluators=[prog, llm_eval], max_revisions=2)

        agent_client = MockLLMClient(
            [
                _make_response(content="bad answer"),
                _make_response(content="good answer"),
            ]
        )
        emitter = _make_emitter()
        agent = ReActAgent(
            name="eval-agent",
            llm_client=agent_client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[_add_tool],
            output_evaluator=composite,
        )

        result = await agent.run("Hi")

        assert result.output == "good answer"
        assert result.termination_reason == "complete"

        # First eval: programmatic rejected (LLM not called)
        # Second eval: programmatic passed, LLM accepted
        assert len(llm_eval_client.calls) == 1


# ──────────────────────────────────────────────────────────
# Truncation-Aware Evaluation Tests
# ──────────────────────────────────────────────────────────


def _make_truncated_response(
    content: str = "truncated output",
    usage: Usage | None = None,
) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=[],
        usage=usage or _make_usage(),
        model="test-model",
        stop_reason="max_tokens",
    )


class TestReActAgentTruncation:
    async def test_truncated_output_skips_evaluation(self) -> None:
        """Truncated response skips evaluator and sends truncation feedback."""
        responses = [
            _make_truncated_response(content="cut off report"),
            _make_response(content="concise report"),
        ]
        client = MockLLMClient(responses)
        emitter = _make_emitter()
        agent = ReActAgent(
            name="eval-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[_add_tool],
            output_evaluator=_always_accept_evaluator(),
        )

        result = await agent.run("Write a report")

        assert result.output == "concise report"
        assert result.termination_reason == "complete"

        eval_events = [e for e in emitter.events if isinstance(e, EvaluationEvent)]
        # First: truncation event, Second: accept from evaluator
        assert len(eval_events) == 2
        assert eval_events[0].evaluator_name == "truncation"
        assert eval_events[0].verdict == "revise"
        assert eval_events[1].verdict == "accept"

    async def test_truncated_output_then_normal_evaluated(self) -> None:
        """First response truncated, second normal — evaluator runs on second."""
        responses = [
            _make_truncated_response(content="too long"),
            _make_response(content="good answer"),
        ]
        client = MockLLMClient(responses)
        emitter = _make_emitter()
        evaluator = _reject_then_accept_evaluator(fail_keyword="bad")
        agent = ReActAgent(
            name="eval-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[_add_tool],
            output_evaluator=evaluator,
        )

        result = await agent.run("Write something")

        assert result.output == "good answer"
        assert result.termination_reason == "complete"

    async def test_truncated_output_exhausts_budget(self) -> None:
        """Truncated response when revision budget is exhausted → evaluation_failed."""
        evaluator = ProgrammaticEvaluator(checks=[], max_revisions=0)
        responses = [
            _make_truncated_response(content="too long"),
        ]
        client = MockLLMClient(responses)
        emitter = _make_emitter()
        agent = ReActAgent(
            name="eval-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[_add_tool],
            output_evaluator=evaluator,
        )

        result = await agent.run("Write something")

        assert result.termination_reason == "evaluation_failed"
        assert result.output == "too long"

        exhausted = [e for e in emitter.events if isinstance(e, EvaluationExhaustedEvent)]
        assert len(exhausted) == 1
        assert exhausted[0].evaluator_name == "truncation"
        assert exhausted[0].verdict == "revise"
        assert exhausted[0].max_revisions == 0

    async def test_truncated_output_no_evaluator_proceeds(self) -> None:
        """Agent without evaluator returns truncated output as-is."""
        responses = [
            _make_truncated_response(content="cut off response"),
        ]
        client = MockLLMClient(responses)
        emitter = _make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[_add_tool],
        )

        result = await agent.run("Hi")

        assert result.output == "cut off response"
        assert result.termination_reason == "complete"
        eval_events = [e for e in emitter.events if isinstance(e, EvaluationEvent)]
        assert len(eval_events) == 0


# ──────────────────────────────────────────────────────────
# Hardening Tests
# ──────────────────────────────────────────────────────────


class TestMaxRevisionsValidation:
    def test_programmatic_evaluator_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            ProgrammaticEvaluator(checks=[], max_revisions=-1)

    def test_llm_evaluator_rejects_negative(self) -> None:
        client = MockLLMClient([])
        with pytest.raises(ValueError, match="non-negative"):
            LLMEvaluator(llm_client=client, criteria="test", max_revisions=-1)

    def test_composite_evaluator_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            CompositeEvaluator(evaluators=[], max_revisions=-1)

    def test_programmatic_evaluator_accepts_zero(self) -> None:
        evaluator = ProgrammaticEvaluator(checks=[], max_revisions=0)
        assert evaluator.max_revisions == 0

    def test_llm_evaluator_accepts_zero(self) -> None:
        client = MockLLMClient([])
        evaluator = LLMEvaluator(llm_client=client, criteria="test", max_revisions=0)
        assert evaluator.max_revisions == 0

    def test_composite_evaluator_accepts_zero(self) -> None:
        evaluator = CompositeEvaluator(evaluators=[], max_revisions=0)
        assert evaluator.max_revisions == 0


class TestScoreValidation:
    def test_score_in_range_accepted(self) -> None:
        result = EvaluationResult(
            verdict=EvaluationVerdict.ACCEPT,
            score=0.5,
            evaluator_name="test",
        )
        assert result.score == 0.5

    def test_score_at_boundaries(self) -> None:
        low = EvaluationResult(verdict=EvaluationVerdict.ACCEPT, score=0.0, evaluator_name="test")
        high = EvaluationResult(verdict=EvaluationVerdict.ACCEPT, score=1.0, evaluator_name="test")
        assert low.score == 0.0
        assert high.score == 1.0

    def test_score_above_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvaluationResult(
                verdict=EvaluationVerdict.ACCEPT,
                score=1.5,
                evaluator_name="test",
            )

    def test_score_below_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvaluationResult(
                verdict=EvaluationVerdict.ACCEPT,
                score=-0.1,
                evaluator_name="test",
            )

    def test_score_none_accepted(self) -> None:
        result = EvaluationResult(
            verdict=EvaluationVerdict.ACCEPT,
            score=None,
            evaluator_name="test",
        )
        assert result.score is None


class _ErrorThenSuccessClient:
    """Client that raises on first call, then returns a response on second."""

    def __init__(self, error: Exception, response: LLMResponse) -> None:
        self._error = error
        self._response = response
        self._call_count = 0

    async def generate(self, *, output_schema: type | None = None, **kwargs: object) -> LLMResponse:
        self._call_count += 1
        if self._call_count == 1:
            raise self._error
        response = self._response
        if output_schema is not None and response.content is not None:
            parsed = output_schema.model_validate_json(response.content)  # type: ignore[attr-defined]
            response = response.model_copy(update={"parsed": parsed})
        return response

    @property
    def call_count(self) -> int:
        return self._call_count


class _AlwaysErrorClient:
    """Client that always raises the given error."""

    def __init__(self, error: Exception) -> None:
        self._error = error
        self._call_count = 0

    @property
    def model(self) -> str | None:
        return None

    async def generate(self, **kwargs: object) -> LLMResponse:
        self._call_count += 1
        raise self._error

    @property
    def call_count(self) -> int:
        return self._call_count


class TestLLMEvaluatorErrorHandling:
    async def test_llm_error_returns_evaluator_error(self) -> None:
        """LLM provider error during evaluation returns EVALUATOR_ERROR with error_detail."""
        from nanitics.infrastructure.errors import LLMProviderError

        # Use _AlwaysErrorClient — MockLLMClient([]) now raises ValueError on
        # exhaustion, which is not an LLMError and would propagate out of the
        # evaluator. This test exercises the evaluator's `except LLMError:` branch.
        client = _AlwaysErrorClient(LLMProviderError("Server error"))
        evaluator = LLMEvaluator(llm_client=client, criteria="Be accurate.")

        result = await evaluator.evaluate("some output", _make_context())

        assert result.verdict == EvaluationVerdict.EVALUATOR_ERROR
        assert result.score is None
        assert result.feedback is not None
        assert "failed" in result.feedback.lower()
        assert result.evaluator_name == "llm"
        assert result.error_detail is not None
        assert "LLMProviderError" in result.error_detail

    async def test_rate_limit_retry_succeeds(self) -> None:
        """LLMRateLimitError triggers a retry; success on second attempt."""
        from nanitics.infrastructure.errors import LLMRateLimitError

        error = LLMRateLimitError("Rate limited", retry_after=0.0)
        response = _make_llm_eval_response(0.9, "Good output", [])
        client = _ErrorThenSuccessClient(error, response)
        evaluator = LLMEvaluator(llm_client=client, criteria="Be accurate.")

        result = await evaluator.evaluate("some output", _make_context())

        assert result.verdict == EvaluationVerdict.ACCEPT
        assert result.score == 0.9
        assert client.call_count == 2

    async def test_rate_limit_retry_fails_returns_error(self) -> None:
        """LLMRateLimitError retry that also fails returns EVALUATOR_ERROR with detail."""
        from nanitics.infrastructure.errors import LLMRateLimitError

        error = LLMRateLimitError("Rate limited", retry_after=0.0)
        client = _AlwaysErrorClient(error)
        evaluator = LLMEvaluator(llm_client=client, criteria="Be accurate.")

        result = await evaluator.evaluate("some output", _make_context())

        assert result.verdict == EvaluationVerdict.EVALUATOR_ERROR
        assert result.error_detail is not None
        assert "LLMRateLimitError" in result.error_detail
        assert client.call_count == 2  # original + 1 retry

    async def test_provider_error_no_retry(self) -> None:
        """Non-rate-limit LLMError fails immediately without retry."""
        from nanitics.infrastructure.errors import LLMProviderError

        error = LLMProviderError("Server error")
        client = _AlwaysErrorClient(error)
        evaluator = LLMEvaluator(llm_client=client, criteria="Be accurate.")

        result = await evaluator.evaluate("some output", _make_context())

        assert result.verdict == EvaluationVerdict.EVALUATOR_ERROR
        assert result.error_detail is not None
        assert "LLMProviderError" in result.error_detail
        assert client.call_count == 1  # no retry

    async def test_parse_failure_includes_error_detail(self) -> None:
        """Unparseable response includes error_detail."""
        bad_response = _make_response(content="not valid json")
        client = MockLLMClient([bad_response])
        evaluator = LLMEvaluator(llm_client=client, criteria="Be accurate.")

        result = await evaluator.evaluate("answer", _make_context())

        assert result.verdict == EvaluationVerdict.EVALUATOR_ERROR
        assert result.error_detail is not None
        assert "LLMSchemaViolationError" in result.error_detail

    async def test_parsed_none_includes_error_detail(self) -> None:
        """Response with parsed=None returns error_detail about schema mismatch."""

        class _NullParsedClient:
            async def generate(self, **kwargs: object) -> LLMResponse:
                return LLMResponse(
                    content="some content",
                    tool_calls=[],
                    usage=_make_usage(),
                    model="test",
                    stop_reason="end_turn",
                    parsed=None,
                )

        evaluator = LLMEvaluator(llm_client=_NullParsedClient(), criteria="Be accurate.")
        result = await evaluator.evaluate("answer", _make_context())

        assert result.verdict == EvaluationVerdict.EVALUATOR_ERROR
        assert result.error_detail == "Response did not match expected schema"


# ──────────────────────────────────────────────────────────
# EvaluationContext Tree Search Metadata
# ──────────────────────────────────────────────────────────


class TestEvaluationContextTreeSearchFields:
    def test_optional_fields_default_to_none(self) -> None:
        ctx = _make_context()
        assert ctx.depth is None
        assert ctx.max_depth is None
        assert ctx.trajectory_length is None
        assert ctx.total_nodes_explored is None

    def test_optional_fields_set(self) -> None:
        ctx = EvaluationContext(
            messages=[Message(role="user", content="test")],
            task_input="test",
            depth=3,
            max_depth=5,
            trajectory_length=4,
            total_nodes_explored=12,
        )
        assert ctx.depth == 3
        assert ctx.max_depth == 5
        assert ctx.trajectory_length == 4
        assert ctx.total_nodes_explored == 12

    def test_immutability_of_new_fields(self) -> None:
        ctx = EvaluationContext(
            messages=[Message(role="user", content="test")],
            task_input="test",
            depth=3,
        )
        with pytest.raises(ValidationError):
            ctx.depth = 5

    async def test_existing_evaluators_unaffected(self) -> None:
        """ProgrammaticEvaluator works with context containing new fields."""
        evaluator = ProgrammaticEvaluator(
            checks=[
                EvaluationCheck(
                    name="non_empty",
                    check=lambda o: len(o) > 0,
                    feedback="Empty",
                ),
            ]
        )
        ctx = EvaluationContext(
            messages=[Message(role="user", content="test")],
            task_input="test",
            depth=3,
            max_depth=5,
            total_nodes_explored=10,
        )
        result = await evaluator.evaluate("hello", ctx)
        assert result.verdict == EvaluationVerdict.ACCEPT


# ──────────────────────────────────────────────────────────
# LLMEvaluator Depth Context in Prompt
# ──────────────────────────────────────────────────────────

from nanitics.capabilities.evaluation.llm_evaluator import _build_evaluation_prompt


class TestLLMEvaluatorDepthContext:
    def test_prompt_excludes_depth_when_none(self) -> None:
        ctx = EvaluationContext(
            messages=[],
            task_input="test task",
        )
        prompt = _build_evaluation_prompt("output", "test task", "criteria", context=ctx)
        assert "Exploration Context" not in prompt

    def test_prompt_includes_depth_when_set(self) -> None:
        ctx = EvaluationContext(
            messages=[],
            task_input="test task",
            depth=2,
            max_depth=5,
            total_nodes_explored=8,
        )
        prompt = _build_evaluation_prompt("output", "test task", "criteria", context=ctx)
        assert "Exploration Context" in prompt
        assert "depth 2 of 5" in prompt
        assert "8 nodes have been explored" in prompt

    def test_prompt_includes_trajectory_length(self) -> None:
        ctx = EvaluationContext(
            messages=[],
            task_input="test task",
            depth=3,
            max_depth=5,
            trajectory_length=4,
            total_nodes_explored=12,
        )
        prompt = _build_evaluation_prompt("output", "test task", "criteria", context=ctx)
        assert "trajectory from root to this node is 4 steps" in prompt

    def test_prompt_omits_trajectory_when_none(self) -> None:
        ctx = EvaluationContext(
            messages=[],
            task_input="test task",
            depth=2,
            max_depth=5,
        )
        prompt = _build_evaluation_prompt("output", "test task", "criteria", context=ctx)
        assert "Exploration Context" in prompt
        assert "trajectory" not in prompt

    def test_prompt_without_context_backward_compatible(self) -> None:
        prompt = _build_evaluation_prompt("output", "test task", "criteria")
        assert "Exploration Context" not in prompt
        assert "test task" in prompt
        assert "output" in prompt
        assert "criteria" in prompt

    async def test_llm_evaluator_passes_context_to_prompt(self) -> None:
        client = MockLLMClient([_make_llm_eval_response(0.9, "Good", [])])
        evaluator = LLMEvaluator(llm_client=client, criteria="Be accurate.", score_threshold=0.7)

        ctx = EvaluationContext(
            messages=[Message(role="user", content="What is AI?")],
            task_input="What is AI?",
            depth=2,
            max_depth=4,
            total_nodes_explored=6,
        )
        await evaluator.evaluate("AI answer", ctx)

        user_message = client.calls[0]["messages"][0].content
        assert "Exploration Context" in user_message
        assert "depth 2 of 4" in user_message


class TestLLMEvaluatorRejectThreshold:
    async def test_score_below_reject_threshold_returns_reject(self) -> None:
        client = MockLLMClient([_make_llm_eval_response(0.2, "Very poor", ["Fundamentally wrong"])])
        evaluator = LLMEvaluator(llm_client=client, criteria="Be accurate.", score_threshold=0.7, reject_threshold=0.4)

        result = await evaluator.evaluate("terrible answer", _make_context())

        assert result.verdict == EvaluationVerdict.REJECT
        assert result.score == 0.2
        assert result.feedback is not None
        assert "Very poor" in result.feedback

    async def test_score_between_reject_and_accept_returns_revise(self) -> None:
        client = MockLLMClient([_make_llm_eval_response(0.5, "Needs work", ["Missing sources"])])
        evaluator = LLMEvaluator(llm_client=client, criteria="Be accurate.", score_threshold=0.7, reject_threshold=0.4)

        result = await evaluator.evaluate("mediocre answer", _make_context())

        assert result.verdict == EvaluationVerdict.REVISE
        assert result.score == 0.5
        assert result.feedback is not None

    async def test_score_above_accept_threshold_returns_accept(self) -> None:
        client = MockLLMClient([_make_llm_eval_response(0.9, "Excellent", [])])
        evaluator = LLMEvaluator(llm_client=client, criteria="Be accurate.", score_threshold=0.7, reject_threshold=0.4)

        result = await evaluator.evaluate("great answer", _make_context())

        assert result.verdict == EvaluationVerdict.ACCEPT
        assert result.score == 0.9

    async def test_no_reject_threshold_never_produces_reject(self) -> None:
        client = MockLLMClient([_make_llm_eval_response(0.1, "Very poor", ["Wrong"])])
        evaluator = LLMEvaluator(llm_client=client, criteria="Be accurate.", score_threshold=0.7)

        result = await evaluator.evaluate("terrible answer", _make_context())

        assert result.verdict == EvaluationVerdict.REVISE
        assert result.score == 0.1

    def test_reject_threshold_gte_score_threshold_raises(self) -> None:
        client = MockLLMClient([])
        with pytest.raises(ValueError, match=r"reject_threshold.*must be less than.*score_threshold"):
            LLMEvaluator(llm_client=client, criteria="test", score_threshold=0.7, reject_threshold=0.7)

    def test_reject_threshold_above_score_threshold_raises(self) -> None:
        client = MockLLMClient([])
        with pytest.raises(ValueError, match=r"reject_threshold.*must be less than.*score_threshold"):
            LLMEvaluator(llm_client=client, criteria="test", score_threshold=0.7, reject_threshold=0.8)


class TestRevisionAttemptConsistency:
    async def test_evaluation_and_revision_events_share_index(self) -> None:
        """EvaluationEvent and EvaluationRevisionEvent use same revision_attempt value."""
        responses = [
            _make_response(content="bad answer"),
            _make_response(content="good answer"),
        ]
        client = MockLLMClient(responses)
        emitter = _make_emitter()
        agent = ReActAgent(
            name="eval-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[_add_tool],
            output_evaluator=_reject_then_accept_evaluator(),
        )

        await agent.run("Hi")

        eval_events = [e for e in emitter.events if isinstance(e, EvaluationEvent)]
        revision_events = [e for e in emitter.events if isinstance(e, EvaluationRevisionEvent)]

        # First eval: revision_attempt=0, verdict=revise
        assert eval_events[0].revision_attempt == 0
        assert eval_events[0].verdict == "revise"
        # Revision event should match: revision_attempt=0
        assert revision_events[0].revision_attempt == 0
        # Second eval: revision_attempt=1, verdict=accept
        assert eval_events[1].revision_attempt == 1
        assert eval_events[1].verdict == "accept"


class TestReasoningAgentTruncation:
    async def test_truncated_output_skips_evaluation(self) -> None:
        """Truncated response skips evaluator and sends truncation feedback."""
        responses = [
            _make_truncated_response(content="cut off"),
            _make_response(content="concise answer"),
        ]
        client = MockLLMClient(responses)
        emitter = _make_emitter()
        agent = ReasoningAgent(
            name="eval-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            output_evaluator=_always_accept_evaluator(),
        )

        result = await agent.run("Hi")

        assert result.output == "concise answer"
        assert result.termination_reason == "complete"

        eval_events = [e for e in emitter.events if isinstance(e, EvaluationEvent)]
        assert len(eval_events) == 2
        assert eval_events[0].evaluator_name == "truncation"
        assert eval_events[0].verdict == "revise"
        assert eval_events[1].verdict == "accept"

    async def test_truncated_output_exhausts_budget(self) -> None:
        """Truncated response when budget exhausted → evaluation_failed."""
        evaluator = ProgrammaticEvaluator(checks=[], max_revisions=0)
        responses = [
            _make_truncated_response(content="cut off"),
        ]
        client = MockLLMClient(responses)
        emitter = _make_emitter()
        agent = ReasoningAgent(
            name="eval-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            output_evaluator=evaluator,
        )

        result = await agent.run("Hi")

        assert result.termination_reason == "evaluation_failed"
        assert result.output == "cut off"

        exhausted = [e for e in emitter.events if isinstance(e, EvaluationExhaustedEvent)]
        assert len(exhausted) == 1
        assert exhausted[0].evaluator_name == "truncation"
        assert exhausted[0].verdict == "revise"
        assert exhausted[0].max_revisions == 0

    async def test_truncated_output_no_evaluator_proceeds(self) -> None:
        """Agent without evaluator returns truncated output as-is."""
        responses = [
            _make_truncated_response(content="cut off response"),
        ]
        client = MockLLMClient(responses)
        emitter = _make_emitter()
        agent = ReasoningAgent(
            name="reasoning-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
        )

        result = await agent.run("Hi")

        assert result.output == "cut off response"
        assert result.termination_reason == "complete"


# ──────────────────────────────────────────────────────────
# EVALUATOR_ERROR → evaluation_skipped Agent Tests
# ──────────────────────────────────────────────────────────


def _evaluator_error_evaluator() -> LLMEvaluator:
    """Evaluator whose LLM call always fails → EVALUATOR_ERROR.

    Uses ``_AlwaysErrorClient(LLMProviderError(...))`` rather than
    ``MockLLMClient([])`` because mock exhaustion now raises ``ValueError``
    (not an ``LLMError``) and would propagate out of the evaluator. We need a
    real provider error to exercise the evaluator's ``except LLMError:`` branch.
    """
    from nanitics.infrastructure.errors import LLMProviderError

    return LLMEvaluator(
        llm_client=_AlwaysErrorClient(LLMProviderError("Server error")),
        criteria="Be accurate.",
    )


class TestReActAgentEvaluatorError:
    async def test_evaluator_error_returns_evaluation_skipped(self) -> None:
        """When evaluator LLM fails, agent treats output as accepted with evaluation_skipped."""
        client = MockLLMClient([_make_response(content="good answer")])
        emitter = _make_emitter()
        agent = ReActAgent(
            name="eval-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[_add_tool],
            output_evaluator=_evaluator_error_evaluator(),
        )

        result = await agent.run("Hi")

        assert result.output == "good answer"
        assert result.termination_reason == "evaluation_skipped"


class TestReasoningAgentEvaluatorError:
    async def test_evaluator_error_returns_evaluation_skipped(self) -> None:
        """When evaluator LLM fails, agent treats output as accepted with evaluation_skipped."""
        client = MockLLMClient([_make_response(content="good answer")])
        emitter = _make_emitter()
        agent = ReasoningAgent(
            name="eval-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            output_evaluator=_evaluator_error_evaluator(),
        )

        result = await agent.run("Hi")

        assert result.output == "good answer"
        assert result.termination_reason == "evaluation_skipped"


# ──────────────────────────────────────────────────────────
# EvaluationExhaustedEvent Negative Tests
# ──────────────────────────────────────────────────────────


class TestEvaluationExhaustedEventNotEmitted:
    async def test_no_exhausted_event_on_accept_react(self) -> None:
        client = MockLLMClient([_make_response(content="good answer")])
        emitter = _make_emitter()
        agent = ReActAgent(
            name="eval-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[_add_tool],
            output_evaluator=_always_accept_evaluator(),
        )

        await agent.run("Hi")

        exhausted = [e for e in emitter.events if isinstance(e, EvaluationExhaustedEvent)]
        assert len(exhausted) == 0

    async def test_no_exhausted_event_on_accept_reasoning(self) -> None:
        client = MockLLMClient([_make_response(content="good answer")])
        emitter = _make_emitter()
        agent = ReasoningAgent(
            name="eval-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            output_evaluator=_always_accept_evaluator(),
        )

        await agent.run("Hi")

        exhausted = [e for e in emitter.events if isinstance(e, EvaluationExhaustedEvent)]
        assert len(exhausted) == 0

    async def test_no_exhausted_event_on_evaluator_error_react(self) -> None:
        client = MockLLMClient([_make_response(content="good answer")])
        emitter = _make_emitter()
        agent = ReActAgent(
            name="eval-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[_add_tool],
            output_evaluator=_evaluator_error_evaluator(),
        )

        result = await agent.run("Hi")

        assert result.termination_reason == "evaluation_skipped"
        exhausted = [e for e in emitter.events if isinstance(e, EvaluationExhaustedEvent)]
        assert len(exhausted) == 0

    async def test_no_exhausted_event_on_evaluator_error_reasoning(self) -> None:
        client = MockLLMClient([_make_response(content="good answer")])
        emitter = _make_emitter()
        agent = ReasoningAgent(
            name="eval-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            output_evaluator=_evaluator_error_evaluator(),
        )

        result = await agent.run("Hi")

        assert result.termination_reason == "evaluation_skipped"
        exhausted = [e for e in emitter.events if isinstance(e, EvaluationExhaustedEvent)]
        assert len(exhausted) == 0
