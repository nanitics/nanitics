import pytest
from pydantic import ValidationError

from nanitics import (
    EvaluationCheck,
    InMemoryEmitter,
    LLMResponse,
    MockLLMClient,
    ProgrammaticEvaluator,
    ReActAgent,
    Usage,
)
from nanitics.composition.multi_agent.supervisor import (
    BudgetTrigger,
    PredicateTrigger,
    QualityTrigger,
    SupervisionAction,
    SupervisionDecision,
    SupervisionResult,
    Supervisor,
)
from nanitics.infrastructure.llm.protocol import Message
from nanitics.infrastructure.observability.events import SupervisionEvent
from nanitics.strategies.agents.base import AgentResult


def make_agent(name: str, responses: list[LLMResponse], emitter: InMemoryEmitter) -> ReActAgent:
    return ReActAgent(
        name=name,
        llm_client=MockLLMClient(responses),
        emitter=emitter,
        system_prompt=f"You are {name}.",
        tools=[],
    )


def make_agent_result(
    output: str | None = "result",
    total_tokens: int = 15,
) -> AgentResult:
    return AgentResult(
        output=output,
        total_steps=1,
        termination_reason="complete",
        messages=[Message(role="user", content="task")],
        usage=Usage(
            input_tokens=10,
            output_tokens=total_tokens - 10,
        ),
    )


# ── Data Model Tests ───────────────────────────────────────


class TestSupervisionAction:
    def test_values(self) -> None:
        assert SupervisionAction.ACCEPT.value == "accept"
        assert SupervisionAction.RETRY.value == "retry"
        assert SupervisionAction.REASSIGN.value == "reassign"
        assert SupervisionAction.ESCALATE.value == "escalate"


class TestSupervisionDecision:
    def test_construction(self) -> None:
        d = SupervisionDecision(
            action=SupervisionAction.RETRY,
            feedback="Improve output",
            trigger_name="quality",
        )
        assert d.action == SupervisionAction.RETRY
        assert d.feedback == "Improve output"
        assert d.reassign_to is None
        assert d.trigger_name == "quality"

    def test_frozen(self) -> None:
        d = SupervisionDecision(
            action=SupervisionAction.ACCEPT,
            trigger_name="test",
        )
        with pytest.raises(ValidationError):
            d.action = SupervisionAction.RETRY


class TestSupervisionResult:
    def test_construction(self) -> None:
        result = make_agent_result()
        sr = SupervisionResult(
            result=result,
            accepted=True,
            total_attempts=1,
            interventions=[],
            final_agent="agent-a",
        )
        assert sr.accepted is True
        assert sr.total_attempts == 1
        assert sr.final_agent == "agent-a"


# ── QualityTrigger Tests ──────────────────────────────────


class TestQualityTrigger:
    def test_name(self) -> None:
        evaluator = ProgrammaticEvaluator(checks=[])
        trigger = QualityTrigger(evaluator)
        assert trigger.name == "quality"

    async def test_accept_returns_none(self) -> None:
        evaluator = ProgrammaticEvaluator(checks=[])  # No checks = always accept
        trigger = QualityTrigger(evaluator)
        result = make_agent_result(output="good output")
        decision = await trigger.check(result, "task")
        assert decision is None

    async def test_revise_returns_retry(self) -> None:
        check = EvaluationCheck(
            name="length",
            check=lambda x: len(x) > 100,
            feedback="Output too short",
        )
        evaluator = ProgrammaticEvaluator(checks=[check])
        trigger = QualityTrigger(evaluator)
        result = make_agent_result(output="short")
        decision = await trigger.check(result, "task")
        assert decision is not None
        assert decision.action == SupervisionAction.RETRY
        assert "Output too short" in (decision.feedback or "")

    async def test_none_output_escalates(self) -> None:
        evaluator = ProgrammaticEvaluator(checks=[])
        trigger = QualityTrigger(evaluator)
        result = make_agent_result(output=None)
        decision = await trigger.check(result, "task")
        assert decision is not None
        assert decision.action == SupervisionAction.ESCALATE

    async def test_reject_verdict_escalates(self) -> None:
        """REJECT verdict (not REVISE or EVALUATOR_ERROR) falls through to ESCALATE."""
        from nanitics.strategies.agents.evaluation import (
            EvaluationContext,
            EvaluationResult,
            EvaluationVerdict,
        )

        class _RejectEvaluator:
            max_revisions = 1

            async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
                return EvaluationResult(
                    verdict=EvaluationVerdict.REJECT,
                    score=None,
                    feedback="Output fundamentally unacceptable",
                    evaluator_name="test",
                )

        trigger = QualityTrigger(_RejectEvaluator())
        result = make_agent_result(output="bad output")
        decision = await trigger.check(result, "task")
        assert decision is not None
        assert decision.action == SupervisionAction.ESCALATE
        assert decision.feedback == "Output fundamentally unacceptable"


# ── BudgetTrigger Tests ───────────────────────────────────


class TestBudgetTrigger:
    def test_name(self) -> None:
        trigger = BudgetTrigger(max_tokens=100)
        assert trigger.name == "budget"

    async def test_under_budget_returns_none(self) -> None:
        trigger = BudgetTrigger(max_tokens=100)
        result = make_agent_result(total_tokens=50)
        decision = await trigger.check(result, "task")
        assert decision is None

    async def test_over_budget_returns_escalate(self) -> None:
        trigger = BudgetTrigger(max_tokens=100)
        result = make_agent_result(total_tokens=150)
        decision = await trigger.check(result, "task")
        assert decision is not None
        assert decision.action == SupervisionAction.ESCALATE
        assert "150/100" in (decision.feedback or "")


# ── PredicateTrigger Tests ────────────────────────────────


class TestPredicateTrigger:
    def test_name(self) -> None:
        trigger = PredicateTrigger(name="custom", predicate=lambda r, t: None)
        assert trigger.name == "custom"

    async def test_delegates_to_callable(self) -> None:
        decision = SupervisionDecision(
            action=SupervisionAction.ESCALATE,
            trigger_name="custom",
        )
        trigger = PredicateTrigger(name="custom", predicate=lambda r, t: decision)
        result = make_agent_result()
        check_result = await trigger.check(result, "task")
        assert check_result is decision

    async def test_returns_none_when_no_issue(self) -> None:
        trigger = PredicateTrigger(name="custom", predicate=lambda r, t: None)
        result = make_agent_result()
        check_result = await trigger.check(result, "task")
        assert check_result is None


# ── Supervisor Tests ──────────────────────────────────────


class TestSupervisor:
    async def test_all_triggers_pass_accepted(self) -> None:
        emitter = make_emitter()
        agent = make_agent("agent-a", [make_response("good")], emitter)
        trigger = PredicateTrigger(name="pass", predicate=lambda r, t: None)
        supervisor = Supervisor(triggers=[trigger], emitter=emitter)

        sr = await supervisor.supervise(agent, "do something")

        assert sr.accepted is True
        assert sr.total_attempts == 1
        assert sr.interventions == []
        assert sr.final_agent == "agent-a"

        # Accept event emitted so frontend can detect the supervisor pattern
        supervision_events = [e for e in emitter.events if isinstance(e, SupervisionEvent)]
        assert len(supervision_events) == 1
        evt = supervision_events[0]
        assert evt.supervised_agent == "agent-a"
        assert evt.action == "accept"
        assert evt.trigger_name == "all_passed"
        assert evt.attempt == 1

    async def test_retry_with_feedback(self) -> None:
        emitter = make_emitter()
        # Agent will be called twice: first attempt + retry
        agent = make_agent(
            "agent-a",
            [make_response("bad"), make_response("good")],
            emitter,
        )

        call_count = 0

        def check_predicate(result, task):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return SupervisionDecision(
                    action=SupervisionAction.RETRY,
                    feedback="Try harder",
                    trigger_name="test",
                )
            return None

        trigger = PredicateTrigger(name="test", predicate=check_predicate)
        supervisor = Supervisor(triggers=[trigger], emitter=emitter)

        sr = await supervisor.supervise(agent, "do something")

        assert sr.accepted is True
        assert sr.total_attempts == 2
        assert len(sr.interventions) == 1
        assert sr.interventions[0].action == SupervisionAction.RETRY

    async def test_retry_feedback_appended_to_task(self) -> None:
        emitter = make_emitter()
        agent = make_agent(
            "agent-a",
            [make_response("bad"), make_response("good")],
            emitter,
        )

        call_count = 0

        def check_predicate(result, task):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return SupervisionDecision(
                    action=SupervisionAction.RETRY,
                    feedback="Be more specific",
                    trigger_name="test",
                )
            return None

        trigger = PredicateTrigger(name="test", predicate=check_predicate)
        supervisor = Supervisor(triggers=[trigger], emitter=emitter)

        await supervisor.supervise(agent, "write a poem")

        # Check the second call to the agent included feedback
        client: MockLLMClient = agent._llm_client  # type: ignore[assignment]
        second_call_messages = client.calls[1]["messages"]
        task_text = str(second_call_messages[0].content)
        assert "## Feedback from review" in task_text
        assert "Be more specific" in task_text

    async def test_retry_exhausts_max_retries(self) -> None:
        emitter = make_emitter()
        agent = make_agent(
            "agent-a",
            [make_response("bad")] * 5,
            emitter,
        )
        trigger = PredicateTrigger(
            name="always-retry",
            predicate=lambda r, t: SupervisionDecision(
                action=SupervisionAction.RETRY,
                feedback="Still bad",
                trigger_name="always-retry",
            ),
        )
        supervisor = Supervisor(triggers=[trigger], emitter=emitter, max_retries=2)

        sr = await supervisor.supervise(agent, "task")

        assert sr.accepted is False
        assert sr.total_attempts == 3  # 1 original + 2 retries
        assert len(sr.interventions) == 3

    async def test_reassign_runs_different_agent(self) -> None:
        emitter = make_emitter()
        agent_a = make_agent("agent-a", [make_response("bad")], emitter)
        agent_b = make_agent("agent-b", [make_response("good")], emitter)

        call_count = 0

        def check_predicate(result, task):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return SupervisionDecision(
                    action=SupervisionAction.REASSIGN,
                    reassign_to="agent-b",
                    trigger_name="test",
                )
            return None

        trigger = PredicateTrigger(name="test", predicate=check_predicate)
        supervisor = Supervisor(
            triggers=[trigger],
            emitter=emitter,
            agents={"agent-b": agent_b},
        )

        sr = await supervisor.supervise(agent_a, "task")

        assert sr.accepted is True
        assert sr.total_attempts == 2
        assert sr.final_agent == "agent-b"

    async def test_reassign_with_feedback_updates_task(self) -> None:
        """REASSIGN decision with feedback prepends feedback to the task for the new agent."""
        emitter = make_emitter()
        agent_a = make_agent("agent-a", [make_response("bad")], emitter)
        agent_b = make_agent("agent-b", [make_response("good")], emitter)

        call_count = 0

        def check_predicate(result, task):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return SupervisionDecision(
                    action=SupervisionAction.REASSIGN,
                    reassign_to="agent-b",
                    feedback="Please be more thorough",
                    trigger_name="test",
                )
            return None

        trigger = PredicateTrigger(name="test", predicate=check_predicate)
        supervisor = Supervisor(
            triggers=[trigger],
            emitter=emitter,
            agents={"agent-b": agent_b},
        )

        sr = await supervisor.supervise(agent_a, "original task")

        assert sr.accepted is True
        assert sr.final_agent == "agent-b"
        # Verify feedback was included in the task sent to agent-b
        client: MockLLMClient = agent_b._llm_client  # type: ignore[assignment]
        task_text = str(client.calls[0]["messages"][0].content)
        assert "## Feedback from review" in task_text
        assert "Please be more thorough" in task_text

    async def test_reassign_not_found_escalates(self) -> None:
        emitter = make_emitter()
        agent = make_agent("agent-a", [make_response("bad")], emitter)
        trigger = PredicateTrigger(
            name="test",
            predicate=lambda r, t: SupervisionDecision(
                action=SupervisionAction.REASSIGN,
                reassign_to="nonexistent",
                trigger_name="test",
            ),
        )
        supervisor = Supervisor(triggers=[trigger], emitter=emitter)

        sr = await supervisor.supervise(agent, "task")

        assert sr.accepted is False
        assert sr.total_attempts == 1

    async def test_escalate_returns_immediately(self) -> None:
        emitter = make_emitter()
        agent = make_agent("agent-a", [make_response("output")], emitter)
        trigger = PredicateTrigger(
            name="test",
            predicate=lambda r, t: SupervisionDecision(
                action=SupervisionAction.ESCALATE,
                trigger_name="test",
            ),
        )
        supervisor = Supervisor(triggers=[trigger], emitter=emitter)

        sr = await supervisor.supervise(agent, "task")

        assert sr.accepted is False
        assert sr.total_attempts == 1
        assert sr.result.output == "output"

    async def test_first_non_none_trigger_wins(self) -> None:
        emitter = make_emitter()
        agent = make_agent("agent-a", [make_response("output")], emitter)

        trigger_pass = PredicateTrigger(name="pass", predicate=lambda r, t: None)
        trigger_escalate = PredicateTrigger(
            name="escalate",
            predicate=lambda r, t: SupervisionDecision(
                action=SupervisionAction.ESCALATE,
                trigger_name="escalate",
            ),
        )
        trigger_retry = PredicateTrigger(
            name="retry",
            predicate=lambda r, t: SupervisionDecision(
                action=SupervisionAction.RETRY,
                feedback="x",
                trigger_name="retry",
            ),
        )
        supervisor = Supervisor(
            triggers=[trigger_pass, trigger_escalate, trigger_retry],
            emitter=emitter,
        )

        sr = await supervisor.supervise(agent, "task")

        assert sr.accepted is False
        assert sr.interventions[0].trigger_name == "escalate"

    async def test_supervision_event_emitted(self) -> None:
        emitter = make_emitter()
        agent = make_agent("agent-a", [make_response("output")], emitter)
        trigger = PredicateTrigger(
            name="test",
            predicate=lambda r, t: SupervisionDecision(
                action=SupervisionAction.ESCALATE,
                trigger_name="test",
            ),
        )
        supervisor = Supervisor(triggers=[trigger], emitter=emitter)

        await supervisor.supervise(agent, "task")

        supervision_events = [e for e in emitter.events if isinstance(e, SupervisionEvent)]
        assert len(supervision_events) == 1
        evt = supervision_events[0]
        assert evt.supervised_agent == "agent-a"
        assert evt.action == "escalate"
        assert evt.trigger_name == "test"
        assert evt.attempt == 1


# ── Integration Test ───────────────────────────────────────


from nanitics.strategies.agents.evaluation import (
    EvaluationContext,
    EvaluationResult,
    EvaluationVerdict,
    OutputEvaluator,
)
from tests.testing_helpers import make_emitter, make_response


def _evaluator_error_evaluator(error_detail: str | None = "LLMProviderError: Server error") -> OutputEvaluator:
    """Create an evaluator that always returns EVALUATOR_ERROR."""

    class _ErrorEvaluator:
        max_revisions = 1

        async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
            return EvaluationResult(
                verdict=EvaluationVerdict.EVALUATOR_ERROR,
                score=None,
                feedback="Evaluation failed: evaluator LLM call failed.",
                evaluator_name="llm",
                error_detail=error_detail,
            )

    return _ErrorEvaluator()


class TestQualityTriggerEvaluatorError:
    async def test_skip_policy_returns_accept(self) -> None:
        """EVALUATOR_ERROR with skip policy returns ACCEPT decision."""
        trigger = QualityTrigger(_evaluator_error_evaluator(), on_evaluator_error="skip")
        result = make_agent_result(output="some output")

        decision = await trigger.check(result, "task")

        assert decision is not None
        assert decision.action == SupervisionAction.ACCEPT
        assert decision.feedback is not None
        assert "skipped" in decision.feedback.lower()

    async def test_escalate_policy_returns_escalate(self) -> None:
        """EVALUATOR_ERROR with escalate policy returns ESCALATE decision."""
        trigger = QualityTrigger(_evaluator_error_evaluator(), on_evaluator_error="escalate")
        result = make_agent_result(output="some output")

        decision = await trigger.check(result, "task")

        assert decision is not None
        assert decision.action == SupervisionAction.ESCALATE
        assert decision.feedback is not None
        assert "Evaluator error" in decision.feedback

    async def test_error_detail_in_feedback(self) -> None:
        """error_detail from EvaluationResult appears in decision feedback."""
        trigger = QualityTrigger(
            _evaluator_error_evaluator(error_detail="LLMRateLimitError: Rate limited"),
            on_evaluator_error="skip",
        )
        result = make_agent_result(output="some output")

        decision = await trigger.check(result, "task")

        assert decision is not None
        assert decision.feedback is not None
        assert "LLMRateLimitError" in decision.feedback

    async def test_default_policy_is_skip(self) -> None:
        """Default on_evaluator_error policy is skip."""
        trigger = QualityTrigger(_evaluator_error_evaluator())
        result = make_agent_result(output="some output")

        decision = await trigger.check(result, "task")

        assert decision is not None
        assert decision.action == SupervisionAction.ACCEPT


class TestSupervisorAcceptFromTrigger:
    async def test_accept_decision_from_trigger_accepted(self) -> None:
        """Supervisor accepts result when trigger returns ACCEPT decision."""
        emitter = make_emitter()
        agent = make_agent("agent-a", [make_response("output")], emitter)
        trigger = QualityTrigger(_evaluator_error_evaluator(), on_evaluator_error="skip")
        supervisor = Supervisor(triggers=[trigger], emitter=emitter)

        sr = await supervisor.supervise(agent, "task")

        assert sr.accepted is True
        assert sr.total_attempts == 1

    async def test_accept_decision_emits_event_with_feedback(self) -> None:
        """Supervisor emits event with feedback when trigger returns ACCEPT."""
        emitter = make_emitter()
        agent = make_agent("agent-a", [make_response("output")], emitter)
        trigger = QualityTrigger(_evaluator_error_evaluator(), on_evaluator_error="skip")
        supervisor = Supervisor(triggers=[trigger], emitter=emitter)

        await supervisor.supervise(agent, "task")

        supervision_events = [e for e in emitter.events if isinstance(e, SupervisionEvent)]
        assert len(supervision_events) == 1
        evt = supervision_events[0]
        assert evt.action == "accept"
        assert evt.trigger_name == "quality"
        assert evt.feedback is not None
        assert "skipped" in evt.feedback.lower()


class TestSupervisorIntegration:
    async def test_quality_trigger_retry_then_accept(self) -> None:
        """Agent output fails quality check, retry succeeds."""
        emitter = make_emitter()

        # First response is too short, second is long enough
        agent = make_agent(
            "writer",
            [
                make_response("short"),
                make_response("This is a sufficiently long and detailed response that passes the quality check"),
            ],
            emitter,
        )

        check = EvaluationCheck(
            name="length",
            check=lambda x: len(x) > 20,
            feedback="Output too short, please elaborate",
        )
        evaluator = ProgrammaticEvaluator(checks=[check])
        trigger = QualityTrigger(evaluator)
        supervisor = Supervisor(triggers=[trigger], emitter=emitter)

        sr = await supervisor.supervise(agent, "write something detailed")

        assert sr.accepted is True
        assert sr.total_attempts == 2
        assert len(sr.interventions) == 1
        assert sr.interventions[0].action == SupervisionAction.RETRY


@pytest.mark.parametrize("value", [0, -1])
def test_budget_trigger_rejects_non_positive(value: int) -> None:
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        BudgetTrigger(max_tokens=value)
