import pytest
from pydantic import ValidationError

from nanitics.composition.multi_agent.supervisor import (
    BudgetTrigger,
    PredicateTrigger,
    QualityTrigger,
    SupervisionAction,
    SupervisionDecision,
    SupervisionResult,
    Supervisor,
)
from nanitics.evaluation import (
    EvaluationCheck,
    ProgrammaticEvaluator,
)
from nanitics.infrastructure import (
    LLMResponse,
    MockLLMClient,
)
from nanitics.infrastructure.llm.protocol import Message
from nanitics.infrastructure.observability.events import SupervisionEvent
from nanitics.strategies import ReActAgent
from nanitics.strategies.agents.base import AgentResult
from nanitics.tracing import (
    InMemoryEmitter,
    Usage,
)


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
            usage=result.usage,
        )
        assert sr.accepted is True
        assert sr.total_attempts == 1
        assert sr.final_agent == "agent-a"
        assert sr.usage == result.usage


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


# ── thread_key propagation ────────────────────────────────


class TestSupervisorThreadKey:
    """Supervisor threading rules:

    * RETRY appends to the supervisee's thread — the agent sees its
      prior attempt and the supervisor's feedback as natural conversation
      turns.
    * REASSIGN switches to the new agent's thread key (or stateless if
      unmapped). The new agent does not inherit the previous one's
      thread.
    """

    def test_thread_keys_default_empty(self) -> None:
        emitter = InMemoryEmitter(trace_id="t")
        sup = Supervisor(triggers=[], emitter=emitter)
        assert sup._thread_keys == {}

    async def test_retry_appends_to_same_thread(self) -> None:
        from nanitics.composition import InMemoryThreadStore

        emitter = InMemoryEmitter(trace_id="t")
        thread_store = InMemoryThreadStore()
        agent = ReActAgent(
            name="worker",
            llm_client=MockLLMClient(
                [
                    LLMResponse(
                        content="first answer",
                        tool_calls=[],
                        usage=Usage(input_tokens=1, output_tokens=1),
                        model="m",
                        stop_reason="end_turn",
                    ),
                    LLMResponse(
                        content="second answer",
                        tool_calls=[],
                        usage=Usage(input_tokens=1, output_tokens=1),
                        model="m",
                        stop_reason="end_turn",
                    ),
                ]
            ),
            emitter=emitter,
            system_prompt="answer",
            tools=[],
            thread_store=thread_store,
        )

        # Trigger that asks for one retry on the first attempt only.
        call_state = {"calls": 0}

        def predicate(_result: AgentResult, _task: str) -> SupervisionDecision | None:
            call_state["calls"] += 1
            if call_state["calls"] == 1:
                return SupervisionDecision(
                    action=SupervisionAction.RETRY,
                    trigger_name="one-retry",
                    feedback="be more thorough",
                )
            return None

        trigger = PredicateTrigger(name="one-retry", predicate=predicate)

        sup = Supervisor(
            triggers=[trigger],
            emitter=emitter,
            thread_keys={"worker": "worker-thread"},
        )
        sr = await sup.supervise(agent, "do the thing")

        assert sr.accepted is True
        assert sr.total_attempts == 2

        # Both attempts wrote to the same thread.
        loaded = await thread_store.load("worker-thread")
        # Two user turns (initial + retry with feedback) and two assistant turns.
        assert sum(1 for m in loaded if m.role == "user") == 2
        assert sum(1 for m in loaded if m.role == "assistant") == 2
        # The retry's user turn carries the feedback string.
        user_contents = [str(m.content) for m in loaded if m.role == "user"]
        assert any("be more thorough" in c for c in user_contents)

    async def test_reassign_switches_thread(self) -> None:
        from nanitics.composition import InMemoryThreadStore

        emitter = InMemoryEmitter(trace_id="t")
        thread_store = InMemoryThreadStore()

        worker = ReActAgent(
            name="worker",
            llm_client=MockLLMClient(
                [
                    LLMResponse(
                        content="worker answer",
                        tool_calls=[],
                        usage=Usage(input_tokens=1, output_tokens=1),
                        model="m",
                        stop_reason="end_turn",
                    ),
                ]
            ),
            emitter=emitter,
            system_prompt="answer",
            tools=[],
            thread_store=thread_store,
        )
        backup = ReActAgent(
            name="backup",
            llm_client=MockLLMClient(
                [
                    LLMResponse(
                        content="backup answer",
                        tool_calls=[],
                        usage=Usage(input_tokens=1, output_tokens=1),
                        model="m",
                        stop_reason="end_turn",
                    ),
                ]
            ),
            emitter=emitter,
            system_prompt="answer",
            tools=[],
            thread_store=thread_store,
        )

        call_state = {"calls": 0}

        def predicate(_result: AgentResult, _task: str) -> SupervisionDecision | None:
            call_state["calls"] += 1
            if call_state["calls"] == 1:
                return SupervisionDecision(
                    action=SupervisionAction.REASSIGN,
                    trigger_name="reassign-once",
                    reassign_to="backup",
                )
            return None

        trigger = PredicateTrigger(name="reassign-once", predicate=predicate)

        sup = Supervisor(
            triggers=[trigger],
            emitter=emitter,
            agents={"backup": backup},
            thread_keys={"worker": "worker-thread", "backup": "backup-thread"},
        )
        sr = await sup.supervise(worker, "do the thing")

        assert sr.accepted is True
        assert sr.final_agent == "backup"

        worker_msgs = await thread_store.load("worker-thread")
        backup_msgs = await thread_store.load("backup-thread")

        # Worker's thread carries only its single attempt.
        assert sum(1 for m in worker_msgs if m.role == "assistant") == 1
        # Backup's thread carries its own attempt — not a continuation
        # of the worker's thread.
        assert sum(1 for m in backup_msgs if m.role == "assistant") == 1


# ── Per-attempt usage aggregation ─────────────────────────


def _agent_with_usage(name: str, emitter: InMemoryEmitter, usages: list[Usage]) -> ReActAgent:
    responses = [
        LLMResponse(
            content=f"out-{i}",
            tool_calls=[],
            usage=u,
            model="m",
            stop_reason="end_turn",
        )
        for i, u in enumerate(usages)
    ]
    return ReActAgent(
        name=name,
        llm_client=MockLLMClient(responses),
        emitter=emitter,
        system_prompt="answer",
        tools=[],
    )


class TestSupervisorUsageAggregation:
    async def test_single_accept_equals_attempt_usage(self) -> None:
        emitter = InMemoryEmitter(trace_id="t")
        agent = _agent_with_usage("a", emitter, [Usage(input_tokens=3, output_tokens=4)])
        sup = Supervisor(triggers=[], emitter=emitter)
        sr = await sup.supervise(agent, "task")
        assert sr.usage == Usage(input_tokens=3, output_tokens=4)
        assert sr.result.usage == sr.usage

    async def test_retry_then_accept_sums_two_attempts(self) -> None:
        emitter = InMemoryEmitter(trace_id="t")
        agent = _agent_with_usage(
            "a",
            emitter,
            [
                Usage(input_tokens=1, output_tokens=2),
                Usage(input_tokens=5, output_tokens=7),
            ],
        )
        calls = {"n": 0}

        def predicate(_r, _t):
            calls["n"] += 1
            if calls["n"] == 1:
                return SupervisionDecision(
                    action=SupervisionAction.RETRY,
                    trigger_name="once",
                    feedback="redo",
                )
            return None

        sup = Supervisor(
            triggers=[PredicateTrigger(name="once", predicate=predicate)],
            emitter=emitter,
        )
        sr = await sup.supervise(agent, "task")
        assert sr.usage == Usage(input_tokens=6, output_tokens=9)
        # final attempt's usage only:
        assert sr.result.usage == Usage(input_tokens=5, output_tokens=7)

    async def test_retry_give_up_sums_all_attempts(self) -> None:
        emitter = InMemoryEmitter(trace_id="t")
        agent = _agent_with_usage(
            "a",
            emitter,
            [
                Usage(input_tokens=1, output_tokens=1),
                Usage(input_tokens=1, output_tokens=1),
            ],
        )

        def always_retry(_r, _t):
            return SupervisionDecision(action=SupervisionAction.RETRY, trigger_name="always", feedback="x")

        sup = Supervisor(
            triggers=[PredicateTrigger(name="always", predicate=always_retry)],
            emitter=emitter,
            max_retries=1,
        )
        sr = await sup.supervise(agent, "task")
        assert sr.accepted is False
        assert sr.usage == Usage(input_tokens=2, output_tokens=2)

    async def test_reassign_sums_across_agents(self) -> None:
        emitter = InMemoryEmitter(trace_id="t")
        worker = _agent_with_usage("worker", emitter, [Usage(input_tokens=2, output_tokens=3)])
        backup = _agent_with_usage("backup", emitter, [Usage(input_tokens=4, output_tokens=5)])
        calls = {"n": 0}

        def predicate(_r, _t):
            calls["n"] += 1
            if calls["n"] == 1:
                return SupervisionDecision(
                    action=SupervisionAction.REASSIGN,
                    trigger_name="reassign-once",
                    reassign_to="backup",
                )
            return None

        sup = Supervisor(
            triggers=[PredicateTrigger(name="reassign-once", predicate=predicate)],
            emitter=emitter,
            agents={"backup": backup},
        )
        sr = await sup.supervise(worker, "task")
        assert sr.accepted is True
        assert sr.usage == Usage(input_tokens=6, output_tokens=8)

    async def test_reassign_to_unknown_returns_partial_sum(self) -> None:
        emitter = InMemoryEmitter(trace_id="t")
        agent = _agent_with_usage("a", emitter, [Usage(input_tokens=2, output_tokens=2)])

        def predicate(_r, _t):
            return SupervisionDecision(
                action=SupervisionAction.REASSIGN,
                trigger_name="bad-reassign",
                reassign_to="nope",
            )

        sup = Supervisor(
            triggers=[PredicateTrigger(name="bad-reassign", predicate=predicate)],
            emitter=emitter,
        )
        sr = await sup.supervise(agent, "task")
        assert sr.accepted is False
        assert sr.usage == Usage(input_tokens=2, output_tokens=2)

    async def test_escalate_returns_single_attempt_usage(self) -> None:
        emitter = InMemoryEmitter(trace_id="t")
        agent = _agent_with_usage("a", emitter, [Usage(input_tokens=9, output_tokens=11)])

        def predicate(_r, _t):
            return SupervisionDecision(
                action=SupervisionAction.ESCALATE,
                trigger_name="escalate-now",
                feedback="bad",
            )

        sup = Supervisor(
            triggers=[PredicateTrigger(name="escalate-now", predicate=predicate)],
            emitter=emitter,
        )
        sr = await sup.supervise(agent, "task")
        assert sr.accepted is False
        assert sr.usage == Usage(input_tokens=9, output_tokens=11)
