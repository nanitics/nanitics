from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import SupervisionEvent
from nanitics.strategies.agents.base import Agent, AgentResult
from nanitics.strategies.agents.evaluation import (
    EvaluationContext,
    EvaluationVerdict,
    OutputEvaluator,
)

# ── Data Models ────────────────────────────────────────────


class SupervisionAction(StrEnum):
    """Actions a supervisor can take after evaluating an agent's result."""

    ACCEPT = "accept"
    RETRY = "retry"
    REASSIGN = "reassign"
    ESCALATE = "escalate"


class SupervisionDecision(BaseModel):
    """A trigger's decision about an agent result.

    Attributes:
        action: What the supervisor should do next.
        feedback: Optional feedback message, appended to the task on retry.
        reassign_to: Target agent name when action is REASSIGN.
        trigger_name: Name of the trigger that produced this decision.
    """

    model_config = ConfigDict(frozen=True)

    action: SupervisionAction
    feedback: str | None = None
    reassign_to: str | None = None
    trigger_name: str


class SupervisionResult(BaseModel):
    """Outcome of a supervised agent run.

    Attributes:
        result: The final ``AgentResult`` from the last agent execution.
        accepted: Whether the result passed all triggers.
        total_attempts: Total number of agent runs performed.
        interventions: All intervention decisions made during supervision.
        final_agent: Name of the agent that produced the final result.
    """

    model_config = ConfigDict(frozen=True)

    result: AgentResult
    accepted: bool
    total_attempts: int
    interventions: list[SupervisionDecision]
    final_agent: str


# ── Trigger Protocol ───────────────────────────────────────


@runtime_checkable
class SupervisionTrigger(Protocol):
    """Protocol for supervision triggers that evaluate agent results.

    Triggers inspect an ``AgentResult`` and return ``None`` to pass (no
    intervention) or a ``SupervisionDecision`` to intervene.
    """

    @property
    def name(self) -> str: ...

    async def check(self, result: AgentResult, task: str) -> SupervisionDecision | None: ...


# ── Built-in Triggers ──────────────────────────────────────


class QualityTrigger:
    """Supervision trigger that evaluates output quality via an ``OutputEvaluator``.

    Maps evaluator verdicts to supervision actions:
    - ACCEPT → pass (no intervention)
    - REVISE → RETRY with feedback
    - REJECT → ESCALATE with feedback
    - EVALUATOR_ERROR → configurable via ``on_evaluator_error``:
      - ``"skip"``: ACCEPT with diagnostic feedback (default)
      - ``"escalate"``: ESCALATE with diagnostic feedback
    - No output → ESCALATE

    Args:
        evaluator: The output evaluator to assess result quality.
        on_evaluator_error: Policy for handling evaluator errors.
    """

    def __init__(
        self,
        evaluator: OutputEvaluator,
        *,
        on_evaluator_error: Literal["skip", "escalate"] = "skip",
    ) -> None:
        self._evaluator = evaluator
        self._on_evaluator_error = on_evaluator_error

    @property
    def name(self) -> str:
        return "quality"

    async def check(self, result: AgentResult, task: str) -> SupervisionDecision | None:
        if result.output is None:
            return SupervisionDecision(
                action=SupervisionAction.ESCALATE,
                trigger_name=self.name,
                feedback="Agent produced no output",
            )

        eval_result = await self._evaluator.evaluate(
            result.output,
            EvaluationContext(messages=result.messages, task_input=task),
        )

        if eval_result.verdict == EvaluationVerdict.ACCEPT:
            return None
        if eval_result.verdict == EvaluationVerdict.REVISE:
            return SupervisionDecision(
                action=SupervisionAction.RETRY,
                feedback=eval_result.feedback,
                trigger_name=self.name,
            )
        if eval_result.verdict == EvaluationVerdict.EVALUATOR_ERROR:
            detail = eval_result.error_detail or eval_result.feedback or "Unknown evaluator error"
            if self._on_evaluator_error == "skip":
                return SupervisionDecision(
                    action=SupervisionAction.ACCEPT,
                    trigger_name=self.name,
                    feedback=f"Evaluator error (skipped): {detail}",
                )
            return SupervisionDecision(
                action=SupervisionAction.ESCALATE,
                trigger_name=self.name,
                feedback=f"Evaluator error: {detail}",
            )
        # REJECT
        return SupervisionDecision(
            action=SupervisionAction.ESCALATE,
            trigger_name=self.name,
            feedback=eval_result.feedback,
        )


class BudgetTrigger:
    """Supervision trigger that escalates when token usage exceeds a budget.

    Args:
        max_tokens: Maximum total tokens allowed. If the agent's result
            exceeds this, the trigger returns ESCALATE.
    """

    def __init__(self, max_tokens: int) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        return "budget"

    async def check(self, result: AgentResult, task: str) -> SupervisionDecision | None:
        used = result.usage.total_tokens
        if used > self._max_tokens:
            return SupervisionDecision(
                action=SupervisionAction.ESCALATE,
                feedback=f"Token budget exceeded: {used}/{self._max_tokens}",
                trigger_name=self.name,
            )
        return None


class PredicateTrigger:
    """Supervision trigger using a custom predicate function.

    The predicate receives the ``AgentResult`` and original task, and
    returns ``None`` to pass or a ``SupervisionDecision`` to intervene.

    Args:
        name: Trigger name for identification in events.
        predicate: Callable returning ``None`` or a ``SupervisionDecision``.
    """

    def __init__(
        self,
        *,
        name: str,
        predicate: Callable[[AgentResult, str], SupervisionDecision | None],
    ) -> None:
        self._name = name
        self._predicate = predicate

    @property
    def name(self) -> str:
        return self._name

    async def check(self, result: AgentResult, task: str) -> SupervisionDecision | None:
        return self._predicate(result, task)


# ── Supervisor ─────────────────────────────────────────────


class Supervisor:
    """Post-execution monitor that evaluates agent results via triggers.

    After each agent run, triggers are checked in order. The first trigger
    that fires determines the next action: retry with feedback, reassign
    to a different agent, or escalate. If no trigger fires, the result is
    accepted.

    Args:
        triggers: Ordered list of triggers to evaluate after each run.
        emitter: Event emitter for supervision events.
        max_retries: Maximum retry attempts before giving up.
        agents: Registry of named agents for reassignment.
    """

    def __init__(
        self,
        *,
        triggers: list[SupervisionTrigger],
        emitter: EventEmitter,
        max_retries: int = 2,
        agents: dict[str, Agent] | None = None,
    ) -> None:
        self._triggers = triggers
        self._emitter = emitter
        self._max_retries = max_retries
        self._agents = agents or {}

    async def supervise(self, agent: Agent, task: str) -> SupervisionResult:
        """Run an agent under supervision.

        Executes the agent, evaluates the result against all triggers,
        and handles retry, reassignment, or escalation as needed.

        Args:
            agent: The agent to supervise.
            task: The task to execute.

        Returns:
            A ``SupervisionResult`` with the final outcome and intervention history.
        """
        current_agent = agent
        current_task = task
        attempts = 0
        retries_remaining = self._max_retries
        interventions: list[SupervisionDecision] = []

        while True:
            result = await current_agent.bind(self._emitter).run(current_task)
            attempts += 1

            decision = await self._check_triggers(result, task)

            if decision is None or decision.action == SupervisionAction.ACCEPT:
                accept_decision = decision or SupervisionDecision(
                    action=SupervisionAction.ACCEPT,
                    trigger_name="all_passed",
                )
                self._emit_event(
                    current_agent.name,
                    accept_decision,
                    attempts,
                )
                return SupervisionResult(
                    result=result,
                    accepted=True,
                    total_attempts=attempts,
                    interventions=interventions,
                    final_agent=current_agent.name,
                )

            interventions.append(decision)
            self._emit_event(current_agent.name, decision, attempts)

            if decision.action == SupervisionAction.RETRY:
                if retries_remaining <= 0:
                    return SupervisionResult(
                        result=result,
                        accepted=False,
                        total_attempts=attempts,
                        interventions=interventions,
                        final_agent=current_agent.name,
                    )
                retries_remaining -= 1
                current_task = f"{task}\n\n## Feedback from review\n{decision.feedback}"

            elif decision.action == SupervisionAction.REASSIGN:
                target = self._agents.get(decision.reassign_to or "")
                if target is None:
                    return SupervisionResult(
                        result=result,
                        accepted=False,
                        total_attempts=attempts,
                        interventions=interventions,
                        final_agent=current_agent.name,
                    )
                current_agent = target
                current_task = task
                if decision.feedback:
                    current_task = f"{task}\n\n## Feedback from review\n{decision.feedback}"

            elif decision.action == SupervisionAction.ESCALATE:
                return SupervisionResult(
                    result=result,
                    accepted=False,
                    total_attempts=attempts,
                    interventions=interventions,
                    final_agent=current_agent.name,
                )

    async def _check_triggers(self, result: AgentResult, task: str) -> SupervisionDecision | None:
        for trigger in self._triggers:
            decision = await trigger.check(result, task)
            if decision is not None:
                return decision
        return None

    def _emit_event(self, agent_name: str, decision: SupervisionDecision, attempt: int) -> None:
        self._emitter.emit(
            SupervisionEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                supervised_agent=agent_name,
                action=decision.action.value,
                trigger_name=decision.trigger_name,
                feedback=decision.feedback,
                reassigned_to=decision.reassign_to,
                attempt=attempt,
            )
        )
