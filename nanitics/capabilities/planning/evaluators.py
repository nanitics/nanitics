from __future__ import annotations

from nanitics.capabilities.evaluation.protocol import (
    EvaluationContext,
    EvaluationResult,
    EvaluationVerdict,
)
from nanitics.capabilities.planning.models import (
    Goal,
    GoalStatus,
    PlanStatus,
    StepStatus,
)
from nanitics.capabilities.planning.store import PlanStore


class PlanAdherenceEvaluator:
    """Evaluates whether an agent has completed all plan steps before finishing.

    Returns REVISE with a list of incomplete steps if any remain not-started
    or in-progress. Accepts if the plan is not found, abandoned, or fully
    completed.
    """

    def __init__(
        self,
        store: PlanStore,
        plan_id: str,
        max_revisions: int = 1,
    ) -> None:
        """Initialize the plan adherence evaluator.

        Args:
            store: Plan store to load the plan from.
            plan_id: ID of the plan to evaluate adherence against.
            max_revisions: Maximum revision attempts before giving up.
        """
        self._store = store
        self._plan_id = plan_id
        self._max_revisions = max_revisions

    @property
    def max_revisions(self) -> int:
        return self._max_revisions

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        plan = await self._store.load(self._plan_id)
        if plan is None:
            return EvaluationResult(
                verdict=EvaluationVerdict.ACCEPT,
                feedback="Plan not found — nothing to enforce.",
                evaluator_name="plan_adherence",
            )

        if plan.status == PlanStatus.abandoned:
            return EvaluationResult(
                verdict=EvaluationVerdict.ACCEPT,
                feedback="Plan was intentionally abandoned.",
                evaluator_name="plan_adherence",
            )

        incomplete = [step for step in plan.steps if step.status in (StepStatus.not_started, StepStatus.in_progress)]

        if not incomplete:
            return EvaluationResult(
                verdict=EvaluationVerdict.ACCEPT,
                score=1.0,
                evaluator_name="plan_adherence",
            )

        descriptions = "\n".join(f"- [{step.status.value}] {step.description}" for step in incomplete)
        return EvaluationResult(
            verdict=EvaluationVerdict.REVISE,
            score=0.0,
            feedback=f"Plan has incomplete steps:\n{descriptions}",
            evaluator_name="plan_adherence",
        )


def _collect_active_goals(goals: list[Goal]) -> list[Goal]:
    """Recursively collect all goals with active status."""
    active: list[Goal] = []
    for goal in goals:
        if goal.status == GoalStatus.active:
            active.append(goal)
        active.extend(_collect_active_goals(goal.subgoals))
    return active


class GoalSatisfactionEvaluator:
    """Evaluates whether all goals in a plan have been resolved.

    Returns REVISE with a list of still-active goals if any remain
    unresolved. Accepts if the plan is not found or all goals have
    a terminal status (achieved, blocked, or abandoned).
    """

    def __init__(
        self,
        store: PlanStore,
        plan_id: str,
        max_revisions: int = 1,
    ) -> None:
        """Initialize the goal satisfaction evaluator.

        Args:
            store: Plan store to load the plan from.
            plan_id: ID of the plan whose goals to evaluate.
            max_revisions: Maximum revision attempts before giving up.
        """
        self._store = store
        self._plan_id = plan_id
        self._max_revisions = max_revisions

    @property
    def max_revisions(self) -> int:
        return self._max_revisions

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        plan = await self._store.load(self._plan_id)
        if plan is None:
            return EvaluationResult(
                verdict=EvaluationVerdict.ACCEPT,
                feedback="Plan not found — nothing to enforce.",
                evaluator_name="goal_satisfaction",
            )

        if plan.status == PlanStatus.abandoned:
            return EvaluationResult(
                verdict=EvaluationVerdict.ACCEPT,
                feedback="Plan was intentionally abandoned.",
                evaluator_name="goal_satisfaction",
            )

        active_goals = _collect_active_goals(plan.goals)

        if not active_goals:
            return EvaluationResult(
                verdict=EvaluationVerdict.ACCEPT,
                score=1.0,
                evaluator_name="goal_satisfaction",
            )

        descriptions = "\n".join(f"- {goal.description} (priority={goal.priority})" for goal in active_goals)
        return EvaluationResult(
            verdict=EvaluationVerdict.REVISE,
            score=0.0,
            feedback=f"Active goals remain unresolved:\n{descriptions}",
            evaluator_name="goal_satisfaction",
        )
