from __future__ import annotations

from typing import Literal

from nanitics.capabilities.evaluation.protocol import (
    EvaluationContext,
    EvaluationResult,
    EvaluationVerdict,
    OutputEvaluator,
)
from nanitics.capabilities.planning.context_provider import PlanningContextProvider
from nanitics.capabilities.planning.evaluators import (
    GoalSatisfactionEvaluator,
    PlanAdherenceEvaluator,
)
from nanitics.capabilities.planning.store import PlanStore
from nanitics.capabilities.planning.tools import create_planning_tools
from nanitics.core.agents.context import ContextContent, ContextProvider
from nanitics.core.tools.function_tool import FunctionTool
from nanitics.infrastructure.llm.protocol import Message


class _BoundPlanningContextProvider:
    """Context provider that delegates to PlanningContextProvider when a plan is active."""

    def __init__(
        self,
        capability: PlanningCapability,
        store: PlanStore,
        detail: Literal["minimal", "normal", "full"],
    ) -> None:
        self._capability = capability
        self._store = store
        self._detail = detail

    async def provide(self, messages: list[Message]) -> ContextContent | None:
        plan_id = self._capability.active_plan_id
        if plan_id is None:
            return None
        provider = PlanningContextProvider(self._store, plan_id, detail=self._detail)
        return await provider.provide(messages)


class _BoundPlanAdherenceEvaluator:
    """Evaluator that delegates to PlanAdherenceEvaluator when a plan is active."""

    def __init__(
        self,
        capability: PlanningCapability,
        store: PlanStore,
        max_revisions: int,
    ) -> None:
        self._capability = capability
        self._store = store
        self._max_revisions = max_revisions

    @property
    def max_revisions(self) -> int:
        return self._max_revisions

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        plan_id = self._capability.active_plan_id
        if plan_id is None:
            return EvaluationResult(
                verdict=EvaluationVerdict.ACCEPT,
                feedback="No active plan — nothing to enforce.",
                evaluator_name="plan_adherence",
            )
        evaluator = PlanAdherenceEvaluator(self._store, plan_id, max_revisions=self._max_revisions)
        return await evaluator.evaluate(output, context)


class _BoundGoalSatisfactionEvaluator:
    """Evaluator that delegates to GoalSatisfactionEvaluator when a plan is active."""

    def __init__(
        self,
        capability: PlanningCapability,
        store: PlanStore,
        max_revisions: int,
    ) -> None:
        self._capability = capability
        self._store = store
        self._max_revisions = max_revisions

    @property
    def max_revisions(self) -> int:
        return self._max_revisions

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        plan_id = self._capability.active_plan_id
        if plan_id is None:
            return EvaluationResult(
                verdict=EvaluationVerdict.ACCEPT,
                feedback="No active plan — nothing to enforce.",
                evaluator_name="goal_satisfaction",
            )
        evaluator = GoalSatisfactionEvaluator(self._store, plan_id, max_revisions=self._max_revisions)
        return await evaluator.evaluate(output, context)


class PlanningCapability:
    """Bundles planning tools, context provider, and evaluator with automatic plan_id wiring.

    When the agent creates a plan via the bundled tools, the capability
    automatically links the context provider and evaluator to that plan.
    This eliminates the need to manually wire plan IDs across components.
    """

    def __init__(
        self,
        store: PlanStore,
        *,
        namespace: str | None = None,
        context_detail: Literal["minimal", "normal", "full"] = "normal",
        evaluator: Literal["adherence", "goal", None] = "adherence",
        max_revisions: int = 1,
    ) -> None:
        """Initialize the planning capability.

        Args:
            store: Plan store for persistence.
            namespace: Optional namespace scoping plans created by these tools.
            context_detail: Detail level for the context provider.
            evaluator: Type of plan-aware evaluator to use, or ``None`` to skip.
            max_revisions: Maximum revision attempts for the evaluator.
        """
        self._active_plan_id: str | None = None
        self._tools = create_planning_tools(store, namespace, on_plan_created=self._handle_plan_created)
        self._context_provider: ContextProvider = _BoundPlanningContextProvider(self, store, context_detail)

        self._output_evaluator: OutputEvaluator | None
        if evaluator == "adherence":
            self._output_evaluator = _BoundPlanAdherenceEvaluator(self, store, max_revisions)
        elif evaluator == "goal":
            self._output_evaluator = _BoundGoalSatisfactionEvaluator(self, store, max_revisions)
        else:
            self._output_evaluator = None

    @property
    def tools(self) -> list[FunctionTool]:
        return self._tools

    @property
    def context_provider(self) -> ContextProvider:
        return self._context_provider

    @property
    def output_evaluator(self) -> OutputEvaluator | None:
        return self._output_evaluator

    @property
    def active_plan_id(self) -> str | None:
        return self._active_plan_id

    def set_active_plan(self, plan_id: str) -> None:
        """Manually set the active plan ID for context and evaluation."""
        self._active_plan_id = plan_id

    def _handle_plan_created(self, plan_id: str) -> None:
        self._active_plan_id = plan_id
