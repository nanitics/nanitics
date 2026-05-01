"""Tests for PlanningCapability: auto-wiring of planning tools, context provider, and evaluator."""

from nanitics import ToolCall, ToolRegistry
from nanitics.capabilities.evaluation.protocol import (
    EvaluationContext,
    EvaluationVerdict,
)
from nanitics.capabilities.planning.capability import PlanningCapability
from nanitics.capabilities.planning.models import Plan, PlanStep, StepStatus
from nanitics.capabilities.planning.store import InMemoryPlanStore
from tests.testing_helpers import make_emitter


def make_store() -> InMemoryPlanStore:
    return InMemoryPlanStore()


def make_eval_context() -> EvaluationContext:
    return EvaluationContext(messages=[], task_input="test task")


# ──────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────


class TestConstruction:
    def test_defaults(self) -> None:
        cap = PlanningCapability(make_store())
        assert cap.active_plan_id is None
        assert cap.output_evaluator is not None
        assert cap.context_provider is not None

    def test_all_options(self) -> None:
        cap = PlanningCapability(
            make_store(),
            namespace="ns",
            context_detail="full",
            evaluator="goal",
            max_revisions=3,
        )
        assert cap.output_evaluator is not None
        assert cap.output_evaluator.max_revisions == 3

    def test_evaluator_none(self) -> None:
        cap = PlanningCapability(make_store(), evaluator=None)
        assert cap.output_evaluator is None


# ──────────────────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────────────────


class TestTools:
    def test_returns_six_tools(self) -> None:
        cap = PlanningCapability(make_store())
        assert len(cap.tools) == 6

    def test_tool_names(self) -> None:
        cap = PlanningCapability(make_store())
        names = {t.schema.name for t in cap.tools}
        assert names == {"create_plan", "get_plan", "update_step", "revise_plan", "update_goal", "create_goal"}


# ──────────────────────────────────────────────────────────
# Context Provider
# ──────────────────────────────────────────────────────────


class TestContextProvider:
    async def test_returns_none_before_plan_creation(self) -> None:
        cap = PlanningCapability(make_store())
        result = await cap.context_provider.provide([])
        assert result is None

    async def test_returns_content_after_set_active_plan(self) -> None:
        store = make_store()
        plan = Plan(name="Test", steps=[PlanStep(description="Do it")])
        await store.save(plan)

        cap = PlanningCapability(store)
        cap.set_active_plan(plan.id)

        result = await cap.context_provider.provide([])
        assert result is not None
        assert "Test" in result.content


# ──────────────────────────────────────────────────────────
# Evaluator — Adherence (default)
# ──────────────────────────────────────────────────────────


class TestAdherenceEvaluator:
    async def test_accepts_before_plan_creation(self) -> None:
        cap = PlanningCapability(make_store())
        assert cap.output_evaluator is not None
        result = await cap.output_evaluator.evaluate("output", make_eval_context())
        assert result.verdict == EvaluationVerdict.ACCEPT

    async def test_revises_with_incomplete_steps(self) -> None:
        store = make_store()
        plan = Plan(
            name="Incomplete",
            steps=[
                PlanStep(description="Done", status=StepStatus.completed),
                PlanStep(description="Not done"),
            ],
        )
        await store.save(plan)

        cap = PlanningCapability(store)
        cap.set_active_plan(plan.id)

        assert cap.output_evaluator is not None
        result = await cap.output_evaluator.evaluate("output", make_eval_context())
        assert result.verdict == EvaluationVerdict.REVISE
        assert "Not done" in (result.feedback or "")

    async def test_accepts_when_all_complete(self) -> None:
        store = make_store()
        plan = Plan(
            name="Complete",
            steps=[PlanStep(description="Done", status=StepStatus.completed)],
        )
        await store.save(plan)

        cap = PlanningCapability(store)
        cap.set_active_plan(plan.id)

        assert cap.output_evaluator is not None
        result = await cap.output_evaluator.evaluate("output", make_eval_context())
        assert result.verdict == EvaluationVerdict.ACCEPT

    def test_max_revisions(self) -> None:
        cap = PlanningCapability(make_store(), max_revisions=5)
        assert cap.output_evaluator is not None
        assert cap.output_evaluator.max_revisions == 5


# ──────────────────────────────────────────────────────────
# Evaluator — Goal Satisfaction
# ──────────────────────────────────────────────────────────


class TestGoalSatisfactionEvaluator:
    async def test_goal_evaluator_accepts_before_plan_creation(self) -> None:
        cap = PlanningCapability(make_store(), evaluator="goal")
        assert cap.output_evaluator is not None
        result = await cap.output_evaluator.evaluate("output", make_eval_context())
        assert result.verdict == EvaluationVerdict.ACCEPT

    async def test_goal_evaluator_type(self) -> None:
        from nanitics.capabilities.planning.capability import _BoundGoalSatisfactionEvaluator

        cap = PlanningCapability(make_store(), evaluator="goal")
        assert isinstance(cap.output_evaluator, _BoundGoalSatisfactionEvaluator)


# ──────────────────────────────────────────────────────────
# End-to-End: create_plan tool fires callback
# ──────────────────────────────────────────────────────────


class TestEndToEnd:
    async def test_create_plan_tool_wires_plan_id(self) -> None:
        store = make_store()
        emitter = make_emitter()
        cap = PlanningCapability(store)

        registry = ToolRegistry(emitter=emitter)
        for t in cap.tools:
            registry.register(t)

        assert cap.active_plan_id is None

        await registry.dispatch(
            ToolCall(
                id="1",
                name="create_plan",
                arguments={"name": "Auto Plan", "steps": ["Step A", "Step B"]},
            )
        )

        assert cap.active_plan_id is not None
        plans = await store.list_plans()
        assert cap.active_plan_id == plans[0].id

        # Context provider now returns content
        ctx = await cap.context_provider.provide([])
        assert ctx is not None
        assert "Auto Plan" in ctx.content

    async def test_set_active_plan_explicit(self) -> None:
        store = make_store()
        plan = Plan(name="Pre-created", steps=[PlanStep(description="Step 1")])
        await store.save(plan)

        cap = PlanningCapability(store)
        cap.set_active_plan(plan.id)

        assert cap.active_plan_id == plan.id
        ctx = await cap.context_provider.provide([])
        assert ctx is not None
        assert "Pre-created" in ctx.content
