from nanitics.capabilities.evaluation.protocol import (
    EvaluationContext,
    EvaluationVerdict,
    OutputEvaluator,
)
from nanitics.capabilities.planning.evaluators import (
    GoalSatisfactionEvaluator,
    PlanAdherenceEvaluator,
)
from nanitics.capabilities.planning.models import (
    Goal,
    GoalStatus,
    Plan,
    PlanStatus,
    PlanStep,
    StepStatus,
)
from nanitics.capabilities.planning.store import InMemoryPlanStore
from nanitics.infrastructure.llm.protocol import Message


def _make_context() -> EvaluationContext:
    return EvaluationContext(
        messages=[Message(role="user", content="do the task")],
        task_input="do the task",
    )


def _plan_with_steps(*statuses: StepStatus) -> Plan:
    steps = [PlanStep(id=f"s{i}", description=f"Step {i}", status=status) for i, status in enumerate(statuses, 1)]
    return Plan(id="plan-1", name="Test Plan", steps=steps)


def _plan_with_goals(*goals: Goal) -> Plan:
    return Plan(id="plan-1", name="Test Plan", goals=list(goals))


# --- PlanAdherenceEvaluator ---


class TestPlanAdherenceEvaluator:
    async def test_satisfies_protocol(self) -> None:
        store = InMemoryPlanStore()
        evaluator = PlanAdherenceEvaluator(store, "plan-1")
        assert isinstance(evaluator, OutputEvaluator)

    async def test_max_revisions(self) -> None:
        store = InMemoryPlanStore()
        evaluator = PlanAdherenceEvaluator(store, "plan-1", max_revisions=3)
        assert evaluator.max_revisions == 3

    async def test_all_steps_completed(self) -> None:
        store = InMemoryPlanStore()
        plan = _plan_with_steps(StepStatus.completed, StepStatus.completed, StepStatus.completed)
        await store.save(plan)

        evaluator = PlanAdherenceEvaluator(store, "plan-1")
        result = await evaluator.evaluate("done", _make_context())

        assert result.verdict == EvaluationVerdict.ACCEPT
        assert result.score == 1.0
        assert result.evaluator_name == "plan_adherence"

    async def test_completed_and_skipped(self) -> None:
        store = InMemoryPlanStore()
        plan = _plan_with_steps(StepStatus.completed, StepStatus.skipped, StepStatus.completed)
        await store.save(plan)

        evaluator = PlanAdherenceEvaluator(store, "plan-1")
        result = await evaluator.evaluate("done", _make_context())

        assert result.verdict == EvaluationVerdict.ACCEPT
        assert result.score == 1.0

    async def test_incomplete_steps_revise(self) -> None:
        store = InMemoryPlanStore()
        plan = _plan_with_steps(StepStatus.completed, StepStatus.not_started, StepStatus.in_progress)
        await store.save(plan)

        evaluator = PlanAdherenceEvaluator(store, "plan-1")
        result = await evaluator.evaluate("done", _make_context())

        assert result.verdict == EvaluationVerdict.REVISE
        assert result.score == 0.0
        assert result.feedback is not None
        assert "Step 2" in result.feedback
        assert "Step 3" in result.feedback

    async def test_all_not_started_revise(self) -> None:
        store = InMemoryPlanStore()
        plan = _plan_with_steps(StepStatus.not_started, StepStatus.not_started)
        await store.save(plan)

        evaluator = PlanAdherenceEvaluator(store, "plan-1")
        result = await evaluator.evaluate("done", _make_context())

        assert result.verdict == EvaluationVerdict.REVISE

    async def test_abandoned_plan_accepts(self) -> None:
        store = InMemoryPlanStore()
        plan = _plan_with_steps(StepStatus.not_started, StepStatus.not_started)
        plan = plan.model_copy(update={"status": PlanStatus.abandoned})
        await store.save(plan)

        evaluator = PlanAdherenceEvaluator(store, "plan-1")
        result = await evaluator.evaluate("done", _make_context())

        assert result.verdict == EvaluationVerdict.ACCEPT
        assert result.feedback is not None
        assert "abandoned" in result.feedback.lower()

    async def test_plan_not_found_accepts(self) -> None:
        store = InMemoryPlanStore()

        evaluator = PlanAdherenceEvaluator(store, "nonexistent")
        result = await evaluator.evaluate("done", _make_context())

        assert result.verdict == EvaluationVerdict.ACCEPT

    async def test_mixed_statuses(self) -> None:
        store = InMemoryPlanStore()
        plan = _plan_with_steps(
            StepStatus.completed,
            StepStatus.skipped,
            StepStatus.failed,
            StepStatus.not_started,
        )
        await store.save(plan)

        evaluator = PlanAdherenceEvaluator(store, "plan-1")
        result = await evaluator.evaluate("done", _make_context())

        # failed counts as terminal — only not_started triggers revise
        assert result.verdict == EvaluationVerdict.REVISE
        assert result.feedback is not None
        assert "Step 4" in result.feedback


# --- GoalSatisfactionEvaluator ---


class TestGoalSatisfactionEvaluator:
    async def test_satisfies_protocol(self) -> None:
        store = InMemoryPlanStore()
        evaluator = GoalSatisfactionEvaluator(store, "plan-1")
        assert isinstance(evaluator, OutputEvaluator)

    async def test_max_revisions(self) -> None:
        store = InMemoryPlanStore()
        evaluator = GoalSatisfactionEvaluator(store, "plan-1", max_revisions=5)
        assert evaluator.max_revisions == 5

    async def test_all_goals_achieved(self) -> None:
        store = InMemoryPlanStore()
        plan = _plan_with_goals(
            Goal(id="g1", description="Goal 1", status=GoalStatus.achieved),
            Goal(id="g2", description="Goal 2", status=GoalStatus.achieved),
        )
        await store.save(plan)

        evaluator = GoalSatisfactionEvaluator(store, "plan-1")
        result = await evaluator.evaluate("done", _make_context())

        assert result.verdict == EvaluationVerdict.ACCEPT
        assert result.score == 1.0
        assert result.evaluator_name == "goal_satisfaction"

    async def test_all_goals_abandoned(self) -> None:
        store = InMemoryPlanStore()
        plan = _plan_with_goals(
            Goal(id="g1", description="Goal 1", status=GoalStatus.abandoned),
        )
        await store.save(plan)

        evaluator = GoalSatisfactionEvaluator(store, "plan-1")
        result = await evaluator.evaluate("done", _make_context())

        assert result.verdict == EvaluationVerdict.ACCEPT

    async def test_active_goals_revise(self) -> None:
        store = InMemoryPlanStore()
        plan = _plan_with_goals(
            Goal(id="g1", description="Goal 1", status=GoalStatus.achieved),
            Goal(id="g2", description="Goal 2", status=GoalStatus.active),
        )
        await store.save(plan)

        evaluator = GoalSatisfactionEvaluator(store, "plan-1")
        result = await evaluator.evaluate("done", _make_context())

        assert result.verdict == EvaluationVerdict.REVISE
        assert result.score == 0.0
        assert result.feedback is not None
        assert "Goal 2" in result.feedback

    async def test_nested_active_subgoal_revise(self) -> None:
        store = InMemoryPlanStore()
        plan = _plan_with_goals(
            Goal(
                id="g1",
                description="Parent",
                status=GoalStatus.achieved,
                subgoals=[
                    Goal(id="g1a", description="Child Active", status=GoalStatus.active),
                ],
            ),
        )
        await store.save(plan)

        evaluator = GoalSatisfactionEvaluator(store, "plan-1")
        result = await evaluator.evaluate("done", _make_context())

        assert result.verdict == EvaluationVerdict.REVISE
        assert result.feedback is not None
        assert "Child Active" in result.feedback

    async def test_nested_all_achieved_accepts(self) -> None:
        store = InMemoryPlanStore()
        plan = _plan_with_goals(
            Goal(
                id="g1",
                description="Parent",
                status=GoalStatus.achieved,
                subgoals=[
                    Goal(id="g1a", description="Child", status=GoalStatus.achieved),
                    Goal(id="g1b", description="Child 2", status=GoalStatus.abandoned),
                ],
            ),
        )
        await store.save(plan)

        evaluator = GoalSatisfactionEvaluator(store, "plan-1")
        result = await evaluator.evaluate("done", _make_context())

        assert result.verdict == EvaluationVerdict.ACCEPT

    async def test_no_goals_accepts(self) -> None:
        store = InMemoryPlanStore()
        plan = Plan(id="plan-1", name="No Goals")
        await store.save(plan)

        evaluator = GoalSatisfactionEvaluator(store, "plan-1")
        result = await evaluator.evaluate("done", _make_context())

        assert result.verdict == EvaluationVerdict.ACCEPT

    async def test_plan_not_found_accepts(self) -> None:
        store = InMemoryPlanStore()

        evaluator = GoalSatisfactionEvaluator(store, "nonexistent")
        result = await evaluator.evaluate("done", _make_context())

        assert result.verdict == EvaluationVerdict.ACCEPT

    async def test_mixed_goal_statuses(self) -> None:
        store = InMemoryPlanStore()
        plan = _plan_with_goals(
            Goal(id="g1", description="Done", status=GoalStatus.achieved),
            Goal(id="g2", description="Blocked", status=GoalStatus.blocked),
            Goal(id="g3", description="Still Active", status=GoalStatus.active),
            Goal(id="g4", description="Dropped", status=GoalStatus.abandoned),
        )
        await store.save(plan)

        evaluator = GoalSatisfactionEvaluator(store, "plan-1")
        result = await evaluator.evaluate("done", _make_context())

        assert result.verdict == EvaluationVerdict.REVISE
        assert result.feedback is not None
        assert "Still Active" in result.feedback
        # blocked is not active — only active status triggers revise
        assert "Blocked" not in result.feedback

    async def test_abandoned_plan_with_active_goals_accepts(self) -> None:
        store = InMemoryPlanStore()
        plan = _plan_with_goals(
            Goal(id="g1", description="Active Goal", status=GoalStatus.active),
        )
        plan = plan.model_copy(update={"status": PlanStatus.abandoned})
        await store.save(plan)

        evaluator = GoalSatisfactionEvaluator(store, "plan-1")
        result = await evaluator.evaluate("done", _make_context())

        assert result.verdict == EvaluationVerdict.ACCEPT
        assert result.feedback is not None
        assert "abandoned" in result.feedback.lower()
