"""Tests for planning data models: construction, defaults, frozen enforcement, model_copy mutations."""

import pytest
from pydantic import ValidationError

from nanitics.capabilities.planning.models import (
    Goal,
    GoalStatus,
    Plan,
    PlanStatus,
    PlanStep,
    StepStatus,
    TaskNode,
    TaskPlan,
)

# ──────────────────────────────────────────────────────────
# StepStatus / PlanStatus / GoalStatus Enums
# ──────────────────────────────────────────────────────────


class TestEnums:
    def test_step_status_values(self) -> None:
        assert set(StepStatus) == {
            StepStatus.not_started,
            StepStatus.in_progress,
            StepStatus.completed,
            StepStatus.skipped,
            StepStatus.failed,
        }

    def test_plan_status_values(self) -> None:
        assert set(PlanStatus) == {
            PlanStatus.active,
            PlanStatus.completed,
            PlanStatus.abandoned,
        }

    def test_goal_status_values(self) -> None:
        assert set(GoalStatus) == {
            GoalStatus.active,
            GoalStatus.achieved,
            GoalStatus.blocked,
            GoalStatus.abandoned,
        }


# ──────────────────────────────────────────────────────────
# PlanStep
# ──────────────────────────────────────────────────────────


class TestPlanStep:
    def test_construction_with_defaults(self) -> None:
        step = PlanStep(description="Do something")
        assert step.description == "Do something"
        assert step.id  # auto-generated
        assert step.status == StepStatus.not_started
        assert step.result is None
        assert step.dependencies == []
        assert step.metadata == {}

    def test_construction_with_all_fields(self) -> None:
        step = PlanStep(
            id="step-1",
            description="Do something",
            status=StepStatus.completed,
            result="Done",
            dependencies=["step-0"],
            metadata={"key": "value"},
        )
        assert step.id == "step-1"
        assert step.status == StepStatus.completed
        assert step.result == "Done"
        assert step.dependencies == ["step-0"]
        assert step.metadata == {"key": "value"}

    def test_frozen(self) -> None:
        step = PlanStep(description="Do something")
        with pytest.raises(ValidationError):
            step.status = StepStatus.completed

    def test_model_copy_updates_status(self) -> None:
        step = PlanStep(id="s1", description="Do something")
        updated = step.model_copy(update={"status": StepStatus.in_progress})
        assert updated.status == StepStatus.in_progress
        assert updated.id == "s1"
        assert step.status == StepStatus.not_started  # original unchanged

    def test_model_copy_adds_result(self) -> None:
        step = PlanStep(id="s1", description="Do something")
        updated = step.model_copy(update={"result": "All done"})
        assert updated.result == "All done"
        assert step.result is None


# ──────────────────────────────────────────────────────────
# Goal
# ──────────────────────────────────────────────────────────


class TestGoal:
    def test_construction_with_defaults(self) -> None:
        goal = Goal(description="Achieve something")
        assert goal.description == "Achieve something"
        assert goal.id  # auto-generated
        assert goal.status == GoalStatus.active
        assert goal.priority == 0
        assert goal.success_criteria is None
        assert goal.subgoals == []

    def test_construction_with_all_fields(self) -> None:
        subgoal = Goal(description="Sub-goal")
        goal = Goal(
            id="g1",
            description="Main goal",
            status=GoalStatus.achieved,
            priority=5,
            success_criteria="All done",
            subgoals=[subgoal],
        )
        assert goal.id == "g1"
        assert goal.priority == 5
        assert goal.success_criteria == "All done"
        assert len(goal.subgoals) == 1
        assert goal.subgoals[0].description == "Sub-goal"

    def test_frozen(self) -> None:
        goal = Goal(description="Test")
        with pytest.raises(ValidationError):
            goal.status = GoalStatus.achieved

    def test_recursive_subgoals(self) -> None:
        leaf = Goal(description="Leaf goal")
        mid = Goal(description="Mid goal", subgoals=[leaf])
        root = Goal(description="Root goal", subgoals=[mid])
        assert root.subgoals[0].subgoals[0].description == "Leaf goal"

    def test_model_copy_updates_status(self) -> None:
        goal = Goal(id="g1", description="Test")
        updated = goal.model_copy(update={"status": GoalStatus.blocked})
        assert updated.status == GoalStatus.blocked
        assert goal.status == GoalStatus.active


# ──────────────────────────────────────────────────────────
# Plan
# ──────────────────────────────────────────────────────────


class TestPlan:
    def test_construction_with_defaults(self) -> None:
        plan = Plan(name="My plan")
        assert plan.name == "My plan"
        assert plan.id  # auto-generated
        assert plan.description is None
        assert plan.status == PlanStatus.active
        assert plan.steps == []
        assert plan.goals == []
        assert plan.namespace is None
        assert plan.metadata == {}
        assert plan.created_at is not None
        assert plan.updated_at is not None

    def test_construction_with_steps_and_goals(self) -> None:
        steps = [PlanStep(description="Step 1"), PlanStep(description="Step 2")]
        goals = [Goal(description="Goal 1")]
        plan = Plan(name="Full plan", steps=steps, goals=goals, namespace="test")
        assert len(plan.steps) == 2
        assert len(plan.goals) == 1
        assert plan.namespace == "test"

    def test_frozen(self) -> None:
        plan = Plan(name="Test")
        with pytest.raises(ValidationError):
            plan.status = PlanStatus.completed

    def test_model_copy_updates_status(self) -> None:
        plan = Plan(id="p1", name="Test")
        updated = plan.model_copy(update={"status": PlanStatus.completed})
        assert updated.status == PlanStatus.completed
        assert plan.status == PlanStatus.active

    def test_model_copy_replaces_steps(self) -> None:
        original_steps = [PlanStep(description="Step 1")]
        plan = Plan(name="Test", steps=original_steps)
        new_steps = [PlanStep(description="Step A"), PlanStep(description="Step B")]
        updated = plan.model_copy(update={"steps": new_steps})
        assert len(updated.steps) == 2
        assert updated.steps[0].description == "Step A"
        assert len(plan.steps) == 1  # original unchanged


# ──────────────────────────────────────────────────────────
# TaskNode / TaskPlan
# ──────────────────────────────────────────────────────────


class TestTaskNode:
    def test_construction_with_defaults(self) -> None:
        node = TaskNode(description="Task A")
        assert node.description == "Task A"
        assert node.id  # auto-generated
        assert node.subtasks == []
        assert node.dependencies == []

    def test_recursive_subtasks(self) -> None:
        leaf = TaskNode(description="Leaf")
        mid = TaskNode(description="Mid", subtasks=[leaf])
        root = TaskNode(description="Root", subtasks=[mid])
        assert root.subtasks[0].subtasks[0].description == "Leaf"

    def test_frozen(self) -> None:
        node = TaskNode(description="Test")
        with pytest.raises(ValidationError):
            node.description = "Changed"

    def test_with_dependencies(self) -> None:
        node = TaskNode(id="t2", description="Task B", dependencies=["t1"])
        assert node.dependencies == ["t1"]


class TestTaskPlan:
    def test_construction_with_defaults(self) -> None:
        tp = TaskPlan(name="Task plan")
        assert tp.name == "Task plan"
        assert tp.id  # auto-generated
        assert tp.root_tasks == []

    def test_construction_with_tasks(self) -> None:
        tasks = [TaskNode(description="T1"), TaskNode(description="T2")]
        tp = TaskPlan(name="Plan", root_tasks=tasks)
        assert len(tp.root_tasks) == 2

    def test_frozen(self) -> None:
        tp = TaskPlan(name="Test")
        with pytest.raises(ValidationError):
            tp.name = "Changed"
