from typing import Any

import pytest

from nanitics.capabilities.planning.models import TaskNode, TaskPlan
from nanitics.composition.orchestration.dag import DAG
from nanitics.composition.orchestration.parallel import Parallel
from nanitics.composition.orchestration.plan_bridge import (
    _is_linear_chain,
    _topological_sort,
    plan_to_workflow,
)
from nanitics.composition.orchestration.protocol import Step, StepResult
from nanitics.composition.orchestration.sequential import Sequential
from nanitics.infrastructure.observability.emitter import InMemoryEmitter


class _FakeStep:
    """Minimal Step implementation for testing."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def execute(self, input: Any) -> StepResult:
        return StepResult(output=f"result:{self._name}")


def _make_step(node: TaskNode) -> Step:
    return _FakeStep(node.description)


def _emitter() -> InMemoryEmitter:
    return InMemoryEmitter(trace_id="test-trace")


class TestPlanToWorkflowBasic:
    def test_empty_plan_raises(self) -> None:
        plan = TaskPlan(name="empty")
        with pytest.raises(ValueError, match="no root tasks"):
            plan_to_workflow(plan, _make_step, emitter=_emitter())

    def test_single_task_returns_sequential(self) -> None:
        plan = TaskPlan(
            name="single",
            root_tasks=[TaskNode(id="a", description="Task A")],
        )
        wf = plan_to_workflow(plan, _make_step, emitter=_emitter())
        assert isinstance(wf, Sequential)
        assert wf.name == "single"

    async def test_single_task_executes(self) -> None:
        plan = TaskPlan(
            name="single",
            root_tasks=[TaskNode(id="a", description="Task A")],
        )
        wf = plan_to_workflow(plan, _make_step, emitter=_emitter())
        result = await wf.execute(None)
        assert result.output == "result:Task A"


class TestIndependentTasks:
    def test_independent_tasks_return_parallel(self) -> None:
        plan = TaskPlan(
            name="parallel-plan",
            root_tasks=[
                TaskNode(id="a", description="Task A"),
                TaskNode(id="b", description="Task B"),
                TaskNode(id="c", description="Task C"),
            ],
        )
        wf = plan_to_workflow(plan, _make_step, emitter=_emitter())
        assert isinstance(wf, Parallel)
        assert wf.name == "parallel-plan"

    async def test_independent_tasks_execute(self) -> None:
        plan = TaskPlan(
            name="parallel-plan",
            root_tasks=[
                TaskNode(id="a", description="Task A"),
                TaskNode(id="b", description="Task B"),
            ],
        )
        wf = plan_to_workflow(plan, _make_step, emitter=_emitter())
        result = await wf.execute(None)
        outputs = result.output
        assert len(outputs) == 2


class TestLinearChain:
    def test_linear_chain_returns_sequential(self) -> None:
        plan = TaskPlan(
            name="seq-plan",
            root_tasks=[
                TaskNode(id="a", description="Task A"),
                TaskNode(id="b", description="Task B", dependencies=["a"]),
                TaskNode(id="c", description="Task C", dependencies=["b"]),
            ],
        )
        wf = plan_to_workflow(plan, _make_step, emitter=_emitter())
        assert isinstance(wf, Sequential)
        assert wf.name == "seq-plan"

    async def test_linear_chain_executes_in_order(self) -> None:
        plan = TaskPlan(
            name="seq-plan",
            root_tasks=[
                TaskNode(id="a", description="Task A"),
                TaskNode(id="b", description="Task B", dependencies=["a"]),
            ],
        )
        wf = plan_to_workflow(plan, _make_step, emitter=_emitter())
        result = await wf.execute("start")
        # Sequential passes output of one step to the next
        assert result.output is not None


class TestMixedDependencies:
    def test_mixed_deps_return_dag(self) -> None:
        # A -> C, B -> C (diamond/fan-in pattern)
        plan = TaskPlan(
            name="dag-plan",
            root_tasks=[
                TaskNode(id="a", description="Task A"),
                TaskNode(id="b", description="Task B"),
                TaskNode(id="c", description="Task C", dependencies=["a", "b"]),
            ],
        )
        wf = plan_to_workflow(plan, _make_step, emitter=_emitter())
        assert isinstance(wf, DAG)
        assert wf.name == "dag-plan"

    async def test_mixed_deps_execute(self) -> None:
        plan = TaskPlan(
            name="dag-plan",
            root_tasks=[
                TaskNode(id="a", description="Task A"),
                TaskNode(id="b", description="Task B"),
                TaskNode(id="c", description="Task C", dependencies=["a", "b"]),
            ],
        )
        wf = plan_to_workflow(plan, _make_step, emitter=_emitter())
        result = await wf.execute("start")
        assert result.output is not None


class TestNestedSubtasks:
    def test_nested_subtasks_create_sub_workflow(self) -> None:
        plan = TaskPlan(
            name="nested",
            root_tasks=[
                TaskNode(
                    id="parent",
                    description="Parent Task",
                    subtasks=[
                        TaskNode(id="child1", description="Child 1"),
                        TaskNode(id="child2", description="Child 2"),
                    ],
                ),
            ],
        )
        wf = plan_to_workflow(plan, _make_step, emitter=_emitter())
        # Single parent with subtasks → sub-workflow from children
        # Two independent children → Parallel
        assert isinstance(wf, Parallel)

    def test_nested_with_sibling_deps(self) -> None:
        plan = TaskPlan(
            name="nested-deps",
            root_tasks=[
                TaskNode(id="a", description="Task A"),
                TaskNode(
                    id="b",
                    description="Parent B",
                    subtasks=[
                        TaskNode(id="b1", description="Child B1"),
                        TaskNode(id="b2", description="Child B2"),
                    ],
                    dependencies=["a"],
                ),
            ],
        )
        wf = plan_to_workflow(plan, _make_step, emitter=_emitter())
        # a -> b (which has subtasks) → needs DAG (b becomes a sub-workflow)
        assert isinstance(wf, DAG)


class TestEventEmission:
    def test_emits_plan_created_event(self) -> None:
        emitter = _emitter()
        plan = TaskPlan(
            id="plan-123",
            name="test-plan",
            root_tasks=[
                TaskNode(id="a", description="Task A"),
                TaskNode(id="b", description="Task B"),
            ],
        )
        plan_to_workflow(plan, _make_step, emitter=emitter)

        events = emitter.events
        plan_events = [e for e in events if e.event_type == "planning.plan.created"]
        assert len(plan_events) == 1
        event = plan_events[0]
        assert event.plan_id == "plan-123"
        assert event.plan_name == "test-plan"
        assert event.step_count == 2
        assert event.goal_count == 0

    def test_leaf_count_with_nested_subtasks(self) -> None:
        emitter = _emitter()
        plan = TaskPlan(
            id="plan-456",
            name="nested-plan",
            root_tasks=[
                TaskNode(
                    id="parent",
                    description="Parent",
                    subtasks=[
                        TaskNode(id="c1", description="Child 1"),
                        TaskNode(id="c2", description="Child 2"),
                    ],
                ),
                TaskNode(id="leaf", description="Leaf"),
            ],
        )
        plan_to_workflow(plan, _make_step, emitter=emitter)

        plan_events = [e for e in emitter.events if e.event_type == "planning.plan.created"]
        assert plan_events[0].step_count == 3  # 2 children + 1 leaf


class TestIsLinearChainFanOut:
    def test_fan_out_is_not_linear(self) -> None:
        """A node depended on by multiple siblings is not a linear chain (fan-out)."""
        nodes = [
            TaskNode(id="a", description="A"),
            TaskNode(id="b", description="B", dependencies=["a"]),
            TaskNode(id="c", description="C", dependencies=["a"]),
        ]
        node_ids = {n.id for n in nodes}
        assert _is_linear_chain(nodes, node_ids) is False

    def test_fan_out_produces_dag(self) -> None:
        """Fan-out pattern at root level produces a DAG workflow."""
        plan = TaskPlan(
            name="fan-out",
            root_tasks=[
                TaskNode(id="a", description="A"),
                TaskNode(id="b", description="B", dependencies=["a"]),
                TaskNode(id="c", description="C", dependencies=["a"]),
            ],
        )
        wf = plan_to_workflow(plan, _make_step, emitter=_emitter())
        assert isinstance(wf, DAG)


class TestTopologicalSortCycle:
    def test_circular_dependency_raises(self) -> None:
        """Circular dependencies in _topological_sort raise ValueError."""
        nodes = [
            TaskNode(id="a", description="A", dependencies=["b"]),
            TaskNode(id="b", description="B", dependencies=["a"]),
        ]
        node_ids = {n.id for n in nodes}
        with pytest.raises(ValueError, match="Circular dependencies"):
            _topological_sort(nodes, node_ids)


class TestCircularDependencies:
    def test_circular_deps_raise_error(self) -> None:
        plan = TaskPlan(
            name="circular",
            root_tasks=[
                TaskNode(id="a", description="Task A", dependencies=["b"]),
                TaskNode(id="b", description="Task B", dependencies=["a"]),
            ],
        )
        with pytest.raises(ValueError, match="cycle"):
            plan_to_workflow(plan, _make_step, emitter=_emitter())

    def test_three_node_cycle_raises_error(self) -> None:
        plan = TaskPlan(
            name="cycle-3",
            root_tasks=[
                TaskNode(id="a", description="Task A", dependencies=["c"]),
                TaskNode(id="b", description="Task B", dependencies=["a"]),
                TaskNode(id="c", description="Task C", dependencies=["b"]),
            ],
        )
        with pytest.raises(ValueError, match="cycle"):
            plan_to_workflow(plan, _make_step, emitter=_emitter())
