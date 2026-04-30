"""Bridge between task decomposition trees and orchestration workflows."""

from __future__ import annotations

from collections.abc import Callable

from nanitics.capabilities.planning.models import TaskNode, TaskPlan
from nanitics.composition.orchestration.adapters import WorkflowStep
from nanitics.composition.orchestration.dag import DAG, DAGNode
from nanitics.composition.orchestration.parallel import Parallel
from nanitics.composition.orchestration.protocol import Step
from nanitics.composition.orchestration.sequential import Sequential
from nanitics.composition.orchestration.workflow import Workflow
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import PlanCreatedEvent


def plan_to_workflow(
    task_plan: TaskPlan,
    step_factory: Callable[[TaskNode], Step],
    *,
    emitter: EventEmitter,
) -> Workflow:
    """Convert a TaskPlan into an executable orchestration Workflow.

    Analyzes dependencies among the root tasks and recursively converts
    TaskNodes into workflows:

    - Leaf nodes with no inter-dependencies → Parallel
    - Linear dependency chains → Sequential
    - Mixed dependency patterns → DAG
    - Nodes with subtasks become sub-workflows that participate in the parent graph

    Args:
        task_plan: Decomposition tree with root tasks to convert.
        step_factory: Converts a leaf TaskNode into a Step. Only called for
            nodes without subtasks.
        emitter: Event emitter for observability.

    Returns:
        A Workflow (Sequential, Parallel, or DAG) based on the dependency structure.

    Raises:
        ValueError: If the task plan has no root tasks.
    """
    if not task_plan.root_tasks:
        raise ValueError("TaskPlan has no root tasks")

    emitter.emit(
        PlanCreatedEvent(
            trace_id=emitter.trace_id,
            span_id=emitter.span_id,
            parent_span_id=emitter.parent_span_id,
            plan_id=task_plan.id,
            plan_name=task_plan.name,
            step_count=_count_leaf_nodes(task_plan.root_tasks),
            goal_count=0,
        )
    )

    return _nodes_to_workflow(
        name=task_plan.name,
        nodes=task_plan.root_tasks,
        step_factory=step_factory,
        emitter=emitter,
    )


def _node_to_step_or_workflow(
    node: TaskNode,
    step_factory: Callable[[TaskNode], Step],
    emitter: EventEmitter,
) -> Step | Workflow:
    """Convert a single TaskNode into a Step (leaf) or Workflow (has subtasks)."""
    if node.subtasks:
        return _nodes_to_workflow(
            name=node.description,
            nodes=node.subtasks,
            step_factory=step_factory,
            emitter=emitter,
        )
    return step_factory(node)


def _nodes_to_workflow(
    *,
    name: str,
    nodes: list[TaskNode],
    step_factory: Callable[[TaskNode], Step],
    emitter: EventEmitter,
) -> Workflow:
    """Convert a list of sibling TaskNodes into the appropriate workflow type."""
    if len(nodes) == 1:
        node = nodes[0]
        if node.subtasks:
            return _nodes_to_workflow(
                name=node.description,
                nodes=node.subtasks,
                step_factory=step_factory,
                emitter=emitter,
            )
        # Single leaf node — wrap in Sequential for consistent Workflow return type
        return Sequential(name=name, steps=[step_factory(node)], emitter=emitter)

    # Collect the set of node IDs at this level for dependency filtering
    node_ids = {n.id for n in nodes}

    # Check if any node has dependencies on siblings at this level
    has_dependencies = any(dep in node_ids for node in nodes for dep in node.dependencies)

    if not has_dependencies:
        # No inter-dependencies → Parallel
        candidate_steps = [_node_to_step_or_workflow(node, step_factory, emitter) for node in nodes]
        # If any sub-item is a Workflow, we need to wrap it as a Step-compatible
        # object. But Workflow doesn't implement Step protocol directly.
        # For simplicity, only use Parallel when all are leaf Steps.
        all_steps = all(isinstance(s, Step) and not isinstance(s, Workflow) for s in candidate_steps)
        if all_steps:
            return Parallel(name=name, steps=candidate_steps, emitter=emitter)
        # Mixed leaves and sub-workflows — fall through to DAG
        return _build_dag(name=name, nodes=nodes, step_factory=step_factory, emitter=emitter)

    # Has dependencies — check if it's a linear chain
    if _is_linear_chain(nodes, node_ids):
        ordered = _topological_sort(nodes, node_ids)
        steps: list[Step] = []
        for node in ordered:
            result = _node_to_step_or_workflow(node, step_factory, emitter)
            if isinstance(result, Workflow):
                # Sub-workflow in a sequential chain — use DAG for flexibility
                return _build_dag(name=name, nodes=nodes, step_factory=step_factory, emitter=emitter)
            steps.append(result)
        return Sequential(name=name, steps=steps, emitter=emitter)

    # Mixed dependency pattern → DAG
    return _build_dag(name=name, nodes=nodes, step_factory=step_factory, emitter=emitter)


def _build_dag(
    *,
    name: str,
    nodes: list[TaskNode],
    step_factory: Callable[[TaskNode], Step],
    emitter: EventEmitter,
) -> DAG:
    """Build a DAG workflow from nodes with dependencies."""
    node_ids = {n.id for n in nodes}
    dag_nodes: dict[str, DAGNode] = {}

    for node in nodes:
        step = _node_to_step_or_workflow(node, step_factory, emitter)
        # DAG expects Step protocol — Workflow implements execute() but not
        # the Step protocol. Wrap sub-workflows as steps.
        if isinstance(step, Workflow):
            step = WorkflowStep(step)
        sibling_deps = [dep for dep in node.dependencies if dep in node_ids]
        dag_nodes[node.id] = DAGNode(step=step, depends_on=sibling_deps)

    return DAG(name=name, nodes=dag_nodes, emitter=emitter)


def _is_linear_chain(nodes: list[TaskNode], node_ids: set[str]) -> bool:
    """Check if nodes form a simple linear dependency chain."""
    # A linear chain: each node depends on at most one sibling,
    # and each node is depended on by at most one sibling.
    dep_count: dict[str, int] = {n.id: 0 for n in nodes}
    depended_by_count: dict[str, int] = {n.id: 0 for n in nodes}

    for node in nodes:
        sibling_deps = [d for d in node.dependencies if d in node_ids]
        if len(sibling_deps) > 1:
            return False
        dep_count[node.id] = len(sibling_deps)
        for dep in sibling_deps:
            depended_by_count[dep] += 1

    # Each node is depended on by at most one other node
    if any(count > 1 for count in depended_by_count.values()):
        return False

    # Exactly one root (no deps) and one tail (not depended on)
    roots = sum(1 for c in dep_count.values() if c == 0)
    tails = sum(1 for c in depended_by_count.values() if c == 0)
    return roots == 1 and tails == 1


def _topological_sort(nodes: list[TaskNode], node_ids: set[str]) -> list[TaskNode]:
    """Topological sort of nodes by their sibling dependencies."""
    node_map = {n.id: n for n in nodes}
    in_degree: dict[str, int] = {}
    adjacency: dict[str, list[str]] = {n.id: [] for n in nodes}

    for node in nodes:
        sibling_deps = [d for d in node.dependencies if d in node_ids]
        in_degree[node.id] = len(sibling_deps)
        for dep in sibling_deps:
            adjacency[dep].append(node.id)

    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    result: list[TaskNode] = []

    while queue:
        nid = queue.pop(0)
        result.append(node_map[nid])
        for dependent in adjacency[nid]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(result) != len(nodes):
        sorted_ids = {n.id for n in result}
        cyclic = [n.id for n in nodes if n.id not in sorted_ids]
        raise ValueError(f"Circular dependencies detected among tasks: {', '.join(cyclic)}")

    return result


def _count_leaf_nodes(nodes: list[TaskNode]) -> int:
    """Count the total number of leaf nodes in a task tree."""
    count = 0
    for node in nodes:
        if node.subtasks:
            count += _count_leaf_nodes(node.subtasks)
        else:
            count += 1
    return count
