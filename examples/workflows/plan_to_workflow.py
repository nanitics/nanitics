"""Plan-to-workflow bridge: converting task plans into executable workflows.

Demonstrates ``plan_to_workflow`` — the function that converts a ``TaskPlan`` (a dependency
tree of ``TaskNode``s) into an executable ``Workflow``, automatically selecting ``Parallel``,
``Sequential``, or ``DAG`` based on the dependency structure. This bridges two SDK systems:
the planning capability produces task plans; orchestration executes workflows.

Related guide: docs/guides/orchestration.md
"""

import asyncio

from examples.helpers import make_emitter
from nanitics import (
    DAG,
    FunctionStep,
    Parallel,
    Sequential,
)
from nanitics.specialized import (
    TaskNode,
    TaskPlan,
    plan_to_workflow,
)


async def main() -> None:
    # --- Section 1: Independent Tasks → Parallel ---
    print("--- Section 1: Independent Tasks → Parallel ---")

    # Three tasks with no dependencies. plan_to_workflow selects Parallel.

    # Empty plan raises ValueError
    empty_plan = TaskPlan(name="empty")
    emitter = make_emitter("plan-bridge-s1")
    try:
        plan_to_workflow(empty_plan, lambda n: FunctionStep(n.id, lambda x: x), emitter=emitter)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "no root tasks" in str(exc).lower()
        print(f"  Empty plan error: {exc}")

    print("✓ Empty plan raises ValueError")

    # Independent tasks
    plan = TaskPlan(
        name="independent-analysis",
        root_tasks=[
            TaskNode(id="revenue", description="Analyze revenue"),
            TaskNode(id="costs", description="Analyze costs"),
            TaskNode(id="growth", description="Analyze growth"),
        ],
    )

    async def analyze(input_data: str) -> str:
        return f"analyzed: {input_data}"

    def step_factory(node: TaskNode) -> FunctionStep:
        return FunctionStep(node.id, analyze)

    emitter = make_emitter("plan-bridge-s1b")
    workflow = plan_to_workflow(plan, step_factory, emitter=emitter)

    assert isinstance(workflow, Parallel), f"Expected Parallel, got {type(workflow).__name__}"
    print(f"  Workflow type: {type(workflow).__name__}")

    result = await workflow.execute("Q4 data")

    # Parallel returns a list of outputs
    assert isinstance(result.output, list)
    assert len(result.output) == 3
    assert all(item == "analyzed: Q4 data" for item in result.output)

    print(f"  Outputs: {result.output}")
    print("✓ Independent tasks produce Parallel workflow with all outputs")

    # --- Section 2: Linear Chain → Sequential ---
    print("\n--- Section 2: Linear Chain → Sequential ---")

    # Three tasks with linear dependencies: extract → transform → load.
    # plan_to_workflow selects Sequential.

    plan = TaskPlan(
        name="etl-pipeline",
        root_tasks=[
            TaskNode(id="extract", description="Extract data"),
            TaskNode(id="transform", description="Transform data", dependencies=["extract"]),
            TaskNode(id="load", description="Load data", dependencies=["transform"]),
        ],
    )

    step_names: list[str] = []

    async def etl_step(input_data: str) -> str:
        return f"[processed] {input_data}"

    def tracking_factory(node: TaskNode) -> FunctionStep:
        step_names.append(node.id)

        async def step_fn(input_data: str) -> str:
            return f"[{node.id}] {input_data}"

        return FunctionStep(node.id, step_fn)

    emitter = make_emitter("plan-bridge-s2")
    workflow = plan_to_workflow(plan, tracking_factory, emitter=emitter)

    assert isinstance(workflow, Sequential), f"Expected Sequential, got {type(workflow).__name__}"
    print(f"  Workflow type: {type(workflow).__name__}")

    # Factory was called in topological order
    assert step_names == ["extract", "transform", "load"]

    result = await workflow.execute("raw data")

    # Each step chains into the next
    assert result.output == "[load] [transform] [extract] raw data"

    print(f"  Output: {result.output}")
    print("✓ Linear chain produces Sequential workflow with chained outputs")

    # --- Section 3: Mixed Dependencies → DAG ---
    print("\n--- Section 3: Mixed Dependencies → DAG ---")

    # Diamond pattern: gather and context are independent, analyze depends on both,
    # report depends on analyze. plan_to_workflow selects DAG.

    plan = TaskPlan(
        name="research-project",
        root_tasks=[
            TaskNode(id="gather", description="Gather data"),
            TaskNode(id="context", description="Gather context"),
            TaskNode(id="analyze", description="Analyze", dependencies=["gather", "context"]),
            TaskNode(id="report", description="Write report", dependencies=["analyze"]),
        ],
    )

    async def gather_fn(input_data: str) -> str:
        return f"data from {input_data}"

    async def context_fn(input_data: str) -> str:
        return f"context for {input_data}"

    async def analyze_fn(input_data: dict) -> str:
        # DAG passes a dict of dependency outputs to nodes with multiple dependencies
        return f"analysis of {input_data}"

    async def report_fn(input_data: str) -> str:
        return f"report: {input_data}"

    def dag_factory(node: TaskNode) -> FunctionStep:
        fns = {
            "gather": gather_fn,
            "context": context_fn,
            "analyze": analyze_fn,
            "report": report_fn,
        }
        return FunctionStep(node.id, fns[node.id])

    emitter = make_emitter("plan-bridge-s3")
    workflow = plan_to_workflow(plan, dag_factory, emitter=emitter)

    assert isinstance(workflow, DAG), f"Expected DAG, got {type(workflow).__name__}"
    print(f"  Workflow type: {type(workflow).__name__}")

    result = await workflow.execute("research topic")

    # The terminal node (report) produces the final output
    assert "report:" in result.output
    assert "analysis of" in result.output

    print(f"  Output: {result.output}")
    print("✓ Mixed dependencies produce DAG workflow respecting dependency order")

    # --- Section 4: Subtasks → Recursive Sub-Workflows ---
    print("\n--- Section 4: Subtasks → Recursive Sub-Workflows ---")

    # A root task with subtasks becomes a sub-workflow. step_factory is only
    # called for leaf nodes (nodes without subtasks).

    leaf_nodes_converted: list[str] = []

    plan = TaskPlan(
        name="nested-project",
        root_tasks=[
            TaskNode(
                id="research",
                description="Research phase",
                subtasks=[
                    TaskNode(id="literature", description="Search literature"),
                    TaskNode(id="patents", description="Search patents"),
                ],
            ),
            TaskNode(id="summarize", description="Summarize findings", dependencies=["research"]),
        ],
    )

    def subtask_factory(node: TaskNode) -> FunctionStep:
        leaf_nodes_converted.append(node.id)

        async def step_fn(input_data: object) -> str:
            return f"{node.id}: done"

        return FunctionStep(node.id, step_fn)

    emitter = make_emitter("plan-bridge-s4")
    workflow = plan_to_workflow(plan, subtask_factory, emitter=emitter)

    # step_factory was only called for leaf nodes, not the "research" parent
    assert "research" not in leaf_nodes_converted
    assert "literature" in leaf_nodes_converted
    assert "patents" in leaf_nodes_converted
    assert "summarize" in leaf_nodes_converted
    # (May be called more than once per node due to internal workflow selection probing)

    print(f"  Leaf nodes converted: {leaf_nodes_converted}")
    print(f"  Workflow type: {type(workflow).__name__}")

    result = await workflow.execute("start")

    print(f"  Output: {result.output}")
    print("✓ Subtasks create nested sub-workflows; step_factory only called for leaves")

    print("\n✅ All sections passed!")


if __name__ == "__main__":
    asyncio.run(main())
