"""DAG: dependency-driven workflow execution.

Demonstrates ``DAG`` — the orchestration workflow that executes steps as a
directed acyclic graph, scheduling nodes concurrently when their dependencies
are satisfied. Covers diamond execution, input routing rules, multiple terminal
nodes, construction-time validation, ``BEST_EFFORT`` failure handling with
transitive dependent skipping, and concurrency limiting.

Related guide: docs/guides/orchestration.md
"""

import asyncio

from examples.helpers import make_emitter
from nanitics.composition import (
    DAG,
    DAGNode,
    FailurePolicy,
    FunctionStep,
)
from nanitics.infrastructure import (
    WorkflowStartEvent,
    WorkflowStepCompleteEvent,
)


async def main() -> None:
    # --- Section 1: Diamond DAG ---
    print("--- Section 1: Diamond DAG ---")

    # Four nodes in a diamond shape: two sources feed into an analyzer,
    # which feeds into a summarizer. Sources run concurrently, downstream
    # nodes wait for their dependencies.
    #
    #   fetch_data ──┐
    #                ├──→ analyze ──→ summarize
    #   fetch_ctx  ──┘

    async def fetch_data(input: str) -> str:
        return f"market data for {input}"

    async def fetch_ctx(input: str) -> str:
        return f"context for {input}"

    async def analyze(inputs: dict[str, str]) -> str:
        return f"analysis of {inputs['fetch_data']} with {inputs['fetch_ctx']}"

    async def summarize(input: str) -> str:
        return f"summary: {input}"

    emitter = make_emitter("dag-s1")

    dag = DAG(
        name="diamond",
        nodes={
            "fetch_data": DAGNode(step=FunctionStep("fetch_data", fetch_data)),
            "fetch_ctx": DAGNode(step=FunctionStep("fetch_ctx", fetch_ctx)),
            "analyze": DAGNode(
                step=FunctionStep("analyze", analyze),
                depends_on=["fetch_data", "fetch_ctx"],
            ),
            "summarize": DAGNode(
                step=FunctionStep("summarize", summarize),
                depends_on=["analyze"],
            ),
        },
        emitter=emitter,
    )

    result = await dag.execute("Q3 earnings")

    # Single terminal node → output is that node's output directly
    assert result.output == "summary: analysis of market data for Q3 earnings with context for Q3 earnings"

    # Metadata tracks all executed nodes
    assert result.metadata["total_steps_executed"] == 4
    assert set(result.metadata["node_results"].keys()) == {
        "fetch_data",
        "fetch_ctx",
        "analyze",
        "summarize",
    }

    # Events: one start event listing all nodes, one step-complete per node
    start_events = [e for e in emitter.events if isinstance(e, WorkflowStartEvent)]
    assert len(start_events) == 1
    assert start_events[0].workflow_type == "dag"
    assert set(start_events[0].metadata["nodes"]) == {
        "fetch_data",
        "fetch_ctx",
        "analyze",
        "summarize",
    }

    step_events = [e for e in emitter.events if isinstance(e, WorkflowStepCompleteEvent)]
    assert len(step_events) == 4

    print(f"  Output: {result.output}")
    print(f"  Steps executed: {result.metadata['total_steps_executed']}")
    print("✓ Diamond DAG executed with concurrent sources and single terminal output")

    # --- Section 2: Input Routing Rules ---
    print("\n--- Section 2: Input Routing Rules ---")

    # Three routing rules based on dependency count:
    #   - No dependencies → receives the original workflow input
    #   - One dependency  → receives that dependency's output directly
    #   - Multiple deps   → receives a dict mapping dep names to outputs
    #
    #   source ──→ single_dep ──┐
    #                           ├──→ multi_dep
    #   independent ────────────┘

    received_inputs: dict[str, object] = {}

    async def source(input: str) -> str:
        received_inputs["source"] = input
        return f"source({input})"

    async def independent(input: str) -> str:
        received_inputs["independent"] = input
        return f"independent({input})"

    async def single_dep(input: str) -> str:
        received_inputs["single_dep"] = input
        return f"single_dep({input})"

    async def multi_dep(inputs: dict[str, str]) -> str:
        received_inputs["multi_dep"] = inputs
        return f"multi_dep({inputs['single_dep']}, {inputs['independent']})"

    emitter = make_emitter("dag-s2")

    dag = DAG(
        name="input-routing",
        nodes={
            "source": DAGNode(step=FunctionStep("source", source)),
            "independent": DAGNode(step=FunctionStep("independent", independent)),
            "single_dep": DAGNode(
                step=FunctionStep("single_dep", single_dep),
                depends_on=["source"],
            ),
            "multi_dep": DAGNode(
                step=FunctionStep("multi_dep", multi_dep),
                depends_on=["single_dep", "independent"],
            ),
        },
        emitter=emitter,
    )

    result = await dag.execute("workflow input")

    # No dependencies → original workflow input
    assert received_inputs["source"] == "workflow input"
    assert received_inputs["independent"] == "workflow input"

    # One dependency → that dependency's output directly (string, not dict)
    assert received_inputs["single_dep"] == "source(workflow input)"
    assert isinstance(received_inputs["single_dep"], str)

    # Multiple dependencies → dict mapping dep names to outputs
    assert isinstance(received_inputs["multi_dep"], dict)
    assert set(received_inputs["multi_dep"].keys()) == {"single_dep", "independent"}

    # Single terminal node (multi_dep) → output is its value directly
    assert result.output == "multi_dep(single_dep(source(workflow input)), independent(workflow input))"

    print(f"  Source received: {received_inputs['source']!r}")
    print(f"  Single dep received: {received_inputs['single_dep']!r} (string, not dict)")
    print(f"  Multi dep received: dict with keys {set(received_inputs['multi_dep'].keys())}")
    print("✓ All three input routing rules demonstrated")

    # --- Section 3: Multiple Terminal Nodes ---
    print("\n--- Section 3: Multiple Terminal Nodes ---")

    # When multiple nodes have no dependents, the output is a dict
    # mapping terminal node names to their outputs.
    #
    #          ┌──→ branch_a
    #   source─┤
    #          └──→ branch_b

    async def source_node(input: str) -> str:
        return f"data({input})"

    async def branch_a(input: str) -> str:
        return f"result A from {input}"

    async def branch_b(input: str) -> str:
        return f"result B from {input}"

    emitter = make_emitter("dag-s3")

    dag = DAG(
        name="fan-out",
        nodes={
            "source": DAGNode(step=FunctionStep("source", source_node)),
            "branch_a": DAGNode(
                step=FunctionStep("branch_a", branch_a),
                depends_on=["source"],
            ),
            "branch_b": DAGNode(
                step=FunctionStep("branch_b", branch_b),
                depends_on=["source"],
            ),
        },
        emitter=emitter,
    )

    result = await dag.execute("topic")

    # Multiple terminal nodes → output is a dict
    assert isinstance(result.output, dict)
    assert result.output == {
        "branch_a": "result A from data(topic)",
        "branch_b": "result B from data(topic)",
    }

    print(f"  Output keys: {list(result.output.keys())}")
    print(f"  branch_a: {result.output['branch_a']}")
    print(f"  branch_b: {result.output['branch_b']}")
    print("✓ Multiple terminal nodes produce a dict of outputs")

    # --- Section 4: Validation ---
    print("\n--- Section 4: Validation ---")

    # DAGs validate at construction time. Invalid graphs never reach execution.

    # 4a: Cycle detection
    emitter = make_emitter("dag-s4")

    try:
        DAG(
            name="cyclic",
            nodes={
                "a": DAGNode(step=FunctionStep("a", fetch_data), depends_on=["b"]),
                "b": DAGNode(step=FunctionStep("b", fetch_data), depends_on=["a"]),
            },
            emitter=emitter,
        )
        assert False, "Should have raised ValueError for cycle"
    except ValueError as exc:
        assert "cycle" in str(exc).lower()
        print(f"  Cycle detected: {exc}")

    # 4b: Dangling dependency reference
    try:
        DAG(
            name="dangling",
            nodes={
                "a": DAGNode(step=FunctionStep("a", fetch_data), depends_on=["nonexistent"]),
            },
            emitter=emitter,
        )
        assert False, "Should have raised ValueError for dangling reference"
    except ValueError as exc:
        assert "nonexistent" in str(exc)
        print(f"  Dangling ref detected: {exc}")

    print("✓ Construction-time validation catches cycles and dangling references")

    # --- Section 5: BEST_EFFORT Failure Policy ---
    print("\n--- Section 5: BEST_EFFORT Failure Policy ---")

    # With BEST_EFFORT, a failed node's transitive dependents are skipped,
    # but independent branches complete normally.
    #
    #          ┌──→ fails ──→ downstream (skipped)
    #   source─┤
    #          └──→ succeeds (independent branch)

    async def source_data(input: str) -> str:
        return f"data({input})"

    async def fails(input: str) -> str:
        raise RuntimeError("processing failed")

    async def downstream(input: str) -> str:
        return f"downstream({input})"

    async def succeeds(input: str) -> str:
        return f"independent result from {input}"

    emitter = make_emitter("dag-s5")

    dag = DAG(
        name="best-effort",
        nodes={
            "source": DAGNode(step=FunctionStep("source", source_data)),
            "fails": DAGNode(
                step=FunctionStep("fails", fails),
                depends_on=["source"],
            ),
            "downstream": DAGNode(
                step=FunctionStep("downstream", downstream),
                depends_on=["fails"],
            ),
            "succeeds": DAGNode(
                step=FunctionStep("succeeds", succeeds),
                depends_on=["source"],
            ),
        },
        failure_policy=FailurePolicy.BEST_EFFORT,
        emitter=emitter,
    )

    result = await dag.execute("input")

    # Workflow completes — BEST_EFFORT absorbs the error
    # Only the surviving terminal node's output is returned
    assert result.output == "independent result from data(input)"

    # Failed node tracked with error details
    assert "fails" in result.metadata["failed_nodes"]
    assert result.metadata["failed_nodes"]["fails"]["error_type"] == "RuntimeError"
    assert result.metadata["failed_nodes"]["fails"]["error_message"] == "processing failed"

    # Transitive dependents of the failed node are skipped
    assert result.metadata["skipped_nodes"] == ["downstream"]

    # Only source + succeeds completed (fails threw, downstream skipped)
    assert result.metadata["total_steps_executed"] == 2

    print(f"  Output: {result.output}")
    print(f"  Failed: {list(result.metadata['failed_nodes'].keys())}")
    print(f"  Skipped: {result.metadata['skipped_nodes']}")
    print(f"  Steps executed: {result.metadata['total_steps_executed']}")
    print("✓ BEST_EFFORT skipped transitive dependents, independent branch completed")

    # --- Section 6: Max Concurrency ---
    print("\n--- Section 6: Max Concurrency ---")

    # max_concurrency limits how many nodes execute simultaneously.
    # Three independent sources feed into a collector.
    #
    #   node_a ──┐
    #   node_b ──┼──→ collector
    #   node_c ──┘

    execution_order: list[str] = []

    async def tracked_node(name: str, input: str) -> str:
        execution_order.append(name)
        return f"{name}({input})"

    async def collector(inputs: dict[str, str]) -> str:
        return f"collected: {', '.join(sorted(inputs.values()))}"

    emitter = make_emitter("dag-s6")

    dag = DAG(
        name="limited-concurrency",
        nodes={
            "node_a": DAGNode(step=FunctionStep("node_a", lambda x: tracked_node("a", x))),
            "node_b": DAGNode(step=FunctionStep("node_b", lambda x: tracked_node("b", x))),
            "node_c": DAGNode(step=FunctionStep("node_c", lambda x: tracked_node("c", x))),
            "collector": DAGNode(
                step=FunctionStep("collector", collector),
                depends_on=["node_a", "node_b", "node_c"],
            ),
        },
        max_concurrency=1,
        emitter=emitter,
    )

    result = await dag.execute("data")

    # All 4 nodes executed despite concurrency limit
    assert result.metadata["total_steps_executed"] == 4

    # Collector received all three source outputs
    assert "a(data)" in result.output
    assert "b(data)" in result.output
    assert "c(data)" in result.output

    # All three sources ran (order may vary, but all executed)
    assert set(execution_order) == {"a", "b", "c"}

    print(f"  Output: {result.output}")
    print(f"  Execution order: {execution_order}")
    print(f"  Steps executed: {result.metadata['total_steps_executed']}")
    print("✓ max_concurrency=1 accepted, all nodes executed correctly")

    print("\n✅ All sections passed!")


if __name__ == "__main__":
    asyncio.run(main())
