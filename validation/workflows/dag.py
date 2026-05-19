"""Diamond DAG: two concurrent sources → analyzer → summarizer, all real agents.

Validates the :class:`DAG` workflow scheduling concurrent sources and a
serial analyzer/summarizer fan-in. The assertions pin the two properties
that separate a real DAG scheduler from a naive serial loop:
**topological ordering** (analyzer runs after both sources; summarizer
runs after analyzer) and **structured fan-in input wiring** (analyzer's
metadata captures both source outputs).

Acceptance criteria:
  - ``WorkflowStartEvent(workflow_type="dag")`` emitted with
    ``metadata["edges"]`` matching the declared dependency graph and
    ``metadata["nodes"]`` listing all four nodes.
  - One ``WorkflowStepCompleteEvent`` per node, exactly four total.
  - Topological timestamp ordering: both source events precede the
    analyzer event, and the analyzer event precedes the summarizer.
  - ``result.metadata["total_steps_executed"] == 4``.
  - ``result.metadata["node_results"]`` contains all four node names,
    each mapped to a non-empty output string.
  - The analyzer's recorded output references both the product and the
    customer context — proves fan-in wiring delivered both sources'
    outputs to the analyzer node (not hallucinated from the input prompt).
  - Final output mentions both the product and the customer context.
"""

from __future__ import annotations

from nanitics.composition import (
    DAG,
    AgentStep,
    DAGNode,
)
from nanitics.infrastructure import WorkflowStartEvent, WorkflowStepCompleteEvent
from nanitics.strategies import ReActAgent
from nanitics.tracing import InMemoryEmitter
from validation.helpers import (
    assert_result_satisfies,
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)


async def test_dag_diamond_briefing(traced_emitter: InMemoryEmitter) -> None:
    product_agent = ReActAgent(
        name="product_details",
        llm_client=make_llm_client("anthropic"),
        emitter=traced_emitter,
        system_prompt=(
            "Describe product X in 1-2 sentences. "
            "Do not ask for clarification or note missing details. "
            "Invent specific plausible particulars (name, scale, features) "
            "and state them as if known."
        ),
        tools=[],
        max_iterations=2,
    )
    customer_agent = ReActAgent(
        name="customer_context",
        llm_client=make_llm_client("anthropic"),
        emitter=traced_emitter,
        system_prompt=(
            "Describe customer Y in 1-2 sentences. "
            "Do not ask for clarification or note missing details. "
            "Invent specific plausible particulars (name, scale, features) "
            "and state them as if known."
        ),
        tools=[],
        max_iterations=2,
    )
    analyzer_agent = ReActAgent(
        name="analyzer",
        llm_client=make_llm_client("anthropic"),
        emitter=traced_emitter,
        system_prompt=(
            "Combine the product details and customer context provided into a concise analysis of fit (1-2 sentences). "
            "Reference specifics from both inputs."
        ),
        tools=[],
        max_iterations=2,
    )
    summarizer_agent = ReActAgent(
        name="summarizer",
        llm_client=make_llm_client("anthropic"),
        emitter=traced_emitter,
        system_prompt=(
            "Produce a brief executive briefing (2-3 sentences) based on the analysis provided. "
            "Mention both the product and the customer context."
        ),
        tools=[],
        max_iterations=2,
    )

    dag = DAG(
        name="briefing-diamond",
        nodes={
            "product_details": DAGNode(step=AgentStep(product_agent)),
            "customer_context": DAGNode(step=AgentStep(customer_agent)),
            "analyzer": DAGNode(
                step=AgentStep(analyzer_agent),
                depends_on=["product_details", "customer_context"],
            ),
            "summarizer": DAGNode(step=AgentStep(summarizer_agent), depends_on=["analyzer"]),
        },
        emitter=traced_emitter,
    )

    result = await run_with_retry(
        lambda: dag.execute("Prepare a briefing on product X for customer Y."),
        max_attempts=2,
    )

    # --- Start event: workflow_type and structural edges metadata ---
    expected_edges = {
        ("product_details", "analyzer"),
        ("customer_context", "analyzer"),
        ("analyzer", "summarizer"),
    }
    start_event = assert_trace_contains(
        traced_emitter,
        WorkflowStartEvent,
        predicate=lambda e: (
            e.workflow_type == "dag"
            and {tuple(edge) for edge in e.metadata.get("edges", [])} == expected_edges
            and set(e.metadata.get("nodes", [])) == {"product_details", "customer_context", "analyzer", "summarizer"}
        ),
    )
    assert start_event.step_count == 4, f"Expected WorkflowStartEvent.step_count == 4, got: {start_event.step_count}"

    # --- Exactly four step-complete events, one per node ---
    step_events = [e for e in traced_emitter.events if isinstance(e, WorkflowStepCompleteEvent)]
    assert len(step_events) == 4, f"Expected exactly 4 WorkflowStepCompleteEvent, got: {len(step_events)}"
    step_by_name = {e.step_name: e for e in step_events}
    assert set(step_by_name.keys()) == {
        "product_details",
        "customer_context",
        "analyzer",
        "summarizer",
    }, f"Expected one step-complete per node, got: {list(step_by_name)}"

    # --- Topological ordering via event timestamps ---
    product_ts = step_by_name["product_details"].timestamp
    customer_ts = step_by_name["customer_context"].timestamp
    analyzer_ts = step_by_name["analyzer"].timestamp
    summarizer_ts = step_by_name["summarizer"].timestamp
    assert product_ts < analyzer_ts, (
        f"Expected product_details to complete before analyzer: {product_ts} vs {analyzer_ts}"
    )
    assert customer_ts < analyzer_ts, (
        f"Expected customer_context to complete before analyzer: {customer_ts} vs {analyzer_ts}"
    )
    assert analyzer_ts < summarizer_ts, (
        f"Expected analyzer to complete before summarizer: {analyzer_ts} vs {summarizer_ts}"
    )

    # --- Result metadata: step count and node_results wiring ---
    assert result.metadata["total_steps_executed"] == 4, (
        f"Expected 4 steps, got: {result.metadata['total_steps_executed']}"
    )
    node_results = result.metadata["node_results"]
    assert set(node_results.keys()) == {
        "product_details",
        "customer_context",
        "analyzer",
        "summarizer",
    }, f"Expected node_results for all four nodes, got: {list(node_results)}"
    for node_name, output in node_results.items():
        assert output is not None, f"Expected non-None output for node {node_name!r}, got: {output!r}"
        assert str(output).strip(), f"Expected non-empty output for node {node_name!r}, got: {output!r}"

    # --- Fan-in proof: analyzer output depends on BOTH source outputs ---
    # A broken fan-in mapping (e.g. only one source reached analyzer) would
    # produce an analyzer output missing specifics from the dropped source.
    analyzer_output = str(node_results["analyzer"])
    product_output = str(node_results["product_details"])
    customer_output = str(node_results["customer_context"])
    await assert_result_satisfies(
        (
            "Product details (source A):\n"
            f"{product_output}\n\n"
            "Customer context (source B):\n"
            f"{customer_output}\n\n"
            "Analyzer output:\n"
            f"{analyzer_output}"
        ),
        (
            "The analyzer output references specific details from BOTH the product details "
            "and the customer context sources — not merely the top-level prompt. If the "
            "analyzer output only reflects one source or only generic prompt content, fail."
        ),
    )

    # --- Final output mentions both entities ---
    await assert_result_satisfies(
        str(result.output or ""),
        "The output is a briefing that mentions both the product and the customer context.",
    )
