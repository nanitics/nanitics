"""Parallel: concurrent execution, aggregation, and failure policies.

Demonstrates ``Parallel`` — the orchestration workflow that fans out the same
input to all steps concurrently and collects results. Covers default list
aggregation, custom aggregators, ``AgentStep`` adapters, both failure policies,
and workflow event verification.

Related guide: docs/guides/orchestration.md
"""

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics.composition import (
    AgentStep,
    FailurePolicy,
    FunctionStep,
    Parallel,
    StepResult,
)
from nanitics.infrastructure import (
    MockLLMClient,
    WorkflowCompleteEvent,
    WorkflowStartEvent,
    WorkflowStepCompleteEvent,
)
from nanitics.strategies import ReActAgent


async def main() -> None:
    # --- Section 1: Basic Parallel with FunctionStep ---
    print("--- Section 1: Basic Parallel with FunctionStep ---")

    # Three async functions run concurrently with the same string input.
    # Default aggregation returns a list of outputs in declaration order.

    async def uppercase(text: str) -> str:
        return text.upper()

    async def word_count(text: str) -> int:
        return len(text.split())

    async def reversed_text(text: str) -> str:
        return text[::-1]

    emitter = make_emitter("parallel-s1")

    workflow = Parallel(
        name="text-transforms",
        steps=[
            FunctionStep(name="uppercase", fn=uppercase),
            FunctionStep(name="word_count", fn=word_count),
            FunctionStep(name="reversed", fn=reversed_text),
        ],
        emitter=emitter,
    )

    result = await workflow.execute("hello world")

    # Output is a list in declaration order — not completion order
    assert result.output == ["HELLO WORLD", 2, "dlrow olleh"]
    assert result.metadata["total_steps_executed"] == 3

    print(f"  Outputs: {result.output}")
    print("✓ All three steps ran concurrently with default list aggregation")

    # --- Section 2: Custom Aggregation ---
    print("\n--- Section 2: Custom Aggregation ---")

    # Same three steps, but an aggregator combines results into a dict.

    def merge_transforms(results: list[StepResult]) -> dict[str, object]:
        return {
            "uppercase": results[0].output,
            "word_count": results[1].output,
            "reversed": results[2].output,
        }

    emitter = make_emitter("parallel-s2")

    workflow = Parallel(
        name="text-transforms-merged",
        steps=[
            FunctionStep(name="uppercase", fn=uppercase),
            FunctionStep(name="word_count", fn=word_count),
            FunctionStep(name="reversed", fn=reversed_text),
        ],
        aggregator=merge_transforms,
        emitter=emitter,
    )

    result = await workflow.execute("hello world")

    # Output is now the dict returned by the aggregator
    assert isinstance(result.output, dict)
    assert result.output == {
        "uppercase": "HELLO WORLD",
        "word_count": 2,
        "reversed": "dlrow olleh",
    }

    print(f"  Merged output: {result.output}")
    print("✓ Custom aggregator combined step results into a dict")

    # --- Section 3: AgentStep — Agents in Parallel ---
    print("\n--- Section 3: AgentStep — Agents in Parallel ---")

    # Two ReActAgents run in parallel via AgentStep.
    # One analyzes pros, the other analyzes cons.

    emitter = make_emitter("parallel-s3")

    pros_agent = ReActAgent(
        name="pros_analyst",
        llm_client=MockLLMClient(responses=[make_response("Fast iteration, low cost")]),
        emitter=emitter,
        system_prompt="List the pros of the given topic.",
        tools=[],
    )

    cons_agent = ReActAgent(
        name="cons_analyst",
        llm_client=MockLLMClient(responses=[make_response("Limited scalability, vendor lock-in")]),
        emitter=emitter,
        system_prompt="List the cons of the given topic.",
        tools=[],
    )

    workflow = Parallel(
        name="pros-cons-analysis",
        steps=[
            AgentStep(pros_agent),
            AgentStep(cons_agent),
        ],
        emitter=emitter,
    )

    result = await workflow.execute("Using a managed database service")

    # Output is a list of the two agents' text outputs
    assert result.output == ["Fast iteration, low cost", "Limited scalability, vendor lock-in"]

    # AgentStep metadata includes agent-specific data
    print(f"  Pros: {result.output[0]}")
    print(f"  Cons: {result.output[1]}")
    print("✓ Two agents ran in parallel via AgentStep")

    # --- Section 4: Failure Policy — ALL_OR_NOTHING ---
    print("\n--- Section 4: Failure Policy — ALL_OR_NOTHING ---")

    # One step succeeds, one raises. Default ALL_OR_NOTHING propagates the error.

    async def succeed(text: str) -> str:
        return f"processed: {text}"

    async def fail(text: str) -> str:
        raise RuntimeError("analysis service unavailable")

    emitter = make_emitter("parallel-s4")

    workflow = Parallel(
        name="fragile-pipeline",
        steps=[
            FunctionStep(name="succeed", fn=succeed),
            FunctionStep(name="fail", fn=fail),
        ],
        emitter=emitter,
    )

    raised = False
    try:
        await workflow.execute("some input")
    except RuntimeError as exc:
        raised = True
        assert "analysis service unavailable" in str(exc)
        print(f"  Caught: {exc}")

    assert raised, "Expected RuntimeError from failing step"
    print("✓ ALL_OR_NOTHING propagated the step failure")

    # --- Section 5: Failure Policy — BEST_EFFORT ---
    print("\n--- Section 5: Failure Policy — BEST_EFFORT ---")

    # Same setup, but BEST_EFFORT returns partial results.

    emitter = make_emitter("parallel-s5")

    workflow = Parallel(
        name="resilient-pipeline",
        steps=[
            FunctionStep(name="succeed", fn=succeed),
            FunctionStep(name="fail", fn=fail),
        ],
        failure_policy=FailurePolicy.BEST_EFFORT,
        emitter=emitter,
    )

    result = await workflow.execute("some input")

    # Only the successful step's output is included
    assert result.output == ["processed: some input"]

    # Failed steps are tracked in metadata
    assert result.metadata["failed_steps"] == ["fail"]
    assert result.metadata["total_steps_executed"] == 1

    print(f"  Output: {result.output}")
    print(f"  Failed steps: {result.metadata['failed_steps']}")
    print("✓ BEST_EFFORT returned partial results with failure tracking")

    # --- Section 6: Workflow Events ---
    print("\n--- Section 6: Workflow Events ---")

    # Verify orchestration-specific events emitted during parallel execution.

    emitter = make_emitter("parallel-s6")

    async def double(n: int) -> int:
        return n * 2

    async def triple(n: int) -> int:
        return n * 3

    workflow = Parallel(
        name="number-transforms",
        steps=[
            FunctionStep(name="double", fn=double),
            FunctionStep(name="triple", fn=triple),
        ],
        emitter=emitter,
    )

    result = await workflow.execute(5)
    assert result.output == [10, 15]

    # WorkflowStartEvent — emitted once with workflow metadata
    start_events = [e for e in emitter.events if isinstance(e, WorkflowStartEvent)]
    assert len(start_events) == 1
    assert start_events[0].workflow_name == "number-transforms"
    assert start_events[0].workflow_type == "parallel"
    assert start_events[0].step_count == 2

    # WorkflowStepCompleteEvent — one per step
    step_events = [e for e in emitter.events if isinstance(e, WorkflowStepCompleteEvent)]
    assert len(step_events) == 2
    step_names = {e.step_name for e in step_events}
    assert step_names == {"double", "triple"}

    # WorkflowCompleteEvent — emitted once on successful completion
    complete_events = [e for e in emitter.events if isinstance(e, WorkflowCompleteEvent)]
    assert len(complete_events) == 1
    assert complete_events[0].workflow_name == "number-transforms"
    assert complete_events[0].workflow_type == "parallel"
    assert complete_events[0].total_steps_executed == 2

    print(f"  Start events: {len(start_events)}")
    print(f"  Step complete events: {len(step_events)}")
    print(f"  Complete events: {len(complete_events)}")
    print("✓ All workflow events emitted correctly")

    print("\n✅ All sections passed!")


if __name__ == "__main__":
    asyncio.run(main())
