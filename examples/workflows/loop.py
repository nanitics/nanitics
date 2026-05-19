"""Loop workflow: iterative execution with termination conditions.

Demonstrates the Loop orchestration pattern — repeatedly executing a step until a
condition is met or max_iterations is reached. Between iterations, the step's output
becomes the next iteration's input, enabling iterative refinement. The condition
callback receives (StepResult, iteration) and returns True to stop.

Related guide: docs/guides/orchestration.md
"""

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics.composition import (
    AgentStep,
    FunctionStep,
    StepResult,
)
from nanitics.infrastructure import (
    MockLLMClient,
    WorkflowCompleteEvent,
    WorkflowStartEvent,
    WorkflowStepCompleteEvent,
)
from nanitics.specialized import Loop
from nanitics.strategies import ReActAgent


async def main() -> None:
    # --- Section 1: Basic Loop with FunctionStep ---
    print("--- Section 1: Basic Loop with FunctionStep ---")

    # A FunctionStep that appends a suffix each iteration. The condition stops
    # after 3 iterations. This shows the core mechanics: condition signature,
    # iteration counting, and input→output chaining.

    async def revise(text: str) -> str:
        return f"{text} [revised]"

    def stop_after_three(result: StepResult, iteration: int) -> bool:
        """Return True to stop the loop."""
        return iteration >= 3

    emitter = make_emitter("loop-s1")

    workflow = Loop(
        name="revision-loop",
        step=FunctionStep("revise", revise),
        condition=stop_after_three,
        max_iterations=10,
        emitter=emitter,
    )

    result = await workflow.execute("draft")

    # Output reflects 3 iterations of transformation
    assert result.output == "draft [revised] [revised] [revised]", f"Got: {result.output}"

    # Metadata tracks iteration count
    assert result.metadata["iterations"] == 3

    # Condition stopped the loop, not the iteration limit
    assert "terminated" not in result.metadata or result.metadata.get("terminated") != "iteration_limit"

    print("  Input:  'draft'")
    print(f"  Output: '{result.output}'")
    print(f"  Iterations: {result.metadata['iterations']}")
    print("✓ Condition callback controls loop termination — output chains between iterations")

    # --- Section 2: Iteration Limit ---
    print("\n--- Section 2: Iteration Limit ---")

    # When the condition never returns True, max_iterations acts as a safety net.
    # The loop returns the best result so far with a metadata flag — no exception.

    async def increment(value: str) -> str:
        count = int(value) + 1
        return str(count)

    def never_stop(result: StepResult, iteration: int) -> bool:
        """Never satisfied — the iteration limit will stop the loop."""
        return False

    emitter = make_emitter("loop-s2")

    workflow = Loop(
        name="counter-loop",
        step=FunctionStep("increment", increment),
        condition=never_stop,
        max_iterations=3,
        emitter=emitter,
    )

    result = await workflow.execute("0")

    # 3 iterations: "0" → "1" → "2" → "3"
    assert result.output == "3", f"Got: {result.output}"

    # Metadata shows the limit was hit
    assert result.metadata["iterations"] == 3
    assert result.metadata["terminated"] == "iteration_limit"

    print("  Input:  '0'")
    print(f"  Output: '{result.output}'")
    print(f"  Iterations: {result.metadata['iterations']}")
    print(f"  Terminated: {result.metadata['terminated']}")
    print("✓ max_iterations is a safety net — returns last result with metadata flag, no exception")

    # --- Section 3: AgentStep Iterative Refinement ---
    print("\n--- Section 3: AgentStep Iterative Refinement ---")

    # The canonical Loop use case: an agent refines its output until a quality
    # check passes. Each iteration is a full agent run. The condition checks
    # for a quality marker in the output.

    emitter = make_emitter("loop-s3")

    client = MockLLMClient(
        responses=[
            # Iteration 1: agent receives original task, produces a draft
            make_response("Draft: This product helps users manage tasks efficiently."),
            # Iteration 2: agent receives the draft as input, refines it
            make_response(
                "APPROVED: This product streamlines task management "
                "with intuitive prioritization and seamless collaboration."
            ),
        ]
    )

    agent = ReActAgent(
        name="writer",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a copywriter. Refine the given text until it meets quality standards.",
        tools=[],
    )

    def quality_check(result: StepResult, iteration: int) -> bool:
        """Stop when the output contains the APPROVED marker."""
        return "APPROVED" in str(result.output)

    workflow = Loop(
        name="writing-loop",
        step=AgentStep(agent),
        condition=quality_check,
        max_iterations=5,
        emitter=emitter,
    )

    result = await workflow.execute("Write a product description for a task management app")

    # Converged on second iteration
    assert result.metadata["iterations"] == 2
    assert "APPROVED" in str(result.output)

    print(f"  Iterations: {result.metadata['iterations']}")
    print(f"  Output: '{result.output}'")
    print("✓ AgentStep + Loop enables iterative refinement — condition drives convergence")

    # --- Section 4: Async Termination Condition ---
    print("\n--- Section 4: Async Termination Condition ---")

    # The condition callable can be async, enabling patterns like database lookups
    # or external service checks between iterations.

    async def count_up(value: str) -> str:
        return str(int(value) + 1)

    async def async_threshold_check(result: StepResult, iteration: int) -> bool:
        """Async condition — e.g., could await a database or API call."""
        # Simulate async evaluation
        await asyncio.sleep(0)
        return int(str(result.output)) >= 3

    emitter = make_emitter("loop-s4")

    workflow = Loop(
        name="async-condition-loop",
        step=FunctionStep("count-up", count_up),
        condition=async_threshold_check,
        max_iterations=10,
        emitter=emitter,
    )

    result = await workflow.execute("0")

    # Stops when output reaches 3: "0" → "1" → "2" → "3" (3 iterations)
    assert result.output == "3", f"Got: {result.output}"
    assert result.metadata["iterations"] == 3

    print(f"  Output: '{result.output}'")
    print(f"  Iterations: {result.metadata['iterations']}")
    print("✓ Async conditions work identically — enables evaluation-based termination")

    # --- Section 5: Workflow Events ---
    print("\n--- Section 5: Workflow Events ---")

    # Loop emits standard workflow events: one WorkflowStartEvent, one
    # WorkflowStepCompleteEvent per iteration, and one WorkflowCompleteEvent.

    async def append_dot(text: str) -> str:
        return f"{text}."

    def stop_after_three_dots(result: StepResult, iteration: int) -> bool:
        return iteration >= 3

    emitter = make_emitter("loop-s5")

    workflow = Loop(
        name="dot-loop",
        step=FunctionStep("append-dot", append_dot),
        condition=stop_after_three_dots,
        max_iterations=10,
        emitter=emitter,
    )

    result = await workflow.execute("start")
    assert result.output == "start..."

    # WorkflowStartEvent — emitted once with workflow metadata
    start_events = [e for e in emitter.events if isinstance(e, WorkflowStartEvent)]
    assert len(start_events) == 1
    assert start_events[0].workflow_name == "dot-loop"
    assert start_events[0].workflow_type == "loop"
    assert start_events[0].step_count == 1

    # WorkflowStepCompleteEvent — one per iteration, 0-indexed step_index
    step_events = [e for e in emitter.events if isinstance(e, WorkflowStepCompleteEvent)]
    assert len(step_events) == 3, f"Expected 3 step events, got {len(step_events)}"
    assert [e.step_index for e in step_events] == [0, 1, 2]
    assert all(e.step_name == "append-dot" for e in step_events)

    # WorkflowCompleteEvent — emitted once on successful completion
    complete_events = [e for e in emitter.events if isinstance(e, WorkflowCompleteEvent)]
    assert len(complete_events) == 1
    assert complete_events[0].workflow_name == "dot-loop"
    assert complete_events[0].workflow_type == "loop"
    assert complete_events[0].total_steps_executed == 3

    print(f"  Start events: {len(start_events)} (workflow_type='{start_events[0].workflow_type}')")
    print(f"  Step complete events: {len(step_events)} (indices: {[e.step_index for e in step_events]})")
    print(f"  Complete events: {len(complete_events)} (total_steps={complete_events[0].total_steps_executed})")
    print("✓ Loop emits standard workflow events — one step-complete per iteration")

    print("\n✅ All sections passed!")


if __name__ == "__main__":
    asyncio.run(main())
