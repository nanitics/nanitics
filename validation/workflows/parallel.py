"""Parallel: concurrent fan-out with aggregation and declaration-order semantics.

Validates the :class:`Parallel` workflow running three steps concurrently
against the same input. Two of the steps are deterministic ``FunctionStep``
transforms (stable, timing-tolerant) and one is a real :class:`ReActAgent`
``AgentStep`` — the agent step proves integration with the real provider
inside the parallel scheduler. A custom ``aggregator`` collapses the three
``StepResult`` objects into a dict keyed by branch name; this pins the
distinguishing property of Parallel over other orchestrators: **every
declared branch ran (not just the one that happened to finish first) and
was delivered to the aggregator in declaration order**.

Acceptance criteria:
  - ``WorkflowStartEvent`` has ``workflow_type == "parallel"`` and
    ``step_count == 3``.
  - Exactly three ``WorkflowStepCompleteEvent`` events — one per branch,
    with ``step_index`` values ``{0, 1, 2}`` pinning declaration order.
  - An ``AgentStartEvent`` is observed for the real-agent branch — proves
    the agent's loop actually ran inside the parallel scheduler (not
    skipped or short-circuited).
  - Every branch's duration is shorter than the sum of all branch
    durations — proves they ran **concurrently**, not serially (a serial
    loop would make the last branch's duration ≈ sum of all).
  - ``result.metadata["total_steps_executed"] == 3``.
  - The aggregator output is a dict with keys for all three branches and
    each value is non-empty — proves the aggregator received every
    branch's result, not a subset.
  - The agent branch's aggregated value references the product topic —
    proves the input was delivered to that branch (not hallucinated).
"""

from __future__ import annotations

import asyncio

import pytest

from nanitics import (
    AgentStep,
    FunctionStep,
    InMemoryEmitter,
    Parallel,
    ReActAgent,
    StepResult,
)
from nanitics.infrastructure import (
    AgentStartEvent,
    WorkflowCompleteEvent,
    WorkflowStartEvent,
    WorkflowStepCompleteEvent,
)
from validation.helpers import (
    assert_result_satisfies,
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)

# Small deterministic delay so the concurrency proof has signal even when one
# branch (the LLM agent) dominates total duration. Serial execution would add
# these delays to the LLM time; parallel execution overlaps them under it.
_BRANCH_DELAY_S = 0.1


async def _uppercase(text: str) -> str:
    await asyncio.sleep(_BRANCH_DELAY_S)
    return text.upper()


async def _word_count(text: str) -> int:
    await asyncio.sleep(_BRANCH_DELAY_S)
    return len(text.split())


def _merge(results: list[StepResult]) -> dict[str, object]:
    # Declaration order: uppercase, word_count, analyst.
    return {
        "uppercase": results[0].output,
        "word_count": results[1].output,
        "analyst": results[2].output,
    }


@pytest.mark.quick
async def test_parallel_fans_out_to_three_branches(traced_emitter: InMemoryEmitter) -> None:
    analyst = ReActAgent(
        name="analyst",
        llm_client=make_llm_client("anthropic"),
        emitter=traced_emitter,
        system_prompt=(
            "You are a product analyst. Given a product description, produce a single-sentence "
            "one-liner that names the product or its category. Keep it under 20 words."
        ),
        tools=[],
        max_iterations=2,
    )

    workflow = Parallel(
        name="product-fanout",
        steps=[
            FunctionStep(name="uppercase", fn=_uppercase),
            FunctionStep(name="word_count", fn=_word_count),
            AgentStep(analyst),
        ],
        aggregator=_merge,
        emitter=traced_emitter,
    )

    input_text = "A managed database service for small e-commerce teams"
    result = await run_with_retry(
        lambda: workflow.execute(input_text),
        max_attempts=2,
    )

    start_event = assert_trace_contains(
        traced_emitter,
        WorkflowStartEvent,
        predicate=lambda e: e.workflow_type == "parallel" and e.step_count == 3,
    )
    assert start_event.workflow_name == "product-fanout"

    step_events = [e for e in traced_emitter.events if isinstance(e, WorkflowStepCompleteEvent)]
    assert len(step_events) == 3, f"Expected exactly 3 WorkflowStepCompleteEvent, got: {len(step_events)}"
    indices = {e.step_index for e in step_events}
    assert indices == {0, 1, 2}, f"Expected step_index set == {{0, 1, 2}}, got: {indices}"
    step_names = {e.step_name for e in step_events}
    assert step_names == {"uppercase", "word_count", "analyst"}, (
        f"Expected branch names {{uppercase, word_count, analyst}}, got: {step_names}"
    )

    assert_trace_contains(
        traced_emitter,
        AgentStartEvent,
        predicate=lambda e: e.agent_name == "analyst",
    )

    # --- Concurrency proof: actual wall-clock < sum of branch durations ---
    # Measure the workflow's true wall clock from the Start/Complete events that
    # bracket the run. Serial execution would make wall clock ≈ sum(durations);
    # concurrent execution collapses it toward max(durations). The strict
    # inequality below holds only when branches overlap in time, and the
    # 100 ms delay in each function branch gives the proof measurable signal
    # even when the LLM branch dominates.
    durations_ms = [e.step_duration_ms for e in step_events if e.step_duration_ms is not None]
    assert len(durations_ms) == 3, f"Expected duration on every step event; got: {durations_ms}"
    total_duration_ms = sum(durations_ms)
    complete_event = assert_trace_contains(
        traced_emitter,
        WorkflowCompleteEvent,
        predicate=lambda e: e.workflow_name == "product-fanout",
    )
    wall_clock_s = (complete_event.timestamp - start_event.timestamp).total_seconds()
    assert wall_clock_s < (total_duration_ms / 1000.0), (
        f"Expected concurrent execution: wall clock {wall_clock_s:.3f}s should be less than "
        f"sum of branch durations {total_duration_ms / 1000.0:.3f}s"
    )

    assert result.metadata["total_steps_executed"] == 3, (
        f"Expected 3 steps, got: {result.metadata['total_steps_executed']}"
    )

    assert isinstance(result.output, dict), f"Expected aggregator dict output, got: {type(result.output)}"
    assert set(result.output.keys()) == {"uppercase", "word_count", "analyst"}, (
        f"Expected aggregator keys for all three branches, got: {list(result.output)}"
    )
    assert result.output["uppercase"] == input_text.upper(), (
        f"uppercase branch produced unexpected output: {result.output['uppercase']!r}"
    )
    assert result.output["word_count"] == len(input_text.split()), (
        f"word_count branch produced unexpected output: {result.output['word_count']!r}"
    )
    analyst_output = str(result.output["analyst"] or "")
    assert analyst_output.strip(), f"Expected non-empty analyst output, got: {analyst_output!r}"

    await assert_result_satisfies(
        analyst_output,
        "The output is a short one-liner describing a database, data service, or e-commerce product.",
    )
