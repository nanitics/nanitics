"""Loop: iterative refinement with a real agent, pinning iteration chaining.

Validates the :class:`Loop` workflow running a real :class:`ReActAgent` that
appends a distinctive tick token to its input each iteration. The system
prompt instructs the agent to echo the input and append a new tick of the
form ``[TICK-<n>]`` where ``<n>`` is one more than the highest tick already
present. This pins the primitive's defining property: **iteration N's
output becomes iteration N+1's input** (a broken loop that re-fed the
original input every time would never accumulate more than one tick).

A second test drives the iteration-limit branch with a deterministic
``FunctionStep`` counter and a condition that never fires, proving the
``terminated == "iteration_limit"`` metadata path.

Acceptance criteria (refinement test):
  - ``WorkflowStartEvent`` has ``workflow_type == "loop"`` and
    ``step_count == 1``.
  - One ``AgentStartEvent`` per iteration — proves the agent loop actually
    ran each round (not cached from the first run).
  - Exactly three ``WorkflowStepCompleteEvent`` events with
    ``step_index`` values ``{0, 1, 2}`` — pins iteration count.
  - ``result.metadata["iterations"] == 3`` and
    ``result.metadata["total_steps_executed"] == 3``.
  - The condition fired on iteration 3, so ``"terminated"`` is absent or
    not ``"iteration_limit"``.
  - Final output contains ``[TICK-1]``, ``[TICK-2]``, and ``[TICK-3]`` —
    proves each iteration's output was fed into the next iteration's
    input. A broken loop re-feeding the original input each time would
    produce at most one tick.

Acceptance criteria (iteration-limit test):
  - ``result.metadata["iterations"] == 3``.
  - ``result.metadata["terminated"] == "iteration_limit"``.
  - Final output is ``"3"`` — proves the counter was chained through 3
    iterations (``"0" → "1" → "2" → "3"``).
"""

from __future__ import annotations

import re

import pytest

from nanitics.composition import (
    AgentStep,
    FunctionStep,
    StepResult,
)
from nanitics.infrastructure import (
    AgentStartEvent,
    WorkflowStartEvent,
    WorkflowStepCompleteEvent,
)
from nanitics.specialized import Loop
from nanitics.strategies import ReActAgent
from nanitics.tracing import InMemoryEmitter
from validation.helpers import (
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)

TICK_PATTERN = re.compile(r"\[TICK-(\d+)\]")


def _stop_after_three(_result: StepResult, iteration: int) -> bool:
    return iteration >= 3


def _never_stop(_result: StepResult, _iteration: int) -> bool:
    return False


@pytest.mark.quick
async def test_loop_accumulates_ticks_across_iterations(traced_emitter: InMemoryEmitter) -> None:
    # The ticker agent's contract — each iteration observes the previous
    # output and appends exactly one new tick. This is load-bearing: the
    # assertion below relies on the tick count strictly increasing.
    ticker = ReActAgent(
        name="ticker",
        llm_client=make_llm_client("anthropic"),
        emitter=traced_emitter,
        system_prompt=(
            "You are a token accumulator. The user's message contains zero or more tags of the form "
            "[TICK-N] where N is a positive integer. Your response MUST:\n"
            "1. Preserve every existing [TICK-N] tag verbatim.\n"
            "2. Append exactly ONE new tag whose number is one greater than the highest existing N "
            "(use [TICK-1] if none exists).\n"
            "Respond with ONLY the updated text — no commentary, no explanation."
        ),
        tools=[],
        max_iterations=2,
    )

    workflow = Loop(
        name="tick-loop",
        step=AgentStep(ticker),
        condition=_stop_after_three,
        max_iterations=5,
        emitter=traced_emitter,
    )

    result = await run_with_retry(
        lambda: workflow.execute("seed"),
        max_attempts=2,
    )

    start_event = assert_trace_contains(
        traced_emitter,
        WorkflowStartEvent,
        predicate=lambda e: e.workflow_type == "loop" and e.step_count == 1,
    )
    assert start_event.workflow_name == "tick-loop"

    agent_start_events = [
        e for e in traced_emitter.events if isinstance(e, AgentStartEvent) and e.agent_name == "ticker"
    ]
    assert len(agent_start_events) == 3, (
        f"Expected 3 AgentStartEvent for ticker (one per iteration), got: {len(agent_start_events)}"
    )

    step_events = [e for e in traced_emitter.events if isinstance(e, WorkflowStepCompleteEvent)]
    assert len(step_events) == 3, f"Expected exactly 3 WorkflowStepCompleteEvent, got: {len(step_events)}"
    indices = [e.step_index for e in step_events]
    assert indices == [0, 1, 2], f"Expected step_index sequence [0, 1, 2], got: {indices}"

    assert result.metadata["iterations"] == 3, f"Expected 3 iterations, got: {result.metadata['iterations']}"
    assert result.metadata["total_steps_executed"] == 3, (
        f"Expected total_steps_executed == 3, got: {result.metadata['total_steps_executed']}"
    )
    assert result.metadata.get("terminated") != "iteration_limit", (
        f"Expected condition-based termination, got terminated={result.metadata.get('terminated')!r}"
    )

    # --- Chaining proof: every tick from 1..3 is present in the final output ---
    final_output = str(result.output or "")
    ticks_found = {int(m.group(1)) for m in TICK_PATTERN.finditer(final_output)}
    assert {1, 2, 3}.issubset(ticks_found), (
        f"Expected ticks {{1, 2, 3}} in final output (proves iteration N output fed iteration N+1 input); "
        f"found ticks {sorted(ticks_found)} in output: {final_output!r}"
    )


@pytest.mark.quick
async def test_loop_iteration_limit_branch(traced_emitter: InMemoryEmitter) -> None:
    async def increment(value: str) -> str:
        return str(int(value) + 1)

    workflow = Loop(
        name="counter-loop",
        step=FunctionStep(name="increment", fn=increment),
        condition=_never_stop,
        max_iterations=3,
        emitter=traced_emitter,
    )

    result = await run_with_retry(
        lambda: workflow.execute("0"),
        max_attempts=2,
    )

    step_events = [e for e in traced_emitter.events if isinstance(e, WorkflowStepCompleteEvent)]
    assert len(step_events) == 3, f"Expected 3 WorkflowStepCompleteEvent, got: {len(step_events)}"
    assert [e.step_index for e in step_events] == [0, 1, 2], (
        f"Expected step_index sequence [0, 1, 2], got: {[e.step_index for e in step_events]}"
    )

    assert result.metadata["iterations"] == 3, f"Expected iterations == 3, got: {result.metadata['iterations']}"
    assert result.metadata["terminated"] == "iteration_limit", (
        f"Expected terminated == 'iteration_limit', got: {result.metadata.get('terminated')!r}"
    )
    assert result.output == "3", f"Expected '0' chained through 3 increments to yield '3', got: {result.output!r}"
