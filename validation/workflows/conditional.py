"""Conditional: router picks exactly one branch; unfired branches leave no trace.

Validates the :class:`Conditional` workflow routing input to exactly one of
two named branches based on a sync router. Each branch is a real
:class:`ReActAgent` with a distinctive system prompt — a ``math_agent`` for
arithmetic questions and a ``history_agent`` for history questions. The
test is parametrized over both inputs so each branch is exercised in its
own run; the key distinguishing assertion is that the **unfired branch
emits no agent-start and no step-complete event** — a correctness failure
in which the router or scheduler accidentally ran both branches would
produce events for the dormant branch.

Acceptance criteria (parametrized, run once per branch):
  - ``WorkflowStartEvent`` has ``workflow_type == "conditional"`` and
    ``step_count == 2`` (number of declared branches).
  - Exactly one ``WorkflowStepCompleteEvent`` total — named after the
    expected branch, with ``step_index == 0``.
  - An ``AgentStartEvent`` for the expected branch's agent is observed.
  - Zero ``AgentStartEvent`` events for the other branch's agent — proves
    the unfired branch did not execute.
  - ``result.metadata["selected_branch"]`` equals the expected branch.
  - ``result.metadata["total_steps_executed"] == 1``.
  - The output content is topically aligned with the fired branch
    (arithmetic answer vs. historical answer).
"""

from __future__ import annotations

import pytest

from nanitics import (
    AgentStep,
    Conditional,
    InMemoryEmitter,
    ReActAgent,
)
from nanitics.infrastructure import (
    AgentStartEvent,
    WorkflowStartEvent,
    WorkflowStepCompleteEvent,
)
from validation.helpers import (
    assert_result_satisfies,
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)

MATH_INPUT = "What is 12 times 7?"
HISTORY_INPUT = "Who was the first president of the United States?"


def _route(text: str) -> str:
    lower = text.lower()
    if any(tok in lower for tok in ("president", "history", "war", "century")):
        return "history"
    return "math"


@pytest.mark.quick
@pytest.mark.parametrize(
    ("input_text", "expected_branch", "other_agent_name", "content_criterion"),
    [
        (
            MATH_INPUT,
            "math",
            "history_agent",
            "The output answers an arithmetic question, stating or computing a product that equals 84.",
        ),
        (
            HISTORY_INPUT,
            "history",
            "math_agent",
            "The output identifies George Washington as the first president of the United States.",
        ),
    ],
    ids=["math-branch", "history-branch"],
)
async def test_conditional_routes_to_single_branch(
    traced_emitter: InMemoryEmitter,
    input_text: str,
    expected_branch: str,
    other_agent_name: str,
    content_criterion: str,
) -> None:
    math_agent = ReActAgent(
        name="math_agent",
        llm_client=make_llm_client("anthropic"),
        emitter=traced_emitter,
        system_prompt="You are an arithmetic tutor. Answer the user's math question in one sentence.",
        tools=[],
        max_iterations=2,
    )
    history_agent = ReActAgent(
        name="history_agent",
        llm_client=make_llm_client("anthropic"),
        emitter=traced_emitter,
        system_prompt="You are a history tutor. Answer the user's history question in one sentence.",
        tools=[],
        max_iterations=2,
    )

    conditional = Conditional(
        name="subject-router",
        router=_route,
        branches={
            "math": AgentStep(math_agent),
            "history": AgentStep(history_agent),
        },
        emitter=traced_emitter,
    )

    result = await run_with_retry(
        lambda: conditional.execute(input_text),
        max_attempts=2,
    )

    start_event = assert_trace_contains(
        traced_emitter,
        WorkflowStartEvent,
        predicate=lambda e: e.workflow_type == "conditional" and e.step_count == 2,
    )
    assert start_event.workflow_name == "subject-router"

    step_events = [e for e in traced_emitter.events if isinstance(e, WorkflowStepCompleteEvent)]
    assert len(step_events) == 1, f"Expected exactly 1 WorkflowStepCompleteEvent, got: {len(step_events)}"
    assert step_events[0].step_name == f"{expected_branch}_agent", (
        f"Expected fired step_name == {expected_branch}_agent!r, got: {step_events[0].step_name!r}"
    )
    assert step_events[0].step_index == 0, (
        f"Expected step_index == 0 for fired branch, got: {step_events[0].step_index}"
    )

    assert_trace_contains(
        traced_emitter,
        AgentStartEvent,
        predicate=lambda e: e.agent_name == f"{expected_branch}_agent",
    )

    # --- Distinguishing assertion: no events from the unfired branch ---
    other_starts = [
        e for e in traced_emitter.events if isinstance(e, AgentStartEvent) and e.agent_name == other_agent_name
    ]
    assert other_starts == [], (
        f"Expected zero AgentStartEvent for unfired branch {other_agent_name!r}; got: {len(other_starts)}"
    )
    other_step_events = [
        e for e in traced_emitter.events if isinstance(e, WorkflowStepCompleteEvent) and e.step_name == other_agent_name
    ]
    assert other_step_events == [], (
        f"Expected zero WorkflowStepCompleteEvent for unfired branch {other_agent_name!r}; "
        f"got: {len(other_step_events)}"
    )

    assert result.metadata["selected_branch"] == expected_branch, (
        f"Expected selected_branch == {expected_branch!r}, got: {result.metadata['selected_branch']!r}"
    )
    assert result.metadata["total_steps_executed"] == 1, (
        f"Expected total_steps_executed == 1, got: {result.metadata['total_steps_executed']}"
    )

    await assert_result_satisfies(str(result.output or ""), content_criterion)
