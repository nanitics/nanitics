"""Safety limits fire under designed-to-loop / over-budget scenarios.

Two scenarios, one file:

1. ``SafetyIterationLimitEvent`` — a real ``ReActAgent`` is given a tool
   whose contract instructs the LLM to keep calling it: every return
   value tells the agent the result is "not specific enough" and asks
   it to refine. With ``max_iterations=2`` the agent is expected to
   terminate at the iteration limit and emit
   ``SafetyIterationLimitEvent``. The tool is the source of the loop
   pressure (rather than self-assessment by the LLM) for reproducibility.

2. ``SafetyToolCallLimitEvent`` — a real ``ReActAgent`` is given a tool
   whose contract instructs the LLM to call it once per target, plus an
   explicit instruction in the task to call the tool for three distinct
   targets. With ``max_tool_calls=2`` the tool-call limiter is expected
   to trigger after the second (or batched >=3rd) call and emit
   ``SafetyToolCallLimitEvent`` with ``termination_reason ==
   "tool_call_limit"``.

Acceptance criteria (iteration-limit scenario):
  - ``result.termination_reason == "iteration_limit"``.
  - ``result.total_steps == 2``.
  - ``result.output is None`` (iteration-limited agents do not produce
    a final answer).
  - Trace contains a ``SafetyIterationLimitEvent`` with
    ``agent_name == "loop-prone"``, ``max_iterations == 2``,
    ``current_iteration == 3`` (the value that triggered the raise),
    and ``step_number == 2`` (emitted before ``step_number += 1`` runs
    for the would-be third iteration).
  - Trace contains at least two ``ToolInvokeEvent`` instances for
    ``refine_search`` — proves the loop was driven by actual tool
    invocations, not a generic two-step pattern.

Acceptance criteria (tool-call-limit scenario):
  - ``result.termination_reason == "tool_call_limit"``.
  - Trace contains at least two ``ToolInvokeEvent`` for ``lookup`` —
    proves the loop ran tool calls up to (and past) the limit.
  - Trace contains a ``SafetyToolCallLimitEvent`` with
    ``agent_name == "over-budget"``, ``max_tool_calls == 2``, and
    ``current_tool_calls > max_tool_calls`` (the value that triggered
    the raise).

If a future LLM is smart enough to break out of the loop and produce a
final answer before iteration 2, or refuses to make enough tool calls,
the script fails loudly — diagnosis means tightening the prompt's loop
pressure or accepting that the scenario needs a different design.
"""

from __future__ import annotations

import pytest

from nanitics import InMemoryEmitter, ReActAgent, tool
from nanitics.infrastructure import (
    SafetyIterationLimitEvent,
    SafetyToolCallLimitEvent,
    ToolInvokeEvent,
)
from validation.helpers import (
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)


@tool(
    "refine_search",
    "Refine and re-run a search with a more specific query. Returns search results.",
)
async def refine_search(query: str) -> str:
    return (
        f"Search result for '{query}' is not specific enough. "
        "Refine your query and call refine_search again with more detail."
    )


@pytest.mark.quick
async def test_iteration_limit_fires(traced_emitter: InMemoryEmitter) -> None:
    client = make_llm_client("anthropic")

    agent = ReActAgent(
        name="loop-prone",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You are a precision-obsessed research assistant. Your answers must "
            "be exact and fully specific. If a search result is not specific "
            "enough, refine it and search again. Do not produce a final answer "
            "until the search returns a fully specific result."
        ),
        tools=[refine_search],
        max_iterations=2,
    )

    result = await run_with_retry(
        lambda: agent.run(
            "Use refine_search to find the exact date of a specific historical event of your choice. "
            "Keep refining until the result is specific."
        ),
        max_attempts=2,
    )

    # --- Result-shape invariants ---
    assert result.termination_reason == "iteration_limit", (
        f"Expected termination_reason='iteration_limit', got: {result.termination_reason!r}"
    )
    assert result.total_steps == 2, f"Expected total_steps=2, got: {result.total_steps}"
    assert result.output is None, f"Expected result.output is None at iteration limit, got: {result.output!r}"

    # --- Trace-shape invariants ---
    # The predicate pins the off-by-one contract of IterationLimiter.step():
    # the raise happens when current_iteration (3) > max_iterations (2), and
    # the emit runs before step_number += 1 for the would-be third iteration,
    # so step_number == 2 at emission.
    assert_trace_contains(
        traced_emitter,
        SafetyIterationLimitEvent,
        predicate=lambda e: (
            e.agent_name == "loop-prone" and e.max_iterations == 2 and e.current_iteration == 3 and e.step_number == 2
        ),
    )

    # The distinguishing narrative is "tool-driven loop". Positively assert
    # that the two executed steps each invoked refine_search — without this,
    # the script would pass if the agent happened to loop for any other
    # reason (e.g. two consecutive non-tool responses).
    tool_invocations = [
        e for e in traced_emitter.events if isinstance(e, ToolInvokeEvent) and e.tool_name == "refine_search"
    ]
    assert len(tool_invocations) >= 2, (
        f"Expected at least 2 ToolInvokeEvent for 'refine_search' (loop pressure from the tool "
        f"contract), got: {len(tool_invocations)}"
    )


# ---------------------------------------------------------------------------
# ToolCallLimiter scenario
# ---------------------------------------------------------------------------


@tool(
    "lookup",
    "Look up a single target by name. Call once per target; returns a short fact.",
)
async def lookup(target: str) -> str:
    # Return something specific but not self-terminating — the LLM must keep
    # calling for each named target. The fact is intentionally terse so the
    # model has no reason to abandon the plan after one call.
    return f"Fact about {target}: acknowledged."


@pytest.mark.quick
async def test_tool_call_limiter_stops_at_max(traced_emitter: InMemoryEmitter) -> None:
    client = make_llm_client("anthropic")

    agent = ReActAgent(
        name="over-budget",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You are a precise research assistant. For every target the user "
            "names, call the `lookup` tool exactly once with that target as "
            "the argument. Do not skip any target. After gathering all facts, "
            "produce a short summary."
        ),
        tools=[lookup],
        max_iterations=6,
        max_tool_calls=2,
    )

    result = await run_with_retry(
        lambda: agent.run(
            "Please look up these three distinct targets one at a time, using "
            "the `lookup` tool once per target: 'alpha', 'beta', and 'gamma'. "
            "Then summarize the facts."
        ),
        max_attempts=2,
    )

    # --- Result-shape invariants ---
    assert result.termination_reason == "tool_call_limit", (
        f"Expected termination_reason='tool_call_limit', got: {result.termination_reason!r}"
    )

    # --- Trace-shape invariants ---
    # Positive narrative: at least two lookup invocations actually executed.
    # Without this, the script would pass if the model never called the tool
    # and we merely happened to terminate for some other reason.
    tool_invocations = [e for e in traced_emitter.events if isinstance(e, ToolInvokeEvent) and e.tool_name == "lookup"]
    assert len(tool_invocations) >= 2, (
        f"Expected at least 2 ToolInvokeEvent for 'lookup' (the limit is 2; "
        f"at least 2 must execute before the limiter raises), got: {len(tool_invocations)}"
    )

    # Pin the limiter event's payload. The raise happens when
    # current_tool_calls > max_tool_calls, so current_tool_calls must exceed
    # max_tool_calls at emission time.
    assert_trace_contains(
        traced_emitter,
        SafetyToolCallLimitEvent,
        predicate=lambda e: (
            e.agent_name == "over-budget" and e.max_tool_calls == 2 and e.current_tool_calls > e.max_tool_calls
        ),
    )
