"""Multi-tool ReAct validation: sequential tool composition with interleaved reasoning.

The task is structured so the second tool call depends on the first tool's
observation — looking up City A's population, then deciding which variant of
City B to consult — so a degenerate agent that batches both tool calls in a
single turn cannot satisfy the problem.

Acceptance criteria:
  - At least two ``AgentStepEvent`` events carry a non-empty ``action`` field
    (proves two interleaved tool-emitting turns, not one batched turn).
  - At least one ``AgentStepEvent`` carries non-empty ``thought`` and
    ``observation`` fields (proves the reasoning-and-action interleaving that
    distinguishes ReAct).
  - Both provided tools are invoked at least once (``ToolInvokeEvent``).
  - Both tool invocations produced a successful ``ToolResultEvent``.
  - ``result.termination_reason == "complete"`` (distinguishes a clean finish
    from an iteration-limit exit).
  - ``2 <= result.total_steps <= 4`` (tight upper bound — two tool turns plus
    a final answer turn).
  - Final answer references facts from both tools.
"""

from __future__ import annotations

import pytest

from nanitics import InMemoryEmitter, ReActAgent, tool
from nanitics.infrastructure import AgentStepEvent, ToolInvokeEvent, ToolResultEvent
from validation.helpers import (
    assert_result_satisfies,
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)


@tool("lookup_population_a", "Look up the population of city A.")
async def lookup_population_a(note: str = "") -> str:
    del note  # unused — tool is a deterministic stub
    return "City A: 850000"


@tool(
    "lookup_population_b",
    "Look up the population of city B. Requires ``reference_population`` "
    "(an integer from a prior lookup) so the tool knows which city-B variant "
    "to report.",
)
async def lookup_population_b(reference_population: int) -> str:
    # The tool deliberately requires the agent to supply a value derived from
    # the first tool's observation, so a batched single-turn solution cannot
    # succeed.
    if reference_population >= 800000:
        return "City B (large-reference variant): 620000"
    return "City B (small-reference variant): 120000"


@pytest.mark.quick
async def test_react_multi_tool(traced_emitter: InMemoryEmitter) -> None:
    client = make_llm_client("anthropic")
    agent = ReActAgent(
        name="react-agent",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You are a helpful assistant. Use the provided tools to look up "
            "information. The second tool REQUIRES you to pass the integer "
            "population returned by the first tool, so always call "
            "``lookup_population_a`` first, read its observation, and then "
            "call ``lookup_population_b`` with the integer you observed."
        ),
        tools=[lookup_population_a, lookup_population_b],
        max_iterations=5,
    )

    result = await run_with_retry(
        lambda: agent.run(
            "First look up the population of City A. Then use that integer as "
            "``reference_population`` when calling lookup_population_b. Finally, "
            "state which city has the larger population and by how much."
        ),
        max_attempts=2,
    )

    # --- Sequential-composition assertion ---
    # Count AgentStepEvents whose `action` field is populated. Two populated
    # actions prove the loop emitted two tool-using turns — a single batched
    # turn would show only one populated action followed by a no-action
    # final-answer step.
    step_events = [e for e in traced_emitter.events if isinstance(e, AgentStepEvent)]
    acting_steps = [e for e in step_events if e.action]
    assert len(acting_steps) >= 2, (
        f"Expected at least 2 AgentStepEvents with non-empty action, got: {[e.action for e in step_events]}"
    )

    # --- Interleaved reasoning-and-action: thought + observation populated ---
    assert any(e.thought for e in step_events), (
        f"Expected at least one AgentStepEvent with non-empty thought, got: {[e.thought for e in step_events]}"
    )
    assert any(e.observation for e in step_events), (
        f"Expected at least one AgentStepEvent with non-empty observation, got: {[e.observation for e in step_events]}"
    )

    # --- Both tools invoked and both returned successfully ---
    tool_calls = [e for e in traced_emitter.events if isinstance(e, ToolInvokeEvent)]
    tool_names = {e.tool_name for e in tool_calls}
    assert "lookup_population_a" in tool_names, f"Expected lookup_population_a, got: {tool_names}"
    assert "lookup_population_b" in tool_names, f"Expected lookup_population_b, got: {tool_names}"

    assert_trace_contains(
        traced_emitter,
        ToolResultEvent,
        predicate=lambda e: e.tool_name == "lookup_population_a" and e.success is True,
    )
    assert_trace_contains(
        traced_emitter,
        ToolResultEvent,
        predicate=lambda e: e.tool_name == "lookup_population_b" and e.success is True,
    )

    # --- Termination and iteration bounds ---
    assert result.termination_reason == "complete", (
        f"Expected termination_reason == 'complete', got: {result.termination_reason!r}"
    )
    assert 2 <= result.total_steps <= 4, f"Expected 2 <= total_steps <= 4, got: {result.total_steps}"

    # --- Fuzzy output ---
    await assert_result_satisfies(
        result.output or "",
        "The output references both City A's population (around 850000) and "
        "City B's population (around 620000), and states that City A is larger "
        "by approximately 230000.",
    )
