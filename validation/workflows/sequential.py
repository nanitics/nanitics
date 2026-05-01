"""Sequential pipeline: researcher → writer, both real agents.

Validates the :class:`Sequential` workflow composing two real
:class:`ReActAgent` steps. The researcher is instructed to include a
distinctive token (``NANITICS-SEQ-7F3A``) in its output; the writer is
instructed to preserve any such token verbatim. The token round-trip
pins the defining property of the primitive: **stage N's output is
stage N+1's input**. A broken pipe that fed the original input into
both stages would produce a writer output lacking the token.

Acceptance criteria:
  - ``AgentStartEvent`` observed for both ``researcher`` and ``writer``
    (proves each wrapped agent's loop actually ran).
  - ``WorkflowStepCompleteEvent`` observed for ``researcher`` at
    ``step_index == 0`` and ``writer`` at ``step_index == 1`` — pins
    execution order.
  - ``result.metadata["total_steps_executed"] == 2``.
  - ``result.metadata["intermediate_results"]`` has the researcher step
    and its output contains the distinctive token — proves the token
    made it out of stage 1.
  - The distinctive token is present in ``result.output`` — proves
    stage 2 received stage 1's output (round-trip).
  - Final output summarizes findings about Python 3.13.
"""

from __future__ import annotations

import pytest

from nanitics import AgentStep, InMemoryEmitter, ReActAgent, Sequential
from nanitics.infrastructure import (
    AgentStartEvent,
    WorkflowStepCompleteEvent,
)
from validation.helpers import (
    assert_result_satisfies,
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)

# Distinctive token injected by researcher and preserved by writer.
# A broken pipe (writer called on the original input) cannot produce it.
PIPE_TOKEN = "NANITICS-SEQ-7F3A"


@pytest.mark.quick
async def test_sequential_research_writer(traced_emitter: InMemoryEmitter) -> None:
    researcher = ReActAgent(
        name="researcher",
        llm_client=make_llm_client("anthropic"),
        emitter=traced_emitter,
        system_prompt=(
            "You are a research assistant. Provide concise key findings on the given topic. "
            f"You MUST end your response with the exact tag [{PIPE_TOKEN}] on its own line — "
            "downstream tooling relies on it."
        ),
        tools=[],
        max_iterations=2,
    )
    writer = ReActAgent(
        name="writer",
        llm_client=make_llm_client("anthropic"),
        emitter=traced_emitter,
        system_prompt=(
            "You are a technical writer. Summarize the research findings provided in 1-2 sentences. "
            f"If the input contains a tag of the form [{PIPE_TOKEN}], you MUST include that exact tag "
            "verbatim somewhere in your output so downstream tooling can trace it."
        ),
        tools=[],
        max_iterations=2,
    )

    workflow = Sequential(
        name="research-pipeline",
        steps=[AgentStep(researcher), AgentStep(writer)],
        emitter=traced_emitter,
    )

    result = await run_with_retry(
        lambda: workflow.execute("What's new in Python 3.13?"),
        max_attempts=2,
    )

    # --- Per-agent AgentStartEvent proves each wrapped agent's loop ran ---
    assert_trace_contains(traced_emitter, AgentStartEvent, predicate=lambda e: e.agent_name == "researcher")
    assert_trace_contains(traced_emitter, AgentStartEvent, predicate=lambda e: e.agent_name == "writer")

    # --- Per-step events with pinned step_index prove execution order ---
    assert_trace_contains(
        traced_emitter,
        WorkflowStepCompleteEvent,
        predicate=lambda e: e.step_name == "researcher" and e.step_index == 0,
    )
    assert_trace_contains(
        traced_emitter,
        WorkflowStepCompleteEvent,
        predicate=lambda e: e.step_name == "writer" and e.step_index == 1,
    )

    # --- Metadata invariants ---
    assert result.metadata["total_steps_executed"] == 2, (
        f"Expected 2 steps, got: {result.metadata['total_steps_executed']}"
    )

    # --- Input-piping proof: token present in researcher's output AND final output ---
    intermediate_results = result.metadata["intermediate_results"]
    assert "researcher" in intermediate_results, (
        f"Expected 'researcher' in intermediate_results, got keys: {list(intermediate_results)}"
    )
    researcher_output = str(intermediate_results["researcher"].output or "")
    assert PIPE_TOKEN in researcher_output, (
        f"Expected researcher output to contain the pipe token {PIPE_TOKEN!r}; got: {researcher_output!r}"
    )
    final_output = str(result.output or "")
    assert PIPE_TOKEN in final_output, (
        f"Expected writer output to echo the pipe token {PIPE_TOKEN!r} (proves stage 1's "
        f"output was fed to stage 2); got: {final_output!r}"
    )

    # --- Fuzzy semantic check on content ---
    await assert_result_satisfies(
        final_output,
        "The output summarizes research findings about Python 3.13.",
    )
