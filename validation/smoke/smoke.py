"""Reference validation script. Copy this shape for new validation scripts.

Every new script must:
  (a) Use a real client from ``make_llm_client``.
  (b) Accept the ``traced_emitter`` fixture — it auto-saves the trace on
      teardown, including when the test fails.
  (c) State its acceptance criteria as executable assertions
      (``assert_trace_contains`` for trace shape, ``assert_result_satisfies``
      for output fuzziness).

Acceptance criteria for this smoke test:
  - ``AgentStartEvent`` is emitted with ``echo`` registered in
    ``tools_available``.
  - At least one ``AgentStepEvent`` is emitted (control loop steps).
  - ``LLMResponseEvent`` is emitted with positive input and output token
    counts (a real provider call happened).
  - ``ToolInvokeEvent`` and ``ToolResultEvent`` are emitted for the ``echo``
    tool (real tool round-trip).
  - ``AgentCompleteEvent`` is emitted with ``termination_reason='complete'``.
  - ``AgentResult`` has a non-empty ``output``, ``total_steps >= 1``,
    ``termination_reason == 'complete'``, and ``usage.total_tokens > 0``.
  - Judge (redundancy) confirms the answer acknowledges "smoke" was echoed.
"""

from __future__ import annotations

import json

import pytest

from nanitics.infrastructure import (
    AgentCompleteEvent,
    AgentStartEvent,
    AgentStepEvent,
    LLMResponseEvent,
    ToolInvokeEvent,
    ToolResultEvent,
)
from nanitics.strategies import (
    ReActAgent,
    tool,
)
from nanitics.tracing import InMemoryEmitter
from validation.helpers import (
    assert_result_satisfies,
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
    save_trace,
    validation_trace_dir,
)


@tool("echo", "Return the input message verbatim.")
async def echo(message: str) -> str:
    return message


@pytest.mark.quick
async def test_smoke_react_agent(traced_emitter: InMemoryEmitter) -> None:
    client = make_llm_client("anthropic")
    agent = ReActAgent(
        name="smoke-agent",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt="You are a helpful assistant. Use the echo tool when asked to echo something.",
        tools=[echo],
    )

    result = await run_with_retry(
        lambda: agent.run("Use the echo tool to repeat the word 'smoke'."),
        max_attempts=2,
    )

    # --- Lifecycle bookends ---
    assert_trace_contains(
        traced_emitter,
        AgentStartEvent,
        predicate=lambda e: "echo" in e.tools_available,
    )
    assert_trace_contains(traced_emitter, AgentStepEvent)
    assert_trace_contains(
        traced_emitter,
        AgentCompleteEvent,
        predicate=lambda e: e.termination_reason == "complete",
    )

    # --- Real provider round-trip ---
    assert_trace_contains(
        traced_emitter,
        LLMResponseEvent,
        predicate=lambda e: e.usage.input_tokens > 0 and e.usage.output_tokens > 0,
    )

    # --- Real tool round-trip ---
    assert_trace_contains(
        traced_emitter,
        ToolInvokeEvent,
        predicate=lambda e: e.tool_name == "echo",
    )
    assert_trace_contains(
        traced_emitter,
        ToolResultEvent,
        predicate=lambda e: e.tool_name == "echo" and e.success is True,
    )

    # --- AgentResult shape ---
    assert result.output, "Agent produced an empty output."
    assert result.total_steps >= 1, f"Expected total_steps >= 1, got {result.total_steps}."
    assert result.termination_reason == "complete", (
        f"Expected termination_reason='complete', got {result.termination_reason!r}."
    )
    assert result.usage.total_tokens > 0, f"Expected positive total_tokens, got {result.usage.total_tokens}."

    # --- Exported trace file exists and parses ---
    # The fixture's finaliser writes the trace on teardown. Flush now so we
    # can assert the serialised form round-trips through ``json.loads``.
    trace_path = validation_trace_dir() / "smoke_react_agent.json"
    save_trace(traced_emitter, trace_path, script="validation/smoke/smoke.py::test_smoke_react_agent")
    assert trace_path.exists(), f"Expected trace file at {trace_path}."
    json.loads(trace_path.read_text())

    # --- Fuzzy redundancy check ---
    await assert_result_satisfies(
        result.output,
        "The output acknowledges that the word 'smoke' was echoed.",
    )
