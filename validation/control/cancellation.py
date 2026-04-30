"""CancellationToken stops a ReActAgent mid-run and emits SafetyCancellationEvent.

Mid-run cancellation of a running async agent is inherently racy when
cancellation is triggered from another coroutine — the agent's loop
checks the token only between steps. To make this deterministic in a
real-LLM integration test, the cancellation is triggered by a *tool*
(``signal_cancel``) that flips the token during its own execution. The
agent then sees ``is_cancelled == True`` on its next loop iteration and
breaks out with ``termination_reason == "cancelled"``.

The task is designed to require more than one step: the system prompt
and user instruction push the model to call ``gather_chunk`` three
times before producing a final answer. ``signal_cancel`` is surfaced as
a side-effect tool the model is instructed to call at a specific
midpoint. This yields:

  - at least one real ``gather_chunk`` invocation (work happened),
  - a ``signal_cancel`` invocation (cancellation was triggered inside
    the agent run, not before it started),
  - termination via the cancellation path, not via iteration limit or
    natural completion.

Acceptance criteria:
  - ``result.termination_reason == "cancelled"``.
  - ``result.output is None`` (cancelled agents do not produce a final
    answer).
  - Trace contains a ``SafetyCancellationEvent`` with
    ``agent_name == "cancelled-agent"``.
  - Trace contains at least one ``ToolInvokeEvent`` for
    ``gather_chunk`` *before* the cancellation event — proves the
    agent did real work before being cancelled.
  - Trace contains a ``ToolInvokeEvent`` for ``signal_cancel`` — pins
    the cancellation pathway (mid-run, via a tool), not a pre-cancel.
  - The number of ``AgentStepEvent`` equals the number of completed
    steps before cancellation (i.e. events follow the step count on
    the result, which is less than the loop's max_iterations).
  - After the ``SafetyCancellationEvent``, no further
    ``LLMRequestEvent`` fires — cancellation halts further provider
    calls.
"""

from __future__ import annotations

from nanitics import CancellationToken, InMemoryEmitter, ReActAgent, tool
from nanitics.infrastructure import (
    AgentStepEvent,
    LLMRequestEvent,
    SafetyCancellationEvent,
    ToolInvokeEvent,
)
from validation.helpers import (
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)


async def test_cancellation_token_stops_agent_mid_run(traced_emitter: InMemoryEmitter) -> None:
    client = make_llm_client("anthropic")

    async def _run() -> object:
        # Reset trace state so trace assertions only see the current attempt.
        traced_emitter.events.clear()

        token = CancellationToken()

        @tool(
            "gather_chunk",
            "Gather a chunk of data for the report. Call with a chunk label like 'a', 'b', or 'c'.",
        )
        async def gather_chunk(label: str) -> str:
            return f"Chunk {label} gathered: 42 rows."

        @tool(
            "signal_cancel",
            "Signal that the run should be cancelled. Call this once when instructed.",
        )
        async def signal_cancel() -> str:
            token.cancel()
            return "Cancellation signalled."

        agent = ReActAgent(
            name="cancelled-agent",
            llm_client=client,
            emitter=traced_emitter,
            system_prompt=(
                "You are a data gathering assistant. Follow the user's instructions "
                "step by step and only call the tools they explicitly ask for."
            ),
            tools=[gather_chunk, signal_cancel],
            cancellation_token=token,
            max_iterations=10,
        )

        return await agent.run(
            "Do the following in order, one tool call per turn: "
            "(1) call `gather_chunk` with label 'a'; "
            "(2) call `gather_chunk` with label 'b'; "
            "(3) call `signal_cancel` to abort the run; "
            "(4) call `gather_chunk` with label 'c'; "
            "(5) summarize everything you gathered. "
            "Call one tool per turn."
        )

    result = await run_with_retry(_run, max_attempts=2)

    # --- Result-shape invariants ---
    assert result.termination_reason == "cancelled", (
        f"Expected termination_reason='cancelled', got: {result.termination_reason!r}"
    )
    assert result.output is None, f"Expected result.output is None after cancellation, got: {result.output!r}"

    # --- Trace-shape invariants ---
    cancel_event = assert_trace_contains(
        traced_emitter,
        SafetyCancellationEvent,
        predicate=lambda e: e.agent_name == "cancelled-agent",
    )

    # The signal_cancel tool must have been invoked — this pins that the
    # cancellation came from inside the run, not from a pre-cancelled token.
    signal_invocations = [
        e for e in traced_emitter.events if isinstance(e, ToolInvokeEvent) and e.tool_name == "signal_cancel"
    ]
    assert len(signal_invocations) >= 1, (
        f"Expected at least one ToolInvokeEvent for 'signal_cancel' (proves mid-run cancellation), "
        f"got: {len(signal_invocations)}"
    )

    # Real work must have happened before the cancellation — otherwise the
    # scenario degenerates into a pre-cancel that doesn't test mid-run
    # behaviour.
    cancel_index = traced_emitter.events.index(cancel_event)
    gather_before_cancel = [
        i
        for i, e in enumerate(traced_emitter.events)
        if isinstance(e, ToolInvokeEvent) and e.tool_name == "gather_chunk" and i < cancel_index
    ]
    assert len(gather_before_cancel) >= 1, (
        f"Expected at least one ToolInvokeEvent for 'gather_chunk' BEFORE SafetyCancellationEvent "
        f"(to prove the agent did work prior to cancellation); "
        f"gather indices before cancel were: {gather_before_cancel}"
    )

    # Step-count invariant: the number of AgentStepEvent matches the step
    # count on the result. This is a shape check — if cancellation short-
    # circuited the step emission or counted an extra step, one of these
    # would drift.
    step_events = [e for e in traced_emitter.events if isinstance(e, AgentStepEvent)]
    assert len(step_events) == result.total_steps, (
        f"Expected {result.total_steps} AgentStepEvent(s) to match result.total_steps, got {len(step_events)}."
    )

    # After cancellation, no further provider calls should be made — the
    # loop check runs at the top of each iteration, so any LLMRequestEvent
    # after the cancellation event would indicate a missed check.
    llm_requests_after_cancel = [
        i for i, e in enumerate(traced_emitter.events) if isinstance(e, LLMRequestEvent) and i > cancel_index
    ]
    assert not llm_requests_after_cancel, (
        f"Expected zero LLMRequestEvent after SafetyCancellationEvent, got indices: {llm_requests_after_cancel}"
    )
