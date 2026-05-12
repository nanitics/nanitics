"""Structured handoff between two real agents.

A sender ``ReActAgent`` researches a deterministic fact about Python's
``asyncio`` module via a lookup tool. A ``HandoffStep`` runs the sender,
applies a ``HandoffTransfer`` strategy that lifts the tool result into a
``HandoffPayload.findings`` entry, renders the payload as markdown, and
emits a ``HandoffEvent``. The rendered markdown is then passed as input
to a receiver ``ReActAgent`` that produces a polished one-paragraph
summary. The deterministic fact carries a sentinel phrase
(``"collects their results in order"``) — if the payload did not actually
flow from sender to receiver, the sentinel would not appear in the
receiver's output.

``HandoffStep`` is used (rather than ``HandoffTransfer`` directly) because
``HandoffEvent`` is only emitted by the step — the strategy itself only
extracts text.

The test is parametrised over ``to_agent`` to prove the ``HandoffEvent``
destination tracks the configured value rather than a constant
(dispatch correctness for "multiple potential targets").

Acceptance criteria:
  - Rendered handoff markdown contains ``## Handoff Context``,
    ``### Task State``, ``### Findings``, and ``### Decisions`` headers
    (pins the conditional-section branches of ``HandoffPayload.render``
    that the builder populates).
  - Trace contains a ``ToolInvokeEvent`` with
    ``tool_name == "lookup_fact"`` (proves the sender actually used its
    tool).
  - Trace contains an ``AgentStepEvent`` with ``agent_name == "writer"``
    (proves the receiver actually executed).
  - Trace contains a ``HandoffEvent`` with ``from_agent == "researcher"``,
    ``to_agent`` matching the configured target, ``payload_size > 0``,
    and ``"output" in payload_fields``.
  - Receiver output contains the sentinel phrase
    ``"collects their results in order"`` — proves the payload flowed
    from sender tool → builder → rendered markdown → receiver input →
    receiver output.
"""

from __future__ import annotations

import pytest

from nanitics import (
    AgentResult,
    InMemoryEmitter,
    ReActAgent,
    tool,
)
from nanitics.infrastructure import (
    AgentStepEvent,
    HandoffEvent,
    ToolInvokeEvent,
)
from nanitics.patterns import (
    HandoffPayload,
    HandoffStep,
    HandoffTransfer,
)
from validation.helpers import (
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)

_ASYNCIO_FACTS = {
    "event_loop": "asyncio uses an event loop to schedule and run coroutines cooperatively.",
    "gather": "asyncio.gather runs awaitables concurrently and collects their results in order.",
    "task": "asyncio.create_task schedules a coroutine to run on the event loop and returns a Task handle.",
    "timeout": "asyncio.timeout provides a context manager for cancelling long-running operations.",
}

# Sentinel phrase lifted verbatim from _ASYNCIO_FACTS["gather"]. Only
# reaches the receiver if the tool result flowed through the builder
# into the rendered payload.
_SENTINEL = "collects their results in order"


@tool("lookup_fact", "Look up a short factual statement about Python's asyncio module by topic key.")
async def lookup_fact(topic: str) -> str:
    fact = _ASYNCIO_FACTS.get(topic)
    if fact is None:
        keys = ", ".join(sorted(_ASYNCIO_FACTS))
        return f"No fact recorded for '{topic}'. Known topics: {keys}."
    return fact


def _build_handoff(result: AgentResult) -> HandoffPayload:
    """Map the sender's AgentResult into a structured HandoffPayload.

    The sender's final output is lifted verbatim into ``findings`` so the
    deterministic tool-result text (which contains the sentinel phrase)
    is preserved through the render → receiver-input path.
    """
    findings_text = result.output or "No findings produced."
    return HandoffPayload(
        task_state="Summarise the researcher's finding about asyncio.gather.",
        findings=[findings_text],
        decisions=[f"Researcher completed in {result.total_steps} step(s)."],
    )


@pytest.mark.quick
@pytest.mark.parametrize("to_agent", ["writer", "editor"])
async def test_handoff_between_agents(traced_emitter: InMemoryEmitter, to_agent: str) -> None:
    client = make_llm_client("anthropic")

    sender = ReActAgent(
        name="researcher",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You are a research specialist. Call the lookup_fact tool with "
            "topic='gather' to retrieve the fact. In your final answer, "
            "report the fact's full text verbatim — do not paraphrase or "
            "truncate it."
        ),
        tools=[lookup_fact],
        max_iterations=3,
    )

    receiver = ReActAgent(
        name="writer",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You are a technical writer. You will receive a structured handoff "
            "context block containing research findings. Produce a polished "
            "one-paragraph summary that quotes the finding's exact phrasing "
            "where it uses the word 'concurrently' — preserve the original "
            "wording including any phrase describing the result ordering."
        ),
        tools=[],
        max_iterations=2,
    )

    transfer_strategy = HandoffTransfer(builder=_build_handoff)
    handoff_step = HandoffStep(
        agent=sender,
        emitter=traced_emitter,
        transfer_strategy=transfer_strategy,
        to_agent=to_agent,
    )

    sender_step_result = await run_with_retry(
        lambda: handoff_step.execute(
            "Look up the asyncio 'gather' fact and report it verbatim so it can be handed off to a downstream writer."
        ),
        max_attempts=2,
    )

    rendered_handoff = sender_step_result.output
    for header in ("## Handoff Context", "### Task State", "### Findings", "### Decisions"):
        assert header in rendered_handoff, f"Expected {header!r} in rendered handoff, got: {rendered_handoff!r}"

    receiver_result = await run_with_retry(
        lambda: receiver.run(rendered_handoff),
        max_attempts=2,
    )

    # --- Trace-shape invariants ---
    assert_trace_contains(
        traced_emitter,
        ToolInvokeEvent,
        predicate=lambda e: e.tool_name == "lookup_fact",
    )
    assert_trace_contains(
        traced_emitter,
        AgentStepEvent,
        predicate=lambda e: e.agent_name == "writer",
    )
    assert_trace_contains(
        traced_emitter,
        HandoffEvent,
        predicate=lambda e: (
            e.from_agent == "researcher"
            and e.to_agent == to_agent
            and e.payload_size > 0
            and "output" in e.payload_fields
        ),
    )

    # --- Sentinel carry-over ---
    # The sentinel is a sub-phrase of the deterministic tool result. It
    # only reaches the receiver's output if (a) the sender invoked the
    # tool, (b) the builder lifted the tool result into findings, (c) the
    # render emitted the findings section, and (d) the receiver read and
    # echoed the phrase. A regression at any stage breaks this assertion.
    receiver_output = receiver_result.output or ""
    assert _SENTINEL in receiver_output, (
        f"Expected sentinel {_SENTINEL!r} in receiver output (carry-over "
        f"from sender's tool result), got: {receiver_output!r}"
    )
