"""A ``return_direct`` tool ends a ReActAgent run on its result, against a real model.

A ``ReActAgent`` is given two tools: ``lookup_customer`` (a normal,
non-terminal tool) and ``submit_proposal`` marked ``return_direct=True``.
The task asks the agent to look up a customer and then submit a proposal.
When the model calls ``submit_proposal``, the run must end on that tool's
``ToolResult`` — no closing prose turn — with the tool's content as the
run output and ``termination_reason == "return_direct"``.

This is the consumer-facing claim (Studio's headless delegation): the
terminal action is a tool call, and the loop must not spend one more LLM
generation producing a message the caller discards. The load-bearing pins
are the ``termination_reason`` value (only reachable via the flag), the
output identity (the run output IS the tool content, not model prose), and
the trace showing the terminal ``ToolInvokeEvent`` fired with no assistant
turn after it.

Acceptance criteria:
  - ``result.termination_reason == "return_direct"``.
  - ``result.output`` equals the ``submit_proposal`` tool's content exactly
    (proves the output is the tool result, not synthesised prose).
  - The trace contains a ``ToolInvokeEvent`` for ``submit_proposal``.
  - No ``tool_result`` message follows the terminal one — i.e. the
    proposal's result is the last ``tool_result`` in the conversation — and
    structured data the tool attached survives on that message's metadata.
"""

from __future__ import annotations

from nanitics.infrastructure import ToolInvokeEvent
from nanitics.strategies import ReActAgent, tool
from nanitics.strategies.tools import ToolResult
from nanitics.tracing import InMemoryEmitter
from validation.helpers import (
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)

_PROPOSAL_CONTENT = "PROPOSAL: upgrade customer cust-1 to the Enterprise plan."


@tool("lookup_customer", "Look up a customer by name. Returns the customer ID.")
async def lookup_customer(name: str) -> str:
    return f"Customer '{name}' has ID cust-1 on the Pro plan."


@tool(
    "submit_proposal",
    "Submit the final proposal for the customer. This ends the task.",
    return_direct=True,
)
async def submit_proposal(proposal: str) -> ToolResult:
    # The terminal tool attaches structured data on metadata — the caller
    # reads it off the last tool_result message rather than from prose.
    return ToolResult(content=_PROPOSAL_CONTENT, metadata={"customer_id": "cust-1"})


async def test_return_direct_ends_run_on_tool_result(
    traced_emitter: InMemoryEmitter,
) -> None:
    client = make_llm_client("anthropic")

    agent = ReActAgent(
        name="proposal-writer",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You are a sales assistant. Look up the customer, then submit a "
            "single proposal with `submit_proposal`. Submitting the proposal "
            "completes the task — do not say anything after submitting."
        ),
        tools=[lookup_customer, submit_proposal],
        max_iterations=5,
    )

    result = await run_with_retry(
        lambda: agent.run("Look up customer 'Acme' and submit a proposal to upgrade them."),
        max_attempts=2,
    )

    # --- Termination invariant: only the flag can produce this value ---
    assert result.termination_reason == "return_direct", (
        f"Expected termination_reason='return_direct', got: {result.termination_reason!r}"
    )

    # --- Output identity: the run output IS the tool content, not prose ---
    assert result.output == _PROPOSAL_CONTENT, (
        f"Expected output to be the tool's content verbatim; got: {result.output!r}"
    )
    assert result.parsed is None

    # --- Trace pin: the terminal tool actually fired ---
    invoke = assert_trace_contains(traced_emitter, ToolInvokeEvent)
    invoke_names = [e.tool_name for e in traced_emitter.events if isinstance(e, ToolInvokeEvent)]
    assert "submit_proposal" in invoke_names, (
        f"Expected a submit_proposal invocation; tool invocations were: {invoke_names}"
    )
    assert invoke is not None

    # --- Conversation pin: the proposal result is the LAST tool_result, and
    # its structured metadata survived onto the message. ---
    tool_results = [m for m in result.messages if m.role == "tool_result"]
    assert tool_results, "expected at least one tool_result message"
    assert tool_results[-1].content == _PROPOSAL_CONTENT
    assert tool_results[-1].metadata == {"customer_id": "cust-1"}
