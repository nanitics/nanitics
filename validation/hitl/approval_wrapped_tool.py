"""ApprovalWrappedTool gating on real-LLM tool invocations: APPROVE and REJECT.

A real ``ReActAgent`` is given a single ``send_email`` tool wrapped in
``ApprovalWrappedTool``. The approval provider's callback is scripted to
return APPROVE or REJECT, and a side-channel call counter on the underlying
tool pins whether the inner function actually executed. Parametrization
exercises the two gate branches:

- APPROVE: human approves → the underlying tool runs exactly once, a
  ``ToolInvokeEvent`` for ``send_email`` appears, and the agent's final
  answer reflects successful delivery.
- REJECT: human rejects → the underlying tool is never called, no
  ``ToolInvokeEvent`` for ``send_email`` appears, and the agent's final
  answer reflects that the action did not happen.

The subjects of the test are the gate's pre-execution approval check, the
exact side-effect count on the inner tool, and the trace evidence that the
wrapped tool was scheduled with the right parameters regardless of
decision.

Acceptance criteria (evaluated for every parametrization):
  - Trace contains exactly one ``HumanInputRequestEvent`` whose
    ``tool_name == "send_email"``, ``request_type == "approval"``,
    ``agent_name == "mailer"`` (the agent whose tool_state supplied the
    name), ``metadata["tool_name"] == "send_email"``, and whose
    ``metadata["parameters"]`` contains ``to`` and ``subject`` keys.
  - Trace contains exactly one ``HumanInputResponseEvent`` whose
    ``decision`` matches the parametrized decision and whose
    ``request_id`` matches the request event.

Acceptance criteria (decision-specific):
  - APPROVE: ``invocations == 1``; trace contains at least one
    ``ToolInvokeEvent`` with ``tool_name == "send_email"``; the agent's
    final answer indicates the email was sent.
  - REJECT: ``invocations == 0``; trace contains zero ``ToolInvokeEvent``
    with ``tool_name == "send_email"``; the agent's final answer reflects
    that the email was not sent.
"""

from __future__ import annotations

import pytest

from nanitics.collaboration.protocol import HumanInputRequest
from nanitics.hitl import (
    ApprovalWrappedTool,
    CallbackHumanInputProvider,
    HumanDecision,
    HumanInputResponse,
)
from nanitics.infrastructure import (
    HumanInputRequestEvent,
    HumanInputResponseEvent,
    ToolInvokeEvent,
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
)


@pytest.mark.quick
@pytest.mark.parametrize(
    ("decision", "criterion"),
    [
        (
            HumanDecision.APPROVE,
            "The answer indicates the email was sent successfully.",
        ),
        (
            HumanDecision.REJECT,
            "The answer indicates the email was NOT sent — rejected, blocked, or otherwise not delivered.",
        ),
    ],
    ids=["approve", "reject"],
)
async def test_approval_wrapped_tool_lifecycle(
    traced_emitter: InMemoryEmitter,
    decision: HumanDecision,
    criterion: str,
) -> None:
    client = make_llm_client("anthropic")

    invocation_log: list[dict[str, str]] = []

    @tool("send_email", "Send an email to a recipient.")
    async def send_email(to: str, subject: str, body: str) -> str:
        invocation_log.append({"to": to, "subject": subject, "body": body})
        return f"Email delivered to {to}."

    def _callback(req: HumanInputRequest) -> HumanInputResponse:
        return HumanInputResponse(
            request_id=req.request_id,
            decision=decision,
            content=("Policy forbids unsolicited email." if decision is HumanDecision.REJECT else None),
        )

    provider = CallbackHumanInputProvider(_callback)
    wrapped_send_email = ApprovalWrappedTool(
        tool=send_email,
        provider=provider,
        emitter=traced_emitter,
    )

    agent = ReActAgent(
        name="mailer",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You are an assistant that can send email. When the user asks "
            "you to send a message, call ``send_email`` exactly once with "
            "sensible ``to``, ``subject``, and ``body`` fields. After the "
            "tool returns — or refuses — state plainly in one sentence "
            "whether the email was actually sent."
        ),
        tools=[wrapped_send_email],
        tool_state={"run_id": "validation-92-mailer", "agent_name": "mailer"},
        max_iterations=3,
    )

    result = await run_with_retry(
        lambda: agent.run(
            "Please send an email to alex@example.com with subject 'Quarterly update' and a short friendly body."
        ),
        max_attempts=2,
    )

    # --- HITL event pair ---
    request_events = [e for e in traced_emitter.events if isinstance(e, HumanInputRequestEvent)]
    response_events = [e for e in traced_emitter.events if isinstance(e, HumanInputResponseEvent)]
    assert len(request_events) == 1, f"Expected exactly one HumanInputRequestEvent; got {len(request_events)}."
    assert len(response_events) == 1, f"Expected exactly one HumanInputResponseEvent; got {len(response_events)}."
    request_event = assert_trace_contains(
        traced_emitter,
        HumanInputRequestEvent,
        predicate=lambda e: e.tool_name == "send_email" and e.request_type == "approval",
    )
    # --- Trace-level shape of the request event: agent_name and metadata
    # land on the event itself (no provider-wrapper workaround needed). ---
    assert request_event.agent_name == "mailer", (
        "Expected the request event's agent_name to reflect the ambient "
        f"tool_state ('mailer'); got {request_event.agent_name!r}."
    )
    assert request_event.metadata.get("tool_name") == "send_email", (
        f"Expected event metadata['tool_name'] == 'send_email'; got {request_event.metadata!r}."
    )
    event_params = request_event.metadata.get("parameters") or {}
    assert "to" in event_params, (
        f"Expected event metadata to expose the 'to' parameter; got parameters={event_params!r}."
    )
    assert "subject" in event_params, (
        f"Expected event metadata to expose the 'subject' parameter; got parameters={event_params!r}."
    )
    response_event = assert_trace_contains(
        traced_emitter,
        HumanInputResponseEvent,
        predicate=lambda e: e.decision == decision.value,
    )
    assert request_event.request_id == response_event.request_id, (
        "Request/response events should share one request_id lifecycle; "
        f"got request={request_event.request_id!r}, "
        f"response={response_event.request_id!r}."
    )

    # --- Decision-specific side-effect + trace invariants ---
    send_email_invokes = [
        e for e in traced_emitter.events if isinstance(e, ToolInvokeEvent) and e.tool_name == "send_email"
    ]
    if decision is HumanDecision.APPROVE:
        assert len(invocation_log) == 1, (
            f"APPROVE: inner tool must run exactly once; got {len(invocation_log)} call(s)."
        )
        assert len(send_email_invokes) >= 1, "APPROVE: expected at least one ToolInvokeEvent for send_email; got zero."
    else:
        assert invocation_log == [], f"REJECT: inner tool must not run; got invocations={invocation_log!r}."
        assert send_email_invokes == [], (
            f"REJECT: trace must contain zero ToolInvokeEvent for send_email; got {len(send_email_invokes)}."
        )

    await assert_result_satisfies(result.output or "", criterion)
