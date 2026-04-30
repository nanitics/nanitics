"""ApprovalGate lifecycle on real-LLM-produced output: APPROVE and REJECT.

A real ``ReActAgent`` drafts a one-sentence product description. The output
flows through an ``ApprovalGate`` backed by a deterministic
``CallbackHumanInputProvider``. The script is parametrized across
``HumanDecision.APPROVE`` and ``HumanDecision.REJECT`` to exercise two of the
gate's four decision branches (APPROVE pass-through vs. reject-to-None) in a
single end-to-end trace each. The subjects of the test are the gate plumbing
(request/response event lifecycle), the decision-branch result shape, and
the trace-level shape of the emitted ``HumanInputRequestEvent``.

Acceptance criteria (evaluated for every parametrization):
  - Trace contains exactly one ``HumanInputRequestEvent`` whose
    ``agent_name == "drafter"`` (the producing agent, threaded onto the
    gate via its ``agent_name`` kwarg), whose ``metadata["step_name"] ==
    "approval_gate"``, whose ``prompt`` equals the configured prompt, and
    whose ``request_type == "approval"``.
  - Trace contains exactly one ``HumanInputResponseEvent`` whose ``decision``
    matches the parametrized decision, whose ``has_content is False``, and
    whose ``wait_duration_ms >= 0``.
  - Request/response event ``request_id`` values match (one lifecycle, not two).

Acceptance criteria (decision-specific, one per parametrization):
  - APPROVE: ``gated.output == result.output`` (pass-through invariant) and
    the gated text is a one-sentence product description for headphones.
  - REJECT: ``gated.output is None`` and ``gated.metadata["rejected"] is
    True`` (the reject branch returns no output and annotates metadata).
"""

from __future__ import annotations

import pytest

from nanitics import (
    ApprovalGate,
    CallbackHumanInputProvider,
    HumanDecision,
    HumanInputResponse,
    InMemoryEmitter,
    ReActAgent,
)
from nanitics.collaboration.protocol import HumanInputRequest
from nanitics.infrastructure import HumanInputRequestEvent, HumanInputResponseEvent
from validation.helpers import (
    assert_result_satisfies,
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)

_GATE_PROMPT = "Approve this product description for publication?"
_GATE_NAME = "approval_gate"
_PRODUCER_AGENT_NAME = "drafter"


@pytest.mark.quick
@pytest.mark.parametrize(
    "decision",
    [HumanDecision.APPROVE, HumanDecision.REJECT],
    ids=["approve", "reject"],
)
async def test_approval_gate_lifecycle(traced_emitter: InMemoryEmitter, decision: HumanDecision) -> None:
    client = make_llm_client("anthropic")

    agent = ReActAgent(
        name="drafter",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You are a marketing copywriter. Draft exactly one sentence — no "
            "more — that describes the product the user names. Output only the "
            "sentence; do not add commentary."
        ),
        tools=[],
        max_iterations=3,
    )

    def _callback(req: HumanInputRequest) -> HumanInputResponse:
        return HumanInputResponse(request_id=req.request_id, decision=decision)

    provider = CallbackHumanInputProvider(_callback)
    gate = ApprovalGate(
        provider=provider,
        emitter=traced_emitter,
        prompt=_GATE_PROMPT,
        name=_GATE_NAME,
        run_id="validation-90-approval-gate",
        agent_name=_PRODUCER_AGENT_NAME,
    )

    # ``run_with_retry`` wraps only the agent call: the LLM step is the only
    # non-deterministic, out-of-process dependency. ``gate.execute`` runs an
    # in-process deterministic callback, so retrying it would mask real bugs.
    result = await run_with_retry(
        lambda: agent.run("Write a one-sentence product description for a noise-cancelling wireless headphone."),
        max_attempts=2,
    )
    gated = await gate.execute(result.output)

    # --- Trace-shape invariants: exactly one request, exactly one response ---
    request_events = [e for e in traced_emitter.events if isinstance(e, HumanInputRequestEvent)]
    response_events = [e for e in traced_emitter.events if isinstance(e, HumanInputResponseEvent)]
    assert len(request_events) == 1, f"Expected exactly one HumanInputRequestEvent; got {len(request_events)}."
    assert len(response_events) == 1, f"Expected exactly one HumanInputResponseEvent; got {len(response_events)}."

    # --- Trace-level shape of the request event: producer agent_name,
    # metadata, prompt, and request_type all land on the event itself
    # (no provider-wrapper workaround needed). ---
    request_event = assert_trace_contains(
        traced_emitter,
        HumanInputRequestEvent,
        predicate=lambda e: e.metadata.get("step_name") == _GATE_NAME and e.prompt == _GATE_PROMPT,
    )
    assert request_event.agent_name == _PRODUCER_AGENT_NAME, (
        "Expected the request event's agent_name to be the producing agent "
        f"({_PRODUCER_AGENT_NAME!r}); got {request_event.agent_name!r}."
    )
    assert request_event.request_type == "approval", (
        f"Expected request_type == 'approval'; got {request_event.request_type!r}."
    )
    response_event = assert_trace_contains(
        traced_emitter,
        HumanInputResponseEvent,
        predicate=lambda e: e.decision == decision.value and e.has_content is False and e.wait_duration_ms >= 0,
    )
    assert request_event.request_id == response_event.request_id, (
        "Request and response events should share one request_id lifecycle; "
        f"got request={request_event.request_id!r}, "
        f"response={response_event.request_id!r}."
    )

    # --- Decision-specific result-shape invariants ---
    if decision is HumanDecision.APPROVE:
        assert gated.output == result.output, (
            "Expected APPROVE to pass input through unchanged; "
            f"gated.output={gated.output!r}, result.output={result.output!r}."
        )
        await assert_result_satisfies(
            gated.output or "",
            "The output is a one-sentence product description for headphones.",
        )
    else:
        assert gated.output is None, f"Expected REJECT to return output=None; got {gated.output!r}."
        assert gated.metadata.get("rejected") is True, (
            f"Expected REJECT to set metadata['rejected']=True; got {gated.metadata!r}."
        )
