"""RevisionGate lifecycle on a real-LLM-produced output: APPROVE and REVISE→APPROVE.

A real ``ReActAgent`` drafts a short product description; the draft flows
through a ``RevisionGate`` whose review is driven by a deterministic
``CallbackHumanInputProvider``. Two parametrizations exercise the two
non-degenerate cycle counts: a single-cycle APPROVE (worker runs once, gate
approves) and a two-cycle REVISE→APPROVE (gate requests revision with
feedback, worker re-runs with feedback appended, gate approves the revision).
The subjects of the test are (a) the emitted HITL request events matching the
cycle count, (b) the revision-lifecycle trace events, and (c) direct evidence
that the producing agent was re-invoked when revision was requested.

Acceptance criteria (evaluated for every parametrization):
  - Trace contains ``expected_cycles`` ``HumanInputRequestEvent`` events,
    each with ``metadata["step_name"] == "review"``, ``agent_name ==
    "drafter"`` (the producer threaded onto the composed ``ApprovalGate``'s
    ``agent_name`` kwarg), and ``request_type == "approval"``.
  - Trace contains exactly one ``RevisionStartEvent`` whose ``step_name ==
    "review"``, ``worker_count == 1``, and ``max_revisions == 3``.
  - Trace contains exactly one ``RevisionCompleteEvent`` whose ``step_name ==
    "review"``, ``final_decision == "approve"``, and ``total_attempts`` matches
    the parametrization (0 for APPROVE, 1 for REVISE→APPROVE).
  - Trace contains ``expected_cycles`` ``HumanInputResponseEvent`` events;
    the final response event pairs with the final request event by
    ``request_id``.
  - The draft agent emitted exactly ``expected_cycles`` ``AgentStartEvent``
    events — direct evidence that REVISE re-invoked the producer.
  - The gated output is a non-empty string (final approved draft).

Acceptance criteria (cycle-specific, one per parametrization):
  - APPROVE: no ``RevisionAttemptEvent`` is present.
  - REVISE→APPROVE: exactly one ``RevisionAttemptEvent`` with
    ``attempt_number == 1`` and ``feedback`` equal to the scripted feedback
    string; the final worker input contains the ``--- Revision Requested``
    marker so we know feedback was threaded back into the re-invocation.
"""

from __future__ import annotations

import pytest

from nanitics import (
    AgentStep,
    ApprovalGate,
    CallbackHumanInputProvider,
    HumanDecision,
    HumanInputResponse,
    InMemoryEmitter,
    ReActAgent,
    RevisionGate,
)
from nanitics.collaboration.protocol import HumanInputRequest
from nanitics.infrastructure import (
    AgentStartEvent,
    HumanInputRequestEvent,
    HumanInputResponseEvent,
    RevisionAttemptEvent,
    RevisionCompleteEvent,
    RevisionStartEvent,
)
from validation.helpers import (
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)

_GATE_NAME = "review"
_FEEDBACK = "Make the sentence more concrete — mention battery life explicitly."


@pytest.mark.quick
@pytest.mark.parametrize(
    ("scripted_decisions", "expected_cycles"),
    [
        ([(HumanDecision.APPROVE, None)], 1),
        (
            [
                (HumanDecision.REVISE, _FEEDBACK),
                (HumanDecision.APPROVE, None),
            ],
            2,
        ),
    ],
    ids=["approve", "revise_then_approve"],
)
async def test_revision_gate_lifecycle(
    traced_emitter: InMemoryEmitter,
    scripted_decisions: list[tuple[HumanDecision, str | None]],
    expected_cycles: int,
) -> None:
    client = make_llm_client("anthropic")

    agent = ReActAgent(
        name="drafter",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You are a marketing copywriter. Draft exactly one sentence — no "
            "more — that describes the product the user names. If the user "
            "supplies reviewer feedback, revise strictly to address that "
            "feedback and change nothing else. Output only the sentence; do "
            "not add commentary."
        ),
        tools=[],
        max_iterations=3,
    )

    cursor = {"index": 0}

    def _callback(req: HumanInputRequest) -> HumanInputResponse:
        decision, content = scripted_decisions[cursor["index"]]
        cursor["index"] += 1
        return HumanInputResponse(
            request_id=req.request_id,
            decision=decision,
            content=content,
        )

    provider = CallbackHumanInputProvider(_callback)
    gate = ApprovalGate(
        provider=provider,
        emitter=traced_emitter,
        prompt="Approve this product description?",
        name=_GATE_NAME,
        allow_revision=True,
        run_id="validation-91-revision-gate",
        agent_name="drafter",
    )
    revision_gate = RevisionGate(
        workers=[AgentStep(agent)],
        gate=gate,
        name=_GATE_NAME,
        emitter=traced_emitter,
        max_revisions=3,
    )

    # Only the agent's LLM call is non-deterministic; the gate callback is
    # in-process and scripted, so the whole revision_gate.execute call is what
    # we retry — retrying just the agent wouldn't retry a judge-flake on the
    # second cycle.
    result = await run_with_retry(
        lambda: revision_gate.execute(
            "Write a one-sentence product description for a noise-cancelling wireless headphone."
        ),
        max_attempts=2,
    )

    # --- Revision lifecycle events ---
    start_events = [e for e in traced_emitter.events if isinstance(e, RevisionStartEvent)]
    complete_events = [e for e in traced_emitter.events if isinstance(e, RevisionCompleteEvent)]
    assert len(start_events) == 1, f"Expected exactly one RevisionStartEvent; got {len(start_events)}."
    assert len(complete_events) == 1, f"Expected exactly one RevisionCompleteEvent; got {len(complete_events)}."
    start_event = assert_trace_contains(
        traced_emitter,
        RevisionStartEvent,
        predicate=lambda e: e.step_name == _GATE_NAME and e.worker_count == 1 and e.max_revisions == 3,
    )
    assert start_event is start_events[0]

    expected_attempts = expected_cycles - 1
    complete_event = assert_trace_contains(
        traced_emitter,
        RevisionCompleteEvent,
        predicate=lambda e: (
            e.step_name == _GATE_NAME and e.final_decision == "approve" and e.total_attempts == expected_attempts
        ),
    )
    assert complete_event is complete_events[0]

    # --- HITL event counts + request_id round-trip on the final pair ---
    request_events = [e for e in traced_emitter.events if isinstance(e, HumanInputRequestEvent)]
    response_events = [e for e in traced_emitter.events if isinstance(e, HumanInputResponseEvent)]
    assert len(request_events) == expected_cycles, (
        f"Expected {expected_cycles} HumanInputRequestEvent(s); got {len(request_events)}."
    )
    assert len(response_events) == expected_cycles, (
        f"Expected {expected_cycles} HumanInputResponseEvent(s); got {len(response_events)}."
    )
    assert request_events[-1].request_id == response_events[-1].request_id, (
        "Final request/response events should share one request_id lifecycle; "
        f"got request={request_events[-1].request_id!r}, "
        f"response={response_events[-1].request_id!r}."
    )

    # --- Trace-level shape of every request event: producer agent_name,
    # step_name metadata, and approval taxonomy all land on the event itself
    # (no provider-wrapper workaround needed). ---
    for event in request_events:
        assert event.agent_name == "drafter", (
            f"Expected every request event to carry the producing agent's name ('drafter'); got {event.agent_name!r}."
        )
        assert event.metadata.get("step_name") == _GATE_NAME, (
            f"Expected metadata['step_name'] == {_GATE_NAME!r}; got {event.metadata!r}."
        )
        assert event.request_type == "approval", f"Expected request_type == 'approval'; got {event.request_type!r}."

    # --- Direct evidence the producer was re-invoked on REVISE ---
    drafter_starts = [e for e in traced_emitter.events if isinstance(e, AgentStartEvent) and e.agent_name == "drafter"]
    assert len(drafter_starts) == expected_cycles, (
        f"Expected the drafter agent to start {expected_cycles} time(s) "
        f"(one per revision cycle); got {len(drafter_starts)}."
    )

    # --- Gated output is a non-empty string (the approved draft) ---
    assert isinstance(result.output, str), (
        f"Expected approved draft to be a string; got {type(result.output).__name__}."
    )
    assert result.output.strip(), f"Expected a non-empty approved draft string; got {result.output!r}."

    # --- Cycle-specific invariants ---
    attempt_events = [e for e in traced_emitter.events if isinstance(e, RevisionAttemptEvent)]
    if expected_cycles == 1:
        assert attempt_events == [], (
            f"APPROVE on first cycle should emit no RevisionAttemptEvent; got {len(attempt_events)}."
        )
    else:
        assert len(attempt_events) == 1, (
            f"REVISE→APPROVE should emit exactly one RevisionAttemptEvent; got {len(attempt_events)}."
        )
        assert_trace_contains(
            traced_emitter,
            RevisionAttemptEvent,
            predicate=lambda e: e.step_name == _GATE_NAME and e.attempt_number == 1 and e.feedback == _FEEDBACK,
        )
        # The second AgentStartEvent's task_input is the feedback-threaded
        # prompt. The REVISE branch wraps previous output + reviewer feedback
        # into a new task string prefixed with markers — if the marker is
        # missing, the producer was not re-invoked with feedback.
        assert "--- Revision Requested" in drafter_starts[1].task_input, (
            "Second drafter invocation should receive feedback-augmented "
            f"input; got task_input={drafter_starts[1].task_input!r}."
        )
