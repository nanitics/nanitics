"""ApprovalGate: workflow step for human approval of content.

Demonstrates ApprovalGate and CallbackHumanInputProvider. Covers all four
decision paths (APPROVE, REJECT, OVERRIDE, REVISE), dynamic prompt generation,
and observability through HumanInputRequestEvent and HumanInputResponseEvent.

Related guide: docs/guides/human-in-the-loop.md
"""

import asyncio

from examples.helpers import make_emitter
from nanitics.hitl import (
    ApprovalGate,
    CallbackHumanInputProvider,
    HumanDecision,
    HumanInputRequest,
    HumanInputResponse,
)
from nanitics.infrastructure import (
    HumanInputRequestEvent,
    HumanInputResponseEvent,
)


async def main() -> None:
    # --- Section 1: Approval — Output Passes Through ---
    print("--- Section 1: Approval — Output Passes Through ---")

    provider = CallbackHumanInputProvider(
        lambda req: HumanInputResponse(
            request_id=req.request_id,
            decision=HumanDecision.APPROVE,
        )
    )
    gate = ApprovalGate(provider=provider, prompt="Approve this draft?", run_id="example-90-approve")
    result = await gate.execute("draft report content")

    assert result.output == "draft report content", "Approved input passes through unchanged"
    assert "rejected" not in result.metadata
    assert "modified" not in result.metadata
    print("✓ Approval passes input through unchanged")

    # --- Section 2: Rejection — Metadata Shows Reason ---
    print("\n--- Section 2: Rejection — Metadata Shows Reason ---")

    provider = CallbackHumanInputProvider(
        lambda req: HumanInputResponse(
            request_id=req.request_id,
            decision=HumanDecision.REJECT,
            content="Not ready for publication",
        )
    )
    gate = ApprovalGate(provider=provider, prompt="Approve this draft?", run_id="example-90-reject")
    result = await gate.execute("incomplete draft")

    assert result.output is None, "Rejected gate returns None output"
    assert result.metadata["rejected"] is True
    assert result.metadata["reason"] == "Not ready for publication"
    print("✓ Rejection returns None with metadata reason")

    # --- Section 3: Override — Human Alters Output ---
    print("\n--- Section 3: Override — Human Alters Output ---")

    provider = CallbackHumanInputProvider(
        lambda req: HumanInputResponse(
            request_id=req.request_id,
            decision=HumanDecision.OVERRIDE,
            content="improved draft content",
        )
    )
    gate = ApprovalGate(provider=provider, prompt="Approve or override?", run_id="example-90-override")
    result = await gate.execute("original draft")

    assert result.output == "improved draft content", "Overridden output is the human's content"
    assert result.metadata["modified"] is True
    print("✓ Override replaces output with human's content")

    # --- Section 4: Revision Request — Feedback for Rework ---
    print("\n--- Section 4: Revision Request — Feedback for Rework ---")

    provider = CallbackHumanInputProvider(
        lambda req: HumanInputResponse(
            request_id=req.request_id,
            decision=HumanDecision.REVISE,
            content="Add more data to section 3",
        )
    )
    gate = ApprovalGate(
        provider=provider,
        prompt="Review this analysis?",
        allow_revision=True,
        run_id="example-90-revise",
    )
    result = await gate.execute("analysis draft")

    assert result.output is None, "Revision returns None — input needs rework"
    assert result.metadata["revision_requested"] is True
    assert result.metadata["feedback"] == "Add more data to section 3"
    print("✓ Revision signals rework with feedback")

    # --- Section 5: Dynamic Prompts and Context ---
    print("\n--- Section 5: Dynamic Prompts and Context ---")

    captured: dict[str, HumanInputRequest] = {}

    def capture_and_approve(req: HumanInputRequest) -> HumanInputResponse:
        captured["request"] = req
        return HumanInputResponse(
            request_id=req.request_id,
            decision=HumanDecision.APPROVE,
        )

    gate = ApprovalGate(
        provider=CallbackHumanInputProvider(capture_and_approve),
        prompt=lambda draft: f"Approve this draft?\n\n{draft[:50]}...",
        context=lambda draft: f"Length: {len(draft)} characters",
        run_id="example-90-dynamic",
    )
    draft = "A longer piece of content that exceeds fifty chars for demonstration"
    result = await gate.execute(draft)

    req = captured["request"]
    assert "Approve this draft?" in req.prompt
    assert draft[:50] in req.prompt
    assert req.context is not None
    assert f"Length: {len(draft)} characters" in req.context
    assert result.output == draft, "Approval still passes input through"
    print("✓ Dynamic prompts and context computed from input")

    # --- Section 6: Observability — Emitted Events ---
    print("\n--- Section 6: Observability — Emitted Events ---")

    emitter = make_emitter("approval-observability")
    provider = CallbackHumanInputProvider(
        lambda req: HumanInputResponse(
            request_id=req.request_id,
            decision=HumanDecision.APPROVE,
        )
    )
    gate = ApprovalGate(
        provider=provider,
        emitter=emitter,
        prompt="Approve for publication?",
        run_id="example-90-events",
    )
    result = await gate.execute("ready content")

    request_events = [e for e in emitter.events if isinstance(e, HumanInputRequestEvent)]
    response_events = [e for e in emitter.events if isinstance(e, HumanInputResponseEvent)]

    assert len(request_events) == 1, "Exactly one request event emitted"
    assert request_events[0].request_type == "approval"
    assert request_events[0].prompt == "Approve for publication?"

    assert len(response_events) == 1, "Exactly one response event emitted"
    assert response_events[0].decision == "approve"
    assert response_events[0].has_content is False
    assert response_events[0].wait_duration_ms >= 0
    print("✓ Events emitted with correct fields for debugging")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
