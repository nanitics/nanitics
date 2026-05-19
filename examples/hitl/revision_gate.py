"""RevisionGate: iterative human review with feedback-driven revision.

Demonstrates the full revision lifecycle: workers produce output, a human reviewer
approves, requests revision with feedback, or rejects. On revision, feedback is
injected into the worker input and workers re-run. The loop ends on approval,
rejection, or when max_revisions is exceeded.

Prerequisite reading: examples/workflows/sequential_pipeline.py (Step/FunctionStep)
and examples/hitl/approval_gate.py (ApprovalGate basics).

Related guide: docs/guides/human-in-the-loop.md
"""

import asyncio

from examples.helpers import make_emitter
from nanitics.composition import FunctionStep
from nanitics.hitl import (
    ApprovalGate,
    CallbackHumanInputProvider,
    HumanDecision,
    HumanInputRequest,
    HumanInputResponse,
    RevisionGate,
)
from nanitics.infrastructure import (
    RevisionAttemptEvent,
    RevisionCompleteEvent,
    RevisionStartEvent,
)
from nanitics.tracing import InMemoryEmitter


def make_gate(
    decisions: list[tuple[HumanDecision, str | None]],
    emitter: InMemoryEmitter | None = None,
    run_id: str = "example-91",
) -> ApprovalGate:
    """Create an ApprovalGate that returns scripted decisions in sequence.

    Each entry is (decision, content) — content is feedback text for REVISE
    or a reason string for REJECT. APPROVE uses None.
    """
    call_index = 0

    def callback(req: HumanInputRequest) -> HumanInputResponse:
        nonlocal call_index
        decision, content = decisions[call_index]
        call_index += 1
        return HumanInputResponse(
            request_id=req.request_id,
            decision=decision,
            content=content,
        )

    return ApprovalGate(
        provider=CallbackHumanInputProvider(callback=callback),
        allow_revision=True,
        emitter=emitter,
        run_id=run_id,
    )


async def main() -> None:
    # --- Section 1: Approve on First Attempt ---
    print("--- Section 1: Approve on First Attempt ---")

    # Simplest path: worker runs, human approves, output returned directly.

    async def analyze(x: str) -> str:
        return f"Analysis of: {x}"

    gate = make_gate([(HumanDecision.APPROVE, None)])
    rg = RevisionGate(
        workers=[FunctionStep("analyst", analyze)],
        gate=gate,
        name="review",
    )

    result = await rg.execute("Q4 revenue trends")

    assert result.output == "Analysis of: Q4 revenue trends"
    assert result.metadata.get("rejected") is not True
    print(f"  Output: {result.output}")

    # --- Section 2: Revise Then Approve ---
    print("\n--- Section 2: Revise Then Approve ---")

    # Core revision loop: worker produces output, human requests revision with
    # feedback, worker re-runs with augmented input, human approves the revision.

    received_inputs: list[str] = []

    async def capture_worker(x: str) -> str:
        received_inputs.append(x)
        return f"output: {x}"

    gate = make_gate(
        [
            (HumanDecision.REVISE, "Add cost comparison"),
            (HumanDecision.APPROVE, None),
        ]
    )
    rg = RevisionGate(
        workers=[FunctionStep("analyst", capture_worker)],
        gate=gate,
        name="review",
        max_revisions=5,
    )

    result = await rg.execute("evaluate vendors")

    # Worker was called twice: original input, then augmented input with feedback.
    assert len(received_inputs) == 2
    assert received_inputs[0] == "evaluate vendors"
    assert "--- Your Previous Output ---" in received_inputs[1]
    assert "--- Revision Requested (attempt 1 of 5) ---" in received_inputs[1]
    assert "Reviewer feedback: Add cost comparison" in received_inputs[1]
    # Final output reflects the second worker execution.
    assert "evaluate vendors" in result.output
    assert "Revision Requested" in result.output
    print(f"  First input:  {received_inputs[0]}")
    print(f"  Second input: {received_inputs[1][:80]}...")
    print(f"  Final output: {result.output[:80]}...")

    # --- Section 3: Reject ---
    print("\n--- Section 3: Reject ---")

    # Human rejects on first review — worker runs once, gate rejects, loop exits.

    call_count = 0

    async def counted_worker(x: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"draft: {x}"

    gate = make_gate([(HumanDecision.REJECT, "Not acceptable")])
    rg = RevisionGate(
        workers=[FunctionStep("writer", counted_worker)],
        gate=gate,
        name="review",
    )

    result = await rg.execute("write proposal")

    assert result.output is None
    assert result.metadata["rejected"] is True
    assert call_count == 1  # Worker only ran once
    print(f"  Rejected: {result.metadata['rejected']}")

    # --- Section 4: Max Revisions Exceeded ---
    print("\n--- Section 4: Max Revisions Exceeded ---")

    # Human keeps requesting revisions past the limit — safety bound kicks in.

    decisions = [(HumanDecision.REVISE, f"feedback {i}") for i in range(5)]
    gate = make_gate(decisions)
    rg = RevisionGate(
        workers=[FunctionStep("worker", analyze)],
        gate=gate,
        name="review",
        max_revisions=2,
    )

    result = await rg.execute("draft report")

    assert result.output is None
    assert result.metadata["rejected"] is True
    assert result.metadata["reason"] == "Maximum revisions exceeded"
    print(f"  Rejected: {result.metadata['rejected']}")
    print(f"  Reason:   {result.metadata['reason']}")

    # --- Section 5: Multiple Workers ---
    print("\n--- Section 5: Multiple Workers ---")

    # Two workers run in parallel — output is a dict keyed by worker name.
    # On revision, both workers re-run with the augmented input.

    counts = {"pricing": 0, "capabilities": 0}

    async def pricing_worker(x: str) -> str:
        counts["pricing"] += 1
        return f"pricing({counts['pricing']}): {x}"

    async def capabilities_worker(x: str) -> str:
        counts["capabilities"] += 1
        return f"capabilities({counts['capabilities']}): {x}"

    gate = make_gate(
        [
            (HumanDecision.REVISE, "More detail on enterprise tier"),
            (HumanDecision.APPROVE, None),
        ]
    )
    workers = [
        FunctionStep("pricing", pricing_worker),
        FunctionStep("capabilities", capabilities_worker),
    ]
    rg = RevisionGate(workers=workers, gate=gate, name="review")

    result = await rg.execute("evaluate vendors")

    # Output is a dict with both worker names as keys.
    assert isinstance(result.output, dict)
    assert "pricing" in result.output
    assert "capabilities" in result.output
    # Both workers ran twice (initial + revision).
    assert counts["pricing"] == 2
    assert counts["capabilities"] == 2
    # Final output reflects second run.
    assert "pricing(2)" in result.output["pricing"]
    assert "capabilities(2)" in result.output["capabilities"]
    print(f"  Output keys: {list(result.output.keys())}")
    print(f"  Pricing:     {result.output['pricing'][:60]}...")
    print(f"  Capabilities:{result.output['capabilities'][:60]}...")

    # --- Section 6: Observability Events ---
    print("\n--- Section 6: Observability Events ---")

    # RevisionGate emits three event types tracking the revision lifecycle.

    emitter = make_emitter("revision-events")
    gate = make_gate(
        [
            (HumanDecision.REVISE, "Fix formatting"),
            (HumanDecision.APPROVE, None),
        ],
        emitter=emitter,
    )
    rg = RevisionGate(
        workers=[FunctionStep("worker", analyze)],
        gate=gate,
        name="review-step",
        emitter=emitter,
    )

    await rg.execute("draft report")

    start_events = [e for e in emitter.events if isinstance(e, RevisionStartEvent)]
    assert len(start_events) == 1
    assert start_events[0].step_name == "review-step"
    assert start_events[0].worker_count == 1
    assert start_events[0].max_revisions == 10
    print(f"  RevisionStartEvent: step={start_events[0].step_name}, workers={start_events[0].worker_count}")

    attempt_events = [e for e in emitter.events if isinstance(e, RevisionAttemptEvent)]
    assert len(attempt_events) == 1
    assert attempt_events[0].attempt_number == 1
    assert attempt_events[0].feedback == "Fix formatting"
    print(f"  RevisionAttemptEvent: attempt={attempt_events[0].attempt_number}, feedback={attempt_events[0].feedback}")

    complete_events = [e for e in emitter.events if isinstance(e, RevisionCompleteEvent)]
    assert len(complete_events) == 1
    assert complete_events[0].total_attempts == 1
    assert complete_events[0].final_decision == "approve"
    print(
        f"  RevisionCompleteEvent: attempts={complete_events[0].total_attempts}, "
        f"decision={complete_events[0].final_decision}"
    )

    # --- Section 7: Output Callback (on_output) ---
    print("\n--- Section 7: Output Callback (on_output) ---")

    # The on_output callback runs after workers produce output, before the gate.
    # Use it for side effects (event emission, logging) or output transformation.
    # Receives (worker_output, attempt, feedback) — attempt=0 and feedback=""
    # on the initial run.

    output_log: list[dict[str, object]] = []

    def handle_output(output: str, attempt: int, feedback: str) -> str:
        output_log.append({"attempt": attempt, "feedback": feedback, "output": output})
        # Transform the output (e.g., parse, enrich, or normalize)
        return f"[attempt {attempt}] {output}"

    gate = make_gate(
        [
            (HumanDecision.REVISE, "Be more concise"),
            (HumanDecision.APPROVE, None),
        ]
    )
    rg = RevisionGate(
        workers=[FunctionStep("analyst", analyze)],
        gate=gate,
        name="review",
        on_output=handle_output,
    )

    result = await rg.execute("Q4 trends")

    # Callback was called twice: initial attempt and after revision.
    assert len(output_log) == 2
    assert output_log[0]["attempt"] == 0
    assert output_log[0]["feedback"] == ""
    assert output_log[1]["attempt"] == 1
    assert output_log[1]["feedback"] == "Be more concise"
    # Final output reflects the transformation from on_output.
    assert result.output.startswith("[attempt 1]")
    print(f"  Callback calls: {len(output_log)}")
    print(f"  First:  attempt={output_log[0]['attempt']}, feedback='{output_log[0]['feedback']}'")
    print(f"  Second: attempt={output_log[1]['attempt']}, feedback='{output_log[1]['feedback']}'")
    print(f"  Final output: {result.output[:60]}...")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
