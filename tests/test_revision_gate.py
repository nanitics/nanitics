"""Tests for RevisionGate: approve, revise, reject, max revisions, events."""

from nanitics.collaboration.approval_gate import ApprovalGate
from nanitics.collaboration.protocol import (
    CallbackHumanInputProvider,
    HumanDecision,
    HumanInputRequest,
    HumanInputResponse,
)
from nanitics.collaboration.revision_gate import RevisionGate
from nanitics.composition.orchestration.adapters import FunctionStep
from nanitics.composition.orchestration.protocol import Step
from nanitics.infrastructure.observability.emitter import InMemoryEmitter
from nanitics.infrastructure.observability.events import (
    RevisionAttemptEvent,
    RevisionCompleteEvent,
    RevisionStartEvent,
)
from tests.testing_helpers import make_emitter

_DEFAULT_RUN_ID = "test-run"


def make_gate(
    decisions: list[tuple[HumanDecision, str | None]],
    allow_revision: bool = True,
    emitter: InMemoryEmitter | None = None,
    run_id: str = _DEFAULT_RUN_ID,
    name: str = "approval_gate",
    observed: list[HumanInputRequest] | None = None,
) -> ApprovalGate:
    """Create a gate that returns decisions in sequence.

    When ``observed`` is provided, every request the callback sees is
    appended to it so tests can assert on request identity.
    """
    call_index = 0

    def callback(req: HumanInputRequest) -> HumanInputResponse:
        nonlocal call_index
        if observed is not None:
            observed.append(req)
        decision, content = decisions[call_index]
        call_index += 1
        return HumanInputResponse(
            request_id=req.request_id,
            decision=decision,
            content=content,
        )

    return ApprovalGate(
        provider=CallbackHumanInputProvider(callback=callback),
        allow_revision=allow_revision,
        emitter=emitter,
        run_id=run_id,
        name=name,
    )


def make_worker(name: str = "worker") -> FunctionStep:
    """Create a simple worker that echoes input with its name."""

    async def fn(x: str) -> str:
        return f"{name}: {x}"

    return FunctionStep(name=name, fn=fn)


class TestRevisionGateConstruction:
    def test_satisfies_step_protocol(self) -> None:
        gate = make_gate([(HumanDecision.APPROVE, None)])
        rg = RevisionGate(workers=[make_worker()], gate=gate, name="review")
        assert isinstance(rg, Step)

    def test_name_property(self) -> None:
        gate = make_gate([(HumanDecision.APPROVE, None)])
        rg = RevisionGate(workers=[make_worker()], gate=gate, name="my-review")
        assert rg.name == "my-review"


class TestRevisionGateSingleWorker:
    async def test_approve_on_first_attempt(self) -> None:
        gate = make_gate([(HumanDecision.APPROVE, None)])
        rg = RevisionGate(
            workers=[make_worker("analyst")],
            gate=gate,
            name="review",
        )
        result = await rg.execute("evaluate vendors")
        assert result.output == "analyst: evaluate vendors"

    async def test_revise_once_then_approve(self) -> None:
        gate = make_gate(
            [
                (HumanDecision.REVISE, "Add cost comparison"),
                (HumanDecision.APPROVE, None),
            ]
        )
        rg = RevisionGate(
            workers=[make_worker("analyst")],
            gate=gate,
            name="review",
        )
        result = await rg.execute("evaluate vendors")
        # After revision, the worker received augmented input including previous output
        assert "evaluate vendors" in result.output
        assert "Your Previous Output" in result.output
        assert "Revision Requested" in result.output
        assert "Add cost comparison" in result.output


class TestRevisionGateMultipleWorkers:
    async def test_parallel_execution_outputs_collected_as_dict(self) -> None:
        gate = make_gate([(HumanDecision.APPROVE, None)])
        workers: list[Step] = [make_worker("pricing"), make_worker("capabilities")]
        rg = RevisionGate(workers=workers, gate=gate, name="review")
        result = await rg.execute("evaluate vendors")
        assert result.output["pricing"] == "pricing: evaluate vendors"
        assert result.output["capabilities"] == "capabilities: evaluate vendors"

    async def test_revise_reruns_all_workers(self) -> None:
        call_counts = {"a": 0, "b": 0}

        async def fn_a(x: str) -> str:
            call_counts["a"] += 1
            return f"a({call_counts['a']}): {x}"

        async def fn_b(x: str) -> str:
            call_counts["b"] += 1
            return f"b({call_counts['b']}): {x}"

        gate = make_gate(
            [
                (HumanDecision.REVISE, "More detail"),
                (HumanDecision.APPROVE, None),
            ]
        )
        workers: list[Step] = [FunctionStep(name="a", fn=fn_a), FunctionStep(name="b", fn=fn_b)]
        rg = RevisionGate(workers=workers, gate=gate, name="review")
        result = await rg.execute("task")

        # Both workers ran twice (once initial, once after revision)
        assert call_counts["a"] == 2
        assert call_counts["b"] == 2
        # Final output reflects second run
        assert "a(2)" in result.output["a"]
        assert "b(2)" in result.output["b"]


class TestRevisionGateMaxRevisions:
    async def test_max_revisions_exceeded_returns_rejection(self) -> None:
        # Always request revision — should hit limit
        decisions: list[tuple[HumanDecision, str | None]] = [(HumanDecision.REVISE, f"feedback {i}") for i in range(5)]
        gate = make_gate(decisions)
        rg = RevisionGate(
            workers=[make_worker()],
            gate=gate,
            name="review",
            max_revisions=3,
        )
        result = await rg.execute("task")
        assert result.output is None
        assert result.metadata["rejected"] is True
        assert result.metadata["reason"] == "Maximum revisions exceeded"


class TestRevisionGateReject:
    async def test_reject_at_gate_returns_immediately(self) -> None:
        gate = make_gate([(HumanDecision.REJECT, "Not acceptable")])
        call_count = 0

        async def fn(x: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"output: {x}"

        worker = FunctionStep(name="worker", fn=fn)
        rg = RevisionGate(workers=[worker], gate=gate, name="review")
        result = await rg.execute("task")
        assert result.output is None
        assert result.metadata["rejected"] is True
        assert call_count == 1  # Worker ran only once


class TestRevisionGateEvents:
    async def test_emits_revision_events(self) -> None:
        emitter = make_emitter()
        gate = make_gate(
            [
                (HumanDecision.REVISE, "Fix formatting"),
                (HumanDecision.APPROVE, None),
            ],
            emitter=emitter,
        )
        rg = RevisionGate(
            workers=[make_worker()],
            gate=gate,
            name="review-step",
            emitter=emitter,
        )
        await rg.execute("task")

        start_events = [e for e in emitter.events if isinstance(e, RevisionStartEvent)]
        assert len(start_events) == 1
        assert start_events[0].step_name == "review-step"
        assert start_events[0].worker_count == 1
        assert start_events[0].max_revisions == 10

        attempt_events = [e for e in emitter.events if isinstance(e, RevisionAttemptEvent)]
        assert len(attempt_events) == 1
        assert attempt_events[0].step_name == "review-step"
        assert attempt_events[0].attempt_number == 1
        assert attempt_events[0].feedback == "Fix formatting"

        complete_events = [e for e in emitter.events if isinstance(e, RevisionCompleteEvent)]
        assert len(complete_events) == 1
        assert complete_events[0].step_name == "review-step"
        assert complete_events[0].total_attempts == 1
        assert complete_events[0].final_decision == "approve"

    async def test_reject_emits_complete_with_reject_decision(self) -> None:
        emitter = make_emitter()
        gate = make_gate([(HumanDecision.REJECT, "No")], emitter=emitter)
        rg = RevisionGate(
            workers=[make_worker()],
            gate=gate,
            name="review",
            emitter=emitter,
        )
        await rg.execute("task")

        complete_events = [e for e in emitter.events if isinstance(e, RevisionCompleteEvent)]
        assert len(complete_events) == 1
        assert complete_events[0].final_decision == "reject"
        assert complete_events[0].total_attempts == 0

    async def test_max_revisions_emits_complete_with_exceeded(self) -> None:
        emitter = make_emitter()
        decisions: list[tuple[HumanDecision, str | None]] = [(HumanDecision.REVISE, "again") for _ in range(3)]
        gate = make_gate(decisions, emitter=emitter)
        rg = RevisionGate(
            workers=[make_worker()],
            gate=gate,
            name="review",
            emitter=emitter,
            max_revisions=2,
        )
        await rg.execute("task")

        complete_events = [e for e in emitter.events if isinstance(e, RevisionCompleteEvent)]
        assert len(complete_events) == 1
        assert complete_events[0].final_decision == "max_revisions_exceeded"


class TestRevisionGateFeedbackFormat:
    async def test_feedback_injected_into_worker_input(self) -> None:
        received_inputs: list[str] = []

        async def capture_fn(x: str) -> str:
            received_inputs.append(x)
            return f"output: {x}"

        gate = make_gate(
            [
                (HumanDecision.REVISE, "Be more specific about pricing"),
                (HumanDecision.APPROVE, None),
            ]
        )
        worker = FunctionStep(name="worker", fn=capture_fn)
        rg = RevisionGate(
            workers=[worker],
            gate=gate,
            name="review",
            max_revisions=5,
        )
        await rg.execute("analyze vendor proposals")

        # First call gets original input
        assert received_inputs[0] == "analyze vendor proposals"
        # Second call gets augmented input with previous output and feedback
        assert received_inputs[1].startswith("analyze vendor proposals")
        assert "--- Your Previous Output ---" in received_inputs[1]
        assert "output: analyze vendor proposals" in received_inputs[1]
        assert "--- Revision Requested (attempt 1 of 5) ---" in received_inputs[1]
        assert "Reviewer feedback: Be more specific about pricing" in received_inputs[1]
        assert (
            "Revise your previous output to address the feedback. Change ONLY what the feedback asks for"
            in received_inputs[1]
        )


class TestRevisionGateOnOutput:
    async def test_on_output_called_with_attempt_and_feedback(self) -> None:
        """Verify callback receives correct arguments over multiple attempts."""
        calls: list[tuple[str, int, str]] = []

        def on_output(output: str, attempt: int, feedback: str) -> None:
            calls.append((output, attempt, feedback))

        gate = make_gate(
            [
                (HumanDecision.REVISE, "Add details"),
                (HumanDecision.APPROVE, None),
            ]
        )
        rg = RevisionGate(
            workers=[make_worker("w")],
            gate=gate,
            name="review",
            on_output=on_output,
        )
        await rg.execute("task")

        assert len(calls) == 2
        # First call: attempt=0, feedback=""
        assert calls[0][1] == 0
        assert calls[0][2] == ""
        # Second call: attempt=1, feedback from revision
        assert calls[1][1] == 1
        assert calls[1][2] == "Add details"

    async def test_on_output_transforms_output(self) -> None:
        """Return a transformed value; verify gate sees it and final result contains it."""

        def on_output(output: str, attempt: int, feedback: str) -> str:
            return f"TRANSFORMED: {output}"

        gate = make_gate([(HumanDecision.APPROVE, None)])
        rg = RevisionGate(
            workers=[make_worker("w")],
            gate=gate,
            name="review",
            on_output=on_output,
        )
        result = await rg.execute("task")
        assert result.output == "TRANSFORMED: w: task"

    async def test_on_output_none_keeps_original(self) -> None:
        """Return None from callback; verify original output is preserved."""

        def on_output(output: str, attempt: int, feedback: str) -> None:
            return None

        gate = make_gate([(HumanDecision.APPROVE, None)])
        rg = RevisionGate(
            workers=[make_worker("w")],
            gate=gate,
            name="review",
            on_output=on_output,
        )
        result = await rg.execute("task")
        assert result.output == "w: task"

    async def test_on_output_not_called_when_not_provided(self) -> None:
        """Default behavior unchanged when on_output is not set."""
        gate = make_gate([(HumanDecision.APPROVE, None)])
        rg = RevisionGate(
            workers=[make_worker("w")],
            gate=gate,
            name="review",
        )
        result = await rg.execute("task")
        assert result.output == "w: task"

    async def test_on_output_initial_attempt_has_empty_feedback(self) -> None:
        """Verify feedback="" and attempt=0 on first call."""
        calls: list[tuple[int, str]] = []

        def on_output(output: str, attempt: int, feedback: str) -> None:
            calls.append((attempt, feedback))

        gate = make_gate([(HumanDecision.APPROVE, None)])
        rg = RevisionGate(
            workers=[make_worker("w")],
            gate=gate,
            name="review",
            on_output=on_output,
        )
        await rg.execute("task")

        assert len(calls) == 1
        assert calls[0] == (0, "")


class TestRevisionGateDeterministicIdentity:
    async def test_revision_slots_progress_by_attempt_number(self) -> None:
        """Across a REVISE → APPROVE cycle, the two gate requests must share
        the same ``{run_id}:{name}`` prefix but differ in their trailing
        ``:N`` revision-count slot so each loop iteration has stable,
        distinguishable identity for suspend/resume replay."""
        observed: list[HumanInputRequest] = []
        gate = make_gate(
            [
                (HumanDecision.REVISE, "more detail"),
                (HumanDecision.APPROVE, None),
            ],
            run_id="r",
            name="gate",
            observed=observed,
        )
        rg = RevisionGate(
            workers=[make_worker("w")],
            gate=gate,
            name="review",
        )
        await rg.execute("task")

        assert len(observed) == 2
        assert observed[0].request_id == "r:gate:0"
        assert observed[1].request_id == "r:gate:1"
