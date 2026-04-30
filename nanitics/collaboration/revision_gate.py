from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from nanitics.collaboration.approval_gate import ApprovalGate
from nanitics.composition.orchestration.protocol import Step, StepResult
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import (
    RevisionAttemptEvent,
    RevisionCompleteEvent,
    RevisionStartEvent,
)


class RevisionGate:
    """Workflow step that loops workers through human review until approval.

    Workers produce output, an ``ApprovalGate`` presents it for review, and
    if the human requests revision, workers re-run with feedback appended.
    Repeats until approval, rejection, or ``max_revisions`` is reached.

    HITL events emitted by the composed ``ApprovalGate`` carry
    ``agent_name`` from the gate's ``agent_name`` kwarg — ``RevisionGate``
    does not take its own ``agent_name``. For the multi-worker case, the
    adopter chooses which producer label to pin on the gate (there is no
    prescribed composite label).

    Args:
        workers: Steps that produce output for review.
        gate: The ApprovalGate used for human review.
        name: Step name for identification.
        emitter: Event emitter for observability.
        max_revisions: Maximum revision iterations before auto-rejection.
        on_output: Optional callback invoked after workers produce output,
            before the gate. Receives ``(worker_output, attempt, feedback)``
            where ``attempt=0`` and ``feedback=""`` on the initial run.
            If it returns a non-None value, that value replaces the worker
            output for the gate and final result.
    """

    def __init__(
        self,
        workers: list[Step],
        gate: ApprovalGate,
        name: str,
        emitter: EventEmitter | None = None,
        max_revisions: int = 10,
        on_output: Callable[[Any, int, str], Any] | None = None,
    ) -> None:
        self._workers = workers
        self._gate = gate
        self._name = name
        self._emitter = emitter
        self._max_revisions = max_revisions
        self._on_output = on_output

    @property
    def name(self) -> str:
        return self._name

    async def execute(self, input: Any) -> StepResult:
        """Run workers, present output for review, and loop on revision requests.

        Returns:
            StepResult reflecting approval, rejection, or max-revisions outcome.
        """
        if self._emitter is not None:
            self._emitter.emit(
                RevisionStartEvent(
                    trace_id=self._emitter.trace_id,
                    span_id=self._emitter.span_id,
                    parent_span_id=self._emitter.parent_span_id,
                    step_name=self._name,
                    worker_count=len(self._workers),
                    max_revisions=self._max_revisions,
                )
            )

        attempt = 0
        current_input = input
        feedback = ""

        while True:
            worker_outputs = await self._run_workers(current_input)

            if self._on_output is not None:
                transformed = self._on_output(worker_outputs, attempt, feedback)
                if transformed is not None:
                    worker_outputs = transformed

            gate_result = await self._gate.execute(worker_outputs, revision_count=attempt)

            if not gate_result.metadata.get("revision_requested"):
                final_decision = "reject" if gate_result.metadata.get("rejected") else "approve"
                self._emit_complete(attempt, final_decision)
                return gate_result

            attempt += 1
            if attempt > self._max_revisions:
                self._emit_complete(attempt - 1, "max_revisions_exceeded")
                return StepResult(
                    output=None,
                    metadata={"rejected": True, "reason": "Maximum revisions exceeded"},
                )

            feedback = gate_result.metadata.get("feedback", "")

            if self._emitter is not None:
                self._emitter.emit(
                    RevisionAttemptEvent(
                        trace_id=self._emitter.trace_id,
                        span_id=self._emitter.span_id,
                        parent_span_id=self._emitter.parent_span_id,
                        step_name=self._name,
                        attempt_number=attempt,
                        feedback=feedback,
                    )
                )

            current_input = (
                f"{input}\n\n"
                f"--- Your Previous Output ---\n"
                f"{worker_outputs}\n\n"
                f"--- Revision Requested (attempt {attempt} of {self._max_revisions}) ---\n"
                f"Reviewer feedback: {feedback}\n"
                f"Revise your previous output to address the feedback. "
                f"Change ONLY what the feedback asks for — keep everything else exactly as it was."
            )

    async def _run_workers(self, input: Any) -> Any:
        if len(self._workers) == 1:
            result = await self._workers[0].execute(input)
            return result.output

        results = await asyncio.gather(*(worker.execute(input) for worker in self._workers))
        return {worker.name: result.output for worker, result in zip(self._workers, results, strict=True)}

    def _emit_complete(self, total_attempts: int, final_decision: str) -> None:
        if self._emitter is not None:
            self._emitter.emit(
                RevisionCompleteEvent(
                    trace_id=self._emitter.trace_id,
                    span_id=self._emitter.span_id,
                    parent_span_id=self._emitter.parent_span_id,
                    step_name=self._name,
                    total_attempts=total_attempts,
                    final_decision=final_decision,
                )
            )
