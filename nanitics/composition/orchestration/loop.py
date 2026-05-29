from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from nanitics.composition.durability.models import RunCheckpoint
from nanitics.composition.durability.store import CheckpointStore
from nanitics.composition.durability.suspension import SuspendExecution
from nanitics.composition.orchestration.protocol import Step, StepResult
from nanitics.composition.orchestration.workflow import Workflow
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import (
    WorkflowStepCompleteEvent,
    WorkflowStepDefinition,
)
from nanitics.infrastructure.observability.storage import PersistentTraceStore
from nanitics.safety.cancellation import CancellationToken


class Loop(Workflow):
    """Repeatedly executes a step until a condition is met or max_iterations is reached.

    Between iterations, the step's output becomes the next iteration's input.
    The condition receives the step result and the 1-indexed iteration number;
    returning ``True`` stops the loop.

    Result metadata includes ``iterations`` and optionally
    ``terminated = "iteration_limit"`` if max_iterations was reached.

    Args:
        name: Workflow identifier.
        step: The step to execute repeatedly.
        condition: Callback ``(result, iteration) -> should_stop``. Can be sync or async.
        max_iterations: Hard upper bound on iterations. Defaults to 10.
        emitter: Event emitter for observability.
        cancellation_token: Optional cooperative cancellation signal.
        checkpoint_store: Optional store for suspension checkpoints.
        run_id: Run identifier for checkpoint records.
    """

    def __init__(
        self,
        *,
        name: str,
        step: Step,
        condition: Callable[[StepResult, int], bool | Awaitable[bool]],
        max_iterations: int = 10,
        emitter: EventEmitter,
        cancellation_token: CancellationToken | None = None,
        checkpoint_store: CheckpointStore | None = None,
        run_id: str | None = None,
        trace_store: PersistentTraceStore | None = None,
    ) -> None:
        super().__init__(
            name=name,
            emitter=emitter,
            cancellation_token=cancellation_token,
            checkpoint_store=checkpoint_store,
            run_id=run_id,
            trace_store=trace_store,
        )
        self._step = step
        self._condition = condition
        self._max_iterations = max_iterations

    def _workflow_type(self) -> str:
        return "loop"

    def _step_count(self) -> int:
        return 1

    def _get_step_definitions(self) -> list[WorkflowStepDefinition]:
        step_type, metadata = self._classify_step(self._step)
        metadata["max_iterations"] = self._max_iterations
        return [
            WorkflowStepDefinition(
                name=self._step.name,
                step_type=step_type,
                index=0,
                metadata=metadata,
            )
        ]

    async def _run(self, input: Any, *, resume_from: RunCheckpoint | None = None) -> StepResult:
        from nanitics.composition.orchestration.adapters import WorkflowStep

        current_input = input
        result: StepResult | None = None
        start_iteration = 1
        child_resume: RunCheckpoint | None = None

        if resume_from is not None:
            state = resume_from.state
            start_iteration = state["iteration"]
            current_input = state.get("last_result", {}).get("output", input) if state.get("last_result") else input
            if isinstance(self._step, WorkflowStep):
                nested = state.get("nested_checkpoint")
                assert nested is not None
                child_resume = self._child_checkpoint(resume_from, nested)
            self._emit_resumed(resume_from, self._step.name)

        for iteration in range(start_iteration, self._max_iterations + 1):
            if self._cancellation_token and self._cancellation_token.is_cancelled:
                return StepResult(
                    output=result.output if result else current_input,
                    metadata={
                        **(result.metadata if result else {}),
                        "iterations": iteration - 1,
                        "terminated": "cancelled",
                        "total_steps_executed": iteration - 1,
                    },
                )

            try:
                with self._emitter.span(f"{self._step.name}-iteration-{iteration}"):
                    bound_step = self._bind_step(self._step)
                    step_start = time.monotonic()
                    if child_resume is not None and iteration == start_iteration:
                        result = await self._execute_resumable(bound_step, current_input, child_resume)
                    else:
                        result = await bound_step.execute(current_input)
                    step_duration_ms = int((time.monotonic() - step_start) * 1000)
            except SuspendExecution as exc:
                checkpoint_state: dict[str, Any] = {
                    "orchestrator_type": "loop",
                    "iteration": iteration,
                    "last_result": ({"output": result.output, "metadata": result.metadata} if result else None),
                    "original_input": input,
                }
                await self._surface_suspension(exc, checkpoint_state, step_name=self._step.name)
                raise

            self._emitter.emit(
                WorkflowStepCompleteEvent(
                    trace_id=self._emitter.trace_id,
                    span_id=self._emitter.span_id,
                    parent_span_id=self._emitter.parent_span_id,
                    workflow_name=self._name,
                    step_name=self._step.name,
                    step_index=iteration - 1,
                    step_duration_ms=step_duration_ms,
                    step_output=str(result.output) if result.output is not None else None,
                    step_metadata=result.metadata,
                )
            )

            should_stop = self._condition(result, iteration)
            if asyncio.iscoroutine(should_stop):
                should_stop = await should_stop

            if should_stop:
                return StepResult(
                    output=result.output,
                    metadata={
                        **result.metadata,
                        "iterations": iteration,
                        "total_steps_executed": iteration,
                    },
                )

            current_input = result.output

        # max_iterations reached — return last result, don't raise
        return StepResult(
            output=result.output if result else current_input,
            metadata={
                **(result.metadata if result else {}),
                "iterations": self._max_iterations,
                "terminated": "iteration_limit",
                "total_steps_executed": self._max_iterations,
            },
        )
