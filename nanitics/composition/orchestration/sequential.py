from __future__ import annotations

import time
from typing import Any

from nanitics.composition.durability.models import RunCheckpoint
from nanitics.composition.durability.store import CheckpointStore
from nanitics.composition.durability.suspension import SuspendExecution
from nanitics.composition.orchestration.protocol import Step, StepResult, _sum_usage
from nanitics.composition.orchestration.workflow import Workflow
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import (
    ExecutionSuspendedEvent,
    Usage,
    WorkflowStepCompleteEvent,
    WorkflowStepDefinition,
)
from nanitics.infrastructure.observability.storage import PersistentTraceStore
from nanitics.safety.cancellation import CancellationToken


class Sequential(Workflow):
    """Executes steps one after another, chaining each step's output as the next step's input.

    The final result contains the last step's output and metadata with
    ``intermediate_results`` (step name → ``StepResult``) and ``total_steps_executed``.
    The returned ``StepResult.usage`` is the aggregated sum across every
    sub-step's ``usage`` (``None`` only when every sub-step contributed
    ``None``). On a cancellation mid-flight, the partial aggregate of the
    completed steps is returned. On resume from a checkpoint, sub-step
    usages are reconstructed from the checkpoint state and folded into the
    final sum.

    Args:
        name: Workflow identifier.
        steps: Ordered list of steps to execute. Must contain at least one step.
        emitter: Event emitter for observability.
        cancellation_token: Optional cooperative cancellation signal.
        checkpoint_store: Optional store for suspension checkpoints.
        run_id: Run identifier for checkpoint records.

    Raises:
        ValueError: If steps list is empty.
    """

    def __init__(
        self,
        *,
        name: str,
        steps: list[Step],
        emitter: EventEmitter,
        cancellation_token: CancellationToken | None = None,
        checkpoint_store: CheckpointStore | None = None,
        run_id: str | None = None,
        trace_store: PersistentTraceStore | None = None,
    ) -> None:
        if not steps:
            raise ValueError("Sequential requires at least one step")
        super().__init__(
            name=name,
            emitter=emitter,
            cancellation_token=cancellation_token,
            checkpoint_store=checkpoint_store,
            run_id=run_id,
            trace_store=trace_store,
        )
        self._steps = steps

    def _workflow_type(self) -> str:
        return "sequential"

    def _step_count(self) -> int:
        return len(self._steps)

    def _get_step_definitions(self) -> list[WorkflowStepDefinition]:
        defs = []
        for i, step in enumerate(self._steps):
            step_type, metadata = self._classify_step(step)
            defs.append(WorkflowStepDefinition(name=step.name, step_type=step_type, index=i, metadata=metadata))
        return defs

    async def _run(self, input: Any, *, resume_from: RunCheckpoint | None = None) -> StepResult:
        current_input = input
        intermediate_results: dict[str, StepResult] = {}
        start_index = 0

        if resume_from is not None:
            state = resume_from.state
            start_index = state["suspended_step_index"]
            current_input = state["last_output"]
            for step_name, d in state["completed_results"].items():
                usage_dict = d.get("usage")
                restored_usage = Usage.model_validate(usage_dict) if usage_dict is not None else None
                intermediate_results[step_name] = StepResult(
                    output=d["output"], metadata=d["metadata"], usage=restored_usage
                )
            self._emit_resumed(resume_from, self._steps[start_index].name)

        for index in range(start_index, len(self._steps)):
            step = self._steps[index]

            if self._cancellation_token and self._cancellation_token.is_cancelled:
                last_result = (
                    intermediate_results[list(intermediate_results.keys())[-1]]
                    if intermediate_results
                    else StepResult(output=current_input)
                )
                return StepResult(
                    output=last_result.output,
                    metadata={
                        **last_result.metadata,
                        "intermediate_results": dict(intermediate_results),
                        "terminated": "cancelled",
                        "total_steps_executed": index,
                    },
                    usage=_sum_usage(r.usage for r in intermediate_results.values()),
                )

            try:
                agent_checkpoint: dict[str, Any] | None = None
                if resume_from is not None and index == start_index:
                    candidate = resume_from.state.get("agent_checkpoint")
                    if candidate:
                        agent_checkpoint = candidate
                bound_step = self._bind_step(step, agent_checkpoint=agent_checkpoint)
                with self._emitter.span(step.name):
                    step_start = time.monotonic()
                    result = await bound_step.execute(current_input)
                    step_duration_ms = int((time.monotonic() - step_start) * 1000)
            except SuspendExecution as exc:
                if self._checkpoint_store:
                    checkpoint_state: dict[str, Any] = {
                        "orchestrator_type": "sequential",
                        "suspended_step_index": index,
                        "completed_results": {
                            k: {
                                "output": v.output,
                                "metadata": v.metadata,
                                "usage": v.usage.model_dump() if v.usage is not None else None,
                            }
                            for k, v in intermediate_results.items()
                        },
                        "last_output": current_input,
                        "original_input": input,
                    }
                    if exc.checkpoint_data:
                        checkpoint_state["agent_checkpoint"] = exc.checkpoint_data
                    checkpoint = await self._save_checkpoint(exc, checkpoint_state)
                    self._emitter.emit(
                        ExecutionSuspendedEvent(
                            trace_id=self._emitter.trace_id,
                            span_id=self._emitter.span_id,
                            parent_span_id=self._emitter.parent_span_id,
                            suspension_id=exc.suspension_info.suspension_id,
                            suspension_type="hitl",
                            checkpoint_id=checkpoint.checkpoint_id,
                            step_name=step.name,
                            agent_name=exc.suspension_info.agent_name,
                        )
                    )
                raise

            intermediate_results[step.name] = result
            current_input = result.output

            self._emitter.emit(
                WorkflowStepCompleteEvent(
                    trace_id=self._emitter.trace_id,
                    span_id=self._emitter.span_id,
                    parent_span_id=self._emitter.parent_span_id,
                    workflow_name=self._name,
                    step_name=step.name,
                    step_index=index,
                    step_duration_ms=step_duration_ms,
                    step_output=str(result.output) if result.output is not None else None,
                    step_metadata=result.metadata,
                )
            )

        return StepResult(
            output=current_input,
            metadata={
                "intermediate_results": dict(intermediate_results),
                "total_steps_executed": len(self._steps),
            },
            usage=_sum_usage(r.usage for r in intermediate_results.values()),
        )
