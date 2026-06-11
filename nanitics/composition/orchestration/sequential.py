from __future__ import annotations

import time
from typing import Any

from nanitics.composition.durability.models import RunCheckpoint
from nanitics.composition.durability.store import CheckpointStore, StepCheckpointSink
from nanitics.composition.durability.suspension import SuspendExecution
from nanitics.composition.orchestration.protocol import Step, StepResult, _sum_usage
from nanitics.composition.orchestration.workflow import Workflow, _AgentStepCheckpointSink
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import (
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
        step_checkpoints: bool = False,
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
            step_checkpoints=step_checkpoints,
        )
        self._steps = steps

    @staticmethod
    def _serialize_results(intermediate_results: dict[str, StepResult]) -> dict[str, Any]:
        """Serialize completed step results for a checkpoint state dict.

        Shared by the suspend path and the step-level cursor checkpoint so the
        two produce an identical ``completed_results`` shape — the resume branch
        reconstructs ``StepResult``s from it the same way for both.
        """
        return {
            k: {
                "output": v.output,
                "metadata": v.metadata,
                "usage": v.usage.model_dump() if v.usage is not None else None,
            }
            for k, v in intermediate_results.items()
        }

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
        from nanitics.composition.orchestration.adapters import WorkflowStep

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
            # A step/crash cursor checkpoint can point one past the last step
            # (interrupted after the final step completed). There is no step to
            # name in that case; the loop below simply finalizes.
            resumed_step_name = self._steps[start_index].name if start_index < len(self._steps) else None
            self._emit_resumed(resume_from, resumed_step_name)

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
                child_resume: RunCheckpoint | None = None
                agent_checkpoint: dict[str, Any] | None = None
                if resume_from is not None and index == start_index:
                    if isinstance(step, WorkflowStep):
                        # Present only for a HITL suspension whose suspended step
                        # is this nested workflow. A step/crash cursor checkpoint
                        # points at a not-yet-started step, so there is no nested
                        # frame and the workflow runs fresh.
                        nested = resume_from.state.get("nested_checkpoint")
                        if nested is not None:
                            child_resume = self._child_checkpoint(resume_from, nested)
                    else:
                        candidate = resume_from.state.get("agent_checkpoint")
                        if candidate:
                            agent_checkpoint = candidate
                # Step-level durability: hand the bound agent a sink that writes
                # an orchestration-shaped cursor (pointing at *this* step, with
                # the agent snapshot under ``agent_checkpoint``) after each
                # completed tool batch, so a mid-agent crash resumes through the
                # branch above without re-firing completed tools. The sink is
                # only consumed by agent steps; other step types ignore it.
                checkpoint_sink: StepCheckpointSink | None = None
                if self._step_checkpoints and self._checkpoint_store is not None:
                    checkpoint_sink = _AgentStepCheckpointSink(
                        self,
                        step_path_prefix=f"sequential#{index}:{step.name}",
                        cursor_state_base={
                            "orchestrator_type": "sequential",
                            "suspended_step_index": index,
                            "completed_results": self._serialize_results(intermediate_results),
                            "last_output": current_input,
                            "original_input": input,
                        },
                    )
                bound_step = self._bind_step(
                    step,
                    agent_checkpoint=agent_checkpoint,
                    checkpoint_sink=checkpoint_sink,
                )
                with self._emitter.span(step.name):
                    step_start = time.monotonic()
                    if child_resume is not None:
                        result = await self._execute_resumable(bound_step, current_input, child_resume)
                    else:
                        result = await bound_step.execute(current_input)
                    step_duration_ms = int((time.monotonic() - step_start) * 1000)
            except SuspendExecution as exc:
                checkpoint_state: dict[str, Any] = {
                    "orchestrator_type": "sequential",
                    "suspended_step_index": index,
                    "completed_results": self._serialize_results(intermediate_results),
                    "last_output": current_input,
                    "original_input": input,
                }
                await self._surface_suspension(exc, checkpoint_state, step_name=step.name)
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

            # Step-level durability: record a cursor pointing at the *next* step
            # plus a journal entry for the result just produced, so an
            # interrupted run resumes here without re-executing this step.
            await self._save_step_checkpoint(
                {
                    "orchestrator_type": "sequential",
                    "suspended_step_index": index + 1,
                    "completed_results": self._serialize_results(intermediate_results),
                    "last_output": current_input,
                    "original_input": input,
                },
                step_path=f"sequential#{index}:{step.name}",
                result_payload={
                    "output": result.output,
                    "metadata": result.metadata,
                    "usage": result.usage.model_dump() if result.usage is not None else None,
                },
            )

        return StepResult(
            output=current_input,
            metadata={
                "intermediate_results": dict(intermediate_results),
                "total_steps_executed": len(self._steps),
            },
            usage=_sum_usage(r.usage for r in intermediate_results.values()),
        )
