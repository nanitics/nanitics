from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from nanitics.composition.durability.models import RunCheckpoint
from nanitics.composition.durability.store import CheckpointStore
from nanitics.composition.durability.suspension import SuspendExecution
from nanitics.composition.orchestration.protocol import FailurePolicy, Step, StepResult
from nanitics.composition.orchestration.workflow import Workflow
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import (
    WorkflowStepCompleteEvent,
    WorkflowStepDefinition,
)
from nanitics.infrastructure.observability.storage import PersistentTraceStore
from nanitics.safety.cancellation import CancellationToken


class Parallel(Workflow):
    """Executes all steps concurrently with the same input.

    By default, output is a list of each step's output in declaration order.
    Provide an ``aggregator`` to combine results differently.

    Args:
        name: Workflow identifier.
        steps: Steps to execute concurrently. Must contain at least one step.
        emitter: Event emitter for observability.
        aggregator: Optional function to combine step results into a single output.
            Receives the list of StepResult objects in declaration order.
        failure_policy: How to handle step failures. Default is ALL_OR_NOTHING.
        cancellation_token: Optional cooperative cancellation signal.
        checkpoint_store: Optional store for suspension checkpoints.
        run_id: Run identifier for checkpoint records.
        step_checkpoints: When ``True`` (and a ``checkpoint_store`` is set), a
            thin cursor checkpoint plus a journal record are written after each
            completed branch, so an interrupted run resumes via
            ``resume_interrupted`` without re-running branches that already
            finished. The completed-branch set is reconstructed from the journal
            (an order-independent union keyed by step path), so concurrent
            completions cannot clobber each other. Opt-in; defaults to ``False``.

    Raises:
        ValueError: If steps list is empty.
    """

    def __init__(
        self,
        *,
        name: str,
        steps: list[Step],
        emitter: EventEmitter,
        aggregator: Callable[[list[StepResult]], Any] | None = None,
        failure_policy: FailurePolicy = FailurePolicy.ALL_OR_NOTHING,
        cancellation_token: CancellationToken | None = None,
        checkpoint_store: CheckpointStore | None = None,
        run_id: str | None = None,
        trace_store: PersistentTraceStore | None = None,
        step_checkpoints: bool = False,
    ) -> None:
        if not steps:
            raise ValueError("Parallel requires at least one step")
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
        self._aggregator = aggregator
        self._failure_policy = failure_policy

    def _workflow_type(self) -> str:
        return "parallel"

    def _step_count(self) -> int:
        return len(self._steps)

    def _get_step_definitions(self) -> list[WorkflowStepDefinition]:
        defs = []
        for i, step in enumerate(self._steps):
            step_type, metadata = self._classify_step(step)
            defs.append(
                WorkflowStepDefinition(
                    name=step.name,
                    step_type=step_type,
                    index=i,
                    parallel_group="parallel",
                    metadata=metadata,
                )
            )
        return defs

    async def _run(self, input: Any, *, resume_from: RunCheckpoint | None = None) -> StepResult:
        from nanitics.composition.orchestration.adapters import WorkflowStep

        # HITL-suspend resume path: only re-execute the suspended branch. Gated
        # on ``suspension_info`` so a step/crash cursor (which has none) routes to
        # the journal-backed step-cursor resume below instead.
        if resume_from is not None and resume_from.suspension_info is not None:
            state = resume_from.state
            suspended_branch = state["suspended_branch"]
            completed_branches: dict[str, Any] = state["completed_branches"]
            self._emit_resumed(resume_from, suspended_branch)

            # Find the suspended step and its index
            suspended_index: int | None = None
            suspended_step: Step | None = None
            for i, step in enumerate(self._steps):
                if step.name == suspended_branch:
                    suspended_index = i
                    suspended_step = step
                    break
            assert suspended_step is not None
            assert suspended_index is not None

            # Re-execute suspended branch (recursing into a nested workflow if needed)
            child_resume: RunCheckpoint | None = None
            if isinstance(suspended_step, WorkflowStep):
                nested = state.get("nested_checkpoint")
                assert nested is not None
                child_resume = self._child_checkpoint(resume_from, nested)
            bound_step = self._bind_step(suspended_step)
            with self._emitter.span(suspended_step.name):
                if child_resume is not None:
                    result = await self._execute_resumable(bound_step, input, child_resume)
                else:
                    result = await bound_step.execute(input)
            self._emitter.emit(
                WorkflowStepCompleteEvent(
                    trace_id=self._emitter.trace_id,
                    span_id=self._emitter.span_id,
                    parent_span_id=self._emitter.parent_span_id,
                    workflow_name=self._name,
                    step_name=suspended_step.name,
                    step_index=suspended_index,
                    step_output=str(result.output) if result.output is not None else None,
                    step_metadata=result.metadata,
                )
            )

            # Merge results: reconstruct full ordered list
            all_results: list[StepResult] = []
            for step in self._steps:
                if step.name == suspended_branch:
                    all_results.append(result)
                elif step.name in completed_branches:
                    all_results.append(StepResult(output=completed_branches[step.name]))
            output = self._aggregator(all_results) if self._aggregator else [r.output for r in all_results]
            return StepResult(
                output=output,
                metadata={"total_steps_executed": len(all_results)},
            )

        # Normal execution path (a fresh run, or a step/crash cursor resume).
        #
        # Step-cursor resume: reconstruct the set of already-completed branches
        # from the append-only journal, NOT from the cursor's state. The journal
        # is keyed by ``(run_id, step_path)`` — an order-independent union — so a
        # branch that finished is replayed from its recorded result regardless of
        # the order or ``created_at`` of the concurrent cursor writes (the cursor
        # only carries ``original_input``). Pre-seeded completed branches are not
        # re-launched. On a fresh run the journal is empty and nothing is seeded.
        index_by_name = {step.name: i for i, step in enumerate(self._steps)}
        completed_results: dict[str, StepResult] = {}
        if resume_from is not None:
            assert self._checkpoint_store is not None
            self._emit_resumed(resume_from, None)
            journal = await self._checkpoint_store.load_journal(self._run_id)
            by_path = {rec.step_path: rec.result for rec in journal}
            for step in self._steps:
                payload = by_path.get(f"parallel#{index_by_name[step.name]}:{step.name}")
                if payload is not None:
                    completed_results[step.name] = self._restore_step_result(payload)

        tasks: dict[str, asyncio.Task[StepResult]] = {}

        async def _run_step(step: Step, index: int) -> StepResult:
            bound_step = self._bind_step(step)
            with self._emitter.span(step.name):
                step_start = time.monotonic()
                result = await bound_step.execute(input)
                step_duration_ms = int((time.monotonic() - step_start) * 1000)
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
            return result

        for index, step in enumerate(self._steps):
            if step.name in completed_results:
                continue  # restored from the journal on resume; do not re-launch
            task = asyncio.create_task(_run_step(step, index))
            tasks[step.name] = task

        # Wait for all tasks, watching for suspension
        suspended_name: str | None = None
        suspended_exc: SuspendExecution | None = None

        pending = set(tasks.values())
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                name = None
                for n, t in tasks.items():
                    if t is task:
                        name = n
                        break
                assert name is not None

                try:
                    result = task.result()
                except SuspendExecution as exc:
                    if suspended_name is None:
                        suspended_name = name
                        suspended_exc = exc
                    # Continue draining remaining tasks
                except Exception:
                    if self._failure_policy == FailurePolicy.ALL_OR_NOTHING:
                        for t in pending:
                            if not t.done():
                                t.cancel()
                        if pending:
                            await asyncio.gather(*pending, return_exceptions=True)
                        raise
                else:
                    completed_results[name] = result
                    # Step-level durability: journal this completed branch + a
                    # thin cursor (carrying only ``original_input``), so a resume
                    # reconstructs the completed set from the journal union.
                    await self._save_step_checkpoint(
                        {"orchestrator_type": "parallel", "original_input": input},
                        step_path=f"parallel#{index_by_name[name]}:{name}",
                        result_payload=self._serialize_step_result(result),
                    )

        if suspended_name is not None and suspended_exc is not None:
            checkpoint_state: dict[str, Any] = {
                "orchestrator_type": "parallel",
                "completed_branches": {k: v.output for k, v in completed_results.items()},
                "suspended_branch": suspended_name,
                "original_input": input,
            }
            await self._surface_suspension(suspended_exc, checkpoint_state, step_name=suspended_name)
            raise suspended_exc

        if self._failure_policy == FailurePolicy.BEST_EFFORT:
            failed_steps: list[str] = []
            successful_results: list[StepResult] = []
            for step in self._steps:
                if step.name in completed_results:
                    successful_results.append(completed_results[step.name])
                else:
                    failed_steps.append(step.name)
            output = (
                self._aggregator(successful_results) if self._aggregator else [r.output for r in successful_results]
            )
            metadata: dict[str, Any] = {"total_steps_executed": len(successful_results)}
            if failed_steps:
                metadata["failed_steps"] = failed_steps
            return StepResult(output=output, metadata=metadata)

        # All tasks completed normally (ALL_OR_NOTHING)
        ordered_results = [completed_results[step.name] for step in self._steps]
        output = self._aggregator(ordered_results) if self._aggregator else [r.output for r in ordered_results]
        return StepResult(output=output, metadata={"total_steps_executed": len(ordered_results)})
