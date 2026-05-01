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
    ExecutionSuspendedEvent,
    WorkflowStepCompleteEvent,
    WorkflowStepDefinition,
)
from nanitics.infrastructure.observability.storage import PersistentTraceStore
from nanitics.safety.cancellation import CancellationToken


class MapReduce(Workflow):
    """Splits input into items, maps a step over each concurrently, then reduces results.

    The ``splitter`` breaks the input into a list of items. Each item is processed
    by the ``step`` concurrently (up to ``max_concurrency``). The ``reducer``
    combines the list of StepResult objects into the final output.

    Result metadata includes ``total_items`` and ``total_steps_executed``.
    With BEST_EFFORT, failed items are tracked in ``metadata["failed_items"]``.

    Args:
        name: Workflow identifier.
        step: The step applied to each item.
        emitter: Event emitter for observability.
        splitter: Function that takes input and returns a list of items. Can be sync or async.
        reducer: Function that takes a list of StepResult and returns the combined output.
            Can be sync or async.
        max_concurrency: Maximum number of items processed simultaneously. None for unlimited.
        failure_policy: How to handle item failures. Default is ALL_OR_NOTHING.
        cancellation_token: Optional cooperative cancellation signal.
        checkpoint_store: Optional store for suspension checkpoints.
        run_id: Run identifier for checkpoint records.
    """

    def __init__(
        self,
        *,
        name: str,
        step: Step,
        emitter: EventEmitter,
        splitter: Callable[[Any], Any],
        reducer: Callable[[list[StepResult]], Any],
        max_concurrency: int | None = None,
        failure_policy: FailurePolicy = FailurePolicy.ALL_OR_NOTHING,
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
        self._splitter = splitter
        self._reducer = reducer
        self._max_concurrency = max_concurrency
        self._failure_policy = failure_policy

    def _workflow_type(self) -> str:
        return "map_reduce"

    def _step_count(self) -> int:
        return 0

    def _get_step_definitions(self) -> list[WorkflowStepDefinition]:
        step_type, metadata = self._classify_step(self._step)
        return [
            WorkflowStepDefinition(name=f"{self._step.name}-map", step_type=step_type, index=0, metadata=metadata),
            WorkflowStepDefinition(name="reduce", step_type="function", index=1),
        ]

    async def _run(self, input: Any, *, resume_from: RunCheckpoint | None = None) -> StepResult:
        # Resume path: skip splitting, use stored items
        if resume_from is not None:
            state = resume_from.state
            items = state["split_items"]
            suspended_item_index = state["suspended_item_index"]
            completed_items: dict[int, StepResult] = {
                int(k): StepResult(output=v) for k, v in state["completed_items"].items()
            }
            self._emit_resumed(resume_from, f"{self._step.name}-{suspended_item_index}")

            # Execute suspended + remaining items
            remaining_indices = [i for i in range(len(items)) if i not in completed_items]
            result = await self._execute_items_with_suspension(
                items,
                remaining_indices,
                input,
            )
            if isinstance(result, StepResult):
                return result  # pragma: no cover — defensive; suspension always raises

            completed_map, failed_items = result

            # Merge all results
            all_results: list[StepResult] = []
            for i in range(len(items)):
                if i in completed_items:
                    all_results.append(completed_items[i])
                elif i in completed_map:
                    all_results.append(completed_map[i])

            reduced = self._reducer(all_results)
            if asyncio.iscoroutine(reduced):
                reduced = await reduced
            metadata: dict[str, Any] = {
                "total_items": len(items),
                "total_steps_executed": len(all_results),
            }
            if self._failure_policy == FailurePolicy.BEST_EFFORT and failed_items:
                metadata["failed_items"] = failed_items
            if self._cancellation_token and self._cancellation_token.is_cancelled:
                metadata["terminated"] = "cancelled"
            return StepResult(output=reduced, metadata=metadata)

        # Normal path: Split
        items = self._splitter(input)
        if asyncio.iscoroutine(items):
            items = await items

        if not items:
            reduced = self._reducer([])
            if asyncio.iscoroutine(reduced):
                reduced = await reduced
            return StepResult(
                output=reduced,
                metadata={"total_items": 0, "total_steps_executed": 0},
            )

        all_indices = list(range(len(items)))
        map_result = await self._execute_items_with_suspension(
            items,
            all_indices,
            input,
        )
        if isinstance(map_result, StepResult):
            return map_result  # pragma: no cover — defensive; suspension always raises

        completed_map, failed_items = map_result
        ordered_results = [completed_map[i] for i in range(len(items)) if i in completed_map]

        # Reduce
        reduced = self._reducer(ordered_results)
        if asyncio.iscoroutine(reduced):
            reduced = await reduced

        metadata = {
            "total_items": len(items),
            "total_steps_executed": len(ordered_results),
        }
        if self._failure_policy == FailurePolicy.BEST_EFFORT and failed_items:
            metadata["failed_items"] = failed_items
        if self._cancellation_token and self._cancellation_token.is_cancelled:
            metadata["terminated"] = "cancelled"

        return StepResult(output=reduced, metadata=metadata)

    async def _execute_items_with_suspension(
        self,
        items: list[Any],
        indices: list[int],
        original_input: Any,
    ) -> tuple[dict[int, StepResult], list[int]] | StepResult:
        """Execute map items.

        Returns (completed_dict, failed_indices) or a StepResult if suspended.
        """
        semaphore = asyncio.Semaphore(self._max_concurrency) if self._max_concurrency is not None else None

        async def _run_item(item: Any, index: int) -> tuple[int, StepResult]:
            if semaphore:
                async with semaphore:
                    r = await self._execute_item(item, index)
                    return (index, r)
            r = await self._execute_item(item, index)
            return (index, r)

        tasks: dict[int, asyncio.Task[tuple[int, StepResult]]] = {}
        for i in indices:
            task = asyncio.create_task(_run_item(items[i], i))
            tasks[i] = task

        completed: dict[int, StepResult] = {}
        failed_items: list[int] = []
        suspended_index: int | None = None
        suspended_exc: SuspendExecution | None = None

        pending = set(tasks.values())
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                idx = None
                for i, t in tasks.items():
                    if t is task:
                        idx = i
                        break
                assert idx is not None

                try:
                    _, result = task.result()
                    if result.metadata.get("terminated") == "cancelled":
                        failed_items.append(idx)
                    else:
                        completed[idx] = result
                except SuspendExecution as exc:
                    if suspended_index is None:
                        suspended_index = idx
                        suspended_exc = exc
                except BaseException:
                    if self._failure_policy == FailurePolicy.ALL_OR_NOTHING:
                        for t in pending:
                            if not t.done():
                                t.cancel()
                        if pending:
                            await asyncio.gather(*pending, return_exceptions=True)
                        raise
                    failed_items.append(idx)

        if suspended_index is not None and suspended_exc is not None:
            if self._checkpoint_store:
                checkpoint_state: dict[str, Any] = {
                    "orchestrator_type": "mapreduce",
                    "completed_items": {str(k): v.output for k, v in completed.items()},
                    "suspended_item_index": suspended_index,
                    "split_items": items,
                    "original_input": original_input,
                }
                if suspended_exc.checkpoint_data:
                    checkpoint_state["agent_checkpoint"] = suspended_exc.checkpoint_data
                checkpoint = await self._save_checkpoint(suspended_exc, checkpoint_state)
                self._emitter.emit(
                    ExecutionSuspendedEvent(
                        trace_id=self._emitter.trace_id,
                        span_id=self._emitter.span_id,
                        parent_span_id=self._emitter.parent_span_id,
                        suspension_id=suspended_exc.suspension_info.suspension_id,
                        suspension_type="hitl",
                        checkpoint_id=checkpoint.checkpoint_id,
                        step_name=f"{self._step.name}-{suspended_index}",
                        agent_name=suspended_exc.suspension_info.agent_name,
                    )
                )
            raise suspended_exc

        return (completed, failed_items)

    async def _execute_item(self, item: Any, index: int) -> StepResult:
        span_name = f"{self._step.name}-{index}"
        with self._emitter.span(span_name):
            bound_step = self._bind_step(self._step)
            if self._cancellation_token and self._cancellation_token.is_cancelled:
                return StepResult(output=None, metadata={"terminated": "cancelled"})
            step_start = time.monotonic()
            result = await bound_step.execute(item)
            step_duration_ms = int((time.monotonic() - step_start) * 1000)
        self._emitter.emit(
            WorkflowStepCompleteEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                workflow_name=self._name,
                step_name=self._step.name,
                step_index=index,
                step_duration_ms=step_duration_ms,
                step_output=str(result.output) if result.output is not None else None,
            )
        )
        return result
