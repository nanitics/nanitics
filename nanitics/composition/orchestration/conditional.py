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
    ExecutionSuspendedEvent,
    WorkflowStepCompleteEvent,
    WorkflowStepDefinition,
)
from nanitics.infrastructure.observability.storage import PersistentTraceStore
from nanitics.safety.cancellation import CancellationToken


class Conditional(Workflow):
    """Routes input to one of multiple branches based on a router function.

    The router receives the input and returns a branch name. The corresponding
    step is executed. If the router returns an unknown branch and a ``default``
    step is provided, the default is used. Otherwise, a ``ValueError`` is raised.

    Result metadata includes ``selected_branch`` with the chosen branch name.

    Args:
        name: Workflow identifier.
        router: Function that takes input and returns a branch name. Can be sync or async.
        branches: Mapping of branch names to steps.
        default: Optional fallback step for unknown branch names.
        emitter: Event emitter for observability.
        cancellation_token: Optional cooperative cancellation signal.
        checkpoint_store: Optional store for suspension checkpoints.
        run_id: Run identifier for checkpoint records.

    Raises:
        ValueError: If branches dict is empty.
    """

    def __init__(
        self,
        *,
        name: str,
        router: Callable[[Any], str | Awaitable[str]],
        branches: dict[str, Step],
        default: Step | None = None,
        emitter: EventEmitter,
        cancellation_token: CancellationToken | None = None,
        checkpoint_store: CheckpointStore | None = None,
        run_id: str | None = None,
        trace_store: PersistentTraceStore | None = None,
    ) -> None:
        if not branches:
            raise ValueError("Conditional requires at least one branch")
        super().__init__(
            name=name,
            emitter=emitter,
            cancellation_token=cancellation_token,
            checkpoint_store=checkpoint_store,
            run_id=run_id,
            trace_store=trace_store,
        )
        self._router = router
        self._branches = branches
        self._default = default

    def _workflow_type(self) -> str:
        return "conditional"

    def _step_count(self) -> int:
        return len(self._branches)

    def _get_step_definitions(self) -> list[WorkflowStepDefinition]:
        defs = []
        for i, (branch_name, step) in enumerate(self._branches.items()):
            step_type, metadata = self._classify_step(step)
            metadata["branch"] = branch_name
            defs.append(
                WorkflowStepDefinition(
                    name=step.name,
                    step_type=step_type,
                    index=i,
                    metadata=metadata,
                )
            )
        if self._default is not None:
            step_type, metadata = self._classify_step(self._default)
            metadata["branch"] = "default"
            defs.append(
                WorkflowStepDefinition(
                    name=self._default.name,
                    step_type=step_type,
                    index=len(self._branches),
                    metadata=metadata,
                )
            )
        return defs

    async def _run(self, input: Any, *, resume_from: RunCheckpoint | None = None) -> StepResult:
        if resume_from is not None:
            # Resume: skip routing, directly execute the stored branch
            state = resume_from.state
            branch_name = state["selected_branch"]
            step = self._branches.get(branch_name)
            if step is None and self._default is not None:
                step = self._default
            assert step is not None
            self._emit_resumed(resume_from, branch_name)
        else:
            branch_name = self._router(input)
            if asyncio.iscoroutine(branch_name):
                branch_name = await branch_name

            step = self._branches.get(branch_name)
            if step is None:
                if self._default is not None:
                    step = self._default
                    branch_name = f"default({branch_name})"
                else:
                    available = list(self._branches.keys())
                    raise ValueError(f"Router returned unknown branch '{branch_name}'. Available branches: {available}")

        try:
            bound_step = self._bind_step(step)
            with self._emitter.span(step.name):
                step_start = time.monotonic()
                result = await bound_step.execute(input)
                step_duration_ms = int((time.monotonic() - step_start) * 1000)
        except SuspendExecution as exc:
            if self._checkpoint_store:
                checkpoint_state: dict[str, Any] = {
                    "orchestrator_type": "conditional",
                    "selected_branch": branch_name,
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

        self._emitter.emit(
            WorkflowStepCompleteEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                workflow_name=self._name,
                step_name=step.name,
                step_index=0,
                step_duration_ms=step_duration_ms,
                step_output=str(result.output) if result.output is not None else None,
            )
        )

        return StepResult(
            output=result.output,
            metadata={
                **result.metadata,
                "selected_branch": branch_name,
                "total_steps_executed": 1,
            },
        )
