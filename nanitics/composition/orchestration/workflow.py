from __future__ import annotations

import time
from abc import ABC, abstractmethod
from contextvars import ContextVar
from typing import Any, cast
from uuid import uuid4

from pydantic import BaseModel

from nanitics.composition.durability.models import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointVersionError,
    RunCheckpoint,
)
from nanitics.composition.durability.store import CheckpointStore
from nanitics.composition.durability.suspension import SuspendExecution
from nanitics.composition.orchestration.protocol import Step, StepResult
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import (
    CheckpointSavedEvent,
    ExecutionResumedEvent,
    ExecutionSuspendedEvent,
    RunCompleteEvent,
    RunFailedEvent,
    RunStartEvent,
    RunSuspendedEvent,
    WorkflowCompleteEvent,
    WorkflowErrorEvent,
    WorkflowStartEvent,
    WorkflowStepDefinition,
    WorkflowStructureEvent,
)
from nanitics.infrastructure.observability.storage import PersistentTraceStore
from nanitics.safety.cancellation import CancellationToken


class Workflow(ABC):
    """Abstract base for all workflow orchestration patterns.

    Provides common infrastructure: event emission, cancellation, checkpoint-based
    suspension and resumption. Subclasses implement ``_run`` with pattern-specific
    execution logic.

    Args:
        name: Workflow identifier used in events and trace spans.
        emitter: Event emitter for observability.
        cancellation_token: Optional cooperative cancellation signal.
        checkpoint_store: Optional store for persisting suspension checkpoints.
        run_id: Identifier for the run, used in checkpoint records.
            Auto-generated if not provided.
        trace_store: Optional persistent trace store for run registration.
    """

    def __init__(
        self,
        *,
        name: str,
        emitter: EventEmitter,
        cancellation_token: CancellationToken | None = None,
        checkpoint_store: CheckpointStore | None = None,
        run_id: str | None = None,
        trace_store: PersistentTraceStore | None = None,
    ) -> None:
        self._name = name
        self._default_emitter = emitter
        self._emitter_var: ContextVar[EventEmitter] = ContextVar(f"workflow_emitter_{id(self)}")
        self._cancellation_token = cancellation_token
        self._checkpoint_store = checkpoint_store
        self._run_id = run_id or str(uuid4())
        self._trace_store = trace_store

    @property
    def name(self) -> str:
        return self._name

    @property
    def _emitter(self) -> EventEmitter:
        """Emitter active for the current asyncio task.

        Resolves to the per-call child emitter when the workflow is
        running under a bound handle produced by :meth:`bind`; otherwise
        to the default emitter supplied at construction.
        """
        return self._emitter_var.get(self._default_emitter)

    def bind(self, parent_emitter: EventEmitter) -> BoundWorkflow:
        """Return a non-mutating per-invocation binding of this workflow.

        Symmetric to :meth:`Agent.bind`. Creates a child emitter from
        ``parent_emitter`` in the calling task and returns a
        :class:`BoundWorkflow` handle that drives ``execute`` without
        mutating the workflow.
        """
        return BoundWorkflow(self, parent_emitter.create_child())

    async def execute(
        self,
        input: Any,
        *,
        resume_from: RunCheckpoint | None = None,
    ) -> StepResult:
        """Execute the workflow.

        Args:
            input: Initial input data for the workflow.
            resume_from: Optional checkpoint to resume from. Skips already-completed
                steps and re-executes from the suspension point.

        Returns:
            A StepResult with the workflow's final output and metadata.

        Raises:
            WorkflowCancelledError: If the cancellation token is set before execution starts.
            SuspendExecution: If a step suspends (e.g., waiting for human input).
            CheckpointVersionError: If the checkpoint schema version doesn't match.
        """
        if resume_from is not None:
            self._validate_checkpoint_version(resume_from)
        with self._emitter.span(self._name):
            self._emit_start()
            self._emit_structure()
            if self._cancellation_token and self._cancellation_token.is_cancelled:
                raise WorkflowCancelledError(f"Workflow '{self._name}' cancelled before start")
            await self._register_run()
            start_time = time.monotonic()
            try:
                result = await self._run(input, resume_from=resume_from)
                self._emit_complete(result)
                await self._complete_run(start_time)
                return result
            except SuspendExecution as exc:
                await self._suspend_run(exc)
                raise
            except Exception as e:
                self._emit_error(e)
                await self._fail_run(e)
                raise

    @abstractmethod
    async def _run(
        self,
        input: Any,
        *,
        resume_from: RunCheckpoint | None = None,
    ) -> StepResult: ...

    @staticmethod
    def _normalize_for_serialization(value: Any) -> Any:
        """Recursively convert Pydantic BaseModel instances to plain dicts.

        Ensures checkpoint state is JSON-serializable by normalizing any
        BaseModel found in the state tree via ``.model_dump(mode="json")``.
        """
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return {k: Workflow._normalize_for_serialization(v) for k, v in value.items()}
        if isinstance(value, list):
            return [Workflow._normalize_for_serialization(item) for item in value]
        return value

    async def _save_checkpoint(
        self,
        suspension: SuspendExecution,
        state: dict[str, Any],
    ) -> RunCheckpoint:
        state = self._normalize_for_serialization(state)
        suspension_info = suspension.suspension_info
        checkpoint = RunCheckpoint(
            run_id=self._run_id,
            checkpoint_type="orchestration",
            state=state,
            suspension_info=suspension_info,
        )
        if self._checkpoint_store:
            await self._checkpoint_store.save(checkpoint)
            self._emitter.emit(
                CheckpointSavedEvent(
                    trace_id=self._emitter.trace_id,
                    span_id=self._emitter.span_id,
                    parent_span_id=self._emitter.parent_span_id,
                    checkpoint_id=checkpoint.checkpoint_id,
                    checkpoint_type="orchestration",
                    run_id=self._run_id,
                )
            )
        return checkpoint

    async def _surface_suspension(
        self,
        exc: SuspendExecution,
        checkpoint_state: dict[str, Any],
        *,
        step_name: str,
    ) -> None:
        """Fold child resume-state into this frame, surface it up, persist at root.

        Common to every orchestrator's suspend path. Folds the leaf agent's
        resume state (``exc.checkpoint_data``) or a nested workflow's frame
        (``exc.orchestration_state``) into ``checkpoint_state``, then sets
        ``checkpoint_state`` as this frame's ``orchestration_state`` so the
        parent can embed it under ``nested_checkpoint``. The consumed carriers
        are cleared so an ancestor does not re-consume them. The checkpoint is
        persisted and :class:`ExecutionSuspendedEvent` emitted only when this
        frame owns a ``checkpoint_store`` — i.e. only at the durable root.
        """
        if exc.checkpoint_data:
            checkpoint_state["agent_checkpoint"] = exc.checkpoint_data
        if exc.orchestration_state is not None:
            checkpoint_state["nested_checkpoint"] = exc.orchestration_state
        exc.checkpoint_data = None
        exc.orchestration_state = checkpoint_state
        if self._checkpoint_store:
            checkpoint = await self._save_checkpoint(exc, checkpoint_state)
            self._emitter.emit(
                ExecutionSuspendedEvent(
                    trace_id=self._emitter.trace_id,
                    span_id=self._emitter.span_id,
                    parent_span_id=self._emitter.parent_span_id,
                    suspension_id=exc.suspension_info.suspension_id,
                    suspension_type="hitl",
                    checkpoint_id=checkpoint.checkpoint_id,
                    step_name=step_name,
                    agent_name=exc.suspension_info.agent_name,
                )
            )

    def _emit_resumed(self, checkpoint: RunCheckpoint, step_name: str | None = None) -> None:
        self._emitter.emit(
            ExecutionResumedEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                checkpoint_id=checkpoint.checkpoint_id,
                suspension_id=checkpoint.suspension_info.suspension_id,
                resumed_from_step=step_name,
            )
        )

    @staticmethod
    async def _execute_resumable(bound_step: Step, input: Any, resume_from: RunCheckpoint) -> StepResult:
        """Execute a bound nested-workflow step on its resume path.

        Callers guard on ``isinstance(step, WorkflowStep)``, so the bound
        step is always a :class:`_BoundWorkflowStep`. The cast narrows the
        ``Step`` protocol to the resume-aware ``execute`` here rather than
        widening the base protocol — non-workflow step types keep the plain
        ``execute(self, input)`` contract.
        """
        resumable = cast("_BoundWorkflowStep", bound_step)
        return await resumable.execute(input, resume_from=resume_from)

    @staticmethod
    def _child_checkpoint(parent: RunCheckpoint, nested_state: dict[str, Any]) -> RunCheckpoint:
        """Reconstruct a nested workflow's checkpoint from an embedded frame.

        On resume, an orchestrator whose suspended step is a
        :class:`WorkflowStep` lifts the child frame stored under its own
        ``state["nested_checkpoint"]`` back into a :class:`RunCheckpoint`
        and threads it down as the child's ``resume_from``. The parent's
        ``run_id``, ``schema_version``, and ``suspension_info`` carry
        through; ``checkpoint_id`` / ``created_at`` default fresh and are
        not load-bearing on the resume read path.
        """
        return RunCheckpoint(
            run_id=parent.run_id,
            checkpoint_type="orchestration",
            schema_version=parent.schema_version,
            state=nested_state,
            suspension_info=parent.suspension_info,
        )

    @staticmethod
    def _validate_checkpoint_version(checkpoint: RunCheckpoint) -> None:
        if checkpoint.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointVersionError(
                f"Checkpoint schema version {checkpoint.schema_version} "
                f"does not match expected version {CHECKPOINT_SCHEMA_VERSION}",
                expected_version=CHECKPOINT_SCHEMA_VERSION,
                actual_version=checkpoint.schema_version,
            )

    def _bind_step(
        self,
        step: Step,
        *,
        agent_checkpoint: dict[str, Any] | None = None,
    ) -> Step:
        """Return a per-call Step bound to this workflow's emitter.

        Non-mutating. ``AgentStep`` and ``HandoffStep`` become
        :class:`_BoundAgentStep` wrappers driving the agent through a
        ``BoundAgent`` handle; ``WorkflowStep`` becomes a
        :class:`_BoundWorkflowStep` driving a ``BoundWorkflow``; other
        step types are returned unchanged. Safe to call concurrently on
        the same workflow — the bound handles carry per-task child
        emitters so no ``self._emitter`` mutation is required.

        Args:
            step: The step to bind.
            agent_checkpoint: Optional per-agent resume-state dict
                persisted by the orchestrator under
                ``checkpoint.state["agent_checkpoint"]``. When non-None
                and ``step`` is an :class:`AgentStep`, the returned bound
                wrapper consumes the checkpoint exactly once on its
                first ``execute`` call (see :class:`_BoundAgentStep`).
                Ignored for other step types — resume state for
                non-agent steps is orchestrator-level, not step-local.
        """
        from nanitics.composition.multi_agent.handoff import HandoffStep
        from nanitics.composition.orchestration.adapters import AgentStep, WorkflowStep

        if isinstance(step, AgentStep):
            return _BoundAgentStep(
                step,
                step.agent.bind(self._emitter),
                agent_checkpoint=agent_checkpoint,
            )
        if isinstance(step, WorkflowStep):
            return _BoundWorkflowStep(step, step.workflow.bind(self._emitter))
        if isinstance(step, HandoffStep):
            return _BoundHandoffStep(step, step.agent.bind(self._emitter), self._emitter)
        return step

    async def _execute_with_emitter(
        self,
        input: Any,
        emitter: EventEmitter,
        *,
        resume_from: RunCheckpoint | None = None,
    ) -> StepResult:
        """Execute the workflow under a per-call emitter.

        Drives :meth:`execute` with ``emitter`` installed on the
        workflow's per-task ``ContextVar`` for the duration of the call.
        Does not mutate ``self``.
        """
        token = self._emitter_var.set(emitter)
        try:
            return await self.execute(input, resume_from=resume_from)
        finally:
            self._emitter_var.reset(token)

    @abstractmethod
    def _workflow_type(self) -> str: ...

    @abstractmethod
    def _step_count(self) -> int: ...

    @abstractmethod
    def _get_step_definitions(self) -> list[WorkflowStepDefinition]: ...

    @staticmethod
    def _classify_step(step: Step) -> tuple[str, dict[str, Any]]:
        """Classify a step's type and extract metadata."""
        from nanitics.composition.orchestration.adapters import AgentStep, FunctionStep, WorkflowStep

        if isinstance(step, AgentStep):
            return "agent", {"agent_name": step.agent.name}
        if isinstance(step, WorkflowStep):
            return "workflow", {"workflow_name": step.workflow.name, "workflow_type": step.workflow._workflow_type()}
        if isinstance(step, FunctionStep):
            return "function", {}
        return "custom", {}

    def _emit_structure(self) -> None:
        self._emitter.emit(
            WorkflowStructureEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                workflow_name=self._name,
                workflow_type=self._workflow_type(),
                steps=self._get_step_definitions(),
            )
        )

    def _emit_start(self) -> None:
        self._emitter.emit(
            WorkflowStartEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                workflow_name=self._name,
                workflow_type=self._workflow_type(),
                step_count=self._step_count(),
            )
        )

    def _emit_complete(self, result: StepResult) -> None:
        total = result.metadata.get("total_steps_executed", self._step_count())
        self._emitter.emit(
            WorkflowCompleteEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                workflow_name=self._name,
                workflow_type=self._workflow_type(),
                total_steps_executed=total,
            )
        )

    def _emit_error(self, error: Exception) -> None:
        self._emitter.emit(
            WorkflowErrorEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                workflow_name=self._name,
                workflow_type=self._workflow_type(),
                error_type=type(error).__name__,
                error_message=str(error),
                failed_step=None,
            )
        )

    # --- Run lifecycle ---

    async def _register_run(self) -> None:
        self._emitter.emit(
            RunStartEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                run_id=self._run_id,
                workflow_name=self._name,
            )
        )
        if self._trace_store is not None:
            await self._trace_store.register_run(
                self._run_id,
                self._emitter.trace_id,
                {"workflow_name": self._name, "workflow_type": self._workflow_type()},
            )

    async def _complete_run(self, start_time: float) -> None:
        duration_ms = int((time.monotonic() - start_time) * 1000)
        self._emitter.emit(
            RunCompleteEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                run_id=self._run_id,
                duration_ms=duration_ms,
            )
        )
        if self._trace_store is not None:
            await self._trace_store.update_run_status(self._run_id, "completed")

    async def _fail_run(self, error: Exception) -> None:
        self._emitter.emit(
            RunFailedEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                run_id=self._run_id,
                error_type=type(error).__name__,
                error_message=str(error),
            )
        )
        if self._trace_store is not None:
            await self._trace_store.update_run_status(self._run_id, "failed", error=str(error))

    async def _suspend_run(self, exc: SuspendExecution) -> None:
        self._emitter.emit(
            RunSuspendedEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                run_id=self._run_id,
                suspension_id=exc.suspension_info.suspension_id,
            )
        )
        if self._trace_store is not None:
            await self._trace_store.update_run_status(self._run_id, "suspended")


# Re-export for isinstance checks
assert issubclass(Workflow, ABC)


class WorkflowCancelledError(Exception):
    """Raised when a workflow is cancelled before execution starts."""


class BoundWorkflow:
    """A per-invocation binding of a :class:`Workflow` to a parent trace.

    Symmetric to :class:`nanitics.strategies.agents.bound.BoundAgent`. Holds a
    child emitter constructed in the calling task and drives
    :meth:`Workflow.execute` under it without mutating the workflow —
    safe to use concurrently on a shared workflow across tasks.
    """

    __slots__ = ("_emitter", "_workflow")

    def __init__(self, workflow: Workflow, emitter: EventEmitter) -> None:
        self._workflow = workflow
        self._emitter = emitter

    @property
    def workflow(self) -> Workflow:
        return self._workflow

    @property
    def emitter(self) -> EventEmitter:
        return self._emitter

    async def execute(self, input: Any, *, resume_from: RunCheckpoint | None = None) -> StepResult:
        return await self._workflow._execute_with_emitter(input, self._emitter, resume_from=resume_from)


class _BoundAgentStep:
    """Wraps an :class:`AgentStep` for per-call execution under a ``BoundAgent``.

    On a resume path, the orchestrator passes ``agent_checkpoint`` —
    the per-agent resume-state dict it persisted on the suspend path.
    The wrapper injects it into the underlying agent via
    ``Agent._set_resume_state`` on its first ``execute`` call and then
    clears its local copy ("consume-once"), so an intra-run retry would
    not re-inject stale state.

    .. deprecated:: 0.5.0
        Reading token usage from ``StepResult.metadata["usage"]`` (the
        dict mirror) is deprecated. Use the typed :attr:`StepResult.usage`
        field instead. The dict mirror is retained alongside the typed
        field for backwards compatibility and will be removed in 1.0.0.
        See ``docs/migrations/step-result-usage.md``.
    """

    def __init__(
        self,
        step: Step,
        bound: Any,
        *,
        agent_checkpoint: dict[str, Any] | None = None,
    ) -> None:
        self._step = step
        self._bound = bound
        self._agent_checkpoint = agent_checkpoint

    @property
    def name(self) -> str:
        return self._step.name

    async def execute(self, input: Any) -> StepResult:
        from nanitics.composition.orchestration.adapters import AgentStep

        if self._agent_checkpoint is not None:
            self._bound.agent._set_resume_state(self._agent_checkpoint)
            self._agent_checkpoint = None
        step = cast(AgentStep, self._step)
        result = await self._bound.run(str(input), thread_key=step._thread_key)
        metadata: dict[str, Any] = {
            "total_steps": result.total_steps,
            "termination_reason": result.termination_reason,
            "usage": result.usage.model_dump(),
        }
        if result.parsed is not None:
            metadata["text_output"] = result.output
            return StepResult(output=result.parsed, metadata=metadata, usage=result.usage)
        return StepResult(output=result.output, metadata=metadata, usage=result.usage)


class _BoundHandoffStep:
    """Wraps a ``HandoffStep`` for per-call execution under a ``BoundAgent``.

    Replicates ``HandoffStep.execute`` — run agent, extract via transfer
    strategy, emit ``HandoffEvent`` — but routes the agent run through
    the bound handle and emits the ``HandoffEvent`` on the workflow's
    per-call child emitter rather than the ``HandoffStep``'s stored
    static emitter.

    .. deprecated:: 0.5.0
        Reading token usage from ``StepResult.metadata["usage"]`` (the
        dict mirror) is deprecated. Use the typed :attr:`StepResult.usage`
        field instead. The dict mirror is retained alongside the typed
        field for backwards compatibility and will be removed in 1.0.0.
        See ``docs/migrations/step-result-usage.md``.
    """

    def __init__(self, step: Step, bound: Any, emitter: EventEmitter) -> None:
        self._step = step
        self._bound = bound
        self._emitter = emitter

    @property
    def name(self) -> str:
        return self._step.name

    async def execute(self, input: Any) -> StepResult:
        from nanitics.composition.multi_agent.handoff import HandoffStep
        from nanitics.infrastructure.observability.events import HandoffEvent

        step = cast(HandoffStep, self._step)
        result = await self._bound.run(str(input), thread_key=step._thread_key)
        handoff_text = await step._transfer_strategy.extract(result)
        payload_fields = list(type(result).model_fields.keys())
        payload_size = len(handoff_text)
        self._emitter.emit(
            HandoffEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                from_agent=step._name,
                to_agent=step._to_agent,
                payload_fields=payload_fields,
                payload_size=payload_size,
            )
        )
        return StepResult(
            output=handoff_text,
            metadata={
                "agent_name": step._agent.name,
                "total_steps": result.total_steps,
                "termination_reason": result.termination_reason,
                "usage": result.usage.model_dump(),
            },
            usage=result.usage,
        )


class _BoundWorkflowStep:
    """Wraps a ``WorkflowStep`` for per-call execution under a ``BoundWorkflow``."""

    def __init__(self, step: Step, bound: Any) -> None:
        self._step = step
        self._bound = bound

    @property
    def name(self) -> str:
        return self._step.name

    async def execute(self, input: Any, *, resume_from: RunCheckpoint | None = None) -> StepResult:
        result: StepResult = await self._bound.execute(input, resume_from=resume_from)
        return result
