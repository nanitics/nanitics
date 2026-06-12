"""Durable-HITL resume abstractions.

Provides :class:`DurableRun` (suspend-side wrapper) and
:class:`ResumeService` (resume-side dispatcher) so consumers stop
hand-rolling the "catch ``SuspendExecution`` / preload response /
re-execute with ``resume_from``" glue. See
``docs/guides/human-in-the-loop.md`` for the full pattern and
``examples/durability/durable_resume_service.py`` for a runnable reference.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nanitics.collaboration.hitl_store import DuplicateHitlResponseError, HitlRequestStore
from nanitics.collaboration.protocol import HumanInputRequest, HumanInputResponse
from nanitics.composition.durability.models import RunCheckpoint, SuspensionInfo
from nanitics.composition.durability.store import CheckpointStore
from nanitics.composition.durability.suspension import SuspendExecution

if TYPE_CHECKING:
    from nanitics.composition.orchestration.workflow import Workflow
    from nanitics.strategies.agents.base import Agent


@dataclass(frozen=True)
class ResumeContext:
    """Inputs a factory needs to reconstruct a :class:`DurableRun` on resume.

    Attributes:
        run_id: The run being resumed.
        checkpoint: The checkpoint loaded from the ``CheckpointStore`` for
            this run — includes ``suspension_info`` and persisted state.
        hitl_store: The HITL request store in use for this process.
        checkpoint_store: The checkpoint store in use for this process.
    """

    run_id: str
    checkpoint: RunCheckpoint
    hitl_store: HitlRequestStore
    checkpoint_store: CheckpointStore


@dataclass(frozen=True)
class SuspendedRun:
    """Serializable handle returned when a run suspends.

    Consumers ship this payload across a process boundary; the resume
    caller routes the corresponding :class:`HumanInputResponse` back
    through :meth:`ResumeService.resume` keyed by ``run_id``.

    Attributes:
        run_id: Identifies the suspended run.
        suspension_info: Details of the suspension that produced this
            payload.
        pending_request: The HITL request awaiting a human response.
        checkpoint_id: Identifier of the checkpoint that was persisted
            at the suspension point — useful for audit/debug.
    """

    run_id: str
    suspension_info: SuspensionInfo
    pending_request: HumanInputRequest
    checkpoint_id: str


@dataclass(frozen=True)
class ResumeResult:
    """Returned when a run reaches completion without (further) suspending.

    Mirrors the shape of :class:`StepResult` so downstream code has the
    same ``output`` / ``metadata`` contract regardless of whether the
    run completed on its first pass or after one or more resumes.

    Attributes:
        run_id: The run that produced this result.
        output: The final output value from the workflow / agent.
        metadata: Workflow-level metadata (intermediate results, step
            counts, token usage, etc.).
    """

    run_id: str
    output: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class DurableRun:
    """Suspend-side wrapper around an :class:`Agent` or :class:`Workflow`.

    Executes the runnable under a ``CheckpointStore`` and converts any
    :class:`SuspendExecution` raised by the runnable into a
    :class:`SuspendedRun` payload. Never propagates ``SuspendExecution``
    to callers — every suspension is a value.

    When ``runnable`` is an :class:`Agent`, it is wrapped internally in
    an ``AgentStep`` + ``Sequential`` so both agent and workflow inputs
    flow through the same checkpoint/suspend machinery. When ``runnable``
    is already a :class:`Workflow`, its ``_checkpoint_store`` and
    ``_run_id`` must agree with the ones passed here.

    Args:
        runnable: The agent or workflow to execute.
        hitl_store: Store for HITL requests/responses.
        checkpoint_store: Store for orchestration checkpoints.
        run_id: Identifier for this run. Required for durable HITL; if
            omitted and the supplied workflow already carries one, it is
            adopted.
        step_checkpoints: When ``True``, the agent-wrapping workflow writes a
            cursor checkpoint after each completed step so the run can be
            resumed via :meth:`resume_from_checkpoint` /
            :meth:`ResumeService.resume_interrupted` after a crash. Applies to
            the ``Agent`` case; when wrapping a ``Workflow`` directly, configure
            ``step_checkpoints`` on that workflow. Defaults to ``False``.

    Raises:
        ValueError: If ``run_id`` cannot be resolved, if a workflow's
            existing ``run_id`` conflicts with the one passed here, or
            if the workflow's ``_checkpoint_store`` differs from
            ``checkpoint_store``.
    """

    def __init__(
        self,
        runnable: Agent | Workflow,
        *,
        hitl_store: HitlRequestStore,
        checkpoint_store: CheckpointStore,
        run_id: str | None = None,
        step_checkpoints: bool = False,
    ) -> None:
        from nanitics.composition.orchestration.adapters import AgentStep
        from nanitics.composition.orchestration.sequential import Sequential
        from nanitics.composition.orchestration.workflow import Workflow
        from nanitics.strategies.agents.base import Agent

        self._hitl_store = hitl_store
        self._checkpoint_store = checkpoint_store

        if isinstance(runnable, Agent):
            resolved_run_id = run_id
            if resolved_run_id is None:
                raise ValueError(
                    "DurableRun requires a run_id when wrapping an Agent; "
                    "durable HITL cannot key checkpoints without one."
                )
            self._workflow: Workflow = Sequential(
                name=f"durable-{runnable.name}",
                steps=[AgentStep(runnable)],
                emitter=runnable._default_emitter,
                checkpoint_store=checkpoint_store,
                run_id=resolved_run_id,
                step_checkpoints=step_checkpoints,
            )
            self._run_id = resolved_run_id
        elif isinstance(runnable, Workflow):
            if run_id is not None and run_id != runnable._run_id:
                raise ValueError(f"DurableRun run_id={run_id!r} conflicts with workflow run_id={runnable._run_id!r}")
            if runnable._checkpoint_store is None:
                raise ValueError("DurableRun requires the supplied workflow to have a checkpoint_store configured")
            if runnable._checkpoint_store is not checkpoint_store:
                raise ValueError("DurableRun checkpoint_store must match the workflow's configured checkpoint_store")
            self._workflow = runnable
            self._run_id = runnable._run_id
        else:
            raise TypeError(f"DurableRun runnable must be an Agent or Workflow; got {type(runnable).__name__}")

    @property
    def run_id(self) -> str:
        """The run identifier used for checkpoint and HITL-store keys."""
        return self._run_id

    async def start(self, input: Any) -> ResumeResult | SuspendedRun:
        """Execute the runnable from the top; return completion or suspension.

        Never raises :class:`SuspendExecution`: a suspension is converted
        to a :class:`SuspendedRun` payload describing the pending HITL
        request and the persisted checkpoint.
        """
        try:
            result = await self._workflow.execute(input)
        except SuspendExecution as exc:
            return await self._build_suspended_run(exc)
        return ResumeResult(
            run_id=self._run_id,
            output=result.output,
            metadata=dict(result.metadata),
        )

    async def _resume(self, checkpoint: RunCheckpoint) -> ResumeResult | SuspendedRun:
        """Re-drive the runnable from ``checkpoint``.

        Used by :class:`ResumeService.resume`. Pulls the original input
        from the checkpoint state (preserved by the orchestrator on the
        suspend path) and wraps the outcome identically to :meth:`start`.
        """
        original_input = checkpoint.state.get("original_input")
        try:
            result = await self._workflow.execute(original_input, resume_from=checkpoint)
        except SuspendExecution as exc:
            return await self._build_suspended_run(exc)
        return ResumeResult(
            run_id=self._run_id,
            output=result.output,
            metadata=dict(result.metadata),
        )

    async def resume_from_checkpoint(self) -> ResumeResult | SuspendedRun:
        """Resume an interrupted run from its latest cursor checkpoint.

        For crash / redeploy recovery under step-level durability: loads the
        most recent checkpoint for this run and re-drives the runnable, which
        skips steps already completed (and journaled) before the interruption.
        Unlike the HITL resume path there is no human response to apply.

        Raises:
            ValueError: If no checkpoint exists for this run, or the latest
                checkpoint is a HITL suspension still awaiting input — route
                that through :meth:`ResumeService.resume` instead.
        """
        checkpoint = await self._checkpoint_store.load(self._run_id)
        if checkpoint is None:
            raise ValueError(f"No checkpoint to resume for run_id={self._run_id!r}")
        if checkpoint.suspension_info is not None:
            raise ValueError(
                f"run_id={self._run_id!r} is suspended awaiting human input "
                f"(checkpoint_reason={checkpoint.checkpoint_reason!r}); resume it via "
                f"ResumeService.resume, not resume_from_checkpoint."
            )
        return await self._resume(checkpoint)

    async def _build_suspended_run(self, exc: SuspendExecution) -> SuspendedRun:
        checkpoint = await self._checkpoint_store.load(self._run_id)
        if checkpoint is None:
            raise RuntimeError(f"Workflow suspended for run_id={self._run_id!r} but no checkpoint was persisted")
        pending = await self._find_pending_request(exc.suspension_info.request_id)
        return SuspendedRun(
            run_id=self._run_id,
            suspension_info=exc.suspension_info,
            pending_request=pending,
            checkpoint_id=checkpoint.checkpoint_id,
        )

    async def _find_pending_request(self, request_id: str) -> HumanInputRequest:
        pending = await self._hitl_store.get_pending_requests(self._run_id)
        for request in pending:
            if request.request_id == request_id:
                return request
        raise RuntimeError(
            f"HITL store has no pending request for request_id={request_id!r} on run_id={self._run_id!r}"
        )


class ResumeService:
    """Resume-side dispatcher.

    Construct once per process with the stores and a factory that knows
    how to rebuild a :class:`DurableRun` for a given :class:`ResumeContext`.
    Every inbound :class:`HumanInputResponse` is routed through
    :meth:`resume`, which loads the checkpoint, validates the response,
    persists it, and re-drives the runnable through the factory-built
    :class:`DurableRun`.

    Args:
        hitl_store: Shared HITL request store.
        checkpoint_store: Shared checkpoint store.
        factory: Callable that returns a fresh :class:`DurableRun` for
            the given :class:`ResumeContext`. Invoked once per call to
            :meth:`resume`.
    """

    def __init__(
        self,
        *,
        hitl_store: HitlRequestStore,
        checkpoint_store: CheckpointStore,
        factory: Callable[[ResumeContext], DurableRun],
    ) -> None:
        self._hitl_store = hitl_store
        self._checkpoint_store = checkpoint_store
        self._factory = factory

    async def resume(
        self,
        run_id: str,
        response: HumanInputResponse,
    ) -> ResumeResult | SuspendedRun:
        """Persist the response and drive the factory-built run forward.

        Loads the checkpoint for ``run_id``, validates that
        ``response.request_id`` matches the checkpoint's pending
        ``suspension_info.request_id``, saves the response, invokes the
        factory to reconstruct a :class:`DurableRun`, and re-drives it.
        Returns :class:`ResumeResult` on completion or :class:`SuspendedRun`
        on a nested suspension.

        Raises:
            ValueError: If no checkpoint exists for ``run_id`` or the
                response ``request_id`` does not match the checkpoint's
                pending request.
            TypeError: If the factory does not return a :class:`DurableRun`.
        """
        checkpoint = await self._checkpoint_store.load(run_id)
        if checkpoint is None:
            raise ValueError(f"No checkpoint for run_id={run_id!r}")

        if checkpoint.suspension_info is None:
            raise ValueError(
                f"Checkpoint for run_id={run_id!r} is not a HITL suspension "
                f"(checkpoint_reason={checkpoint.checkpoint_reason!r}); HITL resume "
                f"requires a suspension checkpoint."
            )
        expected = checkpoint.suspension_info.request_id
        if response.request_id != expected:
            raise ValueError(
                f"Response request_id mismatch for run_id={run_id!r}: "
                f"expected {expected!r}, got {response.request_id!r}"
            )

        # Re-saving the response on a re-driven resume (a worker re-claiming
        # the job after a mid-resume crash) is expected and idempotent — the
        # store is the gate, mirroring the request side's duplicate handling.
        with contextlib.suppress(DuplicateHitlResponseError):
            await self._hitl_store.save_response(response.request_id, response)

        ctx = ResumeContext(
            run_id=run_id,
            checkpoint=checkpoint,
            hitl_store=self._hitl_store,
            checkpoint_store=self._checkpoint_store,
        )
        durable_run = self._factory(ctx)
        if not isinstance(durable_run, DurableRun):
            raise TypeError(f"ResumeService factory must return a DurableRun; got {type(durable_run).__name__}")
        return await durable_run._resume(checkpoint)

    async def resume_interrupted(
        self,
        run_id: str,
    ) -> ResumeResult | SuspendedRun:
        """Resume a crashed / redeployed run from its latest cursor checkpoint.

        The step-level-durability counterpart to :meth:`resume`: loads the
        most recent checkpoint for ``run_id`` and re-drives a factory-built
        :class:`DurableRun`, skipping steps completed before the interruption.
        There is no human response to validate or persist.

        Raises:
            ValueError: If no checkpoint exists for ``run_id``, or the latest
                checkpoint is a HITL suspension still awaiting input — route
                that through :meth:`resume` instead.
            TypeError: If the factory does not return a :class:`DurableRun`.
        """
        checkpoint = await self._checkpoint_store.load(run_id)
        if checkpoint is None:
            raise ValueError(f"No checkpoint for run_id={run_id!r}")
        if checkpoint.suspension_info is not None:
            raise ValueError(
                f"Checkpoint for run_id={run_id!r} is a HITL suspension awaiting "
                f"input (checkpoint_reason={checkpoint.checkpoint_reason!r}); resume it "
                f"via resume(), not resume_interrupted()."
            )

        ctx = ResumeContext(
            run_id=run_id,
            checkpoint=checkpoint,
            hitl_store=self._hitl_store,
            checkpoint_store=self._checkpoint_store,
        )
        durable_run = self._factory(ctx)
        if not isinstance(durable_run, DurableRun):
            raise TypeError(f"ResumeService factory must return a DurableRun; got {type(durable_run).__name__}")
        return await durable_run._resume(checkpoint)


__all__ = [
    "DurableRun",
    "ResumeContext",
    "ResumeResult",
    "ResumeService",
    "SuspendedRun",
]
