from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from nanitics.infrastructure.errors import NaniticsError

CHECKPOINT_SCHEMA_VERSION = 4


class CheckpointVersionError(NaniticsError):
    """Raised when a checkpoint's schema version doesn't match the expected version.

    Attributes:
        expected_version: The version the code expects.
        actual_version: The version found in the checkpoint.
    """

    expected_version: int
    actual_version: int

    def __init__(
        self,
        message: str,
        *,
        expected_version: int,
        actual_version: int,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> None:
        super().__init__(message, trace_id=trace_id, span_id=span_id)
        self.expected_version = expected_version
        self.actual_version = actual_version


class SuspensionInfo(BaseModel):
    """Details about why execution was suspended.

    Carried by ``SuspendExecution`` and stored in checkpoints to provide
    enough context for an external system to display the pending request.

    Two suspension kinds share this shape:

    - ``"hitl"`` — a human must answer before the run continues. The
      ``request_id`` / ``request_type`` / ``prompt`` fields describe the
      pending request.
    - ``"budget_exhausted"`` — a ReAct run hit ``max_iterations`` /
      ``max_tool_calls`` with ``suspend_on_budget=True``, so it parked itself
      resumable instead of ending. There is no human request; the HITL fields
      are empty and ``last_assistant_text`` carries the run's final assistant
      turn so a host can show the partial work. Resume it through
      :meth:`~nanitics.composition.durability.resume.ResumeService.continue_run`
      with the agent rebuilt on a larger budget, not the HITL ``resume`` path.

    Attributes:
        suspension_id: Unique identifier for this suspension.
        suspension_type: The kind of suspension — ``"hitl"`` or
            ``"budget_exhausted"``.
        request_id: The HITL request that caused the suspension (empty for a
            budget suspension).
        request_type: The type of human input requested (empty for a budget
            suspension).
        prompt: The prompt shown to the human (empty for a budget suspension).
        agent_name: Which agent triggered the suspension.
        last_assistant_text: For a budget suspension, the run's final assistant
            turn at the point of exhaustion, so a host can surface the partial
            work without scraping the trace. ``None`` for a HITL suspension.
    """

    model_config = ConfigDict(frozen=True)

    suspension_id: str
    suspension_type: Literal["hitl", "budget_exhausted"] = "hitl"
    request_id: str
    request_type: str
    prompt: str
    agent_name: str | None = None
    last_assistant_text: str | None = None


class RunCheckpoint(BaseModel):
    """Snapshot of workflow or agent state at a durable checkpoint.

    Persisted by a ``CheckpointStore`` and used to resume execution. A
    checkpoint is written either when a run suspends for human input (a HITL
    suspension) or, when step-level durability is enabled, as a thin cursor
    snapshot recording loop position after a completed step. The
    ``checkpoint_reason`` discriminator distinguishes the two without having
    to infer from ``suspension_info`` being unset.

    Attributes:
        checkpoint_id: Unique identifier (auto-generated UUID).
        run_id: The run this checkpoint belongs to.
        checkpoint_type: Whether this is an orchestration or agent checkpoint.
        schema_version: Version for forward-compatibility checking.
        state: Serialized execution state (completed results, position, etc.).
            For a suspended step that is an agent, ``state`` carries an
            optional ``agent_checkpoint`` dict; for a suspended step that is
            a nested ``Workflow`` (via ``WorkflowStep``), it carries an
            optional ``nested_checkpoint`` dict — itself a full orchestrator
            state, recursive to arbitrary nesting depth. The two keys are
            mutually exclusive: a suspended step is either an agent or a
            nested workflow.
        suspension_info: Details about the suspension that produced this
            checkpoint. Set for suspension checkpoints — HITL
            (``checkpoint_reason == "hitl_suspend"``) and budget-exhaustion
            (``checkpoint_reason == "budget_exhausted"``); ``None`` for step /
            crash-safe cursor checkpoints, which are not suspensions.
        checkpoint_reason: Why this checkpoint was written. ``"hitl_suspend"``
            (the default) for a checkpoint produced by a human-in-the-loop
            suspension; ``"budget_exhausted"`` for a ReAct run that parked
            itself on hitting ``max_iterations`` / ``max_tool_calls`` with
            ``suspend_on_budget=True``; ``"step"`` for a thin cursor snapshot
            written after a completed step; ``"crash_safe"`` for a defensively
            written cursor snapshot. A step/crash checkpoint is not a suspension
            and carries no ``suspension_info``.
        created_at: When the checkpoint was created (UTC).
    """

    model_config = ConfigDict(frozen=True)

    checkpoint_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    checkpoint_type: Literal["orchestration", "agent"]
    schema_version: int = CHECKPOINT_SCHEMA_VERSION
    state: dict[str, Any]
    suspension_info: SuspensionInfo | None = None
    checkpoint_reason: Literal["hitl_suspend", "budget_exhausted", "step", "crash_safe"] = "hitl_suspend"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class StepRecord(BaseModel):
    """An append-only record of one completed step's result.

    Written to the step-result journal (see ``CheckpointStore.append_step``)
    when step-level durability is enabled. On resume, the runtime consults the
    journal by ``step_path`` and injects a recorded ``result`` instead of
    re-dispatching the step, so a completed side-effecting step runs at most
    once across the run and all its resumes. The journal carries the
    correctness-critical "what side effects happened" data; the
    ``RunCheckpoint`` cursor carries only "where was I".

    Frozen and JSON-serializable, mirroring how ``RunCheckpoint`` is
    normalized for persistence.

    Attributes:
        run_id: The run this step belongs to. Together with ``step_path`` it
            forms the step key that is stable across the run and all resumes.
        step_path: Deterministic position of the step in the execution tree
            (e.g. ``"seq#2/agent/turn#3/tool#1:send_email"``). Composed from
            positional indices the runtime already holds, not content hashes,
            so it is identical on the original run and every resume.
        step_kind: The kind of step this records.
        result: Serialized step output (e.g. a ``tool_result`` message
            payload) to inject on replay.
        schema_version: Version for forward-compatibility checking; defaults
            to ``CHECKPOINT_SCHEMA_VERSION``.
        created_at: When the step completed and was recorded (UTC).
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    step_path: str
    step_kind: Literal["tool_call", "orchestration_step", "agent_turn"]
    result: dict[str, Any]
    schema_version: int = CHECKPOINT_SCHEMA_VERSION
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
