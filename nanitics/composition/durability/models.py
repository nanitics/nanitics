from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from nanitics.infrastructure.errors import NaniticsError

CHECKPOINT_SCHEMA_VERSION = 2


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

    Attributes:
        suspension_id: Unique identifier for this suspension.
        suspension_type: The kind of suspension (currently always ``"hitl"``).
        request_id: The HITL request that caused the suspension.
        request_type: The type of human input requested.
        prompt: The prompt shown to the human.
        agent_name: Which agent triggered the suspension.
    """

    model_config = ConfigDict(frozen=True)

    suspension_id: str
    suspension_type: Literal["hitl"] = "hitl"
    request_id: str
    request_type: str
    prompt: str
    agent_name: str | None = None


class RunCheckpoint(BaseModel):
    """Snapshot of workflow or agent state at the point of suspension.

    Persisted by a ``CheckpointStore`` and used to resume execution after
    a human responds. Contains the completed step results and the
    suspension details.

    Attributes:
        checkpoint_id: Unique identifier (auto-generated UUID).
        run_id: The run this checkpoint belongs to.
        checkpoint_type: Whether this is an orchestration or agent checkpoint.
        schema_version: Version for forward-compatibility checking.
        state: Serialized execution state (completed results, position, etc.).
        suspension_info: Details about the suspension that produced this checkpoint.
        created_at: When the checkpoint was created (UTC).
    """

    model_config = ConfigDict(frozen=True)

    checkpoint_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    checkpoint_type: Literal["orchestration", "agent"]
    schema_version: int = CHECKPOINT_SCHEMA_VERSION
    state: dict[str, Any]
    suspension_info: SuspensionInfo
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
