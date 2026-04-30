from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class StepStatus(StrEnum):
    """Status of an individual plan step."""

    not_started = "not_started"
    in_progress = "in_progress"
    completed = "completed"
    skipped = "skipped"
    failed = "failed"


class PlanStatus(StrEnum):
    """Overall status of a plan."""

    active = "active"
    completed = "completed"
    abandoned = "abandoned"


class GoalStatus(StrEnum):
    """Status of a goal within a plan."""

    active = "active"
    achieved = "achieved"
    blocked = "blocked"
    abandoned = "abandoned"


class PlanStep(BaseModel):
    """An individual step within a plan.

    Attributes:
        id: Unique identifier (auto-generated UUID).
        description: Human-readable description of this step.
        status: Current execution status.
        result: Optional result text recorded after execution.
        dependencies: IDs of steps that must complete before this one.
        metadata: Arbitrary key-value data attached to this step.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    description: str
    status: StepStatus = StepStatus.not_started
    result: str | None = None
    dependencies: list[str] = []
    metadata: dict[str, Any] = {}


class Goal(BaseModel):
    """A desired outcome within a plan, optionally with subgoals.

    Attributes:
        id: Unique identifier (auto-generated UUID).
        description: Human-readable description of the goal.
        status: Current goal status.
        priority: Numeric priority (higher = more important).
        success_criteria: Optional description of what "achieved" means.
        subgoals: Nested goals that contribute to this one.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    description: str
    status: GoalStatus = GoalStatus.active
    priority: int = 0
    success_criteria: str | None = None
    subgoals: list["Goal"] = []


class Plan(BaseModel):
    """A structured plan with ordered steps and optional goals.

    Attributes:
        id: Unique identifier (auto-generated UUID).
        name: Short name for the plan.
        description: Optional detailed description.
        status: Overall plan status. Auto-completes when all steps finish.
        steps: Ordered list of plan steps.
        goals: Optional goal hierarchy for the plan.
        namespace: Optional scope for filtering plans in a shared store.
        metadata: Arbitrary key-value data attached to the plan.
        created_at: Timestamp when the plan was created.
        updated_at: Timestamp of the last modification.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str | None = None
    status: PlanStatus = PlanStatus.active
    steps: list[PlanStep] = []
    goals: list[Goal] = []
    namespace: str | None = None
    metadata: dict[str, Any] = {}
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TaskNode(BaseModel):
    """A node in a task decomposition tree used by ``plan_to_workflow``.

    Attributes:
        id: Unique identifier (auto-generated UUID).
        description: Human-readable description of the task.
        subtasks: Child tasks that compose this task.
        dependencies: IDs of sibling tasks that must complete first.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    description: str
    subtasks: list["TaskNode"] = []
    dependencies: list[str] = []


class TaskPlan(BaseModel):
    """A task decomposition tree convertible to an orchestration workflow.

    Use ``plan_to_workflow`` to convert a ``TaskPlan`` into an executable
    ``Workflow`` that runs the tasks with appropriate parallelism.

    Attributes:
        id: Unique identifier (auto-generated UUID).
        name: Name for the task plan.
        root_tasks: Top-level tasks in the decomposition tree.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    root_tasks: list[TaskNode] = []
