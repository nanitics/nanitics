from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


class FailurePolicy(Enum):
    """Controls how concurrent workflows handle step failures.

    Applies to Parallel, MapReduce, and DAG workflows.
    """

    ALL_OR_NOTHING = "all_or_nothing"
    """Cancel all pending steps on any failure and propagate the exception."""
    BEST_EFFORT = "best_effort"
    """Continue execution despite failures, tracking failed steps in metadata."""


class StepResult(BaseModel):
    """Result returned by any step execution.

    Attributes:
        output: The step's output value, passed as input to downstream steps.
        metadata: Workflow-level information such as step counts, termination
            reasons, intermediate results, and failure details.
    """

    model_config = ConfigDict(frozen=True)

    output: Any = None
    metadata: dict[str, Any] = {}


@runtime_checkable
class Step(Protocol):
    """Protocol for execution units in workflows.

    Any object with a ``name`` property and an async ``execute`` method
    satisfies this protocol. The SDK provides ``AgentStep`` and
    ``FunctionStep`` as built-in adapters.
    """

    @property
    def name(self) -> str: ...

    async def execute(self, input: Any) -> StepResult:
        """Execute the step with the given input.

        Args:
            input: Data from the previous step or the workflow's initial input.

        Returns:
            A StepResult containing the output and optional metadata.
        """
        ...
