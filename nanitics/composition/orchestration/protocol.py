from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from nanitics.infrastructure.observability.events import Usage


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
            reasons, intermediate results, and failure details. Surfaced on
            :class:`WorkflowStepCompleteEvent` as ``step_metadata``; values
            are coerced to a JSON-safe shape at event-construction time
            (non-serializable objects fall back to ``repr()``), so prefer
            JSON-serializable values to preserve fidelity for event sinks.
        usage: Token usage produced by the step. ``None`` when the step did
            not run an LLM call (e.g. ``FunctionStep``) or when the step's
            underlying agent produced no usage. For ``Sequential`` and
            ``Pipeline`` workflows, the returned ``StepResult.usage`` is the
            aggregated sum of all completed sub-step usages (``None`` only
            when every sub-step contributed ``None``).

    The ``metadata["usage"]`` dict mirror written by ``AgentStep`` and the
    bound agent/handoff step wrappers is deprecated; use ``usage`` (the
    typed field) as the canonical access path. See
    ``docs/migrations/step-result-usage.md``.
    """

    model_config = ConfigDict(frozen=True)

    output: Any = None
    metadata: dict[str, Any] = {}
    usage: Usage | None = None


def _sum_optional(values: Iterable[int | None]) -> int | None:
    present = [v for v in values if v is not None]
    return sum(present) if present else None


def _sum_usage(usages: Iterable[Usage | None]) -> Usage | None:
    """Aggregate a collection of optional ``Usage`` values.

    ``None`` inputs are dropped (no contribution, not coerced to zero).
    Returns ``None`` only when every input is ``None``; otherwise returns
    a new ``Usage`` whose token fields are summed across present values.
    Cache-token fields are summed independently using the same rule, so
    they remain ``None`` if no input carried them.
    """
    materialized = list(usages)
    present = [u for u in materialized if u is not None]
    if not present:
        return None
    return Usage(
        input_tokens=sum(u.input_tokens for u in present),
        output_tokens=sum(u.output_tokens for u in present),
        cache_creation_input_tokens=_sum_optional(u.cache_creation_input_tokens for u in present),
        cache_read_input_tokens=_sum_optional(u.cache_read_input_tokens for u in present),
    )


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


@runtime_checkable
class StepObserver(Protocol):
    """Awaited lifecycle hooks fired around an :class:`AgentStep`'s execution.

    Attach via ``AgentStep(agent, observer=...)`` to observe a step's boundary
    without wrapping the agent in a custom :class:`Step`. Both hooks are
    awaited inline with the step, so an observer can persist per-step progress
    transactionally: ``on_start`` before the agent runs, ``on_complete`` after
    its :class:`StepResult` is composed. A step that suspends (raises
    ``SuspendExecution``) fires ``on_start`` but not ``on_complete`` — it did
    not complete.

    Prefer an observer over a custom ``Step`` for this purpose: a custom step
    that wraps an agent loses the agent's step-level durability, because the
    orchestrator only threads its checkpoint sink into an :class:`AgentStep`.
    An observer keeps the step a first-class ``AgentStep``, so tool-batch
    crash-resume still applies to the agent inside it.
    """

    async def on_start(self, input: Any) -> None:
        """Called with the step's input before the agent runs."""
        ...

    async def on_complete(self, result: StepResult) -> None:
        """Called with the composed :class:`StepResult` after the agent returns."""
        ...
