from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from nanitics.composition.durability.models import RunCheckpoint
from nanitics.composition.orchestration.protocol import StepObserver, StepResult
from nanitics.composition.orchestration.workflow import Workflow
from nanitics.strategies.agents.base import Agent


class AgentStep:
    """Wraps an Agent as a Step for use in workflows.

    Converts the input to a string task, runs the agent, and returns the agent's
    output with metadata including step count, termination reason, and token usage.

    When the wrapped agent has ``output_schema``, the step output is the parsed
    Pydantic model (``result.parsed``) and the text response is preserved in
    ``metadata["text_output"]``. Without ``output_schema``, the step output is
    the text response as before.

    .. deprecated:: 0.5.0
        Reading token usage from ``StepResult.metadata["usage"]`` (the dict
        mirror) is deprecated. Use the typed :attr:`StepResult.usage` field
        instead. The dict mirror is retained alongside the typed field for
        backwards compatibility and will be removed in 1.0.0. See
        ``docs/migrations/step-result-usage.md``.

    Args:
        agent: The agent to execute as a workflow step.
        thread_key: Opaque key identifying the conversation thread this
            step's agent continues across repeated runs of the same
            step. Forwarded to
            :meth:`~nanitics.strategies.agents.base.Agent.run` on each
            execution. The agent must be configured with a
            :class:`~nanitics.composition.threads.ThreadStore` for the
            prefix to be persisted; the key is otherwise accepted and
            ignored. ``None`` (the default) runs the step stateless.
        observer: Optional :class:`~nanitics.composition.orchestration.protocol.StepObserver`
            whose ``on_start`` / ``on_complete`` hooks are awaited around the
            agent run — ``on_start`` before it runs, ``on_complete`` after the
            :class:`StepResult` is composed (skipped if the step suspends). Use
            this to attach per-step lifecycle behaviour (e.g. persisting
            progress) without wrapping the agent in a custom ``Step``, which
            would forfeit the agent's step-level durability. ``None`` (the
            default) attaches no hooks.
    """

    def __init__(
        self,
        agent: Agent,
        *,
        thread_key: str | None = None,
        observer: StepObserver | None = None,
    ) -> None:
        self._agent = agent
        self._thread_key = thread_key
        self._observer = observer

    @property
    def name(self) -> str:
        return self._agent.name

    @property
    def agent(self) -> Agent:
        """The wrapped agent."""
        return self._agent

    async def execute(self, input: Any) -> StepResult:
        if self._observer is not None:
            await self._observer.on_start(input)
        result = await self._agent.run(str(input), thread_key=self._thread_key)
        metadata: dict[str, Any] = {
            "total_steps": result.total_steps,
            "termination_reason": result.termination_reason,
            "usage": result.usage.model_dump(),
        }
        if result.parsed is not None:
            metadata["text_output"] = result.output
            step_result = StepResult(output=result.parsed, metadata=metadata, usage=result.usage)
        else:
            step_result = StepResult(output=result.output, metadata=metadata, usage=result.usage)
        if self._observer is not None:
            await self._observer.on_complete(step_result)
        return step_result


class WorkflowStep:
    """Wraps a Workflow as a Step for use in other workflows.

    Since ``Workflow`` does not directly implement the ``Step`` protocol
    (its ``execute`` accepts an extra ``resume_from`` keyword argument),
    this adapter bridges the gap for composability.

    Args:
        workflow: The workflow to wrap as a step.
    """

    def __init__(self, workflow: Workflow) -> None:
        self._workflow = workflow

    @property
    def name(self) -> str:
        return self._workflow.name

    @property
    def workflow(self) -> Workflow:
        """The wrapped workflow."""
        return self._workflow

    async def execute(self, input: Any, *, resume_from: RunCheckpoint | None = None) -> StepResult:
        return await self._workflow.execute(input, resume_from=resume_from)


class FunctionStep:
    """Wraps an async function as a Step.

    If the function returns a ``StepResult``, it is used directly — including
    its ``usage`` field, which passes through unchanged (whether ``None`` or
    a populated ``Usage``). Otherwise, the return value is wrapped in
    ``StepResult(output=...)`` and the resulting ``usage`` defaults to
    ``None``, since a plain function has no LLM call to attribute tokens to.

    Args:
        name: Step name used in events and trace spans.
        fn: Async function that takes input and returns output.
    """

    def __init__(
        self,
        name: str,
        fn: Callable[[Any], Awaitable[Any]],
    ) -> None:
        self._name = name
        self._fn = fn

    @property
    def name(self) -> str:
        return self._name

    async def execute(self, input: Any) -> StepResult:
        result = await self._fn(input)
        if isinstance(result, StepResult):
            return result
        return StepResult(output=result)
