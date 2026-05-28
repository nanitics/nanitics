from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from nanitics.composition.orchestration.protocol import StepResult
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
    """

    def __init__(self, agent: Agent, *, thread_key: str | None = None) -> None:
        self._agent = agent
        self._thread_key = thread_key

    @property
    def name(self) -> str:
        return self._agent.name

    @property
    def agent(self) -> Agent:
        """The wrapped agent."""
        return self._agent

    async def execute(self, input: Any) -> StepResult:
        result = await self._agent.run(str(input), thread_key=self._thread_key)
        metadata: dict[str, Any] = {
            "total_steps": result.total_steps,
            "termination_reason": result.termination_reason,
            "usage": result.usage.model_dump(),
        }
        if result.parsed is not None:
            metadata["text_output"] = result.output
            return StepResult(output=result.parsed, metadata=metadata)
        return StepResult(output=result.output, metadata=metadata)


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

    async def execute(self, input: Any) -> StepResult:
        return await self._workflow.execute(input)


class FunctionStep:
    """Wraps an async function as a Step.

    If the function returns a ``StepResult``, it is used directly. Otherwise,
    the return value is wrapped in ``StepResult(output=...)``.

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
