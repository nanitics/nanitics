from __future__ import annotations

from typing import Any

from nanitics.composition.multi_agent.context_transfer import (
    ContextTransferStrategy,
    RawOutputTransfer,
)
from nanitics.composition.orchestration.protocol import Step, StepResult
from nanitics.composition.orchestration.sequential import Sequential
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import HandoffEvent
from nanitics.safety.cancellation import CancellationToken
from nanitics.strategies.agents.base import Agent


class HandoffStep:
    """A workflow step that runs an agent and applies a context transfer strategy.

    Implements the ``Step`` protocol. On execution, runs the wrapped agent,
    extracts the result via the transfer strategy, and emits a
    ``HandoffEvent`` linking the source and destination agents.

    Args:
        agent: Agent to run for this step.
        emitter: Event emitter for handoff tracing.
        transfer_strategy: How to extract the agent's result.
            Defaults to ``RawOutputTransfer``.
        name: Step name override. Defaults to ``agent.name``.
        to_agent: Name of the next agent in the chain (for tracing).
    """

    def __init__(
        self,
        *,
        agent: Agent,
        emitter: EventEmitter,
        transfer_strategy: ContextTransferStrategy | None = None,
        name: str | None = None,
        to_agent: str = "unknown",
    ) -> None:
        self._agent = agent
        self._emitter = emitter
        self._transfer_strategy = transfer_strategy or RawOutputTransfer()
        self._name = name or agent.name
        self._to_agent = to_agent

    @property
    def name(self) -> str:
        return self._name

    @property
    def agent(self) -> Agent:
        """The wrapped agent."""
        return self._agent

    async def execute(self, input: Any) -> StepResult:
        """Run the agent with ``input`` as the task and return the transferred output.

        Returns:
            StepResult with the extracted handoff text and metadata including
            ``agent_name``, ``total_steps``, ``termination_reason``, and ``usage``.
        """
        result = await self._agent.bind(self._emitter).run(str(input))

        handoff_text = await self._transfer_strategy.extract(result)

        payload_fields = list(type(result).model_fields.keys())
        payload_size = len(handoff_text)

        self._emitter.emit(
            HandoffEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                from_agent=self._name,
                to_agent=self._to_agent,
                payload_fields=payload_fields,
                payload_size=payload_size,
            )
        )

        return StepResult(
            output=handoff_text,
            metadata={
                "agent_name": self._agent.name,
                "total_steps": result.total_steps,
                "termination_reason": result.termination_reason,
                "usage": result.usage.model_dump(),
            },
        )


def create_handoff_chain(
    *,
    name: str,
    agents: list[Agent],
    emitter: EventEmitter,
    transfer_strategy: ContextTransferStrategy | None = None,
    cancellation_token: CancellationToken | None = None,
) -> Sequential:
    """Build a Sequential workflow that chains agents with handoff steps.

    Each agent's output is passed to the next via the specified transfer
    strategy. The last step always uses ``RawOutputTransfer`` since its
    output is the final result.

    Args:
        name: Workflow name.
        agents: Ordered list of agents (at least 2).
        emitter: Event emitter for workflow and handoff events.
        transfer_strategy: Strategy for intermediate handoffs.
            Defaults to ``RawOutputTransfer``.
        cancellation_token: Shared cancellation signal.

    Returns:
        A ``Sequential`` workflow ready to execute.

    Raises:
        ValueError: If fewer than 2 agents are provided.
    """
    if len(agents) < 2:
        raise ValueError("Handoff chain requires at least 2 agents")

    steps: list[Step] = []
    for i, agent in enumerate(agents):
        is_last = i == len(agents) - 1
        step_strategy = RawOutputTransfer() if is_last else (transfer_strategy or RawOutputTransfer())
        next_agent = "output" if is_last else agents[i + 1].name
        steps.append(
            HandoffStep(
                agent=agent,
                emitter=emitter,
                transfer_strategy=step_strategy,
                to_agent=next_agent,
            )
        )

    return Sequential(
        name=name,
        steps=steps,
        emitter=emitter,
        cancellation_token=cancellation_token,
    )
