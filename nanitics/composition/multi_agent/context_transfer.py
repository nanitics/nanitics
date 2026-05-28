from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from nanitics.infrastructure.llm.protocol import LLMClient, Message
from nanitics.strategies.agents.base import AgentResult


@runtime_checkable
class ContextTransferStrategy(Protocol):
    """Protocol for extracting transferable content from an agent result.

    Implementations control what information flows from one agent to the
    next — full output, trajectory, summary, or a custom extraction.

    Returns ``str`` because the seam between agents is messages. Consumers
    that need a typed object on their side of the seam project from the
    extracted string (or from ``AgentResult`` directly) in their own code;
    the SDK does not ship a parallel typed-transfer protocol.

    Distinct from ``ContextProvider``, which runs *inside one agent* on
    every LLM call to inject dynamic context. ``ContextTransferStrategy``
    runs *between two agents*, once per delegation or handoff edge. See
    ``docs/guides/multi-agent-foundations.md`` § Context Transfer for the
    full contrast.
    """

    async def extract(self, result: AgentResult) -> str: ...


class RawOutputTransfer:
    """Passes the agent's final output string as-is.

    Cheapest strategy — no additional LLM calls. Loses the reasoning
    trajectory; only the final answer is transferred.
    """

    async def extract(self, result: AgentResult) -> str:
        return result.output or ""


class TrajectoryTransfer:
    """Formats the full message history including tool calls.

    Most faithful representation of the agent's reasoning process.
    Can be large for long-running agents.
    """

    async def extract(self, result: AgentResult) -> str:
        return _format_messages(result.messages)


class SummaryTransfer:
    """Uses an LLM to summarize the agent conversation.

    Compresses the trajectory while preserving key findings, decisions,
    and outcomes. Costs one additional LLM call per transfer.

    Args:
        llm_client: LLM client used to generate the summary.
    """

    _PROMPT = (
        "Summarize the following agent conversation concisely. "
        "Capture the key findings, decisions, and final outcome. "
        "Omit redundant steps and failed attempts unless they produced useful information."
    )

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    async def extract(self, result: AgentResult) -> str:
        trajectory = _format_messages(result.messages)
        response = await self._llm_client.generate(
            system_prompt=self._PROMPT,
            messages=[Message(role="user", content=trajectory)],
        )
        return response.content or ""


class CustomTransfer:
    """User-defined extraction function for full control over context transfer.

    Args:
        fn: Callable that takes an ``AgentResult`` and returns a string.
    """

    def __init__(self, fn: Callable[[AgentResult], str]) -> None:
        self._fn = fn

    async def extract(self, result: AgentResult) -> str:
        return self._fn(result)


def _format_messages(messages: list[Message]) -> str:
    parts: list[str] = []
    for msg in messages:
        prefix = msg.role.upper()
        if msg.content:
            parts.append(f"{prefix}: {msg.content}")
        if msg.tool_calls:
            parts.extend(f"{prefix} [tool_call]: {tc.name}({tc.arguments})" for tc in msg.tool_calls)
    return "\n".join(parts)
