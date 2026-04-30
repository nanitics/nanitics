from __future__ import annotations

import json
import time
from typing import Any

from nanitics.collaboration.protocol import (
    HumanDecision,
    HumanInputProvider,
    HumanInputRequest,
    HumanInputType,
)
from nanitics.core.tools.context import _current_tool_context
from nanitics.core.tools.protocol import Tool, ToolResult
from nanitics.infrastructure.llm.protocol import ToolSchema
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import (
    HumanInputRequestEvent,
    HumanInputResponseEvent,
)

_MISSING_CONTEXT_MESSAGE = (
    "ApprovalWrappedTool requires a ToolContext with run_id and tool_call_id — "
    "construct the agent with run_id=… or register the tool on a "
    "ToolRegistry with tool_state={'run_id': …} and dispatch via ToolCall"
)


class ApprovalWrappedTool:
    """Wraps a tool so every invocation requires human approval before execution.

    The human sees the tool name, parameters, and description, then decides
    whether to approve, modify parameters, or reject. The wrapped tool's
    schema has ``requires_approval=True``.

    The wrapper derives its ``request_id`` from the ambient ``ToolContext``
    as ``f"{run_id}:{tool_call_id}"`` so a logical approval keeps the same
    identity across suspend/resume re-execution. The agent must be
    constructed with a ``run_id`` (or the registry's ``tool_state`` must
    supply one) and the tool must be dispatched via a ``ToolCall``.

    Args:
        tool: The tool to wrap.
        provider: Handles delivering the approval request.
        emitter: Event emitter for observability.
    """

    def __init__(
        self,
        tool: Tool,
        provider: HumanInputProvider,
        emitter: EventEmitter | None = None,
    ) -> None:
        self._tool = tool
        self._provider = provider
        self._emitter = emitter

    @property
    def schema(self) -> ToolSchema:
        return self._tool.schema.model_copy(update={"requires_approval": True})

    async def execute(self, **params: Any) -> ToolResult:
        """Request approval, then execute the tool if approved.

        Returns:
            The tool's result on approval, or a rejection message on reject.
        """
        tool_context = _current_tool_context.get()
        if tool_context is None or tool_context.run_id is None or tool_context.tool_call_id is None:
            raise ValueError(_MISSING_CONTEXT_MESSAGE)
        run_id = tool_context.run_id
        tool_call_id = tool_context.tool_call_id
        agent_name = tool_context.state.get("agent_name") if tool_context.state else None

        request_id = f"{run_id}:{tool_call_id}"
        tool_name = self._tool.schema.name
        params_summary = json.dumps(params, default=str)

        prompt = f"Approve tool '{tool_name}' execution?"
        context = f"Parameters:\n{params_summary}\n\nDescription: {self._tool.schema.description}"

        request = HumanInputRequest(
            request_id=request_id,
            run_id=run_id,
            request_type=HumanInputType.APPROVAL,
            prompt=prompt,
            context=context,
            metadata={"tool_name": tool_name, "parameters": params},
            agent_name=agent_name,
        )

        if self._emitter is not None:
            self._emitter.emit(
                HumanInputRequestEvent(
                    trace_id=self._emitter.trace_id,
                    span_id=self._emitter.span_id,
                    parent_span_id=self._emitter.parent_span_id,
                    request_id=request_id,
                    request_type=HumanInputType.APPROVAL.value,
                    prompt=prompt,
                    context=context,
                    agent_name=agent_name,
                    tool_name=tool_name,
                    metadata={"tool_name": tool_name, "parameters": params},
                )
            )

        start = time.monotonic()
        response = await self._provider.request_input(request)
        wait_ms = int((time.monotonic() - start) * 1000)

        if self._emitter is not None:
            self._emitter.emit(
                HumanInputResponseEvent(
                    trace_id=self._emitter.trace_id,
                    span_id=self._emitter.span_id,
                    parent_span_id=self._emitter.parent_span_id,
                    request_id=request_id,
                    decision=response.decision.value,
                    has_content=response.content is not None,
                    wait_duration_ms=wait_ms,
                )
            )

        if response.decision == HumanDecision.APPROVE:
            return await self._tool.execute(**params)

        if response.decision == HumanDecision.OVERRIDE:
            modified_params = dict(params)
            if response.metadata.get("modified_params"):
                modified_params.update(response.metadata["modified_params"])
            return await self._tool.execute(**modified_params)

        reason = response.content or "No reason provided"
        return ToolResult(
            content=f"Action rejected by human: {reason}",
            executed=False,
        )
