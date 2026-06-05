from __future__ import annotations

import time

from nanitics.collaboration.protocol import (
    HumanDecision,
    HumanInputProvider,
    HumanInputRequest,
    HumanInputType,
)
from nanitics.infrastructure.observability.events import (
    HumanInputRequestEvent,
    HumanInputResponseEvent,
)
from nanitics.strategies.tools.context import ToolContext
from nanitics.strategies.tools.function_tool import FunctionTool, tool

_MISSING_CONTEXT_MESSAGE = (
    "HITL tools require a ToolContext with run_id and tool_call_id — "
    "construct the agent with run_id=… or register the tool on a "
    "ToolRegistry with tool_state={'run_id': …} and dispatch via ToolCall"
)


def _require_identity(context: ToolContext | None) -> tuple[str, str, str | None]:
    """Extract (run_id, tool_call_id, agent_name) from the ambient context.

    HITL tools require both ``run_id`` and ``tool_call_id`` to derive a
    deterministic ``request_id`` that survives suspend/resume. ``agent_name``
    is optional and sourced from ``context.state`` when the agent seeds it.
    """
    if context is None or context.run_id is None or context.tool_call_id is None:
        raise ValueError(_MISSING_CONTEXT_MESSAGE)
    agent_name = context.state.get("agent_name") if context.state else None
    return context.run_id, context.tool_call_id, agent_name


def _format_approval_response(decision: HumanDecision, content: str | None) -> str:
    if decision == HumanDecision.APPROVE:
        return "Approved." + (f" Note: {content}" if content else "")
    if decision == HumanDecision.REJECT:
        return f"Rejected. Reason: {content}" if content else "Rejected."
    if decision == HumanDecision.OVERRIDE:
        return f"Approved with overrides: {content}" if content else "Approved with overrides."
    if decision == HumanDecision.ESCALATE:
        return f"Escalated. Note: {content}" if content else "Escalated — human cannot resolve."
    return f"Decision: {decision}" + (f" — {content}" if content else "")


def _format_answer_response(content: str | None) -> str:
    if content:
        return f"Human response: {content}"
    return "Human provided no answer."


def create_request_approval_tool(provider: HumanInputProvider) -> FunctionTool:
    """Create a tool that lets an agent request human approval.

    Emits ``HumanInputRequestEvent`` and ``HumanInputResponseEvent``.

    The tool derives its ``request_id`` from the ambient ``ToolContext`` as
    ``f"{run_id}:{tool_call_id}"`` so a logical request keeps the same
    identity across suspend/resume re-execution. The agent must be
    constructed with a ``run_id`` (or the registry's ``tool_state`` must
    supply one) and the tool must be dispatched via a ``ToolCall``.

    Args:
        provider: The provider that delivers requests to a human.
    """

    @tool(
        name="request_approval",
        description=(
            "Request human authorization before proceeding with an action. Use this "
            "before actions that are significant, irreversible, or where you want "
            "explicit sign-off. The run pauses until the human decides, then continues "
            "with their decision."
        ),
    )
    async def request_approval(
        action: str,
        reason: str,
        details: str | None = None,
        context: ToolContext | None = None,
    ) -> str:
        run_id, tool_call_id, agent_name = _require_identity(context)
        request_id = f"{run_id}:{tool_call_id}"
        prompt = f"Action: {action}\nReason: {reason}"
        if details:
            prompt += f"\nDetails: {details}"

        metadata = {"action": action, "reason": reason}
        request = HumanInputRequest(
            request_id=request_id,
            run_id=run_id,
            request_type=HumanInputType.APPROVAL,
            prompt=prompt,
            context=details,
            metadata=metadata,
            agent_name=agent_name,
        )

        emitter = context.emitter if context is not None else None
        if emitter is not None:
            emitter.emit(
                HumanInputRequestEvent(
                    trace_id=emitter.trace_id,
                    span_id=emitter.span_id,
                    parent_span_id=emitter.parent_span_id,
                    request_id=request_id,
                    request_type=HumanInputType.APPROVAL.value,
                    prompt=prompt,
                    context=details,
                    agent_name=agent_name,
                    metadata=metadata,
                )
            )

        start = time.monotonic()
        response = await provider.request_input(request)
        wait_ms = int((time.monotonic() - start) * 1000)

        if emitter is not None:
            emitter.emit(
                HumanInputResponseEvent(
                    trace_id=emitter.trace_id,
                    span_id=emitter.span_id,
                    parent_span_id=emitter.parent_span_id,
                    request_id=request_id,
                    decision=response.decision.value,
                    has_content=response.content is not None,
                    wait_duration_ms=wait_ms,
                )
            )

        return _format_approval_response(response.decision, response.content)

    return request_approval


def create_ask_human_tool(provider: HumanInputProvider) -> FunctionTool:
    """Create a tool that lets an agent ask the human a question.

    Emits ``HumanInputRequestEvent`` and ``HumanInputResponseEvent``.

    The tool derives its ``request_id`` from the ambient ``ToolContext`` as
    ``f"{run_id}:{tool_call_id}"``. See :func:`create_request_approval_tool`
    for details on identity requirements.

    Args:
        provider: The provider that delivers requests to a human.
    """

    @tool(
        name="ask_human",
        description=(
            "Ask a person a question and receive their answer. Use this when you "
            "need clarification, additional context, or information only a person "
            "can provide. The run pauses until they respond, then continues with "
            "their answer."
        ),
    )
    async def ask_human(
        question: str,
        context_info: str | None = None,
        options: list[str] | None = None,
        context: ToolContext | None = None,
    ) -> str:
        run_id, tool_call_id, agent_name = _require_identity(context)
        request_id = f"{run_id}:{tool_call_id}"

        metadata = {"question": question}
        request = HumanInputRequest(
            request_id=request_id,
            run_id=run_id,
            request_type=HumanInputType.QUESTION,
            prompt=question,
            context=context_info,
            options=options,
            metadata=metadata,
            agent_name=agent_name,
        )

        emitter = context.emitter if context is not None else None
        if emitter is not None:
            emitter.emit(
                HumanInputRequestEvent(
                    trace_id=emitter.trace_id,
                    span_id=emitter.span_id,
                    parent_span_id=emitter.parent_span_id,
                    request_id=request_id,
                    request_type=HumanInputType.QUESTION.value,
                    prompt=question,
                    context=context_info,
                    agent_name=agent_name,
                    metadata=metadata,
                )
            )

        start = time.monotonic()
        response = await provider.request_input(request)
        wait_ms = int((time.monotonic() - start) * 1000)

        if emitter is not None:
            emitter.emit(
                HumanInputResponseEvent(
                    trace_id=emitter.trace_id,
                    span_id=emitter.span_id,
                    parent_span_id=emitter.parent_span_id,
                    request_id=request_id,
                    decision=response.decision.value,
                    has_content=response.content is not None,
                    wait_duration_ms=wait_ms,
                )
            )

        return _format_answer_response(response.content)

    return ask_human


def create_hitl_tools(provider: HumanInputProvider) -> list[FunctionTool]:
    """Create HITL tools that let an agent request human input.

    Convenience function that returns both tools. For individual tools, use
    :func:`create_request_approval_tool` or :func:`create_ask_human_tool`.

    Returns two tools:

    - ``request_approval`` — agent asks for approval before a significant action.
    - ``ask_human`` — agent asks a question when it needs clarification.

    Both emit ``HumanInputRequestEvent`` and ``HumanInputResponseEvent`` and
    both derive ``request_id`` from the ambient ``ToolContext``.

    Args:
        provider: The provider that delivers requests to a human.

    Returns:
        List of two FunctionTool instances.
    """
    return [
        create_request_approval_tool(provider),
        create_ask_human_tool(provider),
    ]
