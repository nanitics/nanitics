from __future__ import annotations

import warnings
from typing import Any

from nanitics.composition.multi_agent.context_transfer import (
    ContextTransferStrategy,
    RawOutputTransfer,
)
from nanitics.infrastructure.llm.protocol import (
    ContentBlock,
    TextContentBlock,
    ToolSchema,
)
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import DelegationEvent
from nanitics.safety.cancellation import CancellationToken
from nanitics.strategies.agents.base import Agent, AgentInput
from nanitics.strategies.tools.protocol import _UNSET, ToolResult, _Unset


class AgentTool:
    """Wraps an Agent behind the Tool protocol so it can be invoked by a calling agent.

    Exposes a single ``task`` parameter. When executed, delegates the task to
    the wrapped agent, applies a ``ContextTransferStrategy`` to extract the
    result, and returns it as a ``ToolResult``. Emits a ``DelegationEvent``
    for trace linking.

    Args:
        agent: The agent to delegate tasks to.
        emitter: Event emitter for tracing delegation.
        description: Tool description visible to the calling agent's LLM.
        name: Tool name override. Defaults to ``agent.name``.
        transfer_strategy: How to extract the delegate's result.
            Defaults to ``RawOutputTransfer``.
        caller_name: Name of the calling agent, used in trace events.
        cancellation_token: Shared cancellation signal propagated to the
            delegate agent.
        content_blocks: Additional content blocks (e.g., images) to include
            alongside the task when calling the delegate. When set, the task
            string becomes a TextContentBlock and is prepended to these blocks
            to form multimodal input. When None, the task string passes
            directly to the delegate.
        thread_key: Opaque key identifying the conversation thread the
            delegate continues across repeated ``execute`` calls. Every
            invocation of this ``AgentTool`` forwards the key to the
            delegate's :meth:`~nanitics.strategies.agents.base.Agent.run`,
            so a coordinator that calls the same tool more than once sees
            the delegate accumulate its prior assistant turns, tool
            calls, and tool results as its own conversation history. The
            delegate must be configured with a
            :class:`~nanitics.composition.threads.ThreadStore` for the
            prefix to be persisted; the key is otherwise accepted and
            ignored. When ``None`` (the default) the delegate runs
            stateless across calls. See ``docs/guides/memory.md`` §
            Behavioral Continuity for the substrate distinction.
        return_direct: When ``True``, the calling ``ReActAgent`` ends its
            run on this delegation and uses the delegate's extracted
            output as the calling run's output, skipping the caller's
            closing LLM turn. Defaults to ``False``. See
            :class:`~nanitics.infrastructure.llm.protocol.ToolSchema`.
    """

    def __init__(
        self,
        *,
        agent: Agent,
        emitter: EventEmitter,
        description: str,
        name: str | None = None,
        transfer_strategy: ContextTransferStrategy | None = None,
        caller_name: str = "",
        cancellation_token: CancellationToken | None = None,
        content_blocks: list[ContentBlock] | None = None,
        thread_key: str | None = None,
        return_direct: bool = False,
    ) -> None:
        self._agent = agent
        self._emitter = emitter
        self._name = name or agent.name
        self._description = description
        self._transfer_strategy = transfer_strategy or RawOutputTransfer()
        self._caller_name = caller_name
        self.cancellation_token = cancellation_token
        self._content_blocks = content_blocks
        self._thread_key = thread_key
        self._return_direct = return_direct

    @property
    def schema(self) -> ToolSchema:
        """Tool schema with a single ``task`` string parameter."""
        return ToolSchema(
            name=self._name,
            description=self._description,
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task to delegate to the agent.",
                    },
                },
                "required": ["task"],
            },
            return_direct=self._return_direct,
        )

    def replace(
        self,
        *,
        name: str | _Unset = _UNSET,
        description: str | _Unset = _UNSET,
        return_direct: bool | _Unset = _UNSET,
    ) -> AgentTool:
        """Return a copy of this delegation tool with the given schema
        metadata replaced.

        Only ``name``, ``description``, and ``return_direct`` may be
        overridden; the wrapped agent, transfer strategy, thread key, content
        blocks, and all other configuration are preserved. Arguments left
        unset keep their current values. Returns a new instance; the original
        is untouched.

        Lets a single delegation be defined once and used both interactively
        (the caller keeps its closing turn) and headlessly
        (``return_direct=True``, the run ends on the delegate's output)::

            headless = delegate_tool.replace(return_direct=True)

        Args:
            name: New tool name, if overriding.
            description: New description, if overriding.
            return_direct: New ``return_direct`` flag, if overriding.

        Returns:
            A new ``AgentTool`` delegating to the same agent, differing only
            in the overridden schema fields.
        """
        return AgentTool(
            agent=self._agent,
            emitter=self._emitter,
            description=self._description if isinstance(description, _Unset) else description,
            name=self._name if isinstance(name, _Unset) else name,
            transfer_strategy=self._transfer_strategy,
            caller_name=self._caller_name,
            cancellation_token=self.cancellation_token,
            content_blocks=self._content_blocks,
            thread_key=self._thread_key,
            return_direct=self._return_direct if isinstance(return_direct, _Unset) else return_direct,
        )

    def with_return_direct(self, value: bool = True) -> AgentTool:
        """Return a copy of this delegation tool with ``return_direct`` set to
        ``value``.

        .. deprecated:: 0.9.0
            Use :meth:`replace` instead:
            ``tool.replace(return_direct=value)``. ``with_return_direct``
            will be removed in 1.0.
        """
        warnings.warn(
            "AgentTool.with_return_direct is deprecated; use replace(return_direct=...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.replace(return_direct=value)

    async def execute(self, **params: Any) -> ToolResult:
        """Run the delegate agent and return its extracted output.

        Emits a ``DelegationEvent``, runs the agent with the given task,
        applies the transfer strategy, and returns a ``ToolResult`` with
        metadata including ``total_steps``, ``termination_reason``, and
        ``usage``.
        """
        task: str = params["task"]

        self._emitter.emit(
            DelegationEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                caller_agent=self._caller_name,
                delegate_agent=self._agent.name,
                task=task,
                transfer_strategy=type(self._transfer_strategy).__name__,
            )
        )

        if self.cancellation_token is not None:
            self._agent.set_cancellation_token(self.cancellation_token)

        agent_input: AgentInput
        if self._content_blocks is not None:
            agent_input = [TextContentBlock(text=task), *self._content_blocks]
        else:
            agent_input = task

        result = await self._agent.bind(self._emitter).run(agent_input, thread_key=self._thread_key)

        content = await self._transfer_strategy.extract(result)

        return ToolResult(
            content=content,
            metadata={
                "total_steps": result.total_steps,
                "termination_reason": result.termination_reason,
                "usage": result.usage.model_dump(),
            },
        )
