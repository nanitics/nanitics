from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from nanitics.infrastructure.llm.protocol import ToolSchema


class ToolResult(BaseModel):
    """Immutable result returned by a tool execution.

    The ``content`` field is the text the LLM sees in the conversation
    history.  ``metadata`` carries structured data for application logic.
    The agent that consumes the registry's dispatch result copies it onto
    the ``tool_result`` ``Message.metadata`` so application code that
    inspects the conversation (for example, a ``TruncationPolicy`` reading
    ``metadata['protected']``) sees what the tool surfaced. LLM providers
    strip ``Message.metadata`` at serialization — it is never sent to the
    model.

    The ``executed`` flag is a wrapper-signalling mechanism consumed by
    :class:`~nanitics.strategies.tools.registry.ToolRegistry`:

    - It defaults to ``True`` so existing tools are unaffected — every
      tool that actually runs its work returns ``executed=True`` by
      construction.
    - Transparent wrappers that short-circuit execution before calling
      the inner tool (for example,
      :class:`~nanitics.collaboration.approval_wrapped.ApprovalWrappedTool`
      on the reject path) should return ``ToolResult(..., executed=False)``
      so the registry suppresses the paired ``ToolInvokeEvent`` /
      ``ToolResultEvent`` pair that would otherwise misrepresent the
      trace as a real invocation.
    - The flag is never sent to the LLM; only ``content`` is. It is a
      decision signal for the registry, not a conversation field.
    - A raised exception from ``execute()`` does not set ``executed`` —
      the flag is a pre-execution gating signal for wrappers, not an
      exception-handling signal. The registry's exception handlers
      continue to emit invoke/result pairs with ``success=False`` on
      raised errors.

    Attributes:
        content: Text returned to the agent.
        metadata: Arbitrary metadata for application use. Propagated onto
            the ``tool_result`` ``Message.metadata``; not sent to the LLM.
        executed: ``True`` (default) when the tool's work ran; ``False``
            when a wrapper short-circuited before executing the inner
            tool, signalling the registry to suppress invoke/result
            events for this call.
    """

    model_config = ConfigDict(frozen=True)

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    executed: bool = Field(
        default=True,
        description=(
            "Whether the underlying tool actually executed. Wrappers that "
            "gate execution (e.g. approval wrappers on reject) set this "
            "to False so the registry suppresses ToolInvokeEvent / "
            "ToolResultEvent for the call."
        ),
    )


@runtime_checkable
class Tool(Protocol):
    """Structural protocol that all tools must satisfy.

    Any object with a ``schema`` property returning a
    :class:`~nanitics.infrastructure.llm.protocol.ToolSchema` and an async
    ``execute(**params)`` method returning a :class:`ToolResult` is a valid
    tool.  No inheritance required.
    """

    @property
    def schema(self) -> ToolSchema:
        """Describe the tool's interface for the LLM."""
        ...

    async def execute(self, **params: Any) -> ToolResult:
        """Run the tool with the given parameters and return a result."""
        ...
