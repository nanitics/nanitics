from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from nanitics.infrastructure.llm.protocol import Message, ToolSchema
from nanitics.infrastructure.observability.emitter import EventEmitter


class ContextContent(BaseModel):
    """Content returned by a context provider for injection into the LLM context.

    Before the LLM sees the content, :meth:`Agent._inject_context`
    wraps the ``content`` string in a namespaced
    ``<nanitics:context provider="…" priority="…" protected="…">…</nanitics:context>``
    block and emits it as a ``role="user"`` message. The wrapper is
    the structural signal that tells the LLM "this is SDK-injected
    context, not user speech." Providers return raw strings; the SDK
    owns the wire shape. See :meth:`Agent._inject_context` for the
    authoritative wrapper spec.

    Attributes:
        content: The text content to inject. Any leading human-readable
            label the provider chooses (e.g. ``[Working Memory]``,
            ``[Past Experiences]``) is preserved verbatim inside the
            wrapper body.
        priority: Ordering priority when multiple providers contribute.
            Lower values are higher priority (injected first). Default: 0.
        protected: If True, this content cannot be truncated by the
            context manager. Default: False.
        provider_name: Identifier for the provider that generated this
            content. Used in observability events and rendered as the
            ``provider`` attribute on the wrapper's opening tag.
    """

    model_config = ConfigDict(frozen=True)

    content: str
    priority: int = 0
    protected: bool = False
    provider_name: str = ""


@runtime_checkable
class ContextProvider(Protocol):
    """Protocol for providing additional context before each LLM call.

    Context providers are called before every LLM invocation. They return
    content that is injected into the message sequence, giving the agent
    access to dynamic state (working memory, episodic memory, shared
    memory board, etc.) without explicit tool calls.
    """

    async def provide(self, messages: list[Message]) -> ContextContent | None:
        """Generate context content based on the current conversation.

        Args:
            messages: The current message history.

        Returns:
            Content to inject, or None if nothing to contribute.
        """
        ...


class ContextManagement(Protocol):
    """Protocol for managing the context window before each LLM call.

    Implementations inspect the full message history and return a trimmed
    or summarized version that fits within the model's token budget. The
    agent calls ``reset()`` at the start of each run so per-run state
    (e.g. summarization memory) does not leak across runs.
    """

    async def prepare(
        self,
        system_prompt: str,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        emitter: EventEmitter | None,
    ) -> list[Message]:
        """Return the message list to send to the LLM for this call."""
        ...

    def reset(self) -> None:
        """Reset any per-run state accumulated during the previous run."""
        ...
