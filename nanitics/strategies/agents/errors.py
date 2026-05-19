from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from nanitics.infrastructure.observability.emitter import EventEmitter


class ErrorHandling(Protocol):
    """Protocol for agent error recovery strategies.

    The base :class:`~nanitics.strategies.agents.base.Agent` delegates to an
    ``ErrorHandling`` instance to decide how to respond to LLM errors,
    tool errors, and schema violations — whether to retry, feed a
    correction prompt back to the model, degrade gracefully, or
    propagate the failure. The default implementation is
    :class:`~nanitics.capabilities.errors.handler.ErrorHandler`.
    """

    async def handle_llm_error(
        self,
        error: Exception,
        retry_fn: Callable[[], Awaitable[object]],
        emitter: EventEmitter | None = None,
    ) -> object:
        """Handle a failed LLM generation, optionally retrying via ``retry_fn``."""
        ...

    def handle_tool_error(
        self,
        error: Exception,
        attempt: int,
        available_tools: list[str],
    ) -> str | None:
        """Return a correction message for the LLM, or ``None`` to give up."""
        ...

    def handle_llm_correction(
        self,
        error: Exception,
        attempt: int,
    ) -> str | None:
        """Return a correction prompt for an LLM schema violation, or ``None``."""
        ...

    def should_degrade(self, error: Exception, attempt: int) -> bool:
        """Whether to stop retrying and return a degraded response instead."""
        ...

    def format_degradation_message(self, error: Exception) -> str:
        """Build the message injected into the conversation on degradation."""
        ...

    def reset(self) -> None:
        """Clear per-run state before a new agent run."""
        ...

    @property
    def total_corrections(self) -> int:
        """Number of correction prompts issued so far in the current run."""
        ...

    @property
    def max_corrections(self) -> int:
        """Maximum correction prompts allowed per run."""
        ...

    def restore(self, total_corrections: int) -> None:
        """Restore correction counter when resuming from a checkpoint."""
        ...
