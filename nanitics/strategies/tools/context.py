from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from nanitics.infrastructure.observability.emitter import EventEmitter


@dataclass(frozen=True)
class ToolContext:
    """Runtime context injected into tool functions.

    Tools receive a ``ToolContext`` instance when they declare a parameter
    with the ``ToolContext`` type annotation.  The context is set
    automatically by :class:`~nanitics.strategies.tools.registry.ToolRegistry`
    during dispatch and is not visible in the tool schema the LLM sees.

    Attributes:
        emitter: The event emitter from the registry, or ``None``.
        state: Per-run state dict supplied via ``tool_state`` on the
            agent or registry.  Shared across all tools in the same run.
        run_id: The run identifier from the agent or registry ``tool_state``,
            or ``None`` if not provided.
        tool_call_id: The ``ToolCall.id`` from the dispatching call,
            or ``None`` if not dispatched via a registry.
    """

    emitter: EventEmitter | None = None
    state: Mapping[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    tool_call_id: str | None = None


_current_tool_context: ContextVar[ToolContext | None] = ContextVar("current_tool_context", default=None)
