from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterable
from typing import Any

from nanitics.infrastructure.errors import (
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolParameterError,
    ToolTimeoutError,
)
from nanitics.infrastructure.llm.protocol import ToolCall, ToolSchema
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import (
    ToolInvokeEvent,
    ToolResultEvent,
)

from .conditional import ConditionalTool
from .context import ToolContext, _current_tool_context
from .function_tool import FunctionTool
from .protocol import Tool, ToolResult


class ToolRegistry:
    """Manages a collection of tools and dispatches LLM tool calls.

    The registry stores tools by name, validates parameters, sets the
    :class:`~nanitics.core.tools.context.ToolContext` for each invocation,
    and emits ``ToolInvokeEvent`` / ``ToolResultEvent`` for observability.

    Agents create a ``ToolRegistry`` internally from the ``tools`` list
    passed at construction.  You can also use it directly for lower-level
    control.

    Args:
        emitter: Event emitter for tool invocation and result events.
        tool_state: Per-run state dict injected into ``ToolContext.state``
            for every tool invocation.
    """

    def __init__(
        self,
        emitter: EventEmitter | None = None,
        tool_state: dict[str, Any] | None = None,
        *,
        emitter_provider: Callable[[], EventEmitter | None] | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._static_emitter = emitter
        self._emitter_provider = emitter_provider
        self._tool_state = tool_state or {}

    @property
    def _emitter(self) -> EventEmitter | None:
        """Emitter used for tool events.

        Resolves through the ``emitter_provider`` callable if one was
        supplied (so the registry follows its owning agent's per-task
        bound emitter); otherwise the static emitter passed at
        construction.
        """
        if self._emitter_provider is not None:
            return self._emitter_provider()
        return self._static_emitter

    def register(self, tool: Tool) -> None:
        """Register a tool.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        name = tool.schema.name
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = tool

    def register_all(self, tools: Iterable[Tool]) -> None:
        """Register multiple tools at once."""
        for t in tools:
            self.register(t)

    def get(self, name: str) -> Tool:
        """Retrieve a tool by name.

        Raises:
            ToolNotFoundError: If no tool with the given name exists.
        """
        if name not in self._tools:
            raise ToolNotFoundError(
                f"Tool not found: {name}",
                tool_name=name,
            )
        return self._tools[name]

    def list_schemas(self) -> list[ToolSchema]:
        """Return the schemas of all currently visible tools.

        Tools wrapped in :class:`ConditionalTool` are excluded when
        their ``is_enabled`` predicate returns ``False`` for the
        current ``tool_state``.
        """
        schemas: list[ToolSchema] = []
        for t in self._tools.values():
            if isinstance(t, ConditionalTool) and not t.is_enabled(self._tool_state):
                continue
            schemas.append(t.schema)
        return schemas

    def unregister(self, name: str) -> None:
        """Remove a tool by name.

        Raises:
            ToolNotFoundError: If no tool with the given name exists.
        """
        if name not in self._tools:
            raise ToolNotFoundError(
                f"Tool not found: {name}",
                tool_name=name,
            )
        del self._tools[name]

    def update_state(self, key: str, value: Any) -> None:
        """Update a single key in the shared tool state.

        The updated value will be visible to all subsequent tool
        invocations via ``ToolContext.state``.
        """
        self._tool_state[key] = value

    def has(self, name: str) -> bool:
        """Check whether a tool with the given name is registered."""
        return name in self._tools

    async def dispatch(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call.

        Looks up the tool, validates required parameters (for
        schema-only tools), sets the ``ToolContext``, runs the tool,
        and emits events.

        Emission rule: a ``ToolInvokeEvent`` followed by a paired
        ``ToolResultEvent`` is emitted when the tool executes —
        successfully, or by raising an exception, or when
        parameter-validation fails pre-execute. When a transparent
        wrapper short-circuits execution by returning
        ``ToolResult(executed=False)``, neither event is emitted; the
        returned result is still propagated to the caller so the agent
        can surface ``result.content`` to the LLM. In every branch
        where both events fire, ``ToolInvokeEvent`` is emitted strictly
        before ``ToolResultEvent``.

        Args:
            tool_call: The tool call from the LLM containing name and
                arguments.

        Returns:
            The tool result.

        Raises:
            ToolNotFoundError: If the tool is not registered.
            ToolParameterError: If required parameters are missing.
            ToolExecutionError: If the tool raises a non-``ToolError``
                exception.
        """
        tool = self.get(tool_call.name)

        if isinstance(tool, ConditionalTool) and not tool.is_enabled(self._tool_state):
            raise ToolNotFoundError(
                f"Tool '{tool_call.name}' is not currently available",
                tool_name=tool_call.name,
            )

        start = time.perf_counter()

        # Validate required parameters for schema-only tools (no Pydantic model).
        # This is logically part of the invoke (we tried to dispatch; validation
        # rejected it pre-execute), so we still emit invoke+result for this path.
        if not (isinstance(tool, FunctionTool) and tool.parameters_model is not None):
            required = tool.schema.parameters.get("required", [])
            missing = [r for r in required if r not in tool_call.arguments]
            if missing:
                duration_ms = (time.perf_counter() - start) * 1000
                error_msg = f"Missing required parameters: {', '.join(missing)}"
                self._emit_invoke(tool_call)
                self._emit_result(
                    tool_call,
                    success=False,
                    error=error_msg,
                    duration_ms=duration_ms,
                )
                msg = f"Parameter validation failed for tool '{tool_call.name}': {error_msg}"
                raise ToolParameterError(
                    msg,
                    tool_name=tool_call.name,
                    reason=error_msg,
                )

        # Execute
        timeout = tool.schema.timeout_seconds
        token = _current_tool_context.set(
            ToolContext(
                emitter=self._emitter,
                state=self._tool_state,
                run_id=self._tool_state.get("run_id"),
                tool_call_id=tool_call.id,
            )
        )
        try:
            if timeout is not None:
                result = await asyncio.wait_for(
                    tool.execute(**tool_call.arguments),
                    timeout=timeout,
                )
            else:
                result = await tool.execute(**tool_call.arguments)
        except TimeoutError:
            duration_ms = (time.perf_counter() - start) * 1000
            error_msg = f"Tool '{tool_call.name}' timed out after {timeout}s"
            self._emit_invoke(tool_call)
            self._emit_result(
                tool_call,
                success=False,
                error=error_msg,
                duration_ms=duration_ms,
            )
            raise ToolTimeoutError(
                error_msg,
                tool_name=tool_call.name,
                timeout_seconds=timeout or 0.0,
            ) from None
        except ToolError as e:
            duration_ms = (time.perf_counter() - start) * 1000
            self._emit_invoke(tool_call)
            self._emit_result(
                tool_call,
                success=False,
                error=str(e),
                duration_ms=duration_ms,
            )
            raise
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            self._emit_invoke(tool_call)
            self._emit_result(
                tool_call,
                success=False,
                error=str(e),
                duration_ms=duration_ms,
            )
            raise ToolExecutionError(
                f"Tool '{tool_call.name}' failed: {e}",
                tool_name=tool_call.name,
            ) from e
        finally:
            _current_tool_context.reset(token)

        # Success path: gate emission on whether the tool actually executed.
        # A wrapper that short-circuits returns executed=False to signal the
        # registry to suppress both events. The result is still returned to
        # the caller so its content can be surfaced to the LLM.
        if result.executed:
            duration_ms = (time.perf_counter() - start) * 1000
            self._emit_invoke(tool_call)
            self._emit_result(
                tool_call,
                success=True,
                result=result.content,
                duration_ms=duration_ms,
            )
        return result

    def _emit_invoke(self, tool_call: ToolCall) -> None:
        if self._emitter is None:
            return
        self._emitter.emit(
            ToolInvokeEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                parameters=tool_call.arguments,
            )
        )

    def _emit_result(
        self,
        tool_call: ToolCall,
        *,
        success: bool,
        error: str | None = None,
        result: str | None = None,
        duration_ms: float,
    ) -> None:
        if self._emitter is None:
            return
        self._emitter.emit(
            ToolResultEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                result=result,
                error=error,
                success=success,
                duration_ms=duration_ms,
            )
        )
