"""ConditionalTool: state-driven tool visibility.

Wraps any :class:`~nanitics.strategies.tools.protocol.Tool` with a predicate
that controls whether the tool appears in the schema list sent to the LLM.
When the predicate returns ``False``, the tool is hidden from the agent
and cannot be invoked.

Example::

    from nanitics import ConditionalTool, tool

    @tool("delete_record", "Delete a database record")
    async def delete_record(record_id: str) -> str:
        return f"Deleted {record_id}"

    conditional = ConditionalTool(
        tool=delete_record,
        is_enabled=lambda state: state.get("approved", False),
    )

    conditional.is_enabled({})                    # False — tool hidden
    conditional.is_enabled({"approved": True})    # True  — tool visible
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from nanitics.infrastructure.llm.protocol import ToolSchema

from .protocol import Tool, ToolResult


class ConditionalTool:
    """Wraps a tool with a predicate controlling its visibility.

    The wrapped tool appears in
    :meth:`~nanitics.strategies.tools.registry.ToolRegistry.list_schemas` only
    when ``is_enabled`` returns ``True`` for the current ``tool_state``.
    Schema and execution delegate unchanged to the inner tool.

    Args:
        tool: The tool to wrap.
        is_enabled: Predicate receiving the ``tool_state`` dict. Returns
            ``True`` when the tool should be visible to the LLM.
    """

    def __init__(
        self,
        tool: Tool,
        is_enabled: Callable[[Mapping[str, Any]], bool],
    ) -> None:
        self._tool = tool
        self._is_enabled = is_enabled

    @property
    def schema(self) -> ToolSchema:
        """Return the inner tool's schema unchanged."""
        return self._tool.schema

    async def execute(self, **params: Any) -> ToolResult:
        """Delegate execution to the inner tool."""
        return await self._tool.execute(**params)

    def is_enabled(self, state: Mapping[str, Any]) -> bool:
        """Evaluate the predicate against the given state."""
        return self._is_enabled(state)
