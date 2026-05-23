"""Regression tests: ``asyncio.CancelledError`` is never wrapped by the
tool-dispatch error machinery.

``CancelledError`` is ``BaseException`` on Python >= 3.8, so it already
escapes ``except Exception`` clauses; these tests pin the contract so a
future broadening to ``except BaseException`` would surface a failing
test rather than silently breaking cancellation.
"""

from __future__ import annotations

import asyncio

import pytest

from nanitics.infrastructure.llm.protocol import ToolCall, ToolSchema
from nanitics.infrastructure.observability.emitter import InMemoryEmitter
from nanitics.strategies.tools import Tool, ToolRegistry
from nanitics.strategies.tools.protocol import ToolResult


class _CancellingTool:
    schema = ToolSchema(
        name="x",
        description="raises CancelledError",
        parameters={"type": "object", "properties": {}, "required": []},
    )

    async def execute(self, **kwargs: object) -> ToolResult:
        raise asyncio.CancelledError()


async def test_tool_registry_propagates_cancelled_error() -> None:
    registry = ToolRegistry(emitter=InMemoryEmitter(trace_id="t"))
    registry.register(_CancellingTool())
    tc = ToolCall(id="1", name="x", arguments={})
    with pytest.raises(asyncio.CancelledError):
        await registry.dispatch(tc)


async def test_mcptool_propagates_cancelled_error() -> None:
    """``MCPTool.execute`` must not wrap ``CancelledError`` into ``LLMProviderError``."""
    from nanitics.infrastructure.mcp._tool import MCPTool

    class _Session:
        async def call_tool(self, name: str, params: dict) -> object:
            raise asyncio.CancelledError()

    schema = ToolSchema(
        name="x",
        description="d",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    tool = MCPTool(schema=schema, mcp_tool_name="x", session=_Session(), default_timeout=None)
    with pytest.raises(asyncio.CancelledError):
        await tool.execute()


# Conform to ``Tool`` protocol — silence linters that flag the cycle.
_ = Tool
