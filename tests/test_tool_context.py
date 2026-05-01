"""Tests for ToolContext injection into tools."""

from uuid import uuid4

import pytest

from nanitics.core.tools import ToolContext, ToolRegistry, tool
from nanitics.core.tools.context import _current_tool_context
from nanitics.infrastructure.errors import ToolExecutionError
from nanitics.infrastructure.llm.protocol import ToolCall
from nanitics.infrastructure.observability.emitter import InMemoryEmitter


def _make_tool_call(name: str = "test_tool", **arguments: object) -> ToolCall:
    return ToolCall(id=str(uuid4()), name=name, arguments=arguments)


class TestToolContextVar:
    async def test_dispatch_sets_contextvar_and_resets_after(self) -> None:
        captured: list[ToolContext | None] = []

        @tool(name="capture", description="Captures context")
        async def capture(context: ToolContext) -> str:
            captured.append(context)
            return "ok"

        emitter = InMemoryEmitter(trace_id="t1")
        registry = ToolRegistry(emitter=emitter)
        registry.register(capture)

        assert _current_tool_context.get() is None

        await registry.dispatch(_make_tool_call("capture"))

        assert captured[0] is not None
        assert captured[0].emitter is emitter
        # Contextvar is reset after dispatch
        assert _current_tool_context.get() is None

    async def test_contextvar_reset_after_error(self) -> None:
        @tool(name="failing", description="Fails")
        async def failing(context: ToolContext) -> str:
            raise RuntimeError("boom")

        emitter = InMemoryEmitter(trace_id="t1")
        registry = ToolRegistry(emitter=emitter)
        registry.register(failing)

        with pytest.raises(ToolExecutionError):
            await registry.dispatch(_make_tool_call("failing"))

        assert _current_tool_context.get() is None


class TestToolContextInjection:
    async def test_tool_with_context_receives_it(self) -> None:
        received_emitter = []

        @tool(name="ctx_tool", description="Tool with context")
        async def ctx_tool(value: str, context: ToolContext) -> str:
            received_emitter.append(context.emitter)
            return f"got {value}"

        emitter = InMemoryEmitter(trace_id="t1")
        registry = ToolRegistry(emitter=emitter)
        registry.register(ctx_tool)

        result = await registry.dispatch(_make_tool_call("ctx_tool", value="hello"))
        assert result.content == "got hello"
        assert received_emitter[0] is emitter

    async def test_tool_without_context_works_unchanged(self) -> None:
        @tool(name="plain", description="No context needed")
        async def plain(x: int, y: int) -> str:
            return str(x + y)

        registry = ToolRegistry()
        registry.register(plain)

        result = await registry.dispatch(_make_tool_call("plain", x=3, y=4))
        assert result.content == "7"

    async def test_context_excluded_from_parameter_schema(self) -> None:
        @tool(name="ctx_tool", description="Tool with context")
        async def ctx_tool(query: str, context: ToolContext) -> str:
            return query

        schema = ctx_tool.schema
        props = schema.parameters.get("properties", {})
        assert "query" in props
        assert "context" not in props

        required = schema.parameters.get("required", [])
        assert "context" not in required

    async def test_context_injection_with_async_tool(self) -> None:
        import asyncio

        captured: list[ToolContext | None] = []

        @tool(name="async_ctx", description="Async tool with context")
        async def async_ctx(context: ToolContext) -> str:
            await asyncio.sleep(0)
            captured.append(context)
            return "done"

        emitter = InMemoryEmitter(trace_id="t1")
        registry = ToolRegistry(emitter=emitter)
        registry.register(async_ctx)

        await registry.dispatch(_make_tool_call("async_ctx"))

        assert captured[0] is not None
        assert captured[0].emitter is emitter

    async def test_context_none_without_emitter(self) -> None:
        captured: list[ToolContext | None] = []

        @tool(name="no_emitter", description="No emitter on registry")
        async def no_emitter(context: ToolContext) -> str:
            captured.append(context)
            return "ok"

        registry = ToolRegistry()  # no emitter
        registry.register(no_emitter)

        await registry.dispatch(_make_tool_call("no_emitter"))

        assert captured[0] is not None
        assert captured[0].emitter is None


class TestToolContextState:
    def test_default_state_is_empty_dict(self) -> None:
        ctx = ToolContext()
        assert ctx.state == {}

    def test_explicit_state_is_accessible(self) -> None:
        ctx = ToolContext(state={"key": "val"})
        assert ctx.state["key"] == "val"

    async def test_registry_passes_tool_state_to_context(self) -> None:
        captured: list[ToolContext | None] = []

        @tool(name="capture_state", description="Capture state")
        async def capture_state(context: ToolContext) -> str:
            captured.append(context)
            return "ok"

        emitter = InMemoryEmitter(trace_id="t1")
        registry = ToolRegistry(emitter=emitter, tool_state={"k": "v"})
        registry.register(capture_state)

        await registry.dispatch(_make_tool_call("capture_state"))

        assert captured[0] is not None
        assert captured[0].state["k"] == "v"

    async def test_registry_without_tool_state_passes_empty_dict(self) -> None:
        captured: list[ToolContext | None] = []

        @tool(name="capture_state", description="Capture state")
        async def capture_state(context: ToolContext) -> str:
            captured.append(context)
            return "ok"

        registry = ToolRegistry()
        registry.register(capture_state)

        await registry.dispatch(_make_tool_call("capture_state"))

        assert captured[0] is not None
        assert captured[0].state == {}

    async def test_tool_can_read_state(self) -> None:
        @tool(name="reader", description="Reads state")
        async def reader(context: ToolContext) -> str:
            return str(context.state.get("greeting", "none"))

        emitter = InMemoryEmitter(trace_id="t1")
        registry = ToolRegistry(emitter=emitter, tool_state={"greeting": "hello"})
        registry.register(reader)

        result = await registry.dispatch(_make_tool_call("reader"))
        assert result.content == "hello"

    async def test_update_state_adds_new_key(self) -> None:
        @tool(name="reader", description="Reads state")
        async def reader(context: ToolContext) -> str:
            return str(context.state.get("b", "missing"))

        emitter = InMemoryEmitter(trace_id="t1")
        registry = ToolRegistry(emitter=emitter, tool_state={"a": 1})
        registry.register(reader)

        registry.update_state("b", 2)

        result = await registry.dispatch(_make_tool_call("reader"))
        assert result.content == "2"


class TestToolContextRunIdAndToolCallId:
    async def test_dispatch_populates_run_id_and_tool_call_id(self) -> None:
        captured: list[ToolContext | None] = []

        @tool(name="capture", description="Captures context")
        async def capture(context: ToolContext) -> str:
            captured.append(context)
            return "ok"

        registry = ToolRegistry(tool_state={"run_id": "r-1"})
        registry.register(capture)

        tc = ToolCall(id="tc-99", name="capture", arguments={})
        await registry.dispatch(tc)

        assert captured[0] is not None
        assert captured[0].run_id == "r-1"
        assert captured[0].tool_call_id == "tc-99"

    async def test_run_id_defaults_to_none_without_tool_state(self) -> None:
        captured: list[ToolContext | None] = []

        @tool(name="capture", description="Captures context")
        async def capture(context: ToolContext) -> str:
            captured.append(context)
            return "ok"

        registry = ToolRegistry()
        registry.register(capture)

        tc = ToolCall(id="tc-1", name="capture", arguments={})
        await registry.dispatch(tc)

        assert captured[0] is not None
        assert captured[0].run_id is None
        assert captured[0].tool_call_id == "tc-1"

    async def test_update_state_overwrites_existing_key(self) -> None:
        @tool(name="reader", description="Reads state")
        async def reader(context: ToolContext) -> str:
            return str(context.state.get("a", "missing"))

        emitter = InMemoryEmitter(trace_id="t1")
        registry = ToolRegistry(emitter=emitter, tool_state={"a": 1})
        registry.register(reader)

        registry.update_state("a", 99)

        result = await registry.dispatch(_make_tool_call("reader"))
        assert result.content == "99"
