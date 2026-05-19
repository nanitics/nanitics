"""Tests for ConditionalTool and ToolRegistry conditional filtering."""

import pytest

from nanitics import (
    ToolRegistry,
    tool,
)
from nanitics.infrastructure.errors import ToolNotFoundError
from nanitics.infrastructure.llm.protocol import ToolCall
from nanitics.infrastructure.observability.events import ToolInvokeEvent
from nanitics.specialized import ConditionalTool
from tests.testing_helpers import make_emitter

# -- Fixtures ----------------------------------------------------------------


@tool("always_tool", "A normal tool")
async def always_tool(x: str) -> str:
    return f"always: {x}"


def _make_conditional(*, enabled: bool = True) -> ConditionalTool:
    @tool("conditional_tool", "A conditional tool")
    async def inner(x: str) -> str:
        return f"conditional: {x}"

    return ConditionalTool(tool=inner, is_enabled=lambda state: enabled)


def _make_state_conditional() -> ConditionalTool:
    @tool("state_tool", "Reads state")
    async def inner(x: str) -> str:
        return f"state: {x}"

    return ConditionalTool(
        tool=inner,
        is_enabled=lambda state: state.get("approved", False),
    )


# -- ConditionalTool unit tests ---------------------------------------------


class TestConditionalToolConstruction:
    def test_wraps_tool_and_stores_predicate(self) -> None:
        ct = _make_conditional(enabled=True)
        assert ct.schema.name == "conditional_tool"
        assert ct.is_enabled({}) is True

    def test_schema_delegates_to_inner_tool(self) -> None:
        ct = _make_conditional()
        assert ct.schema.name == "conditional_tool"
        assert ct.schema.description == "A conditional tool"
        assert "x" in ct.schema.parameters.get("properties", {})

    @pytest.mark.anyio
    async def test_execute_delegates_to_inner_tool(self) -> None:
        ct = _make_conditional()
        result = await ct.execute(x="hello")
        assert result.content == "conditional: hello"

    def test_is_enabled_true_predicate(self) -> None:
        ct = _make_conditional(enabled=True)
        assert ct.is_enabled({}) is True

    def test_is_enabled_false_predicate(self) -> None:
        ct = _make_conditional(enabled=False)
        assert ct.is_enabled({}) is False


# -- ToolRegistry integration tests -----------------------------------------


class TestRegistryListSchemas:
    def test_enabled_conditional_included(self) -> None:
        reg = ToolRegistry()
        reg.register(_make_conditional(enabled=True))
        schemas = reg.list_schemas()
        assert len(schemas) == 1
        assert schemas[0].name == "conditional_tool"

    def test_disabled_conditional_excluded(self) -> None:
        reg = ToolRegistry()
        reg.register(_make_conditional(enabled=False))
        schemas = reg.list_schemas()
        assert len(schemas) == 0

    def test_mix_of_conditional_and_unconditional(self) -> None:
        reg = ToolRegistry()
        reg.register(always_tool)
        reg.register(_make_conditional(enabled=False))
        schemas = reg.list_schemas()
        assert len(schemas) == 1
        assert schemas[0].name == "always_tool"

    def test_predicate_receives_tool_state(self) -> None:
        ct = _make_state_conditional()
        reg = ToolRegistry(tool_state={"approved": True})
        reg.register(ct)
        schemas = reg.list_schemas()
        assert len(schemas) == 1
        assert schemas[0].name == "state_tool"

    def test_predicate_receives_tool_state_false(self) -> None:
        ct = _make_state_conditional()
        reg = ToolRegistry(tool_state={})
        reg.register(ct)
        schemas = reg.list_schemas()
        assert len(schemas) == 0


class TestRegistryDispatch:
    @pytest.mark.anyio
    async def test_enabled_conditional_dispatches(self) -> None:
        emitter = make_emitter()
        reg = ToolRegistry(emitter=emitter)
        reg.register(_make_conditional(enabled=True))
        call = ToolCall(id="tc-1", name="conditional_tool", arguments={"x": "hi"})
        result = await reg.dispatch(call)
        assert result.content == "conditional: hi"
        invoke_events = [e for e in emitter.events if isinstance(e, ToolInvokeEvent)]
        assert len(invoke_events) == 1

    @pytest.mark.anyio
    async def test_disabled_conditional_raises(self) -> None:
        emitter = make_emitter()
        reg = ToolRegistry(emitter=emitter)
        reg.register(_make_conditional(enabled=False))
        call = ToolCall(id="tc-1", name="conditional_tool", arguments={"x": "hi"})
        with pytest.raises(ToolNotFoundError, match="not currently available"):
            await reg.dispatch(call)
        # No ToolInvokeEvent emitted for disabled tool
        invoke_events = [e for e in emitter.events if isinstance(e, ToolInvokeEvent)]
        assert len(invoke_events) == 0


class TestRegistryHas:
    def test_has_returns_true_for_disabled_conditional(self) -> None:
        reg = ToolRegistry()
        reg.register(_make_conditional(enabled=False))
        assert reg.has("conditional_tool") is True
