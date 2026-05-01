"""Tests for Tool protocol, FunctionTool, @tool decorator, and ToolRegistry."""

import asyncio
from typing import Any
from uuid import uuid4

import pytest
from pydantic import BaseModel, ValidationError

from nanitics.core.tools import FunctionTool, Tool, ToolRegistry, ToolResult, tool
from nanitics.core.tools.function_tool import _model_from_function
from nanitics.infrastructure.errors import (
    ToolExecutionError,
    ToolNotFoundError,
    ToolParameterError,
    ToolTimeoutError,
)
from nanitics.infrastructure.llm.protocol import ToolCall, ToolSchema
from nanitics.infrastructure.observability.emitter import InMemoryEmitter
from nanitics.infrastructure.observability.events import (
    ToolInvokeEvent,
    ToolResultEvent,
)

# --- ToolResult ---


class TestToolResult:
    def test_construction(self) -> None:
        result = ToolResult(content="hello")
        assert result.content == "hello"
        assert result.metadata == {}

    def test_with_metadata(self) -> None:
        result = ToolResult(content="hello", metadata={"key": "value"})
        assert result.metadata == {"key": "value"}

    def test_frozen(self) -> None:
        result = ToolResult(content="hello")
        with pytest.raises(ValidationError):
            result.content = "world"

    def test_executed_defaults_to_true(self) -> None:
        result = ToolResult(content="hello")
        assert result.executed is True

    def test_executed_can_be_set_false(self) -> None:
        result = ToolResult(content="hello", executed=False)
        assert result.executed is False

    def test_executed_is_frozen(self) -> None:
        result = ToolResult(content="hello", executed=False)
        with pytest.raises(ValidationError):
            result.executed = True


# --- Tool Protocol ---


class TestToolProtocol:
    def test_manual_class_satisfies_protocol(self) -> None:
        class MyTool:
            @property
            def schema(self) -> ToolSchema:
                return ToolSchema(name="my_tool", description="A tool", parameters={})

            async def execute(self, **params: Any) -> ToolResult:
                return ToolResult(content="done")

        assert isinstance(MyTool(), Tool)

    def test_function_tool_satisfies_protocol(self) -> None:
        class Params(BaseModel):
            x: int

        async def fn(x: int) -> str:
            return str(x)

        ft = FunctionTool(fn=fn, name="t", description="d", parameters_model=Params)
        assert isinstance(ft, Tool)


# --- FunctionTool ---


class Greeting(BaseModel):
    name: str
    excited: bool = False


class TestFunctionToolWithModel:
    async def test_schema_generation(self) -> None:
        async def greet(name: str, excited: bool = False) -> str:
            return f"Hello {name}{'!' if excited else '.'}"

        ft = FunctionTool(
            fn=greet,
            name="greet",
            description="Greet someone",
            parameters_model=Greeting,
        )
        assert ft.schema.name == "greet"
        assert ft.schema.description == "Greet someone"
        assert "name" in ft.schema.parameters.get("properties", {})

    async def test_execution(self) -> None:
        async def greet(name: str, excited: bool = False) -> str:
            return f"Hello {name}{'!' if excited else '.'}"

        ft = FunctionTool(fn=greet, name="greet", description="Greet", parameters_model=Greeting)
        result = await ft.execute(name="World", excited=True)
        assert result.content == "Hello World!"

    async def test_validation_error(self) -> None:
        async def greet(name: str) -> str:
            return f"Hello {name}"

        ft = FunctionTool(fn=greet, name="greet", description="Greet", parameters_model=Greeting)
        with pytest.raises(ToolParameterError) as exc_info:
            await ft.execute(name=123, excited="not_bool")
        assert exc_info.value.tool_name == "greet"
        assert exc_info.value.reason is not None


class TestFunctionToolWithSchema:
    async def test_schema_passthrough(self) -> None:
        raw_schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }

        async def search(query: str) -> str:
            return f"Results for {query}"

        ft = FunctionTool(fn=search, name="search", description="Search", parameters_schema=raw_schema)
        assert ft.schema.parameters == raw_schema
        assert ft.parameters_model is None

    async def test_execution(self) -> None:
        async def search(query: str) -> str:
            return f"Results for {query}"

        ft = FunctionTool(
            fn=search,
            name="search",
            description="Search",
            parameters_schema={"type": "object", "properties": {}},
        )
        result = await ft.execute(query="test")
        assert result.content == "Results for test"


class TestFunctionToolReturnTypes:
    async def test_str_return_wrapped(self) -> None:
        async def fn(x: int) -> str:
            return "hello"

        ft = FunctionTool(
            fn=fn,
            name="t",
            description="d",
            parameters_schema={"type": "object", "properties": {}},
        )
        result = await ft.execute(x=1)
        assert isinstance(result, ToolResult)
        assert result.content == "hello"

    async def test_tool_result_returned_as_is(self) -> None:
        meta = {"key": "value"}

        async def fn(x: int) -> ToolResult:
            return ToolResult(content="hello", metadata=meta)

        ft = FunctionTool(
            fn=fn,
            name="t",
            description="d",
            parameters_schema={"type": "object", "properties": {}},
        )
        result = await ft.execute(x=1)
        assert result.content == "hello"
        assert result.metadata == meta


class TestFunctionToolValidation:
    def test_neither_model_nor_schema_raises(self) -> None:
        async def fn() -> str:
            return ""

        with pytest.raises(ValueError, match="Provide either"):
            FunctionTool(fn=fn, name="t", description="d")

    def test_both_model_and_schema_raises(self) -> None:
        async def fn() -> str:
            return ""

        with pytest.raises(ValueError, match="not both"):
            FunctionTool(
                fn=fn,
                name="t",
                description="d",
                parameters_model=Greeting,
                parameters_schema={"type": "object"},
            )


# --- @tool Decorator ---


class TestToolDecorator:
    def test_auto_generates_from_hints(self) -> None:
        @tool(name="add", description="Add two numbers")
        async def add(a: int, b: int) -> str:
            return str(a + b)

        assert isinstance(add, FunctionTool)
        assert isinstance(add, Tool)
        assert add.schema.name == "add"
        props = add.schema.parameters.get("properties", {})
        assert "a" in props
        assert "b" in props

    def test_optional_params(self) -> None:
        @tool(name="greet", description="Greet")
        async def greet(name: str, excited: bool = False) -> str:
            return f"Hello {name}"

        required = greet.schema.parameters.get("required", [])
        assert "name" in required
        assert "excited" not in required

    def test_with_explicit_model(self) -> None:
        @tool(name="greet", description="Greet", parameters_model=Greeting)
        async def greet(name: str, excited: bool = False) -> str:
            return f"Hello {name}"

        assert greet.parameters_model is Greeting

    async def test_execution(self) -> None:
        @tool(name="add", description="Add")
        async def add(a: int, b: int) -> str:
            return str(a + b)

        result = await add.execute(a=1, b=2)
        assert result.content == "3"


class TestModelFromFunction:
    def test_skips_self_parameter(self) -> None:
        class MyClass:
            async def action(self, x: int) -> str:
                return str(x)

        model = _model_from_function(MyClass.action, "action")
        fields = model.model_fields
        assert "self" not in fields
        assert "x" in fields


# --- ToolRegistry ---


def _make_tool(name: str = "test_tool") -> FunctionTool:
    async def fn(**kwargs: Any) -> str:
        return "ok"

    return FunctionTool(
        fn=fn,
        name=name,
        description=f"Tool {name}",
        parameters_schema={"type": "object", "properties": {}},
    )


def _make_tool_call(name: str = "test_tool", **arguments: Any) -> ToolCall:
    return ToolCall(id=str(uuid4()), name=name, arguments=arguments)


class TestToolRegistryRegistration:
    def test_register_and_lookup(self) -> None:
        registry = ToolRegistry()
        t = _make_tool()
        registry.register(t)
        assert registry.get("test_tool") is t

    def test_not_found_raises(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            registry.get("missing")

    def test_duplicate_raises(self) -> None:
        registry = ToolRegistry()
        registry.register(_make_tool())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(_make_tool())

    def test_list_schemas(self) -> None:
        registry = ToolRegistry()
        registry.register(_make_tool("a"))
        registry.register(_make_tool("b"))
        schemas = registry.list_schemas()
        names = [s.name for s in schemas]
        assert "a" in names
        assert "b" in names

    def test_has(self) -> None:
        registry = ToolRegistry()
        registry.register(_make_tool())
        assert registry.has("test_tool") is True
        assert registry.has("missing") is False

    def test_register_all(self) -> None:
        registry = ToolRegistry()
        registry.register_all([_make_tool("a"), _make_tool("b")])
        assert registry.has("a")
        assert registry.has("b")


class TestToolRegistryDispatch:
    async def test_success(self) -> None:
        async def echo(message: str) -> str:
            return f"Echo: {message}"

        ft = FunctionTool(
            fn=echo,
            name="echo",
            description="Echo",
            parameters_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        )
        registry = ToolRegistry()
        registry.register(ft)

        result = await registry.dispatch(ToolCall(id="1", name="echo", arguments={"message": "hi"}))
        assert result.content == "Echo: hi"

    async def test_not_found(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            await registry.dispatch(_make_tool_call("missing"))

    async def test_parameter_validation_error_with_model(self) -> None:
        class Params(BaseModel):
            x: int

        async def fn(x: int) -> str:
            return str(x)

        ft = FunctionTool(fn=fn, name="typed", description="d", parameters_model=Params)
        registry = ToolRegistry()
        registry.register(ft)

        with pytest.raises(ToolParameterError):
            await registry.dispatch(ToolCall(id="1", name="typed", arguments={"x": "not_an_int"}))

    async def test_missing_required_params_without_model(self) -> None:
        async def fn(query: str) -> str:
            return query

        ft = FunctionTool(
            fn=fn,
            name="search",
            description="Search",
            parameters_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        registry = ToolRegistry()
        registry.register(ft)

        with pytest.raises(ToolParameterError, match="Missing required"):
            await registry.dispatch(ToolCall(id="1", name="search", arguments={}))

    async def test_execution_error(self) -> None:
        async def failing(**kwargs: Any) -> str:
            raise RuntimeError("boom")

        ft = FunctionTool(
            fn=failing,
            name="fail",
            description="Fails",
            parameters_schema={"type": "object", "properties": {}},
        )
        registry = ToolRegistry()
        registry.register(ft)

        with pytest.raises(ToolExecutionError) as exc_info:
            await registry.dispatch(_make_tool_call("fail"))
        assert exc_info.value.__cause__ is not None
        assert "boom" in str(exc_info.value.__cause__)


class TestToolRegistryEvents:
    async def test_emits_invoke_and_result_on_success(self) -> None:
        emitter = InMemoryEmitter(trace_id="t1")

        async def fn(**kwargs: Any) -> str:
            return "ok"

        ft = FunctionTool(
            fn=fn,
            name="t",
            description="d",
            parameters_schema={"type": "object", "properties": {}},
        )
        registry = ToolRegistry(emitter=emitter)
        registry.register(ft)

        call = _make_tool_call("t")
        await registry.dispatch(call)

        invoke_events = [e for e in emitter.events if isinstance(e, ToolInvokeEvent)]
        result_events = [e for e in emitter.events if isinstance(e, ToolResultEvent)]

        assert len(invoke_events) == 1
        assert invoke_events[0].tool_name == "t"
        assert invoke_events[0].tool_call_id == call.id
        assert len(result_events) == 1
        assert result_events[0].success is True
        assert result_events[0].result == "ok"
        assert result_events[0].tool_call_id == call.id
        assert result_events[0].duration_ms >= 0

    async def test_emits_result_on_error(self) -> None:
        emitter = InMemoryEmitter(trace_id="t1")

        async def fn(**kwargs: Any) -> str:
            raise RuntimeError("boom")

        ft = FunctionTool(
            fn=fn,
            name="t",
            description="d",
            parameters_schema={"type": "object", "properties": {}},
        )
        registry = ToolRegistry(emitter=emitter)
        registry.register(ft)

        with pytest.raises(ToolExecutionError):
            await registry.dispatch(_make_tool_call("t"))

        result_events = [e for e in emitter.events if isinstance(e, ToolResultEvent)]
        assert len(result_events) == 1
        assert result_events[0].success is False
        assert result_events[0].error is not None

    async def test_no_events_without_emitter(self) -> None:
        async def fn(**kwargs: Any) -> str:
            return "ok"

        ft = FunctionTool(
            fn=fn,
            name="t",
            description="d",
            parameters_schema={"type": "object", "properties": {}},
        )
        registry = ToolRegistry()  # no emitter
        registry.register(ft)

        result = await registry.dispatch(_make_tool_call("t"))
        assert result.content == "ok"

    async def test_no_events_on_not_found(self) -> None:
        emitter = InMemoryEmitter(trace_id="t1")
        registry = ToolRegistry(emitter=emitter)

        with pytest.raises(ToolNotFoundError):
            await registry.dispatch(_make_tool_call("missing"))

        assert len(emitter.events) == 0


class TestToolRegistryErrorHandling:
    async def test_tool_error_subtypes_not_wrapped(self) -> None:
        """ToolError subtypes from execute() should propagate without wrapping."""

        async def fn(x: int) -> str:
            return str(x)

        ft = FunctionTool(fn=fn, name="typed", description="d", parameters_model=Greeting)
        registry = ToolRegistry()
        registry.register(ft)

        with pytest.raises(ToolParameterError) as exc_info:
            await registry.dispatch(ToolCall(id="1", name="typed", arguments={"name": 123}))
        assert exc_info.value.tool_name == "typed"

    async def test_tool_parameter_error_metadata(self) -> None:
        """ToolParameterError carries tool_name and reason metadata."""

        class Params(BaseModel):
            x: int

        async def fn(x: int) -> str:
            return str(x)

        ft = FunctionTool(fn=fn, name="typed", description="d", parameters_model=Params)
        registry = ToolRegistry()
        registry.register(ft)

        with pytest.raises(ToolParameterError) as exc_info:
            await registry.dispatch(ToolCall(id="1", name="typed", arguments={"x": "bad"}))
        assert exc_info.value.tool_name == "typed"
        assert exc_info.value.reason is not None
        assert exc_info.value.__cause__ is not None

    async def test_tool_error_emits_result_event(self) -> None:
        """ToolError subtypes still emit a result event before re-raising."""
        emitter = InMemoryEmitter(trace_id="t1")

        async def fn(x: int) -> str:
            return str(x)

        ft = FunctionTool(fn=fn, name="typed", description="d", parameters_model=Greeting)
        registry = ToolRegistry(emitter=emitter)
        registry.register(ft)

        with pytest.raises(ToolParameterError):
            await registry.dispatch(ToolCall(id="1", name="typed", arguments={"name": 123}))

        result_events = [e for e in emitter.events if isinstance(e, ToolResultEvent)]
        assert len(result_events) == 1
        assert result_events[0].success is False


class TestToolTimeout:
    async def test_timeout_raises_tool_timeout_error(self) -> None:
        async def slow(**kwargs: Any) -> str:
            await asyncio.sleep(10)
            return "done"

        ft = FunctionTool(
            fn=slow,
            name="slow",
            description="A slow tool",
            parameters_schema={
                "type": "object",
                "properties": {},
            },
        )
        # Override schema with timeout
        ft._schema = ToolSchema(
            name="slow",
            description="A slow tool",
            parameters={"type": "object", "properties": {}},
            timeout_seconds=0.05,
        )
        registry = ToolRegistry()
        registry.register(ft)

        with pytest.raises(ToolTimeoutError) as exc_info:
            await registry.dispatch(_make_tool_call("slow"))
        assert exc_info.value.tool_name == "slow"
        assert exc_info.value.timeout_seconds == 0.05

    async def test_timeout_emits_failure_event(self) -> None:
        emitter = InMemoryEmitter(trace_id="t1")

        async def slow(**kwargs: Any) -> str:
            await asyncio.sleep(10)
            return "done"

        ft = FunctionTool(
            fn=slow,
            name="slow",
            description="A slow tool",
            parameters_schema={
                "type": "object",
                "properties": {},
            },
        )
        ft._schema = ToolSchema(
            name="slow",
            description="A slow tool",
            parameters={"type": "object", "properties": {}},
            timeout_seconds=0.05,
        )
        registry = ToolRegistry(emitter=emitter)
        registry.register(ft)

        with pytest.raises(ToolTimeoutError):
            await registry.dispatch(_make_tool_call("slow"))

        result_events = [e for e in emitter.events if isinstance(e, ToolResultEvent)]
        assert len(result_events) == 1
        assert result_events[0].success is False
        assert result_events[0].error is not None
        assert "timed out" in result_events[0].error

    async def test_no_timeout_when_not_configured(self) -> None:
        async def fast(**kwargs: Any) -> str:
            return "fast"

        ft = FunctionTool(
            fn=fast,
            name="fast",
            description="A fast tool",
            parameters_schema={
                "type": "object",
                "properties": {},
            },
        )
        assert ft.schema.timeout_seconds is None

        registry = ToolRegistry()
        registry.register(ft)
        result = await registry.dispatch(_make_tool_call("fast"))
        assert result.content == "fast"


class TestDispatchEmissionGating:
    """Registry emission is gated on whether the tool actually executed.

    ``ToolResult.executed`` (defaults to True) is the wrapper-signalling
    contract: when a transparent wrapper short-circuits by returning
    ``executed=False``, the registry suppresses both ``ToolInvokeEvent``
    and ``ToolResultEvent``. Every emitting branch must emit invoke
    strictly before result.
    """

    async def test_executed_true_emits_invoke_then_result(self) -> None:
        emitter = InMemoryEmitter(trace_id="t1")

        async def ok(**kwargs: Any) -> str:
            return "ok"

        ft = FunctionTool(
            fn=ok,
            name="ok",
            description="d",
            parameters_schema={"type": "object", "properties": {}},
        )
        registry = ToolRegistry(emitter=emitter)
        registry.register(ft)

        result = await registry.dispatch(_make_tool_call("ok"))
        assert result.executed is True

        invoke_events = [e for e in emitter.events if isinstance(e, ToolInvokeEvent)]
        result_events = [e for e in emitter.events if isinstance(e, ToolResultEvent)]
        assert len(invoke_events) == 1
        assert len(result_events) == 1
        assert result_events[0].success is True
        assert result_events[0].result == "ok"

        invoke_index = emitter.events.index(invoke_events[0])
        result_index = emitter.events.index(result_events[0])
        assert invoke_index < result_index

    async def test_executed_false_suppresses_both_events(self) -> None:
        emitter = InMemoryEmitter(trace_id="t1")

        class ShortCircuitTool:
            @property
            def schema(self) -> ToolSchema:
                return ToolSchema(
                    name="gated",
                    description="A gating wrapper",
                    parameters={"type": "object", "properties": {}},
                )

            async def execute(self, **params: Any) -> ToolResult:
                return ToolResult(content="short-circuited", executed=False)

        registry = ToolRegistry(emitter=emitter)
        registry.register(ShortCircuitTool())

        result = await registry.dispatch(_make_tool_call("gated"))
        assert result.content == "short-circuited"
        assert result.executed is False

        invoke_events = [e for e in emitter.events if isinstance(e, ToolInvokeEvent)]
        result_events = [e for e in emitter.events if isinstance(e, ToolResultEvent)]
        assert len(invoke_events) == 0
        assert len(result_events) == 0

    async def test_tool_error_still_emits_invoke_then_error_result(self) -> None:
        emitter = InMemoryEmitter(trace_id="t1")

        async def raising(**kwargs: Any) -> str:
            raise ToolExecutionError("boom", tool_name="raising")

        ft = FunctionTool(
            fn=raising,
            name="raising",
            description="d",
            parameters_schema={"type": "object", "properties": {}},
        )
        registry = ToolRegistry(emitter=emitter)
        registry.register(ft)

        with pytest.raises(ToolExecutionError):
            await registry.dispatch(_make_tool_call("raising"))

        invoke_events = [e for e in emitter.events if isinstance(e, ToolInvokeEvent)]
        result_events = [e for e in emitter.events if isinstance(e, ToolResultEvent)]
        assert len(invoke_events) == 1
        assert len(result_events) == 1
        assert result_events[0].success is False
        assert result_events[0].error == "boom"

        invoke_index = emitter.events.index(invoke_events[0])
        result_index = emitter.events.index(result_events[0])
        assert invoke_index < result_index

    async def test_parameter_validation_failure_emits_invoke_then_error_result(
        self,
    ) -> None:
        emitter = InMemoryEmitter(trace_id="t1")

        async def needs(query: str) -> str:
            return query

        ft = FunctionTool(
            fn=needs,
            name="needs",
            description="Needs query",
            parameters_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        registry = ToolRegistry(emitter=emitter)
        registry.register(ft)

        with pytest.raises(ToolParameterError):
            await registry.dispatch(ToolCall(id="1", name="needs", arguments={}))

        invoke_events = [e for e in emitter.events if isinstance(e, ToolInvokeEvent)]
        result_events = [e for e in emitter.events if isinstance(e, ToolResultEvent)]
        assert len(invoke_events) == 1
        assert len(result_events) == 1
        assert result_events[0].success is False
        assert result_events[0].error is not None
        assert "Missing required parameters" in result_events[0].error

        invoke_index = emitter.events.index(invoke_events[0])
        result_index = emitter.events.index(result_events[0])
        assert invoke_index < result_index

    async def test_timeout_emits_invoke_then_error_result(self) -> None:
        emitter = InMemoryEmitter(trace_id="t1")

        async def slow(**kwargs: Any) -> str:
            await asyncio.sleep(10)
            return "done"

        ft = FunctionTool(
            fn=slow,
            name="slow",
            description="Slow",
            parameters_schema={"type": "object", "properties": {}},
        )
        ft._schema = ToolSchema(
            name="slow",
            description="Slow",
            parameters={"type": "object", "properties": {}},
            timeout_seconds=0.05,
        )
        registry = ToolRegistry(emitter=emitter)
        registry.register(ft)

        with pytest.raises(ToolTimeoutError):
            await registry.dispatch(_make_tool_call("slow"))

        invoke_events = [e for e in emitter.events if isinstance(e, ToolInvokeEvent)]
        result_events = [e for e in emitter.events if isinstance(e, ToolResultEvent)]
        assert len(invoke_events) == 1
        assert len(result_events) == 1

        invoke_index = emitter.events.index(invoke_events[0])
        result_index = emitter.events.index(result_events[0])
        assert invoke_index < result_index

    async def test_generic_exception_emits_invoke_then_error_result(self) -> None:
        emitter = InMemoryEmitter(trace_id="t1")

        async def bad(**kwargs: Any) -> str:
            raise RuntimeError("generic")

        ft = FunctionTool(
            fn=bad,
            name="bad",
            description="d",
            parameters_schema={"type": "object", "properties": {}},
        )
        registry = ToolRegistry(emitter=emitter)
        registry.register(ft)

        with pytest.raises(ToolExecutionError):
            await registry.dispatch(_make_tool_call("bad"))

        invoke_events = [e for e in emitter.events if isinstance(e, ToolInvokeEvent)]
        result_events = [e for e in emitter.events if isinstance(e, ToolResultEvent)]
        assert len(invoke_events) == 1
        assert len(result_events) == 1
        assert result_events[0].success is False
        assert result_events[0].error == "generic"

        invoke_index = emitter.events.index(invoke_events[0])
        result_index = emitter.events.index(result_events[0])
        assert invoke_index < result_index
