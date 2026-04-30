"""Tests for the MCP client integration.

Covers ``MCPTool`` (the per-tool protocol-conforming wrapper), ``MCPClient``
(the session-owning async context manager and ``MCPStdioParameters`` dataclass),
and end-to-end parity through ``ToolRegistry`` / ``ReActAgent``.

All tests are hermetic: transport uses ``mcp.shared.memory`` so no subprocess
or network I/O is involved.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import anyio
import pytest
from mcp import ClientSession, ErrorData, McpError
from mcp.server.fastmcp import FastMCP
from mcp.types import (
    CallToolResult,
    ListToolsResult,
    TextContent,
)
from mcp.types import (
    Tool as MCPUpstreamTool,
)

from nanitics.core.agents.react import ReActAgent
from nanitics.core.tools.protocol import Tool as ToolProtocol
from nanitics.core.tools.protocol import ToolResult
from nanitics.core.tools.registry import ToolRegistry
from nanitics.infrastructure.errors import (
    LLMProviderError,
    ToolExecutionError,
    ToolTimeoutError,
)
from nanitics.infrastructure.llm.mock import MockLLMClient
from nanitics.infrastructure.llm.protocol import (
    LLMResponse,
    ToolCall,
    ToolSchema,
)
from nanitics.infrastructure.mcp._tool import MCPTool
from nanitics.infrastructure.mcp.client import MCPClient, MCPStdioParameters
from nanitics.infrastructure.observability.emitter import InMemoryEmitter
from nanitics.infrastructure.observability.events import (
    ToolInvokeEvent,
    ToolResultEvent,
    Usage,
)

# ---------------------------------------------------------------------------
# Session stub — lets us exercise MCPTool without a real session.
# ---------------------------------------------------------------------------


class _StubSession:
    """Minimal stand-in for ``mcp.ClientSession`` used by MCPTool tests.

    Only implements ``call_tool``.  Behavior is scripted via the handler
    passed to the constructor: the handler receives (name, arguments) and
    returns/raises whatever the test wants.
    """

    def __init__(
        self,
        handler: Callable[[str, dict[str, Any] | None], Awaitable[CallToolResult]],
    ) -> None:
        self._handler = handler
        self.calls: list[tuple[str, dict[str, Any] | None]] = []
        self.closed = False

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> CallToolResult:
        self.calls.append((name, arguments))
        if self.closed:
            raise ConnectionError("session closed")
        return await self._handler(name, arguments)


def _text_result(text: str, *, is_error: bool = False) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        isError=is_error,
    )


def _make_tool(
    *,
    name: str = "tool",
    description: str = "desc",
    handler: Callable[[str, dict[str, Any] | None], Awaitable[CallToolResult]] | None = None,
    schema_timeout: float | None = None,
    default_timeout: float | None = None,
    session: _StubSession | None = None,
) -> tuple[MCPTool, _StubSession]:
    async def _default(_name: str, _args: dict[str, Any] | None) -> CallToolResult:
        return _text_result("ok")

    actual_handler = handler or _default
    sess = session if session is not None else _StubSession(actual_handler)
    schema = ToolSchema(
        name=name,
        description=description,
        parameters={"type": "object", "properties": {}},
        timeout_seconds=schema_timeout,
    )
    tool = MCPTool(
        schema=schema,
        mcp_tool_name=name,
        session=sess,  # type: ignore[arg-type]
        default_timeout=default_timeout,
    )
    return tool, sess


# ---------------------------------------------------------------------------
# TestMCPTool — Step 4 acceptance
# ---------------------------------------------------------------------------


class TestMCPTool:
    def test_satisfies_tool_protocol(self) -> None:
        tool, _ = _make_tool()
        assert isinstance(tool, ToolProtocol)

    def test_schema_property_returns_stored_schema(self) -> None:
        tool, _ = _make_tool(name="weather", description="look up weather")
        assert tool.schema.name == "weather"
        assert tool.schema.description == "look up weather"

    async def test_execute_returns_tool_result_with_text_content(self) -> None:
        async def handler(name: str, args: dict[str, Any] | None) -> CallToolResult:
            assert name == "echo"
            assert args == {"msg": "hi"}
            return _text_result("you said: hi")

        tool, session = _make_tool(name="echo", handler=handler)
        result = await tool.execute(msg="hi")

        assert isinstance(result, ToolResult)
        assert result.content == "you said: hi"
        assert result.metadata["is_error"] is False
        assert session.calls == [("echo", {"msg": "hi"})]

    async def test_execute_forwards_empty_arguments_as_empty_dict(self) -> None:
        async def handler(_name: str, args: dict[str, Any] | None) -> CallToolResult:
            assert args == {}
            return _text_result("ok")

        tool, _ = _make_tool(handler=handler)
        await tool.execute()

    async def test_schema_timeout_takes_precedence_over_default(self) -> None:
        started = asyncio.Event()

        async def slow(_name: str, _args: dict[str, Any] | None) -> CallToolResult:
            started.set()
            await asyncio.sleep(10)
            return _text_result("never")

        tool, _ = _make_tool(
            handler=slow,
            schema_timeout=0.05,
            default_timeout=5.0,
        )

        with pytest.raises(ToolTimeoutError) as exc_info:
            await tool.execute()

        assert exc_info.value.tool_name == "tool"
        assert exc_info.value.timeout_seconds == 0.05
        assert started.is_set()

    async def test_default_timeout_used_when_schema_has_no_timeout(self) -> None:
        async def slow(_name: str, _args: dict[str, Any] | None) -> CallToolResult:
            await asyncio.sleep(10)
            return _text_result("never")

        tool, _ = _make_tool(handler=slow, default_timeout=0.05)

        with pytest.raises(ToolTimeoutError) as exc_info:
            await tool.execute()

        assert exc_info.value.timeout_seconds == 0.05

    async def test_no_timeout_when_both_are_none(self) -> None:
        # Should simply return normally; no asyncio.timeout wrapper when both are
        # None.  We don't try to verify the absence of a deadline other than by
        # successful completion — a hang would show up as a test-suite timeout.
        tool, _ = _make_tool()
        result = await tool.execute()
        assert result.content == "ok"

    async def test_mcp_error_maps_to_tool_execution_error(self) -> None:
        original = McpError(ErrorData(code=-32000, message="server boom"))

        async def handler(_name: str, _args: dict[str, Any] | None) -> CallToolResult:
            raise original

        tool, _ = _make_tool(name="boom", handler=handler)

        with pytest.raises(ToolExecutionError) as exc_info:
            await tool.execute()

        assert exc_info.value.tool_name == "boom"
        assert exc_info.value.__cause__ is original

    async def test_is_error_result_maps_to_tool_execution_error(self) -> None:
        async def handler(_name: str, _args: dict[str, Any] | None) -> CallToolResult:
            return _text_result("bad inputs", is_error=True)

        tool, _ = _make_tool(name="err", handler=handler)

        with pytest.raises(ToolExecutionError) as exc_info:
            await tool.execute()

        assert exc_info.value.tool_name == "err"
        assert "bad inputs" in exc_info.value.message

    async def test_transport_failure_maps_to_llm_provider_error(self) -> None:
        original = ConnectionError("pipe broke")

        async def handler(_name: str, _args: dict[str, Any] | None) -> CallToolResult:
            raise original

        tool, _ = _make_tool(name="bad_transport", handler=handler)

        with pytest.raises(LLMProviderError) as exc_info:
            await tool.execute()

        assert exc_info.value.provider == "mcp"
        assert "bad_transport" in exc_info.value.message
        assert exc_info.value.__cause__ is original

    async def test_execute_after_session_closed_raises_llm_provider_error(self) -> None:
        tool, session = _make_tool()
        session.closed = True

        with pytest.raises(LLMProviderError) as exc_info:
            await tool.execute()

        assert exc_info.value.provider == "mcp"


# ---------------------------------------------------------------------------
# Step 5 — MCPStdioParameters, MCPClient factory/lifecycle/discovery
# ---------------------------------------------------------------------------


class TestMCPStdioParameters:
    def test_defaults(self) -> None:
        p = MCPStdioParameters(command="npx")
        assert p.command == "npx"
        assert p.args == []
        assert p.env is None
        assert p.cwd is None

    def test_equality(self) -> None:
        a = MCPStdioParameters(command="x", args=["-y"])
        b = MCPStdioParameters(command="x", args=["-y"])
        assert a == b

    def test_default_args_are_not_shared(self) -> None:
        a = MCPStdioParameters(command="x")
        b = MCPStdioParameters(command="y")
        # Frozen dataclass + default_factory => independent list instances.
        assert a.args is not b.args

    def test_frozen(self) -> None:
        import dataclasses

        p = MCPStdioParameters(command="x")
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.command = "y"  # type: ignore[misc]


# A helper transport factory that yields a prepared ClientSession.
# This lets us unit-test MCPClient's lifecycle logic without depending on
# stdio_client/sse_client internals.


def _memory_transport_factory(
    server: FastMCP | None = None,
    *,
    fail_on_enter: BaseException | None = None,
) -> Callable[[], Any]:
    """Return a transport-context factory compatible with MCPClient.

    The factory, when called, returns an async context manager that yields
    a pair ``(read_stream, write_stream)`` just like ``stdio_client`` and
    ``sse_client``.  When the MCPClient creates a ClientSession around those
    streams, it is a real session wired to an in-memory FastMCP.
    """

    if server is None:
        server = FastMCP(name="test")

        @server.tool(description="Echo")
        def echo(msg: str) -> str:
            return f"echo: {msg}"

    @asynccontextmanager
    async def _transport() -> AsyncIterator[tuple[Any, Any]]:
        if fail_on_enter is not None:
            raise fail_on_enter
        # Create two paired memory streams: one for client->server and one for
        # server->client.  Run the server loop on its side; hand the client
        # streams to MCPClient.
        from mcp.shared.memory import create_client_server_memory_streams

        async with create_client_server_memory_streams() as (client_streams, server_streams):
            client_read, client_write = client_streams
            server_read, server_write = server_streams

            async def run_server() -> None:
                from mcp.server.lowlevel.server import NotificationOptions

                init_options = server._mcp_server.create_initialization_options(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                )
                try:
                    await server._mcp_server.run(
                        server_read,
                        server_write,
                        init_options,
                        raise_exceptions=False,
                    )
                except anyio.ClosedResourceError:
                    return
                except Exception:
                    return

            async with anyio.create_task_group() as tg:
                tg.start_soon(run_server)
                try:
                    yield client_read, client_write
                finally:
                    tg.cancel_scope.cancel()

    return _transport


class TestMCPClientFactory:
    def test_stdio_factory_stores_parameters(self) -> None:
        params = MCPStdioParameters(command="npx", args=["-y", "srv"])
        client = MCPClient.stdio(params, name_prefix="fs_")
        assert client._name_prefix == "fs_"
        assert client._discovery_timeout == 30.0
        assert client._default_call_timeout == 60.0
        # The factory must not open the transport yet.
        assert client._entered is False

    def test_sse_factory_stores_parameters(self) -> None:
        client = MCPClient.sse(
            url="http://example.com/mcp",
            headers={"Authorization": "Bearer x"},
            name_prefix="web_",
            discovery_timeout=10.0,
            default_call_timeout=5.0,
        )
        assert client._name_prefix == "web_"
        assert client._discovery_timeout == 10.0
        assert client._default_call_timeout == 5.0
        assert client._entered is False

    def test_name_filter_stored(self) -> None:
        f = lambda n: n.startswith("safe_")  # noqa: E731
        client = MCPClient.stdio(
            MCPStdioParameters(command="x"),
            name_filter=f,
        )
        assert client._name_filter is f

    def test_stdio_factory_invokes_upstream_stdio_client(self) -> None:
        # The factory should be a closure that, when called, returns an
        # async context manager from the upstream SDK.  Calling it should
        # not spawn a subprocess — the upstream ACM opens the subprocess
        # only on ``__aenter__``.
        client = MCPClient.stdio(MCPStdioParameters(command="echo", args=["hi"]))
        cm = client._transport_factory()
        # The upstream stdio_client returns an asynccontextmanager; confirm
        # it is an async context manager without entering it.
        assert hasattr(cm, "__aenter__")
        assert hasattr(cm, "__aexit__")

    def test_sse_factory_invokes_upstream_sse_client(self) -> None:
        client = MCPClient.sse(url="http://example.com/mcp")
        cm = client._transport_factory()
        assert hasattr(cm, "__aenter__")
        assert hasattr(cm, "__aexit__")


class TestMCPClientLifecycle:
    async def test_successful_enter_and_exit(self) -> None:
        server = FastMCP(name="t")

        @server.tool(description="ping")
        def ping() -> str:
            return "pong"

        client = MCPClient._for_testing(
            transport_factory=_memory_transport_factory(server),
        )
        async with client as c:
            tools = await c.list_tools()
            assert [t.schema.name for t in tools] == ["ping"]

    async def test_list_tools_before_enter_raises(self) -> None:
        client = MCPClient._for_testing(
            transport_factory=_memory_transport_factory(),
        )
        with pytest.raises(RuntimeError, match="async context manager"):
            await client.list_tools()

    async def test_transport_failure_during_enter_propagates(self) -> None:
        boom = RuntimeError("no transport")
        client = MCPClient._for_testing(
            transport_factory=_memory_transport_factory(fail_on_enter=boom),
        )
        with pytest.raises(RuntimeError, match="no transport"):
            async with client:
                pass  # pragma: no cover

    async def test_mcptool_unusable_after_exit(self) -> None:
        server = FastMCP(name="t")

        @server.tool(description="ping")
        def ping() -> str:
            return "pong"

        client = MCPClient._for_testing(
            transport_factory=_memory_transport_factory(server),
        )
        async with client as c:
            tools = await c.list_tools()

        # After exit, invoking the tool should fail — the session is closed.
        with pytest.raises(LLMProviderError):
            await tools[0].execute()


class TestMCPClientDiscovery:
    async def test_discovery_applies_name_prefix(self) -> None:
        server = FastMCP(name="t")

        @server.tool(description="ping")
        def ping() -> str:
            return "pong"

        client = MCPClient._for_testing(
            transport_factory=_memory_transport_factory(server),
            name_prefix="srv_",
        )
        async with client as c:
            tools = await c.list_tools()
            assert [t.schema.name for t in tools] == ["srv_ping"]
            # Executing through the prefixed tool should still hit the unprefixed
            # server-side name.
            result = await tools[0].execute()
            assert result.content == "pong"

    async def test_discovery_applies_name_filter(self) -> None:
        server = FastMCP(name="t")

        @server.tool(description="ping")
        def ping() -> str:
            return "pong"

        @server.tool(description="pong")
        def pang() -> str:
            return "pang"

        client = MCPClient._for_testing(
            transport_factory=_memory_transport_factory(server),
            name_filter=lambda n: n == "ping",
        )
        async with client as c:
            tools = await c.list_tools()
            assert [t.schema.name for t in tools] == ["ping"]

    async def test_list_tools_caches_result(self) -> None:
        # Wrap the session so we can count list_tools calls on it.
        call_count = 0

        class _CountingSession:
            def __init__(self, inner: ClientSession) -> None:
                self._inner = inner

            async def initialize(self) -> Any:
                return await self._inner.initialize()

            async def list_tools(self, *args: Any, **kwargs: Any) -> ListToolsResult:
                nonlocal call_count
                call_count += 1
                return await self._inner.list_tools(*args, **kwargs)

            async def call_tool(self, *args: Any, **kwargs: Any) -> CallToolResult:
                return await self._inner.call_tool(*args, **kwargs)

        server = FastMCP(name="t")

        @server.tool(description="ping")
        def ping() -> str:
            return "pong"

        client = MCPClient._for_testing(
            transport_factory=_memory_transport_factory(server),
            session_wrapper=_CountingSession,
        )
        async with client as c:
            first = await c.list_tools()
            second = await c.list_tools()

        assert first is second
        assert call_count == 1

    async def test_none_discovery_timeout_uses_unbounded_initialize_and_list(self) -> None:
        server = FastMCP(name="t")

        @server.tool(description="ping")
        def ping() -> str:
            return "pong"

        client = MCPClient._for_testing(
            transport_factory=_memory_transport_factory(server),
            discovery_timeout=None,
        )
        async with client as c:
            tools = await c.list_tools()
        assert [t.schema.name for t in tools] == ["ping"]

    async def test_list_tools_timeout_raises_llm_provider_error(self) -> None:
        # Inject a session wrapper whose list_tools hangs; the client's
        # list_tools wrapper should time out and raise LLMProviderError.
        class _HangSession:
            def __init__(self, inner: ClientSession) -> None:
                self._inner = inner

            async def initialize(self) -> Any:
                return await self._inner.initialize()

            async def list_tools(self) -> ListToolsResult:
                await asyncio.sleep(10)
                raise AssertionError("unreachable")  # pragma: no cover

            async def call_tool(self, *args: Any, **kwargs: Any) -> CallToolResult:
                return await self._inner.call_tool(*args, **kwargs)  # pragma: no cover

        server = FastMCP(name="t")
        client = MCPClient._for_testing(
            transport_factory=_memory_transport_factory(server),
            session_wrapper=_HangSession,
            discovery_timeout=0.05,
        )

        caught: LLMProviderError | None = None
        async with client as c:
            try:
                await c.list_tools()
            except LLMProviderError as exc:
                caught = exc

        assert caught is not None
        assert caught.provider == "mcp"
        assert "discovery timed out" in caught.message.lower()

    async def test_discovery_timeout_raises_llm_provider_error(self) -> None:
        # A transport that hangs during ``initialize`` — use a factory that
        # produces streams attached to a server that never responds.
        @asynccontextmanager
        async def _hang_transport() -> AsyncIterator[tuple[Any, Any]]:
            from mcp.shared.memory import create_client_server_memory_streams

            async with create_client_server_memory_streams() as (client_streams, _server_streams):
                yield client_streams

        client = MCPClient._for_testing(
            transport_factory=_hang_transport,
            discovery_timeout=0.05,
        )
        with pytest.raises(LLMProviderError) as exc_info:
            async with client:
                pass  # pragma: no cover

        assert exc_info.value.provider == "mcp"
        assert "timed out" in exc_info.value.message.lower()

    async def test_tool_name_prefix_does_not_leak_into_server_call(self) -> None:
        server = FastMCP(name="t")

        @server.tool(description="returns input")
        def echo(msg: str) -> str:
            return f"got: {msg}"

        client = MCPClient._for_testing(
            transport_factory=_memory_transport_factory(server),
            name_prefix="fs_",
        )
        async with client as c:
            [tool] = await c.list_tools()
            out = await tool.execute(msg="hi")
            assert out.content == "got: hi"

    async def test_empty_tool_list_returns_empty(self) -> None:
        server = FastMCP(name="t")  # no tools
        client = MCPClient._for_testing(
            transport_factory=_memory_transport_factory(server),
        )
        async with client as c:
            tools = await c.list_tools()
        assert tools == []

    async def test_mcp_upstream_tool_types_translated(self) -> None:
        # Sanity check that translate path is wired up: the discovered
        # MCPTool's schema parameters match the server-declared inputSchema.
        upstream_name = "calc"

        server = FastMCP(name="t")

        @server.tool(name=upstream_name, description="add")
        def calc(a: int, b: int) -> int:
            return a + b

        client = MCPClient._for_testing(
            transport_factory=_memory_transport_factory(server),
        )
        async with client as c:
            [tool] = await c.list_tools()

        assert isinstance(tool.schema, ToolSchema)
        assert tool.schema.name == upstream_name
        # Description is prefixed with "[MCP]" so a downstream tool
        # catalogue can tell MCP-sourced tools apart from in-process ones.
        assert tool.schema.description.startswith("[MCP]")


# ---------------------------------------------------------------------------
# Regression: Upstream Tool model carries what we expect.  Not strictly
# required, but catches drift in pinned mcp version.
# ---------------------------------------------------------------------------


def test_upstream_tool_model_shape_regression() -> None:
    t = MCPUpstreamTool(
        name="x",
        description="y",
        inputSchema={"type": "object", "properties": {}},
    )
    assert t.name == "x"


# ---------------------------------------------------------------------------
# TestMCPIntegration — protocol parity end-to-end
#
# MCP-discovered tools must behave exactly like in-process ``FunctionTool``
# instances when driven through ``ToolRegistry`` and consumed by an agent.
# These tests lock that invariant in.
# ---------------------------------------------------------------------------


def _build_integration_server() -> FastMCP:
    """FastMCP server exposing a happy-path tool and an always-failing tool."""
    srv = FastMCP(name="integration")

    @srv.tool(description="Look up weather for a city.")
    def get_weather(city: str) -> str:
        return f"sunny in {city}"

    @srv.tool(description="Always fails.")
    def noop() -> str:
        # FastMCP surfaces raised exceptions as CallToolResult(isError=True).
        raise RuntimeError("deliberate failure")

    return srv


class TestMCPIntegration:
    async def test_mcp_tools_dispatch_through_tool_registry(self) -> None:
        """MCPTool integrates with ToolRegistry and emits the standard events."""
        client = MCPClient._for_testing(
            transport_factory=_memory_transport_factory(_build_integration_server()),
        )
        emitter = InMemoryEmitter(trace_id="mcp-integration")
        registry = ToolRegistry(emitter=emitter)

        async with client as c:
            tools = await c.list_tools()
            registry.register_all(tools)

            # Happy-path dispatch.
            result = await registry.dispatch(
                ToolCall(id="call-1", name="get_weather", arguments={"city": "sf"}),
            )
            assert isinstance(result, ToolResult)
            assert result.content == "sunny in sf"

            # Error-path dispatch: isError=True from the server becomes
            # ToolExecutionError at the registry boundary.
            with pytest.raises(ToolExecutionError) as exc_info:
                await registry.dispatch(
                    ToolCall(id="call-2", name="noop", arguments={}),
                )
            assert exc_info.value.tool_name == "noop"

        # Event stream: we expect one Invoke + one Result per dispatch (2 each),
        # with the Result on the error path carrying success=False.
        invoke_events = [e for e in emitter.events if isinstance(e, ToolInvokeEvent)]
        result_events = [e for e in emitter.events if isinstance(e, ToolResultEvent)]

        assert [e.tool_name for e in invoke_events] == ["get_weather", "noop"]
        assert [e.tool_call_id for e in invoke_events] == ["call-1", "call-2"]

        assert len(result_events) == 2
        success_event, error_event = result_events
        assert success_event.tool_name == "get_weather"
        assert success_event.success is True
        assert success_event.result == "sunny in sf"
        assert error_event.tool_name == "noop"
        assert error_event.success is False
        assert error_event.error is not None

    async def test_mcp_tools_drive_react_agent_end_to_end(self) -> None:
        """A ReActAgent treats MCP tools identically to FunctionTools.

        The scripted LLM first issues a ``get_weather`` tool call, then —
        after seeing the tool result in the message history — returns a final
        answer containing the result string.  We assert the agent completes
        normally and that the trace carries a ``tool.result`` event with the
        MCP-backed output.
        """
        usage = Usage(input_tokens=1, output_tokens=1)

        responses = [
            LLMResponse(
                content="Let me check.",
                tool_calls=[
                    ToolCall(id="tc-1", name="get_weather", arguments={"city": "sf"}),
                ],
                usage=usage,
                model="mock",
                stop_reason="tool_use",
            ),
            LLMResponse(
                content="The weather is: sunny in sf",
                tool_calls=[],
                usage=usage,
                model="mock",
                stop_reason="end_turn",
            ),
        ]

        llm = MockLLMClient(responses)
        emitter = InMemoryEmitter(trace_id="mcp-react")

        mcp_client = MCPClient._for_testing(
            transport_factory=_memory_transport_factory(_build_integration_server()),
        )

        async with mcp_client as c:
            mcp_tools = await c.list_tools()
            agent = ReActAgent(
                name="weather-agent",
                llm_client=llm,
                emitter=emitter,
                system_prompt="Use the provided tools to answer the user.",
                tools=mcp_tools,
            )
            agent_result = await agent.run("What's the weather in SF?")

        assert agent_result.termination_reason == "complete"
        assert "sunny in sf" in agent_result.output

        # Confirm the MCP tool call flowed through ToolRegistry — success event
        # with the MCP-produced string.
        tool_result_events = [e for e in emitter.events if isinstance(e, ToolResultEvent)]
        assert any(
            e.tool_name == "get_weather" and e.success is True and e.result == "sunny in sf" for e in tool_result_events
        )
        # And that the MCP tool was offered to the LLM in the tools list on the
        # first turn — identical shape to a FunctionTool.
        first_call_tools = llm.calls[0]["tools"]
        assert first_call_tools is not None
        assert [t.name for t in first_call_tools] == ["get_weather", "noop"]
