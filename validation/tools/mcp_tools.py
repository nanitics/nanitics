"""MCP tool dispatch under a real LLM.

Validates the end-to-end MCP path: in-process ``FastMCP`` server →
:class:`MCPClient` discovery → :class:`ReActAgent` with real LLM → tool
invocation → result. The MCP server runs in-process over an in-memory
transport (matching the hermetic pattern in ``examples/tools/mcp_tools.py``
Section 1) so the script does not require a subprocess or network.

Acceptance criteria:
  - Discovery returns both server-exposed tools (``get_weather`` and
    ``always_fail``).
  - The ``get_weather`` tool's ``schema.description`` carries the ``[MCP]``
    prefix.
  - The forwarded ``inputSchema`` includes both ``city`` and ``units``
    properties (proves non-trivial schema forwarding, not just a single
    string field).
  - Agent invokes ``get_weather`` and a ``ToolInvokeEvent`` is emitted with
    ``tool_name == "get_weather"``.
  - ``ToolResultEvent.success is True`` for the successful invocation.
  - ``ToolInvokeEvent.parameters["city"]`` matches the ``city`` the server
    observed (round-trip fidelity via mutated server-side state).
  - Final answer reports that Amsterdam is sunny and warm.
  - An ``isError=True`` MCP response surfaces as ``ToolExecutionError``
    (error-path contract).
  - After the ``async with`` block exits, ``MCPTool.execute(...)`` raises
    ``LLMProviderError`` because the session streams are closed (post-exit
    lifecycle contract).
"""

from __future__ import annotations

import importlib.util
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

import pytest

from validation.helpers import (
    assert_result_satisfies,
    assert_trace_contains,
    make_llm_client,
    requires_mcp,
    run_with_retry,
)

# Module-level guard — ``requires_mcp`` handles the per-test skip message, but
# the imports below fail at collection time if the ``mcp`` extra is missing.
_has_mcp = importlib.util.find_spec("mcp") is not None

if _has_mcp:
    import anyio
    from mcp.server.fastmcp import FastMCP

    from nanitics.infrastructure import MCPClient, ToolInvokeEvent, ToolResultEvent
    from nanitics.infrastructure.errors import (
        LLMProviderError,
        ToolExecutionError,
    )
    from nanitics.strategies import ReActAgent
    from nanitics.tracing import InMemoryEmitter


class _ServerState:
    """Captures server-side state for round-trip fidelity assertions."""

    def __init__(self) -> None:
        self.last_city: str | None = None
        self.last_units: str | None = None
        self.call_count: int = 0


def _build_weather_server(state: _ServerState) -> FastMCP:
    server = FastMCP(name="weather-demo")

    # Non-trivial schema: ``city`` is a string, ``units`` is a Literal enum so
    # the forwarded ``inputSchema`` exercises more than a single string field.
    @server.tool(description="Look up the current weather for a city.")
    def get_weather(city: str, units: Literal["celsius", "fahrenheit"] = "celsius") -> str:
        state.last_city = city
        state.last_units = units
        state.call_count += 1
        temp = "22C" if units == "celsius" else "72F"
        return f"sunny and {temp} in {city}"

    @server.tool(description="Always fails — exercises the isError path.")
    def always_fail() -> str:
        raise RuntimeError("deliberate failure for validation")

    return server


def _in_memory_transport_factory(server: FastMCP) -> Any:
    @asynccontextmanager
    async def _transport() -> AsyncIterator[tuple[Any, Any]]:
        from mcp.server.lowlevel.server import NotificationOptions
        from mcp.shared.memory import create_client_server_memory_streams

        async with create_client_server_memory_streams() as (client_streams, server_streams):
            client_read, client_write = client_streams
            server_read, server_write = server_streams

            init_options = server._mcp_server.create_initialization_options(
                notification_options=NotificationOptions(),
                experimental_capabilities={},
            )

            async def run_server() -> None:
                try:
                    await server._mcp_server.run(server_read, server_write, init_options, raise_exceptions=False)
                except Exception:
                    return

            async with anyio.create_task_group() as tg:
                tg.start_soon(run_server)
                try:
                    yield client_read, client_write
                finally:
                    tg.cancel_scope.cancel()

    return _transport


@pytest.mark.quick
@requires_mcp
async def test_mcp_tool_dispatch(traced_emitter: InMemoryEmitter) -> None:
    state = _ServerState()
    server = _build_weather_server(state)
    mcp_client = MCPClient._for_testing(transport_factory=_in_memory_transport_factory(server))

    leaked_tool = None
    async with mcp_client as client:
        mcp_tools = await client.list_tools()
        tool_names = sorted(t.schema.name for t in mcp_tools)
        assert tool_names == ["always_fail", "get_weather"], f"Expected two tools, got: {tool_names}"

        get_weather = next(t for t in mcp_tools if t.schema.name == "get_weather")
        assert get_weather.schema.description.startswith("[MCP]"), (
            f"Expected [MCP] prefix, got: {get_weather.schema.description!r}"
        )
        # Non-trivial schema: both parameters should be forwarded.
        properties = get_weather.schema.parameters.get("properties", {})
        assert "city" in properties, f"Expected 'city' in forwarded schema, got: {properties}"
        assert "units" in properties, f"Expected 'units' in forwarded schema, got: {properties}"

        agent = ReActAgent(
            name="mcp-weather-agent",
            llm_client=make_llm_client("anthropic"),
            emitter=traced_emitter,
            system_prompt="Use the provided tools to answer questions about the weather.",
            tools=[get_weather],
            max_iterations=3,
        )
        result = await run_with_retry(
            lambda: agent.run("Use the get_weather tool to look up the weather in Amsterdam."),
            max_attempts=2,
        )

        # --- Error-path assertion: isError=True on the MCP wire maps to
        #     ToolExecutionError for the invoking code. ---
        always_fail = next(t for t in mcp_tools if t.schema.name == "always_fail")
        with pytest.raises(ToolExecutionError):
            await always_fail.execute()

        # Keep a reference to a valid tool so we can probe the post-exit
        # lifecycle contract once the ``async with`` block unwinds.
        leaked_tool = get_weather

    # --- Invocation / result events ---
    invoke = assert_trace_contains(
        traced_emitter,
        ToolInvokeEvent,
        predicate=lambda e: e.tool_name == "get_weather",
    )
    assert_trace_contains(traced_emitter, ToolResultEvent, predicate=lambda e: e.success is True)

    # --- Round-trip fidelity: what the agent sent matches what the server saw. ---
    assert state.call_count >= 1, "Expected the MCP server to observe at least one call"
    invoked_city = invoke.parameters.get("city")
    assert invoked_city == state.last_city, (
        f"Round-trip mismatch: agent sent city={invoked_city!r}, server saw {state.last_city!r}"
    )

    # --- Final answer check ---
    await assert_result_satisfies(
        result.output or "",
        "The output reports that the weather in Amsterdam is sunny and warm.",
    )

    # --- Post-exit lifecycle: MCPTool.execute() after __aexit__ must raise
    #     LLMProviderError because the session streams are closed. ---
    assert leaked_tool is not None
    with pytest.raises(LLMProviderError):
        await leaked_tool.execute(city="Amsterdam")
