"""MCP tools: consume any Model Context Protocol server as ordinary Nanitics tools.

Demonstrates that ``MCPClient`` discovers tools from an MCP server and exposes them
as regular ``Tool``-protocol-conforming objects — no new registry, no new dispatch
path, no new event types.  A ``ReActAgent`` treats MCP-backed tools identically to
locally-defined ``FunctionTool`` instances.

Section 1 (always runs) connects to an in-process ``FastMCP`` server over an
in-memory transport pair, so the example is hermetic — no subprocess, no network,
no real LLM.  Section 2 is a commented-out block showing how to connect to a real
stdio MCP server (``@modelcontextprotocol/server-filesystem``) via ``npx``.
Section 3 demonstrates name-collision handling when connecting to multiple MCP
servers by setting a ``name_prefix`` per client.

Related guide: docs/guides/tools.md (see the "MCP Tools" section).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import anyio
from mcp.server.fastmcp import FastMCP

from examples.helpers import make_emitter, make_response
from nanitics.infrastructure import (
    MCPClient,
    MockLLMClient,
    ToolInvokeEvent,
    ToolResultEvent,
)
from nanitics.strategies import ReActAgent
from nanitics.tracing import ToolCall


def _in_memory_transport_factory(server: FastMCP) -> Any:
    """Return an async-context-manager factory compatible with ``MCPClient``.

    Uses ``mcp.shared.memory.create_client_server_memory_streams`` to pair a
    client-side transport with an in-process FastMCP server — no subprocess,
    no network.  This is the same pattern the MCP integration's own test
    suite uses (see ``tests/infrastructure/test_mcp_client.py``).
    """

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
                    await server._mcp_server.run(
                        server_read,
                        server_write,
                        init_options,
                        raise_exceptions=False,
                    )
                except Exception:
                    # Background server task stops when the client disconnects;
                    # we don't care about the reason here.
                    return

            async with anyio.create_task_group() as tg:
                tg.start_soon(run_server)
                try:
                    yield client_read, client_write
                finally:
                    tg.cancel_scope.cancel()

    return _transport


def _build_weather_server() -> FastMCP:
    server = FastMCP(name="weather-demo")

    @server.tool(description="Look up the current weather for a city.")
    def get_weather(city: str) -> str:
        return f"sunny and 22C in {city}"

    return server


def _build_filesystem_server() -> FastMCP:
    server = FastMCP(name="fs-demo")

    @server.tool(description="List files under a path (demo — always returns the same list).")
    def list_files(path: str) -> str:
        return f"{path}/readme.md\n{path}/main.py"

    return server


async def main() -> None:
    # --- Section 1: ReActAgent with MCP-discovered tools (always runs) ---
    print("--- Section 1: ReActAgent with MCP-discovered tools (hermetic) ---")

    weather_server = _build_weather_server()

    llm = MockLLMClient(
        responses=[
            make_response(
                "Let me check.",
                tool_calls=[ToolCall(id="tc-1", name="get_weather", arguments={"city": "Amsterdam"})],
                stop_reason="tool_use",
            ),
            make_response("The weather in Amsterdam is sunny and 22C."),
        ]
    )
    emitter = make_emitter("mcp-section-1")

    mcp_client = MCPClient._for_testing(
        transport_factory=_in_memory_transport_factory(weather_server),
    )

    async with mcp_client as client:
        mcp_tools = await client.list_tools()
        assert [t.schema.name for t in mcp_tools] == ["get_weather"]
        assert mcp_tools[0].schema.description.startswith("[MCP]")

        agent = ReActAgent(
            name="weather-agent",
            llm_client=llm,
            emitter=emitter,
            system_prompt="Use the provided tools to answer the user.",
            tools=mcp_tools,
        )
        result = await agent.run("What's the weather in Amsterdam?")

    assert result.termination_reason == "complete"
    assert result.output is not None
    assert "sunny" in result.output

    # The MCP tool was dispatched through the standard ToolRegistry and emitted
    # the same events a FunctionTool would have — protocol parity.
    tool_invokes = [e for e in emitter.events if isinstance(e, ToolInvokeEvent)]
    tool_results = [e for e in emitter.events if isinstance(e, ToolResultEvent)]
    assert [e.tool_name for e in tool_invokes] == ["get_weather"]
    assert tool_results[0].success is True
    assert tool_results[0].result is not None
    assert "sunny" in tool_results[0].result

    print(f"  Output: {result.output}")
    print(f"  MCP tools discovered: {[t.schema.name for t in mcp_tools]}")
    print(f"  Events: {len(tool_invokes)} invoke, {len(tool_results)} result")
    print("✓ MCP tools behave exactly like FunctionTools inside the agent loop")

    # --- Section 2: Real stdio MCP server (commented out — requires npx) ---
    print("\n--- Section 2: Real stdio MCP server (commented out — requires npx) ---")
    print("  See the source of this file for the runnable block.")
    # To run against the real MCP filesystem server, install the MCP extra:
    #
    #     pip install nanitics[mcp]
    #
    # and ensure ``npx`` is on your PATH (Node.js 18+).
    #
    # ----------------------------------------------------------------------------
    # import os
    # from nanitics import AnthropicLLMClient, MCPStdioParameters
    #
    # real_llm = AnthropicLLMClient(model="claude-haiku-4-5")  # reads ANTHROPIC_API_KEY
    # params = MCPStdioParameters(
    #     command="npx",
    #     args=["-y", "@modelcontextprotocol/server-filesystem", os.path.expanduser("~")],
    # )
    # async with MCPClient.stdio(params) as client:
    #     tools = await client.list_tools()
    #     agent = ReActAgent(
    #         name="fs-agent",
    #         llm_client=real_llm,
    #         emitter=make_emitter("mcp-fs"),
    #         system_prompt="Use the filesystem tools to help the user.",
    #         tools=tools,
    #     )
    #     result = await agent.run("List the top-level files in my home directory.")
    #     print(result.output)
    # ----------------------------------------------------------------------------

    # --- Section 3: Multiple MCP servers with name_prefix (always runs) ---
    print("\n--- Section 3: Multiple MCP servers with name_prefix ---")

    weather_server = _build_weather_server()
    fs_server = _build_filesystem_server()

    weather_client = MCPClient._for_testing(
        transport_factory=_in_memory_transport_factory(weather_server),
        name_prefix="weather_",
    )
    fs_client = MCPClient._for_testing(
        transport_factory=_in_memory_transport_factory(fs_server),
        name_prefix="fs_",
    )

    async with weather_client as w, fs_client as f:
        weather_tools = await w.list_tools()
        fs_tools = await f.list_tools()
        combined = weather_tools + fs_tools

        # The registry inside the agent accepts both prefixed sets without
        # collision, even though each server also exposes its own short name.
        combined_names = [t.schema.name for t in combined]
        assert combined_names == ["weather_get_weather", "fs_list_files"]

        # Drive each prefixed tool directly — the prefix stays out of the call
        # the MCP client sends to the server.
        weather_result = await combined[0].execute(city="Paris")
        fs_result = await combined[1].execute(path="/tmp")
        assert weather_result.content == "sunny and 22C in Paris"
        assert "/tmp/readme.md" in fs_result.content

    print(f"  Prefixed tools available to the agent: {combined_names}")
    print(f"  weather_get_weather(Paris): {weather_result.content}")
    print(f"  fs_list_files(/tmp): {fs_result.content.splitlines()[0]}, ...")
    print("✓ name_prefix makes cross-server tool sets collision-free")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
