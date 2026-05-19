"""MCP (Model Context Protocol) client integration.

This subpackage exposes an MCP *client* that connects to any MCP-compatible
server over stdio or SSE transport and surfaces its tools as ordinary
Nanitics :class:`~nanitics.strategies.tools.protocol.Tool` instances.  Tools
returned by :meth:`MCPClient.list_tools` satisfy the structural ``Tool``
protocol and dispatch through ``ToolRegistry`` identically to in-process
tools.

The public symbols (``MCPClient``, ``MCPStdioParameters``, ``MCPTool``) are
only available when the optional ``mcp`` extra is installed
(``pip install nanitics[mcp]``).  When the extra is missing, the names are
re-exported as ``None`` so the rest of the SDK imports cleanly.
"""

from __future__ import annotations

try:
    from nanitics.infrastructure.mcp._tool import MCPTool
    from nanitics.infrastructure.mcp.client import MCPClient, MCPStdioParameters
except ImportError:
    MCPClient = None  # type: ignore[assignment,misc]
    MCPStdioParameters = None  # type: ignore[assignment,misc]
    MCPTool = None  # type: ignore[assignment,misc]

__all__ = [
    "MCPClient",
    "MCPStdioParameters",
    "MCPTool",
]
