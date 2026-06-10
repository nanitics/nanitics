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

The missing-extra case is detected explicitly via :func:`importlib.util.find_spec`
rather than by catching :class:`ImportError` around the real imports.  A blanket
``except ImportError`` would also swallow a *circular-import* or a genuinely
broken module and silently null the symbols out — turning a hard bug into an
invisible, order-dependent loss of all MCP support.  Only a truly absent ``mcp``
distribution maps to ``None``; every other import failure propagates.
"""

from __future__ import annotations

import importlib.util

if importlib.util.find_spec("mcp") is not None:
    # The extra is present — import for real and let any failure surface.
    from nanitics.infrastructure.mcp._tool import MCPTool
    from nanitics.infrastructure.mcp.client import MCPClient, MCPStdioParameters
else:
    # The optional ``mcp`` extra is not installed.
    MCPClient = None  # type: ignore[assignment,misc]
    MCPStdioParameters = None  # type: ignore[assignment,misc]
    MCPTool = None  # type: ignore[assignment,misc]

__all__ = [
    "MCPClient",
    "MCPStdioParameters",
    "MCPTool",
]
