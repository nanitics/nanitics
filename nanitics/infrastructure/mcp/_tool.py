"""Tool-protocol-conforming wrapper around a single MCP server tool.

``MCPTool`` is constructed internally by :class:`MCPClient` during
``list_tools()`` — users should not instantiate it directly.  The class is
exposed publicly only for type hints (e.g. ``list[MCPTool]`` as the return
type of ``MCPClient.list_tools``).

Design highlights:

* Structural conformance to :class:`~nanitics.strategies.tools.protocol.Tool` via
  the runtime-checkable ``Tool`` protocol — no inheritance.
* Does NOT own the session.  When the owning ``MCPClient`` exits its
  ``async with`` block, any subsequent call to :meth:`execute` fails fast
  with :class:`~nanitics.infrastructure.errors.LLMProviderError` because the
  underlying session streams are closed.
* Error translation follows the SDK error hierarchy:
  ``CallToolResult.isError`` and upstream ``McpError`` map to
  :class:`~nanitics.infrastructure.errors.ToolExecutionError`; transport
  failures map to :class:`~nanitics.infrastructure.errors.LLMProviderError`
  with ``provider="mcp"``; HTTP 401/403 on the underlying call map to
  :class:`~nanitics.infrastructure.errors.MCPAuthError`; timeouts map to
  :class:`~nanitics.infrastructure.errors.ToolTimeoutError`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from nanitics.infrastructure.errors import (
    LLMProviderError,
    ToolExecutionError,
    ToolTimeoutError,
)
from nanitics.infrastructure.llm.protocol import ToolSchema
from nanitics.infrastructure.mcp._auth_error import _classify_mcp_auth_error
from nanitics.infrastructure.mcp._translation import call_result_to_tool_result
from nanitics.strategies.tools.protocol import ToolResult

try:
    from mcp import McpError
except ImportError as _err:  # pragma: no cover
    raise ImportError("MCPTool requires the 'mcp' extra: pip install nanitics[mcp]") from _err

if TYPE_CHECKING:
    from mcp import ClientSession


class MCPTool:
    """Wrapper exposing a single MCP server tool as a Nanitics ``Tool``.

    Instances are constructed by :meth:`MCPClient.list_tools` and share the
    session owned by that client.  They are valid only within the client's
    ``async with`` block.

    Args:
        schema: The translated ``ToolSchema``.  Its ``name`` may carry a
            prefix applied by the client; the unprefixed server-side name
            is passed separately as ``mcp_tool_name``.
        mcp_tool_name: The name the MCP server uses for this tool.  This
            is the value sent in ``tools/call``.
        session: The MCP ``ClientSession`` owned by the enclosing
            ``MCPClient``.  When the client exits, transport calls on this
            session fail and are translated to :class:`LLMProviderError`.
        default_timeout: Session-level default used when
            ``schema.timeout_seconds`` is ``None``.  ``None`` means no
            deadline at all.
    """

    def __init__(
        self,
        *,
        schema: ToolSchema,
        mcp_tool_name: str,
        session: ClientSession,
        default_timeout: float | None = None,
    ) -> None:
        self._schema = schema
        self._mcp_tool_name = mcp_tool_name
        self._session = session
        self._default_timeout = default_timeout

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    async def execute(self, **params: Any) -> ToolResult:
        effective_timeout = (
            self._schema.timeout_seconds if self._schema.timeout_seconds is not None else self._default_timeout
        )

        try:
            if effective_timeout is not None:
                async with asyncio.timeout(effective_timeout):
                    result = await self._session.call_tool(self._mcp_tool_name, params)
            else:
                result = await self._session.call_tool(self._mcp_tool_name, params)
        except TimeoutError as exc:
            # effective_timeout is necessarily non-None when TimeoutError can
            # fire (we only wrap in asyncio.timeout above); narrow for the
            # type checker.
            assert effective_timeout is not None
            raise ToolTimeoutError(
                f"MCP tool {self._schema.name!r} timed out after {effective_timeout}s",
                tool_name=self._schema.name,
                timeout_seconds=effective_timeout,
            ) from exc
        except McpError as exc:
            raise ToolExecutionError(
                str(exc),
                tool_name=self._schema.name,
            ) from exc
        except asyncio.CancelledError:
            # ``CancelledError`` is ``BaseException`` on Python 3.8+ so it
            # already escapes the ``except Exception`` branch below — this
            # explicit re-raise is a ward against future regressions
            # (e.g. someone broadening to ``except BaseException`` for
            # "robustness" or an anyio shim that surfaces cancellation as
            # a regular ``Exception``).
            raise
        except Exception as exc:
            auth_err = _classify_mcp_auth_error(exc)
            if auth_err is not None:
                raise auth_err from exc
            raise LLMProviderError(
                f"MCP transport failure during {self._schema.name}: {exc}",
                provider="mcp",
            ) from exc

        return call_result_to_tool_result(result, tool_name=self._schema.name)
