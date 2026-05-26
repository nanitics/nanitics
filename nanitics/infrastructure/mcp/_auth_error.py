"""Classification helper for MCP HTTP auth (401/403) failures.

The classifier walks the ``__cause__`` / ``__context__`` chain of an
exception raised at one of the two MCP raise sites (``MCPClient.__aenter__``
and ``MCPTool.execute``) and returns a typed :class:`MCPAuthError` when it
finds an ``httpx.HTTPStatusError`` with status 401 or 403. Otherwise it
returns ``None``, letting the caller fall through to the generic
:class:`LLMProviderError` mapping.
"""

from __future__ import annotations

import httpx

from nanitics.infrastructure.errors import MCPAuthError

_MAX_CHAIN_DEPTH = 10


def _classify_mcp_auth_error(exc: BaseException) -> MCPAuthError | None:
    """Walk the exception chain looking for an MCP auth (401/403) cause.

    Returns a constructed :class:`MCPAuthError` when the chain contains an
    ``httpx.HTTPStatusError`` with status 401 or 403; otherwise returns
    ``None``. Never raises.

    Walks both ``__cause__`` and ``__context__`` so the helper works whether
    the upstream ``mcp`` library uses ``raise ... from ...`` or a bare
    re-raise inside an ``except``. Bounded at 10 hops to defend against
    cycles or pathological nesting depth.
    """

    current: BaseException | None = exc
    depth = 0
    seen: set[int] = set()
    while current is not None and depth < _MAX_CHAIN_DEPTH:
        if id(current) in seen:
            return None
        seen.add(id(current))
        if isinstance(current, httpx.HTTPStatusError):
            status_code = current.response.status_code
            if status_code in (401, 403):
                try:
                    www_authenticate = current.response.headers.get("www-authenticate")
                except Exception:
                    www_authenticate = None
                return MCPAuthError(
                    f"MCP HTTP transport returned {status_code}: {current!s}",
                    status_code=status_code,
                    www_authenticate=www_authenticate,
                )
        current = current.__cause__ if current.__cause__ is not None else current.__context__
        depth += 1
    return None
