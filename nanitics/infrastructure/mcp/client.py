"""User-facing MCP client connector.

``MCPClient`` owns the lifecycle of an MCP session: it enters the transport
context (stdio or SSE), runs the initialization handshake, exposes
``list_tools()`` to discover server-provided tools, and cleans up on exit.
The class is an async context manager — enter the ``async with`` block to
use it, and exit it to release the subprocess or HTTP connection.

``MCPStdioParameters`` mirrors the upstream ``StdioServerParameters`` so
users importing ``nanitics`` do not need to depend on the ``mcp`` package
symbol directly for the common case.

Current scope:

* Transports: stdio (via ``mcp.client.stdio.stdio_client``), SSE (via
  ``mcp.client.sse.sse_client``), and Streamable HTTP (via
  ``mcp.client.streamable_http.streamablehttp_client``).
* Tools only — no resources, prompts, sampling callbacks, or dynamic
  ``tools/list_changed`` re-discovery.
* Tool list is cached on first ``list_tools()`` call.

Lifecycle contract:

* Before ``__aenter__``, calling any method raises ``RuntimeError``.
* Between ``__aenter__`` and ``__aexit__``, ``list_tools()`` returns an
  ``MCPTool`` per server-exposed tool (filtered and prefixed per config).
* After ``__aexit__``, ``MCPTool`` instances returned earlier remain
  accessible but calling ``.execute()`` raises ``LLMProviderError`` because
  the underlying session streams are closed.
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self, TextIO

import httpx

from nanitics.infrastructure.errors import LLMProviderError
from nanitics.infrastructure.mcp._tool import MCPTool
from nanitics.infrastructure.mcp._translation import mcp_tool_to_schema

try:
    from mcp import ClientSession
    from mcp import StdioServerParameters as _UpstreamStdioParameters
    from mcp import stdio_client as _stdio_client
    from mcp.client.sse import sse_client as _sse_client

    # ``streamablehttp_client`` is the legacy upstream symbol kept for its
    # ``headers=`` keyword argument; the newer ``streamable_http_client``
    # requires the caller to pre-construct an ``httpx.AsyncClient``. We
    # consciously prefer the legacy symbol for parameter parity with
    # ``MCPClient.sse``. Silence the import-time DeprecationWarning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from mcp.client.streamable_http import streamablehttp_client as _streamablehttp_client
except ImportError as _err:  # pragma: no cover
    raise ImportError("MCPClient requires the 'mcp' extra: pip install nanitics[mcp]") from _err

if TYPE_CHECKING:
    from types import TracebackType


_MCP_SOURCE_PREFIX = "[MCP]"


# Type alias kept private — the public surface is the ``headers_provider``
# parameter on the two HTTP factories. Documented inline in those factories'
# docstrings, not re-exported.
_HeadersProvider = Callable[[], Awaitable[dict[str, str]]]


class _HeadersProviderAuth(httpx.Auth):
    """Per-request httpx auth that delegates header production to a caller-supplied async provider.

    Lives alongside ``MCPClient`` rather than in a sibling ``_auth.py`` module
    because the class is small, private, and tightly coupled to the two HTTP
    factories that construct it — splitting would force a cross-module import
    for no readability gain. If the file ever grows another auth variant,
    revisit the placement.

    Behaviour:
        * On every outgoing request, ``async_auth_flow`` awaits the stored
          provider and merges the returned mapping over ``request.headers``,
          with provider keys winning on conflict.
        * No caching, no fallback, no retry, no logging. The provider's
          contract (cache internally, refresh near expiry, raise on hard
          failure) is the adopter's responsibility.
        * Provider exceptions propagate unchanged from the awaiting request.
    """

    requires_request_body = False
    requires_response_body = False

    def __init__(self, provider: _HeadersProvider) -> None:
        self._provider = provider

    async def async_auth_flow(
        self,
        request: httpx.Request,
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        overlay = await self._provider()
        for key, value in overlay.items():
            request.headers[key] = value
        yield request


@dataclass(frozen=True)
class MCPStdioParameters:
    """Parameters for connecting to an MCP server over stdio.

    Mirrors :class:`mcp.StdioServerParameters` but re-exported so users can
    import directly from ``nanitics`` without depending on the upstream
    symbol in application code.

    Attributes:
        command: The executable to spawn (``"npx"``, ``"uvx"``, an absolute
            path, etc).
        args: Command-line arguments passed to the executable.
        env: Environment variables for the subprocess.  ``None`` inherits
            the parent process environment.
        cwd: Working directory for the subprocess.  ``None`` uses the
            current working directory.
    """

    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | None = None

    def _to_upstream(self) -> _UpstreamStdioParameters:
        return _UpstreamStdioParameters(
            command=self.command,
            args=list(self.args),
            env=self.env,
            cwd=self.cwd,
        )


class MCPClient:
    """Async context manager that owns an MCP session and exposes its tools.

    Users never call ``__init__`` directly — use :meth:`stdio` or :meth:`sse`
    to construct an instance bound to a transport, then use ``async with``.

    Example:
        >>> params = MCPStdioParameters(
        ...     command="npx",
        ...     args=["-y", "server-filesystem", "/tmp"],
        ... )
        >>> async with MCPClient.stdio(params) as client:
        ...     tools = await client.list_tools()
        ...     # pass ``tools`` to a ReActAgent or register in a ToolRegistry

    Lifecycle:
        * The ``async with`` block bounds the session — tools obtained
          inside the block fail with ``LLMProviderError`` when invoked
          after exit.
        * ``discovery_timeout`` bounds the MCP initialization handshake
          and ``tools/list`` combined; on timeout,
          :class:`~nanitics.infrastructure.errors.LLMProviderError` is
          raised.
        * Per-tool execution timeouts come from
          ``ToolSchema.timeout_seconds`` (rare; server-declared) or the
          ``default_call_timeout`` set on this client.
    """

    def __init__(
        self,
        *,
        transport_factory: Callable[[], contextlib.AbstractAsyncContextManager[tuple[Any, ...]]],
        name_prefix: str = "",
        name_filter: Callable[[str], bool] | None = None,
        discovery_timeout: float | None = 30.0,
        default_call_timeout: float | None = 60.0,
        session_wrapper: Callable[[ClientSession], Any] | None = None,
    ) -> None:
        # Kept public-adjacent: the factories are the supported entry points,
        # but the constructor is the seam tests use to inject an in-memory
        # transport.  We do not document this in user-facing docs.
        self._transport_factory = transport_factory
        self._name_prefix = name_prefix
        self._name_filter = name_filter
        self._discovery_timeout = discovery_timeout
        self._default_call_timeout = default_call_timeout
        self._session_wrapper = session_wrapper

        self._entered: bool = False
        self._session: Any = None
        self._stack: contextlib.AsyncExitStack | None = None
        self._cached_tools: list[MCPTool] | None = None

    # ---- Factories -------------------------------------------------------

    @classmethod
    def stdio(
        cls,
        parameters: MCPStdioParameters,
        *,
        name_prefix: str = "",
        name_filter: Callable[[str], bool] | None = None,
        discovery_timeout: float | None = 30.0,
        default_call_timeout: float | None = 60.0,
        errlog: TextIO | None = None,
    ) -> MCPClient:
        """Connect to an MCP server over stdio (spawning the given command).

        Args:
            parameters: Stdio transport parameters (command, args, env, cwd).
            name_prefix: Prefix prepended to every discovered tool's name.
            name_filter: Predicate over server-side tool names; tools for
                which it returns ``False`` are skipped during discovery.
            discovery_timeout: Bounds the MCP initialization handshake and
                ``tools/list`` combined.
            default_call_timeout: Bounds each ``execute()`` call when the
                tool's schema does not declare its own timeout.
            errlog: Optional writable text stream that receives the child
                process's ``stderr``. ``None`` (the default) preserves the
                upstream behavior of routing child stderr to
                ``sys.stderr``. Stdio-only; the SSE and Streamable HTTP
                transports have no child process and therefore no stderr
                to redirect. The caller owns the stream's lifetime — the
                SDK does not open, close, or seek it.
        """

        upstream = parameters._to_upstream()

        def _factory() -> contextlib.AbstractAsyncContextManager[tuple[Any, ...]]:
            if errlog is None:
                return _stdio_client(upstream)
            return _stdio_client(upstream, errlog=errlog)

        return cls(
            transport_factory=_factory,
            name_prefix=name_prefix,
            name_filter=name_filter,
            discovery_timeout=discovery_timeout,
            default_call_timeout=default_call_timeout,
        )

    @classmethod
    def sse(
        cls,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        headers_provider: _HeadersProvider | None = None,
        name_prefix: str = "",
        name_filter: Callable[[str], bool] | None = None,
        discovery_timeout: float | None = 30.0,
        default_call_timeout: float | None = 60.0,
    ) -> MCPClient:
        """Connect to an MCP server over Server-Sent Events.

        Args:
            url: The MCP server endpoint URL.
            headers: Static headers set once on the underlying
                ``httpx.AsyncClient`` at construction. Recommended for
                stable identity headers (User-Agent, tenant ID, tracing
                correlation IDs) that do not rotate during a session.
            headers_provider: Optional async callable
                ``Callable[[], Awaitable[dict[str, str]]]`` invoked
                **before every outgoing HTTP request** to produce a
                rotating header overlay. SSE collapses "per request"
                naturally to "on connect + on each POST", so the provider
                runs on the initial SSE GET and on every subsequent POST
                to the discovered endpoint — no special-casing.

                The returned mapping is merged into the request's headers
                with **provider keys winning on conflict** against the
                static ``headers`` set, against MCP-protocol headers, or
                any other pre-existing header on the request.

                The provider is **expected to cache internally** and only
                perform real refresh work near token expiry — the standard
                OAuth client pattern. The SDK does not cache provider
                returns.

                Provider exceptions propagate unchanged from the request
                that triggered the invocation. No fallback to stale
                headers, no swallowing, no retry. Other in-flight or
                subsequent requests are unaffected — the session is not
                poisoned. Adopters who need resilience implement it inside
                their provider, where the refresh context lives.
            name_prefix: Prefix prepended to every discovered tool's name.
            name_filter: Predicate over server-side tool names; tools for
                which it returns ``False`` are skipped during discovery.
            discovery_timeout: Bounds the MCP initialization handshake and
                ``tools/list`` combined.
            default_call_timeout: Bounds each ``execute()`` call when the
                tool's schema does not declare its own timeout.
        """

        auth = _HeadersProviderAuth(headers_provider) if headers_provider is not None else None

        def _factory() -> contextlib.AbstractAsyncContextManager[tuple[Any, ...]]:
            return _sse_client(url, headers=headers, auth=auth)

        return cls(
            transport_factory=_factory,
            name_prefix=name_prefix,
            name_filter=name_filter,
            discovery_timeout=discovery_timeout,
            default_call_timeout=default_call_timeout,
        )

    @classmethod
    def streamable_http(
        cls,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        headers_provider: _HeadersProvider | None = None,
        name_prefix: str = "",
        name_filter: Callable[[str], bool] | None = None,
        discovery_timeout: float | None = 30.0,
        default_call_timeout: float | None = 60.0,
    ) -> MCPClient:
        """Connect to an MCP server over the Streamable HTTP transport.

        Mirrors :meth:`sse` parameter-for-parameter. The upstream
        ``streamablehttp_client`` yields a 3-tuple
        ``(read, write, get_session_id)``; the session-id getter is
        intentionally dropped here — surfacing it as a public property is
        a follow-up.

        Args:
            url: The MCP server endpoint URL.
            headers: Static headers set once on the underlying
                ``httpx.AsyncClient`` at construction. Recommended for
                stable identity headers (User-Agent, tenant ID, tracing
                correlation IDs) that do not rotate during a session.
            headers_provider: Optional async callable
                ``Callable[[], Awaitable[dict[str, str]]]`` invoked
                **before every outgoing HTTP request** to produce a
                rotating header overlay. The provider runs on the
                initialization POST, on each ``list_tools`` POST, on each
                ``call_tool`` POST, and on the session-termination DELETE.

                The returned mapping is merged into the request's headers
                with **provider keys winning on conflict** against the
                static ``headers`` set, against MCP-protocol headers
                (``Mcp-Session-Id``, ``Mcp-Protocol-Version``), or any
                other pre-existing header on the request. Adopters are
                expected not to return MCP-protocol headers from the
                provider.

                The provider is **expected to cache internally** and only
                perform real refresh work near token expiry — the standard
                OAuth client pattern. The SDK does not cache provider
                returns.

                Provider exceptions propagate unchanged from the request
                that triggered the invocation. No fallback to stale
                headers, no swallowing, no retry. Other in-flight or
                subsequent requests are unaffected — the session is not
                poisoned. Adopters who need resilience implement it inside
                their provider, where the refresh context lives.
            name_prefix: Prefix prepended to every discovered tool's name.
            name_filter: Predicate over server-side tool names; tools for
                which it returns ``False`` are skipped during discovery.
            discovery_timeout: Bounds the MCP initialization handshake and
                ``tools/list`` combined.
            default_call_timeout: Bounds each ``execute()`` call when the
                tool's schema does not declare its own timeout.
        """

        auth = _HeadersProviderAuth(headers_provider) if headers_provider is not None else None

        def _factory() -> contextlib.AbstractAsyncContextManager[tuple[Any, ...]]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                return _streamablehttp_client(url, headers=headers, auth=auth)

        return cls(
            transport_factory=_factory,
            name_prefix=name_prefix,
            name_filter=name_filter,
            discovery_timeout=discovery_timeout,
            default_call_timeout=default_call_timeout,
        )

    @classmethod
    def _for_testing(
        cls,
        *,
        transport_factory: Callable[[], contextlib.AbstractAsyncContextManager[tuple[Any, ...]]],
        name_prefix: str = "",
        name_filter: Callable[[str], bool] | None = None,
        discovery_timeout: float | None = 30.0,
        default_call_timeout: float | None = 60.0,
        session_wrapper: Callable[[ClientSession], Any] | None = None,
    ) -> MCPClient:
        """Construct an MCPClient bound to a caller-supplied transport factory.

        Test-only seam — keeps the in-memory transport wiring out of the
        public API while exercising the real lifecycle logic.
        """

        return cls(
            transport_factory=transport_factory,
            name_prefix=name_prefix,
            name_filter=name_filter,
            discovery_timeout=discovery_timeout,
            default_call_timeout=default_call_timeout,
            session_wrapper=session_wrapper,
        )

    # ---- Context manager -------------------------------------------------

    async def __aenter__(self) -> Self:
        self._stack = contextlib.AsyncExitStack()
        await self._stack.__aenter__()
        try:
            # 1. Enter the transport context to get streams.
            #    stdio/SSE yield a 2-tuple ``(read, write)``; streamable HTTP
            #    yields a 3-tuple ``(read, write, get_session_id)`` — the
            #    session-id getter is intentionally dropped for now (see
            #    design-rationale §5).
            streams = await self._stack.enter_async_context(self._transport_factory())
            if len(streams) == 3:
                read_stream, write_stream, _get_session_id = streams
            else:
                read_stream, write_stream = streams
            # 2. Open a ClientSession over the streams.
            raw_session: ClientSession = await self._stack.enter_async_context(ClientSession(read_stream, write_stream))
            # 3. Run initialization handshake, bounded by discovery_timeout.
            import asyncio as _asyncio

            try:
                if self._discovery_timeout is not None:
                    async with _asyncio.timeout(self._discovery_timeout):
                        await raw_session.initialize()
                else:
                    await raw_session.initialize()
            except TimeoutError as exc:
                raise LLMProviderError(
                    f"MCP initialization timed out after {self._discovery_timeout}s",
                    provider="mcp",
                ) from exc

            self._session = self._session_wrapper(raw_session) if self._session_wrapper is not None else raw_session
        except BaseException:
            # Roll back anything already pushed onto the stack, then rethrow.
            # `_safe_rollback` suppresses cleanup-phase exceptions so the
            # primary error (LLMProviderError, transport failure, etc.) is
            # what propagates to the caller.
            await _safe_rollback(self._stack)
            self._stack = None
            raise

        self._entered = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._entered = False
        stack = self._stack
        self._stack = None
        self._session = None
        if stack is not None:
            await stack.__aexit__(exc_type, exc, tb)

    # ---- Discovery -------------------------------------------------------

    async def list_tools(self) -> list[MCPTool]:
        """Return the tools exposed by the connected MCP server.

        Cached on first call for the lifetime of the session.  Applies
        ``name_prefix`` and ``name_filter`` configured on this client.
        """

        if not self._entered or self._session is None:
            raise RuntimeError(
                "MCPClient must be used as an async context manager: `async with client as c: await c.list_tools()`."
            )

        if self._cached_tools is not None:
            return self._cached_tools

        import asyncio as _asyncio

        try:
            if self._discovery_timeout is not None:
                async with _asyncio.timeout(self._discovery_timeout):
                    result = await self._session.list_tools()
            else:
                result = await self._session.list_tools()
        except TimeoutError as exc:
            raise LLMProviderError(
                f"MCP tool discovery timed out after {self._discovery_timeout}s",
                provider="mcp",
            ) from exc

        tools: list[MCPTool] = []
        for mcp_tool in result.tools:
            if self._name_filter is not None and not self._name_filter(mcp_tool.name):
                continue
            schema = mcp_tool_to_schema(
                mcp_tool,
                name_prefix=self._name_prefix,
                description_prefix=_MCP_SOURCE_PREFIX,
            )
            tools.append(
                MCPTool(
                    schema=schema,
                    mcp_tool_name=mcp_tool.name,
                    session=self._session,
                    default_timeout=self._default_call_timeout,
                )
            )

        self._cached_tools = tools
        return tools


async def _safe_rollback(stack: contextlib.AsyncExitStack) -> None:
    """Close an ``AsyncExitStack`` silently, swallowing cleanup-only errors.

    During startup rollback (initialization timeout, transport factory error),
    the underlying ``ClientSession``'s background task group can surface
    ``CancelledError``/``ExceptionGroup`` instances as it tears down mid-RPC.
    Those are expected consequences of the failure we are already about to
    raise — surfacing them too would obscure the true error.

    We close the stack here and drop any exceptions it raises.  The primary
    error (transport factory failure, ``LLMProviderError`` from the timeout
    path, etc.) is raised separately by the caller.
    """

    with contextlib.suppress(BaseException):
        await stack.aclose()
