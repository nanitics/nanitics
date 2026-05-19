"""``create_http_tool`` — a curated HTTP request reference tool.

The factory returns a :class:`~nanitics.strategies.tools.protocol.Tool`-conforming
object that issues HTTP requests through :mod:`httpx`.  Requests are gated
by an explicit domain allow-list (or the ``allow_any_domain=True`` escape
hatch), response bodies are bounded by ``max_response_bytes`` (default
1 MiB), and timeouts raise :class:`~nanitics.infrastructure.errors.ToolTimeoutError`.

The tool deliberately does NOT raise on 4xx/5xx statuses — those are
surfaced through ``ToolResult.metadata["status"]`` so the LLM can observe
the status code and adapt.  Only transport failures, timeouts, and
parameter-validation failures raise.

This module requires the ``http-tools`` extra.  When :mod:`httpx` is not
installed, importing the module raises :class:`ImportError` with an
install-hint message; ``nanitics.tools.__init__`` wraps that import in a
``try`` / ``except`` so ``import nanitics.tools`` still succeeds.
"""

from __future__ import annotations

import json
from typing import Any, Literal
from urllib.parse import urljoin, urlparse

from pydantic import BaseModel, Field

from nanitics.infrastructure.errors import (
    ToolExecutionError,
    ToolParameterError,
    ToolTimeoutError,
)
from nanitics.strategies.tools.function_tool import FunctionTool
from nanitics.strategies.tools.protocol import Tool, ToolResult
from nanitics.tools._result_models import HttpResponse

try:
    import httpx
except ImportError as _err:  # pragma: no cover
    raise ImportError("create_http_tool requires the 'http-tools' extra: pip install nanitics[http-tools]") from _err


_DEFAULT_MAX_RESPONSE_BYTES = 1_048_576  # 1 MiB
_REDIRECT_STATUS_CODES: frozenset[int] = frozenset({301, 302, 303, 307, 308})

HttpMethod = Literal["GET", "POST", "PUT", "DELETE"]


class _HttpParams(BaseModel):
    """Parameters accepted by the ``http_request`` tool."""

    method: HttpMethod = Field(description="HTTP method: GET, POST, PUT, or DELETE.")
    url: str = Field(min_length=1, description="Absolute URL to request.")
    headers: dict[str, str] | None = Field(
        default=None,
        description="Request headers; merged on top of the factory's default_headers.",
    )
    query_params: dict[str, str] | None = Field(
        default=None,
        description="Query-string parameters appended to the URL.",
    )
    body: str | dict[str, Any] | None = Field(
        default=None,
        description="Request body. dict is JSON-encoded, str is sent as-is, None omits.",
    )


def _host_allowed(host: str, allowed_domains: list[str]) -> bool:
    """Return ``True`` if *host* matches any entry in *allowed_domains*.

    Comparison is case-insensitive and exact (no wildcard / subdomain
    expansion).
    """
    host_lc = host.lower()
    return any(host_lc == d.lower() for d in allowed_domains)


def _merge_headers(
    default_headers: dict[str, str] | None,
    request_headers: dict[str, str] | None,
) -> dict[str, str]:
    """Merge *default_headers* with *request_headers*.

    Request-level headers take precedence on key collision.
    """
    merged: dict[str, str] = {}
    if default_headers:
        merged.update(default_headers)
    if request_headers:
        merged.update(request_headers)
    return merged


def _render_content(status: int, body: str) -> str:
    """Render the LLM-visible content string for an HTTP response."""
    return f"HTTP {status}\n\n{body}"


def create_http_tool(
    *,
    allowed_domains: list[str] | None = None,
    allow_any_domain: bool = False,
    default_headers: dict[str, str] | None = None,
    request_timeout: float = 30.0,
    max_response_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES,
    max_redirects: int = 5,
    name: str = "http_request",
    description: str | None = None,
) -> Tool:
    """Create an HTTP-request tool with an explicit domain allow-list.

    The returned object satisfies :class:`~nanitics.strategies.tools.protocol.Tool`
    and can be registered in :class:`~nanitics.strategies.ToolRegistry` alongside
    any other tool.  The tool constructs a fresh
    :class:`httpx.AsyncClient` per call and follows redirects manually so
    every hop is validated against the allow-list.

    Security posture: one of ``allowed_domains`` (non-empty) or
    ``allow_any_domain=True`` must be supplied at construction time.  At
    call time, a request whose host does not match any allow-listed domain
    raises :class:`ToolParameterError` so the LLM is told the URL was
    invalid and can correct itself.  Every redirect hop is revalidated
    against the same allow-list — a 302 from an allow-listed origin to a
    disallowed host raises :class:`ToolParameterError` and the redirect
    target is not contacted.  Response bodies are bounded by
    ``max_response_bytes`` and truncated with metadata flags when
    exceeded.

    4xx and 5xx responses are NOT raised — the status is returned via
    ``ToolResult.metadata["status"]``.

    Args:
        allowed_domains: Explicit list of permitted hosts (exact,
            case-insensitive).  Mutually exclusive with
            ``allow_any_domain=True``.
        allow_any_domain: Escape hatch for trusted environments.  When
            ``True``, ``allowed_domains`` may be omitted.
        default_headers: Headers merged into every request.  Per-request
            ``headers`` override these on key collision.
        request_timeout: Per-call timeout in seconds.  When exceeded the
            tool raises :class:`ToolTimeoutError`.
        max_response_bytes: Upper bound on the response body length in
            bytes.  Bodies larger than this are truncated and the
            ``truncated`` / ``bytes_read`` metadata fields record the
            effect.
        max_redirects: Maximum number of redirect hops to follow before
            raising :class:`ToolParameterError`.  ``0`` disables redirect
            following entirely — a 3xx response is surfaced as-is.
            Defaults to ``5``.
        name: Tool name exposed to the LLM.  Defaults to
            ``"http_request"``.
        description: Optional override of the LLM-facing description.

    Returns:
        A :class:`Tool`-conforming object.

    Raises:
        ValueError: If neither ``allowed_domains`` nor ``allow_any_domain``
            is supplied, or if ``allowed_domains`` is empty without
            ``allow_any_domain=True``, or if ``max_redirects`` is
            negative.
    """
    if not allow_any_domain and not allowed_domains:
        raise ValueError("create_http_tool requires a non-empty allowed_domains list or allow_any_domain=True")
    if max_redirects < 0:
        raise ValueError(f"max_redirects must be non-negative, got {max_redirects}")
    # Snapshot of allow-list for the closure.
    effective_allowed: list[str] = list(allowed_domains or [])
    effective_description = description or (
        "Issue an HTTP request (GET, POST, PUT, DELETE) against an allow-listed domain. "
        f"Response body is truncated to {max_response_bytes} bytes."
    )

    async def _execute(
        method: HttpMethod,
        url: str,
        headers: dict[str, str] | None = None,
        query_params: dict[str, str] | None = None,
        body: str | dict[str, Any] | None = None,
    ) -> ToolResult:
        parsed = urlparse(url)
        host = parsed.hostname
        if not host or not parsed.scheme:
            raise ToolParameterError(
                f"Invalid URL: '{url}' — missing scheme or host",
                tool_name="http_request",
                parameter_name="url",
                reason="missing scheme or host",
            )
        if not allow_any_domain and not _host_allowed(host, effective_allowed):
            raise ToolParameterError(
                f"host '{host}' not in allowed_domains",
                tool_name="http_request",
                parameter_name="url",
                reason=f"host '{host}' not in allowed_domains",
            )

        merged_headers = _merge_headers(default_headers, headers)

        request_kwargs: dict[str, Any] = {}
        if isinstance(body, dict):
            request_kwargs["content"] = json.dumps(body).encode("utf-8")
            # Only default Content-Type when the caller did not supply one.
            if not any(k.lower() == "content-type" for k in merged_headers):
                merged_headers["Content-Type"] = "application/json"
        elif isinstance(body, str):
            request_kwargs["content"] = body.encode("utf-8")

        if query_params:
            request_kwargs["params"] = query_params
        if merged_headers:
            request_kwargs["headers"] = merged_headers

        try:
            async with httpx.AsyncClient(
                timeout=request_timeout,
                follow_redirects=False,
            ) as client:
                current_url = url
                hops = 0
                response = await client.request(method, current_url, **request_kwargs)
                while (
                    max_redirects > 0
                    and response.status_code in _REDIRECT_STATUS_CODES
                    and "location" in {k.lower() for k in response.headers}
                ):
                    target = urljoin(str(response.url), response.headers["Location"])
                    parsed_target = urlparse(target)
                    target_host = parsed_target.hostname
                    if not target_host or not parsed_target.scheme:
                        raise ToolParameterError(
                            f"redirect target URL is invalid: {target}",
                            tool_name="http_request",
                            parameter_name="url",
                            reason=f"redirect target URL is invalid: {target}",
                        )
                    if hops >= max_redirects:
                        raise ToolParameterError(
                            f"redirect limit {max_redirects} exceeded",
                            tool_name="http_request",
                            parameter_name="url",
                            reason=f"redirect limit {max_redirects} exceeded",
                        )
                    if not allow_any_domain and not _host_allowed(target_host, effective_allowed):
                        raise ToolParameterError(
                            f"redirect target host '{target_host}' not in allowed_domains",
                            tool_name="http_request",
                            parameter_name="url",
                            reason=f"redirect target host '{target_host}' not in allowed_domains",
                        )
                    hops += 1
                    current_url = target
                    response = await client.request(method, current_url, **request_kwargs)
        except httpx.TimeoutException as exc:
            raise ToolTimeoutError(
                f"http_request timed out after {request_timeout}s",
                tool_name="http_request",
                timeout_seconds=request_timeout,
            ) from exc
        except httpx.RequestError as exc:
            raise ToolExecutionError(
                f"Transport error: {exc}",
                tool_name="http_request",
            ) from exc

        raw = response.content
        if len(raw) > max_response_bytes:
            prefix = raw[:max_response_bytes]
            truncated = True
        else:
            prefix = raw
            truncated = False
        # Fall back to replacement decoding so the LLM still sees something
        # legible when the server returns non-UTF-8 bytes.
        try:
            body_text = prefix.decode("utf-8")
        except UnicodeDecodeError:
            body_text = prefix.decode("utf-8", errors="replace")

        metadata = HttpResponse(
            status=response.status_code,
            headers={k.lower(): v for k, v in response.headers.items()},
            body=body_text,
            truncated=truncated,
            bytes_read=len(prefix),
            url=str(response.url),
        ).model_dump()

        return ToolResult(
            content=_render_content(response.status_code, body_text),
            metadata=metadata,
        )

    return FunctionTool(
        fn=_execute,
        name=name,
        description=effective_description,
        parameters_model=_HttpParams,
    )
