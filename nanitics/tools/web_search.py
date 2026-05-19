"""``create_web_search_tool`` — a curated web-search reference tool.

The factory returns a :class:`~nanitics.strategies.tools.protocol.Tool`-conforming
object that dispatches through :class:`~nanitics.strategies.ToolRegistry`
identically to a :class:`~nanitics.strategies.FunctionTool` — no new registry, no
new event types.  Two backends are supported: Tavily (``provider="tavily"``,
default) and Brave (``provider="brave"``).  A fresh
:class:`httpx.AsyncClient` is constructed per call; the tool does not share
clients across invocations.

This module requires the ``http-tools`` extra.  When :mod:`httpx` is not
installed, importing the module raises :class:`ImportError` with an
install-hint message; ``nanitics.tools.__init__`` wraps that import in a
``try`` / ``except`` so ``import nanitics.tools`` still succeeds.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from nanitics.infrastructure.errors import (
    ToolExecutionError,
    ToolTimeoutError,
)
from nanitics.strategies.tools.function_tool import FunctionTool
from nanitics.strategies.tools.protocol import Tool, ToolResult
from nanitics.tools._result_models import WebSearchResult, WebSearchResultItem

try:
    import httpx
except ImportError as _err:  # pragma: no cover
    raise ImportError(
        "create_web_search_tool requires the 'http-tools' extra: pip install nanitics[http-tools]"
    ) from _err


_TAVILY_URL = "https://api.tavily.com/search"
_BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"

_SUPPORTED_PROVIDERS: tuple[str, ...] = ("tavily", "brave")

# Clamp the LLM-facing snippet preview to keep ``content`` compact; the full
# provider payload lives in ``metadata.raw_response`` for application code.
_SNIPPET_CLAMP_CHARS = 500
# Response-body preview used when the provider returns a non-2xx status so
# the ``ToolExecutionError.message`` stays readable in traces.
_ERROR_BODY_PREVIEW_CHARS = 200


class _WebSearchParams(BaseModel):
    """Parameters accepted by the ``web_search`` tool."""

    query: str = Field(min_length=1, max_length=500, description="Search query text.")
    max_results: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Optional per-call override of the factory's max_results_default.",
    )


def _clamp(text: str, limit: int) -> str:
    """Clamp *text* to *limit* characters."""
    if len(text) <= limit:
        return text
    return text[:limit]


def _render_content(items: list[WebSearchResultItem]) -> str:
    """Render normalized results as a compact markdown bullet list.

    The format is one bullet per result with bolded title, the snippet on
    the next line, and the URL on a third line.  This shape is what the
    LLM finds most useful when deciding whether to click through or refine
    the query.
    """
    if not items:
        return "No results."
    lines: list[str] = []
    for item in items:
        snippet = _clamp(item.snippet, _SNIPPET_CLAMP_CHARS)
        lines.append(f"- **{item.title}**")
        lines.append(f"  {snippet}")
        lines.append(f"  URL: {item.url}")
    return "\n".join(lines)


def _parse_tavily_response(payload: Any) -> list[WebSearchResultItem]:
    """Map the Tavily JSON response into normalized result items.

    Raises :class:`ValueError` if the shape does not match the documented
    contract; the factory wraps that into a :class:`ToolExecutionError`.
    """
    if not isinstance(payload, dict) or "results" not in payload:
        raise ValueError("missing 'results' field")
    results = payload["results"]
    if not isinstance(results, list):
        raise ValueError("'results' is not a list")
    items: list[WebSearchResultItem] = []
    for entry in results:
        if not isinstance(entry, dict):
            raise ValueError("result entry is not an object")
        items.append(
            WebSearchResultItem(
                title=str(entry.get("title", "")),
                url=str(entry.get("url", "")),
                snippet=str(entry.get("content", "")),
                score=float(entry["score"]) if entry.get("score") is not None else None,
            )
        )
    return items


def _parse_brave_response(payload: Any) -> list[WebSearchResultItem]:
    """Map the Brave JSON response into normalized result items."""
    if not isinstance(payload, dict):
        raise ValueError("response is not an object")
    web = payload.get("web")
    if not isinstance(web, dict) or "results" not in web:
        raise ValueError("missing 'web.results' field")
    results = web["results"]
    if not isinstance(results, list):
        raise ValueError("'web.results' is not a list")
    items: list[WebSearchResultItem] = []
    for entry in results:
        if not isinstance(entry, dict):
            raise ValueError("result entry is not an object")
        items.append(
            WebSearchResultItem(
                title=str(entry.get("title", "")),
                url=str(entry.get("url", "")),
                snippet=str(entry.get("description", "")),
                score=None,
            )
        )
    return items


async def _call_tavily(
    client: httpx.AsyncClient,
    *,
    api_key: str,
    query: str,
    max_results: int,
) -> tuple[list[WebSearchResultItem], dict[str, Any]]:
    response = await client.post(
        _TAVILY_URL,
        json={
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": False,
        },
        headers={"Accept": "application/json"},
    )
    if response.status_code >= 400:
        raise ToolExecutionError(
            f"tavily returned {response.status_code}: {_clamp(response.text, _ERROR_BODY_PREVIEW_CHARS)}",
            tool_name="web_search",
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise ToolExecutionError(
            f"Unexpected response from tavily: could not decode JSON ({exc})",
            tool_name="web_search",
        ) from exc
    try:
        items = _parse_tavily_response(payload)
    except ValueError as exc:
        raise ToolExecutionError(
            f"Unexpected response from tavily: {exc}",
            tool_name="web_search",
        ) from exc
    # Parse functions above already verified payload is a dict.
    return items, payload


async def _call_brave(
    client: httpx.AsyncClient,
    *,
    api_key: str,
    query: str,
    max_results: int,
) -> tuple[list[WebSearchResultItem], dict[str, Any]]:
    response = await client.get(
        _BRAVE_URL,
        params={"q": query, "count": str(max_results)},
        headers={
            "X-Subscription-Token": api_key,
            "Accept": "application/json",
        },
    )
    if response.status_code >= 400:
        raise ToolExecutionError(
            f"brave returned {response.status_code}: {_clamp(response.text, _ERROR_BODY_PREVIEW_CHARS)}",
            tool_name="web_search",
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise ToolExecutionError(
            f"Unexpected response from brave: could not decode JSON ({exc})",
            tool_name="web_search",
        ) from exc
    try:
        items = _parse_brave_response(payload)
    except ValueError as exc:
        raise ToolExecutionError(
            f"Unexpected response from brave: {exc}",
            tool_name="web_search",
        ) from exc
    # Parse functions above already verified payload is a dict.
    return items, payload


def create_web_search_tool(
    api_key: str,
    *,
    provider: Literal["tavily", "brave"] = "tavily",
    max_results_default: int = 5,
    request_timeout: float = 30.0,
    name: str = "web_search",
    description: str | None = None,
) -> Tool:
    """Create a web-search tool backed by Tavily or Brave.

    The returned object satisfies :class:`~nanitics.strategies.tools.protocol.Tool`
    and can be registered in :class:`~nanitics.strategies.ToolRegistry` alongside
    any other tool.  The tool emits :class:`~nanitics.events.ToolInvokeEvent`
    and :class:`~nanitics.events.ToolResultEvent` through the registry's
    standard dispatch path.

    The tool constructs a fresh :class:`httpx.AsyncClient` per call; callers
    do not need to manage client lifecycles.

    Args:
        api_key: Provider API key.  Must be non-empty.
        provider: One of ``"tavily"`` or ``"brave"``.
        max_results_default: Default number of results requested from the
            provider (1-20).  Overridable per call via the ``max_results``
            parameter.
        request_timeout: Per-call timeout in seconds.  When exceeded the
            tool raises :class:`ToolTimeoutError`.
        name: Tool name exposed to the LLM.  Defaults to ``"web_search"``.
        description: Optional override of the LLM-facing description.  The
            default includes the provider name so trace consumers can
            distinguish Tavily-backed tools from Brave-backed tools.

    Returns:
        A :class:`Tool`-conforming object.

    Raises:
        ValueError: If ``api_key`` is empty or ``provider`` is unknown.
    """
    if not api_key:
        raise ValueError("api_key must be a non-empty string")
    if provider not in _SUPPORTED_PROVIDERS:
        raise ValueError(f"provider must be one of {_SUPPORTED_PROVIDERS!r}, got {provider!r}")

    effective_description = description or (
        f"Search the web via {provider}. Returns a bulleted list of the top results with title, snippet, and URL."
    )

    async def _execute(query: str, max_results: int | None = None) -> ToolResult:
        effective_max = max_results if max_results is not None else max_results_default
        try:
            async with httpx.AsyncClient(timeout=request_timeout) as client:
                if provider == "tavily":
                    items, raw = await _call_tavily(
                        client,
                        api_key=api_key,
                        query=query,
                        max_results=effective_max,
                    )
                else:
                    items, raw = await _call_brave(
                        client,
                        api_key=api_key,
                        query=query,
                        max_results=effective_max,
                    )
        except httpx.TimeoutException as exc:
            raise ToolTimeoutError(
                f"web_search timed out after {request_timeout}s",
                tool_name="web_search",
                timeout_seconds=request_timeout,
            ) from exc
        except httpx.RequestError as exc:
            raise ToolExecutionError(
                f"Transport error calling {provider}: {exc}",
                tool_name="web_search",
            ) from exc

        metadata = WebSearchResult(
            provider=provider,
            query=query,
            results=items,
            raw_response=raw,
        ).model_dump()

        return ToolResult(content=_render_content(items), metadata=metadata)

    return FunctionTool(
        fn=_execute,
        name=name,
        description=effective_description,
        parameters_model=_WebSearchParams,
    )
