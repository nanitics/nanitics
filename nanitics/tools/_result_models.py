"""Internal result-metadata Pydantic models for the four reference tools.

These models describe the shape of ``ToolResult.metadata`` returned by each
reference tool.  They are re-exported at package level from
:mod:`nanitics.tools` so application code can introspect the structured data
with full type-checker support.  The LLM never sees these objects directly;
the user-facing prose is routed through ``ToolResult.content`` instead.

All models are ``frozen`` so that a ``ToolResult.metadata`` round-tripped
through ``.model_dump()`` cannot be mutated after the tool has reported its
outcome.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class WebSearchResultItem(BaseModel):
    """A single normalized search result across providers."""

    model_config = ConfigDict(frozen=True)

    title: str
    url: str
    snippet: str
    score: float | None = None


class WebSearchResult(BaseModel):
    """Metadata emitted by :func:`create_web_search_tool`."""

    model_config = ConfigDict(frozen=True)

    provider: Literal["tavily", "brave"]
    query: str
    results: list[WebSearchResultItem]
    raw_response: dict[str, Any]


class HttpResponse(BaseModel):
    """Metadata emitted by :func:`create_http_tool`.

    ``body`` is truncated to the factory's ``max_response_bytes``.  When
    truncation occurs, ``truncated`` is ``True`` and ``bytes_read`` records
    the size of the retained prefix.
    """

    model_config = ConfigDict(frozen=True)

    status: int
    headers: dict[str, str]
    body: str
    truncated: bool
    bytes_read: int
    url: str


class FileReadResult(BaseModel):
    """Metadata emitted by :func:`create_file_read_tool`.

    ``path`` is the resolved absolute path, ``size_bytes`` is the full
    on-disk size, and ``bytes_read`` is the number of bytes actually
    returned (bounded by the per-call ``max_bytes`` parameter).
    """

    model_config = ConfigDict(frozen=True)

    path: str
    size_bytes: int
    bytes_read: int
    truncated: bool
    encoding: Literal["utf-8", "base64"]


class CodeExecutionResult(BaseModel):
    """Metadata emitted by :func:`create_code_execution_tool`.

    Mirrors the fields of the underlying
    :class:`~nanitics.safety.ExecutionResult` returned by the sandbox.
    """

    model_config = ConfigDict(frozen=True)

    success: bool
    stdout: str
    stderr: str
    return_value: str | None
    error: str | None
    duration_ms: float
