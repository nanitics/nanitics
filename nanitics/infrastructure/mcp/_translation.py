"""Pure translation helpers between MCP upstream types and Nanitics types.

This module is deliberately kept free of transport, session, and I/O
dependencies so it can be unit-tested in isolation.  ``MCPClient`` and
``MCPTool`` depend on these helpers; the helpers do not depend on them.

Internal — no public re-exports.  ``_`` prefix signals this to IDEs and
library users.
"""

from __future__ import annotations

from typing import Any

from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
    Tool,
)

from nanitics.core.tools.protocol import ToolResult
from nanitics.infrastructure.errors import ToolExecutionError
from nanitics.infrastructure.llm.protocol import ToolSchema

_DEFAULT_EMPTY_PARAMETERS: dict[str, Any] = {"type": "object", "properties": {}}
_ERROR_WITH_NO_MESSAGE = "MCP server reported tool error with no message"


def mcp_tool_to_schema(
    mcp_tool: Tool,
    *,
    name_prefix: str = "",
    description_prefix: str | None = None,
) -> ToolSchema:
    """Translate an upstream MCP ``Tool`` description into a Nanitics ``ToolSchema``.

    Args:
        mcp_tool: The server-provided tool definition.
        name_prefix: Optional string prepended to ``mcp_tool.name`` — used by
            ``MCPClient`` to namespace tools from a specific server (e.g.
            ``"fs_"``).
        description_prefix: Optional string prepended to the tool description
            with a single space separator.  Used to mark the tool as MCP-sourced
            (``"[MCP]"``) for trace observability, since ``ToolInfo`` has no
            dedicated source field.

    Returns:
        A frozen ``ToolSchema`` with the translated fields.  The JSON Schema in
        ``inputSchema`` is forwarded verbatim; if the server provides no
        schema or an empty one, a minimal ``{"type": "object", "properties": {}}``
        is substituted so downstream LLM tool-calling does not choke.
    """

    name = f"{name_prefix}{mcp_tool.name}"
    raw_description = mcp_tool.description or ""
    if description_prefix is not None:
        # Use a single space when the body is non-empty; strip trailing space
        # when the body is empty so the description never ends in whitespace.
        description = f"{description_prefix} {raw_description}".rstrip()
    else:
        description = raw_description

    parameters = mcp_tool.inputSchema if mcp_tool.inputSchema else dict(_DEFAULT_EMPTY_PARAMETERS)

    return ToolSchema(
        name=name,
        description=description,
        parameters=parameters,
    )


def call_result_to_tool_result(result: CallToolResult, *, tool_name: str) -> ToolResult:
    """Translate an MCP ``CallToolResult`` into a Nanitics ``ToolResult``.

    On ``result.isError is True``, raises ``ToolExecutionError`` rather than
    returning — this routes the failure through the standard tool error
    classifier (CORRECTABLE), matching how ``FunctionTool`` raises on failure.

    Text content blocks are concatenated into ``ToolResult.content`` with
    ``"\\n\\n"`` separators — that is the text the LLM sees.  Every content
    block (including images and embedded resources) is also preserved as a
    JSON-serialisable dict in ``metadata["raw_content"]`` so application code
    can access non-text payloads without bypassing the Tool abstraction.

    Args:
        result: The MCP server's response to ``tools/call``.
        tool_name: The prefixed Nanitics-side tool name; used as the
            ``tool_name`` field on any raised ``ToolExecutionError``.

    Raises:
        ToolExecutionError: If ``result.isError`` is ``True`` or if a content
            block has an unrecognised type (the ``ValueError`` from the block
            serialiser is re-raised as ``ToolExecutionError`` with it as
            ``__cause__``).
    """

    if result.isError:
        error_text = "\n\n".join(b.text for b in result.content if isinstance(b, TextContent))
        if not error_text:
            error_text = _ERROR_WITH_NO_MESSAGE
        raise ToolExecutionError(error_text, tool_name=tool_name)

    content_str = "\n\n".join(b.text for b in result.content if isinstance(b, TextContent))

    try:
        raw_content = [_content_block_to_dict(b) for b in result.content]
    except ValueError as exc:
        raise ToolExecutionError(str(exc), tool_name=tool_name) from exc

    metadata: dict[str, Any] = {
        "raw_content": raw_content,
        "is_error": False,
    }
    return ToolResult(content=content_str, metadata=metadata)


def _content_block_to_dict(block: object) -> dict[str, Any]:
    """Serialise a single MCP content block to a JSON-safe dict.

    Handles the three block types the MCP surface currently supports
    (``TextContent``, ``ImageContent``, ``EmbeddedResource``). Any
    other type raises ``ValueError`` — the caller converts that to a
    ``ToolExecutionError`` so the failure surfaces rather than silently
    dropping the block.
    """

    if isinstance(block, TextContent):
        return {"type": "text", "text": block.text}
    if isinstance(block, ImageContent):
        return {
            "type": "image",
            "data": block.data,
            "mime_type": block.mimeType,
        }
    if isinstance(block, EmbeddedResource):
        return {
            "type": "resource",
            "resource": _resource_to_dict(block.resource),
        }
    raise ValueError(f"Unknown MCP content block type: {type(block).__name__}")


def _resource_to_dict(resource: Any) -> dict[str, Any]:
    """Serialise a ``TextResourceContents`` or ``BlobResourceContents`` to a dict.

    The upstream types are pydantic models; ``model_dump(mode="json")`` produces
    a JSON-safe representation (``AnyUrl`` becomes ``str``, etc.).  Unknown
    keys are preserved so future upstream additions don't silently drop data.
    """

    dumped = resource.model_dump(mode="json", exclude_none=True)
    # Drop the ``meta`` field when present and empty so tests/downstream
    # consumers see a tight dict without upstream implementation noise.
    return {k: v for k, v in dumped.items() if k != "meta" or v}
