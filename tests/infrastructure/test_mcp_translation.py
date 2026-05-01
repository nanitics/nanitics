"""Unit tests for MCP schema and result translation helpers.

These tests exercise pure synchronous functions with no transport or session
dependencies.  They verify every branch of the translation logic so that
higher-level tests (``test_mcp_client.py``) can assume translation is
correct.
"""

from __future__ import annotations

import pytest
from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
    Tool,
)
from pydantic import AnyUrl

from nanitics.infrastructure.errors import ToolExecutionError
from nanitics.infrastructure.mcp._translation import (
    _content_block_to_dict,
    call_result_to_tool_result,
    mcp_tool_to_schema,
)

# ---------------------------------------------------------------------------
# mcp_tool_to_schema
# ---------------------------------------------------------------------------


class TestMcpToolToSchema:
    def test_normal_tool_maps_name_description_and_schema(self) -> None:
        mcp_tool = Tool(
            name="get_weather",
            description="Look up weather for a city.",
            inputSchema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )

        schema = mcp_tool_to_schema(mcp_tool)

        assert schema.name == "get_weather"
        assert schema.description == "Look up weather for a city."
        assert schema.parameters == {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }
        assert schema.requires_approval is False
        assert schema.timeout_seconds is None

    def test_empty_description_yields_empty_string(self) -> None:
        mcp_tool = Tool(name="noop", description=None, inputSchema={"type": "object", "properties": {}})

        schema = mcp_tool_to_schema(mcp_tool)

        assert schema.description == ""

    def test_missing_input_schema_defaults_to_empty_object(self) -> None:
        mcp_tool = Tool(name="ping", description="pong", inputSchema={"type": "object"})
        # Force inputSchema to the empty-ish default the helper substitutes
        # when upstream emits a bare schema — mcp.types.Tool requires
        # inputSchema to be a dict, so the translation helper's job is to
        # make sure a minimal object schema is produced regardless.
        mcp_tool_stripped = mcp_tool.model_copy(update={"inputSchema": {}})

        schema = mcp_tool_to_schema(mcp_tool_stripped)

        assert schema.parameters == {"type": "object", "properties": {}}

    def test_name_prefix_prepended(self) -> None:
        mcp_tool = Tool(name="read_file", description="read", inputSchema={"type": "object", "properties": {}})

        schema = mcp_tool_to_schema(mcp_tool, name_prefix="fs_")

        assert schema.name == "fs_read_file"

    def test_description_prefix_prepended_with_space(self) -> None:
        mcp_tool = Tool(
            name="search",
            description="Search the web.",
            inputSchema={"type": "object", "properties": {}},
        )

        schema = mcp_tool_to_schema(mcp_tool, description_prefix="[MCP]")

        assert schema.description == "[MCP] Search the web."

    def test_description_prefix_none_omits_prefix(self) -> None:
        mcp_tool = Tool(
            name="search",
            description="Search the web.",
            inputSchema={"type": "object", "properties": {}},
        )

        schema = mcp_tool_to_schema(mcp_tool, description_prefix=None)

        assert schema.description == "Search the web."

    def test_description_prefix_with_empty_description_is_still_clean(self) -> None:
        mcp_tool = Tool(
            name="nop",
            description=None,
            inputSchema={"type": "object", "properties": {}},
        )

        schema = mcp_tool_to_schema(mcp_tool, description_prefix="[MCP]")

        # Prefix + single space + empty body; rstrip the trailing whitespace so
        # the description never ends in a bare space.
        assert schema.description == "[MCP]"


# ---------------------------------------------------------------------------
# call_result_to_tool_result — success paths
# ---------------------------------------------------------------------------


class TestCallResultToToolResultSuccess:
    def test_text_only_success(self) -> None:
        result = CallToolResult(
            content=[TextContent(type="text", text="It's sunny.")],
            isError=False,
        )

        tool_result = call_result_to_tool_result(result, tool_name="get_weather")

        assert tool_result.content == "It's sunny."
        assert tool_result.metadata["is_error"] is False
        assert tool_result.metadata["raw_content"] == [{"type": "text", "text": "It's sunny."}]

    def test_multiple_text_blocks_joined_with_double_newline(self) -> None:
        result = CallToolResult(
            content=[
                TextContent(type="text", text="Line 1"),
                TextContent(type="text", text="Line 2"),
            ],
            isError=False,
        )

        tool_result = call_result_to_tool_result(result, tool_name="multi")

        assert tool_result.content == "Line 1\n\nLine 2"
        assert tool_result.metadata["raw_content"] == [
            {"type": "text", "text": "Line 1"},
            {"type": "text", "text": "Line 2"},
        ]

    def test_image_content_preserved_in_metadata(self) -> None:
        result = CallToolResult(
            content=[
                TextContent(type="text", text="See image."),
                ImageContent(type="image", data="aGVsbG8=", mimeType="image/png"),
            ],
            isError=False,
        )

        tool_result = call_result_to_tool_result(result, tool_name="render")

        assert tool_result.content == "See image."
        assert tool_result.metadata["raw_content"] == [
            {"type": "text", "text": "See image."},
            {"type": "image", "data": "aGVsbG8=", "mime_type": "image/png"},
        ]

    def test_embedded_text_resource_preserved_in_metadata(self) -> None:
        resource = TextResourceContents(
            uri=AnyUrl("file:///tmp/notes.txt"),
            mimeType="text/plain",
            text="hello world",
        )
        result = CallToolResult(
            content=[
                TextContent(type="text", text="File contents follow."),
                EmbeddedResource(type="resource", resource=resource),
            ],
            isError=False,
        )

        tool_result = call_result_to_tool_result(result, tool_name="read_file")

        assert tool_result.content == "File contents follow."
        raw = tool_result.metadata["raw_content"]
        assert raw[0] == {"type": "text", "text": "File contents follow."}
        assert raw[1]["type"] == "resource"
        assert raw[1]["resource"]["uri"] == "file:///tmp/notes.txt"
        assert raw[1]["resource"]["mimeType"] == "text/plain"
        assert raw[1]["resource"]["text"] == "hello world"

    def test_embedded_blob_resource_preserved_in_metadata(self) -> None:
        resource = BlobResourceContents(
            uri=AnyUrl("file:///tmp/binary.bin"),
            mimeType="application/octet-stream",
            blob="YWJjMTIz",
        )
        result = CallToolResult(
            content=[EmbeddedResource(type="resource", resource=resource)],
            isError=False,
        )

        tool_result = call_result_to_tool_result(result, tool_name="read_binary")

        assert tool_result.content == ""
        raw = tool_result.metadata["raw_content"]
        assert raw[0]["type"] == "resource"
        assert raw[0]["resource"]["uri"] == "file:///tmp/binary.bin"
        assert raw[0]["resource"]["mimeType"] == "application/octet-stream"
        assert raw[0]["resource"]["blob"] == "YWJjMTIz"

    def test_empty_content_yields_empty_string(self) -> None:
        result = CallToolResult(content=[], isError=False)

        tool_result = call_result_to_tool_result(result, tool_name="noop")

        assert tool_result.content == ""
        assert tool_result.metadata["raw_content"] == []
        assert tool_result.metadata["is_error"] is False


# ---------------------------------------------------------------------------
# call_result_to_tool_result — error paths
# ---------------------------------------------------------------------------


class TestCallResultToToolResultError:
    def test_error_result_with_text_raises_tool_execution_error(self) -> None:
        result = CallToolResult(
            content=[TextContent(type="text", text="City not found.")],
            isError=True,
        )

        with pytest.raises(ToolExecutionError) as excinfo:
            call_result_to_tool_result(result, tool_name="get_weather")

        assert excinfo.value.tool_name == "get_weather"
        assert str(excinfo.value) == "City not found."

    def test_error_result_with_multiple_text_blocks_joined(self) -> None:
        result = CallToolResult(
            content=[
                TextContent(type="text", text="Failure"),
                TextContent(type="text", text="Reason: downstream offline."),
            ],
            isError=True,
        )

        with pytest.raises(ToolExecutionError) as excinfo:
            call_result_to_tool_result(result, tool_name="search")

        assert str(excinfo.value) == "Failure\n\nReason: downstream offline."

    def test_error_result_with_no_text_uses_default_message(self) -> None:
        result = CallToolResult(
            content=[ImageContent(type="image", data="AA==", mimeType="image/png")],
            isError=True,
        )

        with pytest.raises(ToolExecutionError) as excinfo:
            call_result_to_tool_result(result, tool_name="render")

        assert str(excinfo.value) == "MCP server reported tool error with no message"

    def test_error_result_with_empty_content_uses_default_message(self) -> None:
        result = CallToolResult(content=[], isError=True)

        with pytest.raises(ToolExecutionError) as excinfo:
            call_result_to_tool_result(result, tool_name="x")

        assert str(excinfo.value) == "MCP server reported tool error with no message"


# ---------------------------------------------------------------------------
# _content_block_to_dict — direct tests for unknown-type branch
# ---------------------------------------------------------------------------


class TestContentBlockToDict:
    def test_text_block(self) -> None:
        block = TextContent(type="text", text="hi")
        assert _content_block_to_dict(block) == {"type": "text", "text": "hi"}

    def test_image_block(self) -> None:
        block = ImageContent(type="image", data="AA==", mimeType="image/jpeg")
        assert _content_block_to_dict(block) == {
            "type": "image",
            "data": "AA==",
            "mime_type": "image/jpeg",
        }

    def test_unknown_block_type_raises_value_error(self) -> None:
        class _FakeBlock:
            """Stand-in for a future content-block type we don't know about."""

        with pytest.raises(ValueError, match="Unknown MCP content block type"):
            _content_block_to_dict(_FakeBlock())  # type: ignore[arg-type]

    def test_unknown_block_type_surfaces_through_call_result(self) -> None:
        class _FakeBlock:
            pass

        # CallToolResult.content is validated by pydantic, so we can't push a
        # fake block through the normal constructor.  Construct the result
        # with valid content, then monkey-patch the content list to include
        # the unknown block — this proves the helper catches and re-raises
        # correctly.
        result = CallToolResult(content=[TextContent(type="text", text="x")], isError=False)
        object.__setattr__(result, "content", [_FakeBlock()])

        with pytest.raises(ToolExecutionError) as excinfo:
            call_result_to_tool_result(result, tool_name="odd")

        assert "Unknown MCP content block type" in str(excinfo.value)
        assert excinfo.value.tool_name == "odd"
        assert isinstance(excinfo.value.__cause__, ValueError)
