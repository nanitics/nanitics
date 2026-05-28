from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import pytest
from pydantic import BaseModel

from nanitics.errors import (
    LLMAuthenticationError,
    LLMContextLengthError,
    LLMOverloadedError,
    LLMProviderError,
    LLMQuotaExhaustedError,
    LLMRateLimitError,
    LLMSchemaViolationError,
)
from nanitics.infrastructure import (
    LLMClient,
    ToolSchema,
)
from nanitics.infrastructure.llm.anthropic import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    AnthropicLLMClient,
    _extract_anthropic_error_message,
    _extract_anthropic_error_type,
    _from_anthropic_response,
    _to_anthropic_messages,
    _to_anthropic_tools,
)
from nanitics.tracing import (
    ImageContentBlock,
    Message,
    TextContentBlock,
    ToolCall,
)

# --- Helpers ---


def _make_text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_use_block(id: str, name: str, input: dict[str, Any]) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.id = id
    block.name = name
    block.input = input
    return block


def _make_thinking_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "thinking"
    block.thinking = text
    return block


def _make_anthropic_response(
    content_blocks: list[MagicMock],
    *,
    stop_reason: str = "end_turn",
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> MagicMock:
    response = MagicMock(spec=anthropic.types.Message)
    response.content = content_blocks
    response.stop_reason = stop_reason
    response.usage = MagicMock()
    response.usage.input_tokens = input_tokens
    response.usage.output_tokens = output_tokens
    return response


def _mock_stream_ctx(*, response: MagicMock | None = None, error: Exception | None = None) -> AsyncMock:
    """Create mock for ``messages.stream()`` async context manager."""
    stream_obj = AsyncMock()
    if response is not None:
        stream_obj.get_final_message = AsyncMock(return_value=response)
    cm = AsyncMock()
    if error is not None:
        cm.__aenter__ = AsyncMock(side_effect=error)
    else:
        cm.__aenter__ = AsyncMock(return_value=stream_obj)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


# --- Message Conversion Tests ---


class TestToAnthropicMessages:
    def test_user_message(self) -> None:
        msgs = [Message(role="user", content="Hello")]
        result = _to_anthropic_messages(msgs)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_user_message_with_name(self) -> None:
        msgs = [Message(role="user", content="Hello", name="alice")]
        result = _to_anthropic_messages(msgs)
        assert result == [{"role": "user", "content": "Hello", "name": "alice"}]

    def test_assistant_text_message(self) -> None:
        msgs = [Message(role="assistant", content="Hi there")]
        result = _to_anthropic_messages(msgs)
        assert result == [{"role": "assistant", "content": [{"type": "text", "text": "Hi there"}]}]

    def test_assistant_message_with_name(self) -> None:
        msgs = [Message(role="assistant", content="Hi", name="bot")]
        result = _to_anthropic_messages(msgs)
        assert result[0]["name"] == "bot"

    def test_assistant_with_tool_calls(self) -> None:
        tc = ToolCall(id="tc-1", name="search", arguments={"q": "test"})
        msgs = [Message(role="assistant", tool_calls=[tc])]
        result = _to_anthropic_messages(msgs)
        assert result == [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tc-1",
                        "name": "search",
                        "input": {"q": "test"},
                    }
                ],
            }
        ]

    def test_assistant_with_text_and_tool_calls(self) -> None:
        tc = ToolCall(id="tc-1", name="search", arguments={"q": "test"})
        msgs = [Message(role="assistant", content="Let me search", tool_calls=[tc])]
        result = _to_anthropic_messages(msgs)
        assert len(result) == 1
        blocks = result[0]["content"]
        assert blocks[0] == {"type": "text", "text": "Let me search"}
        assert blocks[1]["type"] == "tool_use"

    def test_tool_result_message(self) -> None:
        msgs = [Message(role="tool_result", content="result data", tool_call_id="tc-1")]
        result = _to_anthropic_messages(msgs)
        assert result == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tc-1",
                        "content": "result data",
                    }
                ],
            }
        ]

    def test_consecutive_tool_results_grouped(self) -> None:
        msgs = [
            Message(role="tool_result", content="result 1", tool_call_id="tc-1"),
            Message(role="tool_result", content="result 2", tool_call_id="tc-2"),
        ]
        result = _to_anthropic_messages(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["tool_use_id"] == "tc-1"
        assert result[0]["content"][1]["tool_use_id"] == "tc-2"

    def test_tool_result_after_user_not_grouped(self) -> None:
        msgs = [
            Message(role="user", content="Hello"),
            Message(role="tool_result", content="result", tool_call_id="tc-1"),
        ]
        result = _to_anthropic_messages(msgs)
        assert len(result) == 2

    def test_empty_content(self) -> None:
        msgs = [Message(role="user", content=None)]
        result = _to_anthropic_messages(msgs)
        assert result == [{"role": "user", "content": ""}]

    def test_empty_assistant_content_becomes_non_empty(self) -> None:
        """Empty assistant content is replaced with a space to avoid Anthropic rejection."""
        msgs = [Message(role="assistant", content="")]
        result = _to_anthropic_messages(msgs)
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == " "

    def test_none_assistant_content_becomes_non_empty(self) -> None:
        """None assistant content is replaced with a space to avoid Anthropic rejection."""
        msgs = [Message(role="assistant", content=None)]
        result = _to_anthropic_messages(msgs)
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == " "

    def test_full_conversation(self) -> None:
        msgs = [
            Message(role="user", content="What is 2+2?"),
            Message(
                role="assistant",
                content="Let me calculate",
                tool_calls=[ToolCall(id="tc-1", name="calc", arguments={"expr": "2+2"})],
            ),
            Message(role="tool_result", content="4", tool_call_id="tc-1"),
            Message(role="assistant", content="The answer is 4"),
        ]
        result = _to_anthropic_messages(msgs)
        assert len(result) == 4
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "user"  # tool_result → user role
        assert result[3]["role"] == "assistant"


# --- Message Caching Tests ---


class TestToAnthropicMessagesCaching:
    def test_messages_last_user_gets_cache_control_when_enabled(self) -> None:
        msgs = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
            Message(role="user", content="How are you?"),
        ]
        result = _to_anthropic_messages(msgs, enable_caching=True)
        # Last user message should have cache_control
        assert result[2]["content"] == [
            {"type": "text", "text": "How are you?", "cache_control": {"type": "ephemeral"}}
        ]
        # First user message should NOT have cache_control
        assert result[0]["content"] == "Hello"

    def test_messages_tool_result_gets_cache_control_when_enabled(self) -> None:
        msgs = [
            Message(role="user", content="Use tool"),
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="tc-1", name="search", arguments={"q": "test"})],
            ),
            Message(role="tool_result", content="result 1", tool_call_id="tc-1"),
            Message(role="tool_result", content="result 2", tool_call_id="tc-2"),
        ]
        result = _to_anthropic_messages(msgs, enable_caching=True)
        # The grouped tool_result user message is the last user message
        last_user = result[2]
        assert last_user["role"] == "user"
        assert len(last_user["content"]) == 2
        # cache_control on last block only
        assert "cache_control" not in last_user["content"][0]
        assert last_user["content"][1]["cache_control"] == {"type": "ephemeral"}

    def test_messages_no_cache_control_when_disabled(self) -> None:
        msgs = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
            Message(role="user", content="How are you?"),
        ]
        result = _to_anthropic_messages(msgs, enable_caching=False)
        # No cache_control anywhere
        assert result[0]["content"] == "Hello"
        assert result[2]["content"] == "How are you?"

    def test_messages_empty_list_no_error(self) -> None:
        result = _to_anthropic_messages([], enable_caching=True)
        assert result == []


# --- Vision Content Block Tests ---


class TestToAnthropicMessagesVision:
    def test_user_message_with_text_content_blocks(self) -> None:
        msgs = [
            Message(
                role="user",
                content=[TextContentBlock(text="Describe this image")],
            )
        ]
        result = _to_anthropic_messages(msgs)
        assert result == [{"role": "user", "content": [{"type": "text", "text": "Describe this image"}]}]

    def test_user_message_with_image_base64(self) -> None:
        msgs = [
            Message(
                role="user",
                content=[
                    ImageContentBlock(media_type="image/png", data="iVBORw0KGgo="),
                ],
            )
        ]
        result = _to_anthropic_messages(msgs)
        assert result == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "iVBORw0KGgo=",
                        },
                    }
                ],
            }
        ]

    def test_user_message_with_image_url(self) -> None:
        msgs = [
            Message(
                role="user",
                content=[
                    ImageContentBlock(media_type="image/jpeg", data="https://example.com/img.jpg"),
                ],
            )
        ]
        result = _to_anthropic_messages(msgs)
        assert result == [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "url", "url": "https://example.com/img.jpg"}},
                ],
            }
        ]

    def test_user_message_with_mixed_content(self) -> None:
        msgs = [
            Message(
                role="user",
                content=[
                    TextContentBlock(text="What's in this image?"),
                    ImageContentBlock(media_type="image/png", data="iVBORw0KGgo="),
                    TextContentBlock(text="Be specific."),
                ],
            )
        ]
        result = _to_anthropic_messages(msgs)
        assert len(result) == 1
        blocks = result[0]["content"]
        assert len(blocks) == 3
        assert blocks[0] == {"type": "text", "text": "What's in this image?"}
        assert blocks[1]["type"] == "image"
        assert blocks[1]["source"]["type"] == "base64"
        assert blocks[2] == {"type": "text", "text": "Be specific."}

    def test_content_blocks_with_caching(self) -> None:
        msgs = [
            Message(
                role="user",
                content=[
                    TextContentBlock(text="Describe this"),
                    ImageContentBlock(media_type="image/png", data="iVBORw0KGgo="),
                ],
            )
        ]
        result = _to_anthropic_messages(msgs, enable_caching=True)
        blocks = result[0]["content"]
        # cache_control on last block
        assert "cache_control" not in blocks[0]
        assert blocks[1]["cache_control"] == {"type": "ephemeral"}


# --- Tool Schema Conversion Tests ---


class TestToAnthropicTools:
    def test_basic_conversion(self) -> None:
        tools = [
            ToolSchema(
                name="search",
                description="Search the web",
                parameters={
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            )
        ]
        result = _to_anthropic_tools(tools)
        assert result == [
            {
                "name": "search",
                "description": "Search the web",
                "input_schema": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            }
        ]

    def test_multiple_tools(self) -> None:
        tools = [
            ToolSchema(name="a", description="Tool A", parameters={}),
            ToolSchema(name="b", description="Tool B", parameters={}),
        ]
        result = _to_anthropic_tools(tools)
        assert len(result) == 2
        assert result[0]["name"] == "a"
        assert result[1]["name"] == "b"


# --- Response Conversion Tests ---


class TestFromAnthropicResponse:
    def test_text_response(self) -> None:
        response = _make_anthropic_response([_make_text_block("Hello")])
        result = _from_anthropic_response(response, "claude-test")
        assert result.content == "Hello"
        assert result.tool_calls == []
        assert result.model == "claude-test"
        assert result.stop_reason == "end_turn"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 20
        assert result.usage.total_tokens == 30

    def test_tool_use_response(self) -> None:
        block = _make_tool_use_block("tc-1", "search", {"q": "test"})
        response = _make_anthropic_response([block], stop_reason="tool_use")
        result = _from_anthropic_response(response, "claude-test")
        assert result.content is None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "tc-1"
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].arguments == {"q": "test"}
        assert result.stop_reason == "tool_use"

    def test_mixed_content(self) -> None:
        blocks = [
            _make_text_block("Let me search"),
            _make_tool_use_block("tc-1", "search", {"q": "test"}),
        ]
        response = _make_anthropic_response(blocks, stop_reason="tool_use")
        result = _from_anthropic_response(response, "claude-test")
        assert result.content == "Let me search"
        assert len(result.tool_calls) == 1

    def test_empty_content(self) -> None:
        response = _make_anthropic_response([])
        result = _from_anthropic_response(response, "claude-test")
        assert result.content is None
        assert result.tool_calls == []

    def test_usage_tracking(self) -> None:
        response = _make_anthropic_response(
            [_make_text_block("Hi")],
            input_tokens=100,
            output_tokens=50,
        )
        result = _from_anthropic_response(response, "claude-test")
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50
        assert result.usage.total_tokens == 150

    def test_cache_token_extraction(self) -> None:
        response = _make_anthropic_response(
            [_make_text_block("Hi")],
            input_tokens=100,
            output_tokens=50,
        )
        response.usage.cache_creation_input_tokens = 500
        response.usage.cache_read_input_tokens = 3000
        result = _from_anthropic_response(response, "claude-test")
        assert result.usage.cache_creation_input_tokens == 500
        assert result.usage.cache_read_input_tokens == 3000

    def test_cache_tokens_none_when_absent(self) -> None:
        response = _make_anthropic_response(
            [_make_text_block("Hi")],
            input_tokens=100,
            output_tokens=50,
        )
        # MagicMock returns MagicMock for missing attrs by default,
        # so delete them to trigger getattr fallback
        del response.usage.cache_creation_input_tokens
        del response.usage.cache_read_input_tokens
        result = _from_anthropic_response(response, "claude-test")
        assert result.usage.cache_creation_input_tokens is None
        assert result.usage.cache_read_input_tokens is None


# --- Reasoning Extraction Tests ---


class TestReasoningExtraction:
    def test_thinking_only_then_final_text(self) -> None:
        """Thinking block + final text: ``reasoning_text`` is the thinking; ``content`` is the text."""
        blocks = [
            _make_thinking_block("Let me think about this."),
            _make_text_block("The answer is 42."),
        ]
        response = _make_anthropic_response(blocks, stop_reason="end_turn")
        result = _from_anthropic_response(response, "claude-test")
        assert result.content == "The answer is 42."
        assert result.reasoning_text == "Let me think about this."

    def test_text_before_tool_use(self) -> None:
        """Text block before ``tool_use``: text becomes ``reasoning_text``."""
        blocks = [
            _make_text_block("Let me search for that."),
            _make_tool_use_block("tc-1", "search", {"q": "test"}),
        ]
        response = _make_anthropic_response(blocks, stop_reason="tool_use")
        result = _from_anthropic_response(response, "claude-test")
        assert result.content == "Let me search for that."
        assert result.reasoning_text == "Let me search for that."
        assert len(result.tool_calls) == 1

    def test_thinking_and_text_before_tool_use(self) -> None:
        """Thinking + text-before-tool-use: both concatenated in ``reasoning_text``."""
        blocks = [
            _make_thinking_block("I should look this up."),
            _make_text_block("Let me search."),
            _make_tool_use_block("tc-1", "search", {"q": "test"}),
        ]
        response = _make_anthropic_response(blocks, stop_reason="tool_use")
        result = _from_anthropic_response(response, "claude-test")
        assert result.content == "Let me search."
        assert result.reasoning_text == "I should look this up.\n\nLet me search."
        assert len(result.tool_calls) == 1

    def test_plain_text_final_answer_has_no_reasoning(self) -> None:
        """Plain text final answer (no thinking, no tool use): ``reasoning_text is None``."""
        blocks = [_make_text_block("The answer is 42.")]
        response = _make_anthropic_response(blocks, stop_reason="end_turn")
        result = _from_anthropic_response(response, "claude-test")
        assert result.content == "The answer is 42."
        assert result.reasoning_text is None

    def test_structured_output_with_thinking(self) -> None:
        """Thinking block + structured-output tool_use: ``reasoning_text`` holds the thinking only.

        The tool_use arguments become structured output downstream; the
        extractor here only sees content blocks. Text blocks that precede
        the structured-output tool_use are also reasoning.
        """
        blocks = [
            _make_thinking_block("I need to enumerate the fields."),
            _make_tool_use_block("tc-1", "structured_output", {"field": "value"}),
        ]
        response = _make_anthropic_response(blocks, stop_reason="tool_use")
        result = _from_anthropic_response(response, "claude-test")
        assert result.reasoning_text == "I need to enumerate the fields."

    def test_tool_use_only_no_reasoning(self) -> None:
        """Tool use with no preceding thinking or text: ``reasoning_text is None``."""
        blocks = [_make_tool_use_block("tc-1", "search", {"q": "test"})]
        response = _make_anthropic_response(blocks, stop_reason="tool_use")
        result = _from_anthropic_response(response, "claude-test")
        assert result.content is None
        assert result.reasoning_text is None

    def test_empty_content_blocks_no_reasoning(self) -> None:
        """Empty content-block list: ``reasoning_text is None``."""
        response = _make_anthropic_response([], stop_reason="end_turn")
        result = _from_anthropic_response(response, "claude-test")
        assert result.reasoning_text is None

    def test_text_after_tool_use_not_reasoning(self) -> None:
        """Text blocks that *follow* a ``tool_use`` (no later tool_use) are final answer,
        not reasoning. Only text blocks that precede a tool_use count.
        """
        blocks = [
            _make_tool_use_block("tc-1", "search", {"q": "x"}),
            _make_text_block("Here is what I found."),
        ]
        response = _make_anthropic_response(blocks, stop_reason="tool_use")
        result = _from_anthropic_response(response, "claude-test")
        # Text that comes after the tool_use is not reasoning — no later
        # tool_use follows it.
        assert result.reasoning_text is None


# --- AnthropicLLMClient Tests ---


class TestAnthropicLLMClient:
    def _make_client(self, **kwargs: Any) -> AnthropicLLMClient:
        kwargs.setdefault("model", "claude-test")
        kwargs.setdefault("api_key", "test-key")
        return AnthropicLLMClient(**kwargs)

    async def test_protocol_conformance(self) -> None:
        client = self._make_client()
        assert isinstance(client, LLMClient)

    def test_model_property(self) -> None:
        client = self._make_client(model="claude-3-opus")
        assert client.model == "claude-3-opus"

    def test_disables_anthropic_builtin_retries(self) -> None:
        client = self._make_client()
        assert client._client.max_retries == 0

    def test_preflight_raises_when_no_key_and_no_env(self) -> None:
        with patch.dict("os.environ", {}, clear=True), pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            AnthropicLLMClient(model="claude-test")

    def test_preflight_passes_when_api_key_provided(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            client = AnthropicLLMClient(model="claude-test", api_key="explicit")
        assert client.model == "claude-test"

    def test_preflight_passes_when_env_set(self) -> None:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "from-env"}, clear=True):
            client = AnthropicLLMClient(model="claude-test")
        assert client.model == "claude-test"

    async def test_successful_text_response(self) -> None:
        client = self._make_client()
        mock_response = _make_anthropic_response([_make_text_block("Hello!")])

        with patch.object(
            client._client.messages, "stream", return_value=_mock_stream_ctx(response=mock_response)
        ) as mock_stream:
            result = await client.generate(
                system_prompt="You are helpful",
                messages=[Message(role="user", content="Hi")],
            )

        assert result.content == "Hello!"
        assert result.model == "claude-test"
        assert result.stop_reason == "end_turn"
        mock_stream.assert_called_once()
        call_kwargs = mock_stream.call_args.kwargs
        assert call_kwargs["system"] == "You are helpful"
        assert call_kwargs["model"] == "claude-test"

    async def test_on_token_callback_receives_text_stream(self) -> None:
        client = self._make_client()
        mock_response = _make_anthropic_response([_make_text_block("Hello!")])

        async def _text_stream():
            yield "Hel"
            yield "lo!"

        stream_obj = AsyncMock()
        stream_obj.get_final_message = AsyncMock(return_value=mock_response)
        stream_obj.text_stream = _text_stream()

        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=stream_obj)
        cm.__aexit__ = AsyncMock(return_value=False)

        tokens: list[str] = []
        with patch.object(client._client.messages, "stream", return_value=cm):
            await client.generate(
                system_prompt="You are helpful",
                messages=[Message(role="user", content="Hi")],
                on_token=tokens.append,
            )
        assert tokens == ["Hel", "lo!"]

    async def test_successful_tool_use_response(self) -> None:
        client = self._make_client()
        block = _make_tool_use_block("tc-1", "search", {"q": "test"})
        mock_response = _make_anthropic_response([block], stop_reason="tool_use")

        tools = [
            ToolSchema(
                name="search",
                description="Search",
                parameters={"type": "object", "properties": {"q": {"type": "string"}}},
            )
        ]

        with patch.object(
            client._client.messages, "stream", return_value=_mock_stream_ctx(response=mock_response)
        ) as mock_stream:
            result = await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="search for test")],
                tools=tools,
            )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"
        call_kwargs = mock_stream.call_args.kwargs
        assert "tools" in call_kwargs

    async def test_mutually_exclusive_tools_and_schema(self) -> None:
        client = self._make_client()

        class MyOutput(BaseModel):
            answer: str

        with pytest.raises(ValueError, match="mutually exclusive"):
            await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="hi")],
                tools=[ToolSchema(name="t", description="d", parameters={})],
                output_schema=MyOutput,
            )

    async def test_rate_limit_error_mapping(self) -> None:
        client = self._make_client()
        mock_response = MagicMock()
        mock_response.headers = {"retry-after": "30"}
        mock_response.status_code = 429

        exc = anthropic.RateLimitError(
            message="Rate limited",
            response=mock_response,
            body=None,
        )

        with patch.object(client._client.messages, "stream", return_value=_mock_stream_ctx(error=exc)):
            with pytest.raises(LLMRateLimitError) as exc_info:
                await client.generate(
                    system_prompt="test",
                    messages=[Message(role="user", content="hi")],
                )
            assert exc_info.value.retry_after == 30.0

    async def test_context_length_error_mapping(self) -> None:
        client = self._make_client()
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.headers = {}

        exc = anthropic.BadRequestError(
            message="prompt is too long: token count exceeds limit",
            response=mock_response,
            body={
                "error": {
                    "type": "invalid_request_error",
                    "message": "prompt is too long: token count exceeds limit",
                }
            },
        )

        with (
            patch.object(client._client.messages, "stream", return_value=_mock_stream_ctx(error=exc)),
            pytest.raises(LLMContextLengthError),
        ):
            await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="hi")],
            )

    async def test_bad_request_non_context_error_mapping(self) -> None:
        client = self._make_client()
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.headers = {}

        exc = anthropic.BadRequestError(
            message="invalid model",
            response=mock_response,
            body=None,
        )

        with (
            patch.object(client._client.messages, "stream", return_value=_mock_stream_ctx(error=exc)),
            pytest.raises(LLMProviderError),
        ):
            await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="hi")],
            )

    async def test_authentication_error_mapping(self) -> None:
        client = self._make_client()
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {}

        exc = anthropic.AuthenticationError(
            message="Invalid API key",
            response=mock_response,
            body={"error": {"type": "authentication_error", "message": "Invalid API key"}},
        )

        with patch.object(client._client.messages, "stream", return_value=_mock_stream_ctx(error=exc)):
            with pytest.raises(LLMAuthenticationError) as exc_info:
                await client.generate(
                    system_prompt="test",
                    messages=[Message(role="user", content="hi")],
                )
            assert exc_info.value.provider == "anthropic"
            assert exc_info.value.status_code == 401
            assert exc_info.value.provider_error_type == "authentication_error"

    async def test_api_status_error_mapping(self) -> None:
        client = self._make_client()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {}

        exc = anthropic.APIStatusError(
            message="Internal server error",
            response=mock_response,
            body=None,
        )

        with patch.object(client._client.messages, "stream", return_value=_mock_stream_ctx(error=exc)):
            with pytest.raises(LLMProviderError) as exc_info:
                await client.generate(
                    system_prompt="test",
                    messages=[Message(role="user", content="hi")],
                )
            assert exc_info.value.status_code == 500
            assert exc_info.value.provider_error_type is None

    async def test_connection_error_mapping(self) -> None:
        client = self._make_client()
        exc = anthropic.APIConnectionError(request=MagicMock())

        with patch.object(client._client.messages, "stream", return_value=_mock_stream_ctx(error=exc)):
            with pytest.raises(LLMProviderError) as exc_info:
                await client.generate(
                    system_prompt="test",
                    messages=[Message(role="user", content="hi")],
                )
            assert exc_info.value.provider == "anthropic"

    async def test_request_deadline(self) -> None:
        client = self._make_client(request_timeout=0.1)

        async def _hanging_aenter(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(10)  # pragma: no cover

        cm = AsyncMock()
        cm.__aenter__ = _hanging_aenter
        cm.__aexit__ = AsyncMock(return_value=False)

        with patch.object(client._client.messages, "stream", return_value=cm):
            with pytest.raises(LLMProviderError, match="timed out") as exc_info:
                await client.generate(
                    system_prompt="test",
                    messages=[Message(role="user", content="hi")],
                )
            assert exc_info.value.provider == "anthropic"

    def test_request_timeout_default(self) -> None:
        client = self._make_client()
        assert client._request_timeout == 300.0

    def test_enable_caching_defaults_to_false(self) -> None:
        # Cache writes cost ~1.25× input; a single call pays the premium
        # for no benefit. Callers with stable, repeated prefixes opt in.
        client = self._make_client()
        assert client._enable_caching is False


# --- Typed Error Subclass Mapping Tests (Phase: typed-llm-error-subclasses) ---


class TestExtractAnthropicErrorType:
    """Defensive shape checks for the body parser."""

    def test_missing_body(self) -> None:
        exc = MagicMock(spec=[])  # no body attribute
        assert _extract_anthropic_error_type(exc) is None

    def test_body_is_none(self) -> None:
        exc = MagicMock()
        exc.body = None
        assert _extract_anthropic_error_type(exc) is None

    def test_body_is_non_dict(self) -> None:
        exc = MagicMock()
        exc.body = "not-a-dict"
        assert _extract_anthropic_error_type(exc) is None

    def test_body_dict_without_error_key(self) -> None:
        exc = MagicMock()
        exc.body = {"other": "data"}
        assert _extract_anthropic_error_type(exc) is None

    def test_body_dict_with_non_dict_error(self) -> None:
        exc = MagicMock()
        exc.body = {"error": "not-a-dict"}
        assert _extract_anthropic_error_type(exc) is None

    def test_body_dict_with_non_string_error_type(self) -> None:
        exc = MagicMock()
        exc.body = {"error": {"type": 42}}
        assert _extract_anthropic_error_type(exc) is None

    def test_happy_path(self) -> None:
        exc = MagicMock()
        exc.body = {"error": {"type": "insufficient_quota", "message": "..."}}
        assert _extract_anthropic_error_type(exc) == "insufficient_quota"


class TestExtractAnthropicErrorMessage:
    """Defensive shape checks for the anchored-phrase body-message parser."""

    def test_missing_body(self) -> None:
        exc = MagicMock(spec=[])
        assert _extract_anthropic_error_message(exc) == ""

    def test_body_is_none(self) -> None:
        exc = MagicMock()
        exc.body = None
        assert _extract_anthropic_error_message(exc) == ""

    def test_body_is_non_dict(self) -> None:
        exc = MagicMock()
        exc.body = "not-a-dict"
        assert _extract_anthropic_error_message(exc) == ""

    def test_body_dict_with_non_dict_error(self) -> None:
        exc = MagicMock()
        exc.body = {"error": "not-a-dict"}
        assert _extract_anthropic_error_message(exc) == ""

    def test_body_dict_with_non_string_message(self) -> None:
        exc = MagicMock()
        exc.body = {"error": {"message": 42}}
        assert _extract_anthropic_error_message(exc) == ""

    def test_happy_path(self) -> None:
        exc = MagicMock()
        exc.body = {"error": {"type": "x", "message": "hello"}}
        assert _extract_anthropic_error_message(exc) == "hello"


class TestAnthropicTypedErrorMapping:
    """End-to-end mapping of upstream Anthropic exceptions to typed SDK subclasses."""

    def _make_client(self) -> AnthropicLLMClient:
        return AnthropicLLMClient(model="claude-test", api_key="test-key")

    async def _raise_via_client(self, exc: Exception) -> Any:
        client = self._make_client()
        with patch.object(client._client.messages, "stream", return_value=_mock_stream_ctx(error=exc)):
            await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="hi")],
            )

    async def test_rate_limit_with_insufficient_quota_routes_to_quota_exhausted(self) -> None:
        mock_response = MagicMock()
        mock_response.headers = {}
        mock_response.status_code = 429
        exc = anthropic.RateLimitError(
            message="Quota exhausted",
            response=mock_response,
            body={"error": {"type": "insufficient_quota", "message": "Quota exhausted"}},
        )
        with pytest.raises(LLMQuotaExhaustedError) as exc_info:
            await self._raise_via_client(exc)
        assert exc_info.value.provider == "anthropic"
        assert exc_info.value.status_code == 429
        assert exc_info.value.provider_error_type == "insufficient_quota"

    async def test_rate_limit_without_quota_signal_still_routes_to_rate_limit(self) -> None:
        # Regression: a generic 429 with no insufficient_quota body must
        # continue to route through LLMRateLimitError so existing
        # retry-with-backoff behaviour is preserved.
        mock_response = MagicMock()
        mock_response.headers = {"retry-after": "10"}
        mock_response.status_code = 429
        exc = anthropic.RateLimitError(
            message="Rate limited",
            response=mock_response,
            body={"error": {"type": "rate_limit_error", "message": "Slow down"}},
        )
        with pytest.raises(LLMRateLimitError) as exc_info:
            await self._raise_via_client(exc)
        assert exc_info.value.retry_after == 10.0

    async def test_bad_request_credit_balance_routes_to_quota_exhausted(self) -> None:
        mock_response = MagicMock()
        mock_response.headers = {}
        mock_response.status_code = 400
        exc = anthropic.BadRequestError(
            message="Your credit balance is too low",
            response=mock_response,
            body={
                "error": {
                    "type": "invalid_request_error",
                    "message": "Your credit balance is too low to access the API.",
                }
            },
        )
        with pytest.raises(LLMQuotaExhaustedError) as exc_info:
            await self._raise_via_client(exc)
        assert exc_info.value.status_code == 400
        assert exc_info.value.provider == "anthropic"
        assert exc_info.value.provider_error_type == "invalid_request_error"

    async def test_bad_request_other_invalid_request_routes_to_provider_error(self) -> None:
        mock_response = MagicMock()
        mock_response.headers = {}
        mock_response.status_code = 400
        exc = anthropic.BadRequestError(
            message="invalid",
            response=mock_response,
            body={"error": {"type": "invalid_request_error", "message": "invalid model"}},
        )
        with pytest.raises(LLMProviderError) as exc_info:
            await self._raise_via_client(exc)
        assert type(exc_info.value) is LLMProviderError
        assert exc_info.value.provider_error_type == "invalid_request_error"

    async def test_api_status_403_billing_routes_to_quota_exhausted(self) -> None:
        mock_response = MagicMock()
        mock_response.headers = {}
        mock_response.status_code = 403
        exc = anthropic.APIStatusError(
            message="Forbidden",
            response=mock_response,
            body={
                "error": {
                    "type": "permission_error",
                    "message": "Your credit balance is too low.",
                }
            },
        )
        with pytest.raises(LLMQuotaExhaustedError) as exc_info:
            await self._raise_via_client(exc)
        assert exc_info.value.status_code == 403
        assert exc_info.value.provider_error_type == "permission_error"

    async def test_api_status_403_non_billing_routes_to_provider_error(self) -> None:
        mock_response = MagicMock()
        mock_response.headers = {}
        mock_response.status_code = 403
        exc = anthropic.APIStatusError(
            message="Forbidden",
            response=mock_response,
            body={"error": {"type": "permission_error", "message": "Account suspended."}},
        )
        with pytest.raises(LLMProviderError) as exc_info:
            await self._raise_via_client(exc)
        assert type(exc_info.value) is LLMProviderError
        assert exc_info.value.status_code == 403
        assert exc_info.value.provider_error_type == "permission_error"

    async def test_api_status_529_routes_to_overloaded(self) -> None:
        mock_response = MagicMock()
        mock_response.headers = {}
        mock_response.status_code = 529
        exc = anthropic.APIStatusError(
            message="Overloaded",
            response=mock_response,
            body={"error": {"type": "overloaded_error", "message": "Overloaded"}},
        )
        with pytest.raises(LLMOverloadedError) as exc_info:
            await self._raise_via_client(exc)
        assert exc_info.value.status_code == 529
        assert exc_info.value.provider_error_type == "overloaded_error"

    async def test_api_status_other_5xx_routes_to_provider_error_with_type(self) -> None:
        mock_response = MagicMock()
        mock_response.headers = {}
        mock_response.status_code = 502
        exc = anthropic.APIStatusError(
            message="Bad gateway",
            response=mock_response,
            body={"error": {"type": "api_error", "message": "upstream"}},
        )
        with pytest.raises(LLMProviderError) as exc_info:
            await self._raise_via_client(exc)
        assert type(exc_info.value) is LLMProviderError
        assert exc_info.value.status_code == 502
        assert exc_info.value.provider_error_type == "api_error"


# --- Structured Output Tests ---


class TestStructuredOutput:
    async def test_structured_output_success(self) -> None:
        class MyOutput(BaseModel):
            answer: str
            confidence: float

        client = AnthropicLLMClient(model="claude-test", api_key="test-key")
        tool_block = _make_tool_use_block(
            "tc-1",
            STRUCTURED_OUTPUT_TOOL_NAME,
            {"answer": "42", "confidence": 0.95},
        )
        mock_response = _make_anthropic_response([tool_block], stop_reason="tool_use")

        with patch.object(
            client._client.messages, "stream", return_value=_mock_stream_ctx(response=mock_response)
        ) as mock_stream:
            result = await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="hi")],
                output_schema=MyOutput,
            )

        assert isinstance(result.parsed, MyOutput)
        assert result.parsed.answer == "42"
        assert result.parsed.confidence == 0.95
        assert result.tool_calls == []
        # Content should be JSON of the arguments
        assert result.content is not None
        assert '"answer"' in result.content
        # Verify tool_choice was forced
        call_kwargs = mock_stream.call_args.kwargs
        assert call_kwargs["tool_choice"] == {
            "type": "tool",
            "name": STRUCTURED_OUTPUT_TOOL_NAME,
        }

    async def test_structured_output_validation_failure(self) -> None:
        class StrictOutput(BaseModel):
            count: int

        client = AnthropicLLMClient(model="claude-test", api_key="test-key")
        tool_block = _make_tool_use_block(
            "tc-1",
            STRUCTURED_OUTPUT_TOOL_NAME,
            {"count": "not_a_number"},
        )
        mock_response = _make_anthropic_response([tool_block], stop_reason="tool_use")

        with (
            patch.object(client._client.messages, "stream", return_value=_mock_stream_ctx(response=mock_response)),
            pytest.raises(LLMSchemaViolationError),
        ):
            await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="hi")],
                output_schema=StrictOutput,
            )


# --- Prompt Caching Tests ---


class TestPromptCaching:
    async def test_system_prompt_cached_when_enabled(self) -> None:
        client = AnthropicLLMClient(model="claude-test", api_key="test-key", enable_caching=True)
        mock_response = _make_anthropic_response([_make_text_block("Hi")])

        with patch.object(
            client._client.messages, "stream", return_value=_mock_stream_ctx(response=mock_response)
        ) as mock_stream:
            await client.generate(
                system_prompt="You are helpful",
                messages=[Message(role="user", content="Hi")],
            )

        call_kwargs = mock_stream.call_args.kwargs
        assert call_kwargs["system"] == [
            {"type": "text", "text": "You are helpful", "cache_control": {"type": "ephemeral"}}
        ]
        # Message-level caching: last user message should have cache_control
        sent_messages = call_kwargs["messages"]
        last_user = next(m for m in reversed(sent_messages) if m["role"] == "user")
        assert isinstance(last_user["content"], list)
        assert last_user["content"][-1]["cache_control"] == {"type": "ephemeral"}

    async def test_tools_last_gets_cache_control_when_enabled(self) -> None:
        client = AnthropicLLMClient(model="claude-test", api_key="test-key", enable_caching=True)
        mock_response = _make_anthropic_response([_make_text_block("Hi")])
        tools = [
            ToolSchema(name="a", description="Tool A", parameters={"type": "object"}),
            ToolSchema(name="b", description="Tool B", parameters={"type": "object"}),
        ]

        with patch.object(
            client._client.messages, "stream", return_value=_mock_stream_ctx(response=mock_response)
        ) as mock_stream:
            await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="Hi")],
                tools=tools,
            )

        call_kwargs = mock_stream.call_args.kwargs
        sent_tools = call_kwargs["tools"]
        assert "cache_control" not in sent_tools[0]
        assert sent_tools[1]["cache_control"] == {"type": "ephemeral"}

    async def test_caching_disabled_plain_system_and_no_tool_cache_control(self) -> None:
        client = AnthropicLLMClient(model="claude-test", api_key="test-key", enable_caching=False)
        mock_response = _make_anthropic_response([_make_text_block("Hi")])
        tools = [
            ToolSchema(name="a", description="Tool A", parameters={"type": "object"}),
        ]

        with patch.object(
            client._client.messages, "stream", return_value=_mock_stream_ctx(response=mock_response)
        ) as mock_stream:
            await client.generate(
                system_prompt="You are helpful",
                messages=[Message(role="user", content="Hi")],
                tools=tools,
            )

        call_kwargs = mock_stream.call_args.kwargs
        assert call_kwargs["system"] == "You are helpful"
        assert "cache_control" not in call_kwargs["tools"][0]

    async def test_no_tools_only_system_cached(self) -> None:
        client = AnthropicLLMClient(model="claude-test", api_key="test-key", enable_caching=True)
        mock_response = _make_anthropic_response([_make_text_block("Hi")])

        with patch.object(
            client._client.messages, "stream", return_value=_mock_stream_ctx(response=mock_response)
        ) as mock_stream:
            await client.generate(
                system_prompt="You are helpful",
                messages=[Message(role="user", content="Hi")],
            )

        call_kwargs = mock_stream.call_args.kwargs
        assert call_kwargs["system"] == [
            {"type": "text", "text": "You are helpful", "cache_control": {"type": "ephemeral"}}
        ]
        assert "tools" not in call_kwargs

    async def test_system_prompt_sections_selective_caching(self) -> None:
        from nanitics.infrastructure.llm.protocol import SystemPromptSection

        client = AnthropicLLMClient(model="claude-test", api_key="test-key", enable_caching=True)
        mock_response = _make_anthropic_response([_make_text_block("Hi")])
        sections = [
            SystemPromptSection(content="Static base prompt", cacheable=True),
            SystemPromptSection(content="Dynamic state", cacheable=False),
            SystemPromptSection(content="Language instructions", cacheable=True),
        ]

        with patch.object(
            client._client.messages, "stream", return_value=_mock_stream_ctx(response=mock_response)
        ) as mock_stream:
            await client.generate(
                system_prompt="Static base prompt\n\nDynamic state\n\nLanguage instructions",
                messages=[Message(role="user", content="Hi")],
                system_prompt_sections=sections,
            )

        call_kwargs = mock_stream.call_args.kwargs
        system_blocks = call_kwargs["system"]
        assert len(system_blocks) == 3
        # Prefix-based caching: only the LAST cacheable section gets cache_control
        assert system_blocks[0] == {"type": "text", "text": "Static base prompt"}
        assert system_blocks[1] == {"type": "text", "text": "Dynamic state"}
        assert system_blocks[2] == {
            "type": "text",
            "text": "Language instructions",
            "cache_control": {"type": "ephemeral"},
        }

    async def test_system_prompt_sections_many_cacheable_only_last_gets_cache_control(self) -> None:
        from nanitics.infrastructure.llm.protocol import SystemPromptSection

        client = AnthropicLLMClient(model="claude-test", api_key="test-key", enable_caching=True)
        mock_response = _make_anthropic_response([_make_text_block("Hi")])
        sections = [
            SystemPromptSection(content="Base prompt", cacheable=True),
            SystemPromptSection(content="ICP Profile", cacheable=True),
            SystemPromptSection(content="Candidates", cacheable=True),
        ]

        with patch.object(
            client._client.messages, "stream", return_value=_mock_stream_ctx(response=mock_response)
        ) as mock_stream:
            await client.generate(
                system_prompt="ignored",
                messages=[Message(role="user", content="Hi")],
                system_prompt_sections=sections,
            )

        system_blocks = mock_stream.call_args.kwargs["system"]
        assert len(system_blocks) == 3
        # Only the last cacheable section gets cache_control (prefix-based caching)
        assert "cache_control" not in system_blocks[0]
        assert "cache_control" not in system_blocks[1]
        assert system_blocks[2]["cache_control"] == {"type": "ephemeral"}

    async def test_system_prompt_sections_all_non_cacheable(self) -> None:
        from nanitics.infrastructure.llm.protocol import SystemPromptSection

        client = AnthropicLLMClient(model="claude-test", api_key="test-key", enable_caching=True)
        mock_response = _make_anthropic_response([_make_text_block("Hi")])
        sections = [
            SystemPromptSection(content="Dynamic A", cacheable=False),
            SystemPromptSection(content="Dynamic B", cacheable=False),
        ]

        with patch.object(
            client._client.messages, "stream", return_value=_mock_stream_ctx(response=mock_response)
        ) as mock_stream:
            await client.generate(
                system_prompt="ignored",
                messages=[Message(role="user", content="Hi")],
                system_prompt_sections=sections,
            )

        system_blocks = mock_stream.call_args.kwargs["system"]
        assert len(system_blocks) == 2
        assert "cache_control" not in system_blocks[0]
        assert "cache_control" not in system_blocks[1]

    async def test_cache_control_budget_with_sections_tools_and_messages(self) -> None:
        """Total cache_control blocks must never exceed 4, even with many cacheable sections."""
        from nanitics.infrastructure.llm.protocol import SystemPromptSection

        client = AnthropicLLMClient(model="claude-test", api_key="test-key", enable_caching=True)
        mock_response = _make_anthropic_response([_make_text_block("Hi")])
        sections = [
            SystemPromptSection(content="Base", cacheable=True),
            SystemPromptSection(content="ICP", cacheable=True),
            SystemPromptSection(content="Candidates", cacheable=True),
        ]
        tools = [
            ToolSchema(name="tool_a", description="A", parameters={"type": "object", "properties": {}}),
            ToolSchema(name="tool_b", description="B", parameters={"type": "object", "properties": {}}),
        ]

        with patch.object(
            client._client.messages, "stream", return_value=_mock_stream_ctx(response=mock_response)
        ) as mock_stream:
            await client.generate(
                system_prompt="ignored",
                messages=[Message(role="user", content="Hi")],
                system_prompt_sections=sections,
                tools=tools,
            )

        call_kwargs = mock_stream.call_args.kwargs
        total_cache_blocks = 0
        for block in call_kwargs["system"]:
            if "cache_control" in block:
                total_cache_blocks += 1
        for tool in call_kwargs["tools"]:
            if "cache_control" in tool:
                total_cache_blocks += 1
        for msg in call_kwargs["messages"]:
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "cache_control" in block:
                        total_cache_blocks += 1
        assert total_cache_blocks <= 4, f"Expected at most 4 cache_control blocks, found {total_cache_blocks}"
        # Specifically: 1 system + 1 tools + 1 messages = 3
        assert total_cache_blocks == 3

    async def test_cache_control_budget_worst_case_with_structured_output(self) -> None:
        """Structured output + many cacheable sections + long history stays ≤4.

        Anthropic caps ``cache_control`` breakpoints at 4 per request. The
        worst-case composition this SDK can emit is:

        - ``enable_caching=True``
        - ``output_schema`` set (the synthetic ``structured_output`` tool
          receives a cache_control block — mutually exclusive with
          user-provided ``tools``)
        - ≥5 cacheable ``system_prompt_sections`` (only the last receives
          cache_control under prefix-based caching)
        - a long multi-turn message history where the last user message
          has several content blocks (only the last user message's last
          content block receives cache_control)

        The walk below counts cache_control occurrences across ``system``
        blocks, ``tools`` entries, and every message's content list —
        **not via a helper** (the test must fail fast if a future change
        hides breakpoints behind an abstraction).
        """
        from nanitics.infrastructure.llm.protocol import SystemPromptSection

        class StructuredOut(BaseModel):
            answer: str

        client = AnthropicLLMClient(model="claude-test", api_key="test-key", enable_caching=True)
        tool_block = _make_tool_use_block("tc-1", STRUCTURED_OUTPUT_TOOL_NAME, {"answer": "42"})
        mock_response = _make_anthropic_response([tool_block], stop_reason="tool_use")
        sections = [
            SystemPromptSection(content="Base", cacheable=True),
            SystemPromptSection(content="Environment", cacheable=True),
            SystemPromptSection(content="Planning", cacheable=True),
            SystemPromptSection(content="ICP", cacheable=True),
            SystemPromptSection(content="Candidates", cacheable=True),
        ]
        messages = [
            Message(role="user", content="turn-1 question"),
            Message(role="assistant", content="turn-1 answer"),
            Message(role="user", content="turn-2 question"),
            Message(role="assistant", content="turn-2 answer"),
            Message(role="user", content="turn-3 question"),
        ]

        with patch.object(
            client._client.messages, "stream", return_value=_mock_stream_ctx(response=mock_response)
        ) as mock_stream:
            await client.generate(
                system_prompt="ignored",
                messages=messages,
                system_prompt_sections=sections,
                output_schema=StructuredOut,
            )

        call_kwargs = mock_stream.call_args.kwargs
        total_cache_blocks = 0
        # System section blocks.
        for block in call_kwargs["system"]:
            if "cache_control" in block:
                total_cache_blocks += 1
        # Tools (synthetic structured_output tool, in this composition).
        for tool_entry in call_kwargs["tools"]:
            if "cache_control" in tool_entry:
                total_cache_blocks += 1
        # Message content blocks.
        for msg in call_kwargs["messages"]:
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "cache_control" in block:
                        total_cache_blocks += 1

        assert total_cache_blocks <= 4, (
            f"Anthropic caps cache_control breakpoints at 4 per request; "
            f"worst-case composition produced {total_cache_blocks}."
        )
        # Today's behaviour: last cacheable system section (1) + synthetic
        # structured_output tool (1) + last content block of last user
        # message (1) = 3. Guard against an unintended fifth breakpoint.
        assert total_cache_blocks == 3

    async def test_system_prompt_sections_ignored_when_caching_disabled(self) -> None:
        from nanitics.infrastructure.llm.protocol import SystemPromptSection

        client = AnthropicLLMClient(model="claude-test", api_key="test-key", enable_caching=False)
        mock_response = _make_anthropic_response([_make_text_block("Hi")])
        sections = [
            SystemPromptSection(content="Base", cacheable=True),
            SystemPromptSection(content="State", cacheable=False),
        ]

        with patch.object(
            client._client.messages, "stream", return_value=_mock_stream_ctx(response=mock_response)
        ) as mock_stream:
            await client.generate(
                system_prompt="Base\n\nState",
                messages=[Message(role="user", content="Hi")],
                system_prompt_sections=sections,
            )

        call_kwargs = mock_stream.call_args.kwargs
        # When caching is disabled, system_prompt_sections are ignored; plain string used
        assert call_kwargs["system"] == "Base\n\nState"

    async def test_output_schema_synthetic_tool_gets_cache_control(self) -> None:
        class MyOutput(BaseModel):
            answer: str

        client = AnthropicLLMClient(model="claude-test", api_key="test-key", enable_caching=True)
        tool_block = _make_tool_use_block("tc-1", STRUCTURED_OUTPUT_TOOL_NAME, {"answer": "42"})
        mock_response = _make_anthropic_response([tool_block], stop_reason="tool_use")

        with patch.object(
            client._client.messages, "stream", return_value=_mock_stream_ctx(response=mock_response)
        ) as mock_stream:
            await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="hi")],
                output_schema=MyOutput,
            )

        call_kwargs = mock_stream.call_args.kwargs
        synthetic_tool = call_kwargs["tools"][0]
        assert synthetic_tool["name"] == STRUCTURED_OUTPUT_TOOL_NAME
        assert synthetic_tool["cache_control"] == {"type": "ephemeral"}


# --- _to_anthropic_tools caching ---


class TestToAnthropicToolsCaching:
    def test_cache_control_on_last_tool(self) -> None:
        tools = [
            ToolSchema(name="a", description="A", parameters={}),
            ToolSchema(name="b", description="B", parameters={}),
        ]
        result = _to_anthropic_tools(tools, enable_caching=True)
        assert "cache_control" not in result[0]
        assert result[1]["cache_control"] == {"type": "ephemeral"}

    def test_no_cache_control_when_disabled(self) -> None:
        tools = [
            ToolSchema(name="a", description="A", parameters={}),
        ]
        result = _to_anthropic_tools(tools, enable_caching=False)
        assert "cache_control" not in result[0]

    def test_empty_tools_no_error(self) -> None:
        result = _to_anthropic_tools([], enable_caching=True)
        assert result == []
