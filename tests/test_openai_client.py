from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest
from pydantic import BaseModel

from nanitics.errors import (
    LLMContextLengthError,
    LLMProviderError,
    LLMRateLimitError,
    LLMSchemaViolationError,
)
from nanitics.infrastructure import (
    LLMClient,
    ToolSchema,
)
from nanitics.infrastructure.llm._openai_format import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    _from_openai_response,
    _map_stop_reason,
    _parse_tool_calls,
    _to_openai_messages,
    _to_openai_tools,
)
from nanitics.infrastructure.llm.openai import OpenAILLMClient
from nanitics.tracing import (
    ImageContentBlock,
    Message,
    TextContentBlock,
    ToolCall,
)

# --- Helpers ---


def _make_tool_call_obj(id: str, name: str, arguments: dict[str, Any]) -> MagicMock:
    """Build a non-streaming tool-call object with .id/.function.name/.function.arguments attrs."""
    tc = MagicMock()
    tc.id = id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


def _make_openai_response(
    content: str | None = None,
    tool_calls: list[MagicMock] | None = None,
    *,
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
) -> MagicMock:
    """Build a non-streaming ChatCompletion response mock."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _make_error_response(status_code: int, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        headers=headers or {},
    )


def _make_tool_call_delta(
    index: int,
    *,
    id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> MagicMock:
    """Build a streaming tool_call delta object."""
    delta = MagicMock()
    delta.index = index
    delta.id = id
    if name is None and arguments is None:
        delta.function = None
    else:
        func = MagicMock()
        func.name = name
        func.arguments = arguments
        delta.function = func
    return delta


def _make_stream_chunk(
    *,
    content: str | None = None,
    tool_calls: list[MagicMock] | None = None,
    finish_reason: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    include_choice: bool = True,
) -> MagicMock:
    """Build a streaming chunk mock."""
    chunk = MagicMock()
    if prompt_tokens is not None and completion_tokens is not None:
        chunk.usage = MagicMock()
        chunk.usage.prompt_tokens = prompt_tokens
        chunk.usage.completion_tokens = completion_tokens
    else:
        chunk.usage = None

    if include_choice:
        delta = MagicMock()
        delta.content = content
        delta.tool_calls = tool_calls
        choice = MagicMock()
        choice.delta = delta
        choice.finish_reason = finish_reason
        chunk.choices = [choice]
    else:
        chunk.choices = []
    return chunk


async def _async_iter(items: list[Any]) -> AsyncIterator[Any]:
    for item in items:
        yield item


def _mock_stream(chunks: list[MagicMock]) -> Any:
    """Produce a value returned by an awaited create(..., stream=True) call:
    an object that is itself async-iterable over the chunks."""

    class _AsyncStream:
        def __aiter__(self) -> AsyncIterator[MagicMock]:
            return _async_iter(chunks)

    return _AsyncStream()


# --- Message Conversion Tests ---


class TestToOpenAIMessages:
    def test_system_prompt_first(self) -> None:
        result = _to_openai_messages("You are helpful", [])
        assert result == [{"role": "system", "content": "You are helpful"}]

    def test_user_message(self) -> None:
        msgs = [Message(role="user", content="Hello")]
        result = _to_openai_messages("sys", msgs)
        assert result[1] == {"role": "user", "content": "Hello"}

    def test_assistant_text_message(self) -> None:
        msgs = [Message(role="assistant", content="Hi there")]
        result = _to_openai_messages("sys", msgs)
        assert result[1] == {"role": "assistant", "content": "Hi there"}

    def test_assistant_with_tool_calls(self) -> None:
        tc = ToolCall(id="tc-1", name="search", arguments={"q": "test"})
        msgs = [Message(role="assistant", tool_calls=[tc])]
        result = _to_openai_messages("sys", msgs)
        assert result[1]["role"] == "assistant"
        assert len(result[1]["tool_calls"]) == 1
        assert result[1]["tool_calls"][0]["id"] == "tc-1"
        assert result[1]["tool_calls"][0]["type"] == "function"
        assert result[1]["tool_calls"][0]["function"]["name"] == "search"
        # arguments are serialized as a JSON string
        assert json.loads(result[1]["tool_calls"][0]["function"]["arguments"]) == {"q": "test"}
        # No content key when tool_calls are present without text
        assert "content" not in result[1]

    def test_assistant_with_text_and_tool_calls(self) -> None:
        tc = ToolCall(id="tc-1", name="search", arguments={"q": "test"})
        msgs = [Message(role="assistant", content="Let me search", tool_calls=[tc])]
        result = _to_openai_messages("sys", msgs)
        assert result[1]["content"] == "Let me search"
        assert len(result[1]["tool_calls"]) == 1

    def test_tool_result_message(self) -> None:
        msgs = [Message(role="tool_result", content="result data", tool_call_id="tc-1")]
        result = _to_openai_messages("sys", msgs)
        assert result[1] == {
            "role": "tool",
            "tool_call_id": "tc-1",
            "content": "result data",
        }

    def test_tool_result_empty_content(self) -> None:
        msgs = [Message(role="tool_result", tool_call_id="tc-1")]
        result = _to_openai_messages("sys", msgs)
        assert result[1]["content"] == ""

    def test_empty_user_content(self) -> None:
        msgs = [Message(role="user", content=None)]
        result = _to_openai_messages("sys", msgs)
        assert result[1] == {"role": "user", "content": ""}

    def test_empty_assistant_content_becomes_non_empty(self) -> None:
        """Empty assistant content is replaced with a space to avoid OpenAI rejection."""
        msgs = [Message(role="assistant", content="")]
        result = _to_openai_messages("sys", msgs)
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == " "

    def test_none_assistant_content_becomes_non_empty(self) -> None:
        msgs = [Message(role="assistant", content=None)]
        result = _to_openai_messages("sys", msgs)
        assert result[1]["content"] == " "

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
        result = _to_openai_messages("sys", msgs)
        assert len(result) == 5
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert result[3]["role"] == "tool"
        assert result[4]["role"] == "assistant"


# --- Vision Content Block Tests ---


class TestToOpenAIMessagesVision:
    def test_user_message_with_text_content_blocks(self) -> None:
        msgs = [
            Message(
                role="user",
                content=[TextContentBlock(text="Describe this image")],
            )
        ]
        result = _to_openai_messages("sys", msgs)
        assert result[1] == {
            "role": "user",
            "content": [{"type": "text", "text": "Describe this image"}],
        }

    def test_user_message_with_image_base64(self) -> None:
        msgs = [
            Message(
                role="user",
                content=[ImageContentBlock(media_type="image/png", data="iVBORw0KGgo=")],
            )
        ]
        result = _to_openai_messages("sys", msgs)
        assert result[1] == {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                }
            ],
        }

    def test_user_message_with_image_url(self) -> None:
        msgs = [
            Message(
                role="user",
                content=[ImageContentBlock(media_type="image/jpeg", data="https://example.com/img.jpg")],
            )
        ]
        result = _to_openai_messages("sys", msgs)
        assert result[1] == {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
            ],
        }

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
        result = _to_openai_messages("sys", msgs)
        blocks = result[1]["content"]
        assert len(blocks) == 3
        assert blocks[0] == {"type": "text", "text": "What's in this image?"}
        assert blocks[1] == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
        }
        assert blocks[2] == {"type": "text", "text": "Be specific."}


# --- Tool Schema Conversion Tests ---


class TestToOpenAITools:
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
        result = _to_openai_tools(tools)
        assert result == [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                },
            }
        ]

    def test_multiple_tools(self) -> None:
        tools = [
            ToolSchema(name="a", description="Tool A", parameters={}),
            ToolSchema(name="b", description="Tool B", parameters={}),
        ]
        result = _to_openai_tools(tools)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "a"
        assert result[1]["function"]["name"] == "b"


# --- Stop Reason Mapping ---


class TestMapStopReason:
    def test_stop_maps_to_end_turn(self) -> None:
        assert _map_stop_reason("stop") == "end_turn"

    def test_tool_calls_maps_to_tool_use(self) -> None:
        assert _map_stop_reason("tool_calls") == "tool_use"

    def test_length_maps_to_max_tokens(self) -> None:
        assert _map_stop_reason("length") == "max_tokens"

    def test_none_maps_to_end_turn(self) -> None:
        assert _map_stop_reason(None) == "end_turn"


# --- Tool Call Parsing ---


class TestParseToolCalls:
    def test_parse_tool_call_object(self) -> None:
        tc_obj = _make_tool_call_obj("tc-1", "search", {"q": "test"})
        result = _parse_tool_calls([tc_obj])
        assert len(result) == 1
        assert result[0].id == "tc-1"
        assert result[0].name == "search"
        assert result[0].arguments == {"q": "test"}

    def test_parse_tool_call_dict_with_string_arguments(self) -> None:
        raw = [{"id": "tc-1", "function": {"name": "search", "arguments": json.dumps({"q": "test"})}}]
        result = _parse_tool_calls(raw)
        assert result[0].arguments == {"q": "test"}

    def test_parse_tool_call_dict_with_empty_arguments_string(self) -> None:
        raw = [{"id": "tc-1", "function": {"name": "noop", "arguments": ""}}]
        result = _parse_tool_calls(raw)
        assert result[0].arguments == {}

    def test_parse_tool_call_dict_with_dict_arguments(self) -> None:
        raw = [{"id": "tc-1", "function": {"name": "search", "arguments": {"q": "test"}}}]
        result = _parse_tool_calls(raw)
        assert result[0].arguments == {"q": "test"}


# --- Response Parsing ---


class TestParseResponse:
    def test_text_response(self) -> None:
        response = _make_openai_response(content="Hello")
        result = _from_openai_response(response, "gpt-4o-mini")
        assert result.content == "Hello"
        assert result.tool_calls == []
        assert result.model == "gpt-4o-mini"
        assert result.stop_reason == "end_turn"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 20
        assert result.usage.total_tokens == 30

    def test_tool_use_response(self) -> None:
        tc = _make_tool_call_obj("tc-1", "search", {"q": "test"})
        response = _make_openai_response(tool_calls=[tc], finish_reason="tool_calls")
        result = _from_openai_response(response, "gpt-4o-mini")
        assert result.content is None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "tc-1"
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].arguments == {"q": "test"}
        assert result.stop_reason == "tool_use"

    def test_usage_tracking(self) -> None:
        response = _make_openai_response(content="Hi", prompt_tokens=100, completion_tokens=50)
        result = _from_openai_response(response, "gpt-4o-mini")
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50
        assert result.usage.total_tokens == 150

    def test_cache_tokens_always_none(self) -> None:
        response = _make_openai_response(content="Hi")
        result = _from_openai_response(response, "gpt-4o-mini")
        assert result.usage.cache_creation_input_tokens is None
        assert result.usage.cache_read_input_tokens is None

    def test_missing_usage_defaults_to_zero(self) -> None:
        response = _make_openai_response(content="Hi")
        response.usage = None
        result = _from_openai_response(response, "gpt-4o-mini")
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0


# --- Reasoning Extraction Tests ---


class TestReasoningExtraction:
    def test_tool_calls_with_content(self) -> None:
        """``tool_calls`` + non-empty content: ``reasoning_text`` is the content."""
        tc = _make_tool_call_obj("tc-1", "search", {"q": "test"})
        response = _make_openai_response(
            content="Let me search for that.",
            tool_calls=[tc],
            finish_reason="tool_calls",
        )
        result = _from_openai_response(response, "gpt-4o-mini")
        assert result.content == "Let me search for that."
        assert result.reasoning_text == "Let me search for that."
        assert len(result.tool_calls) == 1

    def test_tool_calls_with_empty_content_is_none(self) -> None:
        """``tool_calls`` + empty-string content: ``reasoning_text is None``."""
        tc = _make_tool_call_obj("tc-1", "search", {"q": "test"})
        response = _make_openai_response(
            content="",
            tool_calls=[tc],
            finish_reason="tool_calls",
        )
        result = _from_openai_response(response, "gpt-4o-mini")
        assert result.reasoning_text is None

    def test_tool_calls_with_none_content_is_none(self) -> None:
        """``tool_calls`` + ``content is None``: ``reasoning_text is None``."""
        tc = _make_tool_call_obj("tc-1", "search", {"q": "test"})
        response = _make_openai_response(
            content=None,
            tool_calls=[tc],
            finish_reason="tool_calls",
        )
        result = _from_openai_response(response, "gpt-4o-mini")
        assert result.reasoning_text is None

    def test_final_content_only_has_no_reasoning(self) -> None:
        """Final-answer response (no tool calls): ``reasoning_text is None``."""
        response = _make_openai_response(content="The answer is 42.")
        result = _from_openai_response(response, "gpt-4o-mini")
        assert result.content == "The answer is 42."
        assert result.reasoning_text is None

    def test_structured_output_has_no_reasoning(self) -> None:
        """Structured-output responses (tool_calls consumed into parsed) still have
        ``reasoning_text`` reflecting the pre-extraction state. The OpenAI client
        rewrites ``content`` and ``tool_calls`` after parsing, but
        ``reasoning_text`` tracks prose-before-tool-call, which is independent of
        the structured-output rewrite. When structured output is requested and the
        tool-call carries no prose, ``reasoning_text is None``.
        """
        tc = _make_tool_call_obj("tc-1", STRUCTURED_OUTPUT_TOOL_NAME, {"answer": "42"})
        response = _make_openai_response(
            content=None,
            tool_calls=[tc],
            finish_reason="tool_calls",
        )
        result = _from_openai_response(response, "gpt-4o-mini")
        assert result.reasoning_text is None


# --- OpenAILLMClient Tests ---


class TestOpenAILLMClient:
    def _make_client(self, **kwargs: Any) -> OpenAILLMClient:
        kwargs.setdefault("model", "gpt-4o-mini")
        kwargs.setdefault("api_key", "test-key")
        return OpenAILLMClient(**kwargs)

    async def test_protocol_conformance(self) -> None:
        client = self._make_client()
        assert isinstance(client, LLMClient)

    def test_model_property(self) -> None:
        client = self._make_client(model="gpt-4o-mini")
        assert client.model == "gpt-4o-mini"

    def test_disables_openai_builtin_retries(self) -> None:
        client = self._make_client()
        assert client._client.max_retries == 0

    def test_base_url_passthrough(self) -> None:
        client = self._make_client(base_url="https://custom.example.com/v1")
        assert str(client._client.base_url).startswith("https://custom.example.com/v1")

    def test_preflight_raises_when_no_key_and_no_env(self) -> None:
        with patch.dict("os.environ", {}, clear=True), pytest.raises(LLMProviderError, match="OPENAI_API_KEY"):
            OpenAILLMClient(model="gpt-4o-mini")

    def test_preflight_passes_when_api_key_provided(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            client = OpenAILLMClient(model="gpt-4o-mini", api_key="explicit")
        assert client.model == "gpt-4o-mini"

    def test_preflight_passes_when_env_set(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "from-env"}, clear=True):
            client = OpenAILLMClient(model="gpt-4o-mini")
        assert client.model == "gpt-4o-mini"

    def test_preflight_passes_for_local_llm_with_dummy_key(self) -> None:
        # Local OpenAI-compatible endpoints (Ollama, vLLM, LM Studio) accept
        # any non-empty string as the key; docs/guides/local-llms.md shows this pattern.
        with patch.dict("os.environ", {}, clear=True):
            client = OpenAILLMClient(
                model="gpt-4o-mini",
                api_key="ollama",
                base_url="http://localhost:11434/v1",
            )
        assert client.model == "gpt-4o-mini"

    def test_request_timeout_default(self) -> None:
        client = self._make_client()
        assert client._request_timeout == 300.0

    async def test_successful_text_response(self) -> None:
        client = self._make_client()
        mock_response = _make_openai_response(content="Hello!")

        with patch.object(
            client._client.chat.completions, "create", new=AsyncMock(return_value=mock_response)
        ) as mock_create:
            result = await client.generate(
                system_prompt="You are helpful",
                messages=[Message(role="user", content="Hi")],
            )

        assert result.content == "Hello!"
        assert result.model == "gpt-4o-mini"
        assert result.stop_reason == "end_turn"
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["messages"][0] == {"role": "system", "content": "You are helpful"}

    async def test_successful_tool_use_response(self) -> None:
        client = self._make_client()
        tc = _make_tool_call_obj("tc-1", "search", {"q": "test"})
        mock_response = _make_openai_response(tool_calls=[tc], finish_reason="tool_calls")

        tools = [
            ToolSchema(
                name="search",
                description="Search",
                parameters={"type": "object", "properties": {"q": {"type": "string"}}},
            )
        ]

        with patch.object(
            client._client.chat.completions, "create", new=AsyncMock(return_value=mock_response)
        ) as mock_create:
            result = await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="search for test")],
                tools=tools,
            )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"
        call_kwargs = mock_create.call_args.kwargs
        assert "tools" in call_kwargs
        assert call_kwargs["tool_choice"] == "auto"

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

    async def test_rate_limit_error_mapping_with_retry_after(self) -> None:
        client = self._make_client()
        response = _make_error_response(429, headers={"retry-after": "30"})
        exc = openai.RateLimitError(message="Rate limited", response=response, body=None)

        with patch.object(client._client.chat.completions, "create", new=AsyncMock(side_effect=exc)):
            with pytest.raises(LLMRateLimitError) as exc_info:
                await client.generate(
                    system_prompt="test",
                    messages=[Message(role="user", content="hi")],
                )
            assert exc_info.value.retry_after == 30.0

    async def test_rate_limit_error_mapping_without_retry_after(self) -> None:
        client = self._make_client()
        response = _make_error_response(429)
        exc = openai.RateLimitError(message="Rate limited", response=response, body=None)

        with patch.object(client._client.chat.completions, "create", new=AsyncMock(side_effect=exc)):
            with pytest.raises(LLMRateLimitError) as exc_info:
                await client.generate(
                    system_prompt="test",
                    messages=[Message(role="user", content="hi")],
                )
            assert exc_info.value.retry_after is None

    async def test_context_length_error_mapping(self) -> None:
        client = self._make_client()
        response = _make_error_response(400)
        exc = openai.BadRequestError(
            message="This model's maximum context length is 8192 tokens",
            response=response,
            body={
                "code": "context_length_exceeded",
                "message": "This model's maximum context length is 8192 tokens",
            },
        )

        with (
            patch.object(client._client.chat.completions, "create", new=AsyncMock(side_effect=exc)),
            pytest.raises(LLMContextLengthError),
        ):
            await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="hi")],
            )

    async def test_bad_request_non_context_error_mapping(self) -> None:
        client = self._make_client()
        response = _make_error_response(400)
        exc = openai.BadRequestError(message="invalid model id", response=response, body=None)

        with patch.object(client._client.chat.completions, "create", new=AsyncMock(side_effect=exc)):
            with pytest.raises(LLMProviderError) as exc_info:
                await client.generate(
                    system_prompt="test",
                    messages=[Message(role="user", content="hi")],
                )
            assert exc_info.value.provider == "openai"
            assert exc_info.value.status_code == 400

    async def test_authentication_error_mapping(self) -> None:
        client = self._make_client()
        response = _make_error_response(401)
        exc = openai.AuthenticationError(message="Invalid API key", response=response, body=None)

        with patch.object(client._client.chat.completions, "create", new=AsyncMock(side_effect=exc)):
            with pytest.raises(LLMProviderError) as exc_info:
                await client.generate(
                    system_prompt="test",
                    messages=[Message(role="user", content="hi")],
                )
            assert exc_info.value.provider == "openai"
            assert exc_info.value.status_code == 401

    async def test_api_status_error_mapping(self) -> None:
        client = self._make_client()
        response = _make_error_response(500)
        exc = openai.APIStatusError(message="Server error", response=response, body=None)

        with patch.object(client._client.chat.completions, "create", new=AsyncMock(side_effect=exc)):
            with pytest.raises(LLMProviderError) as exc_info:
                await client.generate(
                    system_prompt="test",
                    messages=[Message(role="user", content="hi")],
                )
            assert exc_info.value.status_code == 500

    async def test_connection_error_mapping(self) -> None:
        client = self._make_client()
        exc = openai.APIConnectionError(request=MagicMock())

        with patch.object(client._client.chat.completions, "create", new=AsyncMock(side_effect=exc)):
            with pytest.raises(LLMProviderError) as exc_info:
                await client.generate(
                    system_prompt="test",
                    messages=[Message(role="user", content="hi")],
                )
            assert exc_info.value.provider == "openai"
            assert exc_info.value.status_code is None

    async def test_request_deadline(self) -> None:
        client = self._make_client(request_timeout=0.1)

        async def _slow_create(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(10)  # pragma: no cover

        with (
            patch.object(client._client.chat.completions, "create", new=AsyncMock(side_effect=_slow_create)),
            pytest.raises(LLMProviderError, match="timed out") as exc_info,
        ):
            await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="hi")],
            )
        assert exc_info.value.provider == "openai"

    async def test_request_deadline_none_disables(self) -> None:
        client = self._make_client(request_timeout=None)
        mock_response = _make_openai_response(content="Hello!")

        with patch.object(client._client.chat.completions, "create", new=AsyncMock(return_value=mock_response)):
            result = await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="hi")],
            )
        assert result.content == "Hello!"

    async def test_usage_stats_mapped(self) -> None:
        client = self._make_client()
        mock_response = _make_openai_response(content="Hi", prompt_tokens=100, completion_tokens=50)

        with patch.object(client._client.chat.completions, "create", new=AsyncMock(return_value=mock_response)):
            result = await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="hi")],
            )

        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50
        assert result.usage.total_tokens == 150
        assert result.usage.cache_creation_input_tokens is None
        assert result.usage.cache_read_input_tokens is None


# --- Structured Output Tests ---


class TestStructuredOutput:
    async def test_structured_output_success(self) -> None:
        class MyOutput(BaseModel):
            answer: str
            confidence: float

        client = OpenAILLMClient(model="gpt-4o-mini", api_key="test-key")
        tc = _make_tool_call_obj(
            "tc-1",
            STRUCTURED_OUTPUT_TOOL_NAME,
            {"answer": "42", "confidence": 0.95},
        )
        mock_response = _make_openai_response(tool_calls=[tc], finish_reason="tool_calls")

        with patch.object(
            client._client.chat.completions, "create", new=AsyncMock(return_value=mock_response)
        ) as mock_create:
            result = await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="hi")],
                output_schema=MyOutput,
            )

        assert isinstance(result.parsed, MyOutput)
        assert result.parsed.answer == "42"
        assert result.parsed.confidence == 0.95
        assert result.tool_calls == []
        assert result.content is not None
        assert '"answer"' in result.content
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["tool_choice"] == {
            "type": "function",
            "function": {"name": STRUCTURED_OUTPUT_TOOL_NAME},
        }

    async def test_structured_output_validation_failure(self) -> None:
        class StrictOutput(BaseModel):
            count: int

        client = OpenAILLMClient(model="gpt-4o-mini", api_key="test-key")
        tc = _make_tool_call_obj(
            "tc-1",
            STRUCTURED_OUTPUT_TOOL_NAME,
            {"count": "not_a_number"},
        )
        mock_response = _make_openai_response(tool_calls=[tc], finish_reason="tool_calls")

        with (
            patch.object(client._client.chat.completions, "create", new=AsyncMock(return_value=mock_response)),
            pytest.raises(LLMSchemaViolationError),
        ):
            await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="hi")],
                output_schema=StrictOutput,
            )

    async def test_structured_output_disables_streaming(self) -> None:
        class MyOutput(BaseModel):
            answer: str

        client = OpenAILLMClient(model="gpt-4o-mini", api_key="test-key")
        tc = _make_tool_call_obj("tc-1", STRUCTURED_OUTPUT_TOOL_NAME, {"answer": "42"})
        mock_response = _make_openai_response(tool_calls=[tc], finish_reason="tool_calls")
        tokens: list[str] = []

        with patch.object(
            client._client.chat.completions, "create", new=AsyncMock(return_value=mock_response)
        ) as mock_create:
            result = await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="hi")],
                output_schema=MyOutput,
                on_token=tokens.append,
            )

        assert tokens == []
        assert isinstance(result.parsed, MyOutput)
        # Non-streaming call — no "stream": True
        assert "stream" not in mock_create.call_args.kwargs


# --- Streaming Tests ---


class TestStreaming:
    async def test_streaming_calls_on_token(self) -> None:
        client = OpenAILLMClient(model="gpt-4o-mini", api_key="test-key")
        tokens_received: list[str] = []

        chunks = [
            _make_stream_chunk(content="Hello"),
            _make_stream_chunk(content=" world"),
            _make_stream_chunk(finish_reason="stop", prompt_tokens=10, completion_tokens=5),
        ]

        with patch.object(
            client._client.chat.completions,
            "create",
            new=AsyncMock(return_value=_mock_stream(chunks)),
        ) as mock_create:
            result = await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="hi")],
                on_token=tokens_received.append,
            )

        assert tokens_received == ["Hello", " world"]
        assert result.content == "Hello world"
        assert result.stop_reason == "end_turn"
        # Ensure include_usage was requested on the streaming call
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["stream"] is True
        assert call_kwargs["stream_options"] == {"include_usage": True}
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    async def test_streaming_with_tool_calls(self) -> None:
        client = OpenAILLMClient(model="gpt-4o-mini", api_key="test-key")
        tokens_received: list[str] = []

        chunks = [
            _make_stream_chunk(
                tool_calls=[_make_tool_call_delta(0, id="tc-1", name="search", arguments='{"q":')],
            ),
            _make_stream_chunk(
                tool_calls=[_make_tool_call_delta(0, arguments='"test"}')],
            ),
            _make_stream_chunk(finish_reason="tool_calls", prompt_tokens=10, completion_tokens=5),
        ]

        tools = [ToolSchema(name="search", description="Search", parameters={})]

        with patch.object(
            client._client.chat.completions,
            "create",
            new=AsyncMock(return_value=_mock_stream(chunks)),
        ):
            result = await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="go")],
                tools=tools,
                on_token=tokens_received.append,
            )

        assert tokens_received == []
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "tc-1"
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].arguments == {"q": "test"}
        assert result.stop_reason == "tool_use"

    async def test_streaming_skips_empty_choices(self) -> None:
        """Chunks with empty 'choices' list are ignored."""
        client = OpenAILLMClient(model="gpt-4o-mini", api_key="test-key")
        tokens_received: list[str] = []

        chunks = [
            _make_stream_chunk(include_choice=False),  # empty choices
            _make_stream_chunk(content="Hello"),
            _make_stream_chunk(finish_reason="stop", prompt_tokens=5, completion_tokens=2),
        ]

        with patch.object(
            client._client.chat.completions,
            "create",
            new=AsyncMock(return_value=_mock_stream(chunks)),
        ):
            result = await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="hi")],
                on_token=tokens_received.append,
            )

        assert tokens_received == ["Hello"]
        assert result.content == "Hello"

    async def test_streaming_tool_call_delta_without_function(self) -> None:
        """Tool-call deltas without a function field (e.g., only .id updates) are safely ignored."""
        client = OpenAILLMClient(model="gpt-4o-mini", api_key="test-key")

        chunks = [
            _make_stream_chunk(
                tool_calls=[_make_tool_call_delta(0, id="tc-1", name="noop", arguments="{}")],
            ),
            # A second delta with just an index and no id/name/arguments — function is None
            _make_stream_chunk(
                tool_calls=[_make_tool_call_delta(0)],
            ),
            _make_stream_chunk(finish_reason="tool_calls", prompt_tokens=5, completion_tokens=2),
        ]

        tools = [ToolSchema(name="noop", description="No args", parameters={})]

        with patch.object(
            client._client.chat.completions,
            "create",
            new=AsyncMock(return_value=_mock_stream(chunks)),
        ):
            result = await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="go")],
                tools=tools,
                on_token=lambda t: None,
            )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "noop"
        assert result.tool_calls[0].arguments == {}

    async def test_streaming_tool_call_empty_arguments_string(self) -> None:
        """Tool call with only empty-string arguments resolves to {}."""
        client = OpenAILLMClient(model="gpt-4o-mini", api_key="test-key")

        chunks = [
            _make_stream_chunk(
                tool_calls=[_make_tool_call_delta(0, id="tc-1", name="noop", arguments="")],
            ),
            _make_stream_chunk(finish_reason="tool_calls", prompt_tokens=5, completion_tokens=2),
        ]

        tools = [ToolSchema(name="noop", description="No args", parameters={})]

        with patch.object(
            client._client.chat.completions,
            "create",
            new=AsyncMock(return_value=_mock_stream(chunks)),
        ):
            result = await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="go")],
                tools=tools,
                on_token=lambda t: None,
            )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments == {}

    async def test_streaming_reasoning_text_from_prose_before_tool_call(self) -> None:
        """Streaming with prose content + tool_calls populates ``reasoning_text``."""
        client = OpenAILLMClient(model="gpt-4o-mini", api_key="test-key")
        tokens_received: list[str] = []

        chunks = [
            _make_stream_chunk(content="Let me "),
            _make_stream_chunk(content="search."),
            _make_stream_chunk(
                tool_calls=[_make_tool_call_delta(0, id="tc-1", name="search", arguments='{"q":"x"}')],
            ),
            _make_stream_chunk(finish_reason="tool_calls", prompt_tokens=10, completion_tokens=5),
        ]

        tools = [ToolSchema(name="search", description="Search", parameters={})]

        with patch.object(
            client._client.chat.completions,
            "create",
            new=AsyncMock(return_value=_mock_stream(chunks)),
        ):
            result = await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="go")],
                tools=tools,
                on_token=tokens_received.append,
            )

        assert result.content == "Let me search."
        assert result.reasoning_text == "Let me search."
        assert len(result.tool_calls) == 1
