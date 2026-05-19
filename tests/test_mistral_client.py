from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
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
from nanitics.infrastructure.llm.mistral import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    MistralLLMClient,
    _map_stop_reason,
    _parse_response,
    _parse_tool_calls,
    _to_mistral_messages,
    _to_mistral_tools,
)
from nanitics.tracing import (
    ImageContentBlock,
    Message,
    TextContentBlock,
    ToolCall,
)

# --- Helpers ---


def _make_response_json(
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    model: str = "mistral-small-latest",
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant"}
    if content is not None:
        message["content"] = content
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chat-123",
        "choices": [{"message": message, "finish_reason": finish_reason, "index": 0}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "model": model,
    }


def _make_tool_call_json(id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _make_httpx_response(data: dict[str, Any], status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=data,
        request=httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions"),
    )


def _make_error_response(status_code: int, body: str = "", headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        text=body,
        request=httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions"),
        headers=headers or {},
    )


def _make_sse_lines(*chunks: str) -> list[str]:
    """Build SSE lines from data payloads."""
    lines = [f"data: {c}" for c in chunks]
    lines.append("data: [DONE]")
    return lines


# --- Message Conversion Tests ---


class TestToMistralMessages:
    def test_system_prompt_first(self) -> None:
        result = _to_mistral_messages("You are helpful", [])
        assert result == [{"role": "system", "content": "You are helpful"}]

    def test_user_message(self) -> None:
        msgs = [Message(role="user", content="Hello")]
        result = _to_mistral_messages("sys", msgs)
        assert result[1] == {"role": "user", "content": "Hello"}

    def test_assistant_text_message(self) -> None:
        msgs = [Message(role="assistant", content="Hi there")]
        result = _to_mistral_messages("sys", msgs)
        assert result[1] == {"role": "assistant", "content": "Hi there"}

    def test_assistant_with_tool_calls(self) -> None:
        tc = ToolCall(id="tc-1", name="search", arguments={"q": "test"})
        msgs = [Message(role="assistant", tool_calls=[tc])]
        result = _to_mistral_messages("sys", msgs)
        assert result[1]["role"] == "assistant"
        assert len(result[1]["tool_calls"]) == 1
        assert result[1]["tool_calls"][0]["id"] == "tc-1"
        assert result[1]["tool_calls"][0]["type"] == "function"
        assert result[1]["tool_calls"][0]["function"]["name"] == "search"
        assert json.loads(result[1]["tool_calls"][0]["function"]["arguments"]) == {"q": "test"}

    def test_assistant_with_text_and_tool_calls(self) -> None:
        tc = ToolCall(id="tc-1", name="search", arguments={"q": "test"})
        msgs = [Message(role="assistant", content="Let me search", tool_calls=[tc])]
        result = _to_mistral_messages("sys", msgs)
        assert result[1]["content"] == "Let me search"
        assert len(result[1]["tool_calls"]) == 1

    def test_tool_result_message(self) -> None:
        msgs = [Message(role="tool_result", content="result data", tool_call_id="tc-1")]
        result = _to_mistral_messages("sys", msgs)
        assert result[1] == {
            "role": "tool",
            "tool_call_id": "tc-1",
            "content": "result data",
        }

    def test_tool_results_not_grouped(self) -> None:
        msgs = [
            Message(role="tool_result", content="result 1", tool_call_id="tc-1"),
            Message(role="tool_result", content="result 2", tool_call_id="tc-2"),
        ]
        result = _to_mistral_messages("sys", msgs)
        # system + 2 tool messages = 3
        assert len(result) == 3
        assert result[1]["role"] == "tool"
        assert result[2]["role"] == "tool"

    def test_empty_content(self) -> None:
        msgs = [Message(role="user", content=None)]
        result = _to_mistral_messages("sys", msgs)
        assert result[1] == {"role": "user", "content": ""}

    def test_empty_assistant_content_becomes_non_empty(self) -> None:
        """Empty assistant content is replaced with a space to avoid Mistral rejection."""
        msgs = [Message(role="assistant", content="")]
        result = _to_mistral_messages("sys", msgs)
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == " "

    def test_none_assistant_content_becomes_non_empty(self) -> None:
        """None assistant content is replaced with a space to avoid Mistral rejection."""
        msgs = [Message(role="assistant", content=None)]
        result = _to_mistral_messages("sys", msgs)
        assert result[1]["role"] == "assistant"
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
        result = _to_mistral_messages("sys", msgs)
        assert len(result) == 5  # system + 4 messages
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert result[3]["role"] == "tool"
        assert result[4]["role"] == "assistant"


# --- Vision Content Block Tests ---


class TestToMistralMessagesVision:
    def test_user_message_with_text_content_blocks(self) -> None:
        msgs = [
            Message(
                role="user",
                content=[TextContentBlock(text="Describe this image")],
            )
        ]
        result = _to_mistral_messages("sys", msgs)
        assert result[1] == {
            "role": "user",
            "content": [{"type": "text", "text": "Describe this image"}],
        }

    def test_user_message_with_image_base64(self) -> None:
        msgs = [
            Message(
                role="user",
                content=[
                    ImageContentBlock(media_type="image/png", data="iVBORw0KGgo="),
                ],
            )
        ]
        result = _to_mistral_messages("sys", msgs)
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
                content=[
                    ImageContentBlock(media_type="image/jpeg", data="https://example.com/img.jpg"),
                ],
            )
        ]
        result = _to_mistral_messages("sys", msgs)
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
        result = _to_mistral_messages("sys", msgs)
        blocks = result[1]["content"]
        assert len(blocks) == 3
        assert blocks[0] == {"type": "text", "text": "What's in this image?"}
        assert blocks[1] == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
        }
        assert blocks[2] == {"type": "text", "text": "Be specific."}


# --- Tool Schema Conversion Tests ---


class TestToMistralTools:
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
        result = _to_mistral_tools(tools)
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
        result = _to_mistral_tools(tools)
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
    def test_parse_single_tool_call(self) -> None:
        raw = [_make_tool_call_json("tc-1", "search", {"q": "test"})]
        result = _parse_tool_calls(raw)
        assert len(result) == 1
        assert result[0].id == "tc-1"
        assert result[0].name == "search"
        assert result[0].arguments == {"q": "test"}

    def test_parse_dict_arguments(self) -> None:
        raw = [{"id": "tc-1", "function": {"name": "search", "arguments": {"q": "test"}}}]
        result = _parse_tool_calls(raw)
        assert result[0].arguments == {"q": "test"}


# --- Response Parsing ---


class TestParseResponse:
    def test_text_response(self) -> None:
        data = _make_response_json(content="Hello")
        result = _parse_response(data, "mistral-small-latest")
        assert result.content == "Hello"
        assert result.tool_calls == []
        assert result.model == "mistral-small-latest"
        assert result.stop_reason == "end_turn"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 20
        assert result.usage.total_tokens == 30

    def test_tool_use_response(self) -> None:
        tc = _make_tool_call_json("tc-1", "search", {"q": "test"})
        data = _make_response_json(tool_calls=[tc], finish_reason="tool_calls")
        result = _parse_response(data, "mistral-small-latest")
        assert result.content is None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "tc-1"
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].arguments == {"q": "test"}
        assert result.stop_reason == "tool_use"

    def test_usage_tracking(self) -> None:
        data = _make_response_json(content="Hi", prompt_tokens=100, completion_tokens=50)
        result = _parse_response(data, "mistral-small-latest")
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50
        assert result.usage.total_tokens == 150

    def test_cache_tokens_always_none(self) -> None:
        data = _make_response_json(content="Hi")
        result = _parse_response(data, "mistral-small-latest")
        assert result.usage.cache_creation_input_tokens is None
        assert result.usage.cache_read_input_tokens is None


# --- Reasoning Extraction Tests ---


class TestReasoningExtraction:
    def test_tool_calls_with_content(self) -> None:
        """``tool_calls`` + non-empty content: ``reasoning_text`` is the content."""
        tc = _make_tool_call_json("tc-1", "search", {"q": "test"})
        data = _make_response_json(
            content="Let me search for that.",
            tool_calls=[tc],
            finish_reason="tool_calls",
        )
        result = _parse_response(data, "mistral-small-latest")
        assert result.content == "Let me search for that."
        assert result.reasoning_text == "Let me search for that."
        assert len(result.tool_calls) == 1

    def test_tool_calls_with_empty_content_is_none(self) -> None:
        """``tool_calls`` + empty-string content: ``reasoning_text is None``."""
        tc = _make_tool_call_json("tc-1", "search", {"q": "test"})
        data = _make_response_json(
            content="",
            tool_calls=[tc],
            finish_reason="tool_calls",
        )
        result = _parse_response(data, "mistral-small-latest")
        assert result.reasoning_text is None

    def test_tool_calls_without_content_is_none(self) -> None:
        """``tool_calls`` with no content key in the message: ``reasoning_text is None``."""
        tc = _make_tool_call_json("tc-1", "search", {"q": "test"})
        data = _make_response_json(tool_calls=[tc], finish_reason="tool_calls")
        result = _parse_response(data, "mistral-small-latest")
        assert result.reasoning_text is None

    def test_final_content_only_has_no_reasoning(self) -> None:
        """Final-answer response (no tool calls): ``reasoning_text is None``."""
        data = _make_response_json(content="The answer is 42.")
        result = _parse_response(data, "mistral-small-latest")
        assert result.content == "The answer is 42."
        assert result.reasoning_text is None

    def test_structured_output_has_no_reasoning(self) -> None:
        """Structured-output responses carry a forced ``structured_output``
        tool-call with no prose content; ``reasoning_text is None``.
        """
        tc = _make_tool_call_json("tc-1", STRUCTURED_OUTPUT_TOOL_NAME, {"answer": "42"})
        data = _make_response_json(tool_calls=[tc], finish_reason="tool_calls")
        result = _parse_response(data, "mistral-small-latest")
        assert result.reasoning_text is None


# --- MistralLLMClient Tests ---


class TestMistralLLMClient:
    def _make_client(self, **kwargs: Any) -> MistralLLMClient:
        kwargs.setdefault("model", "mistral-small-latest")
        kwargs.setdefault("api_key", "test-key")
        return MistralLLMClient(**kwargs)

    async def test_protocol_conformance(self) -> None:
        client = self._make_client()
        assert isinstance(client, LLMClient)

    def test_model_property(self) -> None:
        client = self._make_client(model="pixtral-12b-2409")
        assert client.model == "pixtral-12b-2409"

    async def test_successful_text_response(self) -> None:
        client = self._make_client()
        response_data = _make_response_json(content="Hello!")
        mock_response = _make_httpx_response(response_data)

        with patch.object(client._client, "post", return_value=mock_response) as mock_post:
            result = await client.generate(
                system_prompt="You are helpful",
                messages=[Message(role="user", content="Hi")],
            )

        assert result.content == "Hello!"
        assert result.model == "mistral-small-latest"
        assert result.stop_reason == "end_turn"
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert body["model"] == "mistral-small-latest"
        assert body["messages"][0] == {"role": "system", "content": "You are helpful"}

    async def test_successful_tool_use_response(self) -> None:
        client = self._make_client()
        tc = _make_tool_call_json("tc-1", "search", {"q": "test"})
        response_data = _make_response_json(tool_calls=[tc], finish_reason="tool_calls")
        mock_response = _make_httpx_response(response_data)

        tools = [
            ToolSchema(
                name="search",
                description="Search",
                parameters={"type": "object", "properties": {"q": {"type": "string"}}},
            )
        ]

        with patch.object(client._client, "post", return_value=mock_response) as mock_post:
            result = await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="search for test")],
                tools=tools,
            )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"
        body = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert "tools" in body
        assert body["tool_choice"] == "auto"

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
        error_response = _make_error_response(429, body="Rate limited", headers={"retry-after": "30"})
        mock_post = AsyncMock(
            side_effect=httpx.HTTPStatusError("Rate limited", request=error_response.request, response=error_response)
        )

        with patch.object(client._client, "post", mock_post):
            with pytest.raises(LLMRateLimitError) as exc_info:
                await client.generate(
                    system_prompt="test",
                    messages=[Message(role="user", content="hi")],
                )
            assert exc_info.value.retry_after == 30.0

    async def test_context_length_error_mapping(self) -> None:
        client = self._make_client()
        error_response = _make_error_response(
            400,
            body=json.dumps(
                {
                    "code": "context_length_exceeded",
                    "message": "prompt is too long: token count exceeds limit",
                }
            ),
        )
        mock_post = AsyncMock(
            side_effect=httpx.HTTPStatusError("Bad request", request=error_response.request, response=error_response)
        )

        with patch.object(client._client, "post", mock_post), pytest.raises(LLMContextLengthError):
            await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="hi")],
            )

    async def test_bad_request_non_context_error_mapping(self) -> None:
        client = self._make_client()
        error_response = _make_error_response(400, body="invalid model")
        mock_post = AsyncMock(
            side_effect=httpx.HTTPStatusError("Bad request", request=error_response.request, response=error_response)
        )

        with patch.object(client._client, "post", mock_post):
            with pytest.raises(LLMProviderError) as exc_info:
                await client.generate(
                    system_prompt="test",
                    messages=[Message(role="user", content="hi")],
                )
            assert exc_info.value.provider == "mistral"

    async def test_authentication_error_mapping(self) -> None:
        client = self._make_client()
        error_response = _make_error_response(401, body="Invalid API key")
        mock_post = AsyncMock(
            side_effect=httpx.HTTPStatusError("Unauthorized", request=error_response.request, response=error_response)
        )

        with patch.object(client._client, "post", mock_post):
            with pytest.raises(LLMProviderError) as exc_info:
                await client.generate(
                    system_prompt="test",
                    messages=[Message(role="user", content="hi")],
                )
            assert exc_info.value.provider == "mistral"
            assert exc_info.value.status_code == 401

    async def test_server_error_mapping(self) -> None:
        client = self._make_client()
        error_response = _make_error_response(500, body="Internal server error")
        mock_post = AsyncMock(
            side_effect=httpx.HTTPStatusError("Server error", request=error_response.request, response=error_response)
        )

        with patch.object(client._client, "post", mock_post):
            with pytest.raises(LLMProviderError) as exc_info:
                await client.generate(
                    system_prompt="test",
                    messages=[Message(role="user", content="hi")],
                )
            assert exc_info.value.status_code == 500

    async def test_unmatched_status_code_error_mapping(self) -> None:
        """Status codes not matching 429/401/400/5xx hit the catchall raise."""
        client = self._make_client()
        error_response = _make_error_response(404, body="Not found")
        mock_post = AsyncMock(
            side_effect=httpx.HTTPStatusError("Not found", request=error_response.request, response=error_response)
        )

        with patch.object(client._client, "post", mock_post):
            with pytest.raises(LLMProviderError) as exc_info:
                await client.generate(
                    system_prompt="test",
                    messages=[Message(role="user", content="hi")],
                )
            assert exc_info.value.status_code == 404
            assert exc_info.value.provider == "mistral"

    async def test_connection_error_mapping(self) -> None:
        client = self._make_client()
        mock_post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

        with patch.object(client._client, "post", mock_post):
            with pytest.raises(LLMProviderError) as exc_info:
                await client.generate(
                    system_prompt="test",
                    messages=[Message(role="user", content="hi")],
                )
            assert exc_info.value.provider == "mistral"

    async def test_timeout_exception_mapping(self) -> None:
        client = self._make_client()
        mock_post = AsyncMock(side_effect=httpx.TimeoutException("Read timed out"))

        with patch.object(client._client, "post", mock_post):
            with pytest.raises(LLMProviderError) as exc_info:
                await client.generate(
                    system_prompt="test",
                    messages=[Message(role="user", content="hi")],
                )
            assert exc_info.value.provider == "mistral"
            assert exc_info.value.status_code is None

    async def test_request_deadline_standard_request(self) -> None:
        client = self._make_client(request_timeout=0.1)

        async def _slow_post(*args: Any, **kwargs: Any) -> httpx.Response:
            await asyncio.sleep(10)
            return _make_httpx_response(_make_response_json(content="late"))  # pragma: no cover

        with (
            patch.object(client._client, "post", side_effect=_slow_post),
            pytest.raises(LLMProviderError, match="timed out"),
        ):
            await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="hi")],
            )

    async def test_request_deadline_streaming(self) -> None:
        client = self._make_client(request_timeout=0.1)

        async def _slow_stream(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(10)  # pragma: no cover

        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__ = _slow_stream
        mock_stream_cm.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(client._client, "stream", return_value=mock_stream_cm),
            pytest.raises(LLMProviderError, match="timed out"),
        ):
            await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="hi")],
                on_token=lambda t: None,
            )

    async def test_request_deadline_none_disables(self) -> None:
        client = self._make_client(request_timeout=None)
        response_data = _make_response_json(content="Hello!")
        mock_response = _make_httpx_response(response_data)

        with patch.object(client._client, "post", return_value=mock_response):
            result = await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="hi")],
            )
        assert result.content == "Hello!"

    def test_request_timeout_default(self) -> None:
        client = self._make_client()
        assert client._request_timeout == 300.0

    def test_api_key_from_environment(self) -> None:
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "env-key"}):
            client = MistralLLMClient(model="mistral-small-latest")
        assert client._api_key == "env-key"

    def test_api_key_missing_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True), pytest.raises(LLMProviderError, match="MISTRAL_API_KEY"):
            MistralLLMClient(model="mistral-small-latest")

    def test_api_key_parameter_overrides_env(self) -> None:
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "env-key"}):
            client = MistralLLMClient(model="mistral-small-latest", api_key="param-key")
        assert client._api_key == "param-key"

    async def test_usage_stats_mapped(self) -> None:
        client = self._make_client()
        response_data = _make_response_json(content="Hi", prompt_tokens=100, completion_tokens=50)
        mock_response = _make_httpx_response(response_data)

        with patch.object(client._client, "post", return_value=mock_response):
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

        client = MistralLLMClient(model="mistral-small-latest", api_key="test-key")
        tc = _make_tool_call_json(
            "tc-1",
            STRUCTURED_OUTPUT_TOOL_NAME,
            {"answer": "42", "confidence": 0.95},
        )
        response_data = _make_response_json(tool_calls=[tc], finish_reason="tool_calls")
        mock_response = _make_httpx_response(response_data)

        with patch.object(client._client, "post", return_value=mock_response) as mock_post:
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
        # Verify tool_choice was forced
        body = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert body["tool_choice"] == {
            "type": "function",
            "function": {"name": STRUCTURED_OUTPUT_TOOL_NAME},
        }

    async def test_structured_output_validation_failure(self) -> None:
        class StrictOutput(BaseModel):
            count: int

        client = MistralLLMClient(model="mistral-small-latest", api_key="test-key")
        tc = _make_tool_call_json(
            "tc-1",
            STRUCTURED_OUTPUT_TOOL_NAME,
            {"count": "not_a_number"},
        )
        response_data = _make_response_json(tool_calls=[tc], finish_reason="tool_calls")
        mock_response = _make_httpx_response(response_data)

        with patch.object(client._client, "post", return_value=mock_response), pytest.raises(LLMSchemaViolationError):
            await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="hi")],
                output_schema=StrictOutput,
            )


# --- Streaming Tests ---


class TestStreaming:
    async def test_streaming_calls_on_token(self) -> None:
        client = MistralLLMClient(model="mistral-small-latest", api_key="test-key")
        tokens_received: list[str] = []

        chunk1 = json.dumps(
            {
                "choices": [{"delta": {"content": "Hello"}, "index": 0}],
            }
        )
        chunk2 = json.dumps(
            {
                "choices": [{"delta": {"content": " world"}, "index": 0}],
            }
        )
        chunk3 = json.dumps(
            {
                "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )

        sse_lines = _make_sse_lines(chunk1, chunk2, chunk3)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = lambda: _async_iter(sse_lines)

        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=False)

        with patch.object(client._client, "stream", return_value=mock_stream_cm):
            result = await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="hi")],
                on_token=tokens_received.append,
            )

        assert tokens_received == ["Hello", " world"]
        assert result.content == "Hello world"
        assert result.stop_reason == "end_turn"

    async def test_streaming_disabled_for_structured_output(self) -> None:
        class MyOutput(BaseModel):
            answer: str

        client = MistralLLMClient(model="mistral-small-latest", api_key="test-key")
        tokens_received: list[str] = []

        tc = _make_tool_call_json("tc-1", STRUCTURED_OUTPUT_TOOL_NAME, {"answer": "42"})
        response_data = _make_response_json(tool_calls=[tc], finish_reason="tool_calls")
        mock_response = _make_httpx_response(response_data)

        with patch.object(client._client, "post", return_value=mock_response):
            result = await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="hi")],
                output_schema=MyOutput,
                on_token=tokens_received.append,
            )

        # on_token should NOT have been called (structured output uses non-streaming)
        assert tokens_received == []
        assert isinstance(result.parsed, MyOutput)

    async def test_streaming_with_tool_calls(self) -> None:
        client = MistralLLMClient(model="mistral-small-latest", api_key="test-key")
        tokens_received: list[str] = []

        chunk1 = json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "tc-1", "function": {"name": "search", "arguments": '{"q":'}}
                            ]
                        },
                        "index": 0,
                    }
                ],
            }
        )
        chunk2 = json.dumps(
            {
                "choices": [
                    {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"test"}'}}]}, "index": 0}
                ],
            }
        )
        chunk3 = json.dumps(
            {
                "choices": [{"delta": {}, "finish_reason": "tool_calls", "index": 0}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )

        sse_lines = _make_sse_lines(chunk1, chunk2, chunk3)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = lambda: _async_iter(sse_lines)

        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=False)

        tools = [ToolSchema(name="search", description="Search", parameters={})]

        with patch.object(client._client, "stream", return_value=mock_stream_cm):
            result = await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="search for test")],
                tools=tools,
                on_token=tokens_received.append,
            )

        assert tokens_received == []  # no text content
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].arguments == {"q": "test"}
        assert result.stop_reason == "tool_use"

    async def test_streaming_skips_non_data_lines(self) -> None:
        """Lines not starting with 'data: ' are ignored (e.g. SSE comments)."""
        client = MistralLLMClient(model="mistral-small-latest", api_key="test-key")
        tokens_received: list[str] = []

        content_chunk = json.dumps({"choices": [{"delta": {"content": "Hi"}, "index": 0}]})
        # Mix in a non-data line (SSE comment) before the real data
        lines = [": heartbeat", f"data: {content_chunk}", "data: [DONE]"]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = lambda: _async_iter(lines)

        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=False)

        with patch.object(client._client, "stream", return_value=mock_stream_cm):
            result = await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="hi")],
                on_token=tokens_received.append,
            )

        assert tokens_received == ["Hi"]
        assert result.content == "Hi"

    async def test_streaming_skips_empty_choices(self) -> None:
        """Chunks with empty 'choices' list are ignored."""
        client = MistralLLMClient(model="mistral-small-latest", api_key="test-key")
        tokens_received: list[str] = []

        no_choices_chunk = json.dumps({"choices": []})
        content_chunk = json.dumps({"choices": [{"delta": {"content": "Hello"}, "index": 0}]})
        lines = [f"data: {no_choices_chunk}", f"data: {content_chunk}", "data: [DONE]"]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = lambda: _async_iter(lines)

        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=False)

        with patch.object(client._client, "stream", return_value=mock_stream_cm):
            result = await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="hi")],
                on_token=tokens_received.append,
            )

        assert tokens_received == ["Hello"]
        assert result.content == "Hello"

    async def test_streaming_tool_call_empty_arguments_string(self) -> None:
        """Tool call with accumulated empty-string arguments resolves to {} not JSON parse."""
        client = MistralLLMClient(model="mistral-small-latest", api_key="test-key")

        # Tool call delta with empty arguments string
        chunk = json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [{"index": 0, "id": "tc-1", "function": {"name": "noop", "arguments": ""}}]
                        },
                        "index": 0,
                    }
                ]
            }
        )
        finish_chunk = json.dumps(
            {
                "choices": [{"delta": {}, "finish_reason": "tool_calls", "index": 0}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            }
        )
        lines = [f"data: {chunk}", f"data: {finish_chunk}", "data: [DONE]"]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = lambda: _async_iter(lines)

        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=False)

        tools = [ToolSchema(name="noop", description="No args", parameters={})]

        with patch.object(client._client, "stream", return_value=mock_stream_cm):
            result = await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="go")],
                tools=tools,
                on_token=lambda t: None,
            )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "noop"
        assert result.tool_calls[0].arguments == {}

    async def test_streaming_reasoning_text_from_prose_before_tool_call(self) -> None:
        """Streamed content tokens followed by tool-call deltas surface as
        ``reasoning_text``. Prose preceding a tool call is reasoning, not the
        final answer."""
        client = MistralLLMClient(model="mistral-small-latest", api_key="test-key")
        tokens_received: list[str] = []

        # Stream prose tokens first, then a tool call delta — the accumulated
        # prose must end up in ``reasoning_text`` and ``content``.
        chunk1 = json.dumps({"choices": [{"delta": {"content": "Let me "}, "index": 0}]})
        chunk2 = json.dumps({"choices": [{"delta": {"content": "search."}, "index": 0}]})
        chunk3 = json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "tc-1",
                                    "function": {"name": "search", "arguments": '{"q":"x"}'},
                                }
                            ]
                        },
                        "index": 0,
                    }
                ]
            }
        )
        chunk4 = json.dumps(
            {
                "choices": [{"delta": {}, "finish_reason": "tool_calls", "index": 0}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )
        sse_lines = _make_sse_lines(chunk1, chunk2, chunk3, chunk4)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = lambda: _async_iter(sse_lines)

        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=False)

        tools = [ToolSchema(name="search", description="Search", parameters={})]

        with patch.object(client._client, "stream", return_value=mock_stream_cm):
            result = await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="search for x")],
                tools=tools,
                on_token=tokens_received.append,
            )

        assert tokens_received == ["Let me ", "search."]
        assert result.reasoning_text == "Let me search."
        assert result.content == "Let me search."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"


async def _async_iter(items: list[str]) -> AsyncIterator[str]:
    for item in items:
        yield item
