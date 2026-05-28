from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import litellm
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
from nanitics.infrastructure.llm._openai_format import STRUCTURED_OUTPUT_TOOL_NAME
from nanitics.infrastructure.llm.litellm import (
    LiteLLMClient,
    _extract_litellm_error_type,
    _extract_litellm_quota_signal,
)
from nanitics.tracing import (
    Message,
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
    """Build a non-streaming OpenAI-shaped ChatCompletion response mock (as LiteLLM returns)."""
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


def _make_httpx_response(status_code: int, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        request=httpx.Request("POST", "https://example.com"),
        headers=headers or {},
    )


def _make_tool_call_delta(
    index: int,
    *,
    id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> MagicMock:
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
    class _AsyncStream:
        def __aiter__(self) -> AsyncIterator[MagicMock]:
            return _async_iter(chunks)

    return _AsyncStream()


# --- Construction & protocol conformance ---


class TestLiteLLMClientConstruction:
    def test_protocol_conformance(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        assert isinstance(client, LLMClient)

    def test_model_property_returns_input_string(self) -> None:
        client = LiteLLMClient(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")
        assert client.model == "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"

    def test_stores_extra_kwargs(self) -> None:
        client = LiteLLMClient(
            model="bedrock/foo",
            extra_kwargs={"aws_region_name": "us-east-1"},
        )
        assert client._extra_kwargs == {"aws_region_name": "us-east-1"}

    def test_extra_kwargs_defaults_to_empty_dict(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        assert client._extra_kwargs == {}

    def test_base_url_stored(self) -> None:
        client = LiteLLMClient(model="ollama/llama3", base_url="http://localhost:11434")
        assert client._base_url == "http://localhost:11434"

    def test_request_timeout_default(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        assert client._request_timeout == 300.0

    def test_max_tokens_default(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        assert client._max_tokens == 16_384


# --- generate() — non-streaming happy paths ---


class TestLiteLLMClientGenerate:
    async def test_successful_text_response(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        mock_response = _make_openai_response(content="Hello from LiteLLM!")

        with patch(
            "nanitics.infrastructure.llm.litellm.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ) as mock_ac:
            result = await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
            )

        assert result.content == "Hello from LiteLLM!"
        assert result.model == "openai/gpt-4o-mini"
        assert result.stop_reason == "end_turn"
        call_kwargs = mock_ac.call_args.kwargs
        assert call_kwargs["model"] == "openai/gpt-4o-mini"
        assert call_kwargs["num_retries"] == 0
        # First message is the system message
        assert call_kwargs["messages"][0] == {"role": "system", "content": "sys"}

    async def test_successful_tool_use_response(self) -> None:
        client = LiteLLMClient(model="anthropic/claude-haiku-4-5")
        tc = _make_tool_call_obj("tc-1", "search", {"q": "x"})
        mock_response = _make_openai_response(tool_calls=[tc], finish_reason="tool_calls")

        tools = [
            ToolSchema(
                name="search",
                description="Search",
                parameters={"type": "object", "properties": {"q": {"type": "string"}}},
            )
        ]
        with patch(
            "nanitics.infrastructure.llm.litellm.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ) as mock_ac:
            result = await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="search for x")],
                tools=tools,
            )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"
        call_kwargs = mock_ac.call_args.kwargs
        assert call_kwargs["tool_choice"] == "auto"
        assert "tools" in call_kwargs

    async def test_mutually_exclusive_tools_and_output_schema(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o-mini")

        class MySchema(BaseModel):
            answer: str

        with pytest.raises(ValueError, match="mutually exclusive"):
            await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
                tools=[ToolSchema(name="t", description="d", parameters={})],
                output_schema=MySchema,
            )

    async def test_message_conversion_is_delegated_to_shared_helper(self) -> None:
        """Assistant with tool_calls goes through _openai_format without modification."""
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        mock_response = _make_openai_response(content="ok")

        with patch(
            "nanitics.infrastructure.llm.litellm.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ) as mock_ac:
            await client.generate(
                system_prompt="sys",
                messages=[
                    Message(role="user", content="hi"),
                    Message(
                        role="assistant",
                        content="t",
                        tool_calls=[ToolCall(id="tc-1", name="search", arguments={"q": "x"})],
                    ),
                    Message(role="tool_result", tool_call_id="tc-1", content="result"),
                ],
            )

        call_kwargs = mock_ac.call_args.kwargs
        msgs = call_kwargs["messages"]
        assert msgs[2]["role"] == "assistant"
        assert msgs[2]["tool_calls"][0]["id"] == "tc-1"
        assert msgs[3]["role"] == "tool"

    async def test_max_tokens_passed_through(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o-mini", max_tokens=256)
        mock_response = _make_openai_response(content="ok")

        with patch(
            "nanitics.infrastructure.llm.litellm.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ) as mock_ac:
            await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
            )

        assert mock_ac.call_args.kwargs["max_tokens"] == 256

    async def test_api_key_passed_through(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o-mini", api_key="sk-test")
        mock_response = _make_openai_response(content="ok")

        with patch(
            "nanitics.infrastructure.llm.litellm.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ) as mock_ac:
            await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
            )

        assert mock_ac.call_args.kwargs["api_key"] == "sk-test"

    async def test_usage_stats_mapped(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        mock_response = _make_openai_response(content="Hi", prompt_tokens=100, completion_tokens=50)

        with patch(
            "nanitics.infrastructure.llm.litellm.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
            )

        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50
        assert result.usage.total_tokens == 150
        assert result.usage.cache_creation_input_tokens is None
        assert result.usage.cache_read_input_tokens is None


# --- Reasoning Extraction Tests ---


class TestReasoningExtraction:
    """End-to-end reasoning_text extraction via LiteLLM's OpenAI-shape normalization."""

    async def test_tool_calls_with_content(self) -> None:
        """``tool_calls`` + non-empty content: ``reasoning_text`` is the content."""
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        tc = _make_tool_call_obj("tc-1", "search", {"q": "test"})
        mock_response = _make_openai_response(
            content="Let me search for that.",
            tool_calls=[tc],
            finish_reason="tool_calls",
        )
        tools = [
            ToolSchema(
                name="search",
                description="Search",
                parameters={"type": "object", "properties": {"q": {"type": "string"}}},
            )
        ]
        with patch(
            "nanitics.infrastructure.llm.litellm.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="search for x")],
                tools=tools,
            )
        assert result.content == "Let me search for that."
        assert result.reasoning_text == "Let me search for that."

    async def test_tool_calls_with_empty_content_is_none(self) -> None:
        """``tool_calls`` + empty-string content: ``reasoning_text is None``."""
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        tc = _make_tool_call_obj("tc-1", "search", {"q": "test"})
        mock_response = _make_openai_response(
            content="",
            tool_calls=[tc],
            finish_reason="tool_calls",
        )
        tools = [
            ToolSchema(
                name="search",
                description="Search",
                parameters={"type": "object", "properties": {"q": {"type": "string"}}},
            )
        ]
        with patch(
            "nanitics.infrastructure.llm.litellm.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="search for x")],
                tools=tools,
            )
        assert result.reasoning_text is None

    async def test_final_content_only_has_no_reasoning(self) -> None:
        """Final-answer response (no tool calls): ``reasoning_text is None``."""
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        mock_response = _make_openai_response(content="The answer is 42.")
        with patch(
            "nanitics.infrastructure.llm.litellm.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
            )
        assert result.content == "The answer is 42."
        assert result.reasoning_text is None

    async def test_structured_output_has_no_reasoning(self) -> None:
        """Structured output (forced ``structured_output`` tool-call with no prose):
        ``reasoning_text is None``.
        """

        class MyOutput(BaseModel):
            answer: str

        client = LiteLLMClient(model="openai/gpt-4o-mini")
        tc = _make_tool_call_obj("tc-1", STRUCTURED_OUTPUT_TOOL_NAME, {"answer": "42"})
        mock_response = _make_openai_response(
            content=None,
            tool_calls=[tc],
            finish_reason="tool_calls",
        )
        with patch(
            "nanitics.infrastructure.llm.litellm.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
                output_schema=MyOutput,
            )
        assert result.reasoning_text is None
        assert result.parsed is not None


# --- Structured output ---


class TestLiteLLMClientStructuredOutput:
    async def test_structured_output_success(self) -> None:
        class MyOutput(BaseModel):
            answer: str
            confidence: float

        client = LiteLLMClient(model="openai/gpt-4o-mini")
        tc = _make_tool_call_obj(
            "tc-1",
            STRUCTURED_OUTPUT_TOOL_NAME,
            {"answer": "42", "confidence": 0.95},
        )
        mock_response = _make_openai_response(tool_calls=[tc], finish_reason="tool_calls")

        with patch(
            "nanitics.infrastructure.llm.litellm.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ) as mock_ac:
            result = await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
                output_schema=MyOutput,
            )

        assert isinstance(result.parsed, MyOutput)
        assert result.parsed.answer == "42"
        assert result.parsed.confidence == 0.95
        assert result.tool_calls == []
        assert result.content is not None
        assert '"answer"' in result.content
        call_kwargs = mock_ac.call_args.kwargs
        assert call_kwargs["tool_choice"] == {
            "type": "function",
            "function": {"name": STRUCTURED_OUTPUT_TOOL_NAME},
        }

    async def test_structured_output_validation_failure(self) -> None:
        class StrictOutput(BaseModel):
            count: int

        client = LiteLLMClient(model="openai/gpt-4o-mini")
        tc = _make_tool_call_obj(
            "tc-1",
            STRUCTURED_OUTPUT_TOOL_NAME,
            {"count": "not_a_number"},
        )
        mock_response = _make_openai_response(tool_calls=[tc], finish_reason="tool_calls")

        with (
            patch(
                "nanitics.infrastructure.llm.litellm.litellm.acompletion",
                new=AsyncMock(return_value=mock_response),
            ),
            pytest.raises(LLMSchemaViolationError),
        ):
            await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
                output_schema=StrictOutput,
            )

    async def test_structured_output_disables_streaming(self) -> None:
        """When output_schema is set, the client uses the non-streaming path even if on_token is provided."""

        class MyOutput(BaseModel):
            answer: str

        client = LiteLLMClient(model="openai/gpt-4o-mini")
        tc = _make_tool_call_obj("tc-1", STRUCTURED_OUTPUT_TOOL_NAME, {"answer": "42"})
        mock_response = _make_openai_response(tool_calls=[tc], finish_reason="tool_calls")
        tokens: list[str] = []

        with patch(
            "nanitics.infrastructure.llm.litellm.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ) as mock_ac:
            result = await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
                output_schema=MyOutput,
                on_token=tokens.append,
            )

        assert tokens == []
        assert isinstance(result.parsed, MyOutput)
        assert "stream" not in mock_ac.call_args.kwargs


# --- Error mapping ---


class TestLiteLLMClientErrors:
    async def _patch_and_raise(self, exc: Exception) -> Any:
        return patch(
            "nanitics.infrastructure.llm.litellm.litellm.acompletion",
            new=AsyncMock(side_effect=exc),
        )

    async def test_rate_limit_error_with_retry_after(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        resp = _make_httpx_response(429, headers={"retry-after": "30"})
        exc = litellm.RateLimitError(
            message="rate limited",
            llm_provider="openai",
            model="gpt-4o-mini",
            response=resp,
        )

        with (
            await self._patch_and_raise(exc),
            pytest.raises(LLMRateLimitError) as exc_info,
        ):
            await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
            )
        assert exc_info.value.retry_after == 30.0

    async def test_rate_limit_error_without_retry_after(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        resp = _make_httpx_response(429)
        exc = litellm.RateLimitError(
            message="rate limited",
            llm_provider="openai",
            model="gpt-4o-mini",
            response=resp,
        )

        with (
            await self._patch_and_raise(exc),
            pytest.raises(LLMRateLimitError) as exc_info,
        ):
            await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
            )
        assert exc_info.value.retry_after is None

    async def test_rate_limit_error_without_response(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        exc = litellm.RateLimitError(
            message="rate limited",
            llm_provider="openai",
            model="gpt-4o-mini",
            response=None,
        )

        with (
            await self._patch_and_raise(exc),
            pytest.raises(LLMRateLimitError) as exc_info,
        ):
            await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
            )
        assert exc_info.value.retry_after is None

    async def test_rate_limit_error_with_non_numeric_retry_after(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        resp = _make_httpx_response(429, headers={"retry-after": "not-a-number"})
        exc = litellm.RateLimitError(
            message="rate limited",
            llm_provider="openai",
            model="gpt-4o-mini",
            response=resp,
        )
        with (
            await self._patch_and_raise(exc),
            pytest.raises(LLMRateLimitError) as exc_info,
        ):
            await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
            )
        assert exc_info.value.retry_after is None

    async def test_context_window_exceeded_error(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        resp = _make_httpx_response(400)
        exc = litellm.ContextWindowExceededError(
            message="context length exceeded",
            model="gpt-4o-mini",
            llm_provider="openai",
            response=resp,
        )

        with (
            await self._patch_and_raise(exc),
            pytest.raises(LLMContextLengthError),
        ):
            await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
            )

    async def test_bad_request_with_overflow_like_message_is_provider_error(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        resp = _make_httpx_response(400)
        exc = litellm.BadRequestError(
            message="Request exceeds the token limit for this model.",
            model="gpt-4o-mini",
            llm_provider="openai",
            response=resp,
        )

        with (
            await self._patch_and_raise(exc),
            pytest.raises(LLMProviderError) as exc_info,
        ):
            await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
            )
        assert exc_info.value.provider == "litellm"
        assert exc_info.value.status_code == 400

    async def test_bad_request_non_context(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        resp = _make_httpx_response(400)
        exc = litellm.BadRequestError(
            message="invalid model id",
            model="gpt-4o-mini",
            llm_provider="openai",
            response=resp,
        )

        with (
            await self._patch_and_raise(exc),
            pytest.raises(LLMProviderError) as exc_info,
        ):
            await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
            )
        assert exc_info.value.provider == "litellm"
        assert exc_info.value.status_code == 400

    async def test_authentication_error(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        resp = _make_httpx_response(401)
        exc = litellm.AuthenticationError(
            message="invalid api key",
            llm_provider="openai",
            model="gpt-4o-mini",
            response=resp,
        )

        with (
            await self._patch_and_raise(exc),
            pytest.raises(LLMAuthenticationError) as exc_info,
        ):
            await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
            )
        assert exc_info.value.provider == "litellm"
        assert exc_info.value.status_code == 401

    async def test_permission_denied_error(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        resp = _make_httpx_response(403)
        exc = litellm.PermissionDeniedError(
            message="forbidden",
            llm_provider="openai",
            model="gpt-4o-mini",
            response=resp,
        )

        with (
            await self._patch_and_raise(exc),
            pytest.raises(LLMAuthenticationError) as exc_info,
        ):
            await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
            )
        assert exc_info.value.status_code == 403

    async def test_not_found_error(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        resp = _make_httpx_response(404)
        exc = litellm.NotFoundError(
            message="model not found",
            model="gpt-4o-mini",
            llm_provider="openai",
            response=resp,
        )

        with (
            await self._patch_and_raise(exc),
            pytest.raises(LLMProviderError) as exc_info,
        ):
            await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
            )
        assert exc_info.value.status_code == 404

    async def test_unprocessable_entity_error(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        resp = _make_httpx_response(422)
        exc = litellm.UnprocessableEntityError(
            message="unprocessable",
            model="gpt-4o-mini",
            llm_provider="openai",
            response=resp,
        )

        with (
            await self._patch_and_raise(exc),
            pytest.raises(LLMProviderError) as exc_info,
        ):
            await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
            )
        assert exc_info.value.status_code == 422

    async def test_internal_server_error(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        resp = _make_httpx_response(500)
        exc = litellm.InternalServerError(
            message="server down",
            llm_provider="openai",
            model="gpt-4o-mini",
            response=resp,
        )

        with (
            await self._patch_and_raise(exc),
            pytest.raises(LLMProviderError) as exc_info,
        ):
            await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
            )
        assert exc_info.value.status_code == 500

    async def test_api_connection_error(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        exc = litellm.APIConnectionError(
            message="connection error",
            llm_provider="openai",
            model="gpt-4o-mini",
        )

        with (
            await self._patch_and_raise(exc),
            pytest.raises(LLMProviderError) as exc_info,
        ):
            await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
            )
        assert exc_info.value.provider == "litellm"
        assert exc_info.value.status_code is None

    async def test_generic_api_error(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        exc = litellm.APIError(
            status_code=502,
            message="bad gateway",
            llm_provider="openai",
            model="gpt-4o-mini",
        )

        with (
            await self._patch_and_raise(exc),
            pytest.raises(LLMProviderError) as exc_info,
        ):
            await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
            )
        assert exc_info.value.provider == "litellm"
        assert exc_info.value.status_code == 502
        assert type(exc_info.value) is LLMProviderError

    async def test_timeout_mapping(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o-mini", request_timeout=0.1)

        async def _slow_acompletion(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(10)  # pragma: no cover

        with (
            patch(
                "nanitics.infrastructure.llm.litellm.litellm.acompletion",
                new=AsyncMock(side_effect=_slow_acompletion),
            ),
            pytest.raises(LLMProviderError, match="timed out") as exc_info,
        ):
            await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
            )
        assert exc_info.value.provider == "litellm"

    async def test_request_timeout_none_disables_deadline(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o-mini", request_timeout=None)
        mock_response = _make_openai_response(content="hi")

        with patch(
            "nanitics.infrastructure.llm.litellm.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
            )
        assert result.content == "hi"


# --- Streaming ---


class TestLiteLLMClientStreaming:
    async def test_streaming_calls_on_token(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        tokens: list[str] = []

        chunks = [
            _make_stream_chunk(content="Hel"),
            _make_stream_chunk(content="lo"),
            _make_stream_chunk(finish_reason="stop", prompt_tokens=10, completion_tokens=5),
        ]

        with patch(
            "nanitics.infrastructure.llm.litellm.litellm.acompletion",
            new=AsyncMock(return_value=_mock_stream(chunks)),
        ) as mock_ac:
            result = await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
                on_token=tokens.append,
            )

        assert tokens == ["Hel", "lo"]
        assert result.content == "Hello"
        assert result.stop_reason == "end_turn"
        call_kwargs = mock_ac.call_args.kwargs
        assert call_kwargs["stream"] is True
        assert call_kwargs["stream_options"] == {"include_usage": True}
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    async def test_streaming_with_tool_calls(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        tokens: list[str] = []

        chunks = [
            _make_stream_chunk(tool_calls=[_make_tool_call_delta(0, id="tc-1", name="search", arguments='{"q":')]),
            _make_stream_chunk(tool_calls=[_make_tool_call_delta(0, arguments='"test"}')]),
            _make_stream_chunk(finish_reason="tool_calls", prompt_tokens=5, completion_tokens=3),
        ]

        tools = [ToolSchema(name="search", description="Search", parameters={})]

        with patch(
            "nanitics.infrastructure.llm.litellm.litellm.acompletion",
            new=AsyncMock(return_value=_mock_stream(chunks)),
        ):
            result = await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="go")],
                tools=tools,
                on_token=tokens.append,
            )

        assert tokens == []
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "tc-1"
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].arguments == {"q": "test"}
        assert result.stop_reason == "tool_use"

    async def test_streaming_skips_empty_choices(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        tokens: list[str] = []

        chunks = [
            _make_stream_chunk(include_choice=False),
            _make_stream_chunk(content="Hello"),
            _make_stream_chunk(finish_reason="stop", prompt_tokens=5, completion_tokens=2),
        ]

        with patch(
            "nanitics.infrastructure.llm.litellm.litellm.acompletion",
            new=AsyncMock(return_value=_mock_stream(chunks)),
        ):
            result = await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
                on_token=tokens.append,
            )

        assert tokens == ["Hello"]
        assert result.content == "Hello"

    async def test_streaming_tool_call_delta_without_function(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        chunks = [
            _make_stream_chunk(tool_calls=[_make_tool_call_delta(0, id="tc-1", name="noop", arguments="{}")]),
            _make_stream_chunk(
                tool_calls=[_make_tool_call_delta(0)],
            ),
            _make_stream_chunk(finish_reason="tool_calls", prompt_tokens=5, completion_tokens=2),
        ]
        tools = [ToolSchema(name="noop", description="No args", parameters={})]

        with patch(
            "nanitics.infrastructure.llm.litellm.litellm.acompletion",
            new=AsyncMock(return_value=_mock_stream(chunks)),
        ):
            result = await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="go")],
                tools=tools,
                on_token=lambda t: None,
            )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "noop"
        assert result.tool_calls[0].arguments == {}

    async def test_streaming_tool_call_empty_arguments_string(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        chunks = [
            _make_stream_chunk(tool_calls=[_make_tool_call_delta(0, id="tc-1", name="noop", arguments="")]),
            _make_stream_chunk(finish_reason="tool_calls", prompt_tokens=5, completion_tokens=2),
        ]
        tools = [ToolSchema(name="noop", description="No args", parameters={})]

        with patch(
            "nanitics.infrastructure.llm.litellm.litellm.acompletion",
            new=AsyncMock(return_value=_mock_stream(chunks)),
        ):
            result = await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="go")],
                tools=tools,
                on_token=lambda t: None,
            )

        assert result.tool_calls[0].arguments == {}

    async def test_streaming_reasoning_text_from_prose_before_tool_call(self) -> None:
        """Streamed content tokens followed by tool-call deltas surface as
        ``reasoning_text``. Prose preceding a tool call is reasoning, not the
        final answer."""
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        tokens: list[str] = []

        chunks = [
            _make_stream_chunk(content="Let me "),
            _make_stream_chunk(content="search."),
            _make_stream_chunk(tool_calls=[_make_tool_call_delta(0, id="tc-1", name="search", arguments='{"q":"x"}')]),
            _make_stream_chunk(finish_reason="tool_calls", prompt_tokens=5, completion_tokens=3),
        ]

        tools = [ToolSchema(name="search", description="Search", parameters={})]

        with patch(
            "nanitics.infrastructure.llm.litellm.litellm.acompletion",
            new=AsyncMock(return_value=_mock_stream(chunks)),
        ):
            result = await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="search for x")],
                tools=tools,
                on_token=tokens.append,
            )

        assert tokens == ["Let me ", "search."]
        assert result.reasoning_text == "Let me search."
        assert result.content == "Let me search."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"


# --- Passthrough behaviour ---


class TestLiteLLMClientPassthrough:
    async def test_extra_kwargs_forwarded_to_acompletion(self) -> None:
        client = LiteLLMClient(
            model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
            extra_kwargs={"aws_region_name": "us-east-1"},
        )
        mock_response = _make_openai_response(content="ok")

        with patch(
            "nanitics.infrastructure.llm.litellm.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ) as mock_ac:
            await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
            )

        call_kwargs = mock_ac.call_args.kwargs
        assert call_kwargs["aws_region_name"] == "us-east-1"

    async def test_base_url_forwarded_as_api_base(self) -> None:
        client = LiteLLMClient(model="ollama/llama3", base_url="http://localhost:11434")
        mock_response = _make_openai_response(content="ok")

        with patch(
            "nanitics.infrastructure.llm.litellm.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ) as mock_ac:
            await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
            )

        assert mock_ac.call_args.kwargs["api_base"] == "http://localhost:11434"

    async def test_base_url_not_passed_when_none(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        mock_response = _make_openai_response(content="ok")

        with patch(
            "nanitics.infrastructure.llm.litellm.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ) as mock_ac:
            await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
            )

        assert "api_base" not in mock_ac.call_args.kwargs

    async def test_api_key_not_passed_when_none(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        mock_response = _make_openai_response(content="ok")

        with patch(
            "nanitics.infrastructure.llm.litellm.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ) as mock_ac:
            await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
            )

        assert "api_key" not in mock_ac.call_args.kwargs

    async def test_num_retries_is_zero(self) -> None:
        """LiteLLM's built-in retry layer is disabled — the SDK's RetryPolicy owns retries."""
        client = LiteLLMClient(model="openai/gpt-4o-mini")
        mock_response = _make_openai_response(content="ok")

        with patch(
            "nanitics.infrastructure.llm.litellm.litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ) as mock_ac:
            await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
            )

        assert mock_ac.call_args.kwargs["num_retries"] == 0

    async def test_model_string_passthrough_provider_prefixed(self) -> None:
        """Provider-prefixed model strings are forwarded verbatim to LiteLLM."""
        for model_str in [
            "openai/gpt-4o-mini",
            "anthropic/claude-haiku-4-5",
            "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
            "gemini/gemini-2.0-flash",
            "ollama/llama3",
        ]:
            client = LiteLLMClient(model=model_str)
            mock_response = _make_openai_response(content="ok")
            with patch(
                "nanitics.infrastructure.llm.litellm.litellm.acompletion",
                new=AsyncMock(return_value=mock_response),
            ) as mock_ac:
                result = await client.generate(
                    system_prompt="sys",
                    messages=[Message(role="user", content="hi")],
                )
            assert mock_ac.call_args.kwargs["model"] == model_str
            assert result.model == model_str


# --- Typed Error Subclass Mapping Tests (Phase: typed-llm-error-subclasses) ---


class TestExtractLitellmErrorType:
    def test_missing_body(self) -> None:
        exc = MagicMock(spec=[])
        assert _extract_litellm_error_type(exc) is None

    def test_body_is_none(self) -> None:
        exc = MagicMock()
        exc.body = None
        assert _extract_litellm_error_type(exc) is None

    def test_body_is_non_dict(self) -> None:
        exc = MagicMock()
        exc.body = "x"
        assert _extract_litellm_error_type(exc) is None

    def test_body_without_error(self) -> None:
        exc = MagicMock()
        exc.body = {"other": 1}
        assert _extract_litellm_error_type(exc) is None

    def test_body_with_non_dict_error(self) -> None:
        exc = MagicMock()
        exc.body = {"error": "x"}
        assert _extract_litellm_error_type(exc) is None

    def test_body_with_non_string_type(self) -> None:
        exc = MagicMock()
        exc.body = {"error": {"type": 42}}
        assert _extract_litellm_error_type(exc) is None

    def test_happy_path(self) -> None:
        exc = MagicMock()
        exc.body = {"error": {"type": "insufficient_quota"}}
        assert _extract_litellm_error_type(exc) == "insufficient_quota"


class TestExtractLitellmQuotaSignal:
    def test_no_body(self) -> None:
        exc = MagicMock()
        exc.body = None
        assert _extract_litellm_quota_signal(exc) is None

    def test_non_dict_body(self) -> None:
        exc = MagicMock()
        exc.body = "x"
        assert _extract_litellm_quota_signal(exc) is None

    def test_no_error_obj(self) -> None:
        exc = MagicMock()
        exc.body = {}
        assert _extract_litellm_quota_signal(exc) is None

    def test_non_dict_error_obj(self) -> None:
        exc = MagicMock()
        exc.body = {"error": "x"}
        assert _extract_litellm_quota_signal(exc) is None

    def test_insufficient_quota_type(self) -> None:
        exc = MagicMock()
        exc.body = {"error": {"type": "insufficient_quota"}}
        assert _extract_litellm_quota_signal(exc) == "insufficient_quota"

    def test_gemini_resource_exhausted_status(self) -> None:
        exc = MagicMock()
        exc.body = {"error": {"status": "RESOURCE_EXHAUSTED", "message": "x"}}
        assert _extract_litellm_quota_signal(exc) == "RESOURCE_EXHAUSTED"

    def test_other_error_type_returns_none(self) -> None:
        exc = MagicMock()
        exc.body = {"error": {"type": "rate_limit_exceeded"}}
        assert _extract_litellm_quota_signal(exc) is None

    def test_non_string_error_type(self) -> None:
        exc = MagicMock()
        exc.body = {"error": {"type": 42}}
        assert _extract_litellm_quota_signal(exc) is None

    def test_non_string_status(self) -> None:
        exc = MagicMock()
        exc.body = {"error": {"status": 42}}
        assert _extract_litellm_quota_signal(exc) is None

    def test_other_status_returns_none(self) -> None:
        exc = MagicMock()
        exc.body = {"error": {"status": "DEADLINE_EXCEEDED"}}
        assert _extract_litellm_quota_signal(exc) is None


class TestLiteLLMTypedErrorMapping:
    """Verify upstream LiteLLM exceptions map to the typed SDK subclasses."""

    def _make_client(self) -> LiteLLMClient:
        return LiteLLMClient(model="openai/gpt-4o-mini")

    async def _raise_via_client(self, exc: Exception) -> None:
        client = self._make_client()
        with patch(
            "nanitics.infrastructure.llm.litellm.litellm.acompletion",
            new=AsyncMock(side_effect=exc),
        ):
            await client.generate(
                system_prompt="sys",
                messages=[Message(role="user", content="hi")],
            )

    async def test_rate_limit_insufficient_quota_routes_to_quota(self) -> None:
        resp = _make_httpx_response(429)
        exc = litellm.RateLimitError(
            message="quota",
            llm_provider="openai",
            model="gpt-4o-mini",
            response=resp,
        )
        exc.body = {"error": {"type": "insufficient_quota", "message": "x"}}
        with pytest.raises(LLMQuotaExhaustedError) as exc_info:
            await self._raise_via_client(exc)
        assert exc_info.value.provider == "litellm"
        assert exc_info.value.status_code == 429
        assert exc_info.value.provider_error_type == "insufficient_quota"

    async def test_rate_limit_gemini_resource_exhausted_routes_to_quota(self) -> None:
        resp = _make_httpx_response(429)
        exc = litellm.RateLimitError(
            message="quota",
            llm_provider="gemini",
            model="gemini-2.0-flash",
            response=resp,
        )
        exc.body = {"error": {"status": "RESOURCE_EXHAUSTED", "message": "x"}}
        with pytest.raises(LLMQuotaExhaustedError) as exc_info:
            await self._raise_via_client(exc)
        assert exc_info.value.provider == "litellm"
        assert exc_info.value.status_code == 429
        assert exc_info.value.provider_error_type == "RESOURCE_EXHAUSTED"

    async def test_rate_limit_sparse_body_routes_to_rate_limit(self) -> None:
        # Regression: a 429 with no positive quota signal preserves the
        # existing LLMRateLimitError (RETRYABLE) path.
        resp = _make_httpx_response(429, headers={"retry-after": "7"})
        exc = litellm.RateLimitError(
            message="slow",
            llm_provider="openai",
            model="gpt-4o-mini",
            response=resp,
        )
        with pytest.raises(LLMRateLimitError) as exc_info:
            await self._raise_via_client(exc)
        assert exc_info.value.retry_after == 7.0

    async def test_api_error_529_routes_to_overloaded(self) -> None:
        exc = litellm.APIError(
            status_code=529,
            message="overloaded",
            llm_provider="anthropic",
            model="claude-haiku-4-5",
        )
        with pytest.raises(LLMOverloadedError) as exc_info:
            await self._raise_via_client(exc)
        assert exc_info.value.provider == "litellm"
        assert exc_info.value.status_code == 529

    async def test_api_error_overloaded_body_type_routes_to_overloaded(self) -> None:
        exc = litellm.APIError(
            status_code=503,
            message="overloaded",
            llm_provider="openai",
            model="gpt-4o-mini",
        )
        exc.body = {"error": {"type": "overloaded_error", "message": "x"}}
        with pytest.raises(LLMOverloadedError) as exc_info:
            await self._raise_via_client(exc)
        assert exc_info.value.provider == "litellm"
        assert exc_info.value.status_code == 503
        assert exc_info.value.provider_error_type == "overloaded_error"

    async def test_api_error_other_routes_to_provider_error_with_type(self) -> None:
        exc = litellm.APIError(
            status_code=502,
            message="bad gw",
            llm_provider="openai",
            model="gpt-4o-mini",
        )
        exc.body = {"error": {"type": "server_error"}}
        with pytest.raises(LLMProviderError) as exc_info:
            await self._raise_via_client(exc)
        assert type(exc_info.value) is LLMProviderError
        assert exc_info.value.status_code == 502
        assert exc_info.value.provider_error_type == "server_error"

    async def test_bad_request_populates_provider_error_type(self) -> None:
        resp = _make_httpx_response(400)
        exc = litellm.BadRequestError(
            message="invalid",
            model="gpt-4o-mini",
            llm_provider="openai",
            response=resp,
        )
        exc.body = {"error": {"type": "invalid_request_error"}}
        with pytest.raises(LLMProviderError) as exc_info:
            await self._raise_via_client(exc)
        assert type(exc_info.value) is LLMProviderError
        assert exc_info.value.provider_error_type == "invalid_request_error"

    async def test_not_found_populates_provider_error_type(self) -> None:
        resp = _make_httpx_response(404)
        exc = litellm.NotFoundError(
            message="not found",
            model="gpt-4o-mini",
            llm_provider="openai",
            response=resp,
        )
        exc.body = {"error": {"type": "not_found"}}
        with pytest.raises(LLMProviderError) as exc_info:
            await self._raise_via_client(exc)
        assert exc_info.value.provider_error_type == "not_found"

    async def test_unprocessable_populates_provider_error_type(self) -> None:
        resp = _make_httpx_response(422)
        exc = litellm.UnprocessableEntityError(
            message="bad input",
            model="gpt-4o-mini",
            llm_provider="openai",
            response=resp,
        )
        with pytest.raises(LLMProviderError) as exc_info:
            await self._raise_via_client(exc)
        # Empty body — provider_error_type is None
        assert exc_info.value.provider_error_type is None
        assert exc_info.value.status_code == 422

    async def test_internal_server_populates_provider_error_type(self) -> None:
        resp = _make_httpx_response(500)
        exc = litellm.InternalServerError(
            message="server down",
            llm_provider="openai",
            model="gpt-4o-mini",
            response=resp,
        )
        exc.body = {"error": {"type": "server_error"}}
        with pytest.raises(LLMProviderError) as exc_info:
            await self._raise_via_client(exc)
        assert exc_info.value.provider_error_type == "server_error"
        assert exc_info.value.status_code == 500
