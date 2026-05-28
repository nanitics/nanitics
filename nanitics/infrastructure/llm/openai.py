from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel

from nanitics.infrastructure.errors import (
    LLMAuthenticationError,
    LLMContextLengthError,
    LLMOverloadedError,
    LLMProviderError,
    LLMQuotaExhaustedError,
    LLMRateLimitError,
    LLMSchemaViolationError,
)
from nanitics.infrastructure.llm._openai_format import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    _build_openai_kwargs,
    _from_openai_response,
    _map_stop_reason,
    _to_openai_messages,
)
from nanitics.infrastructure.llm._openai_profiles import profile_for
from nanitics.infrastructure.llm.protocol import (
    LLMResponse,
    Message,
    SystemPromptSection,
    ToolCall,
    ToolSchema,
)
from nanitics.infrastructure.observability.events import Usage

try:
    import openai
except ImportError as _err:  # pragma: no cover
    raise ImportError(
        "OpenAILLMClient requires the 'openai' package, which ships with nanitics. "
        "Reinstall with: pip install --force-reinstall nanitics"
    ) from _err

__all__ = ["OpenAILLMClient"]


def _extract_openai_error_type(e: Exception) -> str | None:
    """Return ``body.error.type`` from an OpenAI SDK exception, defensively.

    OpenAI error responses follow the shape
    ``{"error": {"type": str, "code": str, "message": str, ...}}``. The
    ``body`` attribute can be ``None``, a non-dict, or partially populated.
    Returns ``None`` for anything other than a well-formed ``error.type``
    string.
    """
    body = getattr(e, "body", None) or {}
    if not isinstance(body, dict):
        return None
    err_obj = body.get("error")
    if not isinstance(err_obj, dict):
        return None
    err_type = err_obj.get("type")
    if not isinstance(err_type, str):
        return None
    return err_type


class OpenAILLMClient:
    """LLM client for the OpenAI Chat Completions API.

    Handles message formatting, tool-call serialization, structured output
    via the tool-use pattern, multimodal input, streaming, and maps API
    errors to the SDK error hierarchy.

    Structured output uses a forced tool call (``structured_output``)
    rather than OpenAI's native ``response_format={"type": "json_schema"}``
    so that agents written against Anthropic or Mistral behave identically
    when switched to OpenAI without prompt retuning.

    Client-side token counting with ``tiktoken`` is not required — this
    client only reports usage returned by the API. Install the optional
    ``nanitics[openai-tokenizer]`` extra if you want to count tokens
    yourself; the SDK's own context-management path uses
    ``EstimateTokenCounter`` which is provider-agnostic.

    Per-model request-shape variance (e.g. ``max_tokens`` vs.
    ``max_completion_tokens``) is resolved from
    :mod:`nanitics.infrastructure.llm._openai_profiles`, which is the
    authoritative source for the parameter-name mapping.

    Args:
        model: OpenAI model identifier (e.g., ``"gpt-4o-mini"``).
        api_key: API key. Falls back to the ``OPENAI_API_KEY`` environment
            variable via the OpenAI SDK's own resolution.
        base_url: Custom API base URL. Points ``OpenAILLMClient`` at any
            OpenAI-compatible endpoint — Azure, proxies, Ollama, vLLM,
            LM Studio, llama.cpp's server, or any other compatible service.
            See ``docs/guides/local-llms.md`` for recipes.
        max_tokens: Maximum tokens to generate per call (default: 16,384).
        request_timeout: Deadline in seconds for each ``generate()`` call,
            or ``None`` to disable.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 16_384,
        request_timeout: float | None = 300.0,
    ) -> None:
        if api_key is None and not os.environ.get("OPENAI_API_KEY"):
            raise LLMProviderError(
                "OPENAI_API_KEY env var is not set and no api_key= was provided. "
                "Set OPENAI_API_KEY or pass api_key= explicitly "
                "(local OpenAI-compatible endpoints like Ollama accept any non-empty string).",
                provider="openai",
            )
        self._model = model
        self._max_tokens = max_tokens
        self._request_timeout = request_timeout
        self._profile = profile_for(model)
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
        )

    @property
    def model(self) -> str | None:
        return self._model

    async def generate(
        self,
        *,
        system_prompt: str,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        output_schema: type[BaseModel] | None = None,
        on_token: Callable[[str], None] | None = None,
        system_prompt_sections: list[SystemPromptSection] | None = None,
    ) -> LLMResponse:
        if output_schema is not None and tools is not None:
            raise ValueError("Cannot provide both output_schema and tools — they are mutually exclusive")

        request_tools: list[ToolSchema] | None
        tool_choice: Any
        if output_schema is not None:
            request_tools = [
                ToolSchema(
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    description=f"Return structured output matching the {output_schema.__name__} schema.",
                    parameters=output_schema.model_json_schema(),
                )
            ]
            tool_choice = {
                "type": "function",
                "function": {"name": STRUCTURED_OUTPUT_TOOL_NAME},
            }
        elif tools is not None:
            request_tools = tools
            tool_choice = "auto"
        else:
            request_tools = None
            tool_choice = None

        kwargs = _build_openai_kwargs(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=_to_openai_messages(system_prompt, messages),
            tools=request_tools,
            tool_choice=tool_choice,
            profile=self._profile,
        )

        use_streaming = on_token is not None and output_schema is None

        async def _api_call() -> LLMResponse:
            try:
                if use_streaming:
                    assert on_token is not None  # guaranteed by use_streaming
                    return await self._stream_request(kwargs, on_token)
                return await self._standard_request(kwargs)
            except openai.RateLimitError as e:
                # OpenAI 429 with ``error.type == "insufficient_quota"`` is a
                # billing-state condition, not a transient rate-limit —
                # retrying never recovers within the budget window. Route
                # to ``LLMQuotaExhaustedError`` (FATAL); other 429 shapes
                # preserve the existing ``LLMRateLimitError`` retry path.
                err_type = _extract_openai_error_type(e)
                if err_type == "insufficient_quota":
                    raise LLMQuotaExhaustedError(
                        str(e),
                        status_code=e.status_code,
                        provider="openai",
                        provider_error_type="insufficient_quota",
                    ) from e
                retry_after: float | None = None
                response = getattr(e, "response", None)
                if response is not None:
                    raw = response.headers.get("retry-after")
                    if raw is not None:
                        with contextlib.suppress(ValueError, TypeError):
                            retry_after = float(raw)
                raise LLMRateLimitError(str(e), retry_after=retry_after) from e
            except openai.BadRequestError as e:
                # Structured-first classification: OpenAI returns a top-level
                # ``code`` in ``BadRequestError.body``; ``"context_length_exceeded"``
                # is the sole overflow signal. Anything else (including
                # ``"unsupported_parameter"``) is a provider error, not an
                # overflow — substring heuristics over the message (e.g.
                # matching the word "token") misclassified parameter-shape
                # failures as context overflow.
                code = getattr(e, "code", None)
                if code == "context_length_exceeded":
                    raise LLMContextLengthError(str(e)) from e
                raise LLMProviderError(
                    str(e),
                    status_code=e.status_code,
                    provider="openai",
                    provider_error_type=_extract_openai_error_type(e),
                ) from e
            except openai.AuthenticationError as e:
                raise LLMAuthenticationError(
                    str(e),
                    status_code=e.status_code,
                    provider="openai",
                    provider_error_type=_extract_openai_error_type(e),
                ) from e
            except openai.APIStatusError as e:
                err_type = _extract_openai_error_type(e)
                if err_type == "overloaded_error":
                    raise LLMOverloadedError(
                        str(e),
                        status_code=e.status_code,
                        provider="openai",
                        provider_error_type="overloaded_error",
                    ) from e
                raise LLMProviderError(
                    str(e),
                    status_code=e.status_code,
                    provider="openai",
                    provider_error_type=err_type,
                ) from e
            except openai.APIConnectionError as e:
                raise LLMProviderError(str(e), provider="openai") from e

        llm_response = await self._with_deadline(_api_call())

        if output_schema is not None:
            for tc in llm_response.tool_calls:
                if tc.name == STRUCTURED_OUTPUT_TOOL_NAME:
                    try:
                        parsed = output_schema.model_validate(tc.arguments)
                    except Exception as e:
                        raise LLMSchemaViolationError(
                            f"Response did not match schema: {e}",
                            expected_schema=json.dumps(output_schema.model_json_schema()),
                            received=json.dumps(tc.arguments),
                        ) from e
                    llm_response = llm_response.model_copy(
                        update={
                            "parsed": parsed,
                            "content": json.dumps(tc.arguments),
                            "tool_calls": [],
                        }
                    )
                    break

        return llm_response

    async def _with_deadline(self, coro: Awaitable[LLMResponse]) -> LLMResponse:
        if self._request_timeout is None:
            return await coro
        try:
            async with asyncio.timeout(self._request_timeout):
                return await coro
        except TimeoutError:
            raise LLMProviderError(
                f"Request timed out after {self._request_timeout}s",
                provider="openai",
            ) from None

    async def _standard_request(self, kwargs: dict[str, Any]) -> LLMResponse:
        response = await self._client.chat.completions.create(**kwargs)
        return _from_openai_response(response, self._model)

    async def _stream_request(self, kwargs: dict[str, Any], on_token: Callable[[str], None]) -> LLMResponse:
        stream_kwargs = {
            **kwargs,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        content_parts: list[str] = []
        tool_calls_accum: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        input_tokens = 0
        output_tokens = 0

        stream = await self._client.chat.completions.create(**stream_kwargs)
        async for chunk in stream:
            if chunk.usage is not None:
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens

            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            if delta.content:
                content_parts.append(delta.content)
                on_token(delta.content)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_accum:
                        tool_calls_accum[idx] = {
                            "id": tc_delta.id or "",
                            "function": {"name": "", "arguments": ""},
                        }
                    entry = tool_calls_accum[idx]
                    if tc_delta.id:
                        entry["id"] = tc_delta.id
                    func = tc_delta.function
                    if func is not None:
                        if func.name:
                            entry["function"]["name"] = func.name
                        if func.arguments:
                            entry["function"]["arguments"] += func.arguments

            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason

        content = "".join(content_parts) if content_parts else None
        tool_calls: list[ToolCall] = []
        for _idx in sorted(tool_calls_accum.keys()):
            tc_data = tool_calls_accum[_idx]
            arguments_raw = tc_data["function"]["arguments"]
            if isinstance(arguments_raw, str) and arguments_raw:
                arguments = json.loads(arguments_raw)
            else:
                arguments = {}
            tool_calls.append(
                ToolCall(
                    id=tc_data["id"],
                    name=tc_data["function"]["name"],
                    arguments=arguments,
                )
            )

        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
        )

        # Extract reasoning_text on the tool-use path; empty content on
        # the tool-use path maps to None. On the final-answer path,
        # reasoning_text is None — the full response is in content.
        reasoning_text: str | None = None
        if tool_calls and isinstance(content, str) and content:
            reasoning_text = content

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            model=self._model,
            stop_reason=_map_stop_reason(finish_reason),
            reasoning_text=reasoning_text,
        )
