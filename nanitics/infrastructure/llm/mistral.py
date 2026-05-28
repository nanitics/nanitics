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
from nanitics.infrastructure.llm.protocol import (
    ContentBlock,
    LLMResponse,
    Message,
    SystemPromptSection,
    TextContentBlock,
    ToolCall,
    ToolSchema,
)
from nanitics.infrastructure.observability.events import Usage

try:
    import httpx
except ImportError as _err:  # pragma: no cover
    raise ImportError("MistralLLMClient requires the 'mistral' extra: pip install nanitics[mistral]") from _err

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
STRUCTURED_OUTPUT_TOOL_NAME = "structured_output"


def _content_block_to_mistral(block: ContentBlock) -> dict[str, Any]:
    """Convert an SDK ContentBlock to Mistral's format."""
    if isinstance(block, TextContentBlock):
        return {"type": "text", "text": block.text}
    # ImageContentBlock
    if block.data.startswith(("http://", "https://")):
        url = block.data
    else:
        url = f"data:{block.media_type};base64,{block.data}"
    return {"type": "image_url", "image_url": {"url": url}}


def _to_mistral_messages(system_prompt: str, messages: list[Message]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    for msg in messages:
        if msg.role == "user":
            if isinstance(msg.content, list):
                content: Any = [_content_block_to_mistral(b) for b in msg.content]
            else:
                content = msg.content or ""
            result.append({"role": "user", "content": content})

        elif msg.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant"}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
                if msg.content:
                    entry["content"] = msg.content
            else:
                # Defensive guard: Mistral rejects empty assistant content
                entry["content"] = msg.content or " "
            result.append(entry)

        elif msg.role == "tool_result":
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content or "",
                }
            )

    return result


def _to_mistral_tools(tools: list[ToolSchema]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def _map_stop_reason(finish_reason: str | None) -> str:
    if finish_reason == "tool_calls":
        return "tool_use"
    if finish_reason == "length":
        return "max_tokens"
    return "end_turn"


def _parse_tool_calls(raw_tool_calls: list[dict[str, Any]]) -> list[ToolCall]:
    result: list[ToolCall] = []
    for tc in raw_tool_calls:
        func = tc.get("function", {})
        arguments = func.get("arguments", "{}")
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        result.append(
            ToolCall(
                id=tc["id"],
                name=func["name"],
                arguments=arguments,
            )
        )
    return result


def _parse_response(data: dict[str, Any], model: str) -> LLMResponse:
    choice = data["choices"][0]
    message = choice["message"]

    content = message.get("content")
    raw_tool_calls = message.get("tool_calls") or []
    tool_calls = _parse_tool_calls(raw_tool_calls)

    usage_data = data.get("usage", {})
    input_tokens = usage_data.get("prompt_tokens", 0)
    output_tokens = usage_data.get("completion_tokens", 0)

    usage = Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
    )

    # Extract reasoning_text: on the tool-use path, ``message.content``
    # carries the prose-before-tool-call. Empty-string content with
    # ``tool_calls`` present maps to ``None`` (empty is not reasoning).
    # On the final-answer path, reasoning_text is ``None``.
    reasoning_text: str | None = None
    if tool_calls and isinstance(content, str) and content:
        reasoning_text = content

    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        usage=usage,
        model=model,
        stop_reason=_map_stop_reason(choice.get("finish_reason")),
        reasoning_text=reasoning_text,
    )


def _extract_mistral_error_code(body: str) -> str | None:
    """Return ``code`` from a Mistral error body, or ``None``.

    Mistral follows OpenAI's error schema and returns
    ``{"code": str, "message": str, ...}``. The body is the raw HTTP
    response text — parse defensively, returning ``None`` for non-JSON,
    non-dict, or missing/non-string ``code``.
    """
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    raw_code = parsed.get("code")
    if not isinstance(raw_code, str):
        return None
    return raw_code


def _handle_error(exc: httpx.HTTPStatusError) -> None:
    status = exc.response.status_code
    body = exc.response.text
    code = _extract_mistral_error_code(body)

    if status == 429:
        # Mistral 429 with a structured quota code routes to FATAL
        # ``LLMQuotaExhaustedError`` — retrying against an exhausted
        # quota never recovers within the budget window. Other 429
        # shapes preserve the existing rate-limit retry path.
        if code in {"insufficient_quota", "quota_exceeded"}:
            raise LLMQuotaExhaustedError(
                body,
                status_code=429,
                provider="mistral",
                provider_error_type=code,
            ) from exc
        retry_after: float | None = None
        raw = exc.response.headers.get("retry-after")
        if raw is not None:
            with contextlib.suppress(ValueError, TypeError):
                retry_after = float(raw)
        raise LLMRateLimitError(body, retry_after=retry_after) from exc

    if status == 401:
        raise LLMAuthenticationError(
            body,
            status_code=401,
            provider="mistral",
            provider_error_type=code,
        ) from exc

    if status == 400:
        # Structured-first classification: Mistral follows OpenAI's error
        # schema, returning ``{"code": "context_length_exceeded", ...}`` for
        # overflow. Parse the body as JSON and read the structured code;
        # fall back to ``LLMProviderError`` for any non-JSON or other code.
        if code == "context_length_exceeded":
            raise LLMContextLengthError(body) from exc
        raise LLMProviderError(
            body,
            status_code=status,
            provider="mistral",
            provider_error_type=code,
        ) from exc

    if status >= 500:
        # Mistral does not document a dedicated overload code as of
        # writing; ``code == "overloaded"`` is the positive-signal
        # predicate. Absence of the signal leaves the existing
        # ``LLMProviderError`` raise in place.
        if code == "overloaded":
            raise LLMOverloadedError(
                body,
                status_code=status,
                provider="mistral",
                provider_error_type="overloaded",
            ) from exc
        raise LLMProviderError(
            body,
            status_code=status,
            provider="mistral",
            provider_error_type=code,
        ) from exc

    raise LLMProviderError(
        body,
        status_code=status,
        provider="mistral",
        provider_error_type=code,
    ) from exc


class MistralLLMClient:
    """LLM client for the Mistral La Plateforme API.

    Uses httpx for HTTP communication. Supports text generation, tool calling,
    structured output via tool-use pattern, and streaming.

    Args:
        model: Mistral model identifier (e.g., ``"mistral-small-latest"``).
        api_key: API key. Falls back to the ``MISTRAL_API_KEY`` environment
            variable if not provided.
        max_tokens: Maximum tokens to generate per call (default: 16,384).
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 16_384,
        request_timeout: float | None = 300.0,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._request_timeout = request_timeout
        resolved_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not resolved_key:
            raise LLMProviderError(
                "MISTRAL_API_KEY env var is not set and no api_key= was provided. "
                "Set MISTRAL_API_KEY or pass api_key= explicitly.",
                provider="mistral",
            )
        self._api_key = resolved_key
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(120.0, connect=10.0),
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

        body: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": _to_mistral_messages(system_prompt, messages),
        }

        if output_schema is not None:
            schema = output_schema.model_json_schema()
            tool_def = {
                "type": "function",
                "function": {
                    "name": STRUCTURED_OUTPUT_TOOL_NAME,
                    "description": f"Return structured output matching the {output_schema.__name__} schema.",
                    "parameters": schema,
                },
            }
            body["tools"] = [tool_def]
            body["tool_choice"] = {
                "type": "function",
                "function": {"name": STRUCTURED_OUTPUT_TOOL_NAME},
            }
        elif tools is not None:
            body["tools"] = _to_mistral_tools(tools)
            body["tool_choice"] = "auto"

        use_streaming = on_token is not None and output_schema is None
        if use_streaming:
            body["stream"] = True

        async def _api_call() -> LLMResponse:
            try:
                if use_streaming:
                    assert on_token is not None  # guaranteed by use_streaming condition
                    return await self._stream_request(body, on_token)
                return await self._standard_request(body)
            except httpx.HTTPStatusError as exc:
                _handle_error(exc)
                raise  # pragma: no cover  # unreachable, _handle_error always raises
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                raise LLMProviderError(str(exc), provider="mistral") from exc

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
            return await asyncio.wait_for(coro, timeout=self._request_timeout)
        except TimeoutError:
            raise LLMProviderError(
                f"Request timed out after {self._request_timeout}s",
                provider="mistral",
            ) from None

    async def _standard_request(self, body: dict[str, Any]) -> LLMResponse:
        response = await self._client.post(MISTRAL_API_URL, json=body)
        response.raise_for_status()
        return _parse_response(response.json(), self._model)

    async def _stream_request(self, body: dict[str, Any], on_token: Callable[[str], None]) -> LLMResponse:
        content_parts: list[str] = []
        tool_calls_accum: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage_data: dict[str, Any] = {}

        async with self._client.stream("POST", MISTRAL_API_URL, json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                chunk = json.loads(payload)

                if "usage" in chunk:
                    usage_data = chunk["usage"]

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})

                if delta.get("content"):
                    text = delta["content"]
                    content_parts.append(text)
                    on_token(text)

                if delta.get("tool_calls"):
                    for tc_delta in delta["tool_calls"]:
                        idx = tc_delta.get("index", 0)
                        if idx not in tool_calls_accum:
                            tool_calls_accum[idx] = {
                                "id": tc_delta.get("id", ""),
                                "function": {"name": "", "arguments": ""},
                            }
                        entry = tool_calls_accum[idx]
                        if tc_delta.get("id"):
                            entry["id"] = tc_delta["id"]
                        func = tc_delta.get("function", {})
                        if func.get("name"):
                            entry["function"]["name"] = func["name"]
                        if func.get("arguments"):
                            entry["function"]["arguments"] += func["arguments"]

                if choices[0].get("finish_reason"):
                    finish_reason = choices[0]["finish_reason"]

        content = "".join(content_parts) if content_parts else None
        tool_calls: list[ToolCall] = []
        for _idx in sorted(tool_calls_accum.keys()):
            tc_data = tool_calls_accum[_idx]
            arguments = tc_data["function"]["arguments"]
            if isinstance(arguments, str) and arguments:
                arguments = json.loads(arguments)
            elif isinstance(arguments, str):
                arguments = {}
            tool_calls.append(
                ToolCall(
                    id=tc_data["id"],
                    name=tc_data["function"]["name"],
                    arguments=arguments,
                )
            )

        input_tokens = usage_data.get("prompt_tokens", 0)
        output_tokens = usage_data.get("completion_tokens", 0)
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
