from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from nanitics.infrastructure.errors import (
    LLMContextLengthError,
    LLMProviderError,
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
    import anthropic
except ImportError as _err:  # pragma: no cover
    raise ImportError(
        "AnthropicLLMClient requires the 'anthropic' package, which ships with nanitics. "
        "Reinstall with: pip install --force-reinstall nanitics"
    ) from _err

STRUCTURED_OUTPUT_TOOL_NAME = "structured_output"


def _content_block_to_anthropic(block: ContentBlock) -> dict[str, Any]:
    """Convert an SDK ContentBlock to Anthropic's format."""
    if isinstance(block, TextContentBlock):
        return {"type": "text", "text": block.text}
    # ImageContentBlock
    if block.data.startswith(("http://", "https://")):
        return {"type": "image", "source": {"type": "url", "url": block.data}}
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": block.media_type, "data": block.data},
    }


def _to_anthropic_messages(messages: list[Message], *, enable_caching: bool = False) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == "user":
            content_val: list[dict[str, Any]] | str
            if isinstance(msg.content, list):
                content_val = [_content_block_to_anthropic(b) for b in msg.content]
            else:
                content_val = msg.content or ""
            entry: dict[str, Any] = {"role": "user", "content": content_val}
            if msg.name is not None:
                entry["name"] = msg.name
            result.append(entry)

        elif msg.role == "assistant":
            content_blocks: list[dict[str, Any]] = []
            if msg.content:
                content_blocks.append({"type": "text", "text": msg.content})
            if msg.tool_calls:
                content_blocks.extend(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                    for tc in msg.tool_calls
                )
            entry = {
                "role": "assistant",
                # Defensive guard: Anthropic rejects empty assistant content
                "content": content_blocks if content_blocks else (msg.content or " "),
            }
            if msg.name is not None:
                entry["name"] = msg.name
            result.append(entry)

        elif msg.role == "tool_result":
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": msg.tool_call_id,
                "content": msg.content or "",
            }
            # Group consecutive tool_result messages into one user message
            if (
                result
                and result[-1]["role"] == "user"
                and isinstance(result[-1]["content"], list)
                and all(isinstance(b, dict) and b.get("type") == "tool_result" for b in result[-1]["content"])
            ):
                result[-1]["content"].append(tool_result_block)
            else:
                result.append({"role": "user", "content": [tool_result_block]})

    if enable_caching and result:
        # Find the last user-role message and add cache_control to its last content block
        for entry in reversed(result):
            if entry["role"] == "user":
                content = entry["content"]
                if isinstance(content, str):
                    entry["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
                elif isinstance(content, list) and content:
                    content[-1]["cache_control"] = {"type": "ephemeral"}
                break

    return result


def _to_anthropic_tools(tools: list[ToolSchema], *, enable_caching: bool = False) -> list[dict[str, Any]]:
    result = [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        for t in tools
    ]
    if enable_caching and result:
        result[-1]["cache_control"] = {"type": "ephemeral"}
    return result


def _from_anthropic_response(response: anthropic.types.Message, model: str) -> LLMResponse:
    content_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    reasoning_parts: list[str] = []

    blocks = list(response.content)
    # Walk the block sequence once, extracting ``content`` / ``tool_calls``
    # and ``reasoning_text`` in a single pass. A ``text`` block is reasoning
    # only when a ``tool_use`` block follows it in the same response —
    # trailing text on non-tool responses is the final answer and belongs
    # in ``content`` only.
    for i, block in enumerate(blocks):
        if block.type == "text":
            content_parts.append(block.text)
            if any(later.type == "tool_use" for later in blocks[i + 1 :]):
                reasoning_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                )
            )
        elif block.type == "thinking":
            reasoning_parts.append(block.thinking)

    content = "\n".join(content_parts) if content_parts else None
    reasoning_text = "\n\n".join(reasoning_parts) if reasoning_parts else None

    usage = Usage(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cache_creation_input_tokens=getattr(response.usage, "cache_creation_input_tokens", None),
        cache_read_input_tokens=getattr(response.usage, "cache_read_input_tokens", None),
    )

    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        usage=usage,
        model=model,
        stop_reason=response.stop_reason or "end_turn",
        reasoning_text=reasoning_text,
    )


class AnthropicLLMClient:
    """LLM client for the Anthropic API (Claude models).

    Handles message formatting, tool call serialization, structured output
    via tool-use, and maps API errors to the SDK error hierarchy.

    Args:
        model: Anthropic model identifier (e.g., ``"claude-haiku-4-5-20251001"``).
        api_key: API key. Falls back to the ``ANTHROPIC_API_KEY`` environment
            variable if not provided.
        max_tokens: Maximum tokens to generate per call (default: 64,000).
        enable_caching: Off by default. Anthropic cache writes cost ~1.25× a
            baseline input token and reads cost ~0.1×, so caching is a net
            loss on a single call and only breaks even once the cached
            prefix is reused ≥2 times within the 5-minute TTL. Opt in for
            multi-turn loops or repeated calls that share a stable prefix.

    Raises:
        ValueError: If no ``api_key`` is provided and ``ANTHROPIC_API_KEY``
            is unset. This is a construction-time misconfiguration; live-API
            failures surface as ``LLMProviderError`` (or its subclasses) at
            generation time.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 64_000,
        enable_caching: bool = False,
        request_timeout: float | None = 300.0,
    ) -> None:
        if api_key is None and not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError(
                "ANTHROPIC_API_KEY env var is not set and no api_key= was provided. "
                "Set ANTHROPIC_API_KEY or pass api_key= explicitly."
            )
        self._model = model
        self._max_tokens = max_tokens
        self._enable_caching = enable_caching
        self._request_timeout = request_timeout
        self._client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=0)

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

        if self._enable_caching and system_prompt_sections is not None:
            # Structured sections: apply cache_control selectively
            blocks: list[dict[str, Any]] = []
            last_cacheable_idx = -1
            for i, section in enumerate(system_prompt_sections):
                blocks.append({"type": "text", "text": section.content})
                if section.cacheable:
                    last_cacheable_idx = i
            if last_cacheable_idx >= 0:
                blocks[last_cacheable_idx]["cache_control"] = {"type": "ephemeral"}
            system_value: str | list[dict[str, Any]] = blocks
        elif self._enable_caching:
            system_value = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
        else:
            system_value = system_prompt

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system_value,
            "messages": _to_anthropic_messages(messages, enable_caching=self._enable_caching),
        }

        if output_schema is not None:
            schema = output_schema.model_json_schema()
            tool_def: dict[str, Any] = {
                "name": STRUCTURED_OUTPUT_TOOL_NAME,
                "description": (f"Return structured output matching the {output_schema.__name__} schema."),
                "input_schema": schema,
            }
            if self._enable_caching:
                tool_def["cache_control"] = {"type": "ephemeral"}
            kwargs["tools"] = [tool_def]
            kwargs["tool_choice"] = {
                "type": "tool",
                "name": STRUCTURED_OUTPUT_TOOL_NAME,
            }
        elif tools is not None:
            kwargs["tools"] = _to_anthropic_tools(tools, enable_caching=self._enable_caching)

        try:
            timeout_ctx = (
                asyncio.timeout(self._request_timeout)
                if self._request_timeout is not None
                else contextlib.nullcontext()
            )
            async with timeout_ctx, self._client.messages.stream(**kwargs) as stream:
                if on_token is not None and output_schema is None:
                    async for text in stream.text_stream:
                        on_token(text)
                response = await stream.get_final_message()
        except TimeoutError:
            raise LLMProviderError(
                f"Request timed out after {self._request_timeout}s",
                provider="anthropic",
            ) from None
        except anthropic.RateLimitError as e:
            retry_after = None
            if hasattr(e, "response") and e.response is not None:
                raw = e.response.headers.get("retry-after")
                if raw is not None:
                    with contextlib.suppress(ValueError, TypeError):
                        retry_after = float(raw)
            raise LLMRateLimitError(str(e), retry_after=retry_after) from e
        except anthropic.BadRequestError as e:
            # Structured-first classification: Anthropic has no structured
            # error code for context overflow (the 400 ``error.type`` is the
            # generic ``"invalid_request_error"``), so the overflow signal is
            # the documented anchored phrase ``"prompt is too long"``. Any
            # other invalid_request_error surfaces as ``LLMProviderError``.
            # See Anthropic API error reference for the exact phrasing.
            body = getattr(e, "body", None) or {}
            err_obj = body.get("error", {}) if isinstance(body, dict) else {}
            err_type = err_obj.get("type") if isinstance(err_obj, dict) else None
            err_message = err_obj.get("message", "") if isinstance(err_obj, dict) else ""
            if err_type == "invalid_request_error" and "prompt is too long" in err_message:
                raise LLMContextLengthError(str(e)) from e
            raise LLMProviderError(str(e), status_code=e.status_code, provider="anthropic") from e
        except anthropic.AuthenticationError as e:
            raise LLMProviderError(str(e), status_code=e.status_code, provider="anthropic") from e
        except anthropic.APIStatusError as e:
            raise LLMProviderError(str(e), status_code=e.status_code, provider="anthropic") from e
        except anthropic.APIConnectionError as e:
            raise LLMProviderError(str(e), provider="anthropic") from e

        llm_response = _from_anthropic_response(response, self._model)

        if output_schema is not None:
            # Extract structured data from the tool call
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
