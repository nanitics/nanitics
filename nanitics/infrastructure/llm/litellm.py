"""LiteLLM adapter client — single `LLMClient` routing 100+ providers via LiteLLM.

This client wraps ``litellm.acompletion()`` so any provider LiteLLM supports
(OpenAI, Anthropic, Bedrock, Vertex, Gemini, Cohere, Together, Groq, Ollama,
vLLM, Azure, and many more) can be used with Nanitics agents without a
provider-specific native client. Use native clients (``AnthropicLLMClient``,
``OpenAILLMClient``, ``MistralLLMClient``) when they exist — they provide
stronger error classification, native cache-token reporting, and a lighter
dependency footprint.

Trade-offs accepted deliberately in exchange for breadth:

- ``Retry-After`` headers are parsed opportunistically but not guaranteed
  across providers. When a ``RateLimitError`` does not carry a parseable
  header, ``retry_after`` is ``None`` and the SDK's ``RetryPolicy`` falls
  back to its default backoff schedule.
- Cache tokens are not surfaced. ``Usage.cache_creation_input_tokens`` and
  ``Usage.cache_read_input_tokens`` are always ``None`` because LiteLLM
  does not normalize cache-token fields consistently across providers.
- LiteLLM's own router, callbacks, budget tracking, caching layer, proxy,
  and observability integrations are intentionally NOT exposed. Nanitics
  ships equivalents (``RoutingLLMClient``, ``EventEmitter``, trace stores,
  ``RetryPolicy``). Users who need LiteLLM's router should use it directly
  and wrap the result in their own ``LLMClient`` adapter.

Install with ``pip install nanitics[litellm]`` — LiteLLM has a moderate
transitive footprint (tiktoken, tokenizers, httpx, aiohttp, jinja2, etc.)
so this extra is opt-in rather than bundled by default.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel

from nanitics.infrastructure.errors import (
    LLMContextLengthError,
    LLMProviderError,
    LLMRateLimitError,
    LLMSchemaViolationError,
)
from nanitics.infrastructure.llm._openai_format import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    _from_openai_response,
    _map_stop_reason,
    _to_openai_messages,
    _to_openai_tools,
)
from nanitics.infrastructure.llm.protocol import (
    LLMResponse,
    Message,
    SystemPromptSection,
    ToolCall,
    ToolSchema,
)
from nanitics.infrastructure.observability.events import Usage

try:
    import litellm
except ImportError as _err:  # pragma: no cover
    raise ImportError("LiteLLMClient requires the 'litellm' extra: pip install nanitics[litellm]") from _err


class LiteLLMClient:
    """Adapter ``LLMClient`` that routes any LiteLLM-supported provider through the agent loop.

    The ``model`` string must be provider-prefixed per LiteLLM's convention
    (e.g., ``"openai/gpt-4o-mini"``, ``"anthropic/claude-haiku-4-5"``,
    ``"bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"``,
    ``"gemini/gemini-2.0-flash"``, ``"ollama/llama3"``). It is forwarded
    verbatim to ``litellm.acompletion()`` and returned by the ``model``
    property.

    The ``provider`` field on any emitted ``LLMProviderError`` /
    ``LLMRateLimitError`` is always ``"litellm"`` — the actual underlying
    provider is implicit in the ``model`` string. This tells trace consumers
    that the call went through the adapter rather than a native client.

    Args:
        model: Provider-prefixed LiteLLM model identifier.
        api_key: Optional API key override. When unset, LiteLLM resolves the
            provider-appropriate environment variable itself (``OPENAI_API_KEY``,
            ``ANTHROPIC_API_KEY``, ``GEMINI_API_KEY``, etc.).
        base_url: Optional API base URL, forwarded as ``api_base``. Supports
            Azure deployments, Ollama servers, vLLM, and custom proxies.
        max_tokens: Maximum tokens to generate per call (default: 16,384).
            LiteLLM normalizes the parameter name per provider.
        request_timeout: Deadline in seconds for each ``generate()`` call,
            or ``None`` to disable.
        extra_kwargs: Escape hatch for provider-specific parameters. The
            dict is merged into ``litellm.acompletion()``'s call kwargs
            verbatim. Use for things like ``aws_region_name`` (Bedrock),
            ``vertex_project``/``vertex_location`` (Vertex), or Azure
            deployment identifiers.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 16_384,
        request_timeout: float | None = 300.0,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._max_tokens = max_tokens
        self._request_timeout = request_timeout
        self._extra_kwargs: dict[str, Any] = dict(extra_kwargs) if extra_kwargs else {}

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

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": _to_openai_messages(system_prompt, messages),
            "num_retries": 0,
        }
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        if self._base_url is not None:
            kwargs["api_base"] = self._base_url
        kwargs.update(self._extra_kwargs)

        if output_schema is not None:
            schema = output_schema.model_json_schema()
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": STRUCTURED_OUTPUT_TOOL_NAME,
                        "description": f"Return structured output matching the {output_schema.__name__} schema.",
                        "parameters": schema,
                    },
                }
            ]
            kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": STRUCTURED_OUTPUT_TOOL_NAME},
            }
        elif tools is not None:
            kwargs["tools"] = _to_openai_tools(tools)
            kwargs["tool_choice"] = "auto"

        use_streaming = on_token is not None and output_schema is None

        async def _api_call() -> LLMResponse:
            try:
                if use_streaming:
                    assert on_token is not None  # guaranteed by use_streaming
                    return await self._stream_request(kwargs, on_token)
                return await self._standard_request(kwargs)
            except litellm.RateLimitError as e:
                retry_after: float | None = None
                response = getattr(e, "response", None)
                if response is not None:
                    raw = response.headers.get("retry-after")
                    if raw is not None:
                        with contextlib.suppress(ValueError, TypeError):
                            retry_after = float(raw)
                raise LLMRateLimitError(str(e), retry_after=retry_after) from e
            except litellm.ContextWindowExceededError as e:
                raise LLMContextLengthError(str(e)) from e
            except litellm.BadRequestError as e:
                # Structured-first classification: LiteLLM already maps
                # context overflow to ``ContextWindowExceededError`` (handled
                # one branch up). A residual ``BadRequestError`` therefore
                # never represents overflow — surface it as a provider error
                # with no substring fallback.
                raise LLMProviderError(str(e), status_code=getattr(e, "status_code", 400), provider="litellm") from e
            except litellm.AuthenticationError as e:
                raise LLMProviderError(str(e), status_code=getattr(e, "status_code", 401), provider="litellm") from e
            except litellm.PermissionDeniedError as e:
                raise LLMProviderError(str(e), status_code=getattr(e, "status_code", 403), provider="litellm") from e
            except litellm.NotFoundError as e:
                raise LLMProviderError(str(e), status_code=getattr(e, "status_code", 404), provider="litellm") from e
            except litellm.UnprocessableEntityError as e:
                raise LLMProviderError(str(e), status_code=getattr(e, "status_code", 422), provider="litellm") from e
            except litellm.InternalServerError as e:
                raise LLMProviderError(str(e), status_code=getattr(e, "status_code", 500), provider="litellm") from e
            except litellm.APIConnectionError as e:
                raise LLMProviderError(str(e), status_code=None, provider="litellm") from e
            except litellm.APIError as e:
                raise LLMProviderError(str(e), status_code=getattr(e, "status_code", None), provider="litellm") from e

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
                provider="litellm",
            ) from None

    async def _standard_request(self, kwargs: dict[str, Any]) -> LLMResponse:
        response = await litellm.acompletion(**kwargs)
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

        stream = await litellm.acompletion(**stream_kwargs)
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
