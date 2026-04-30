from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel

from nanitics.infrastructure.errors import LLMSchemaViolationError
from nanitics.infrastructure.llm.protocol import (
    LLMResponse,
    Message,
    SystemPromptSection,
    ToolSchema,
)


class MockLLMClient:
    """LLM client that returns scripted responses for testing.

    Responses are returned in order. Raises ``ValueError`` when all scripted
    responses have been consumed. This guarantees the retry loop classifies
    exhaustion as ``FATAL`` and terminates instantly rather than retrying —
    real provider failures use ``LLMProviderError`` and remain retryable.
    Records every call in ``calls`` for test assertions.

    Each response can be either a static ``LLMResponse`` or a callable
    that receives the current message list and returns an ``LLMResponse``.
    Callable responses enable dynamic mock behavior where later responses
    depend on earlier tool results (e.g., referencing runtime-generated IDs).

    Args:
        responses: Ordered list of ``LLMResponse`` objects or callables
            ``(list[Message]) -> LLMResponse`` to return.
        reasoning_texts: Optional parallel list of per-index
            ``reasoning_text`` overrides. When provided, its length must
            equal ``len(responses)`` (raises ``ValueError`` otherwise).
            For each index ``i``, if the resolved response is a static
            ``LLMResponse`` and ``reasoning_texts[i]`` is not ``None``,
            the returned response is rebuilt with
            ``reasoning_text=reasoning_texts[i]``. Callable responses are
            not overridden — the callable is responsible for its own
            ``reasoning_text``.

    Attributes:
        calls: List of dicts recording each ``generate()`` invocation
            with keys ``system_prompt``, ``messages``, ``tools``,
            ``output_schema``, and ``system_prompt_sections``.
    """

    def __init__(
        self,
        responses: Sequence[LLMResponse | Callable[[list[Message]], LLMResponse]],
        *,
        reasoning_texts: list[str | None] | None = None,
    ) -> None:
        if reasoning_texts is not None and len(reasoning_texts) != len(responses):
            raise ValueError(
                f"reasoning_texts must have the same length as responses "
                f"(got {len(reasoning_texts)} vs {len(responses)})"
            )
        self._responses = responses
        self._reasoning_texts = reasoning_texts
        self._index = 0
        self.calls: list[dict[str, Any]] = []

    @property
    def model(self) -> str | None:
        return None

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

        self.calls.append(
            {
                "system_prompt": system_prompt,
                "messages": messages,
                "tools": tools,
                "output_schema": output_schema,
                "system_prompt_sections": system_prompt_sections,
            }
        )

        if self._index >= len(self._responses):
            raise ValueError("MockLLMClient: no more scripted responses")

        raw = self._responses[self._index]
        current_index = self._index
        self._index += 1

        # Resolve callable responses — invoke with messages to get the actual response.
        # Callables own their own reasoning_text; the mock does not override.
        response: LLMResponse
        if callable(raw):
            response = raw(messages)
        else:
            response = raw
            if self._reasoning_texts is not None and self._reasoning_texts[current_index] is not None:
                # Static response: apply the scripted reasoning_text override.
                response = response.model_copy(update={"reasoning_text": self._reasoning_texts[current_index]})

        if on_token is not None and response.content:
            for word in response.content.split(" "):
                on_token(word + " ")

        if output_schema is not None and response.content is not None:
            try:
                parsed = output_schema.model_validate_json(response.content)
                response = response.model_copy(update={"parsed": parsed})
            except Exception as exc:
                raise LLMSchemaViolationError(
                    f"MockLLMClient: response content does not match {output_schema.__name__} schema: {exc}",
                    expected_schema=output_schema.__name__,
                    received=response.content,
                ) from exc

        return response
