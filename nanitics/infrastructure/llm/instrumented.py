from __future__ import annotations

import time
from collections.abc import Callable

from pydantic import BaseModel

from nanitics.infrastructure.llm.protocol import (
    LLMClient,
    LLMResponse,
    Message,
    SystemPromptSection,
    ToolSchema,
)
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import (
    LLMRequestEvent,
    LLMResponseEvent,
)


class InstrumentedLLMClient:
    """Decorator that wraps an ``LLMClient`` to emit trace events for every call.

    Emits ``LLMRequestEvent`` before and ``LLMResponseEvent`` after each
    ``generate()`` call, closing the observability gap for LLM calls made
    outside the agent loop (e.g., evaluators, context transfer, bid generators).

    Args:
        client: The real LLM client to delegate to.
        emitter: Event emitter for trace events.
        label: Optional label identifying the caller purpose
            (e.g., ``"evaluator"``, ``"summary_transfer"``).
    """

    def __init__(
        self,
        client: LLMClient,
        emitter: EventEmitter,
        label: str | None = None,
    ) -> None:
        self._client = client
        self._emitter = emitter
        self._label = label

    @property
    def model(self) -> str | None:
        return self._client.model if hasattr(self._client, "model") else None

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
        self._emitter.emit(
            LLMRequestEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                model_name=self._client.model or "",
                system_prompt=system_prompt,
                messages=[m.model_dump() for m in messages],
                tools=[t.model_dump() for t in tools] if tools else None,
                output_schema=(output_schema.model_json_schema() if output_schema else None),
                label=self._label,
            )
        )

        start = time.perf_counter()

        response = await self._client.generate(
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            output_schema=output_schema,
            on_token=on_token,
            system_prompt_sections=system_prompt_sections,
        )

        duration_ms = (time.perf_counter() - start) * 1000

        self._emitter.emit(
            LLMResponseEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                model_name=response.model,
                content=response.content,
                tool_calls=([tc.model_dump() for tc in response.tool_calls] if response.tool_calls else None),
                usage=response.usage,
                duration_ms=duration_ms,
                label=self._label,
            )
        )

        return response
