from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from nanitics.infrastructure.llm.protocol import (
    LLMClient,
    LLMResponse,
    Message,
    SystemPromptSection,
    ToolSchema,
)
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import ModelRoutingEvent


@dataclass
class RoutingContext:
    """Request context available to routing strategies.

    Attributes:
        system_prompt: The system prompt being sent.
        messages: Conversation history.
        tools: Tool schemas, if any.
        output_schema: Pydantic model for structured output, if any.
    """

    system_prompt: str
    messages: list[Message]
    tools: list[ToolSchema] | None
    output_schema: type[BaseModel] | None


@runtime_checkable
class RoutingStrategy(Protocol):
    """Protocol for LLM routing strategies.

    Implementations return a client key from ``select()`` that maps to one
    of the clients registered in ``RoutingLLMClient``.
    """

    def select(self, context: RoutingContext) -> str: ...


class RoutingLLMClient:
    """LLM client that routes requests to one of multiple backing clients.

    Implements the ``LLMClient`` protocol. On each ``generate()`` call, the
    routing strategy selects a client key. If the key is unknown and a
    ``default`` is set, the default client is used; otherwise a
    ``ValueError`` is raised.

    Emits ``ModelRoutingEvent`` if an ``emitter`` is provided.

    After each successful generation, calls ``strategy.on_response(key,
    response)`` if the strategy implements that method (e.g.,
    ``CostBudgetRouting`` uses this to track token usage).

    Args:
        clients: Mapping of string keys to ``LLMClient`` instances.
        strategy: Routing strategy that selects a client key.
        default: Fallback key when the strategy returns an unknown key.
        emitter: Event emitter for routing events.
    """

    def __init__(
        self,
        *,
        clients: dict[str, LLMClient],
        strategy: RoutingStrategy,
        default: str | None = None,
        emitter: EventEmitter | None = None,
    ) -> None:
        if not clients:
            raise ValueError("clients must be non-empty")
        if default is not None and default not in clients:
            raise ValueError(f"default key {default!r} not in clients")
        self._clients = clients
        self._strategy = strategy
        self._default = default
        self._emitter = emitter

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
        context = RoutingContext(
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            output_schema=output_schema,
        )

        key = self._strategy.select(context)

        if key not in self._clients:
            if self._default is not None:
                key = self._default
            else:
                raise ValueError(f"Strategy returned unknown key {key!r}; available: {list(self._clients)}")

        if self._emitter is not None:
            self._emitter.emit(
                ModelRoutingEvent(
                    trace_id=self._emitter.trace_id,
                    span_id=self._emitter.span_id,
                    parent_span_id=self._emitter.parent_span_id,
                    strategy_name=type(self._strategy).__name__,
                    selected_key=key,
                    available_keys=list(self._clients),
                )
            )

        response = await self._clients[key].generate(
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            output_schema=output_schema,
            on_token=on_token,
            system_prompt_sections=system_prompt_sections,
        )

        on_response = getattr(self._strategy, "on_response", None)
        if callable(on_response):
            on_response(key, response)

        return response


class RuleBasedRouting:
    """Routes requests via a user-defined function.

    Args:
        rule: Callable that receives a ``RoutingContext`` and returns a
            client key string.
    """

    def __init__(self, *, rule: Callable[[RoutingContext], str]) -> None:
        self._rule = rule

    def select(self, context: RoutingContext) -> str:
        return self._rule(context)


class CostBudgetRouting:
    """Routes requests based on cumulative token usage against a budget.

    Tracks total tokens consumed and selects a client key based on
    threshold ratios. When usage exceeds a threshold, the corresponding
    client key is selected. Multiple thresholds are checked in ascending
    order — the highest matching threshold wins.

    Implements ``on_response()`` to automatically track token usage after
    each LLM call.

    Args:
        budget: Total token budget.
        thresholds: List of ``(ratio, key)`` tuples. When
            ``used / budget >= ratio``, the corresponding key is selected.
        default: Client key to use when no threshold is exceeded.
    """

    def __init__(
        self,
        *,
        budget: int,
        thresholds: list[tuple[float, str]],
        default: str,
    ) -> None:
        if budget <= 0:
            raise ValueError("budget must be positive")
        self._budget = budget
        self._thresholds = sorted(thresholds, key=lambda t: t[0])
        self._default = default
        self._used = 0

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        return self._budget - self._used

    def select(self, context: RoutingContext) -> str:
        ratio = self._used / self._budget if self._budget > 0 else 0.0
        selected = self._default
        for threshold, key in self._thresholds:
            if ratio >= threshold:
                selected = key
        return selected

    def on_response(self, key: str, response: LLMResponse) -> None:
        self._used += response.usage.total_tokens
