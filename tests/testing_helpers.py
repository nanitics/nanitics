"""Shared test helper functions.

Importable by test files via ``from helpers import make_response, ...``.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from nanitics.composition import FunctionStep
from nanitics.infrastructure import LLMResponse
from nanitics.tracing import (
    InMemoryEmitter,
    ToolCall,
    Usage,
)


def make_usage(input_tokens: int = 10, output_tokens: int = 5) -> Usage:
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def make_response(
    content: str | None = "response",
    tool_calls: list[ToolCall] | None = None,
    usage: Usage | None = None,
    parsed: Any = None,
) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        usage=usage or make_usage(),
        model="test-model",
        stop_reason="end_turn",
        parsed=parsed,
    )


def make_emitter() -> InMemoryEmitter:
    return InMemoryEmitter(trace_id="test-trace")


def make_step(name: str, fn: Callable[[Any], Awaitable[Any]] | None = None) -> FunctionStep:
    if fn is None:

        async def identity(x: Any) -> Any:
            return x

        fn = identity
    return FunctionStep(name=name, fn=fn)
