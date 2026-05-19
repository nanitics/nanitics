"""Shared helpers for SDK examples. Not an example itself."""

from nanitics.infrastructure import LLMResponse
from nanitics.tracing import (
    InMemoryEmitter,
    Usage,
)


def make_emitter(trace_id: str = "example-trace") -> InMemoryEmitter:
    """Create a standard emitter for examples."""
    return InMemoryEmitter(trace_id=trace_id)


def make_usage(input_tokens: int = 10, output_tokens: int = 5) -> Usage:
    """Create a standard usage object."""
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def make_response(content: str, **kwargs) -> LLMResponse:
    """Create a standard LLM response."""
    return LLMResponse(
        content=content,
        tool_calls=kwargs.get("tool_calls", []),
        usage=kwargs.get("usage", make_usage()),
        model=kwargs.get("model", "mock-model"),
        stop_reason=kwargs.get("stop_reason", "end_turn"),
        parsed=kwargs.get("parsed"),
    )
