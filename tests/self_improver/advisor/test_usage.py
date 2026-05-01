"""Unit tests for :mod:`self_improver.advisor._usage`.

Covers ``aggregate_usage`` in isolation so the input/output token-summing
branches are exercised directly rather than through the full ``analyze()``
path.
"""

from __future__ import annotations

from self_improver.advisor._usage import aggregate_usage

from nanitics.infrastructure.observability.emitter import InMemoryEmitter
from nanitics.infrastructure.observability.events import (
    AgentStartEvent,
    LLMResponseEvent,
    Usage,
)


def _emit_response(
    emitter: InMemoryEmitter,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_creation_input_tokens: int | None = None,
    cache_read_input_tokens: int | None = None,
) -> None:
    emitter.emit(
        LLMResponseEvent(
            trace_id=emitter.trace_id,
            span_id=emitter.span_id,
            model_name="mock",
            content="ok",
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
            ),
            duration_ms=1.0,
        )
    )


class _NoEventsEmitter:
    """Minimal emitter that does not buffer events (simulates a streaming
    adopter-supplied emitter). Exposes just enough protocol for the
    usage aggregator to detect the missing ``events`` attribute and
    return a zero usage without crashing.
    """

    trace_id = "no-events"
    span_id = "root"
    parent_span_id: str | None = None

    def emit(self, event: object) -> None:  # pragma: no cover - unused in tests
        raise NotImplementedError


class TestAggregateUsageSumsResponses:
    def test_sums_input_and_output_tokens_across_events(self) -> None:
        emitter = InMemoryEmitter(trace_id="t")
        _emit_response(emitter, input_tokens=10, output_tokens=4)
        _emit_response(emitter, input_tokens=3, output_tokens=7)
        _emit_response(emitter, input_tokens=5, output_tokens=2)

        usage = aggregate_usage(emitter)

        assert usage.input_tokens == 18
        assert usage.output_tokens == 13

    def test_ignores_non_llm_response_events(self) -> None:
        emitter = InMemoryEmitter(trace_id="t")
        emitter.emit(
            AgentStartEvent(
                trace_id="t",
                span_id="s",
                agent_name="demo",
                task_input="Do.",
                tools_available=[],
            )
        )
        _emit_response(emitter, input_tokens=5, output_tokens=5)

        usage = aggregate_usage(emitter)

        assert usage.input_tokens == 5
        assert usage.output_tokens == 5

    def test_no_events_returns_zero_usage(self) -> None:
        emitter = InMemoryEmitter(trace_id="t")

        usage = aggregate_usage(emitter)

        assert usage.input_tokens == 0
        assert usage.output_tokens == 0


class TestAggregateUsageEmitterCompatibility:
    def test_emitter_without_events_attribute_returns_zero(self) -> None:
        usage = aggregate_usage(_NoEventsEmitter())  # type: ignore[arg-type]

        assert usage.input_tokens == 0
        assert usage.output_tokens == 0


class TestAggregateUsageCacheTokens:
    def test_sums_cache_creation_and_read_when_reported(self) -> None:
        emitter = InMemoryEmitter(trace_id="t")
        _emit_response(
            emitter,
            input_tokens=10,
            output_tokens=4,
            cache_creation_input_tokens=8000,
            cache_read_input_tokens=0,
        )
        _emit_response(
            emitter,
            input_tokens=12,
            output_tokens=6,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=8000,
        )

        usage = aggregate_usage(emitter)

        assert usage.input_tokens == 22
        assert usage.output_tokens == 10
        assert usage.cache_creation_input_tokens == 8000
        assert usage.cache_read_input_tokens == 8000

    def test_cache_fields_stay_none_when_no_response_reports_them(self) -> None:
        emitter = InMemoryEmitter(trace_id="t")
        _emit_response(emitter, input_tokens=5, output_tokens=3)

        usage = aggregate_usage(emitter)

        assert usage.cache_creation_input_tokens is None
        assert usage.cache_read_input_tokens is None

    def test_mixed_reporters_treat_none_as_zero_contribution(self) -> None:
        emitter = InMemoryEmitter(trace_id="t")
        _emit_response(emitter, input_tokens=5, output_tokens=3)  # no cache fields
        _emit_response(
            emitter,
            input_tokens=7,
            output_tokens=2,
            cache_creation_input_tokens=100,
            cache_read_input_tokens=None,
        )

        usage = aggregate_usage(emitter)

        assert usage.cache_creation_input_tokens == 100
        assert usage.cache_read_input_tokens is None
