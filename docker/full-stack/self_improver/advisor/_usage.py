"""Usage aggregation for the advisor runtime.

Internal module — not part of the public surface. ``aggregate_usage`` sums
input, output, and cache tokens from every :class:`LLMResponseEvent` on an
emitter into a single :class:`Usage`, so the advisor's public
:class:`AdvisorReport.usage` reflects total token spend across every
specialist call within a run.
"""

from __future__ import annotations

from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import LLMResponseEvent, Usage


def aggregate_usage(emitter: EventEmitter) -> Usage:
    """Sum token counts across every :class:`LLMResponseEvent` on ``emitter``.

    Reads the events list exposed on :class:`InMemoryEmitter` (which the
    :class:`EventEmitter` protocol does not declare but every in-tree
    implementation provides). Emitters that do not buffer events return a
    zero usage because there is nothing to sum; adopters wiring such an
    emitter are opting out of usage accounting intentionally.

    Cache fields (``cache_creation_input_tokens`` /
    ``cache_read_input_tokens``) are ``None`` on :class:`Usage` when the
    provider did not report them at all; if any response reports a number
    the aggregate surfaces the sum so the cache-write premium and cache-read
    discount are visible in the advisor's usage line.
    """
    events = getattr(emitter, "events", None)
    if events is None:
        return Usage(input_tokens=0, output_tokens=0)

    input_tokens = 0
    output_tokens = 0
    cache_creation: int | None = None
    cache_read: int | None = None
    for event in events:
        if not isinstance(event, LLMResponseEvent):
            continue
        input_tokens += event.usage.input_tokens
        output_tokens += event.usage.output_tokens
        if event.usage.cache_creation_input_tokens is not None:
            cache_creation = (cache_creation or 0) + event.usage.cache_creation_input_tokens
        if event.usage.cache_read_input_tokens is not None:
            cache_read = (cache_read or 0) + event.usage.cache_read_input_tokens

    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
    )


__all__ = ["aggregate_usage"]
