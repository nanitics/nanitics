"""RedactionHook — the SDK's adopter-surface redaction seam for trace events.

The SDK emits trace events whose payloads are authored by adopter code
(prompts, tool inputs and outputs, custom event fields, tool exception
messages). When that content contains data an adopter does not want
persisted or streamed — PII, proprietary strings, anything the adopter's
threat model says is sensitive — the adopter wires a
:class:`RedactionHook` into :class:`TraceCollector` or
:meth:`TracedExecutor.execute`. The hook runs *before* the event is
persisted and *before* it is pushed to the SSE live-stream queue, so any
field the hook scrubs is scrubbed on both downstream surfaces.

The SDK does not ship a default implementation. Regex lists and
credential shapes drift with provider evolution and adopter content is
adopter-owned; a shipped default would be a promise the SDK cannot
honour. See ``docs/guides/observability.md#trace-surface-hygiene``
for design framing and a copy-paste example hook.

The SDK-side no-leakage guarantee (provider credentials, auth headers,
and raw HTTP context are never written into event payloads by SDK
emission code) is orthogonal to this seam and is enforced by the
release-gate invariant test in
``tests/test_no_leakage_invariant.py``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from nanitics.infrastructure.observability.events import TraceEvent


@runtime_checkable
class RedactionHook(Protocol):
    """Adopter-supplied redactor invoked once per emitted trace event.

    Call site and ordering. The hook runs inside
    :meth:`TraceCollector.handle` *after* the event arrives from the
    emitter and *before* (a) the :class:`TraceEventRecord` is built for
    persistence and (b) the payload is pushed to the SSE queue. Whatever
    the hook returns is what the persistence layer and live consumers
    see. The hook does **not** run inside :meth:`EventEmitter.emit` or
    against :attr:`InMemoryEmitter.events` — in-process listeners observe
    the un-redacted event.

    Return contract. Implementations must return a
    :class:`TraceEvent`. Returning the same instance is allowed when no
    field needs scrubbing. Returning a modified copy is the usual case
    (events are frozen Pydantic models, so use ``model_copy(update=...)``
    rather than attempting to mutate in place — mutation raises
    :class:`pydantic.ValidationError`). Returning ``None`` or any
    non-:class:`TraceEvent` value causes downstream record construction
    to raise.

    Fields the adopter should preserve. The Observatory UI, the trace
    analyzer, and any downstream correlation rely on the tracing
    skeleton:

    - ``event_id``
    - ``trace_id``
    - ``span_id``
    - ``parent_span_id``
    - ``timestamp``
    - ``event_type``

    Adopters are free to scrub any other field. The SDK does not enforce
    skeleton preservation at runtime — the contract is documentary —
    because a hook that legitimately wants to rewrite a ``trace_id`` for
    tenant isolation must be allowed to do so.

    Exception semantics. Exceptions propagate. If the hook raises, the
    event is neither persisted nor enqueued; the caller of the agent run
    observes the exception. This is deliberate: swallowing a hook
    exception and silently persisting the un-redacted event would defeat
    the security property the adopter wired the hook in for.

    Implementations. Implementations may be stateless or stateful. They
    may branch on :func:`isinstance` or on :attr:`TraceEvent.event_type`
    to decide what to scrub per event type. The protocol is
    :func:`~typing.runtime_checkable`, so ``isinstance(obj,
    RedactionHook)`` can be used to validate an injected hook.
    """

    def redact(self, event: TraceEvent) -> TraceEvent:
        """Return a (possibly redacted) replacement for *event*.

        Args:
            event: The original event as emitted by the SDK.

        Returns:
            A :class:`TraceEvent` to persist and stream in place of the
            original. May be the same instance if no redaction is
            needed.
        """
        ...
