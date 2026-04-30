"""Trace ingestion for the advisor runtime.

The advisor reads the canonical Nanitics trace envelope produced by
:mod:`validation.helpers.trace` (and equivalent emitter-export helpers):
``{trace_id, exported_at, script, summary, events: [...]}``. Each event dict
is validated through the :data:`TraceEvent` discriminated union so advisor
specialists can reason about typed events rather than raw dicts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import TypeAdapter, ValidationError

from nanitics.infrastructure.observability.events import TraceEvent

_TRACE_EVENT_ADAPTER: TypeAdapter[TraceEvent] = TypeAdapter(TraceEvent)
_REQUIRED_ENVELOPE_KEYS = ("trace_id", "events")


class MalformedTraceError(ValueError):
    """A trace file's envelope or events fail validation.

    The advisor surfaces ingestion failures explicitly rather than silently
    discarding malformed traces; downstream specialists should only ever see
    an event list that has already round-tripped through the discriminated
    union.
    """

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Malformed trace at {path}: {reason}")


@runtime_checkable
class TraceAdapter(Protocol):
    """Converts a trace file into a validated ``list[TraceEvent]``.

    The single shipping implementation is :class:`NaniticsTraceAdapter`.
    Adopters with non-Nanitics traces (OTEL, LangSmith) implement this
    protocol; those adapters are explicitly post-launch.
    """

    def load(self, source: Path) -> list[TraceEvent]:
        """Read ``source`` and return its validated event list."""
        ...


class NaniticsTraceAdapter:
    """Loads the canonical Nanitics JSON trace envelope.

    The envelope shape is produced by :func:`validation.helpers.trace.save_trace`
    and equivalent emitter-export helpers: ``{trace_id, exported_at, script,
    summary, events: [...]}``. Each event dict is validated through the
    :data:`TraceEvent` discriminated union. The envelope keys beyond
    ``trace_id`` and ``events`` are preserved on disk for operators but are
    not required by the adapter — the advisor only needs the event list and
    the trace id for attribution.
    """

    def load(self, source: Path) -> list[TraceEvent]:
        try:
            text = source.read_text(encoding="utf-8")
        except OSError as exc:
            raise MalformedTraceError(source, f"cannot read file: {exc}") from exc

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MalformedTraceError(source, f"invalid JSON: {exc}") from exc

        if not isinstance(payload, dict):
            raise MalformedTraceError(
                source,
                f"envelope must be a JSON object (got {type(payload).__name__!r})",
            )

        for key in _REQUIRED_ENVELOPE_KEYS:
            if key not in payload:
                raise MalformedTraceError(source, f"missing required envelope key '{key}'")

        raw_events = payload["events"]
        if not isinstance(raw_events, list):
            raise MalformedTraceError(
                source,
                f"'events' must be a list (got {type(raw_events).__name__!r})",
            )

        events: list[TraceEvent] = []
        for index, raw in enumerate(raw_events):
            try:
                event = _TRACE_EVENT_ADAPTER.validate_python(raw)
            except ValidationError as exc:
                raise MalformedTraceError(
                    source,
                    f"event at index {index} failed TraceEvent validation: {exc}",
                ) from exc
            events.append(event)

        return events


def load_trace(source: Path, *, adapter: TraceAdapter | None = None) -> list[TraceEvent]:
    """Load a trace file via ``adapter`` (defaults to :class:`NaniticsTraceAdapter`).

    This is the public convenience helper. Programmatic callers who already
    hold a :class:`TraceAdapter` can call ``adapter.load(source)`` directly;
    this helper simply removes the need to instantiate the default adapter
    for the common case.

    Args:
        source: Path to a trace file on disk.
        adapter: Optional alternate adapter. When ``None``, a new
            :class:`NaniticsTraceAdapter` is instantiated per call.

    Returns:
        The validated event list.

    Raises:
        MalformedTraceError: The adapter rejected the file.
    """
    resolved = adapter if adapter is not None else NaniticsTraceAdapter()
    return resolved.load(source)


__all__ = [
    "MalformedTraceError",
    "NaniticsTraceAdapter",
    "TraceAdapter",
    "load_trace",
]
