from __future__ import annotations

import warnings
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable
from uuid import uuid4

from nanitics.infrastructure.observability.events import (
    SpanEndEvent,
    SpanStartEvent,
    TraceEvent,
)


@runtime_checkable
class EventEmitter(Protocol):
    """Protocol for event emission and trace management.

    Implementations collect events, support real-time listeners, and
    manage hierarchical spans for structured tracing. Every agent
    requires an ``EventEmitter``.
    """

    @property
    def trace_id(self) -> str:
        """Unique identifier for the current trace."""
        ...

    @property
    def span_id(self) -> str:
        """Identifier of the currently active span."""
        ...

    @property
    def parent_span_id(self) -> str | None:
        """Identifier of the parent span, or ``None`` at the root."""
        ...

    def emit(self, event: TraceEvent) -> None:
        """Record an event and notify all listeners."""
        ...

    def add_listener(self, callback: Callable[[TraceEvent], None]) -> None:
        """Register a callback invoked for every emitted event."""
        ...

    def span(self, name: str) -> AbstractContextManager[None]:
        """Open a named child span as a context manager."""
        ...

    def create_child(self) -> EventEmitter:
        """Create a child emitter linked to this emitter's trace and current span.

        The child shares the parent's ``trace_id`` and links its root span
        to the parent's current ``span_id`` via ``parent_span_id``. It copies
        the parent's listeners but maintains an independent span stack.
        """
        ...


class InMemoryEmitter:
    """In-memory event emitter with hierarchical span support.

    Collects all events in ``events``. Supports real-time listeners and
    manages a span stack via context variables (async-safe).

    Args:
        trace_id: Unique identifier for this trace.
        max_events: If set, keeps only the most recent *n* events to
            cap memory usage.

    Attributes:
        events: List of all emitted events (most recent last).
    """

    def __init__(
        self,
        trace_id: str,
        *,
        max_events: int | None = None,
        parent_span_id: str | None = None,
        initial_span_id: str | None = None,
    ) -> None:
        self._trace_id = trace_id
        self._max_events = max_events
        self._root_parent_span_id = parent_span_id
        # ``initial_span_id`` lets ``create_child`` seed the stack with the
        # parent emitter's current span_id, so the first span opened in the
        # child parents under that span rather than under a synthetic UUID
        # that no event ever names.
        root_span_id = initial_span_id if initial_span_id is not None else str(uuid4())
        self._span_stack: ContextVar[list[str]] = ContextVar("span_stack")
        self._span_stack.set([root_span_id])
        self.events: list[TraceEvent] = []
        self._listeners: list[Callable[[TraceEvent], None]] = []

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def span_id(self) -> str:
        return self._span_stack.get()[-1]

    @property
    def parent_span_id(self) -> str | None:
        stack = self._span_stack.get()
        if len(stack) > 1:
            return stack[-2]
        return self._root_parent_span_id

    def emit(self, event: TraceEvent) -> None:
        self.events.append(event)
        if self._max_events is not None and len(self.events) > self._max_events:
            self.events = self.events[-self._max_events :]
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception as exc:
                warnings.warn(f"Event listener failed: {exc}", stacklevel=2)

    def add_listener(self, callback: Callable[[TraceEvent], None]) -> None:
        self._listeners.append(callback)

    def _append_forwarded(self, event: TraceEvent) -> None:
        """Append a child-emitter event to this emitter's ``events`` list.

        Does not re-run listeners: re-emitting would cause duplicate listener
        callbacks and, if the listener itself forwards, infinite recursion.
        Used only as the forwarding sink installed by ``create_child``.
        """
        self.events.append(event)
        if self._max_events is not None and len(self.events) > self._max_events:
            self.events = self.events[-self._max_events :]

    def create_child(self) -> InMemoryEmitter:
        """Create a child emitter linked to this emitter's trace and current span.

        The child's ``emit`` path forwards every event into this emitter's
        ``events`` list via ``_append_forwarded`` so that composite-agent
        inner events surface in the outer emitter's trace. Existing listeners
        are copied so real-time consumers (e.g. ``InstrumentedLLMClient``
        label partitioning) continue to receive inner events through the
        normal listener mechanism.

        The child's stack is seeded with this emitter's current ``span_id``,
        so the first span the child opens parents under that span — not under
        a fresh synthetic UUID. Without this, every span tree built through
        ``Agent.bind`` (i.e., every ``AgentTool`` delegation, ``ReflexionAgent``
        inner attempt, and workflow child) would orphan its top-level span in
        the persisted store.
        """
        child = InMemoryEmitter(
            trace_id=self._trace_id,
            max_events=self._max_events,
            parent_span_id=self.parent_span_id,
            initial_span_id=self.span_id,
        )
        for listener in self._listeners:
            child.add_listener(listener)
        child.add_listener(self._append_forwarded)
        return child

    @contextmanager
    def span(self, name: str) -> Generator[None]:
        new_span_id = str(uuid4())
        parent = self.span_id
        stack = self._span_stack.get()
        new_stack = [*stack, new_span_id]
        token = self._span_stack.set(new_stack)

        start_time = datetime.now(UTC)
        self.emit(
            SpanStartEvent(
                trace_id=self._trace_id,
                span_id=new_span_id,
                parent_span_id=parent,
                timestamp=start_time,
                name=name,
            )
        )

        try:
            yield
        finally:
            end_time = datetime.now(UTC)
            duration_ms = (end_time - start_time).total_seconds() * 1000
            self.emit(
                SpanEndEvent(
                    trace_id=self._trace_id,
                    span_id=new_span_id,
                    parent_span_id=parent,
                    timestamp=end_time,
                    name=name,
                    duration_ms=duration_ms,
                )
            )
            self._span_stack.reset(token)
