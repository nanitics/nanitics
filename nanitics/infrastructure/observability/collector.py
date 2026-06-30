"""TraceCollector — classify, buffer, flush, and optionally stream trace events.

Receives :class:`TraceEvent` objects (typically via
:meth:`EventEmitter.add_listener`), classifies their level, buffers them
in memory, and periodically flushes to a :class:`PersistentTraceStore`.
Optionally pushes qualifying events to an :class:`asyncio.Queue` for
SSE streaming.
"""

from __future__ import annotations

import asyncio
import warnings

from nanitics.infrastructure.observability.events import TraceEvent
from nanitics.infrastructure.observability.levels import (
    TraceLevel,
    classify_level,
    is_level_included,
)
from nanitics.infrastructure.observability.redaction import RedactionHook
from nanitics.infrastructure.observability.storage import (
    PersistentTraceStore,
    TraceEventRecord,
)

DEFAULT_FLUSH_INTERVAL: float = 0.5

# A batch that fails this many consecutive flushes is treated as permanently
# un-storable — e.g. third-party tool output carrying a byte the store rejects
# (a Postgres ``text``/``jsonb`` column refuses a NUL) — and dropped, rather than
# re-buffered to be re-attempted forever. Without the cap, one poison batch
# re-queued at the head fails every later flush, disabling the whole run's trace
# and flooding the log. See :meth:`TraceCollector.flush`.
MAX_FLUSH_ATTEMPTS: int = 3


class TraceCollector:
    """Collects SDK trace events, buffers them, and flushes to a persistent store.

    Register as a listener on an :class:`EventEmitter`::

        collector = TraceCollector(store=store, parent_id="run-123")
        emitter.add_listener(collector.handle, internal=True)

    Args:
        store: Persistent store to flush events to.
        parent_id: Foreign key value grouping events (e.g. run ID).
        queue: Optional async queue for live SSE streaming.
        min_level: Minimum level for events pushed to *queue*.
        flush_interval: Seconds between automatic flushes.
        redaction_hook: Optional adopter-supplied
            :class:`RedactionHook`. When provided, ``handle()`` calls
            ``redaction_hook.redact(event)`` exactly once per event and
            uses the returned event for both the
            :class:`TraceEventRecord` built for persistence and the
            payload pushed to *queue*. The hook runs before record
            construction and before queue push, so scrubbed fields are
            scrubbed on both downstream surfaces. If the hook raises,
            the exception propagates and the event is neither persisted
            nor enqueued (fail-closed). See
            ``docs/guides/observability.md#trace-surface-hygiene``.
    """

    def __init__(
        self,
        store: PersistentTraceStore,
        parent_id: str,
        *,
        queue: asyncio.Queue[dict[str, object]] | None = None,
        min_level: TraceLevel = "info",
        flush_interval: float = DEFAULT_FLUSH_INTERVAL,
        max_buffer_size: int = 10_000,
        redaction_hook: RedactionHook | None = None,
    ) -> None:
        self._store = store
        self._parent_id = parent_id
        self._queue = queue
        self._min_level = min_level
        self._flush_interval = flush_interval
        self._max_buffer_size = max_buffer_size
        self._redaction_hook = redaction_hook

        self._buffer: list[TraceEventRecord] = []
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task[None] | None = None
        self._consecutive_failures: int = 0

    def handle(self, event: TraceEvent) -> None:
        """Classify, buffer, and optionally enqueue a trace event.

        This is the callback to pass to :meth:`EventEmitter.add_listener`.
        If a ``redaction_hook`` is configured, it is invoked before the
        :class:`TraceEventRecord` is built and before any SSE queue
        push; any exception it raises propagates out of this method
        without persisting or enqueuing the event.
        """
        if self._redaction_hook is not None:
            event = self._redaction_hook.redact(event)
        level = classify_level(event.event_type)
        payload = event.model_dump(mode="json")

        record = TraceEventRecord(
            event_type=event.event_type,
            level=level,
            trace_id=event.trace_id,
            span_id=event.span_id,
            parent_span_id=event.parent_span_id,
            payload=payload,
            sdk_timestamp=event.timestamp,
        )

        self._buffer.append(record)
        self._ensure_flush_loop()

        if self._queue is not None and is_level_included(level, self._min_level):
            self._queue.put_nowait(
                {
                    "event_type": "trace",
                    "payload": {
                        "sdk_event_type": event.event_type,
                        "level": level,
                        "trace_id": event.trace_id,
                        "span_id": event.span_id,
                        "parent_span_id": event.parent_span_id,
                        "timestamp": event.timestamp.isoformat(),
                        **payload,
                    },
                }
            )

    async def flush(self) -> None:
        """Persist all buffered events to the store.

        A failed write returns the batch to the buffer and retries it on the
        next flush, so a transient store outage loses nothing. But after
        :data:`MAX_FLUSH_ATTEMPTS` consecutive failures the batch is treated as
        permanently un-storable and dropped with a single warning, rather than
        re-buffered to fail forever — one un-storable event (e.g. fetched
        content carrying a byte the store rejects) costs that batch, not the
        run's whole trace. Events appended to the buffer during the dropped
        attempt are not part of that batch and flush on the next cycle.
        """
        async with self._lock:
            if not self._buffer:
                return
            batch = self._buffer[:]
            self._buffer.clear()

        try:
            await self._store.save_events_batch(self._parent_id, batch)
            self._consecutive_failures = 0
        except Exception as exc:
            self._consecutive_failures += 1
            if self._consecutive_failures >= MAX_FLUSH_ATTEMPTS:
                # The head batch has failed every attempt in the window — drop it
                # so it cannot block every later event from persisting. The
                # snapshot above already removed it from the buffer, so anything
                # that arrived since survives.
                self._consecutive_failures = 0
                warnings.warn(
                    f"EventCollector dropping {len(batch)} un-storable events after "
                    f"{MAX_FLUSH_ATTEMPTS} consecutive flush failures: {exc}",
                    stacklevel=2,
                )
                return
            # Transient failure — put events back so they aren't lost.
            async with self._lock:
                self._buffer = batch + self._buffer
                if len(self._buffer) > self._max_buffer_size:
                    self._buffer = self._buffer[-self._max_buffer_size :]
            warnings.warn(
                f"EventCollector flush failed ({self._consecutive_failures} consecutive): {exc}",
                stacklevel=2,
            )

    async def close(self) -> None:
        """Final flush and cleanup. Cancel the periodic flush task."""
        if self._flush_task is not None:
            self._flush_task.cancel()
            self._flush_task = None
        await self.flush()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_flush_loop(self) -> None:
        """Start the periodic flush loop if not already running."""
        if self._flush_task is None:
            self._flush_task = asyncio.create_task(self._periodic_flush())

    async def _periodic_flush(self) -> None:
        """Background coroutine that flushes the buffer at a fixed interval."""
        try:
            while True:
                await asyncio.sleep(self._flush_interval)
                await self.flush()
        except asyncio.CancelledError:
            pass
