"""TracedExecutor — run lifecycle management with real-time event persistence.

Composes :class:`InMemoryEmitter`, :class:`TraceCollector`, and
:class:`PersistentTraceStore` into a single entry point that wraps any
async execution with run registration, streaming event persistence, and
status finalisation.

Typical usage::

    executor = TracedExecutor(trace_store)
    run_id, result = await executor.execute(
        lambda emitter, run_id: my_agent_factory(emitter).run(input),
        metadata={"agent": "extraction"},
    )

The factory receives two positional arguments: the run-scoped
:class:`EventEmitter` and the pre-generated ``run_id``. The ``run_id``
passed into the factory is identical to the first element of the
returned ``(run_id, result)`` tuple — making it available inside the
factory lets callers key external durable state (HITL requests,
resumable workflows, auction waiter registries) on the Observatory
``run_id`` before ``execute`` returns.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar
from uuid import uuid4

from nanitics.infrastructure.observability.collector import TraceCollector
from nanitics.infrastructure.observability.emitter import EventEmitter, InMemoryEmitter
from nanitics.infrastructure.observability.levels import TraceLevel
from nanitics.infrastructure.observability.redaction import RedactionHook
from nanitics.infrastructure.observability.storage import PersistentTraceStore

T = TypeVar("T")


class TracedExecutor:
    """Wraps async execution with run lifecycle management and event persistence.

    Handles the full run lifecycle: generate IDs, register the run, create an
    emitter, wire a :class:`TraceCollector` for real-time persistence, execute
    the callable, and finalise run status. Events are persisted continuously
    (not batched after completion), so failed and suspended runs retain their
    trace data.

    Args:
        trace_store: Persistent store for run records and trace events.
    """

    def __init__(self, trace_store: PersistentTraceStore) -> None:
        self._trace_store = trace_store

    async def execute(
        self,
        fn: Callable[[EventEmitter, str], Awaitable[T]],
        metadata: dict[str, Any] | None = None,
        *,
        queue: asyncio.Queue[dict[str, object]] | None = None,
        min_level: TraceLevel = "info",
        redaction_hook: RedactionHook | None = None,
    ) -> tuple[str, T]:
        """Execute *fn* with full run lifecycle and trace persistence.

        Args:
            fn: Async callable receiving an :class:`EventEmitter` and the
                pre-generated ``run_id`` and returning a result. Free to
                create agents, workflows, or any async work. The ``run_id``
                passed in is identical to the first element of the returned
                tuple, so callers that need to key external durable state on
                the Observatory ``run_id`` before ``execute`` returns can
                use the in-factory value.
            metadata: Optional dictionary persisted with the run record.
            queue: Optional async queue for live SSE streaming. The internal
                :class:`TraceCollector` pushes qualifying events here.
            min_level: Minimum event level pushed to *queue*. Defaults to
                ``"info"``.
            redaction_hook: Optional adopter-supplied
                :class:`RedactionHook` forwarded to the internal
                :class:`TraceCollector`. When provided, the hook runs
                once per emitted event before persistence and before
                SSE queue push, so any field the hook scrubs is scrubbed
                on both downstream surfaces. A hook that raises causes
                the affected event to be neither persisted nor enqueued
                and the run to fail fast (see
                ``docs/guides/observability.md#trace-surface-hygiene``).
                The parameter lives on ``execute()`` rather than
                ``__init__`` so redaction policy can vary per run
                (tenant, user, request).

        Returns:
            A ``(run_id, result)`` tuple where *run_id* is the generated
            identifier and *result* is whatever *fn* returned.

        Raises:
            SuspendExecution: Re-raised after persisting events and marking
                the run as suspended.
            Exception: Any other exception is re-raised after persisting
                events and marking the run as failed.
        """
        # Deferred import: ``SuspendExecution`` lives in ``composition.durability``
        # which layers above ``infrastructure``. Importing it at module level
        # would invert the dependency direction.
        from nanitics.composition.durability.suspension import SuspendExecution

        run_id = str(uuid4())
        trace_id = str(uuid4())

        await self._trace_store.register_run(run_id, trace_id, metadata or {})

        emitter = InMemoryEmitter(trace_id=trace_id)
        collector = TraceCollector(
            store=self._trace_store,
            parent_id=run_id,
            queue=queue,
            min_level=min_level,
            redaction_hook=redaction_hook,
        )
        emitter.add_listener(collector.handle)

        try:
            result = await fn(emitter, run_id)
        except SuspendExecution:
            await collector.close()
            await self._trace_store.update_run_status(run_id, "suspended")
            raise
        except Exception as exc:
            await collector.close()
            await self._trace_store.update_run_status(run_id, "failed", error=str(exc))
            raise

        await collector.close()
        result_str = _serialize_result(result)
        await self._trace_store.update_run_status(run_id, "completed", result=result_str)

        return run_id, result


def _serialize_result(result: object) -> str:
    """Best-effort serialization of a run result for storage."""
    if isinstance(result, str):
        return result
    dump = getattr(result, "model_dump_json", None)
    if callable(dump):
        return dump()  # type: ignore[no-any-return]
    return json.dumps(result, default=str)
