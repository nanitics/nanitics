"""SSE streaming for observatory — polls PersistentTraceStore for new events."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from nanitics.infrastructure.observability.levels import (
    LEVEL_ORDER,
    TraceLevel,
    is_level_included,
)
from nanitics.infrastructure.observability.storage import RunStatus
from nanitics.observatory.service import _event_to_response

if TYPE_CHECKING:
    from starlette.requests import Request

    from nanitics.infrastructure.observability.storage import PersistentTraceStore

_POLL_INTERVAL_S = 0.3
_KEEPALIVE_INTERVAL_S = 15.0
_POLL_BATCH_SIZE = 50

_TERMINAL_STATUSES: frozenset[RunStatus] = frozenset({"completed", "failed"})


async def stream_run_events(
    store: PersistentTraceStore,
    run_id: str,
    request: Request,
    *,
    min_level: TraceLevel = "info",
    last_event_id: int | None = None,
) -> AsyncGenerator[str]:
    """Async generator yielding SSE-formatted messages for a run.

    Polls the store for new events and streams them to the client.
    Terminates when the run reaches a terminal status and all events
    have been delivered, or when the client disconnects.
    """
    cursor = last_event_id
    levels = [lvl for lvl in LEVEL_ORDER if is_level_included(lvl, min_level)]
    seconds_since_last_message = 0.0

    while not await request.is_disconnected():
        events = await store.query_events(
            run_id,
            levels=levels,
            after_id=cursor,
            limit=_POLL_BATCH_SIZE,
        )

        if events:
            for event in events:
                response = _event_to_response(event)
                data = json.dumps(response.model_dump(mode="json"))
                yield f"id: {event.id}\nevent: trace\ndata: {data}\n\n"
                cursor = event.id
            seconds_since_last_message = 0.0
            # More events may be waiting — skip sleep when we got a full batch
            if len(events) >= _POLL_BATCH_SIZE:
                continue
        else:
            run = await store.get_run(run_id)
            if run is not None and run.status in _TERMINAL_STATUSES:
                # Drain: re-poll to catch events saved between query_events
                # and get_run. Only emit run_complete once no events remain.
                drain = await store.query_events(run_id, levels=levels, after_id=cursor, limit=_POLL_BATCH_SIZE)
                if drain:
                    for event in drain:
                        response = _event_to_response(event)
                        ev_data = json.dumps(response.model_dump(mode="json"))
                        yield f"id: {event.id}\nevent: trace\ndata: {ev_data}\n\n"
                        cursor = event.id
                    continue
                data = json.dumps({"status": run.status})
                yield f"event: run_complete\ndata: {data}\n\n"
                return

            seconds_since_last_message += _POLL_INTERVAL_S
            if seconds_since_last_message >= _KEEPALIVE_INTERVAL_S:
                yield ": keepalive\n\n"
                seconds_since_last_message = 0.0

        await asyncio.sleep(_POLL_INTERVAL_S)
