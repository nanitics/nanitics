"""Tests for observatory SSE streaming."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient

from nanitics.infrastructure.observability.levels import TraceLevel
from nanitics.infrastructure.observability.storage import (
    InMemoryPersistentTraceStore,
    RunRecord,
    TraceEventRecord,
)
from nanitics.observatory.router import create_observatory_api_router

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_TIME = datetime(2026, 3, 5, 12, 0, 0, tzinfo=UTC)


def _make_event(
    event_type: str,
    trace_id: str = "trace-1",
    span_id: str = "span-1",
    parent_span_id: str | None = None,
    payload: dict | None = None,
    level: TraceLevel = "info",
    time_offset_ms: int = 0,
) -> TraceEventRecord:
    return TraceEventRecord(
        event_type=event_type,
        level=level,
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        payload=payload or {},
        sdk_timestamp=_BASE_TIME + timedelta(milliseconds=time_offset_ms),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> InMemoryPersistentTraceStore:
    return InMemoryPersistentTraceStore()


@pytest.fixture
def client(store: InMemoryPersistentTraceStore) -> AsyncClient:
    from fastapi import FastAPI

    app = FastAPI()
    router = create_observatory_api_router(store)
    app.include_router(router, prefix="/api/observatory")
    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    )


def _parse_sse_messages(raw: str) -> list[dict]:
    """Parse raw SSE text into a list of message dicts with event/data/id fields."""
    messages: list[dict] = []
    current: dict = {}
    for line in raw.split("\n"):
        if line.startswith("id: "):
            current["id"] = line[4:]
        elif line.startswith("event: "):
            current["event"] = line[7:]
        elif line.startswith("data: "):
            current["data"] = line[6:]
        elif line.startswith(": "):
            messages.append({"comment": line[2:]})
        elif line == "" and current:
            messages.append(current)
            current = {}
    if current:
        messages.append(current)
    return messages


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSSEStream:
    async def test_404_for_missing_run(self, client: AsyncClient) -> None:
        resp = await client.get("/api/observatory/runs/no-such-run/stream")
        assert resp.status_code == 404

    async def test_event_delivery(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        """Events in the store appear in the SSE stream."""
        await store.register_run("run-1", "trace-1", {})
        await store.save_events_batch(
            "run-1",
            [
                _make_event("agent.start", payload={"agent_name": "a"}),
                _make_event(
                    "agent.complete",
                    payload={"agent_name": "a", "total_steps": 1, "termination_reason": "done"},
                    time_offset_ms=10,
                ),
            ],
        )
        await store.update_run_status("run-1", "completed")

        async with client.stream("GET", "/api/observatory/runs/run-1/stream") as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"
            body = ""
            async for chunk in resp.aiter_text():
                body += chunk
                if "run_complete" in body:
                    break

        messages = _parse_sse_messages(body)
        trace_msgs = [m for m in messages if m.get("event") == "trace"]
        assert len(trace_msgs) == 2
        first = json.loads(trace_msgs[0]["data"])
        assert first["event_type"] == "agent.start"
        assert trace_msgs[0]["id"] == str(first["id"])

    async def test_run_complete_event(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        """Stream emits run_complete and closes when run is terminal."""
        await store.register_run("run-1", "trace-1", {})
        await store.update_run_status("run-1", "completed")

        async with client.stream("GET", "/api/observatory/runs/run-1/stream") as resp:
            body = ""
            async for chunk in resp.aiter_text():
                body += chunk
                if "run_complete" in body:
                    break

        messages = _parse_sse_messages(body)
        complete_msgs = [m for m in messages if m.get("event") == "run_complete"]
        assert len(complete_msgs) == 1
        data = json.loads(complete_msgs[0]["data"])
        assert data["status"] == "completed"

    async def test_failed_run_complete(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        """Stream emits run_complete with failed status."""
        await store.register_run("run-1", "trace-1", {})
        await store.update_run_status("run-1", "failed", error="boom")

        async with client.stream("GET", "/api/observatory/runs/run-1/stream") as resp:
            body = ""
            async for chunk in resp.aiter_text():
                body += chunk
                if "run_complete" in body:
                    break

        messages = _parse_sse_messages(body)
        complete_msgs = [m for m in messages if m.get("event") == "run_complete"]
        assert len(complete_msgs) == 1
        data = json.loads(complete_msgs[0]["data"])
        assert data["status"] == "failed"

    async def test_cursor_resumption(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        """Last-Event-ID resumes from cursor — only newer events delivered."""
        await store.register_run("run-1", "trace-1", {})
        await store.save_events_batch(
            "run-1",
            [
                _make_event("agent.start", payload={"agent_name": "a"}),
                _make_event(
                    "agent.complete",
                    payload={"agent_name": "a", "total_steps": 1, "termination_reason": "done"},
                    time_offset_ms=10,
                ),
            ],
        )
        await store.update_run_status("run-1", "completed")

        # Resume from event id 1 — should only get event id 2 + run_complete
        async with client.stream(
            "GET",
            "/api/observatory/runs/run-1/stream",
            headers={"Last-Event-ID": "1"},
        ) as resp:
            body = ""
            async for chunk in resp.aiter_text():
                body += chunk
                if "run_complete" in body:
                    break

        messages = _parse_sse_messages(body)
        trace_msgs = [m for m in messages if m.get("event") == "trace"]
        assert len(trace_msgs) == 1
        data = json.loads(trace_msgs[0]["data"])
        assert data["event_type"] == "agent.complete"

    async def test_level_filtering(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        """min_level filters out events below the threshold."""
        await store.register_run("run-1", "trace-1", {})
        await store.save_events_batch(
            "run-1",
            [
                _make_event("agent.start", level="info", payload={"agent_name": "a"}),
                _make_event(
                    "llm.response",
                    level="debug",
                    payload={"usage": {"input_tokens": 10, "output_tokens": 5}},
                    time_offset_ms=10,
                ),
                _make_event(
                    "span.start",
                    level="verbose",
                    payload={"name": "x"},
                    time_offset_ms=20,
                ),
            ],
        )
        await store.update_run_status("run-1", "completed")

        # Default min_level=info should only include info events
        async with client.stream("GET", "/api/observatory/runs/run-1/stream") as resp:
            body = ""
            async for chunk in resp.aiter_text():
                body += chunk
                if "run_complete" in body:
                    break

        messages = _parse_sse_messages(body)
        trace_msgs = [m for m in messages if m.get("event") == "trace"]
        assert len(trace_msgs) == 1
        data = json.loads(trace_msgs[0]["data"])
        assert data["event_type"] == "agent.start"

        # min_level=debug should include info + debug
        async with client.stream("GET", "/api/observatory/runs/run-1/stream?min_level=debug") as resp:
            body = ""
            async for chunk in resp.aiter_text():
                body += chunk
                if "run_complete" in body:
                    break

        messages = _parse_sse_messages(body)
        trace_msgs = [m for m in messages if m.get("event") == "trace"]
        assert len(trace_msgs) == 2

    async def test_cache_headers(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        """SSE response has correct cache and buffering headers."""
        await store.register_run("run-1", "trace-1", {})
        await store.update_run_status("run-1", "completed")

        async with client.stream("GET", "/api/observatory/runs/run-1/stream") as resp:
            assert resp.headers["cache-control"] == "no-cache"
            assert resp.headers["x-accel-buffering"] == "no"
            # Drain the stream
            async for _ in resp.aiter_text():
                pass


class TestSSEStreamUnit:
    """Unit tests for the stream_run_events generator directly."""

    async def test_keepalive_emitted(self, store: InMemoryPersistentTraceStore) -> None:
        """Keepalive comment sent after idle period."""
        import nanitics.observatory.streaming as streaming_mod
        from nanitics.observatory.streaming import stream_run_events

        # Patch intervals for fast testing
        original_poll = streaming_mod._POLL_INTERVAL_S
        original_keepalive = streaming_mod._KEEPALIVE_INTERVAL_S
        streaming_mod._POLL_INTERVAL_S = 0.01
        streaming_mod._KEEPALIVE_INTERVAL_S = 0.03

        try:
            await store.register_run("run-1", "trace-1", {})

            class FakeRequest:
                _disconnected = False
                _call_count = 0

                async def is_disconnected(self) -> bool:
                    self._call_count += 1
                    # Disconnect after enough polls to allow a keepalive
                    if self._call_count > 8:
                        self._disconnected = True
                    return self._disconnected

                @property
                def headers(self) -> dict:
                    return {}

            request = FakeRequest()
            chunks = [
                chunk
                async for chunk in stream_run_events(
                    store,
                    "run-1",
                    request,  # type: ignore[arg-type]
                    min_level="info",
                )
            ]

            keepalives = [c for c in chunks if c.startswith(": keepalive")]
            assert len(keepalives) >= 1
        finally:
            streaming_mod._POLL_INTERVAL_S = original_poll
            streaming_mod._KEEPALIVE_INTERVAL_S = original_keepalive

    async def test_no_events_lost_on_terminal_transition(self, store: InMemoryPersistentTraceStore) -> None:
        """Events saved between query_events and get_run are not lost.

        Simulates the race: query_events returns empty, then events are
        saved and run goes terminal before the next poll. The drain loop
        must deliver those events before emitting run_complete.
        """
        import nanitics.observatory.streaming as streaming_mod
        from nanitics.observatory.streaming import stream_run_events

        original_poll = streaming_mod._POLL_INTERVAL_S
        streaming_mod._POLL_INTERVAL_S = 0.01

        try:
            await store.register_run("run-1", "trace-1", {})

            # Track calls to get_run so we can inject events right before
            # the terminal status is observed.
            original_get_run = store.get_run
            _get_run_calls = 0

            async def patched_get_run(run_id: str) -> RunRecord | None:
                nonlocal _get_run_calls
                _get_run_calls += 1
                if _get_run_calls == 1:
                    # First time terminal check fires: inject late events
                    # and set terminal status — simulating the race.
                    await store.save_events_batch(
                        "run-1",
                        [
                            _make_event("agent.start", payload={"agent_name": "late"}),
                            _make_event(
                                "agent.complete",
                                payload={"agent_name": "late", "total_steps": 1, "termination_reason": "done"},
                                time_offset_ms=10,
                            ),
                        ],
                    )
                    await store.update_run_status("run-1", "completed")
                return await original_get_run(run_id)

            store.get_run = patched_get_run  # type: ignore[method-assign]

            class FakeRequest:
                _call_count = 0

                async def is_disconnected(self) -> bool:
                    self._call_count += 1
                    return self._call_count > 30

            request = FakeRequest()
            chunks = [
                chunk
                async for chunk in stream_run_events(
                    store,
                    "run-1",
                    request,  # type: ignore[arg-type]
                    min_level="info",
                )
            ]

            trace_chunks = [c for c in chunks if "event: trace" in c]
            complete_chunks = [c for c in chunks if "event: run_complete" in c]
            assert len(trace_chunks) == 2, f"Expected 2 trace events, got {len(trace_chunks)}"
            assert len(complete_chunks) == 1
            # Trace events must come before run_complete
            last_trace_idx = max(i for i, c in enumerate(chunks) if "event: trace" in c)
            complete_idx = next(i for i, c in enumerate(chunks) if "event: run_complete" in c)
            assert last_trace_idx < complete_idx
        finally:
            streaming_mod._POLL_INTERVAL_S = original_poll

    async def test_events_arrive_during_stream(self, store: InMemoryPersistentTraceStore) -> None:
        """Events added during streaming are picked up by polling."""
        import nanitics.observatory.streaming as streaming_mod
        from nanitics.observatory.streaming import stream_run_events

        original_poll = streaming_mod._POLL_INTERVAL_S
        streaming_mod._POLL_INTERVAL_S = 0.01

        try:
            await store.register_run("run-1", "trace-1", {})

            class FakeRequest:
                _call_count = 0

                async def is_disconnected(self) -> bool:
                    self._call_count += 1
                    return self._call_count > 10

                @property
                def headers(self) -> dict:
                    return {}

            request = FakeRequest()

            # Add events after a short delay
            async def add_events() -> None:
                await asyncio.sleep(0.03)
                await store.save_events_batch(
                    "run-1",
                    [_make_event("agent.start", payload={"agent_name": "a"})],
                )

            task = asyncio.create_task(add_events())
            chunks = [
                chunk
                async for chunk in stream_run_events(
                    store,
                    "run-1",
                    request,  # type: ignore[arg-type]
                    min_level="info",
                )
            ]
            await task

            trace_chunks = [c for c in chunks if "event: trace" in c]
            assert len(trace_chunks) >= 1
        finally:
            streaming_mod._POLL_INTERVAL_S = original_poll

    async def test_drain_repoll_emits_late_events(self, store: InMemoryPersistentTraceStore) -> None:
        """Drain re-poll emits events that arrived between query_events and get_run."""
        import nanitics.observatory.streaming as streaming_mod
        from nanitics.observatory.streaming import stream_run_events

        original_poll = streaming_mod._POLL_INTERVAL_S
        streaming_mod._POLL_INTERVAL_S = 0.01

        try:
            await store.register_run("run-1", "trace-1", {})
            # Pre-complete the run and seed one event
            await store.save_events_batch(
                "run-1",
                [_make_event("agent.start", payload={"agent_name": "pre"})],
            )
            await store.update_run_status("run-1", "completed")

            # Intercept query_events: first call returns normal events,
            # second call returns empty (triggers terminal check),
            # third call (drain) returns a late event.
            original_query = store.query_events
            _call_count = 0

            async def patched_query(
                run_id: str, *, levels: list | None = None, after_id: int | None = None, limit: int = 100
            ) -> list:
                nonlocal _call_count
                _call_count += 1
                if _call_count == 1:
                    return await original_query(run_id, levels=levels, after_id=after_id, limit=limit)
                if _call_count == 2:
                    return []  # trigger terminal check
                if _call_count == 3:
                    # Drain: inject a late event
                    late_payload = {"agent_name": "late", "total_steps": 1, "termination_reason": "done"}
                    await store.save_events_batch(
                        "run-1",
                        [_make_event("agent.complete", payload=late_payload, time_offset_ms=50)],
                    )
                    return await original_query(run_id, levels=levels, after_id=after_id, limit=limit)
                return []

            store.query_events = patched_query  # type: ignore[method-assign]

            class FakeRequest:
                _call_count = 0

                async def is_disconnected(self) -> bool:
                    self._call_count += 1
                    return self._call_count > 30

                @property
                def headers(self) -> dict:
                    return {}

            request = FakeRequest()
            chunks = [
                chunk
                async for chunk in stream_run_events(
                    store,
                    "run-1",
                    request,  # type: ignore[arg-type]
                    min_level="info",
                )
            ]

            trace_chunks = [c for c in chunks if "event: trace" in c]
            complete_chunks = [c for c in chunks if "event: run_complete" in c]
            # Should have events from first poll + drain events
            assert len(trace_chunks) >= 2
            assert len(complete_chunks) == 1
        finally:
            streaming_mod._POLL_INTERVAL_S = original_poll

    async def test_full_batch_skips_sleep(self, store: InMemoryPersistentTraceStore) -> None:
        """When query_events returns a full batch, the loop continues without sleeping."""
        import nanitics.observatory.streaming as streaming_mod
        from nanitics.observatory.streaming import stream_run_events

        original_poll = streaming_mod._POLL_INTERVAL_S
        streaming_mod._POLL_INTERVAL_S = 0.01

        try:
            await store.register_run("run-1", "trace-1", {})
            # Seed >= _POLL_BATCH_SIZE events so the first poll returns a full batch
            batch = [
                _make_event(
                    "agent.step",
                    payload={"agent_name": "a", "step": i, "total_steps": 0, "termination_reason": ""},
                    time_offset_ms=i,
                )
                for i in range(streaming_mod._POLL_BATCH_SIZE + 5)
            ]
            await store.save_events_batch("run-1", batch)
            await store.update_run_status("run-1", "completed")

            class FakeRequest:
                _call_count = 0

                async def is_disconnected(self) -> bool:
                    self._call_count += 1
                    return self._call_count > 20

                @property
                def headers(self) -> dict:
                    return {}

            request = FakeRequest()
            chunks = [
                chunk
                async for chunk in stream_run_events(
                    store,
                    "run-1",
                    request,  # type: ignore[arg-type]
                    min_level="info",
                )
            ]

            trace_chunks = [c for c in chunks if "event: trace" in c]
            # All events delivered (first batch of 50 + second poll for remaining)
            assert len(trace_chunks) == streaming_mod._POLL_BATCH_SIZE + 5
        finally:
            streaming_mod._POLL_INTERVAL_S = original_poll
