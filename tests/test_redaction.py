"""Tests for the RedactionHook protocol and its wire-in points.

Covers:

- The protocol surface (runtime-checkable, single method).
- :class:`TraceCollector` wiring: hook applied to the record payload and
  to the SSE queue payload; exception propagation (fail-closed);
  no-hook default path is unchanged.
- :class:`TracedExecutor` wiring: ``execute(redaction_hook=...)``
  forwards to the internal collector so persisted events reflect the
  redaction.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from nanitics.infrastructure.observability.collector import TraceCollector
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import (
    AgentStartEvent,
    LLMRequestEvent,
    TraceEvent,
)
from nanitics.infrastructure.observability.executor import TracedExecutor
from nanitics.infrastructure.observability.redaction import RedactionHook
from nanitics.infrastructure.observability.storage import (
    InMemoryPersistentTraceStore,
    PersistentTraceStore,
)

# --- Helpers ---


def _make_mock_store() -> AsyncMock:
    store = AsyncMock(spec=PersistentTraceStore)
    store.save_events_batch = AsyncMock()
    return store


def _make_agent_start(agent_name: str = "test-agent") -> AgentStartEvent:
    return AgentStartEvent(
        trace_id="trace-1",
        span_id="span-1",
        parent_span_id=None,
        timestamp=datetime.now(UTC),
        agent_name=agent_name,
        task_input="do things",
        tools_available=["tool_a"],
    )


def _make_llm_request(system_prompt: str = "you are helpful") -> LLMRequestEvent:
    return LLMRequestEvent(
        trace_id="trace-1",
        span_id="span-1",
        parent_span_id=None,
        timestamp=datetime.now(UTC),
        model_name="claude-3",
        messages=[{"role": "user", "content": "hi"}],
        system_prompt=system_prompt,
    )


class _ReplaceAgentName:
    """Example hook that scrubs :class:`AgentStartEvent.agent_name`."""

    def __init__(self, replacement: str = "[REDACTED]") -> None:
        self._replacement = replacement
        self.calls: list[TraceEvent] = []

    def redact(self, event: TraceEvent) -> TraceEvent:
        self.calls.append(event)
        if isinstance(event, AgentStartEvent):
            return event.model_copy(update={"agent_name": self._replacement})
        return event


class _ReplacePrompt:
    """Example hook that scrubs :class:`LLMRequestEvent.system_prompt`."""

    def __init__(self, replacement: str = "[REDACTED PROMPT]") -> None:
        self._replacement = replacement

    def redact(self, event: TraceEvent) -> TraceEvent:
        if isinstance(event, LLMRequestEvent):
            return event.model_copy(update={"system_prompt": self._replacement})
        return event


class _AlwaysRaises:
    """Example hook that fails closed."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def redact(self, event: TraceEvent) -> TraceEvent:
        raise self._exc


class _Identity:
    """Example hook that returns events unchanged."""

    def redact(self, event: TraceEvent) -> TraceEvent:
        return event


# --- Protocol surface ---


class TestProtocolSurface:
    """The RedactionHook protocol is runtime-checkable and has a single method."""

    def test_runtime_checkable_accepts_conforming_class(self) -> None:
        assert isinstance(_Identity(), RedactionHook)

    def test_runtime_checkable_rejects_non_conforming_class(self) -> None:
        class NotAHook:
            pass

        assert not isinstance(NotAHook(), RedactionHook)

    def test_protocol_exposes_redact_method(self) -> None:
        hook = _Identity()
        event = _make_agent_start()
        returned = hook.redact(event)
        assert returned is event


# --- TraceCollector wiring ---


class TestCollectorRedactionWiring:
    """The hook transforms both the persisted record and the queue payload."""

    async def test_hook_transforms_persistence_payload(self) -> None:
        store = _make_mock_store()
        hook = _ReplaceAgentName(replacement="scrubbed")
        collector = TraceCollector(
            store=store,
            parent_id="run-1",
            flush_interval=10,
            redaction_hook=hook,
        )

        collector.handle(_make_agent_start(agent_name="real-agent"))
        await collector.flush()

        records = store.save_events_batch.call_args[0][1]
        assert len(records) == 1
        assert records[0].payload["agent_name"] == "scrubbed"
        # The hook saw the un-redacted input.
        assert isinstance(hook.calls[0], AgentStartEvent)
        assert hook.calls[0].agent_name == "real-agent"
        await collector.close()

    async def test_hook_transforms_queue_payload(self) -> None:
        store = _make_mock_store()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        hook = _ReplaceAgentName(replacement="scrubbed")
        collector = TraceCollector(
            store=store,
            parent_id="run-1",
            queue=queue,
            min_level="info",
            flush_interval=10,
            redaction_hook=hook,
        )

        collector.handle(_make_agent_start(agent_name="real-agent"))

        msg = queue.get_nowait()
        assert msg["payload"]["agent_name"] == "scrubbed"
        await collector.close()

    async def test_hook_called_exactly_once_per_event(self) -> None:
        store = _make_mock_store()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        hook = _ReplaceAgentName()
        collector = TraceCollector(
            store=store,
            parent_id="run-1",
            queue=queue,
            min_level="info",
            flush_interval=10,
            redaction_hook=hook,
        )

        collector.handle(_make_agent_start())

        assert len(hook.calls) == 1
        await collector.close()

    async def test_hook_exception_propagates_and_skips_persistence(self) -> None:
        store = _make_mock_store()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        hook = _AlwaysRaises(RuntimeError("redaction failed"))
        collector = TraceCollector(
            store=store,
            parent_id="run-1",
            queue=queue,
            min_level="info",
            flush_interval=10,
            redaction_hook=hook,
        )

        with pytest.raises(RuntimeError, match="redaction failed"):
            collector.handle(_make_agent_start())

        # Event neither persisted nor enqueued.
        assert collector._buffer == []
        assert queue.empty()
        await collector.close()

    async def test_no_hook_default_matches_original_behavior(self) -> None:
        """Without a hook, the collector emits events verbatim."""
        store = _make_mock_store()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        collector = TraceCollector(
            store=store,
            parent_id="run-1",
            queue=queue,
            min_level="info",
            flush_interval=10,
        )

        event = _make_agent_start(agent_name="real-agent")
        collector.handle(event)
        await collector.flush()

        records = store.save_events_batch.call_args[0][1]
        assert records[0].payload["agent_name"] == "real-agent"
        msg = queue.get_nowait()
        assert msg["payload"]["agent_name"] == "real-agent"
        await collector.close()


# --- TracedExecutor wiring ---


class TestExecutorRedactionWiring:
    """``TracedExecutor.execute(redaction_hook=...)`` forwards to the collector."""

    async def test_execute_forwards_hook_and_redacts_persisted_events(self) -> None:
        store = InMemoryPersistentTraceStore()
        executor = TracedExecutor(store)
        hook = _ReplacePrompt(replacement="[SCRUBBED PROMPT]")

        async def workload(emitter: EventEmitter, run_id: str) -> str:
            del run_id  # unused in this factory
            emitter.emit(
                LLMRequestEvent(
                    trace_id=emitter.trace_id,
                    span_id=emitter.span_id,
                    parent_span_id=emitter.parent_span_id,
                    model_name="claude-3",
                    messages=[{"role": "user", "content": "hi"}],
                    system_prompt="you are helpful with secret context",
                )
            )
            return "done"

        run_id, result = await executor.execute(workload, redaction_hook=hook)
        assert result == "done"

        events = await store.query_events(run_id)
        llm_events = [e for e in events if e.payload.get("event_type") == "llm.request"]
        assert len(llm_events) == 1
        assert llm_events[0].payload["system_prompt"] == "[SCRUBBED PROMPT]"

    async def test_execute_without_hook_persists_unchanged_events(self) -> None:
        store = InMemoryPersistentTraceStore()
        executor = TracedExecutor(store)

        async def workload(emitter: EventEmitter, run_id: str) -> str:
            del run_id  # unused in this factory
            emitter.emit(
                LLMRequestEvent(
                    trace_id=emitter.trace_id,
                    span_id=emitter.span_id,
                    parent_span_id=emitter.parent_span_id,
                    model_name="claude-3",
                    messages=[{"role": "user", "content": "hi"}],
                    system_prompt="original prompt",
                )
            )
            return "done"

        run_id, _ = await executor.execute(workload)

        events = await store.query_events(run_id)
        llm_events = [e for e in events if e.payload.get("event_type") == "llm.request"]
        assert llm_events[0].payload["system_prompt"] == "original prompt"
