"""Unit tests for :mod:`self_improver.advisor.trace_adapter`.

Covers envelope validation, discriminated-union validation per event,
protocol conformance, and custom-adapter delegation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from self_improver.advisor import (
    MalformedTraceError,
    NaniticsTraceAdapter,
    TraceAdapter,
    load_trace,
)

from nanitics.infrastructure.observability.events import (
    AgentStartEvent,
    SpanStartEvent,
)

_FIXTURE_TRACE = Path(__file__).parent / "fixtures" / "nanitics_sample_trace.json"


def _write_envelope(path: Path, envelope: dict[str, Any]) -> None:
    path.write_text(json.dumps(envelope), encoding="utf-8")


class TestNaniticsTraceAdapter:
    def test_loads_fixture_trace_successfully(self) -> None:
        events = NaniticsTraceAdapter().load(_FIXTURE_TRACE)
        assert len(events) > 0
        # The fixture's first event is a span.start for "coordinator".
        assert isinstance(events[0], SpanStartEvent)
        assert events[0].name == "coordinator"

    def test_every_event_is_discriminated_union_instance(self) -> None:
        events = NaniticsTraceAdapter().load(_FIXTURE_TRACE)
        # Spot-check that agent.start events are typed as AgentStartEvent,
        # confirming the discriminator resolved correctly.
        agent_starts = [e for e in events if isinstance(e, AgentStartEvent)]
        assert agent_starts, "fixture should contain at least one agent.start event"
        for event in agent_starts:
            assert event.event_type == "agent.start"
            assert event.agent_name

    def test_missing_file_raises_malformed_trace_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist.json"
        with pytest.raises(MalformedTraceError) as exc_info:
            NaniticsTraceAdapter().load(missing)
        assert "cannot read file" in exc_info.value.reason

    def test_invalid_json_raises_malformed_trace_error(self, tmp_path: Path) -> None:
        target = tmp_path / "bad.json"
        target.write_text("not valid json {{{", encoding="utf-8")
        with pytest.raises(MalformedTraceError) as exc_info:
            NaniticsTraceAdapter().load(target)
        assert "invalid JSON" in exc_info.value.reason

    def test_non_object_envelope_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "bad.json"
        target.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
        with pytest.raises(MalformedTraceError) as exc_info:
            NaniticsTraceAdapter().load(target)
        assert "envelope must be a JSON object" in exc_info.value.reason

    def test_missing_events_key_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "envelope.json"
        _write_envelope(target, {"trace_id": "x"})
        with pytest.raises(MalformedTraceError) as exc_info:
            NaniticsTraceAdapter().load(target)
        assert "missing required envelope key 'events'" in exc_info.value.reason

    def test_missing_trace_id_key_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "envelope.json"
        _write_envelope(target, {"events": []})
        with pytest.raises(MalformedTraceError) as exc_info:
            NaniticsTraceAdapter().load(target)
        assert "missing required envelope key 'trace_id'" in exc_info.value.reason

    def test_non_list_events_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "envelope.json"
        _write_envelope(target, {"trace_id": "x", "events": {"not": "a list"}})
        with pytest.raises(MalformedTraceError) as exc_info:
            NaniticsTraceAdapter().load(target)
        assert "'events' must be a list" in exc_info.value.reason

    def test_unknown_event_type_raises_with_index(self, tmp_path: Path) -> None:
        target = tmp_path / "envelope.json"
        _write_envelope(
            target,
            {
                "trace_id": "x",
                "events": [
                    {
                        "event_id": "eid-1",
                        "trace_id": "x",
                        "span_id": "s1",
                        "timestamp": "2026-01-01T00:00:00Z",
                        "event_type": "not.a.real.event.type",
                    }
                ],
            },
        )
        with pytest.raises(MalformedTraceError) as exc_info:
            NaniticsTraceAdapter().load(target)
        assert "index 0" in exc_info.value.reason
        assert "TraceEvent validation" in exc_info.value.reason

    def test_malformed_event_body_raises_with_index(self, tmp_path: Path) -> None:
        target = tmp_path / "envelope.json"
        # agent.start requires an ``agent_name`` field — omit it.
        _write_envelope(
            target,
            {
                "trace_id": "x",
                "events": [
                    {
                        "event_id": "eid-1",
                        "trace_id": "x",
                        "span_id": "s1",
                        "timestamp": "2026-01-01T00:00:00Z",
                        "event_type": "agent.start",
                        "tools_available": [],
                    }
                ],
            },
        )
        with pytest.raises(MalformedTraceError) as exc_info:
            NaniticsTraceAdapter().load(target)
        assert "index 0" in exc_info.value.reason


class TestLoadTrace:
    def test_delegates_to_default_adapter_when_none_supplied(self) -> None:
        events = load_trace(_FIXTURE_TRACE)
        assert len(events) > 0

    def test_delegates_to_custom_adapter(self, tmp_path: Path) -> None:
        captured: dict[str, Path] = {}

        class RecordingAdapter:
            def load(self, source: Path) -> list[Any]:
                captured["source"] = source
                return []

        adapter: TraceAdapter = RecordingAdapter()
        fake = tmp_path / "fake.json"
        result = load_trace(fake, adapter=adapter)
        assert result == []
        assert captured["source"] == fake


class TestTraceAdapterProtocol:
    def test_default_adapter_satisfies_protocol(self) -> None:
        assert isinstance(NaniticsTraceAdapter(), TraceAdapter)

    def test_arbitrary_object_with_load_satisfies_protocol(self) -> None:
        class DuckAdapter:
            def load(self, source: Path) -> list[Any]:
                return []

        assert isinstance(DuckAdapter(), TraceAdapter)

    def test_object_without_load_does_not_satisfy_protocol(self) -> None:
        class NotAnAdapter:
            pass

        assert not isinstance(NotAnAdapter(), TraceAdapter)
