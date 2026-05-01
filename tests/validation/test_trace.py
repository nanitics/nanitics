"""Unit tests for `validation.helpers.trace`.

Covers the derived summary block, bare-filename resolution, envelope
serialisation, and `VALIDATION_TRACE_DIR` override.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from nanitics.infrastructure.observability.emitter import InMemoryEmitter
from nanitics.infrastructure.observability.events import (
    AgentStepEvent,
    LLMResponseEvent,
    ToolInvokeEvent,
    ToolResultEvent,
    Usage,
)
from validation.helpers.trace import save_trace, validation_trace_dir


def _populated_emitter(model: str = "claude-haiku-4-5-20251001") -> InMemoryEmitter:
    emitter = InMemoryEmitter(trace_id="trace-1")
    # Two LLM responses for token accumulation.
    for i in range(2):
        emitter.emit(
            LLMResponseEvent(
                trace_id=emitter.trace_id,
                span_id=emitter.span_id,
                timestamp=datetime(2026, 4, 14, 12, i, 0, tzinfo=UTC),
                model_name=model,
                content="hi",
                usage=Usage(input_tokens=10, output_tokens=5),
                duration_ms=50,
            )
        )
    emitter.emit(
        AgentStepEvent(
            trace_id=emitter.trace_id,
            span_id=emitter.span_id,
            timestamp=datetime(2026, 4, 14, 12, 2, 0, tzinfo=UTC),
            agent_name="a",
            step_number=1,
        )
    )
    emitter.emit(
        ToolInvokeEvent(
            trace_id=emitter.trace_id,
            span_id=emitter.span_id,
            timestamp=datetime(2026, 4, 14, 12, 3, 0, tzinfo=UTC),
            tool_call_id="c",
            tool_name="echo",
            parameters={"x": 1},
        )
    )
    emitter.emit(
        ToolResultEvent(
            trace_id=emitter.trace_id,
            span_id=emitter.span_id,
            timestamp=datetime(2026, 4, 14, 12, 4, 0, tzinfo=UTC),
            tool_call_id="c",
            tool_name="echo",
            result="ok",
            success=True,
            duration_ms=10,
        )
    )
    return emitter


def test_save_trace_envelope_and_summary(tmp_path: Path) -> None:
    emitter = _populated_emitter()
    target = tmp_path / "out.json"
    written = save_trace(emitter, target)
    assert written == target

    data = json.loads(target.read_text())
    assert data["trace_id"] == "trace-1"
    assert data["exported_at"].endswith("Z")
    assert data["script"]  # non-empty
    summary = data["summary"]
    assert summary["event_count"] == 5
    assert summary["total_input_tokens"] == 20
    assert summary["total_output_tokens"] == 10
    assert summary["tool_calls"] == 1
    assert summary["tool_results"] == 1
    assert summary["iterations"] == 1
    assert summary["provider"] == "anthropic"
    assert summary["model"] == "claude-haiku-4-5-20251001"
    assert summary["duration_ms"] > 0
    assert summary["error_events"] == 0
    assert len(data["events"]) == 5


def test_save_trace_with_empty_events(tmp_path: Path) -> None:
    emitter = InMemoryEmitter(trace_id="empty")
    target = tmp_path / "empty.json"
    save_trace(emitter, target)
    data = json.loads(target.read_text())
    assert data["summary"]["event_count"] == 0
    assert data["summary"]["duration_ms"] == 0
    assert data["summary"]["provider"] is None
    assert data["summary"]["model"] is None
    assert data["events"] == []


def test_save_trace_provider_from_litellm_string(tmp_path: Path) -> None:
    emitter = _populated_emitter(model="openai/gpt-4o-mini")
    save_trace(emitter, tmp_path / "a.json")
    data = json.loads((tmp_path / "a.json").read_text())
    assert data["summary"]["provider"] == "openai"
    assert data["summary"]["model"] == "openai/gpt-4o-mini"


def test_save_trace_provider_from_gpt_prefix(tmp_path: Path) -> None:
    emitter = _populated_emitter(model="gpt-4o-mini-mini")
    save_trace(emitter, tmp_path / "a.json")
    data = json.loads((tmp_path / "a.json").read_text())
    assert data["summary"]["provider"] == "openai"


def test_save_trace_provider_from_mistral_prefix(tmp_path: Path) -> None:
    emitter = _populated_emitter(model="mistral-large-latest")
    save_trace(emitter, tmp_path / "a.json")
    data = json.loads((tmp_path / "a.json").read_text())
    assert data["summary"]["provider"] == "mistral"


def test_save_trace_provider_unknown_prefix(tmp_path: Path) -> None:
    emitter = _populated_emitter(model="custom-model")
    save_trace(emitter, tmp_path / "a.json")
    data = json.loads((tmp_path / "a.json").read_text())
    assert data["summary"]["provider"] is None
    assert data["summary"]["model"] == "custom-model"


def test_save_trace_counts_error_events(tmp_path: Path) -> None:
    emitter = InMemoryEmitter(trace_id="err")
    emitter.emit(
        AgentStepEvent(
            trace_id="err",
            span_id=emitter.span_id,
            agent_name="a",
            step_number=1,
        )
    )
    # Use a real ErrorEvent-named type.
    from nanitics.infrastructure.observability.events import AgentErrorEvent

    emitter.emit(
        AgentErrorEvent(
            trace_id="err",
            span_id=emitter.span_id,
            agent_name="a",
            error_type="BoomError",
            error_message="boom",
            error_metadata={},
        )
    )
    save_trace(emitter, tmp_path / "err.json")
    data = json.loads((tmp_path / "err.json").read_text())
    assert data["summary"]["error_events"] == 1


def test_save_trace_bare_filename_uses_validation_trace_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VALIDATION_TRACE_DIR", str(tmp_path))
    emitter = InMemoryEmitter(trace_id="trace-bare")
    written = save_trace(emitter, "out.json")
    assert written == tmp_path / "out.json"
    assert written.exists()


def test_validation_trace_dir_respects_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    override = tmp_path / "sub"
    monkeypatch.setenv("VALIDATION_TRACE_DIR", str(override))
    got = validation_trace_dir()
    assert got == override
    assert got.exists()


def test_validation_trace_dir_default_creates_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VALIDATION_TRACE_DIR", raising=False)
    # Redirect the repo-root computation to a tmp path.
    monkeypatch.chdir(tmp_path)

    import validation.helpers.trace as trace_mod

    repo_root = tmp_path
    monkeypatch.setattr(trace_mod, "__file__", str(repo_root / "validation" / "helpers" / "trace.py"))
    # Emulate the directory structure
    (repo_root / "validation" / "helpers").mkdir(parents=True)

    got = validation_trace_dir()
    # Strict contract: flat `<repo_root>/validation/traces`, no timestamp or
    # script-name segment, created if missing.
    assert got == repo_root / "validation" / "traces"
    assert got.exists()
    assert got.is_dir()


def test_save_trace_rerun_overwrites_same_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-running ``save_trace`` with the same bare filename must overwrite in place."""
    monkeypatch.setenv("VALIDATION_TRACE_DIR", str(tmp_path))

    first = InMemoryEmitter(trace_id="validation-smoke")
    second = InMemoryEmitter(trace_id="validation-smoke")

    first_path = save_trace(first, "smoke.json")
    second_path = save_trace(second, "smoke.json")

    assert first_path == second_path == tmp_path / "smoke.json"
    # Exactly one file at the stable path.
    assert [p.name for p in tmp_path.iterdir()] == ["smoke.json"]
    data = json.loads(second_path.read_text())
    assert data["trace_id"] == "validation-smoke"


def test_save_trace_script_name_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("sys.argv", [""])
    emitter = InMemoryEmitter(trace_id="t")
    save_trace(emitter, tmp_path / "out.json")
    data = json.loads((tmp_path / "out.json").read_text())
    assert data["script"] == "unknown"
