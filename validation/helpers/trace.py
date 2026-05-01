"""Trace export: serialize an emitter's event list to a self-contained JSON file.

See :func:`export_trace` for the on-disk schema.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from nanitics import InMemoryEmitter, TraceEvent


def _resolve_script_name() -> str:
    """Best-effort script name derivation from ``sys.argv``."""
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0:
        name = Path(argv0).name
        if name:
            return name
    return "unknown"


def _derive_summary(events: list[TraceEvent]) -> dict[str, Any]:
    """Compute the derived summary block from the event list."""
    event_count = len(events)
    duration_ms = 0.0
    if event_count >= 2:
        first_ts = events[0].timestamp
        last_ts = events[-1].timestamp
        duration_ms = (last_ts - first_ts).total_seconds() * 1000

    total_input_tokens = 0
    total_output_tokens = 0
    cache_creation_tokens = 0
    cache_read_tokens = 0
    tool_calls = 0
    tool_results = 0
    iterations = 0
    error_events = 0
    provider: str | None = None
    model: str | None = None
    first_llm_model: str | None = None

    for event in events:
        cls_name = type(event).__name__
        if cls_name == "LLMResponseEvent":
            usage = getattr(event, "usage", None)
            if usage is not None:
                total_input_tokens += getattr(usage, "input_tokens", 0) or 0
                total_output_tokens += getattr(usage, "output_tokens", 0) or 0
                cache_creation_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
                cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
            if first_llm_model is None:
                first_llm_model = getattr(event, "model_name", None)
        elif cls_name == "ToolInvokeEvent":
            tool_calls += 1
        elif cls_name == "ToolResultEvent":
            tool_results += 1
        elif cls_name == "AgentStepEvent":
            iterations += 1
        if "Error" in cls_name:
            error_events += 1

    model = first_llm_model
    if model is not None:
        # Best-effort provider extraction: the "anthropic/..." litellm-style string or the
        # human-readable "claude-..." / "gpt-..." prefix.
        if "/" in model:
            provider = model.split("/", 1)[0]
        elif model.startswith("claude"):
            provider = "anthropic"
        elif model.startswith("gpt"):
            provider = "openai"
        elif model.startswith("mistral"):
            provider = "mistral"

    return {
        "duration_ms": duration_ms,
        "error_events": error_events,
        "event_count": event_count,
        "iterations": iterations,
        "model": model,
        "provider": provider,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "cache_creation_tokens": cache_creation_tokens,
        "cache_read_tokens": cache_read_tokens,
    }


def save_trace(
    emitter: InMemoryEmitter,
    path: str | Path,
    *,
    script: str | None = None,
) -> Path:
    """Write ``emitter.events`` to a JSON file in the authoritative trace format.

    The envelope contains ``trace_id``, ``exported_at``, ``script``,
    ``summary`` (derived), and ``events`` (each via ``model_dump(mode="json")``).

    If ``path`` is a bare filename, it is resolved against
    :func:`validation_trace_dir` so scripts can write a short ``"smoke.json"``.

    The write is atomic: the payload is staged in a sibling ``.tmp`` file and
    renamed over the target, so a ``SIGINT`` or crash mid-write cannot leave a
    half-written trace on disk.

    Args:
        emitter: The in-memory emitter whose events are serialised.
        path: Destination file path. Absolute paths are honoured as-is.
        script: Optional explicit script label recorded in the envelope. When
            omitted, falls back to a best-effort ``sys.argv[0]`` derivation
            (unreliable under pytest — prefer passing the pytest node id).

    Returns:
        The resolved :class:`Path` that was written.
    """
    target = Path(path)
    if not target.is_absolute() and target.parent == Path():
        target = validation_trace_dir() / target.name

    target.parent.mkdir(parents=True, exist_ok=True)

    events_list: list[TraceEvent] = list(emitter.events)
    payload = {
        "trace_id": emitter.trace_id,
        "exported_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "script": script if script is not None else _resolve_script_name(),
        "summary": _derive_summary(events_list),
        "events": [event.model_dump(mode="json") for event in events_list],
    }

    tmp = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return target


def validation_trace_dir() -> Path:
    """Return the target directory for traces, creating it if needed.

    Default: ``<repo_root>/validation/traces/`` — a flat, stable directory.
    Combined with the ``save_trace(emitter, validation_trace_dir() / "<name>.json")``
    call-site convention used by every validation script, this guarantees
    re-runs of the same script overwrite the same file at a predictable
    path that ``/analyze-run`` can address deterministically.

    Override via the ``VALIDATION_TRACE_DIR`` env var (absolute path) for
    CI bundling or one-off isolation. The override is honoured as-is and
    created if missing; no timestamp or script-name segment is appended.
    """
    override = os.environ.get("VALIDATION_TRACE_DIR")
    if override:
        target = Path(override)
        target.mkdir(parents=True, exist_ok=True)
        return target

    repo_root = Path(__file__).resolve().parents[2]
    target = repo_root / "validation" / "traces"
    target.mkdir(parents=True, exist_ok=True)
    return target
