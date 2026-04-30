"""Public ``analyze()`` entry point and :class:`AdvisorReport` data model.

This module composes the rubric and ranking primitives
(:mod:`self_improver.advisor.rubric`, :mod:`self_improver.advisor.ranking`) with the
parallel specialist runner (:func:`self_improver.advisor._specialists.run_specialist`)
into the single callable :func:`analyze`. The callable is the
adopter-facing Python API; the CLI wraps it without reaching into the
runtime modules.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, TypeAdapter

from nanitics.infrastructure.llm.protocol import LLMClient
from nanitics.infrastructure.observability.emitter import (
    EventEmitter,
    InMemoryEmitter,
)
from nanitics.infrastructure.observability.events import TraceEvent, Usage
from self_improver.advisor._specialists import _LAUNCH_TARGET_DIMENSIONS, run_specialist
from self_improver.advisor._usage import aggregate_usage
from self_improver.advisor.proposal import Proposal, RubricSource
from self_improver.advisor.ranking import rank_proposals
from self_improver.advisor.rubric import load_rubrics
from self_improver.advisor.trace_adapter import TraceAdapter, load_trace

# A single ``TypeAdapter`` instance validates pre-loaded
# :class:`TraceEvent` inputs without the per-call construction cost the
# Pydantic docs warn about.
_TRACE_EVENT_ADAPTER: TypeAdapter[TraceEvent] = TypeAdapter(TraceEvent)


class AdvisorReport(BaseModel):
    """The single composed output of a full advisor run.

    A frozen Pydantic model so callers can round-trip through JSON without
    losing fidelity and the public surface remains stable across versions.

    Attributes:
        trace_id: The id of the trace analyzed. For file-loaded traces this
            matches the envelope's ``trace_id``; for event-list inputs it
            is taken from the first event's ``trace_id``.
        generated_at: UTC timestamp captured at :func:`analyze` entry.
        proposals: The already-ranked proposal list. Sorted by severity
            (critical → warning → observation), then descending
            ``ranking_score``, then ASCII-ascending ``rubric_id`` as the
            deterministic tie-break.
        usage: Aggregated input/output token usage across every LLM call
            in the run.
        rubric_counts: ``{RubricSource.BUILTIN: n, RubricSource.CUSTOM: m}``
            derived from the ranked proposals' ``rubric_source`` values.
            Sources with zero proposals are omitted from the mapping.
        target_dimensions_analyzed: Ordered list of the target dimensions
            the run covered. At launch this is always the three launch
            specialists' dimensions; the field is a list rather than a
            constant because future expansions (e.g., the deferred
            ``agent_strategy`` specialist) will extend it.
    """

    model_config = ConfigDict(frozen=True)

    trace_id: str
    generated_at: datetime
    proposals: list[Proposal]
    usage: Usage
    rubric_counts: dict[RubricSource, int]
    target_dimensions_analyzed: list[str]


def _normalize_trace(
    trace: Path | list[TraceEvent],
    *,
    adapter: TraceAdapter | None,
) -> tuple[list[TraceEvent], str]:
    """Return (events, trace_id) from either a path or pre-loaded events."""
    if isinstance(trace, Path):
        loaded_events = load_trace(trace, adapter=adapter)
        payload = json.loads(trace.read_text(encoding="utf-8"))
        return loaded_events, payload["trace_id"]

    validated_events: list[TraceEvent] = [_TRACE_EVENT_ADAPTER.validate_python(entry) for entry in trace]
    if not validated_events:
        raise ValueError("trace event list must contain at least one event to derive a trace_id")
    return validated_events, validated_events[0].trace_id


async def analyze(
    trace: Path | list[TraceEvent],
    *,
    llm_client: LLMClient,
    rubrics: list[Path] | None = None,
    adapter: TraceAdapter | None = None,
    emitter: EventEmitter | None = None,
) -> AdvisorReport:
    """Analyze ``trace`` and return an :class:`AdvisorReport`.

    Composes the whole advisor runtime: trace ingestion, rubric loading,
    sequential-then-parallel specialist dispatch, ranking, and report
    construction. The first launch specialist runs alone; once it
    completes, the remaining specialists run concurrently via
    :func:`asyncio.gather`. The staggered pattern lets a cache-aware LLM
    client write the shared trace prefix on the first call and serve it
    as a cache read to the parallel fan-out, which three fully-parallel
    cold calls cannot do. Any specialist exception propagates to the
    caller without masking. The function does **not** write files —
    callers who want artifacts on disk pass the returned
    :class:`AdvisorReport` to :func:`write_report`.

    Args:
        trace: Either a :class:`pathlib.Path` to a Nanitics trace envelope
            (loaded via ``adapter`` or the default
            :class:`self_improver.advisor.NaniticsTraceAdapter`) or a pre-loaded
            list of :class:`TraceEvent` values. When a list is supplied,
            ``adapter`` is ignored.
        llm_client: LLM client shared by every specialist. Required —
            programmatic callers wire their own provider explicitly.
        rubrics: Optional adopter-custom rubric file/directory paths merged
            with the builtin corpus. Passed through to
            :func:`load_rubrics` with ``include_builtins=True``.
        adapter: Optional alternate :class:`TraceAdapter`. Ignored when
            ``trace`` is a pre-loaded event list.
        emitter: Optional caller-supplied :class:`EventEmitter`. When
            ``None``, a local :class:`InMemoryEmitter` is constructed with
            an ``advisor-`` prefixed trace id so the advisor's own run is
            traced without colliding with the analyzed trace's id.

    Returns:
        A :class:`AdvisorReport` with ranked proposals, summed usage, and
        metadata.

    Raises:
        Exception: Any exception raised during trace ingestion, rubric
            loading, or specialist execution propagates to the caller.
            Specialist failures are not swallowed.
    """
    events, trace_id = _normalize_trace(trace, adapter=adapter)
    loaded_rubrics = load_rubrics(rubrics, include_builtins=True)
    resolved_emitter: EventEmitter = emitter if emitter is not None else InMemoryEmitter(trace_id=f"advisor-{trace_id}")

    # Stagger the dispatch: the first specialist runs alone so a
    # cache-aware client can write the shared trace prefix; the
    # remaining specialists then fan out in parallel and read the
    # already-cached prefix. Three fully-parallel cold calls would all
    # race to write, defeating the cross-specialist cache.
    first_dimension, *remaining_dimensions = _LAUNCH_TARGET_DIMENSIONS
    first_proposals = await run_specialist(
        target_dimension=first_dimension,
        rubrics=loaded_rubrics,
        trace_events=events,
        llm_client=llm_client,
        emitter=resolved_emitter,
    )
    remaining_results = await asyncio.gather(
        *(
            run_specialist(
                target_dimension=target_dimension,
                rubrics=loaded_rubrics,
                trace_events=events,
                llm_client=llm_client,
                emitter=resolved_emitter,
            )
            for target_dimension in remaining_dimensions
        )
    )
    specialist_results = [first_proposals, *remaining_results]
    proposals: list[Proposal] = [p for specialist_proposals in specialist_results for p in specialist_proposals]
    ranked = rank_proposals(proposals)
    rubric_counts = dict(Counter(p.rubric_source for p in ranked))
    usage = aggregate_usage(resolved_emitter)

    return AdvisorReport(
        trace_id=trace_id,
        generated_at=datetime.now(UTC),
        proposals=ranked,
        usage=usage,
        rubric_counts=rubric_counts,
        target_dimensions_analyzed=list(_LAUNCH_TARGET_DIMENSIONS),
    )


def write_report(
    report: AdvisorReport,
    *,
    json_path: Path | None = None,
    markdown_path: Path | None = None,
) -> None:
    """Atomically write ``report`` to disk as JSON and/or Markdown.

    Either or both paths may be supplied — writes are independent. When
    a path is ``None``, that format is skipped. Each output is staged in
    a sibling ``.tmp`` file and :func:`os.replace`'d into place, so a
    crash mid-write cannot leave a half-written artifact.

    Args:
        report: The :class:`AdvisorReport` to render.
        json_path: Optional path for the JSON artifact. Rendered via
            :class:`self_improver.advisor.JSONFormatter`.
        markdown_path: Optional path for the Markdown artifact. Rendered
            via :class:`self_improver.advisor.MarkdownFormatter`.
    """
    from self_improver.advisor.formatters import JSONFormatter, MarkdownFormatter

    if json_path is not None:
        _atomic_write(json_path, JSONFormatter().render(report))
    if markdown_path is not None:
        _atomic_write(markdown_path, MarkdownFormatter().render(report))


def _atomic_write(target: Path, content: str) -> None:
    """Stage ``content`` to a sibling ``.tmp`` file and rename into place."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, target)


__all__ = [
    "AdvisorReport",
    "analyze",
    "write_report",
]
