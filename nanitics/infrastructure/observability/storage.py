from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from nanitics.infrastructure.observability.events import (
    LLMResponseEvent,
    TraceEvent,
)
from nanitics.infrastructure.observability.levels import TraceLevel

RunStatus = Literal["running", "completed", "failed", "suspended", "rejected"]
"""Valid run lifecycle statuses."""


class _UnsetType:
    """Sentinel marker — distinct from ``None`` and any user-supplied value.

    Used by ``list_runs`` / ``count_runs`` to distinguish "do not filter on
    ``parent_run_id``" (the default) from the legitimate ``None`` value
    (which means "filter to top-level runs only"). See the run hierarchy
    section of the observability guide for the three-state semantics.
    """

    _instance: _UnsetType | None = None

    def __new__(cls) -> _UnsetType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "_UNSET"

    def __bool__(self) -> bool:
        return False


_UNSET: Final[_UnsetType] = _UnsetType()

DEFAULT_RUNS_LIMIT = 50
DEFAULT_EVENTS_LIMIT = 100
MAX_RUNS_LIMIT = 100
MAX_EVENTS_LIMIT = 500


class Trace(BaseModel):
    """A complete execution trace: a trace ID and its events."""

    model_config = ConfigDict(frozen=True)

    trace_id: str
    events: list[TraceEvent]


class TraceSummary(BaseModel):
    """Aggregated metadata for a trace, returned by queries.

    Token totals are summed from all ``LLMResponseEvent`` events in the trace.
    """

    model_config = ConfigDict(frozen=True)

    trace_id: str
    start_time: datetime
    end_time: datetime | None
    event_count: int
    total_input_tokens: int
    total_output_tokens: int


class TraceQuery(BaseModel):
    """Filter and pagination parameters for querying traces."""

    start_time: datetime | None = None
    end_time: datetime | None = None
    limit: int = 100
    offset: int = 0


@runtime_checkable
class TraceStore(Protocol):
    """Protocol for persisting and querying execution traces."""

    async def save_trace(self, trace: Trace) -> None:
        """Persist a complete trace."""
        ...

    async def get_trace(self, trace_id: str) -> Trace | None:
        """Retrieve a trace by ID, or ``None`` if not found."""
        ...

    async def query_traces(self, query: TraceQuery) -> list[TraceSummary]:
        """Filter and paginate traces, returning summaries."""
        ...


def _build_summary(trace: Trace) -> TraceSummary:
    """Build a ``TraceSummary`` by aggregating events in a trace."""
    events = trace.events
    start_time = events[0].timestamp
    end_time = events[-1].timestamp if len(events) > 1 else None
    total_input = 0
    total_output = 0
    for event in events:
        if isinstance(event, LLMResponseEvent):
            total_input += event.usage.input_tokens
            total_output += event.usage.output_tokens
    return TraceSummary(
        trace_id=trace.trace_id,
        start_time=start_time,
        end_time=end_time,
        event_count=len(events),
        total_input_tokens=total_input,
        total_output_tokens=total_output,
    )


class InMemoryTraceStore:
    """In-memory trace store backed by a dictionary.

    Suitable for testing and local development. Not persistent.
    """

    def __init__(self) -> None:
        self._traces: dict[str, Trace] = {}

    async def save_trace(self, trace: Trace) -> None:
        self._traces[trace.trace_id] = trace

    async def get_trace(self, trace_id: str) -> Trace | None:
        return self._traces.get(trace_id)

    async def query_traces(self, query: TraceQuery) -> list[TraceSummary]:
        matching: list[tuple[datetime, Trace]] = []

        for trace in self._traces.values():
            if not trace.events:
                continue
            trace_start = trace.events[0].timestamp
            if query.start_time is not None and trace_start < query.start_time:
                continue
            if query.end_time is not None and trace_start >= query.end_time:
                continue
            matching.append((trace_start, trace))

        matching.sort(key=lambda item: item[0], reverse=True)

        page = matching[query.offset : query.offset + query.limit]
        return [_build_summary(trace) for _, trace in page]


# ---------------------------------------------------------------------------
# Persistent trace storage — individual event persistence with query support
# ---------------------------------------------------------------------------


class TraceEventRecord(BaseModel):
    """A single trace event prepared for persistence."""

    model_config = ConfigDict(frozen=True)

    event_type: str
    level: TraceLevel
    trace_id: str
    span_id: str
    parent_span_id: str | None
    payload: dict[str, Any]  # serialised event data
    sdk_timestamp: datetime


class StoredTraceEvent(TraceEventRecord):
    """A trace event as returned from the store, with a database-assigned ID."""

    id: int


class TerminationReason(StrEnum):
    """Closed set of run termination reasons emitted by SDK strategies.

    Unknown strings from future or third-party strategies map to ``OTHER``
    with the original string preserved in ``RunResult.termination_reason_raw``.
    """

    COMPLETED = "completed"
    ITERATION_LIMIT = "iteration_limit"
    TOOL_CALL_LIMIT = "tool_call_limit"
    EVALUATION_FAILED = "evaluation_failed"
    EVALUATION_SKIPPED = "evaluation_skipped"
    CANCELLED = "cancelled"
    ERROR = "error"
    OTHER = "other"


class RunResult(BaseModel):
    """Structured result of a completed run.

    ``messages`` is intentionally absent: message logs live as trace events
    and are fetched via :meth:`PersistentTraceStore.get_run_messages`.
    """

    model_config = ConfigDict(frozen=True)

    output: str | None = None
    termination_reason: TerminationReason | None = None
    termination_reason_raw: str | None = None
    total_steps: int | None = None


class RunRecord(BaseModel):
    """A registered run with lifecycle state.

    ``parent_run_id`` links a specialist (child) run to the run that
    dispatched it. ``None`` means the run is top-level. See
    :meth:`PostgresTraceStore.register_run` for the cascade contract and the
    note on ``trace_id`` non-enforcement between parent and child.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    trace_id: str
    status: RunStatus
    started_at: datetime
    completed_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    result: RunResult | None = None
    parent_run_id: str | None = None


class TraceSummaryStats(BaseModel):
    """Aggregated statistics for all trace events under a parent."""

    model_config = ConfigDict(frozen=True)

    total_events: int
    events_by_level: dict[TraceLevel, int]
    llm_calls: int
    tool_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_duration_ms: int | None
    agent_names: list[str]
    errors: int
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0


@runtime_checkable
class PersistentTraceStore(Protocol):
    """Protocol for persisting and querying individual trace events."""

    async def save_events_batch(self, parent_id: str, events: list[TraceEventRecord]) -> None:
        """Bulk-insert trace event records."""
        ...

    async def query_events(
        self,
        parent_id: str,
        *,
        levels: list[TraceLevel] | None = None,
        event_types: list[str] | None = None,
        after_id: int | None = None,
        limit: int = DEFAULT_EVENTS_LIMIT,
    ) -> list[StoredTraceEvent]:
        """Filtered, cursor-paginated retrieval of trace events."""
        ...

    async def get_event(self, event_id: int) -> StoredTraceEvent | None:
        """Retrieve a single event by database ID."""
        ...

    async def get_summary(self, parent_id: str) -> TraceSummaryStats:
        """Return aggregated statistics for all events under *parent_id*."""
        ...

    # --- Run management ---

    async def register_run(
        self,
        run_id: str,
        trace_id: str,
        metadata: dict[str, Any],
        *,
        parent_run_id: str | None = None,
    ) -> None:
        """Register a new run with initial 'running' status."""
        ...

    async def update_run_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        error: str | None = None,
        result: RunResult | None = None,
    ) -> None:
        """Update run status (completed, failed, suspended) and optional error/result."""
        ...

    async def get_run(self, run_id: str) -> RunRecord | None:
        """Retrieve a run by ID, or ``None`` if not found."""
        ...

    async def list_runs(
        self,
        *,
        status: str | None = None,
        started_after: datetime | None = None,
        started_before: datetime | None = None,
        sort: str = "started_at_desc",
        search: str | None = None,
        parent_run_id: str | None | _UnsetType = _UNSET,
        limit: int = DEFAULT_RUNS_LIMIT,
        offset: int = 0,
    ) -> list[RunRecord]:
        """List runs with optional filters, ordered by *sort*."""
        ...

    async def count_runs(
        self,
        *,
        status: str | None = None,
        started_after: datetime | None = None,
        started_before: datetime | None = None,
        search: str | None = None,
        parent_run_id: str | None | _UnsetType = _UNSET,
    ) -> int:
        """Return total count of runs matching the given filters."""
        ...

    async def delete_run(self, run_id: str) -> bool:
        """Delete a run and all associated trace events. Return False if not found."""
        ...

    # --- Hierarchy queries ---

    async def get_span_tree(self, trace_id: str) -> list[StoredTraceEvent]:
        """Return all events for a trace ordered for tree reconstruction."""
        ...

    async def get_events_by_span(self, trace_id: str, span_id: str) -> list[StoredTraceEvent]:
        """Return events within a specific span."""
        ...


class InMemoryPersistentTraceStore:
    """In-memory implementation of :class:`PersistentTraceStore` for testing."""

    def __init__(self) -> None:
        self._events: list[StoredTraceEvent] = []
        self._id_counter: int = 0
        self._parent_index: dict[str, list[int]] = {}
        self._runs: dict[str, RunRecord] = {}

    async def save_events_batch(self, parent_id: str, events: list[TraceEventRecord]) -> None:
        indices = self._parent_index.setdefault(parent_id, [])
        for e in events:
            self._id_counter += 1
            stored = StoredTraceEvent(id=self._id_counter, **e.model_dump())
            self._events.append(stored)
            indices.append(len(self._events) - 1)

    async def query_events(
        self,
        parent_id: str,
        *,
        levels: list[TraceLevel] | None = None,
        event_types: list[str] | None = None,
        after_id: int | None = None,
        limit: int = DEFAULT_EVENTS_LIMIT,
    ) -> list[StoredTraceEvent]:
        indices = self._parent_index.get(parent_id, [])
        result: list[StoredTraceEvent] = []
        for idx in indices:
            e = self._events[idx]
            if after_id is not None and e.id <= after_id:
                continue
            if levels and e.level not in levels:
                continue
            if event_types and e.event_type not in event_types:
                continue
            result.append(e)
            if len(result) >= limit:
                break
        return result

    async def get_event(self, event_id: int) -> StoredTraceEvent | None:
        for e in self._events:
            if e.id == event_id:
                return e
        return None

    async def get_summary(self, parent_id: str) -> TraceSummaryStats:
        indices = self._parent_index.get(parent_id, [])
        events = [self._events[i] for i in indices]

        events_by_level: dict[TraceLevel, int] = {}
        llm_calls = 0
        tool_calls = 0
        input_tokens = 0
        output_tokens = 0
        cache_creation_tokens = 0
        cache_read_tokens = 0
        agent_names: list[str] = []
        errors = 0
        first_ts: datetime | None = None
        last_ts: datetime | None = None

        for e in events:
            events_by_level[e.level] = events_by_level.get(e.level, 0) + 1

            if e.event_type == "llm.response":
                llm_calls += 1
                usage = e.payload.get("usage", e.payload)
                input_tokens += int(usage.get("input_tokens", 0))
                output_tokens += int(usage.get("output_tokens", 0))
                cache_creation_tokens += int(usage.get("cache_creation_input_tokens", 0) or 0)
                cache_read_tokens += int(usage.get("cache_read_input_tokens", 0) or 0)

            if e.event_type == "tool.invoke":
                tool_calls += 1

            if e.event_type == "agent.start":
                name = e.payload.get("agent_name")
                if name and name not in agent_names:
                    agent_names.append(name)

            if e.event_type.startswith("agent.error") or e.event_type.startswith("workflow.error"):
                errors += 1

            if first_ts is None or e.sdk_timestamp < first_ts:
                first_ts = e.sdk_timestamp
            if last_ts is None or e.sdk_timestamp > last_ts:
                last_ts = e.sdk_timestamp

        duration_ms: int | None = None
        if first_ts and last_ts and first_ts != last_ts:
            duration_ms = int((last_ts - first_ts).total_seconds() * 1000)

        return TraceSummaryStats(
            total_events=len(events),
            events_by_level=events_by_level,
            llm_calls=llm_calls,
            tool_calls=tool_calls,
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            total_duration_ms=duration_ms,
            agent_names=agent_names,
            errors=errors,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
        )

    # --- Run management ---

    async def register_run(
        self,
        run_id: str,
        trace_id: str,
        metadata: dict[str, Any],
        *,
        parent_run_id: str | None = None,
    ) -> None:
        self._runs[run_id] = RunRecord(
            id=run_id,
            trace_id=trace_id,
            status="running",
            started_at=datetime.now(UTC),
            metadata=metadata,
            parent_run_id=parent_run_id,
        )

    async def update_run_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        error: str | None = None,
        result: RunResult | None = None,
    ) -> None:
        run = self._runs.get(run_id)
        if run is None:
            return
        completed_at = datetime.now(UTC) if status in ("completed", "failed") else run.completed_at
        self._runs[run_id] = RunRecord(
            id=run.id,
            trace_id=run.trace_id,
            status=status,
            started_at=run.started_at,
            completed_at=completed_at,
            metadata=run.metadata,
            error=error,
            result=result if result is not None else run.result,
            parent_run_id=run.parent_run_id,
        )

    async def get_run(self, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)

    async def list_runs(
        self,
        *,
        status: str | None = None,
        started_after: datetime | None = None,
        started_before: datetime | None = None,
        sort: str = "started_at_desc",
        search: str | None = None,
        parent_run_id: str | None | _UnsetType = _UNSET,
        limit: int = DEFAULT_RUNS_LIMIT,
        offset: int = 0,
    ) -> list[RunRecord]:
        runs = self._filter_runs(
            status=status,
            started_after=started_after,
            started_before=started_before,
            search=search,
            parent_run_id=parent_run_id,
        )
        runs = self._sort_runs(runs, sort)
        return runs[offset : offset + limit]

    async def count_runs(
        self,
        *,
        status: str | None = None,
        started_after: datetime | None = None,
        started_before: datetime | None = None,
        search: str | None = None,
        parent_run_id: str | None | _UnsetType = _UNSET,
    ) -> int:
        return len(
            self._filter_runs(
                status=status,
                started_after=started_after,
                started_before=started_before,
                search=search,
                parent_run_id=parent_run_id,
            )
        )

    def _filter_runs(
        self,
        *,
        status: str | None = None,
        started_after: datetime | None = None,
        started_before: datetime | None = None,
        search: str | None = None,
        parent_run_id: str | None | _UnsetType = _UNSET,
    ) -> list[RunRecord]:
        runs = list(self._runs.values())
        if status is not None:
            runs = [r for r in runs if r.status == status]
        if started_after is not None:
            runs = [r for r in runs if r.started_at >= started_after]
        if started_before is not None:
            runs = [r for r in runs if r.started_at < started_before]
        if search is not None:
            needle = search.lower()
            runs = [r for r in runs if needle in json.dumps(r.metadata).lower()]
        if not isinstance(parent_run_id, _UnsetType):
            runs = [r for r in runs if r.parent_run_id == parent_run_id]
        return runs

    @staticmethod
    def _sort_runs(runs: list[RunRecord], sort: str) -> list[RunRecord]:
        if sort == "started_at_asc":
            runs.sort(key=lambda r: r.started_at)
        elif sort == "duration_desc":
            runs.sort(
                key=lambda r: (r.completed_at - r.started_at).total_seconds() if r.completed_at else 0,
                reverse=True,
            )
        elif sort == "duration_asc":
            runs.sort(
                key=lambda r: (r.completed_at - r.started_at).total_seconds() if r.completed_at else float("inf"),
            )
        else:  # started_at_desc (default)
            runs.sort(key=lambda r: r.started_at, reverse=True)
        return runs

    async def delete_run(self, run_id: str) -> bool:
        if run_id not in self._runs:
            return False
        del self._runs[run_id]
        child_ids = [rid for rid, r in self._runs.items() if r.parent_run_id == run_id]
        for child_id in child_ids:
            await self.delete_run(child_id)
        removed_indices = set(self._parent_index.pop(run_id, []))
        if removed_indices:
            # Rebuild events list and remap indices for remaining parents
            new_events: list[StoredTraceEvent] = []
            old_to_new: dict[int, int] = {}
            for i, e in enumerate(self._events):
                if i not in removed_indices:
                    old_to_new[i] = len(new_events)
                    new_events.append(e)
            self._events = new_events
            self._parent_index = {
                pid: [old_to_new[i] for i in indices if i in old_to_new] for pid, indices in self._parent_index.items()
            }
        return True

    # --- Hierarchy queries ---

    async def get_span_tree(self, trace_id: str) -> list[StoredTraceEvent]:
        result = [e for e in self._events if e.trace_id == trace_id]
        result.sort(key=lambda e: (e.sdk_timestamp, e.id))
        return result

    async def get_events_by_span(self, trace_id: str, span_id: str) -> list[StoredTraceEvent]:
        result = [e for e in self._events if e.trace_id == trace_id and e.span_id == span_id]
        result.sort(key=lambda e: (e.sdk_timestamp, e.id))
        return result


# ---------------------------------------------------------------------------
# StoredTraceEvent → TraceEvent conversion helper
# ---------------------------------------------------------------------------


_TRACE_EVENT_ADAPTER: TypeAdapter[TraceEvent] = TypeAdapter(TraceEvent)


class MalformedStoredEventError(ValueError):
    """A stored trace event's payload failed ``TraceEvent`` validation.

    Raised by :func:`trace_events_from_stored` when a row's ``payload``
    does not round-trip through the ``TraceEvent`` discriminated union.
    Distinct from errors raised by external trace-file parsers, which
    signal a malformed trace file on disk; both are
    :class:`ValueError` subclasses so callers doing broad error handling
    catch either with a single ``except ValueError``.
    """

    def __init__(self, *, row_id: int, event_type: str, reason: str) -> None:
        self.row_id = row_id
        self.event_type = event_type
        self.reason = reason
        super().__init__(f"Malformed stored event id={row_id} event_type={event_type!r}: {reason}")


def trace_events_from_stored(
    events: Iterable[StoredTraceEvent],
) -> list[TraceEvent]:
    """Convert stored trace-event rows into validated ``TraceEvent`` values.

    This is the canonical "trace-as-data" read path for in-process
    consumers of :class:`PersistentTraceStore`. The store's
    ``get_span_tree``/``query_events`` methods return
    :class:`StoredTraceEvent` rows whose ``payload`` is a ``dict``
    (produced on the write side by ``event.model_dump(mode="json")``).
    Tools that reason about typed events — for example external
    trace-analysis tools — ingest the strongly-typed ``TraceEvent``
    discriminated union instead. This helper closes that gap without
    re-doing the conversion in every adopter's codebase.

    Behaviour:

    - Input order is preserved. The helper does not sort, deduplicate,
      or filter — the caller controls ordering by choosing which store
      method produced the input.
    - Any ``StoredTraceEvent`` whose ``payload`` fails validation raises
      :class:`MalformedStoredEventError` with the row's ``id`` and
      ``event_type``; malformed rows are never silently skipped.
    - Redaction runs at emission time, so payloads that reach this
      helper are already redacted. The helper does not re-apply
      redaction and does not warn.

    Args:
        events: Iterable of :class:`StoredTraceEvent` rows. Any iterable
            works — lists, generators, async-to-sync adapter output.

    Returns:
        A concrete (materialised) ``list[TraceEvent]`` in input order.

    Raises:
        MalformedStoredEventError: A row's payload did not validate
            against the ``TraceEvent`` discriminated union.
    """
    result: list[TraceEvent] = []
    for stored in events:
        try:
            event = _TRACE_EVENT_ADAPTER.validate_python(stored.payload)
        except ValidationError as exc:
            raise MalformedStoredEventError(
                row_id=stored.id,
                event_type=stored.event_type,
                reason=str(exc),
            ) from exc
        result.append(event)
    return result
