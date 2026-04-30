"""Response models and request schemas for the observatory API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from nanitics.infrastructure.observability.storage import RunStatus

WorkflowStepStatus = Literal["pending", "running", "completed", "error", "skipped"]
"""Runtime status of a workflow step as surfaced by the observatory."""


class TraceEventResponse(BaseModel):
    """A single trace event as returned by the API."""

    id: int
    event_type: str
    level: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    timestamp: str
    payload: dict[str, Any]


class TraceSummaryResponse(BaseModel):
    """Aggregated statistics for trace events under a parent."""

    total_events: int
    events_by_level: dict[str, int]
    llm_calls: int
    tool_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_duration_ms: int | None
    agent_names: list[str]
    errors: int
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0


# ---------------------------------------------------------------------------
# Run models
# ---------------------------------------------------------------------------


class RunResponse(BaseModel):
    """Serialized run record."""

    id: str
    trace_id: str
    status: RunStatus
    started_at: str
    completed_at: str | None
    metadata: dict[str, Any]
    error: str | None
    result: str | None


class RunListItem(BaseModel):
    """A run paired with its summary statistics."""

    run: RunResponse
    summary: TraceSummaryResponse


class RunListResponse(BaseModel):
    """Paginated list of runs with inline summaries."""

    runs: list[RunListItem]
    total: int


class RunDetailResponse(BaseModel):
    """Run record with summary statistics."""

    run: RunResponse
    summary: TraceSummaryResponse


class RunCreateRequest(BaseModel):
    """Request body for POST /runs."""

    run_id: str
    trace_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunStatusUpdateRequest(BaseModel):
    """Request body for PATCH /runs/{run_id}/status."""

    status: RunStatus
    error: str | None = None


# ---------------------------------------------------------------------------
# Span tree models
# ---------------------------------------------------------------------------


class SpanSummary(BaseModel):
    """Per-span aggregated statistics."""

    event_count: int
    duration_ms: float | None
    has_errors: bool
    agent_name: str | None
    agent_type: str | None


class SpanTreeNodeResponse(BaseModel):
    """A node in the span tree with events, children, and summary."""

    span_id: str
    parent_span_id: str | None
    name: str
    summary: SpanSummary
    events: list[TraceEventResponse]
    children: list[SpanTreeNodeResponse]


class SpanTreeResponse(BaseModel):
    """Wrapper for the full span tree."""

    trace_id: str
    root: SpanTreeNodeResponse


class SpanEventsResponse(BaseModel):
    """Events within a specific span."""

    span_id: str
    events: list[TraceEventResponse]


# ---------------------------------------------------------------------------
# Agent models
# ---------------------------------------------------------------------------


class AgentStatsResponse(BaseModel):
    """Per-agent computed statistics."""

    llm_calls: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    duration_ms: float | None
    errors: int
    iterations: int


class AgentInfoResponse(BaseModel):
    """Agent metadata and stats."""

    agent_name: str
    agent_type: str | None
    span_id: str
    capabilities: list[str]
    stats: AgentStatsResponse


class AgentListResponse(BaseModel):
    """List of agents in a run."""

    agents: list[AgentInfoResponse]


class AgentDetailResponse(BaseModel):
    """Detailed agent view with events and span subtree."""

    agent: AgentInfoResponse
    events: list[TraceEventResponse]
    span_tree: SpanTreeNodeResponse


# ---------------------------------------------------------------------------
# Workflow models
# ---------------------------------------------------------------------------


class WorkflowStepResponse(BaseModel):
    """A workflow step with definition and runtime status."""

    name: str
    step_type: str
    index: int | None
    depends_on: list[str]
    parallel_group: str | None
    status: WorkflowStepStatus
    duration_ms: float | None
    agent_span_id: str | None
    metadata: dict[str, Any]


class WorkflowDAGResponse(BaseModel):
    """Complete workflow DAG structure."""

    workflow_name: str
    workflow_type: str
    steps: list[WorkflowStepResponse]


# ---------------------------------------------------------------------------
# Event list model
# ---------------------------------------------------------------------------


class EventListResponse(BaseModel):
    """Paginated event list."""

    events: list[TraceEventResponse]
    has_more: bool
