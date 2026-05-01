"""Observatory service — business logic composing PersistentTraceStore primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from nanitics.infrastructure.observability.levels import is_level_included
from nanitics.infrastructure.observability.storage import DEFAULT_RUNS_LIMIT, RunStatus
from nanitics.observatory.models import (
    AgentDetailResponse,
    AgentInfoResponse,
    AgentListResponse,
    AgentStatsResponse,
    EventListResponse,
    RunDetailResponse,
    RunListItem,
    RunListResponse,
    RunResponse,
    SpanEventsResponse,
    SpanSummary,
    SpanTreeNodeResponse,
    SpanTreeResponse,
    TraceEventResponse,
    TraceSummaryResponse,
    WorkflowDAGResponse,
    WorkflowStepResponse,
    WorkflowStepStatus,
)

if TYPE_CHECKING:
    from datetime import datetime

    from nanitics.infrastructure.observability.levels import TraceLevel
    from nanitics.infrastructure.observability.storage import (
        PersistentTraceStore,
        RunRecord,
        StoredTraceEvent,
        TraceSummaryStats,
    )

SortOption = Literal["started_at_desc", "started_at_asc", "duration_desc", "duration_asc"]
"""Valid sort options for run listing."""


def _event_to_response(e: StoredTraceEvent) -> TraceEventResponse:
    return TraceEventResponse(
        id=e.id,
        event_type=e.event_type,
        level=e.level,
        trace_id=e.trace_id,
        span_id=e.span_id,
        parent_span_id=e.parent_span_id,
        timestamp=e.sdk_timestamp.isoformat(),
        payload=e.payload,
    )


def _run_to_response(run: RunRecord) -> RunResponse:
    return RunResponse(
        id=run.id,
        trace_id=run.trace_id,
        status=run.status,
        started_at=run.started_at.isoformat(),
        completed_at=run.completed_at.isoformat() if run.completed_at else None,
        metadata=dict(run.metadata),
        error=run.error,
        result=run.result,
    )


def _stats_to_response(stats: TraceSummaryStats) -> TraceSummaryResponse:
    return TraceSummaryResponse(
        total_events=stats.total_events,
        events_by_level={str(lvl): count for lvl, count in stats.events_by_level.items()},
        llm_calls=stats.llm_calls,
        tool_calls=stats.tool_calls,
        total_input_tokens=stats.total_input_tokens,
        total_output_tokens=stats.total_output_tokens,
        total_duration_ms=stats.total_duration_ms,
        agent_names=list(stats.agent_names),
        errors=stats.errors,
        cache_creation_tokens=stats.cache_creation_tokens,
        cache_read_tokens=stats.cache_read_tokens,
    )


# ---------------------------------------------------------------------------
# Internal tree-building helpers
# ---------------------------------------------------------------------------


@dataclass
class _SpanNode:
    """Mutable node used during tree construction."""

    span_id: str
    parent_span_id: str | None = None
    name: str = ""
    events: list[StoredTraceEvent] = field(default_factory=list)
    children: list[_SpanNode] = field(default_factory=list)
    agent_name: str | None = None
    agent_type: str | None = None
    duration_ms: float | None = None
    has_errors: bool = False


def _build_span_tree(events: list[StoredTraceEvent]) -> _SpanNode:
    """Build a nested span tree from a flat list of events.

    Returns a synthetic root node whose children are the top-level spans.
    """
    nodes: dict[str, _SpanNode] = {}

    for e in events:
        node = nodes.setdefault(
            e.span_id,
            _SpanNode(span_id=e.span_id, parent_span_id=e.parent_span_id),
        )
        # First event with parent_span_id wins (update if not set yet)
        if node.parent_span_id is None and e.parent_span_id is not None:
            node.parent_span_id = e.parent_span_id
        node.events.append(e)

        if e.event_type == "span.start":
            node.name = e.payload.get("name", "")
        elif e.event_type == "span.end":
            node.duration_ms = e.payload.get("duration_ms")
        elif e.event_type == "agent.start":
            node.agent_name = e.payload.get("agent_name")
            node.agent_type = e.payload.get("agent_type")
            if not node.name:
                node.name = e.payload.get("agent_name", "")
        elif e.event_type.startswith("agent.error") or e.event_type.startswith("workflow.error"):
            node.has_errors = True

    # Wire parent-child relationships
    root = _SpanNode(span_id="__root__", name="root")
    for node in nodes.values():
        parent = nodes.get(node.parent_span_id) if node.parent_span_id else None
        if parent is not None:
            parent.children.append(node)
        else:
            root.children.append(node)

    return root


def _span_node_to_response(
    node: _SpanNode,
    min_level: TraceLevel | None = None,
) -> SpanTreeNodeResponse:
    """Convert a mutable _SpanNode to a response model, optionally filtering events."""
    events = node.events
    if min_level is not None:
        events = [e for e in events if is_level_included(e.level, min_level)]

    return SpanTreeNodeResponse(
        span_id=node.span_id,
        parent_span_id=node.parent_span_id,
        name=node.name,
        summary=SpanSummary(
            event_count=len(events),
            duration_ms=node.duration_ms,
            has_errors=node.has_errors,
            agent_name=node.agent_name,
            agent_type=node.agent_type,
        ),
        events=[_event_to_response(e) for e in events],
        children=[_span_node_to_response(c, min_level) for c in node.children],
    )


def _compute_agent_stats(events: list[StoredTraceEvent]) -> AgentStatsResponse:
    """Compute stats from a list of events in an agent's span subtree."""
    llm_calls = 0
    tool_calls = 0
    input_tokens = 0
    output_tokens = 0
    errors = 0
    iterations = 0
    first_ts = None
    last_ts = None

    for e in events:
        if e.event_type == "llm.response":
            llm_calls += 1
            usage = e.payload.get("usage", e.payload)
            input_tokens += int(usage.get("input_tokens", 0))
            output_tokens += int(usage.get("output_tokens", 0))
        elif e.event_type == "tool.invoke":
            tool_calls += 1
        elif e.event_type == "agent.step":
            iterations += 1
        elif e.event_type.startswith("agent.error") or e.event_type.startswith("workflow.error"):
            errors += 1

        if first_ts is None or e.sdk_timestamp < first_ts:
            first_ts = e.sdk_timestamp
        if last_ts is None or e.sdk_timestamp > last_ts:
            last_ts = e.sdk_timestamp

    duration_ms: float | None = None
    if first_ts and last_ts and first_ts != last_ts:
        duration_ms = (last_ts - first_ts).total_seconds() * 1000

    return AgentStatsResponse(
        llm_calls=llm_calls,
        tool_calls=tool_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        duration_ms=duration_ms,
        errors=errors,
        iterations=iterations,
    )


def _collect_subtree_events(node: _SpanNode) -> list[StoredTraceEvent]:
    """Recursively collect all events from a node and its descendants."""
    result = list(node.events)
    for child in node.children:
        result.extend(_collect_subtree_events(child))
    return result


def _find_node(root: _SpanNode, span_id: str) -> _SpanNode | None:
    """Find a node in the tree by span_id (depth-first)."""
    if root.span_id == span_id:
        return root
    for child in root.children:
        found = _find_node(child, span_id)
        if found is not None:
            return found
    return None


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ObservatoryService:
    """Business logic layer composing PersistentTraceStore primitives.

    All methods are async. Instantiate with an ``InMemoryPersistentTraceStore``
    for testing.
    """

    def __init__(self, store: PersistentTraceStore) -> None:
        self._store = store

    # --- Run management ---

    async def list_runs(
        self,
        *,
        status: RunStatus | None = None,
        started_after: datetime | None = None,
        started_before: datetime | None = None,
        sort: SortOption = "started_at_desc",
        search: str | None = None,
        limit: int = DEFAULT_RUNS_LIMIT,
        offset: int = 0,
    ) -> RunListResponse:
        import asyncio

        runs, total = await asyncio.gather(
            self._store.list_runs(
                status=status,
                started_after=started_after,
                started_before=started_before,
                search=search,
                sort=sort,
                limit=limit,
                offset=offset,
            ),
            self._store.count_runs(
                status=status,
                started_after=started_after,
                started_before=started_before,
                search=search,
            ),
        )

        # Fetch summaries for each run in parallel
        summaries = await asyncio.gather(*(self._store.get_summary(r.id) for r in runs))

        return RunListResponse(
            runs=[
                RunListItem(
                    run=_run_to_response(r),
                    summary=_stats_to_response(s),
                )
                for r, s in zip(runs, summaries, strict=True)
            ],
            total=total,
        )

    async def get_run(self, run_id: str) -> RunDetailResponse | None:
        run = await self._store.get_run(run_id)
        if run is None:
            return None
        stats = await self._store.get_summary(run_id)
        return RunDetailResponse(
            run=_run_to_response(run),
            summary=_stats_to_response(stats),
        )

    async def register_run(
        self,
        run_id: str,
        trace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> RunResponse:
        await self._store.register_run(run_id, trace_id, metadata or {})
        run = await self._store.get_run(run_id)
        assert run is not None  # just registered
        return _run_to_response(run)

    async def update_run_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        error: str | None = None,
    ) -> None:
        await self._store.update_run_status(run_id, status, error=error)

    async def delete_run(self, run_id: str) -> bool:
        return await self._store.delete_run(run_id)

    # --- Trace hierarchy ---

    async def get_span_tree(
        self,
        trace_id: str,
        *,
        min_level: TraceLevel | None = None,
    ) -> SpanTreeResponse:
        events = await self._store.get_span_tree(trace_id)
        root = _build_span_tree(events)
        root_response = _span_node_to_response(root, min_level)
        return SpanTreeResponse(trace_id=trace_id, root=root_response)

    async def get_events_for_span(
        self,
        trace_id: str,
        span_id: str,
        *,
        levels: list[TraceLevel] | None = None,
        event_types: list[str] | None = None,
    ) -> SpanEventsResponse:
        events = await self._store.get_events_by_span(trace_id, span_id)
        if levels:
            events = [e for e in events if e.level in levels]
        if event_types:
            events = [e for e in events if e.event_type in event_types]
        return SpanEventsResponse(
            span_id=span_id,
            events=[_event_to_response(e) for e in events],
        )

    # --- Agent-scoped queries ---

    async def list_agents(self, trace_id: str) -> AgentListResponse:
        all_events = await self._store.get_span_tree(trace_id)
        root = _build_span_tree(all_events)

        agents: list[AgentInfoResponse] = []
        self._collect_agents(root, agents)
        return AgentListResponse(agents=agents)

    def _collect_agents(
        self,
        node: _SpanNode,
        agents: list[AgentInfoResponse],
    ) -> None:
        if node.agent_name is not None:
            subtree_events = _collect_subtree_events(node)
            stats = _compute_agent_stats(subtree_events)
            # Extract capabilities from the AgentStartEvent payload
            capabilities: list[str] = []
            for e in node.events:
                if e.event_type == "agent.start":
                    capabilities = e.payload.get("capabilities", [])
                    break
            agents.append(
                AgentInfoResponse(
                    agent_name=node.agent_name,
                    agent_type=node.agent_type,
                    span_id=node.span_id,
                    capabilities=capabilities,
                    stats=stats,
                )
            )
        for child in node.children:
            self._collect_agents(child, agents)

    async def get_agent_detail(self, trace_id: str, span_id: str) -> AgentDetailResponse | None:
        all_events = await self._store.get_span_tree(trace_id)
        root = _build_span_tree(all_events)
        node = _find_node(root, span_id)
        if node is None or node.agent_name is None:
            return None

        subtree_events = _collect_subtree_events(node)
        stats = _compute_agent_stats(subtree_events)
        capabilities: list[str] = []
        for e in node.events:
            if e.event_type == "agent.start":
                capabilities = e.payload.get("capabilities", [])
                break

        agent_info = AgentInfoResponse(
            agent_name=node.agent_name,
            agent_type=node.agent_type,
            span_id=node.span_id,
            capabilities=capabilities,
            stats=stats,
        )

        return AgentDetailResponse(
            agent=agent_info,
            events=[_event_to_response(e) for e in subtree_events],
            span_tree=_span_node_to_response(node),
        )

    # --- Workflow structure ---

    async def get_workflow_structure(self, trace_id: str) -> WorkflowDAGResponse | None:
        all_events = await self._store.get_span_tree(trace_id)

        # Find the WorkflowStructureEvent
        structure_event = None
        for e in all_events:
            if e.event_type == "workflow.structure":
                structure_event = e
                break

        if structure_event is None:
            return None

        # Build step completion map: step_name -> completion info
        completions: dict[str, StoredTraceEvent] = {}
        for e in all_events:
            if e.event_type == "workflow.step.complete":
                completions[e.payload.get("step_name", "")] = e

        # Build agent start map: step_name -> True if agent.start matches
        agent_started: set[str] = set()
        for e in all_events:
            if e.event_type == "agent.start":
                agent_started.add(e.payload.get("agent_name", ""))

        # Detect workflow error and the failed step
        failed_step_name: str | None = None
        for e in all_events:
            if e.event_type == "workflow.error":
                failed_step_name = e.payload.get("failed_step")
                break

        steps_data = structure_event.payload.get("steps", [])

        # Build dependency graph for transitive "skipped" computation
        dependents: dict[str, list[str]] = {}
        for step in steps_data:
            for dep in step.get("depends_on", []):
                dependents.setdefault(dep, []).append(step.get("name", ""))

        # Compute transitively skipped steps (downstream of failed step)
        skipped_steps: set[str] = set()
        if failed_step_name is not None:
            queue = list(dependents.get(failed_step_name, []))
            while queue:
                dep_name = queue.pop()
                if dep_name not in skipped_steps:
                    skipped_steps.add(dep_name)
                    queue.extend(dependents.get(dep_name, []))

        steps: list[WorkflowStepResponse] = []
        for step in steps_data:
            step_name = step.get("name", "")
            completion = completions.get(step_name)

            # Priority: completed > error > skipped > running > pending
            status: WorkflowStepStatus
            if completion:
                status = "completed"
            elif step_name == failed_step_name:
                status = "error"
            elif step_name in skipped_steps:
                status = "skipped"
            elif step_name in agent_started:
                status = "running"
            else:
                status = "pending"

            duration_ms = completion.payload.get("step_duration_ms") if completion else None
            # Try to find the agent span_id for agent-type steps
            agent_span_id: str | None = None
            if step.get("step_type") == "agent":
                for e in all_events:
                    if e.event_type == "agent.start" and e.payload.get("agent_name") == step_name:
                        agent_span_id = e.span_id
                        break

            steps.append(
                WorkflowStepResponse(
                    name=step_name,
                    step_type=step.get("step_type", "custom"),
                    index=step.get("index"),
                    depends_on=step.get("depends_on", []),
                    parallel_group=step.get("parallel_group"),
                    status=status,
                    duration_ms=duration_ms,
                    agent_span_id=agent_span_id,
                    metadata=step.get("metadata", {}),
                )
            )

        return WorkflowDAGResponse(
            workflow_name=structure_event.payload.get("workflow_name", ""),
            workflow_type=structure_event.payload.get("workflow_type", ""),
            steps=steps,
        )

    # --- Event detail ---

    async def get_event(self, event_id: int) -> TraceEventResponse | None:
        event = await self._store.get_event(event_id)
        if event is None:
            return None
        return _event_to_response(event)

    # --- Event listing (flat) ---

    async def query_events(
        self,
        parent_id: str,
        *,
        levels: list[TraceLevel] | None = None,
        event_types: list[str] | None = None,
        after_id: int | None = None,
        limit: int = 100,
    ) -> EventListResponse:
        # Fetch one extra to detect has_more
        events = await self._store.query_events(
            parent_id,
            levels=levels,
            event_types=event_types,
            after_id=after_id,
            limit=limit + 1,
        )
        has_more = len(events) > limit
        events = events[:limit]
        return EventListResponse(
            events=[_event_to_response(e) for e in events],
            has_more=has_more,
        )

    # --- Statistics ---

    async def get_run_summary(self, parent_id: str) -> TraceSummaryResponse:
        stats = await self._store.get_summary(parent_id)
        return _stats_to_response(stats)

    async def get_agent_stats(self, trace_id: str, span_id: str) -> AgentStatsResponse | None:
        all_events = await self._store.get_span_tree(trace_id)
        root = _build_span_tree(all_events)
        node = _find_node(root, span_id)
        if node is None:
            return None
        subtree_events = _collect_subtree_events(node)
        return _compute_agent_stats(subtree_events)
