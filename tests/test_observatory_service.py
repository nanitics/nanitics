"""Unit tests for ObservatoryService."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from nanitics.infrastructure.observability.levels import TraceLevel
from nanitics.infrastructure.observability.storage import (
    InMemoryPersistentTraceStore,
    TraceEventRecord,
)
from nanitics.observatory.models import SpanTreeNodeResponse, TraceEventResponse
from nanitics.observatory.service import ObservatoryService

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


async def _seed_multi_agent_trace(
    store: InMemoryPersistentTraceStore,
    run_id: str = "run-1",
    trace_id: str = "trace-1",
) -> None:
    """Seed a realistic multi-agent trace: parent agent -> 2 child agents with tools and LLM calls."""
    await store.register_run(run_id, trace_id, {"type": "test"})

    events = [
        # Root span
        _make_event(
            "span.start",
            trace_id=trace_id,
            span_id="span-root",
            payload={"name": "root"},
            level="verbose",
            time_offset_ms=0,
        ),
        # Parent agent
        _make_event(
            "agent.start",
            trace_id=trace_id,
            span_id="span-parent",
            parent_span_id="span-root",
            payload={
                "agent_name": "orchestrator",
                "agent_type": "supervisor",
                "task_input": "do the thing",
                "tools_available": ["delegate"],
                "capabilities": ["planning", "delegation"],
            },
            time_offset_ms=10,
        ),
        _make_event(
            "agent.step",
            trace_id=trace_id,
            span_id="span-parent",
            parent_span_id="span-root",
            payload={"agent_name": "orchestrator", "step_number": 1},
            level="debug",
            time_offset_ms=20,
        ),
        _make_event(
            "llm.response",
            trace_id=trace_id,
            span_id="span-parent",
            parent_span_id="span-root",
            payload={"usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}},
            level="debug",
            time_offset_ms=30,
        ),
        # Child agent 1
        _make_event(
            "agent.start",
            trace_id=trace_id,
            span_id="span-child-1",
            parent_span_id="span-parent",
            payload={
                "agent_name": "researcher",
                "agent_type": "worker",
                "task_input": "research task",
                "tools_available": ["search"],
                "capabilities": ["web_search"],
            },
            time_offset_ms=40,
        ),
        _make_event(
            "agent.step",
            trace_id=trace_id,
            span_id="span-child-1",
            parent_span_id="span-parent",
            payload={"agent_name": "researcher", "step_number": 1},
            level="debug",
            time_offset_ms=50,
        ),
        _make_event(
            "llm.response",
            trace_id=trace_id,
            span_id="span-child-1",
            parent_span_id="span-parent",
            payload={"usage": {"input_tokens": 200, "output_tokens": 80, "total_tokens": 280}},
            level="debug",
            time_offset_ms=60,
        ),
        _make_event(
            "tool.invoke",
            trace_id=trace_id,
            span_id="span-child-1",
            parent_span_id="span-parent",
            payload={"tool_name": "search", "input": "query"},
            level="debug",
            time_offset_ms=70,
        ),
        _make_event(
            "agent.complete",
            trace_id=trace_id,
            span_id="span-child-1",
            parent_span_id="span-parent",
            payload={"agent_name": "researcher", "total_steps": 1, "termination_reason": "done"},
            time_offset_ms=80,
        ),
        # Child agent 2
        _make_event(
            "agent.start",
            trace_id=trace_id,
            span_id="span-child-2",
            parent_span_id="span-parent",
            payload={
                "agent_name": "writer",
                "agent_type": "worker",
                "task_input": "write task",
                "tools_available": [],
                "capabilities": [],
            },
            time_offset_ms=90,
        ),
        _make_event(
            "agent.error",
            trace_id=trace_id,
            span_id="span-child-2",
            parent_span_id="span-parent",
            payload={
                "agent_name": "writer",
                "error_type": "RuntimeError",
                "error_message": "oops",
                "error_metadata": {},
            },
            time_offset_ms=100,
        ),
        # Parent agent completes
        _make_event(
            "agent.complete",
            trace_id=trace_id,
            span_id="span-parent",
            parent_span_id="span-root",
            payload={"agent_name": "orchestrator", "total_steps": 1, "termination_reason": "done"},
            time_offset_ms=110,
        ),
        _make_event(
            "span.end",
            trace_id=trace_id,
            span_id="span-root",
            payload={"name": "root", "duration_ms": 110.0},
            level="verbose",
            time_offset_ms=110,
        ),
    ]
    await store.save_events_batch(run_id, events)


# ---------------------------------------------------------------------------
# Tests: Run management
# ---------------------------------------------------------------------------


class TestRunManagement:
    @pytest.fixture
    def store(self) -> InMemoryPersistentTraceStore:
        return InMemoryPersistentTraceStore()

    @pytest.fixture
    def service(self, store: InMemoryPersistentTraceStore) -> ObservatoryService:
        return ObservatoryService(store)

    async def test_register_and_get_run(self, service: ObservatoryService) -> None:
        run = await service.register_run("run-1", "trace-1", {"key": "val"})
        assert run.id == "run-1"
        assert run.trace_id == "trace-1"
        assert run.status == "running"
        assert run.metadata == {"key": "val"}

        detail = await service.get_run("run-1")
        assert detail is not None
        assert detail.run.id == "run-1"

    async def test_get_run_not_found(self, service: ObservatoryService) -> None:
        result = await service.get_run("nonexistent")
        assert result is None

    async def test_list_runs(
        self,
        service: ObservatoryService,
    ) -> None:
        await service.register_run("run-1", "trace-1")
        await service.register_run("run-2", "trace-2")

        result = await service.list_runs()
        assert result.total == 2
        assert len(result.runs) == 2
        # Each item has run + summary
        assert result.runs[0].run is not None
        assert result.runs[0].summary is not None

    async def test_list_runs_with_status_filter(self, service: ObservatoryService) -> None:
        await service.register_run("run-1", "trace-1")
        await service.register_run("run-2", "trace-2")
        await service.update_run_status("run-1", "completed")

        result = await service.list_runs(status="completed")
        assert result.total == 1
        assert result.runs[0].run.id == "run-1"
        assert result.runs[0].run.status == "completed"

    async def test_list_runs_pagination(self, service: ObservatoryService) -> None:
        for i in range(5):
            await service.register_run(f"run-{i}", f"trace-{i}")

        page = await service.list_runs(limit=2, offset=0)
        assert len(page.runs) == 2
        assert page.total == 5

    async def test_list_runs_inline_summaries(
        self,
        store: InMemoryPersistentTraceStore,
        service: ObservatoryService,
    ) -> None:
        await service.register_run("run-1", "trace-1")
        # Seed some events so summary is non-trivial
        await store.save_events_batch(
            "run-1",
            [
                TraceEventRecord(
                    event_type="llm.response",
                    level="debug",
                    trace_id="trace-1",
                    span_id="s1",
                    parent_span_id=None,
                    payload={"usage": {"input_tokens": 100, "output_tokens": 50}},
                    sdk_timestamp=datetime.now(tz=UTC),
                ),
            ],
        )

        result = await service.list_runs()
        assert len(result.runs) == 1
        item = result.runs[0]
        assert item.run.id == "run-1"
        assert item.summary.llm_calls == 1
        assert item.summary.total_input_tokens == 100
        assert item.summary.total_output_tokens == 50

    async def test_update_run_status(self, service: ObservatoryService) -> None:
        await service.register_run("run-1", "trace-1")
        await service.update_run_status("run-1", "failed", error="boom")

        detail = await service.get_run("run-1")
        assert detail is not None
        assert detail.run.status == "failed"
        assert detail.run.error == "boom"

    async def test_delete_run(self, service: ObservatoryService) -> None:
        await service.register_run("run-1", "trace-1")
        result = await service.delete_run("run-1")
        assert result is True
        assert await service.get_run("run-1") is None

    async def test_delete_run_not_found(self, service: ObservatoryService) -> None:
        result = await service.delete_run("nonexistent")
        assert result is False


# ---------------------------------------------------------------------------
# Tests: Span tree reconstruction
# ---------------------------------------------------------------------------


class TestSpanTree:
    @pytest.fixture
    def store(self) -> InMemoryPersistentTraceStore:
        return InMemoryPersistentTraceStore()

    @pytest.fixture
    def service(self, store: InMemoryPersistentTraceStore) -> ObservatoryService:
        return ObservatoryService(store)

    async def test_tree_reconstruction_multi_agent(
        self,
        store: InMemoryPersistentTraceStore,
        service: ObservatoryService,
    ) -> None:
        await _seed_multi_agent_trace(store)

        tree = await service.get_span_tree("trace-1")
        assert tree.trace_id == "trace-1"

        # Root is the synthetic root; its children are the top-level spans
        root = tree.root
        assert root.span_id == "__root__"
        assert len(root.children) == 1  # span-root

        span_root = root.children[0]
        assert span_root.span_id == "span-root"
        assert span_root.summary.duration_ms == 110.0

        # span-root has one child: span-parent
        assert len(span_root.children) == 1
        parent = span_root.children[0]
        assert parent.span_id == "span-parent"
        assert parent.summary.agent_name == "orchestrator"
        assert parent.summary.agent_type == "supervisor"

        # span-parent has two children: span-child-1, span-child-2
        assert len(parent.children) == 2
        child_ids = {c.span_id for c in parent.children}
        assert child_ids == {"span-child-1", "span-child-2"}

        # child-2 has errors
        child_2 = next(c for c in parent.children if c.span_id == "span-child-2")
        assert child_2.summary.has_errors is True

    async def test_tree_with_min_level_filter(
        self,
        store: InMemoryPersistentTraceStore,
        service: ObservatoryService,
    ) -> None:
        await _seed_multi_agent_trace(store)

        tree = await service.get_span_tree("trace-1", min_level="info")
        # Gather all events across the tree
        all_events = _collect_tree_events(tree.root)
        assert len(all_events) > 0, "Expected some info-level events"
        # debug and verbose events should be excluded
        for e in all_events:
            assert e.level not in ("debug", "verbose"), f"Unexpected level {e.level}"

    async def test_empty_trace(self, service: ObservatoryService) -> None:
        tree = await service.get_span_tree("nonexistent-trace")
        assert tree.root.span_id == "__root__"
        assert tree.root.children == []


# ---------------------------------------------------------------------------
# Tests: Agent discovery
# ---------------------------------------------------------------------------


class TestAgentDiscovery:
    @pytest.fixture
    def store(self) -> InMemoryPersistentTraceStore:
        return InMemoryPersistentTraceStore()

    @pytest.fixture
    def service(self, store: InMemoryPersistentTraceStore) -> ObservatoryService:
        return ObservatoryService(store)

    async def test_list_agents(
        self,
        store: InMemoryPersistentTraceStore,
        service: ObservatoryService,
    ) -> None:
        await _seed_multi_agent_trace(store)

        agents = await service.list_agents("trace-1")
        assert len(agents.agents) == 3

        by_name = {a.agent_name: a for a in agents.agents}
        assert "orchestrator" in by_name
        assert "researcher" in by_name
        assert "writer" in by_name

        orch = by_name["orchestrator"]
        assert orch.agent_type == "supervisor"
        assert orch.capabilities == ["planning", "delegation"]

        researcher = by_name["researcher"]
        assert researcher.agent_type == "worker"
        assert researcher.stats.tool_calls == 1
        assert researcher.stats.llm_calls == 1
        assert researcher.stats.input_tokens == 200
        assert researcher.stats.output_tokens == 80
        assert researcher.stats.iterations == 1

    async def test_get_agent_detail(
        self,
        store: InMemoryPersistentTraceStore,
        service: ObservatoryService,
    ) -> None:
        await _seed_multi_agent_trace(store)

        detail = await service.get_agent_detail("trace-1", "span-child-1")
        assert detail is not None
        assert detail.agent.agent_name == "researcher"
        assert detail.agent.stats.tool_calls == 1
        assert len(detail.events) > 0
        assert detail.span_tree.span_id == "span-child-1"

    async def test_get_agent_detail_not_found(self, service: ObservatoryService) -> None:
        result = await service.get_agent_detail("trace-1", "nonexistent")
        assert result is None

    async def test_agent_stats_computation(
        self,
        store: InMemoryPersistentTraceStore,
        service: ObservatoryService,
    ) -> None:
        await _seed_multi_agent_trace(store)

        # Orchestrator stats include its own events but child subtrees are
        # separate spans under it, so they get counted too
        stats = await service.get_agent_stats("trace-1", "span-parent")
        assert stats is not None
        # orchestrator has 1 llm.response of its own + 1 from child researcher
        assert stats.llm_calls == 2
        # 1 tool call from researcher
        assert stats.tool_calls == 1
        # Tokens: 100+200=300 input, 50+80=130 output
        assert stats.input_tokens == 300
        assert stats.output_tokens == 130
        # Errors: 1 from writer child
        assert stats.errors == 1
        # Iterations: 1 from self + 1 from researcher
        assert stats.iterations == 2

    async def test_agent_with_errors(
        self,
        store: InMemoryPersistentTraceStore,
        service: ObservatoryService,
    ) -> None:
        await _seed_multi_agent_trace(store)

        detail = await service.get_agent_detail("trace-1", "span-child-2")
        assert detail is not None
        assert detail.agent.agent_name == "writer"
        assert detail.agent.stats.errors == 1


# ---------------------------------------------------------------------------
# Tests: Workflow DAG
# ---------------------------------------------------------------------------


class TestWorkflowDag:
    @pytest.fixture
    def store(self) -> InMemoryPersistentTraceStore:
        return InMemoryPersistentTraceStore()

    @pytest.fixture
    def service(self, store: InMemoryPersistentTraceStore) -> ObservatoryService:
        return ObservatoryService(store)

    async def test_workflow_structure_extraction(
        self,
        store: InMemoryPersistentTraceStore,
        service: ObservatoryService,
    ) -> None:
        await store.register_run("run-1", "trace-1", {})
        events = [
            _make_event(
                "workflow.structure",
                span_id="span-wf",
                payload={
                    "workflow_name": "my-workflow",
                    "workflow_type": "sequential",
                    "steps": [
                        {
                            "name": "step-a",
                            "step_type": "agent",
                            "index": 0,
                            "depends_on": [],
                            "metadata": {},
                        },
                        {
                            "name": "step-b",
                            "step_type": "function",
                            "index": 1,
                            "depends_on": ["step-a"],
                            "metadata": {},
                        },
                    ],
                },
            ),
            _make_event(
                "agent.start",
                span_id="span-agent-a",
                parent_span_id="span-wf",
                payload={
                    "agent_name": "step-a",
                    "task_input": "do stuff",
                    "tools_available": [],
                },
                time_offset_ms=10,
            ),
            _make_event(
                "workflow.step.complete",
                span_id="span-wf",
                payload={
                    "workflow_name": "my-workflow",
                    "step_name": "step-a",
                    "step_index": 0,
                    "step_duration_ms": 500,
                },
                time_offset_ms=20,
            ),
        ]
        await store.save_events_batch("run-1", events)

        dag = await service.get_workflow_structure("trace-1")
        assert dag is not None
        assert dag.workflow_name == "my-workflow"
        assert dag.workflow_type == "sequential"
        assert len(dag.steps) == 2

        step_a = dag.steps[0]
        assert step_a.name == "step-a"
        assert step_a.status == "completed"
        assert step_a.duration_ms == 500
        assert step_a.agent_span_id == "span-agent-a"

        step_b = dag.steps[1]
        assert step_b.name == "step-b"
        assert step_b.status == "pending"
        assert step_b.depends_on == ["step-a"]

    async def test_no_workflow_structure(
        self,
        store: InMemoryPersistentTraceStore,
        service: ObservatoryService,
    ) -> None:
        await store.register_run("run-1", "trace-1", {})
        events = [
            _make_event(
                "agent.start",
                payload={
                    "agent_name": "solo",
                    "task_input": "hi",
                    "tools_available": [],
                },
            ),
        ]
        await store.save_events_batch("run-1", events)

        dag = await service.get_workflow_structure("trace-1")
        assert dag is None

    async def test_parallel_workflow(
        self,
        store: InMemoryPersistentTraceStore,
        service: ObservatoryService,
    ) -> None:
        await store.register_run("run-1", "trace-1", {})
        events = [
            _make_event(
                "workflow.structure",
                span_id="span-wf",
                payload={
                    "workflow_name": "parallel-wf",
                    "workflow_type": "parallel",
                    "steps": [
                        {
                            "name": "task-1",
                            "step_type": "agent",
                            "index": 0,
                            "depends_on": [],
                            "parallel_group": "group-a",
                            "metadata": {},
                        },
                        {
                            "name": "task-2",
                            "step_type": "agent",
                            "index": 1,
                            "depends_on": [],
                            "parallel_group": "group-a",
                            "metadata": {},
                        },
                    ],
                },
            ),
            _make_event(
                "workflow.step.complete",
                span_id="span-wf",
                payload={
                    "workflow_name": "parallel-wf",
                    "step_name": "task-1",
                    "step_index": 0,
                    "step_duration_ms": 300,
                },
                time_offset_ms=10,
            ),
            _make_event(
                "workflow.step.complete",
                span_id="span-wf",
                payload={
                    "workflow_name": "parallel-wf",
                    "step_name": "task-2",
                    "step_index": 1,
                    "step_duration_ms": 450,
                },
                time_offset_ms=20,
            ),
        ]
        await store.save_events_batch("run-1", events)

        dag = await service.get_workflow_structure("trace-1")
        assert dag is not None
        assert len(dag.steps) == 2
        assert all(s.status == "completed" for s in dag.steps)
        assert all(s.parallel_group == "group-a" for s in dag.steps)

    async def test_running_status_from_agent_start(
        self,
        store: InMemoryPersistentTraceStore,
        service: ObservatoryService,
    ) -> None:
        """Steps with agent.start but no completion get status='running'."""
        await store.register_run("run-1", "trace-1", {})
        events = [
            _make_event(
                "workflow.structure",
                span_id="span-wf",
                payload={
                    "workflow_name": "wf",
                    "workflow_type": "sequential",
                    "steps": [
                        {"name": "step-a", "step_type": "agent", "index": 0, "depends_on": [], "metadata": {}},
                        {"name": "step-b", "step_type": "agent", "index": 1, "depends_on": ["step-a"], "metadata": {}},
                    ],
                },
            ),
            _make_event(
                "agent.start",
                span_id="span-agent-a",
                parent_span_id="span-wf",
                payload={"agent_name": "step-a", "task_input": "go", "tools_available": []},
                time_offset_ms=10,
            ),
        ]
        await store.save_events_batch("run-1", events)

        dag = await service.get_workflow_structure("trace-1")
        assert dag is not None
        step_map = {s.name: s for s in dag.steps}
        assert step_map["step-a"].status == "running"
        assert step_map["step-b"].status == "pending"

    async def test_error_status_from_workflow_error(
        self,
        store: InMemoryPersistentTraceStore,
        service: ObservatoryService,
    ) -> None:
        """Failed step gets status='error', downstream steps get 'skipped'."""
        await store.register_run("run-1", "trace-1", {})
        events = [
            _make_event(
                "workflow.structure",
                span_id="span-wf",
                payload={
                    "workflow_name": "wf",
                    "workflow_type": "dag",
                    "steps": [
                        {"name": "fetch", "step_type": "function", "index": 0, "depends_on": [], "metadata": {}},
                        {"name": "parse", "step_type": "function", "index": 1, "depends_on": ["fetch"], "metadata": {}},
                        {
                            "name": "validate",
                            "step_type": "function",
                            "index": 2,
                            "depends_on": ["fetch"],
                            "metadata": {},
                        },
                        {
                            "name": "combine",
                            "step_type": "function",
                            "index": 3,
                            "depends_on": ["parse", "validate"],
                            "metadata": {},
                        },
                    ],
                },
            ),
            _make_event(
                "workflow.step.complete",
                span_id="span-wf",
                payload={"workflow_name": "wf", "step_name": "fetch", "step_index": 0, "step_duration_ms": 100},
                time_offset_ms=10,
            ),
            _make_event(
                "workflow.error",
                span_id="span-wf",
                payload={
                    "workflow_name": "wf",
                    "workflow_type": "dag",
                    "error_type": "RuntimeError",
                    "error_message": "parse failed",
                    "failed_step": "parse",
                },
                time_offset_ms=20,
            ),
        ]
        await store.save_events_batch("run-1", events)

        dag = await service.get_workflow_structure("trace-1")
        assert dag is not None
        step_map = {s.name: s for s in dag.steps}
        assert step_map["fetch"].status == "completed"
        assert step_map["parse"].status == "error"
        assert step_map["validate"].status == "pending"  # Not downstream of failed step
        assert step_map["combine"].status == "skipped"  # Transitively depends on "parse"

    async def test_transitive_skipped_status(
        self,
        store: InMemoryPersistentTraceStore,
        service: ObservatoryService,
    ) -> None:
        """All transitive dependents of a failed step are skipped."""
        await store.register_run("run-1", "trace-1", {})
        events = [
            _make_event(
                "workflow.structure",
                span_id="span-wf",
                payload={
                    "workflow_name": "wf",
                    "workflow_type": "sequential",
                    "steps": [
                        {"name": "a", "step_type": "function", "index": 0, "depends_on": [], "metadata": {}},
                        {"name": "b", "step_type": "function", "index": 1, "depends_on": ["a"], "metadata": {}},
                        {"name": "c", "step_type": "function", "index": 2, "depends_on": ["b"], "metadata": {}},
                    ],
                },
            ),
            _make_event(
                "workflow.error",
                span_id="span-wf",
                payload={
                    "workflow_name": "wf",
                    "workflow_type": "sequential",
                    "error_type": "ValueError",
                    "error_message": "bad",
                    "failed_step": "a",
                },
                time_offset_ms=10,
            ),
        ]
        await store.save_events_batch("run-1", events)

        dag = await service.get_workflow_structure("trace-1")
        assert dag is not None
        step_map = {s.name: s for s in dag.steps}
        assert step_map["a"].status == "error"
        assert step_map["b"].status == "skipped"
        assert step_map["c"].status == "skipped"

    async def test_completed_overrides_running(
        self,
        store: InMemoryPersistentTraceStore,
        service: ObservatoryService,
    ) -> None:
        """If both agent.start and workflow.step.complete exist, status is 'completed'."""
        await store.register_run("run-1", "trace-1", {})
        events = [
            _make_event(
                "workflow.structure",
                span_id="span-wf",
                payload={
                    "workflow_name": "wf",
                    "workflow_type": "sequential",
                    "steps": [
                        {"name": "s1", "step_type": "agent", "index": 0, "depends_on": [], "metadata": {}},
                    ],
                },
            ),
            _make_event(
                "agent.start",
                span_id="span-agent",
                parent_span_id="span-wf",
                payload={"agent_name": "s1", "task_input": "go", "tools_available": []},
                time_offset_ms=10,
            ),
            _make_event(
                "workflow.step.complete",
                span_id="span-wf",
                payload={"workflow_name": "wf", "step_name": "s1", "step_index": 0, "step_duration_ms": 100},
                time_offset_ms=20,
            ),
        ]
        await store.save_events_batch("run-1", events)

        dag = await service.get_workflow_structure("trace-1")
        assert dag is not None
        assert dag.steps[0].status == "completed"


# ---------------------------------------------------------------------------
# Tests: Event queries
# ---------------------------------------------------------------------------


class TestEventQueries:
    @pytest.fixture
    def store(self) -> InMemoryPersistentTraceStore:
        return InMemoryPersistentTraceStore()

    @pytest.fixture
    def service(self, store: InMemoryPersistentTraceStore) -> ObservatoryService:
        return ObservatoryService(store)

    async def test_query_events(
        self,
        store: InMemoryPersistentTraceStore,
        service: ObservatoryService,
    ) -> None:
        await _seed_multi_agent_trace(store)

        result = await service.query_events("run-1")
        assert len(result.events) > 0

    async def test_query_events_with_limit(
        self,
        store: InMemoryPersistentTraceStore,
        service: ObservatoryService,
    ) -> None:
        await _seed_multi_agent_trace(store)

        result = await service.query_events("run-1", limit=2)
        assert len(result.events) == 2
        assert result.has_more is True

    async def test_get_event(
        self,
        store: InMemoryPersistentTraceStore,
        service: ObservatoryService,
    ) -> None:
        await _seed_multi_agent_trace(store)
        # Event IDs start at 1
        event = await service.get_event(1)
        assert event is not None
        assert event.id == 1

    async def test_get_event_not_found(self, service: ObservatoryService) -> None:
        result = await service.get_event(99999)
        assert result is None

    async def test_get_events_for_span(
        self,
        store: InMemoryPersistentTraceStore,
        service: ObservatoryService,
    ) -> None:
        await _seed_multi_agent_trace(store)

        result = await service.get_events_for_span("trace-1", "span-child-1")
        assert result.span_id == "span-child-1"
        assert len(result.events) > 0
        assert all(e.span_id == "span-child-1" for e in result.events)

    async def test_get_events_for_span_with_level_filter(
        self,
        store: InMemoryPersistentTraceStore,
        service: ObservatoryService,
    ) -> None:
        await _seed_multi_agent_trace(store)

        result = await service.get_events_for_span("trace-1", "span-child-1", levels=["info"])
        for e in result.events:
            assert e.level == "info"

    async def test_get_events_for_span_with_event_type_filter(
        self,
        store: InMemoryPersistentTraceStore,
        service: ObservatoryService,
    ) -> None:
        await _seed_multi_agent_trace(store)

        result = await service.get_events_for_span("trace-1", "span-child-1", event_types=["tool.invoke"])
        assert len(result.events) == 1
        assert result.events[0].event_type == "tool.invoke"

    async def test_query_events_with_after_id_cursor(
        self,
        store: InMemoryPersistentTraceStore,
        service: ObservatoryService,
    ) -> None:
        await _seed_multi_agent_trace(store)

        # Get the first page
        first_page = await service.query_events("run-1", limit=3)
        assert len(first_page.events) == 3
        last_id = first_page.events[-1].id

        # Resume from cursor
        second_page = await service.query_events("run-1", after_id=last_id, limit=3)
        assert len(second_page.events) > 0
        # All events in second page should have ids > last_id
        for e in second_page.events:
            assert e.id > last_id
        # No overlap between pages
        first_ids = {e.id for e in first_page.events}
        second_ids = {e.id for e in second_page.events}
        assert first_ids.isdisjoint(second_ids)

    async def test_query_events_with_level_filter(
        self,
        store: InMemoryPersistentTraceStore,
        service: ObservatoryService,
    ) -> None:
        await _seed_multi_agent_trace(store)

        result = await service.query_events("run-1", levels=["info"])
        assert len(result.events) > 0
        for e in result.events:
            assert e.level == "info"

    async def test_query_events_with_event_type_filter(
        self,
        store: InMemoryPersistentTraceStore,
        service: ObservatoryService,
    ) -> None:
        await _seed_multi_agent_trace(store)

        result = await service.query_events("run-1", event_types=["llm.response"])
        assert len(result.events) > 0
        for e in result.events:
            assert e.event_type == "llm.response"

    async def test_get_agent_stats_nonexistent_span(
        self,
        store: InMemoryPersistentTraceStore,
        service: ObservatoryService,
    ) -> None:
        await _seed_multi_agent_trace(store)

        result = await service.get_agent_stats("trace-1", "nonexistent")
        assert result is None

    async def test_get_run_summary(
        self,
        store: InMemoryPersistentTraceStore,
        service: ObservatoryService,
    ) -> None:
        await _seed_multi_agent_trace(store)

        summary = await service.get_run_summary("run-1")
        assert summary.total_events > 0
        assert summary.agent_names  # should have agent names

    async def test_single_event_trace(
        self,
        store: InMemoryPersistentTraceStore,
        service: ObservatoryService,
    ) -> None:
        await store.register_run("run-1", "trace-1", {})
        events = [
            _make_event(
                "agent.start",
                payload={
                    "agent_name": "solo",
                    "task_input": "hello",
                    "tools_available": [],
                },
            ),
        ]
        await store.save_events_batch("run-1", events)

        tree = await service.get_span_tree("trace-1")
        assert tree.root.span_id == "__root__"
        assert len(tree.root.children) == 1

        agents = await service.list_agents("trace-1")
        assert len(agents.agents) == 1
        assert agents.agents[0].agent_name == "solo"


# ---------------------------------------------------------------------------
# Helper for collecting events from tree
# ---------------------------------------------------------------------------


def _collect_tree_events(
    node: SpanTreeNodeResponse,
) -> list[TraceEventResponse]:
    """Recursively collect all TraceEventResponse objects from a tree."""
    result = list(node.events)
    for child in node.children:
        result.extend(_collect_tree_events(child))
    return result


class TestBuildSpanTreeParentUpdate:
    """Test late parent_span_id discovery in _build_span_tree."""

    async def test_late_parent_span_id_assignment(self) -> None:
        store = InMemoryPersistentTraceStore()
        service = ObservatoryService(store)
        await service.register_run("run-1", "trace-1")
        # First event for span-child arrives WITHOUT parent_span_id
        # Second event for the same span arrives WITH parent_span_id
        events = [
            _make_event(
                "agent.start",
                trace_id="trace-1",
                span_id="span-parent",
                payload={"agent_name": "parent"},
            ),
            _make_event(
                "agent.step",
                trace_id="trace-1",
                span_id="span-child",
                parent_span_id=None,
                payload={"step_number": 1},
                time_offset_ms=10,
            ),
            _make_event(
                "agent.complete",
                trace_id="trace-1",
                span_id="span-child",
                parent_span_id="span-parent",
                payload={"agent_name": "child", "total_steps": 1, "termination_reason": "done"},
                time_offset_ms=20,
            ),
        ]
        await store.save_events_batch("run-1", events)
        tree = await service.get_span_tree("trace-1")
        # span-child should be parented under span-parent
        parent_node = next(c for c in tree.root.children if c.span_id == "span-parent")
        child_ids = {c.span_id for c in parent_node.children}
        assert "span-child" in child_ids
