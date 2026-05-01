"""Integration tests for the observatory FastAPI router."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from nanitics.infrastructure.observability.levels import TraceLevel
from nanitics.infrastructure.observability.storage import (
    InMemoryPersistentTraceStore,
    TraceEventRecord,
)
from nanitics.observatory.router import create_observatory_router

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


async def _seed_run_with_events(
    store: InMemoryPersistentTraceStore,
    run_id: str = "run-1",
    trace_id: str = "trace-1",
) -> None:
    """Seed a run with a small multi-agent trace."""
    await store.register_run(run_id, trace_id, {"type": "test"})
    events = [
        _make_event(
            "span.start",
            trace_id=trace_id,
            span_id="span-root",
            payload={"name": "root"},
            level="verbose",
        ),
        _make_event(
            "agent.start",
            trace_id=trace_id,
            span_id="span-agent",
            parent_span_id="span-root",
            payload={
                "agent_name": "worker",
                "agent_type": "basic",
                "task_input": "task",
                "tools_available": ["tool_a"],
                "capabilities": ["planning"],
            },
            time_offset_ms=10,
        ),
        _make_event(
            "agent.step",
            trace_id=trace_id,
            span_id="span-agent",
            parent_span_id="span-root",
            payload={"agent_name": "worker", "step_number": 1},
            level="debug",
            time_offset_ms=20,
        ),
        _make_event(
            "llm.response",
            trace_id=trace_id,
            span_id="span-agent",
            parent_span_id="span-root",
            payload={"usage": {"input_tokens": 100, "output_tokens": 50}},
            level="debug",
            time_offset_ms=30,
        ),
        _make_event(
            "tool.invoke",
            trace_id=trace_id,
            span_id="span-agent",
            parent_span_id="span-root",
            payload={"tool_name": "tool_a", "input": "x"},
            level="debug",
            time_offset_ms=40,
        ),
        _make_event(
            "agent.complete",
            trace_id=trace_id,
            span_id="span-agent",
            parent_span_id="span-root",
            payload={"agent_name": "worker", "total_steps": 1, "termination_reason": "done"},
            time_offset_ms=50,
        ),
        _make_event(
            "span.end",
            trace_id=trace_id,
            span_id="span-root",
            payload={"name": "root", "duration_ms": 50.0},
            level="verbose",
            time_offset_ms=50,
        ),
    ]
    await store.save_events_batch(run_id, events)


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
    router = create_observatory_router(store)
    app.include_router(router, prefix="/api/observatory")
    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    )


# ---------------------------------------------------------------------------
# Tests: Run endpoints
# ---------------------------------------------------------------------------


class TestListRuns:
    async def test_empty(self, client: AsyncClient) -> None:
        resp = await client.get("/api/observatory/runs")
        assert resp.status_code == 200
        body = resp.json()
        assert body["runs"] == []
        assert body["total"] == 0

    async def test_returns_runs(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        await store.register_run("run-1", "trace-1", {})
        await store.register_run("run-2", "trace-2", {})
        resp = await client.get("/api/observatory/runs")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        assert len(body["runs"]) == 2

    async def test_status_filter(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        await store.register_run("run-1", "trace-1", {})
        await store.register_run("run-2", "trace-2", {})
        await store.update_run_status("run-2", "completed")
        resp = await client.get("/api/observatory/runs?status=completed")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["runs"][0]["run"]["id"] == "run-2"

    async def test_pagination(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        for i in range(5):
            await store.register_run(f"run-{i}", f"trace-{i}", {})
        resp = await client.get("/api/observatory/runs?limit=2&offset=0")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["runs"]) == 2
        assert body["total"] == 5


class TestGetRun:
    async def test_found(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        await _seed_run_with_events(store)
        resp = await client.get("/api/observatory/runs/run-1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["run"]["id"] == "run-1"
        assert body["run"]["trace_id"] == "trace-1"
        assert "summary" in body

    async def test_not_found(self, client: AsyncClient) -> None:
        resp = await client.get("/api/observatory/runs/nonexistent")
        assert resp.status_code == 404


class TestCreateRun:
    async def test_creates_run(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        resp = await client.post(
            "/api/observatory/runs",
            json={"run_id": "new-run", "trace_id": "new-trace", "metadata": {"key": "val"}},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["id"] == "new-run"
        assert body["trace_id"] == "new-trace"
        assert body["metadata"] == {"key": "val"}
        # Verify it's in the store
        run = await store.get_run("new-run")
        assert run is not None

    async def test_default_metadata(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/observatory/runs",
            json={"run_id": "r", "trace_id": "t"},
        )
        assert resp.status_code == 201
        assert resp.json()["metadata"] == {}


class TestDeleteRun:
    async def test_deletes_run(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        await store.register_run("run-1", "trace-1", {})
        resp = await client.delete("/api/observatory/runs/run-1")
        assert resp.status_code == 204
        run = await store.get_run("run-1")
        assert run is None

    async def test_not_found(self, client: AsyncClient) -> None:
        resp = await client.delete("/api/observatory/runs/nonexistent")
        assert resp.status_code == 404


class TestUpdateRunStatus:
    async def test_updates_status(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        await store.register_run("run-1", "trace-1", {})
        resp = await client.patch(
            "/api/observatory/runs/run-1/status",
            json={"status": "completed"},
        )
        assert resp.status_code == 204
        run = await store.get_run("run-1")
        assert run is not None
        assert run.status == "completed"

    async def test_updates_with_error(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        await store.register_run("run-1", "trace-1", {})
        resp = await client.patch(
            "/api/observatory/runs/run-1/status",
            json={"status": "failed", "error": "something broke"},
        )
        assert resp.status_code == 204
        run = await store.get_run("run-1")
        assert run is not None
        assert run.status == "failed"
        assert run.error == "something broke"

    async def test_not_found(self, client: AsyncClient) -> None:
        resp = await client.patch(
            "/api/observatory/runs/nonexistent/status",
            json={"status": "completed"},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Tests: Trace hierarchy endpoints
# ---------------------------------------------------------------------------


class TestSpanTree:
    async def test_returns_tree(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        await _seed_run_with_events(store)
        resp = await client.get("/api/observatory/runs/run-1/tree")
        assert resp.status_code == 200
        body = resp.json()
        assert body["trace_id"] == "trace-1"
        assert "root" in body
        root = body["root"]
        assert root["span_id"] == "__root__"
        # Should have children
        assert len(root["children"]) > 0

    async def test_min_level_filter(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        await _seed_run_with_events(store)
        # With info level, only info events
        resp = await client.get("/api/observatory/runs/run-1/tree?min_level=info")
        assert resp.status_code == 200
        body = resp.json()
        # Traverse and check event levels
        all_events = _collect_events_from_tree(body["root"])
        for ev in all_events:
            assert ev["level"] == "info"

    async def test_run_not_found(self, client: AsyncClient) -> None:
        resp = await client.get("/api/observatory/runs/nonexistent/tree")
        assert resp.status_code == 404

    async def test_invalid_min_level(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        await _seed_run_with_events(store)
        resp = await client.get("/api/observatory/runs/run-1/tree?min_level=invalid")
        assert resp.status_code == 422


class TestSpanEvents:
    async def test_returns_events(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        await _seed_run_with_events(store)
        resp = await client.get("/api/observatory/runs/run-1/spans/span-agent/events")
        assert resp.status_code == 200
        body = resp.json()
        assert body["span_id"] == "span-agent"
        assert len(body["events"]) > 0

    async def test_with_level_filter(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        await _seed_run_with_events(store)
        resp = await client.get("/api/observatory/runs/run-1/spans/span-agent/events?levels=info")
        assert resp.status_code == 200
        body = resp.json()
        for ev in body["events"]:
            assert ev["level"] == "info"

    async def test_with_type_filter(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        await _seed_run_with_events(store)
        resp = await client.get("/api/observatory/runs/run-1/spans/span-agent/events?event_types=llm.response")
        assert resp.status_code == 200
        body = resp.json()
        for ev in body["events"]:
            assert ev["event_type"] == "llm.response"

    async def test_run_not_found(self, client: AsyncClient) -> None:
        resp = await client.get("/api/observatory/runs/nonexistent/spans/s/events")
        assert resp.status_code == 404

    async def test_invalid_level(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        await _seed_run_with_events(store)
        resp = await client.get("/api/observatory/runs/run-1/spans/span-agent/events?levels=invalid")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Tests: Agent endpoints
# ---------------------------------------------------------------------------


class TestListAgents:
    async def test_returns_agents(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        await _seed_run_with_events(store)
        resp = await client.get("/api/observatory/runs/run-1/agents")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["agents"]) == 1
        agent = body["agents"][0]
        assert agent["agent_name"] == "worker"
        assert agent["agent_type"] == "basic"
        assert "stats" in agent

    async def test_run_not_found(self, client: AsyncClient) -> None:
        resp = await client.get("/api/observatory/runs/nonexistent/agents")
        assert resp.status_code == 404


class TestGetAgentDetail:
    async def test_found(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        await _seed_run_with_events(store)
        resp = await client.get("/api/observatory/runs/run-1/agents/span-agent")
        assert resp.status_code == 200
        body = resp.json()
        assert body["agent"]["agent_name"] == "worker"
        assert "events" in body
        assert "span_tree" in body

    async def test_agent_not_found(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        await _seed_run_with_events(store)
        resp = await client.get("/api/observatory/runs/run-1/agents/nonexistent")
        assert resp.status_code == 404

    async def test_run_not_found(self, client: AsyncClient) -> None:
        resp = await client.get("/api/observatory/runs/nonexistent/agents/span-1")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Tests: Workflow endpoint
# ---------------------------------------------------------------------------


class TestWorkflow:
    async def test_returns_workflow(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        await store.register_run("run-1", "trace-1", {})
        events = [
            _make_event(
                "workflow.structure",
                trace_id="trace-1",
                span_id="span-wf",
                payload={
                    "workflow_name": "my_workflow",
                    "workflow_type": "sequential",
                    "steps": [
                        {"name": "step_a", "step_type": "agent", "depends_on": []},
                        {"name": "step_b", "step_type": "agent", "depends_on": ["step_a"]},
                    ],
                },
            ),
            _make_event(
                "workflow.step.complete",
                trace_id="trace-1",
                span_id="span-wf",
                payload={"step_name": "step_a", "step_duration_ms": 100.0},
                time_offset_ms=100,
            ),
        ]
        await store.save_events_batch("run-1", events)

        resp = await client.get("/api/observatory/runs/run-1/workflow")
        assert resp.status_code == 200
        body = resp.json()
        assert body["workflow_name"] == "my_workflow"
        assert len(body["steps"]) == 2
        assert body["steps"][0]["status"] == "completed"
        assert body["steps"][1]["status"] == "pending"

    async def test_no_workflow(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        await _seed_run_with_events(store)
        resp = await client.get("/api/observatory/runs/run-1/workflow")
        assert resp.status_code == 404

    async def test_run_not_found(self, client: AsyncClient) -> None:
        resp = await client.get("/api/observatory/runs/nonexistent/workflow")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Tests: Flat event endpoints
# ---------------------------------------------------------------------------


class TestListEvents:
    async def test_returns_events(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        await _seed_run_with_events(store)
        resp = await client.get("/api/observatory/runs/run-1/events")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["events"]) > 0
        assert "has_more" in body

    async def test_level_filter(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        await _seed_run_with_events(store)
        resp = await client.get("/api/observatory/runs/run-1/events?level=info")
        assert resp.status_code == 200
        body = resp.json()
        for ev in body["events"]:
            assert ev["level"] == "info"

    async def test_event_type_filter(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        await _seed_run_with_events(store)
        resp = await client.get("/api/observatory/runs/run-1/events?event_types=agent.start,agent.complete")
        assert resp.status_code == 200
        body = resp.json()
        for ev in body["events"]:
            assert ev["event_type"] in ("agent.start", "agent.complete")

    async def test_pagination_limit(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        await _seed_run_with_events(store)
        resp = await client.get("/api/observatory/runs/run-1/events?limit=2")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["events"]) == 2
        assert body["has_more"] is True

    async def test_pagination_cursor(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        await _seed_run_with_events(store)
        # Get first page
        resp1 = await client.get("/api/observatory/runs/run-1/events?limit=2")
        body1 = resp1.json()
        last_id = body1["events"][-1]["id"]
        # Get second page
        resp2 = await client.get(f"/api/observatory/runs/run-1/events?limit=2&after={last_id}")
        assert resp2.status_code == 200
        body2 = resp2.json()
        for ev in body2["events"]:
            assert ev["id"] > last_id

    async def test_run_not_found(self, client: AsyncClient) -> None:
        resp = await client.get("/api/observatory/runs/nonexistent/events")
        assert resp.status_code == 404


class TestGetEvent:
    async def test_found(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        await _seed_run_with_events(store)
        resp = await client.get("/api/observatory/events/1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == 1

    async def test_not_found(self, client: AsyncClient) -> None:
        resp = await client.get("/api/observatory/events/99999")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Tests: Summary endpoint
# ---------------------------------------------------------------------------


class TestSummary:
    async def test_returns_summary(self, client: AsyncClient, store: InMemoryPersistentTraceStore) -> None:
        await _seed_run_with_events(store)
        resp = await client.get("/api/observatory/runs/run-1/summary")
        assert resp.status_code == 200
        body = resp.json()
        assert "total_events" in body
        assert "llm_calls" in body
        assert "tool_calls" in body

    async def test_run_not_found(self, client: AsyncClient) -> None:
        resp = await client.get("/api/observatory/runs/nonexistent/summary")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Tests: Embedded UI
# ---------------------------------------------------------------------------


class TestEmbeddedUI:
    async def test_root_serves_fallback_when_no_static_dir(self, client: AsyncClient) -> None:
        resp = await client.get("/api/observatory/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "observatory-build" in resp.text

    async def test_root_serves_index_when_built(self, store: InMemoryPersistentTraceStore, tmp_path: Path) -> None:
        (tmp_path / "index.html").write_text("<html>observatory</html>")

        from fastapi import FastAPI

        app = FastAPI()
        router = create_observatory_router(store, static_dir=tmp_path)
        app.include_router(router, prefix="/api/observatory")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/api/observatory/")
            assert resp.status_code == 200
            assert "observatory" in resp.text

    async def test_assets_not_found(self, client: AsyncClient) -> None:
        resp = await client.get("/api/observatory/assets/nonexistent.js")
        assert resp.status_code == 404

    async def test_assets_not_found_with_static_dir(self, store: InMemoryPersistentTraceStore, tmp_path: Path) -> None:
        (tmp_path / "assets").mkdir()
        (tmp_path / "index.html").write_text("<html></html>")

        from fastapi import FastAPI

        app = FastAPI()
        router = create_observatory_router(store, static_dir=tmp_path)
        app.include_router(router, prefix="/api/observatory")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/api/observatory/assets/missing.js")
            assert resp.status_code == 404

    async def test_assets_served_with_cache_headers(self, store: InMemoryPersistentTraceStore, tmp_path: Path) -> None:
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        (assets_dir / "app.js").write_bytes(b"console.log('hello');")
        (tmp_path / "index.html").write_text("<html></html>")

        from fastapi import FastAPI

        app = FastAPI()
        router = create_observatory_router(store, static_dir=tmp_path)
        app.include_router(router, prefix="/api/observatory")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/api/observatory/assets/app.js")
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "application/javascript"
            assert "immutable" in resp.headers["cache-control"]
            assert resp.content == b"console.log('hello');"

    async def test_assets_without_cache_headers(self, store: InMemoryPersistentTraceStore, tmp_path: Path) -> None:
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        (assets_dir / "logo.png").write_bytes(b"\x89PNG")
        (tmp_path / "index.html").write_text("<html></html>")

        from fastapi import FastAPI

        app = FastAPI()
        router = create_observatory_router(store, static_dir=tmp_path)
        app.include_router(router, prefix="/api/observatory")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/api/observatory/assets/logo.png")
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "image/png"
            assert "cache-control" not in resp.headers

    async def test_assets_path_traversal_blocked(self, store: InMemoryPersistentTraceStore, tmp_path: Path) -> None:
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        (tmp_path / "secret.txt").write_text("secret")
        (tmp_path / "index.html").write_text("<html></html>")

        from fastapi import FastAPI

        app = FastAPI()
        router = create_observatory_router(store, static_dir=tmp_path)
        app.include_router(router, prefix="/api/observatory")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/api/observatory/assets/../secret.txt")
            assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_events_from_tree(node: dict) -> list[dict]:
    """Recursively collect all events from a span tree node."""
    events = list(node.get("events", []))
    for child in node.get("children", []):
        events.extend(_collect_events_from_tree(child))
    return events
