"""Deterministic unit tests for the retrospective self-improver runner.

No Docker, no real Postgres, no real LLM — ``MockLLMClient`` scripts
the task-agent call, the advisor's ``analyze`` is patched at call site
via ``unittest.mock.patch("self_improver.runner.advisor_analyze", ...)``
so the stub is visible in each test (preferred over a shared conftest
fixture), and ``InMemoryPersistentTraceStore`` backs the shared
``TracedExecutor`` so trace reads are assertable.

``docker/full-stack/`` is not a Python package, but each runner inside
it is. The runtime image copies the three runner packages onto
``/srv``; this test file adds that directory to ``sys.path`` so
``self_improver`` imports the same way it does in production —
coverage then counts the lines that actually run in the image.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nanitics import (
    InMemoryPersistentTraceStore,
    MockLLMClient,
    TracedExecutor,
)
from nanitics.infrastructure.llm.protocol import LLMResponse, ToolCall
from nanitics.infrastructure.observability.events import (
    AgentStartEvent,
    AgentStepEvent,
    ToolInfo,
    Usage,
)
from nanitics.infrastructure.observability.storage import (
    StoredTraceEvent,
    TraceEventRecord,
)

# ---------------------------------------------------------------------------
# Path setup — make ``self_improver`` importable as a top-level package
# mirroring how ``/srv/`` lays it out in the compose image.
# ---------------------------------------------------------------------------

_FULL_STACK_DIR = Path(__file__).resolve().parent.parent / "docker" / "full-stack"
if str(_FULL_STACK_DIR) not in sys.path:
    sys.path.insert(0, str(_FULL_STACK_DIR))


from runners import ShellContext
from self_improver import runner as runner_module
from self_improver.advisor.analyze import AdvisorReport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _llm_final(text: str) -> LLMResponse:
    """Minimal assistant-final response."""
    return LLMResponse(
        content=text,
        tool_calls=[],
        usage=Usage(input_tokens=5, output_tokens=5),
        model="mock",
        stop_reason="end_turn",
    )


def _llm_tool_call(name: str, arguments: dict[str, Any], *, call_id: str = "call-1") -> LLMResponse:
    """Minimal tool-use response scripted for ``ReActAgent``'s tool-use loop."""
    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        usage=Usage(input_tokens=5, output_tokens=5),
        model="mock",
        stop_reason="tool_use",
    )


def _empty_report(trace_id: str) -> AdvisorReport:
    return AdvisorReport(
        trace_id=trace_id,
        generated_at=datetime.now(UTC),
        proposals=[],
        usage=Usage(input_tokens=0, output_tokens=0),
        rubric_counts={},
        target_dimensions_analyzed=["prompts"],
    )


def _stub_analyze_factory(captured: dict[str, Any]) -> Any:
    """Return an ``advisor_analyze`` stub that records the events it was called with.

    Test sites bind the stub via ``unittest.mock.patch`` directly — this
    factory is a small convenience to record call arguments without
    re-writing the side-effect body in every test.
    """

    async def _stub(
        events: list[Any],
        *,
        llm_client: Any,
        rubrics: Any | None = None,
        adapter: Any | None = None,
        emitter: Any | None = None,
    ) -> AdvisorReport:
        captured["events"] = events
        captured["llm_client"] = llm_client
        captured["emitter"] = emitter
        trace_id = events[0].trace_id if events else "stub-trace"
        return _empty_report(trace_id)

    return _stub


def _build_context(
    *,
    script: list[LLMResponse] | None = None,
    trace_store: InMemoryPersistentTraceStore | None = None,
) -> tuple[ShellContext, InMemoryPersistentTraceStore]:
    """Assemble a ``ShellContext`` wired to in-memory infrastructure."""
    store = trace_store if trace_store is not None else InMemoryPersistentTraceStore()
    executor = TracedExecutor(store)
    pool = MagicMock(name="asyncpg-pool-stub")
    responses = script if script is not None else [_llm_final("done")]
    context = ShellContext(
        executor=executor,
        trace_store=store,
        pool=pool,
        build_client=lambda: MockLLMClient(list(responses)),
    )
    return context, store


def _build_app(context: ShellContext) -> FastAPI:
    app = FastAPI()
    runner_module.register(app, context)
    return app


# ---------------------------------------------------------------------------
# Scenario 1 — task-mode happy path.
# ---------------------------------------------------------------------------


def test_task_mode_happy_path() -> None:
    """Empty POST body drives the bundled task; advisor is stubbed empty."""
    # Two scripted LLM calls drive one tool call then a final answer.
    script = [
        _llm_tool_call("list_bundled_docs", {}),
        _llm_final("01-overview.md is the starting point."),
    ]
    context, _store = _build_context(script=script)
    app = _build_app(context)

    captured: dict[str, Any] = {}
    with (
        patch(
            "self_improver.runner.advisor_analyze",
            side_effect=_stub_analyze_factory(captured),
        ),
        TestClient(app) as client,
    ):
        response = client.post("/runners/self-improver/run", json={})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["task_run_id"] is not None
    assert body["critic_run_id"] is not None
    # Run ids are distinct UUIDs per ``TracedExecutor`` — task and
    # critic are independent runs that land as two entries in Observatory.
    assert body["task_run_id"] != body["critic_run_id"]
    assert body["task_trace_id"]  # present
    # The critic received the events the store actually has under the
    # task's trace id — the helper round-tripped them.
    assert captured["events"], "advisor_analyze received no events"
    assert captured["events"][0].trace_id == body["task_trace_id"]


# ---------------------------------------------------------------------------
# Scenario 2 — referenced-trace mode.
# ---------------------------------------------------------------------------


def test_referenced_trace_mode() -> None:
    """When ``trace_id`` is supplied, the task phase is skipped entirely."""
    # Pre-seed the store with a known event set under trace id "seeded".
    seeded_trace = "seeded"
    seeded_event = AgentStartEvent(
        trace_id=seeded_trace,
        span_id="span-root",
        timestamp=datetime(2026, 4, 16, tzinfo=UTC),
        agent_name="external-task-agent",
        task_input="seeded task",
        model_name="claude-haiku",
        tools_available=["some_tool"],
        tool_schemas=[ToolInfo(name="some_tool", description="Does a thing.")],
    )
    store = InMemoryPersistentTraceStore()
    import asyncio

    async def _seed() -> None:
        await store.register_run("run-seed", seeded_trace, metadata={})
        record = TraceEventRecord(
            event_type=seeded_event.event_type,
            level="info",
            trace_id=seeded_event.trace_id,
            span_id=seeded_event.span_id,
            parent_span_id=seeded_event.parent_span_id,
            payload=seeded_event.model_dump(mode="json"),
            sdk_timestamp=seeded_event.timestamp,
        )
        await store.save_events_batch("run-seed", [record])

    asyncio.run(_seed())

    context, _ = _build_context(trace_store=store)
    app = _build_app(context)

    captured: dict[str, Any] = {}
    with (
        patch(
            "self_improver.runner.advisor_analyze",
            side_effect=_stub_analyze_factory(captured),
        ),
        TestClient(app) as client,
    ):
        response = client.post(
            "/runners/self-improver/run",
            json={"trace_id": seeded_trace},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["task_run_id"] is None
    assert body["task_trace_id"] == seeded_trace
    # advisor_analyze was called with the single seeded event.
    assert len(captured["events"]) == 1
    assert captured["events"][0].trace_id == seeded_trace
    assert captured["events"][0].agent_name == "external-task-agent"


# ---------------------------------------------------------------------------
# Scenario 3 — missing trace id returns 404.
# ---------------------------------------------------------------------------


def test_missing_trace_id_returns_404() -> None:
    """A ``trace_id`` that has no stored events yields a 404."""
    context, _ = _build_context()
    app = _build_app(context)

    with (
        patch(
            "self_improver.runner.advisor_analyze",
            new=AsyncMock(side_effect=AssertionError("advisor must not be called")),
        ),
        TestClient(app) as client,
    ):
        response = client.post(
            "/runners/self-improver/run",
            json={"trace_id": "does-not-exist"},
        )

    assert response.status_code == 404
    body = response.json()
    assert body["detail"]["error"] == "trace_not_found"
    assert body["detail"]["trace_id"] == "does-not-exist"


# ---------------------------------------------------------------------------
# Scenario 4 — corpus tool path guard.
# ---------------------------------------------------------------------------


class TestCorpusToolPathGuard:
    """``read_bundled_doc`` must refuse paths outside the corpus directory."""

    def test_rejects_parent_traversal(self) -> None:
        import asyncio

        with pytest.raises(ValueError, match="refusing to read outside corpus"):
            asyncio.run(runner_module.read_bundled_doc.execute(filename="../etc/passwd"))

    def test_rejects_nested_traversal(self) -> None:
        import asyncio

        # Nested traversal that actually escapes the corpus directory.
        with pytest.raises(ValueError, match="refusing to read outside corpus"):
            asyncio.run(runner_module.read_bundled_doc.execute(filename="a/../../etc/passwd"))

    def test_rejects_absolute_path(self) -> None:
        import asyncio

        with pytest.raises(ValueError, match="refusing to read outside corpus"):
            asyncio.run(runner_module.read_bundled_doc.execute(filename="/etc/passwd"))

    def test_missing_file_raises_file_not_found(self) -> None:
        import asyncio

        with pytest.raises(FileNotFoundError, match="corpus file not found"):
            asyncio.run(runner_module.read_bundled_doc.execute(filename="no-such-file.md"))

    def test_happy_path_reads_real_corpus_file(self) -> None:
        """A legitimate filename returns the file's text."""
        import asyncio

        result = asyncio.run(runner_module.read_bundled_doc.execute(filename="01-overview.md"))
        assert "Observability" in result.content


# ---------------------------------------------------------------------------
# Scenario 5 — corpus tool manifest matches the directory.
# ---------------------------------------------------------------------------


def test_corpus_tool_manifest_matches_corpus_directory() -> None:
    """``list_bundled_docs`` returns every ``.md`` file in the corpus."""
    import asyncio

    result = asyncio.run(runner_module.list_bundled_docs.execute())
    manifest = set(result.content.splitlines())
    on_disk = {p.name for p in runner_module.CORPUS_DIR.iterdir() if p.is_file() and p.suffix == ".md"}
    assert manifest == on_disk
    assert manifest, "corpus must not be empty"


def test_corpus_tool_manifest_raises_when_directory_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Deleting the corpus directory surfaces as a clear error, not silence."""
    import asyncio

    missing = tmp_path / "no-such-corpus"
    monkeypatch.setattr(runner_module, "CORPUS_DIR", missing)
    with pytest.raises(FileNotFoundError, match="corpus directory missing"):
        asyncio.run(runner_module.list_bundled_docs.execute())


# ---------------------------------------------------------------------------
# Scenario 6 — task agent iteration cap is 6.
# ---------------------------------------------------------------------------


def test_task_agent_iteration_cap_is_six() -> None:
    """The built agent's limiter is pinned to :data:`TASK_ITERATION_CAP`."""
    from nanitics.infrastructure.observability.emitter import InMemoryEmitter

    client = MockLLMClient([])
    emitter = InMemoryEmitter(trace_id="cap-test")
    agent = runner_module.build_task_agent(client, emitter)
    assert agent._limiter.max_iterations == 6
    # Also exactly the two corpus tools — nothing more, nothing less.
    tool_names = {s.name for s in agent._tool_registry.list_schemas()}
    assert tool_names == {"list_bundled_docs", "read_bundled_doc"}


# ---------------------------------------------------------------------------
# Scenario 7 — extra fields in the request body return 422.
# ---------------------------------------------------------------------------


def test_extra_fields_in_request_return_422() -> None:
    """``RunRequest.model_config = ConfigDict(extra='forbid')`` is honoured."""
    context, _ = _build_context()
    app = _build_app(context)

    with TestClient(app) as client:
        response = client.post(
            "/runners/self-improver/run",
            json={"task_input": "x", "unknown": "y"},
        )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Module-constants smoke check — the registration contract.
# ---------------------------------------------------------------------------


def test_registration_is_in_registrations() -> None:
    """``runners.REGISTRATIONS`` must include a ``self-improver`` entry."""
    from runners import REGISTRATIONS

    slugs = [r.slug for r in REGISTRATIONS]
    assert "self-improver" in slugs


def test_caching_client_passthrough_for_non_anthropic() -> None:
    """``_build_caching_client`` is a no-op for non-Anthropic clients.

    OpenAI-backed deployments run the critic without cache opt-in —
    this pins that behaviour.
    """
    mock_client = MockLLMClient([])
    result = runner_module._build_caching_client(mock_client)
    assert result is mock_client


def test_caching_client_missing_env_returns_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the env vars needed to reconstruct are missing, return the base client.

    Defensive path — the live-compose smoke always sets the env vars, so
    the Anthropic-with-caching construction path is covered by the step-9
    live run. Here we confirm the fallback does not explode.
    """
    from nanitics import AnthropicLLMClient

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("NANITICS_LLM_MODEL", raising=False)

    base = AnthropicLLMClient(model="", api_key="unused")
    result = runner_module._build_caching_client(base)
    assert result is base


def test_caching_client_reconstructs_with_caching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reconstruct an ``AnthropicLLMClient`` with caching when env is set."""
    from nanitics import AnthropicLLMClient

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("NANITICS_LLM_MODEL", "claude-haiku-4-5-20251001")

    base = AnthropicLLMClient(model="claude-haiku-4-5-20251001", api_key="test-key")
    result = runner_module._build_caching_client(base)
    # Not the same instance — it's a fresh client constructed with caching.
    assert isinstance(result, AnthropicLLMClient)
    assert result is not base
    # The caching flag is a private attribute on AnthropicLLMClient; read it
    # here because this is the runner's own reconstruction path and the test
    # would otherwise have no way to distinguish the two clients.
    assert result._enable_caching is True


# ---------------------------------------------------------------------------
# Stored-event consumption — confirms `trace_events_from_stored` is wired up.
# ---------------------------------------------------------------------------


def test_referenced_trace_mode_feeds_typed_events_to_analyze() -> None:
    """The helper converts ``StoredTraceEvent`` rows to ``TraceEvent`` before analyze.

    Regression pin — if someone removes the ``trace_events_from_stored``
    call and passes raw stored rows, advisor's type adapter blows up;
    this test asserts the typed shape.
    """
    seeded_trace = "seeded-typed"
    seeded_event = AgentStepEvent(
        trace_id=seeded_trace,
        span_id="span-root",
        timestamp=datetime(2026, 4, 16, tzinfo=UTC),
        agent_name="external",
        step_number=1,
        thought="seed",
        action="noop",
        observation="noop-result",
    )
    store = InMemoryPersistentTraceStore()
    import asyncio

    async def _seed() -> None:
        await store.register_run("run-seed-typed", seeded_trace, metadata={})
        record = TraceEventRecord(
            event_type=seeded_event.event_type,
            level="info",
            trace_id=seeded_event.trace_id,
            span_id=seeded_event.span_id,
            parent_span_id=seeded_event.parent_span_id,
            payload=seeded_event.model_dump(mode="json"),
            sdk_timestamp=seeded_event.timestamp,
        )
        await store.save_events_batch("run-seed-typed", [record])

    asyncio.run(_seed())

    context, _ = _build_context(trace_store=store)
    app = _build_app(context)

    captured: dict[str, Any] = {}
    with (
        patch(
            "self_improver.runner.advisor_analyze",
            side_effect=_stub_analyze_factory(captured),
        ),
        TestClient(app) as client,
    ):
        response = client.post(
            "/runners/self-improver/run",
            json={"trace_id": seeded_trace},
        )

    assert response.status_code == 200
    # The event reaching analyze must be a typed ``AgentStepEvent``, not
    # a raw ``StoredTraceEvent`` — that is what
    # ``trace_events_from_stored`` exists to guarantee.
    passed = captured["events"]
    assert len(passed) == 1
    assert isinstance(passed[0], AgentStepEvent)
    assert passed[0].step_number == 1


# Reference to StoredTraceEvent so linters don't drop the import when the
# test expansion above is refactored — the module-level import is used
# intentionally for cross-test readability.
_ = StoredTraceEvent
