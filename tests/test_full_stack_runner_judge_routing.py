"""Unit + end-to-end tests for the judge-routing runner.

Covers the eleven tool factories in ``judge_routing.tools`` and the
module-state reset fixture in ``judge_routing.fixtures``, plus the
end-to-end ``/handle`` route, the runner-local
``_GroundedJudgeRouter`` cost grounding, the trace shape (1 start +
4 ranking + 1 allocated + 1 complete), and the failure paths (422 on
empty body, 503 on judge LLM raise / unknown-agent ranking).

``docker/full-stack/`` is not a Python package; the runtime image lays
its files out flat on ``/srv``. Tests add that directory to ``sys.path``
so ``judge_routing.tools`` imports as a package submodule — mirroring
the image layout exactly, so coverage counts the lines that actually
run in production.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from nanitics import (
    BiddableAgent,
    FixedBidGenerator,
    InMemoryPersistentTraceStore,
    LLMResponse,
    MockLLMClient,
    ReActAgent,
    TracedExecutor,
    Usage,
)
from nanitics.infrastructure.errors import ToolParameterError
from nanitics.infrastructure.observability.emitter import InMemoryEmitter

# ── Path setup — make ``judge_routing`` importable as a package ──

_FULL_STACK_DIR = Path(__file__).resolve().parent.parent / "docker" / "full-stack"
if str(_FULL_STACK_DIR) not in sys.path:
    sys.path.insert(0, str(_FULL_STACK_DIR))

from judge_routing import fixtures, tools

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_mutable_state() -> Iterator[None]:
    """Clear module-level mutable state before and after each test.

    The mutation tools write to ``fixtures.MUTABLE_STATE``; without
    this fixture state would leak across tests.
    """
    fixtures.reset_state()
    yield
    fixtures.reset_state()


# ── reset_state ───────────────────────────────────────────────


def test_reset_state_clears_all_buckets() -> None:
    fixtures.MUTABLE_STATE["refunds"].append({"x": 1})
    fixtures.MUTABLE_STATE["password_resets"]["ACC"] = "now"
    fixtures.MUTABLE_STATE["profile_updates"]["ACC"] = {"k": "v"}
    fixtures.MUTABLE_STATE["bugs"].append({"x": 1})

    fixtures.reset_state()

    assert fixtures.MUTABLE_STATE == {
        "refunds": [],
        "password_resets": {},
        "profile_updates": {},
        "bugs": [],
    }


# ── Lookup helpers raise on unknown ids ───────────────────────


def test_find_invoice_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown invoice_id"):
        fixtures.find_invoice("missing")


def test_find_account_by_id_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown account_id"):
        fixtures.find_account_by_id("missing")


def test_find_account_by_email_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown email"):
        fixtures.find_account_by_email("missing@example.com")


def test_find_clause_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown policy clause"):
        fixtures.find_clause("POL-REFUND", "999")


# ── Billing tools ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_lookup_invoice_returns_invoice_metadata() -> None:
    tool = tools.build_lookup_invoice_tool()
    result = await tool.execute(invoice_id="INV-1001")

    assert "INV-1001" in result.content
    assert result.metadata["invoice_id"] == "INV-1001"
    assert result.metadata["amount"] == 49.0


@pytest.mark.asyncio
async def test_lookup_invoice_unknown_raises() -> None:
    tool = tools.build_lookup_invoice_tool()
    with pytest.raises(ValueError, match="unknown invoice_id"):
        await tool.execute(invoice_id="MISSING")


@pytest.mark.asyncio
async def test_lookup_invoice_missing_param_raises() -> None:
    tool = tools.build_lookup_invoice_tool()
    with pytest.raises(ToolParameterError):
        await tool.execute()


@pytest.mark.asyncio
async def test_issue_refund_appends_record() -> None:
    tool = tools.build_issue_refund_tool()
    result = await tool.execute(invoice_id="INV-1002", amount=49.0, reason="Duplicate charge")

    assert "REF-0001" in result.content
    assert fixtures.MUTABLE_STATE["refunds"][0]["invoice_id"] == "INV-1002"
    assert fixtures.MUTABLE_STATE["refunds"][0]["amount"] == 49.0


@pytest.mark.asyncio
async def test_issue_refund_invalid_amount_raises() -> None:
    tool = tools.build_issue_refund_tool()
    with pytest.raises(ToolParameterError):
        await tool.execute(invoice_id="INV-1002", amount=0, reason="")


@pytest.mark.asyncio
async def test_issue_refund_unknown_invoice_raises() -> None:
    tool = tools.build_issue_refund_tool()
    with pytest.raises(ValueError, match="unknown invoice_id"):
        await tool.execute(invoice_id="MISSING", amount=10.0, reason="x")


# ── Technical tools ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_kb_finds_match() -> None:
    tool = tools.build_search_kb_tool()
    result = await tool.execute(query="webhook")

    assert "KB-002" in result.content
    assert result.metadata["hits"][0]["article_id"] == "KB-002"


@pytest.mark.asyncio
async def test_search_kb_no_match_returns_empty_hits() -> None:
    tool = tools.build_search_kb_tool()
    result = await tool.execute(query="quantum")

    assert "No KB articles match" in result.content
    assert result.metadata["hits"] == []


@pytest.mark.asyncio
async def test_search_kb_missing_query_raises() -> None:
    tool = tools.build_search_kb_tool()
    with pytest.raises(ToolParameterError):
        await tool.execute(query="")


@pytest.mark.asyncio
async def test_check_service_status_returns_status() -> None:
    tool = tools.build_check_service_status_tool()
    result = await tool.execute(service="api")

    assert "operational" in result.content
    assert result.metadata == {"service": "api", "status": "operational"}


@pytest.mark.asyncio
async def test_check_service_status_unknown_raises() -> None:
    tool = tools.build_check_service_status_tool()
    with pytest.raises(ValueError, match="unknown service"):
        await tool.execute(service="ghost")


@pytest.mark.asyncio
async def test_check_service_status_missing_param_raises() -> None:
    tool = tools.build_check_service_status_tool()
    with pytest.raises(ToolParameterError):
        await tool.execute()


@pytest.mark.asyncio
async def test_escalate_bug_appends_record() -> None:
    tool = tools.build_escalate_bug_tool()
    first = await tool.execute(summary="Webhook delivery flaky")
    second = await tool.execute(summary="API timeouts on /v1/x")

    assert "BUG-0001" in first.content
    assert "BUG-0002" in second.content
    assert len(fixtures.MUTABLE_STATE["bugs"]) == 2


@pytest.mark.asyncio
async def test_escalate_bug_empty_summary_raises() -> None:
    tool = tools.build_escalate_bug_tool()
    with pytest.raises(ToolParameterError):
        await tool.execute(summary="")


# ── Account tools ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_lookup_account_returns_account() -> None:
    tool = tools.build_lookup_account_tool()
    result = await tool.execute(email="ada@example.com")

    assert "Ada Lovelace" in result.content
    assert result.metadata["account_id"] == "ACC-001"


@pytest.mark.asyncio
async def test_lookup_account_unknown_email_raises() -> None:
    tool = tools.build_lookup_account_tool()
    with pytest.raises(ValueError, match="unknown email"):
        await tool.execute(email="ghost@example.com")


@pytest.mark.asyncio
async def test_lookup_account_missing_email_raises() -> None:
    tool = tools.build_lookup_account_tool()
    with pytest.raises(ToolParameterError):
        await tool.execute()


@pytest.mark.asyncio
async def test_reset_password_records_reset() -> None:
    tool = tools.build_reset_password_tool()
    result = await tool.execute(account_id="ACC-001")

    assert "ACC-001" in result.content
    assert "ACC-001" in fixtures.MUTABLE_STATE["password_resets"]


@pytest.mark.asyncio
async def test_reset_password_unknown_account_raises() -> None:
    tool = tools.build_reset_password_tool()
    with pytest.raises(ValueError, match="unknown account_id"):
        await tool.execute(account_id="MISSING")


@pytest.mark.asyncio
async def test_reset_password_missing_param_raises() -> None:
    tool = tools.build_reset_password_tool()
    with pytest.raises(ToolParameterError):
        await tool.execute()


@pytest.mark.asyncio
async def test_update_profile_merges_changes() -> None:
    tool = tools.build_update_profile_tool()
    await tool.execute(account_id="ACC-002", changes={"name": "Grace H."})
    result = await tool.execute(account_id="ACC-002", changes={"subscription_tier": "pro"})

    assert "ACC-002" in result.content
    stored = fixtures.MUTABLE_STATE["profile_updates"]["ACC-002"]
    assert stored == {"name": "Grace H.", "subscription_tier": "pro"}


@pytest.mark.asyncio
async def test_update_profile_empty_changes_raises() -> None:
    tool = tools.build_update_profile_tool()
    with pytest.raises(ValueError, match="changes must be a non-empty mapping"):
        await tool.execute(account_id="ACC-001", changes={})


@pytest.mark.asyncio
async def test_update_profile_unknown_account_raises() -> None:
    tool = tools.build_update_profile_tool()
    with pytest.raises(ValueError, match="unknown account_id"):
        await tool.execute(account_id="MISSING", changes={"name": "x"})


@pytest.mark.asyncio
async def test_update_profile_missing_account_id_raises() -> None:
    tool = tools.build_update_profile_tool()
    with pytest.raises(ToolParameterError):
        await tool.execute(changes={"name": "x"})


# ── Policy tools ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_lookup_policy_finds_match() -> None:
    tool = tools.build_lookup_policy_tool()
    result = await tool.execute(topic="refund")

    assert "POL-REFUND" in result.content
    assert result.metadata["hits"][0]["policy_id"] == "POL-REFUND"


@pytest.mark.asyncio
async def test_lookup_policy_no_match_returns_empty_hits() -> None:
    tool = tools.build_lookup_policy_tool()
    result = await tool.execute(topic="quantum")

    assert "No policy clauses match" in result.content
    assert result.metadata["hits"] == []


@pytest.mark.asyncio
async def test_lookup_policy_missing_topic_raises() -> None:
    tool = tools.build_lookup_policy_tool()
    with pytest.raises(ToolParameterError):
        await tool.execute(topic="")


@pytest.mark.asyncio
async def test_cite_clause_returns_body() -> None:
    tool = tools.build_cite_clause_tool()
    result = await tool.execute(policy_id="POL-DATA", section="3.4")

    assert "30 days" in result.content
    assert result.metadata == {"policy_id": "POL-DATA", "section": "3.4"}


@pytest.mark.asyncio
async def test_cite_clause_unknown_raises() -> None:
    tool = tools.build_cite_clause_tool()
    with pytest.raises(ValueError, match="unknown policy clause"):
        await tool.execute(policy_id="POL-DATA", section="9.9")


@pytest.mark.asyncio
async def test_cite_clause_missing_params_raises() -> None:
    tool = tools.build_cite_clause_tool()
    with pytest.raises(ToolParameterError):
        await tool.execute(policy_id="POL-DATA")


# ── Tool bundles ──────────────────────────────────────────────


def test_billing_tools_bundle_contents() -> None:
    bundle = tools.billing_tools()
    assert [t.schema.name for t in bundle] == ["lookup_invoice", "issue_refund"]


def test_technical_tools_bundle_contents() -> None:
    bundle = tools.technical_tools()
    assert [t.schema.name for t in bundle] == [
        "search_kb",
        "check_service_status",
        "escalate_bug",
    ]


def test_account_tools_bundle_contents() -> None:
    bundle = tools.account_tools()
    assert [t.schema.name for t in bundle] == [
        "lookup_account",
        "reset_password",
        "update_profile",
    ]


def test_policy_tools_bundle_contents() -> None:
    bundle = tools.policy_tools()
    assert [t.schema.name for t in bundle] == ["lookup_policy", "cite_clause"]


# ── Runner roster ─────────────────────────────────────────────


def test_runner_module_constants() -> None:
    """Slug / title / description / roster size are stable."""
    import judge_routing.runner as runner

    assert runner.RUNNER_SLUG == "judge-routing"
    assert runner.RUNNER_TITLE
    assert runner.RUNNER_DESCRIPTION
    assert len(runner.SPECIALIST_SPECS) == 4


def test_runner_specs_carry_tools() -> None:
    """Each spec's ``tools`` bundle matches its specialty."""
    import judge_routing.runner as runner

    by_name = {s.name: s for s in runner.SPECIALIST_SPECS}
    assert [t.schema.name for t in by_name["billing-specialist"].tools] == [
        "lookup_invoice",
        "issue_refund",
    ]
    assert [t.schema.name for t in by_name["technical-specialist"].tools] == [
        "search_kb",
        "check_service_status",
        "escalate_bug",
    ]
    assert [t.schema.name for t in by_name["account-specialist"].tools] == [
        "lookup_account",
        "reset_password",
        "update_profile",
    ]
    assert [t.schema.name for t in by_name["policy-specialist"].tools] == [
        "lookup_policy",
        "cite_clause",
    ]


# ── End-to-end fixtures ───────────────────────────────────────


def _grounded_ranking_json(
    confidences: list[float],
    complexities: list[int],
    names: list[str] | None = None,
    capabilities: list[list[str]] | None = None,
    reasonings: list[str] | None = None,
) -> str:
    """Build a ``_GroundedJudgeRankingSchema``-shaped JSON payload."""
    if names is None:
        names = [
            "billing-specialist",
            "technical-specialist",
            "account-specialist",
            "policy-specialist",
        ]
    if capabilities is None:
        capabilities = [["routing"]] * len(names)
    if reasonings is None:
        reasonings = ["scripted ranking entry"] * len(names)
    entries = []
    for name, conf, complexity, caps, reasoning in zip(
        names, confidences, complexities, capabilities, reasonings, strict=True
    ):
        caps_json = "[" + ", ".join(f'"{c}"' for c in caps) + "]"
        entries.append(
            f'{{"agent_name": "{name}", "confidence": {conf}, "capabilities": {caps_json}, '
            f'"complexity": {complexity}, "reasoning": "{reasoning}"}}'
        )
    return '{"ranking": [' + ", ".join(entries) + "]}"


def _llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=[],
        usage=Usage(input_tokens=8, output_tokens=4),
        model="mock",
        stop_reason="end_turn",
    )


def _build_e2e_specialists(scripted: list[LLMResponse] | None = None) -> tuple[list[BiddableAgent], MockLLMClient]:
    """Four specialists with a shared :class:`MockLLMClient`.

    The shared client is scripted up front (typically with the judge's
    ranking response followed by the winning ReActAgent's final
    assistant message). ``MockLLMClient`` does not expose a public
    ``responses`` mutator, so construction-time scripting is the test
    contract.
    """
    import judge_routing.runner as runner

    shared_client = MockLLMClient(scripted if scripted is not None else [])
    placeholder_emitter = InMemoryEmitter(trace_id="judge-routing-e2e")
    specialists: list[BiddableAgent] = []
    for spec in runner.SPECIALIST_SPECS:
        agent = ReActAgent(
            name=spec.name,
            llm_client=shared_client,
            emitter=placeholder_emitter,
            system_prompt=spec.system_prompt,
            tools=[],  # tests don't drive the ReAct tool loop here
            max_iterations=1,
        )
        specialists.append(BiddableAgent(agent=agent, bid_generator=FixedBidGenerator(confidence=0.0)))
    return specialists, shared_client


@pytest.fixture
def e2e_runner_module() -> Iterator[Any]:
    """Import ``judge_routing.runner`` with module-level state reset."""
    import judge_routing.runner as runner

    runner._specialists = []
    runner._executor = None
    runner._llm_client = None
    yield runner
    runner._specialists = []
    runner._executor = None
    runner._llm_client = None


def _make_runner_app(
    runner_module: Any,
    *,
    specialists: list[BiddableAgent],
    shared_client: MockLLMClient,
    trace_store: InMemoryPersistentTraceStore | None = None,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FastAPI, InMemoryPersistentTraceStore]:
    """Build a fresh FastAPI app with the judge-routing runner mounted."""
    store = trace_store if trace_store is not None else InMemoryPersistentTraceStore()
    executor = TracedExecutor(store)
    pool = MagicMock(name="asyncpg-pool-stub")

    monkeypatch.setattr(
        runner_module,
        "_build_specialists",
        lambda ctx: (specialists, shared_client),
    )

    from runners import ShellContext

    context = ShellContext(
        executor=executor,
        trace_store=store,
        pool=pool,
        build_client=lambda: shared_client,
    )

    app = FastAPI()
    runner_module.register(app, context)
    return app, store


async def _events_for_run(store: InMemoryPersistentTraceStore, run_id: str) -> list[Any]:
    events: list[Any] = []
    after: int | None = None
    while True:
        batch = await store.query_events(run_id, after_id=after, limit=500)
        if not batch:
            break
        events.extend(batch)
        after = batch[-1].id
    return events


# ── End-to-end happy path ─────────────────────────────────────


@pytest.mark.asyncio
async def test_handle_happy_path_routes_to_winner(
    e2e_runner_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The judge-ranked top candidate answers the request."""
    scripted = [
        _llm_response(
            _grounded_ranking_json(
                confidences=[0.9, 0.4, 0.3, 0.2],
                complexities=[3, 2, 1, 1],
            )
        ),
        _llm_response("Answer from billing-specialist"),
    ]
    specialists, shared_client = _build_e2e_specialists(scripted)

    app, store = _make_runner_app(
        e2e_runner_module,
        specialists=specialists,
        shared_client=shared_client,
        monkeypatch=monkeypatch,
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/runners/judge-routing/handle",
            json={"request_text": "Why is invoice INV-1001 unpaid?"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["winner"] == "billing-specialist"
    assert body["answer"] == "Answer from billing-specialist"
    assert len(body["ranking"]) == 4
    assert body["trace_url"] == f"/api/observatory/runs/{body['run_id']}"

    # Cost-grounding arithmetic: every ranking entry's estimated_cost
    # equals base_rate * complexity for the matching spec, in the
    # scripted order.
    import judge_routing.runner as runner

    base_rates = {s.name: s.base_rate for s in runner.SPECIALIST_SPECS}
    expected_costs = {
        "billing-specialist": base_rates["billing-specialist"] * 3,
        "technical-specialist": base_rates["technical-specialist"] * 2,
        "account-specialist": base_rates["account-specialist"] * 1,
        "policy-specialist": base_rates["policy-specialist"] * 1,
    }
    by_name = {entry["agent_name"]: entry for entry in body["ranking"]}
    for name, expected in expected_costs.items():
        assert by_name[name]["estimated_cost"] == pytest.approx(expected)

    # Trace shape: 1 start, 4 ranking, 1 allocated, 1 complete.
    events = await _events_for_run(store, body["run_id"])
    event_types = [e.event_type for e in events]
    assert event_types.count("multi_agent.judge_routing.start") == 1
    assert event_types.count("multi_agent.judge_routing.ranking") == 4
    assert event_types.count("multi_agent.judge_routing.allocated") == 1
    assert event_types.count("multi_agent.judge_routing.complete") == 1

    # Judge-phase LLM call carries label="judge".
    judge_calls = [e for e in events if e.event_type == "llm.response" and e.payload.get("label") == "judge"]
    assert len(judge_calls) == 1


# ── /handle 422 on empty body ─────────────────────────────────


def test_handle_empty_request_text_returns_422(
    e2e_runner_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specialists, shared_client = _build_e2e_specialists()
    app, _ = _make_runner_app(
        e2e_runner_module,
        specialists=specialists,
        shared_client=shared_client,
        monkeypatch=monkeypatch,
    )
    with TestClient(app) as client:
        response = client.post("/runners/judge-routing/handle", json={"request_text": ""})
    assert response.status_code == 422


# ── /handle 503 when judge LLM raises ─────────────────────────


def test_handle_returns_503_when_judge_llm_raises(
    e2e_runner_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty MockLLMClient raises on its first call — surfaces as 503."""
    specialists, shared_client = _build_e2e_specialists()
    # Don't extend ``shared_client.responses`` — first call will raise.
    app, _ = _make_runner_app(
        e2e_runner_module,
        specialists=specialists,
        shared_client=shared_client,
        monkeypatch=monkeypatch,
    )
    with TestClient(app) as client:
        response = client.post(
            "/runners/judge-routing/handle",
            json={"request_text": "anything"},
        )
    assert response.status_code == 503


# ── /handle 503 on unknown-agent ranking ──────────────────────


def test_handle_returns_503_when_judge_names_unknown_agent(
    e2e_runner_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The judge ranks a non-existent agent first — runner surfaces 503."""
    scripted = [
        _llm_response(
            _grounded_ranking_json(
                confidences=[0.9],
                complexities=[2],
                names=["ghost-specialist"],
                capabilities=[["nothing"]],
            )
        )
    ]
    specialists, shared_client = _build_e2e_specialists(scripted)
    app, _ = _make_runner_app(
        e2e_runner_module,
        specialists=specialists,
        shared_client=shared_client,
        monkeypatch=monkeypatch,
    )
    with TestClient(app) as client:
        response = client.post(
            "/runners/judge-routing/handle",
            json={"request_text": "anything"},
        )
    assert response.status_code == 503
    assert "unknown_agent" in response.json()["detail"]


# ── /runners index entry ──────────────────────────────────────


def test_runners_index_lists_judge_routing_registration() -> None:
    """``REGISTRATIONS`` includes the judge-routing entry from runners.py."""
    import importlib.util
    from types import ModuleType

    def _load(name: str, path: Path) -> ModuleType:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    for name in ("llm_provider", "runners", "app"):
        sys.modules.pop(name, None)

    _load("llm_provider", _FULL_STACK_DIR / "llm_provider.py")
    runners_mod = _load("runners", _FULL_STACK_DIR / "runners.py")

    slugs = [r.slug for r in runners_mod.REGISTRATIONS]
    assert "judge-routing" in slugs
    judge = next(r for r in runners_mod.REGISTRATIONS if r.slug == "judge-routing")
    assert judge.title == "Judge-routed request handling"
    assert "specialists" in judge.description.lower() or "judge" in judge.description.lower()


# ── _build_specialists builds tooled BiddableAgents ──────────


def test_build_specialists_constructs_four_tooled_biddable_agents(
    e2e_runner_module: Any,
) -> None:
    client = MockLLMClient([])
    context = MagicMock()
    context.build_client = lambda: client

    specialists, returned_client = e2e_runner_module._build_specialists(context)

    assert returned_client is client
    assert len(specialists) == 4
    assert [s.agent.name for s in specialists] == [
        "billing-specialist",
        "technical-specialist",
        "account-specialist",
        "policy-specialist",
    ]
    # Each spec carried a tool bundle into the agent constructor — the
    # roster invariant is pinned by ``test_runner_specs_carry_tools``;
    # here we only verify the BiddableAgents were built on the shared
    # client.
    assert all(s.agent.name for s in specialists)


# ── _GroundedJudgeRouter rejects out-of-band complexity ──────


@pytest.mark.asyncio
async def test_grounded_judge_router_rejects_out_of_band_complexity(
    e2e_runner_module: Any,
) -> None:
    """A ``complexity`` outside [1, 5] raises ``ValueError`` from ``run``."""
    bad_response = _llm_response(
        _grounded_ranking_json(
            confidences=[0.9, 0.4, 0.3, 0.2],
            complexities=[7, 1, 1, 1],
        )
    )
    shared_client = MockLLMClient([bad_response])
    specialists, _ = _build_e2e_specialists()
    # Replace agents' client with the bad-response one so the judge
    # call uses the scripted bad-complexity response.
    placeholder_emitter = InMemoryEmitter(trace_id="judge-routing-bad")
    base_rates = {s.name: 0.01 for s in e2e_runner_module.SPECIALIST_SPECS}

    router = e2e_runner_module._GroundedJudgeRouter(
        participants=specialists,
        judge_llm=shared_client,
        emitter=placeholder_emitter,
        base_rates=base_rates,
    )
    with pytest.raises(ValueError, match="complexity"):
        await router.run("anything")


# ── _GroundedJudgeRouter branches: empty / threshold / agent error ──


@pytest.mark.asyncio
async def test_grounded_judge_router_empty_ranking(
    e2e_runner_module: Any,
) -> None:
    """An empty judge ranking sets ``judge_error='empty_ranking'``."""
    empty = _llm_response('{"ranking": []}')
    shared_client = MockLLMClient([empty])
    specialists, _ = _build_e2e_specialists()
    placeholder_emitter = InMemoryEmitter(trace_id="judge-empty")
    router = e2e_runner_module._GroundedJudgeRouter(
        participants=specialists,
        judge_llm=shared_client,
        emitter=placeholder_emitter,
        base_rates={s.name: 0.01 for s in e2e_runner_module.SPECIALIST_SPECS},
    )
    result = await router.run("anything")
    assert result.allocated is False
    assert result.winner is None
    assert result.judge_error == "empty_ranking"


@pytest.mark.asyncio
async def test_grounded_judge_router_below_threshold(
    e2e_runner_module: Any,
) -> None:
    """A top-confidence below ``min_confidence_threshold`` rejects allocation."""
    low = _llm_response(
        _grounded_ranking_json(
            confidences=[0.1, 0.05, 0.05, 0.05],
            complexities=[1, 1, 1, 1],
        )
    )
    shared_client = MockLLMClient([low])
    specialists, _ = _build_e2e_specialists()
    placeholder_emitter = InMemoryEmitter(trace_id="judge-low")
    router = e2e_runner_module._GroundedJudgeRouter(
        participants=specialists,
        judge_llm=shared_client,
        emitter=placeholder_emitter,
        base_rates={s.name: 0.01 for s in e2e_runner_module.SPECIALIST_SPECS},
        min_confidence_threshold=0.5,
    )
    result = await router.run("anything")
    assert result.allocated is False
    assert result.winner is None


@pytest.mark.asyncio
async def test_grounded_judge_router_swallows_winning_agent_exception(
    e2e_runner_module: Any,
) -> None:
    """If the winning agent raises, ``execution_error`` is populated."""
    # Two scripted responses: the judge ranking, then an empty
    # MockLLMClient call which raises (the winning agent will reach for
    # the same shared client but it has no further responses).
    only_ranking = _llm_response(
        _grounded_ranking_json(
            confidences=[0.9, 0.4, 0.3, 0.2],
            complexities=[3, 2, 1, 1],
        )
    )
    shared_client = MockLLMClient([only_ranking])
    # Rebind agents to use ``shared_client`` directly so the agent
    # call exhausts (only the judge ranking is scripted) and raises.
    placeholder_emitter = InMemoryEmitter(trace_id="judge-agent-raise")
    new_specialists: list[BiddableAgent] = []
    for spec in e2e_runner_module.SPECIALIST_SPECS:
        agent = ReActAgent(
            name=spec.name,
            llm_client=shared_client,
            emitter=placeholder_emitter,
            system_prompt=spec.system_prompt,
            tools=[],
            max_iterations=1,
        )
        new_specialists.append(BiddableAgent(agent=agent, bid_generator=FixedBidGenerator(confidence=0.0)))

    router = e2e_runner_module._GroundedJudgeRouter(
        participants=new_specialists,
        judge_llm=shared_client,
        emitter=placeholder_emitter,
        base_rates={s.name: 0.01 for s in e2e_runner_module.SPECIALIST_SPECS},
    )
    result = await router.run("anything")
    # winner was selected but execution raised
    assert result.winner is not None
    assert result.execution_result is None
    assert result.execution_error is not None


# ── /handle without prior register raises RuntimeError ───────


@pytest.mark.asyncio
async def test_handle_request_without_register_raises_runtime_error(
    e2e_runner_module: Any,
) -> None:
    """Calling ``_handle_request`` before ``register`` is a programming error."""
    # ``e2e_runner_module`` fixture resets _executor / _llm_client to None.
    request = e2e_runner_module._HandleRequest(request_text="anything")
    with pytest.raises(RuntimeError, match="not registered"):
        await e2e_runner_module._handle_request(request)
