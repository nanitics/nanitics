"""Deterministic unit tests for the auction-routed request handling runner.

No Docker, no real LLM — ``MockLLMClient`` scripts every provider call
and ``InMemoryPersistentTraceStore`` backs the shared ``TracedExecutor``
so the Observatory trace shape is assertable.

``docker/full-stack/`` is not a Python package; the runtime image lays
its files out flat on ``/srv``. Tests add that directory to ``sys.path``
so ``auction_routing.runner`` imports as a package submodule — mirroring
the image layout exactly, so coverage counts the lines that actually run
in production.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from nanitics.composition import BiddableAgent
from nanitics.composition.multi_agent.bidding import BidGenerator
from nanitics.infrastructure import (
    LLMResponse,
    MockLLMClient,
)
from nanitics.infrastructure.observability.emitter import EventEmitter, InMemoryEmitter
from nanitics.specialized import (
    Bid,
    FixedBidGenerator,
)
from nanitics.strategies import ReActAgent
from nanitics.tracing import (
    InMemoryPersistentTraceStore,
    TracedExecutor,
    Usage,
)

# ---------------------------------------------------------------------------
# Path setup — make ``auction_routing`` importable as a package.
# ---------------------------------------------------------------------------

_FULL_STACK_DIR = Path(__file__).resolve().parent.parent / "docker" / "full-stack"
if str(_FULL_STACK_DIR) not in sys.path:
    sys.path.insert(0, str(_FULL_STACK_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _llm_response(content: str) -> LLMResponse:
    """Build a minimal assistant-final :class:`LLMResponse`."""
    return LLMResponse(
        content=content,
        tool_calls=[],
        usage=Usage(input_tokens=5, output_tokens=5),
        model="mock",
        stop_reason="end_turn",
    )


def _build_scripted_specialists(
    confidences: list[float],
    estimated_costs: list[float | None] | None = None,
) -> list[BiddableAgent]:
    """Four specialists paired with ``FixedBidGenerator``s.

    Each specialist's ``ReActAgent`` is wired to a fresh ``MockLLMClient``
    scripted with a single assistant message naming the specialist — the
    bidding auction only drives the winner's agent, so three of the four
    ``MockLLMClient``s never return their scripted response. ``FixedBidGenerator``
    bypasses the runner's ``_GroundedCostBidGenerator`` so tests can pin
    exact confidence/cost combinations.
    """
    import auction_routing.runner as rar

    if estimated_costs is None:
        estimated_costs = [None, None, None, None]

    placeholder_emitter = InMemoryEmitter(trace_id="auction-routing-test")
    specialists: list[BiddableAgent] = []
    for spec, confidence, cost in zip(rar.SPECIALIST_SPECS, confidences, estimated_costs, strict=True):
        client = MockLLMClient([_llm_response(f"Answer from {spec.name}")])
        specialists.append(
            BiddableAgent(
                agent=ReActAgent(
                    name=spec.name,
                    llm_client=client,
                    emitter=placeholder_emitter,
                    system_prompt="test prompt",
                    tools=[],
                    max_iterations=1,
                ),
                bid_generator=FixedBidGenerator(confidence=confidence, estimated_cost=cost),
            )
        )
    return specialists


def _grounded_bid_json(confidence: float, complexity: int, reasoning: str = "scripted bid") -> str:
    """Build a ``_GroundedCostBidSchema``-shaped JSON blob."""
    return (
        f'{{"confidence": {confidence}, "capabilities": ["billing", "routing"], '
        f'"complexity": {complexity}, "reasoning": "{reasoning}"}}'
    )


def _build_grounded_specialists() -> tuple[list[BiddableAgent], MockLLMClient]:
    """Four specialists wired to the runner's ``_GroundedCostBidGenerator``.

    The shared ``MockLLMClient`` is scripted with five responses — four
    grounded-bid JSON blobs (one per participant) and one assistant-final
    answer for the winning ``ReActAgent``.
    """
    import auction_routing.runner as rar

    shared_client = MockLLMClient(
        [
            LLMResponse(
                content=_grounded_bid_json(0.9, 3, "uniquely positioned"),
                tool_calls=[],
                usage=Usage(input_tokens=12, output_tokens=8),
                model="mock",
                stop_reason="end_turn",
            ),
            LLMResponse(
                content=_grounded_bid_json(0.4, 2, "adjacent expertise"),
                tool_calls=[],
                usage=Usage(input_tokens=11, output_tokens=7),
                model="mock",
                stop_reason="end_turn",
            ),
            LLMResponse(
                content=_grounded_bid_json(0.4, 1, "adjacent"),
                tool_calls=[],
                usage=Usage(input_tokens=11, output_tokens=7),
                model="mock",
                stop_reason="end_turn",
            ),
            LLMResponse(
                content=_grounded_bid_json(0.0, 1, "out of scope"),
                tool_calls=[],
                usage=Usage(input_tokens=11, output_tokens=7),
                model="mock",
                stop_reason="end_turn",
            ),
            _llm_response("Answer from the winner"),
        ]
    )

    placeholder_emitter = InMemoryEmitter(trace_id="auction-routing-grounded-test")
    specialists: list[BiddableAgent] = [
        BiddableAgent(
            agent=ReActAgent(
                name=spec.name,
                llm_client=shared_client,
                emitter=placeholder_emitter,
                system_prompt="test prompt",
                tools=[],
                max_iterations=1,
            ),
            bid_generator=rar._GroundedCostBidGenerator(
                llm_client=shared_client,
                agent_description=spec.agent_description,
                out_of_scope=spec.out_of_scope,
                base_rate=spec.base_rate,
            ),
        )
        for spec in rar.SPECIALIST_SPECS
    ]
    return specialists, shared_client


@pytest.fixture
def runner_module() -> Iterator[Any]:
    """Import ``auction_routing.runner`` with module-level state reset.

    The runner module stores its specialists / executor as module-level
    globals. Tests construct a fresh ``FastAPI`` app and call
    ``register()`` inside the test, so the globals get fresh values
    each run.
    """
    import auction_routing.runner as rar

    rar._specialists = []
    rar._executor = None

    return rar


def _make_runner_app(
    runner_module: Any,
    *,
    specialists_factory: Any,
    trace_store: InMemoryPersistentTraceStore | None = None,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FastAPI, InMemoryPersistentTraceStore]:
    """Build a fresh FastAPI app with the runner mounted.

    Injects the specialists via the module's test seam so ``register()``
    never reaches for a real LLM.
    """
    store = trace_store if trace_store is not None else InMemoryPersistentTraceStore()
    executor = TracedExecutor(store)
    pool = MagicMock(name="asyncpg-pool-stub")

    monkeypatch.setattr(runner_module, "_build_specialists", specialists_factory)

    from runners import ShellContext

    context = ShellContext(
        executor=executor,
        trace_store=store,
        pool=pool,
        build_client=lambda: MockLLMClient([]),
    )

    app = FastAPI()
    runner_module.register(app, context)
    return app, store


# ---------------------------------------------------------------------------
# Scenario 1 — happy path: highest-confidence bid wins.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_specialist_answered_happy_path(
    runner_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 0.9 bid wins outright over a 0.5 / 0.4 / 0.3 field.

    Asserts the response ``run_id`` is the Observatory run_id (a run
    record exists in the trace store under that id) and that
    ``trace_url`` tails match the envelope's ``run_id``.
    """
    app, store = _make_runner_app(
        runner_module,
        specialists_factory=lambda ctx: _build_scripted_specialists([0.9, 0.5, 0.4, 0.3]),
        monkeypatch=monkeypatch,
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/runners/auction-routing/handle",
            json={"request_text": "I need help with a billing question."},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "specialist_answered"
    assert body["winner"] == "billing-specialist"
    assert body["answer"] == "Answer from billing-specialist"
    assert len(body["bids"]) == 4
    assert "hitl_request_id" not in body  # HITL branch removed
    assert body["trace_url"] == f"/api/observatory/runs/{body['run_id']}"
    run = await store.get_run(body["run_id"])
    assert run is not None
    assert run.id == body["run_id"]


# ---------------------------------------------------------------------------
# Scenario 2 — original failing prompt: 90/90 tie, ``LowestCost`` breaks it.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calibrated_tie_breaks_to_lowest_cost(
    runner_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two specialists tie at the top confidence; cheaper bid wins.

    Pins the regression: before the refit, two 90% bids resolved by
    "first-listed wins" — billing-specialist always took the call. With
    ``HighestConfidence(tiebreaker=LowestCost())`` and grounded cost,
    the cheaper specialist wins deterministically. Here account-specialist
    bids 0.9 at cost 0.015 and billing-specialist bids 0.9 at cost 0.06,
    so account-specialist takes the call.
    """
    app, _store = _make_runner_app(
        runner_module,
        specialists_factory=lambda ctx: _build_scripted_specialists(
            [0.9, 0.5, 0.9, 0.4],
            estimated_costs=[0.06, 0.06, 0.015, 0.05],
        ),
        monkeypatch=monkeypatch,
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/runners/auction-routing/handle",
            json={"request_text": "My invoice shows the wrong amount."},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "specialist_answered"
    assert body["winner"] == "account-specialist"


# ---------------------------------------------------------------------------
# Scenario 3 — bid-phase telemetry rolls up; cost is grounded by the
# runner's ``_GroundedCostBidGenerator`` (base_rate × complexity).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_specialist_answered_summary_rolls_up_grounded_bid_calls(
    runner_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Summary aggregates four bid LLM calls; each ``estimated_cost`` is grounded.

    Uses the runner's production ``_GroundedCostBidGenerator``. The bid
    phase emits ``llm.response`` events via
    ``InstrumentedLLMClient(..., label="bid")``, and those events roll
    up into the run's summary under the same ``run_id`` the ``/handle``
    response carries. Each ``BidReceivedEvent`` carries
    ``estimated_cost == base_rate * complexity`` for its specialist.
    """
    trace_store = InMemoryPersistentTraceStore()
    specialists, _client = _build_grounded_specialists()
    app, store = _make_runner_app(
        runner_module,
        specialists_factory=lambda ctx: specialists,
        trace_store=trace_store,
        monkeypatch=monkeypatch,
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/runners/auction-routing/handle",
            json={"request_text": "I need help with a billing question."},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "specialist_answered"
    run_id = body["run_id"]

    # Five LLM calls total: four bids + one winner answer.
    summary = await store.get_summary(run_id)
    assert summary.llm_calls >= 5

    events = await _events_for_run(store, run_id)
    bid_response_events = [e for e in events if e.event_type == "llm.response" and e.payload.get("label") == "bid"]
    assert len(bid_response_events) == 4
    assert summary.total_input_tokens > 0
    assert summary.total_output_tokens > 0

    # Grounded-cost arithmetic: every BidReceivedEvent's estimated_cost
    # equals base_rate * complexity for the matching spec, in the
    # scripted order from ``_build_grounded_specialists``.
    import auction_routing.runner as rar

    expected_costs = {
        rar.SPECIALIST_SPECS[0].name: rar.SPECIALIST_SPECS[0].base_rate * 3,
        rar.SPECIALIST_SPECS[1].name: rar.SPECIALIST_SPECS[1].base_rate * 2,
        rar.SPECIALIST_SPECS[2].name: rar.SPECIALIST_SPECS[2].base_rate * 1,
        rar.SPECIALIST_SPECS[3].name: rar.SPECIALIST_SPECS[3].base_rate * 1,
    }
    bid_events = [e for e in events if e.event_type == "multi_agent.bidding.bid"]
    assert len(bid_events) == 4
    for e in bid_events:
        assert e.payload["estimated_cost"] == pytest.approx(expected_costs[e.payload["agent_name"]])

    # The response envelope's bids carry the same grounded costs.
    by_name = {b["agent_name"]: b for b in body["bids"]}
    for name, expected in expected_costs.items():
        assert by_name[name]["estimated_cost"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Scenario 4 — invalid request body → 422 (Pydantic min_length=1).
# ---------------------------------------------------------------------------


def test_handle_empty_request_text_returns_422(
    runner_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pydantic's ``min_length=1`` guard surfaces as a 422 unprocessable-entity."""
    app, _ = _make_runner_app(
        runner_module,
        specialists_factory=lambda ctx: _build_scripted_specialists([0.9, 0.9, 0.9, 0.9]),
        monkeypatch=monkeypatch,
    )

    with TestClient(app) as client:
        response = client.post("/runners/auction-routing/handle", json={"request_text": ""})

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Scenario 5 — _build_specialists default path constructs the four agents.
# ---------------------------------------------------------------------------


def test_build_specialists_constructs_four_biddable_agents(
    runner_module: Any,
) -> None:
    """The default path builds four BiddableAgents bound to the shared client.

    Each specialist's bid generator is the runner's
    ``_GroundedCostBidGenerator`` (the production wiring), and the
    agents are listed in the documented roster order.
    """
    client = MockLLMClient([])
    context = MagicMock()
    context.build_client = lambda: client

    specialists = runner_module._build_specialists(context)

    assert len(specialists) == 4
    assert [s.agent.name for s in specialists] == [
        "billing-specialist",
        "technical-specialist",
        "account-specialist",
        "policy-specialist",
    ]
    for s in specialists:
        assert isinstance(s.bid_generator, runner_module._GroundedCostBidGenerator)


# ---------------------------------------------------------------------------
# Scenario 6 — trace shape: bidding events on the specialist-answered path.
# ---------------------------------------------------------------------------


async def _events_for_run(store: InMemoryPersistentTraceStore, run_id: str) -> list[Any]:
    """Read every persisted event for *run_id*."""
    events: list[Any] = []
    after: int | None = None
    while True:
        batch = await store.query_events(run_id, after_id=after, limit=500)
        if not batch:
            break
        events.extend(batch)
        after = batch[-1].id
    return events


@pytest.mark.asyncio
async def test_trace_envelope_specialist_path(
    runner_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bidding-event envelope: 1 start, 4 bids, 1 allocated, 1 complete."""
    store = InMemoryPersistentTraceStore()
    app, _ = _make_runner_app(
        runner_module,
        specialists_factory=lambda ctx: _build_scripted_specialists([0.9, 0.4, 0.4, 0.4]),
        trace_store=store,
        monkeypatch=monkeypatch,
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/runners/auction-routing/handle",
            json={"request_text": "billing question"},
        )
    assert response.status_code == 200
    trace_url = response.json()["trace_url"]
    run_id = trace_url.rsplit("/", 1)[-1]

    events = await _events_for_run(store, run_id)
    event_types = [e.event_type for e in events]

    assert event_types.count("multi_agent.bidding.start") == 1
    assert event_types.count("multi_agent.bidding.bid") == 4
    assert event_types.count("multi_agent.bidding.allocated") == 1
    assert event_types.count("multi_agent.bidding.complete") == 1
    allocated = next(e for e in events if e.event_type == "multi_agent.bidding.allocated")
    assert allocated.payload["winner"] == "billing-specialist"
    assert allocated.payload["rejection_reason"] is None
    complete = next(e for e in events if e.event_type == "multi_agent.bidding.complete")
    assert complete.payload["allocated"] is True


# ---------------------------------------------------------------------------
# Scenario 7 — /runners index entry.
# ---------------------------------------------------------------------------


def test_runners_index_lists_auction_routing_registration() -> None:
    """``GET /runners`` returns the auction-routing registration shape.

    Uses the live ``REGISTRATIONS`` list (built from module imports in
    ``runners.py``), not a monkeypatched copy — so this test fails
    loudly if the append-to-REGISTRATIONS regresses.
    """
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
    assert "auction-routing" in slugs

    auction = next(r for r in runners_mod.REGISTRATIONS if r.slug == "auction-routing")
    assert auction.title == "Auction-routed request handling"
    assert "specialists" in auction.description.lower() or "bid" in auction.description.lower()


# ---------------------------------------------------------------------------
# Scenario 8 — winner's agent raises → ``answer`` is null (Bidding swallows
# winner-execution errors and surfaces ``execution_result=None``).
# ---------------------------------------------------------------------------


def test_specialist_answered_none_answer_when_agent_raises(
    runner_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Winning bid fires; the agent call raises; ``answer`` is ``null``.

    ``Bidding`` swallows exceptions from the winning agent and sets
    ``execution_result=None``. The runner must surface that as
    ``answer: null`` in the envelope rather than synthesising a string.
    """

    def _specialists_with_empty_client(_ctx: Any) -> list[BiddableAgent]:
        import auction_routing.runner as rar

        placeholder_emitter = InMemoryEmitter(trace_id="auction-routing-test")
        specialists: list[BiddableAgent] = []
        for spec, confidence in zip(rar.SPECIALIST_SPECS, [0.9, 0.5, 0.4, 0.3], strict=True):
            # Empty MockLLMClient — any LLM call raises ``ValueError``.
            client = MockLLMClient([])
            specialists.append(
                BiddableAgent(
                    agent=ReActAgent(
                        name=spec.name,
                        llm_client=client,
                        emitter=placeholder_emitter,
                        system_prompt="test prompt",
                        tools=[],
                        max_iterations=1,
                    ),
                    bid_generator=FixedBidGenerator(confidence=confidence),
                )
            )
        return specialists

    app, _ = _make_runner_app(
        runner_module,
        specialists_factory=_specialists_with_empty_client,
        monkeypatch=monkeypatch,
    )

    with TestClient(app) as client:
        response = client.post(
            "/runners/auction-routing/handle",
            json={"request_text": "question"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "specialist_answered"
    assert body["winner"] == "billing-specialist"
    assert body["answer"] is None


# ---------------------------------------------------------------------------
# Scenario 9 — _GroundedCostBidGenerator: complexity outside 1–5 raises.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grounded_bid_generator_rejects_out_of_band_complexity(
    runner_module: Any,
) -> None:
    """A ``complexity`` value outside [1, 5] raises ``ValueError``.

    Surfaces the contract violation rather than silently coercing — the
    calibrated prompt anchors complexity at 1–5; anything else means the
    LLM ignored the schema.
    """
    bad_response = LLMResponse(
        content='{"confidence": 0.9, "capabilities": ["x"], "complexity": 9, "reasoning": "y"}',
        tool_calls=[],
        usage=Usage(input_tokens=5, output_tokens=5),
        model="mock",
        stop_reason="end_turn",
    )
    client = MockLLMClient([bad_response])

    spec = runner_module.SPECIALIST_SPECS[0]
    generator = runner_module._GroundedCostBidGenerator(
        llm_client=client,
        agent_description=spec.agent_description,
        out_of_scope=spec.out_of_scope,
        base_rate=spec.base_rate,
    )

    emitter = InMemoryEmitter(trace_id="grounded-bid-test")
    with pytest.raises(ValueError, match="complexity"):
        await generator.generate(spec.name, "any task", emitter=emitter)


@pytest.mark.asyncio
async def test_grounded_bid_generator_clamps_confidence(
    runner_module: Any,
) -> None:
    """``confidence`` is clamped into ``[0.0, 1.0]`` (matches LLMBidGenerator)."""
    overshoot = LLMResponse(
        content='{"confidence": 1.7, "capabilities": ["x"], "complexity": 2, "reasoning": "y"}',
        tool_calls=[],
        usage=Usage(input_tokens=5, output_tokens=5),
        model="mock",
        stop_reason="end_turn",
    )
    client = MockLLMClient([overshoot])

    spec = runner_module.SPECIALIST_SPECS[0]
    generator = runner_module._GroundedCostBidGenerator(
        llm_client=client,
        agent_description=spec.agent_description,
        out_of_scope=spec.out_of_scope,
        base_rate=spec.base_rate,
    )

    emitter = InMemoryEmitter(trace_id="grounded-bid-clamp-test")
    bid = await generator.generate(spec.name, "any task", emitter=emitter)
    assert bid.confidence == 1.0
    assert bid.estimated_cost == pytest.approx(spec.base_rate * 2)


# ---------------------------------------------------------------------------
# Scenario 10 — auction env vars no longer exist anywhere in the codebase.
# ---------------------------------------------------------------------------


def test_auction_env_vars_are_removed_from_codebase() -> None:
    """The HITL-related env vars must not appear in any tracked source.

    Pins the plan's invariant: dropping the HITL branch means the
    config knobs go too. ``grep`` would catch a stray reference; this
    test is the in-process equivalent so CI flags regressions.
    """
    repo_root = Path(__file__).resolve().parent.parent
    for var in ("NANITICS_AUCTION_MIN_CONFIDENCE", "NANITICS_HITL_TIMEOUT_SECONDS"):
        for path in (
            repo_root / "docker" / "full-stack" / "docker-compose.yml",
            repo_root / "docker" / "full-stack" / ".env.example",
            repo_root / "docker" / "full-stack" / "auction_routing" / "runner.py",
            repo_root / "docker" / "full-stack" / "auction_routing" / "README.md",
            repo_root / "docker" / "full-stack" / "DEMO.md",
        ):
            assert var not in path.read_text(), f"{var} should be removed from {path}"


# ---------------------------------------------------------------------------
# Scenario 11 — _SpecialistSpec carries out_of_scope and base_rate, all four
# entries populate them.
# ---------------------------------------------------------------------------


def test_all_specs_carry_out_of_scope_and_base_rate(
    runner_module: Any,
) -> None:
    """Every spec must populate the calibration anchors and cost anchor.

    Pinned by test so a future spec addition cannot regress the
    grounded-cost or out-of-scope contract.
    """
    for spec in runner_module.SPECIALIST_SPECS:
        assert spec.out_of_scope, f"{spec.name} must declare out_of_scope"
        assert spec.base_rate > 0, f"{spec.name} must declare a positive base_rate"


# ---------------------------------------------------------------------------
# Scenario 12 — register() has no env-var inputs; failures from env are gone.
# ---------------------------------------------------------------------------


def test_register_takes_no_env_input(
    runner_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``register()`` no longer reads env vars; the previous knobs are gone.

    The plan removes ``NANITICS_AUCTION_MIN_CONFIDENCE`` and
    ``NANITICS_HITL_TIMEOUT_SECONDS`` entirely. ``register()`` is a pure
    function of ``app`` + ``context`` now.
    """
    monkeypatch.setattr(runner_module, "_build_specialists", lambda ctx: [])

    from runners import ShellContext

    store = InMemoryPersistentTraceStore()
    context = ShellContext(
        executor=TracedExecutor(store),
        trace_store=store,
        pool=MagicMock(),
        build_client=lambda: MockLLMClient([]),
    )

    app = FastAPI()
    runner_module.register(app, context)

    # The route is mounted; no startup hooks needed because no schema
    # work is required (HITL store is gone).
    assert any(getattr(r, "path", None) == "/runners/auction-routing/handle" for r in app.routes)
    assert app.router.on_startup == []


# ---------------------------------------------------------------------------
# Fixture teardown hook — keep runner module globals tidy across the suite.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_runner_module_globals() -> AsyncIterator[None]:
    """Clear the runner module's in-process state after every test."""
    yield
    import auction_routing.runner as rar

    rar._specialists = []
    rar._executor = None


# ---------------------------------------------------------------------------
# Protocol smoke: BidGenerator is satisfied by _GroundedCostBidGenerator.
# Also exercises a no-op return path so ``BidGenerator`` import isn't unused.
# ---------------------------------------------------------------------------


def test_grounded_bid_generator_satisfies_bid_generator_protocol(
    runner_module: Any,
) -> None:
    """``_GroundedCostBidGenerator`` satisfies the SDK's ``BidGenerator`` protocol."""
    spec = runner_module.SPECIALIST_SPECS[0]
    generator = runner_module._GroundedCostBidGenerator(
        llm_client=MockLLMClient([]),
        agent_description=spec.agent_description,
        out_of_scope=spec.out_of_scope,
        base_rate=spec.base_rate,
    )
    # Runtime-checkable Protocol membership check.
    assert isinstance(generator, BidGenerator)
    # Sanity: ``Bid`` and ``EventEmitter`` imports are reachable from
    # the test surface, mirroring the runner's public-API touchpoints.
    bid = Bid(
        agent_name="x",
        confidence=0.5,
        capabilities=[],
        estimated_cost=0.0,
        reasoning="probe",
    )
    assert bid.agent_name == "x"
    emitter: EventEmitter = InMemoryEmitter(trace_id="probe")
    assert emitter.trace_id == "probe"
