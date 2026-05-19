"""Judge-routed request handling runner.

Four tool-using specialists are routed by a single comparative-judgment
LLM call (the judge), and the winning specialist answers using the
in-memory billing/technical/account/policy fixtures from
:mod:`fixtures` and the tools from :mod:`tools`.

Cost is grounded the same way as the auction-routing runner: the judge
returns a 1–5 ``complexity`` per candidate, and ``estimated_cost`` is
post-multiplied with each spec's ``base_rate``. The judge call is
instrumented (``label="judge"``) so its tokens roll into the run
summary alongside the winning agent's calls.

There is no HITL branch on this runner; the judge always allocates to
the top-ranked candidate, and consumers can build a confidence gate
client-side from ``ranking[0].confidence`` in the response.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from nanitics.composition import (
    DEFAULT_CALIBRATED_JUDGE_PROMPT_TEMPLATE,
    BiddableAgent,
    JudgeRouter,
    JudgeRouterResult,
    RankedCandidate,
)
from nanitics.infrastructure.llm.instrumented import InstrumentedLLMClient
from nanitics.infrastructure.llm.protocol import LLMClient, Message
from nanitics.infrastructure.observability.emitter import EventEmitter, InMemoryEmitter
from nanitics.infrastructure.observability.events import (
    JudgeAllocatedEvent,
    JudgeRankingEvent,
    JudgeRoutingCompleteEvent,
    JudgeRoutingStartEvent,
)
from nanitics.specialized import FixedBidGenerator
from nanitics.strategies import (
    ReActAgent,
    Tool,
)

from .tools import account_tools, billing_tools, policy_tools, technical_tools

if TYPE_CHECKING:
    from runners import ShellContext


# ── Module constants ──────────────────────────────────────────

RUNNER_SLUG = "judge-routing"
RUNNER_TITLE = "Judge-routed request handling"
RUNNER_DESCRIPTION = (
    "Four tool-using specialists are routed by a single comparative-"
    "judgment LLM call; the winning specialist answers using the "
    "in-memory billing/technical/account/policy fixtures."
)

_AGENT_MAX_ITERATIONS = 4

# Complexity is anchored 1–5 in the judge prompt and post-multiplied
# against each specialist's ``base_rate`` to ground ``estimated_cost``.
_MIN_COMPLEXITY = 1
_MAX_COMPLEXITY = 5


# ── Specialist roster ─────────────────────────────────────────


@dataclass(frozen=True)
class _SpecialistSpec:
    """Static configuration for one specialist on the judge-routing runner.

    Mirrors the auction-routing spec shape (``out_of_scope`` +
    ``base_rate`` for the cost-grounding multiplier) but adds a ``tools``
    bundle since judge-routing's specialists actually use tools.

    Attributes:
        name: Slug used as the agent name (kebab-case).
        system_prompt: ReActAgent system prompt; mentions only this
            specialist's own tools — the judge, not the agent, decides
            routing.
        agent_description: In-scope summary surfaced into the judge prompt.
        out_of_scope: Single sentence describing what this specialist
            explicitly does NOT handle.
        base_rate: Per-call USD anchor multiplied by the judge's 1–5
            complexity estimate to ground ``estimated_cost``.
        tools: Tool bundle attached to the specialist's
            :class:`~nanitics.ReActAgent`. Provided as a default factory
            so the dataclass remains frozen + hashable while still
            building fresh tool instances per spec.
    """

    name: str
    system_prompt: str
    agent_description: str
    out_of_scope: str
    base_rate: float
    tools: list[Tool] = field(default_factory=list)


SPECIALIST_SPECS: list[_SpecialistSpec] = [
    _SpecialistSpec(
        name="billing-specialist",
        system_prompt=(
            "You are a billing specialist. Use ``lookup_invoice`` to "
            "fetch invoice details and ``issue_refund`` to record a "
            "refund. Answer concisely in 2–4 sentences citing concrete "
            "ids and amounts."
        ),
        agent_description=(
            "Billing specialist handling invoices, refunds, payment disputes, and subscription lifecycle."
        ),
        out_of_scope=(
            "Does not handle access or authentication issues, password "
            "resets, product bugs, integration failures, or "
            "policy/compliance questions."
        ),
        base_rate=0.02,
        tools=billing_tools(),
    ),
    _SpecialistSpec(
        name="technical-specialist",
        system_prompt=(
            "You are a technical support specialist. Use ``search_kb``, "
            "``check_service_status``, and ``escalate_bug`` to "
            "investigate. Answer in 2–4 sentences citing the article id "
            "or service status."
        ),
        agent_description=(
            "Technical support for product bugs, integrations, API errors, and deployment troubleshooting."
        ),
        out_of_scope=(
            "Does not handle billing, invoicing, refunds, account "
            "access/password resets, or policy/compliance questions."
        ),
        base_rate=0.03,
        tools=technical_tools(),
    ),
    _SpecialistSpec(
        name="account-specialist",
        system_prompt=(
            "You are an account specialist. Use ``lookup_account`` to "
            "find the account, then ``reset_password`` or "
            "``update_profile`` as appropriate. Answer in 2–4 sentences."
        ),
        agent_description=(
            "Account management for access, authentication issues, password resets, and profile changes."
        ),
        out_of_scope=(
            "Does not handle billing/invoicing, product bugs or integration failures, or policy/compliance questions."
        ),
        base_rate=0.015,
        tools=account_tools(),
    ),
    _SpecialistSpec(
        name="policy-specialist",
        system_prompt=(
            "You are a policy specialist. Use ``lookup_policy`` to find "
            "relevant clauses and ``cite_clause`` to quote one exactly. "
            "Answer in 2–4 sentences citing the policy id and section."
        ),
        agent_description=(
            "Policy specialist for terms-of-service, acceptable-use, data handling, and compliance guidance."
        ),
        out_of_scope=("Does not handle billing, account access/password resets, or product/technical troubleshooting."),
        base_rate=0.025,
        tools=policy_tools(),
    ),
]


# ── Grounded judge schema + router ────────────────────────────


class _GroundedRankedCandidateSchema(BaseModel):
    """Per-candidate schema returned by the judge.

    Extends the SDK's ranking entry with a 1–5 ``complexity`` integer so
    the runner can multiply by ``base_rate`` and replace the LLM's
    free-form ``estimated_cost`` with a grounded value.
    """

    agent_name: str
    confidence: float
    capabilities: list[str]
    complexity: int
    reasoning: str


class _GroundedJudgeRankingSchema(BaseModel):
    """Top-level judge schema: a list of grounded ranking entries."""

    ranking: list[_GroundedRankedCandidateSchema]


class _GroundedJudgeRouter(JudgeRouter):
    """Runner-local subclass that grounds ``estimated_cost`` per candidate.

    The SDK's :class:`~nanitics.JudgeRouter` is intentionally simple —
    its schema returns a free-form ``estimated_cost``. This runner adds
    a 1–5 ``complexity`` field per candidate (anchored in the prompt
    extension below) and post-multiplies each spec's ``base_rate`` to
    produce a deterministic ``estimated_cost``.

    Mirrors the auction-routing runner's ``_GroundedCostBidGenerator``
    pattern: cost grounding is a runner-side decision, kept out of the
    SDK so the SDK primitive stays minimal.

    The base ``run`` method is overridden because the SDK hardcodes its
    output schema; the override is otherwise behaviour-equivalent (one
    start, N ranking events, one allocated, one complete).
    """

    _COMPLEXITY_INSTRUCTION = (
        "\n\nFor every candidate, also estimate task complexity on a "
        "1–5 integer scale (1 = trivial lookup; 3 = typical request; "
        "5 = deep investigation). Return it as the ``complexity`` "
        "field on each ranking entry."
    )

    def __init__(
        self,
        *,
        participants: list[BiddableAgent],
        judge_llm: LLMClient,
        emitter: EventEmitter,
        base_rates: dict[str, float],
        prompt_template: str | None = None,
        min_confidence_threshold: float | None = None,
    ) -> None:
        resolved_template = DEFAULT_CALIBRATED_JUDGE_PROMPT_TEMPLATE if prompt_template is None else prompt_template
        super().__init__(
            participants=participants,
            judge_llm=judge_llm,
            emitter=emitter,
            prompt_template=resolved_template + self._COMPLEXITY_INSTRUCTION,
            min_confidence_threshold=min_confidence_threshold,
        )
        self._base_rates = base_rates

    async def run(self, task: str) -> JudgeRouterResult:
        self._emitter.emit(
            JudgeRoutingStartEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                task=task,
                participant_names=[p.agent.name for p in self._participants],
            )
        )

        participant_block = "\n".join(f"- {p.agent.name}" for p in self._participants)
        prompt = self._prompt_template.format(participants=participant_block, task=task)

        instrumented = InstrumentedLLMClient(self._judge_llm, emitter=self._emitter, label="judge")
        result = await instrumented.generate(
            system_prompt="You are a comparative routing judge.",
            messages=[Message(role="user", content=prompt)],
            output_schema=_GroundedJudgeRankingSchema,
        )
        parsed = cast(_GroundedJudgeRankingSchema, result.parsed)

        ranking: list[RankedCandidate] = []
        for item in parsed.ranking:
            if item.complexity < _MIN_COMPLEXITY or item.complexity > _MAX_COMPLEXITY:
                # Surface the contract violation rather than coerce —
                # the calibrated prompt anchors complexity at 1–5.
                raise ValueError(f"complexity must be in [{_MIN_COMPLEXITY}, {_MAX_COMPLEXITY}]; got {item.complexity}")
            base_rate = self._base_rates.get(item.agent_name)
            estimated_cost = base_rate * item.complexity if base_rate is not None else None
            ranking.append(
                RankedCandidate(
                    agent_name=item.agent_name,
                    confidence=max(0.0, min(1.0, item.confidence)),
                    capabilities=item.capabilities,
                    estimated_cost=estimated_cost,
                    reasoning=item.reasoning,
                )
            )

        for rank_index, candidate in enumerate(ranking):
            self._emitter.emit(
                JudgeRankingEvent(
                    trace_id=self._emitter.trace_id,
                    span_id=self._emitter.span_id,
                    parent_span_id=self._emitter.parent_span_id,
                    agent_name=candidate.agent_name,
                    rank=rank_index,
                    confidence=candidate.confidence,
                    reasoning=candidate.reasoning,
                    estimated_cost=candidate.estimated_cost,
                )
            )

        winner: RankedCandidate | None
        judge_error: str | None = None
        rejection_reason: str | None = None

        known_names = {p.agent.name for p in self._participants}

        if not ranking:
            winner = None
            judge_error = "empty_ranking"
            rejection_reason = "empty_ranking"
        else:
            top = ranking[0]
            if top.agent_name not in known_names:
                winner = None
                judge_error = f"unknown_agent: {top.agent_name}"
                rejection_reason = "unknown_agent"
            elif self._min_confidence_threshold is not None and top.confidence < self._min_confidence_threshold:
                winner = None
                rejection_reason = "below_threshold"
            else:
                winner = top

        self._emitter.emit(
            JudgeAllocatedEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                winner=winner.agent_name if winner else None,
                confidence=winner.confidence if winner else None,
                total_candidates=len(ranking),
                rejection_reason=rejection_reason,
            )
        )

        execution_result: Any = None
        execution_error: str | None = None
        allocated = winner is not None

        if winner is not None:
            winning_agent = next(p.agent for p in self._participants if p.agent.name == winner.agent_name)
            try:
                agent_result = await winning_agent.bind(self._emitter).run(task)
                execution_result = agent_result.output
            except Exception as exc:
                execution_result = None
                execution_error = str(exc)

        self._emitter.emit(
            JudgeRoutingCompleteEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                winner=winner.agent_name if winner else None,
                total_participants=len(self._participants),
                allocated=allocated,
                judge_error=judge_error,
            )
        )

        return JudgeRouterResult(
            winner=winner,
            ranking=ranking,
            execution_result=execution_result,
            allocated=allocated,
            judge_error=judge_error,
            execution_error=execution_error,
        )


# ── Request / response models ─────────────────────────────────


class _HandleRequest(BaseModel):
    """Request body for ``POST /runners/judge-routing/handle``."""

    model_config = ConfigDict(extra="forbid")

    request_text: str = Field(..., min_length=1, max_length=4000)


class _RankingEntry(BaseModel):
    """One ranking entry surfaced in the response envelope."""

    model_config = ConfigDict(frozen=True)

    agent_name: str
    confidence: float
    capabilities: list[str]
    estimated_cost: float | None
    reasoning: str


class _HandleResponse(BaseModel):
    """Response body for ``POST /runners/judge-routing/handle``.

    There is no HITL branch on this runner — successful requests always
    return the winning specialist's answer. Consumers can build their
    own confidence gate from ``ranking[0].confidence``.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    winner: str
    ranking: list[_RankingEntry]
    answer: str | None
    trace_url: str


# ── Shared runner state ───────────────────────────────────────

# All runner-scoped state lives in module globals. ``register()``
# populates these at lifespan startup. Tests construct a fresh module
# (via the shell-modules by-path loader) so state never bleeds across
# tests.
_specialists: list[BiddableAgent] = []
_executor: Any = None
_llm_client: LLMClient | None = None


# ── Test seams ────────────────────────────────────────────────


def _build_specialists(context: ShellContext) -> tuple[list[BiddableAgent], LLMClient]:
    """Construct the four tool-using specialists against a shared LLM client.

    Each :class:`BiddableAgent` carries a placeholder ``FixedBidGenerator``
    — :class:`JudgeRouter` ignores the bid generator (the type is shared
    with :class:`Bidding` so adopters can flip primitives without
    rebuilding agents).

    Returns the specialist list and the shared LLM client; the latter
    backs the judge call so judge-phase spend rolls into the same run
    summary as the winning agent's calls.
    """
    client = context.build_client()
    placeholder_emitter = InMemoryEmitter(trace_id="judge-routing-placeholder")
    specialists: list[BiddableAgent] = []
    for spec in SPECIALIST_SPECS:
        agent = ReActAgent(
            name=spec.name,
            llm_client=client,
            emitter=placeholder_emitter,
            system_prompt=spec.system_prompt,
            tools=list(spec.tools),
            max_iterations=_AGENT_MAX_ITERATIONS,
        )
        # The ``FixedBidGenerator`` placeholder is unused by JudgeRouter —
        # type uniformity with Bidding lets adopters flip primitives.
        specialists.append(BiddableAgent(agent=agent, bid_generator=FixedBidGenerator(confidence=0.0)))
    return specialists, client


# ── Helpers ───────────────────────────────────────────────────


def _ranking_summaries(candidates: list[RankedCandidate]) -> list[_RankingEntry]:
    """Convert SDK :class:`RankedCandidate`s into the response envelope subset."""
    return [
        _RankingEntry(
            agent_name=c.agent_name,
            confidence=c.confidence,
            capabilities=list(c.capabilities),
            estimated_cost=c.estimated_cost,
            reasoning=c.reasoning,
        )
        for c in candidates
    ]


# ── Route handlers ────────────────────────────────────────────


async def _handle_request(request: _HandleRequest) -> _HandleResponse:
    """Drive one judge-routed request end-to-end.

    Builds a fresh :class:`_GroundedJudgeRouter` per call (the participants
    and base-rates come from module state populated in ``register``) and
    runs it inside the shared :class:`TracedExecutor` so the trace
    persists in the Observatory. There is no HITL branch — the judge
    always allocates.

    Failure paths:

    - 400 if ``request_text`` is empty (Pydantic's ``min_length=1``
      surfaces as a 422 by default; the explicit guard here surfaces
      runner-specific 400 semantics for empty bodies).
    - 503 if the judge LLM raises (the runner does not swallow it; the
      executor's traced ``_factory`` re-raises and the route handler
      converts it to ``HTTPException``).
    """
    if _executor is None:
        # ``register()`` has not been called — fail loudly.
        raise RuntimeError("judge-routing runner not registered")

    base_rates = {spec.name: spec.base_rate for spec in SPECIALIST_SPECS}

    outcome: dict[str, Any] = {
        "winner": None,
        "ranking": [],
        "answer": None,
        "judge_error": None,
    }

    async def _factory(emitter: EventEmitter, run_id: str) -> None:
        del run_id
        assert _llm_client is not None
        router = _GroundedJudgeRouter(
            participants=_specialists,
            judge_llm=_llm_client,
            emitter=emitter,
            base_rates=base_rates,
            min_confidence_threshold=None,
        )
        result = await router.run(request.request_text)
        outcome["ranking"] = _ranking_summaries(result.ranking)
        outcome["winner"] = result.winner.agent_name if result.winner else None
        outcome["answer"] = result.execution_result
        outcome["judge_error"] = result.judge_error

    try:
        run_id, _ = await _executor.execute(_factory, metadata={"runner": RUNNER_SLUG})
    except Exception as exc:
        # The judge LLM (or any other component) raised inside the
        # traced factory. Surface as 503 — the run is broken, not the
        # client request.
        raise HTTPException(status_code=503, detail=f"judge_llm_error: {exc}") from exc

    if outcome["winner"] is None:
        # Empty ranking or unknown agent — judge failed structurally.
        raise HTTPException(status_code=503, detail=f"judge_routing_failed: {outcome['judge_error']}")

    return _HandleResponse(
        run_id=run_id,
        winner=outcome["winner"],
        ranking=outcome["ranking"],
        answer=outcome["answer"],
        trace_url=f"/api/observatory/runs/{run_id}",
    )


# ── Registration ──────────────────────────────────────────────


def register(app: FastAPI, context: ShellContext) -> None:
    """Mount the judge-routing runner onto *app*.

    Builds the specialist roster, captures the shared LLM client for
    the judge call, and registers the single
    ``/runners/judge-routing/handle`` route.
    """
    global _specialists, _executor, _llm_client

    specialists, client = _build_specialists(context)
    _specialists = specialists
    _llm_client = client
    _executor = context.executor

    app.post("/runners/judge-routing/handle", response_model=_HandleResponse)(_handle_request)
