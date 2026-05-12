"""Auction-routed request handling runner.

Four specialist agents bid on every incoming request through the SDK's
:class:`~nanitics.Bidding` primitive; the winner's
:class:`~nanitics.ReActAgent` answers. Bids carry calibrated confidences
(via :data:`~nanitics.DEFAULT_CALIBRATED_BID_PROMPT_TEMPLATE`) and a
grounded cost (per-specialist ``base_rate`` × LLM-estimated complexity
1–5), and the auction selects the winner via
``HighestConfidence(tiebreaker=LowestCost())``. There is no HITL branch
on this runner — the bidding-based path always allocates to the
highest-confidence specialist and returns its answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from nanitics import (
    BiddableAgent,
    ReActAgent,
)
from nanitics.composition.multi_agent.bidding import BidGenerator
from nanitics.experimental import (
    DEFAULT_CALIBRATED_BID_PROMPT_TEMPLATE,
    Bid,
    Bidding,
    HighestConfidence,
    LowestCost,
)
from nanitics.infrastructure.llm.instrumented import InstrumentedLLMClient
from nanitics.infrastructure.llm.protocol import LLMClient, Message
from nanitics.infrastructure.observability.emitter import EventEmitter, InMemoryEmitter

if TYPE_CHECKING:
    from runners import ShellContext


# ── Module constants ──────────────────────────────────────────

RUNNER_SLUG = "auction-routing"
RUNNER_TITLE = "Auction-routed request handling"
RUNNER_DESCRIPTION = (
    "Four specialists bid on incoming requests with calibrated "
    "confidences and grounded per-call cost; the auction routes to the "
    "highest-confidence specialist, with the cheaper bid winning on "
    "strict ties."
)

_AGENT_MAX_ITERATIONS = 3

# Complexity is anchored 1–5 in the bid prompt and post-multiplied
# against each specialist's ``base_rate`` to ground ``estimated_cost``.
_MIN_COMPLEXITY = 1
_MAX_COMPLEXITY = 5


# ── Specialist roster ─────────────────────────────────────────


@dataclass(frozen=True)
class _SpecialistSpec:
    """Static configuration for one specialist in the auction.

    Attributes:
        name: Slug used as the agent name (kebab-case).
        system_prompt: ReActAgent system prompt for the winning-agent run.
        agent_description: In-scope summary surfaced into the bid prompt.
        out_of_scope: Single sentence describing what this specialist
            explicitly does NOT handle. Joined with ``agent_description``
            into the bid prompt body so the calibration anchors have
            something to bite on (a billing question is "out of scope"
            for the technical-specialist, etc.).
        base_rate: Per-call USD anchor for the grounded-cost multiplier.
            ``estimated_cost = base_rate * complexity`` where complexity
            is the LLM's 1–5 estimate.
    """

    name: str
    system_prompt: str
    agent_description: str
    out_of_scope: str
    base_rate: float


SPECIALIST_SPECS: list[_SpecialistSpec] = [
    _SpecialistSpec(
        name="billing-specialist",
        system_prompt=(
            "You are a billing specialist. Handle billing, invoicing, "
            "refunds, and subscription changes. Answer concisely in 2–4 "
            "sentences."
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
    ),
    _SpecialistSpec(
        name="technical-specialist",
        system_prompt=(
            "You are a technical support specialist. Handle product bugs, "
            "integration failures, API errors, and environment "
            "troubleshooting. Answer in 2–4 sentences."
        ),
        agent_description=(
            "Technical support for product bugs, integrations, API errors, and deployment troubleshooting."
        ),
        out_of_scope=(
            "Does not handle billing, invoicing, refunds, account "
            "access/password resets, or policy/compliance questions."
        ),
        base_rate=0.03,
    ),
    _SpecialistSpec(
        name="account-specialist",
        system_prompt=(
            "You are an account specialist. Handle access, password resets, "
            "profile updates, and account lifecycle. Answer in 2–4 sentences."
        ),
        agent_description=(
            "Account management for access, authentication issues, password resets, and profile changes."
        ),
        out_of_scope=(
            "Does not handle billing/invoicing, product bugs or integration failures, or policy/compliance questions."
        ),
        base_rate=0.015,
    ),
    _SpecialistSpec(
        name="policy-specialist",
        system_prompt=(
            "You are a policy specialist. Handle terms-of-service questions, "
            "acceptable-use, data-handling, and compliance matters. Answer "
            "in 2–4 sentences."
        ),
        agent_description=(
            "Policy specialist for terms-of-service, acceptable-use, data handling, and compliance guidance."
        ),
        out_of_scope=("Does not handle billing, account access/password resets, or product/technical troubleshooting."),
        base_rate=0.025,
    ),
]


# ── Grounded-cost bid generator ───────────────────────────────


class _GroundedCostBidSchema(BaseModel):
    """Schema used by :class:`_GroundedCostBidGenerator`.

    Extends the SDK's bid schema with ``complexity`` so the runner can
    multiply by ``base_rate`` and replace the LLM-hallucinated round-
    number costs that produced the original $50/$100 figures.
    """

    confidence: float
    capabilities: list[str]
    complexity: int
    reasoning: str


class _GroundedCostBidGenerator:
    """LLMBidGenerator analogue that grounds ``estimated_cost``.

    Mirrors :class:`~nanitics.LLMBidGenerator`'s tracing contract — one
    ``LLMRequestEvent`` + one ``LLMResponseEvent`` per call, both
    labelled ``"bid"`` — but extends the structured-output schema with a
    1–5 ``complexity`` field and post-multiplies ``base_rate`` to
    produce a deterministic ``estimated_cost``.

    Implemented in the runner rather than the SDK because cost grounding
    is a runner-side decision (different deployments anchor cost
    differently); keeping the SDK's :class:`LLMBidGenerator` simple
    follows the plan's "subclass per-runner" guidance.

    Args:
        llm_client: LLM client used to generate the structured bid.
        agent_description: In-scope description, joined with
            ``out_of_scope`` into the calibrated prompt body.
        out_of_scope: Single sentence describing what the specialist
            explicitly does not handle.
        base_rate: Per-call USD anchor multiplied by ``complexity`` to
            ground ``estimated_cost``.
        prompt_template: Calibrated prompt template (defaults to
            :data:`~nanitics.DEFAULT_CALIBRATED_BID_PROMPT_TEMPLATE`),
            extended with the explicit complexity instruction below.
    """

    _COMPLEXITY_INSTRUCTION = (
        "\n\nAlso estimate task complexity on a 1–5 integer scale "
        "(1 = trivial lookup; 3 = typical request; 5 = deep "
        "investigation). Return it as the ``complexity`` field."
    )

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        agent_description: str,
        out_of_scope: str,
        base_rate: float,
        prompt_template: str = DEFAULT_CALIBRATED_BID_PROMPT_TEMPLATE,
    ) -> None:
        self._llm_client = llm_client
        self._composed_description = f"{agent_description} {out_of_scope}"
        self._base_rate = base_rate
        self._prompt_template = prompt_template + self._COMPLEXITY_INSTRUCTION

    async def generate(
        self,
        agent_name: str,
        task: str,
        *,
        emitter: EventEmitter,
    ) -> Bid:
        instrumented = InstrumentedLLMClient(
            self._llm_client,
            emitter=emitter,
            label="bid",
        )
        prompt = self._prompt_template.format(
            agent_name=agent_name,
            agent_description=self._composed_description,
            task=task,
        )
        result = await instrumented.generate(
            system_prompt="You are a bid evaluator.",
            messages=[Message(role="user", content=prompt)],
            output_schema=_GroundedCostBidSchema,
        )
        parsed = cast(_GroundedCostBidSchema, result.parsed)
        if parsed.complexity < _MIN_COMPLEXITY or parsed.complexity > _MAX_COMPLEXITY:
            # Surface failures — the calibrated prompt anchors complexity
            # at 1–5; an out-of-band value is a contract violation we
            # propagate rather than silently coerce.
            raise ValueError(f"complexity must be in [{_MIN_COMPLEXITY}, {_MAX_COMPLEXITY}]; got {parsed.complexity}")
        estimated_cost = self._base_rate * parsed.complexity
        return Bid(
            agent_name=agent_name,
            confidence=max(0.0, min(1.0, parsed.confidence)),
            capabilities=parsed.capabilities,
            estimated_cost=estimated_cost,
            reasoning=parsed.reasoning,
        )


# ── Request / response models ─────────────────────────────────


class _HandleRequest(BaseModel):
    """Request body for ``POST /runners/auction-routing/handle``."""

    model_config = ConfigDict(extra="forbid")

    request_text: str = Field(..., min_length=1, max_length=4000)


class _BidSummary(BaseModel):
    """Subset of a :class:`~nanitics.Bid` returned in the response envelope."""

    model_config = ConfigDict(frozen=True)

    agent_name: str
    confidence: float
    capabilities: list[str]
    estimated_cost: float | None
    reasoning: str


class _HandleResponse(BaseModel):
    """Response body for ``POST /runners/auction-routing/handle``.

    The ``outcome`` literal is narrowed to ``"specialist_answered"``
    because the runner no longer has a HITL branch — every successful
    request resolves through the auction.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    outcome: Literal["specialist_answered"]
    winner: str
    bids: list[_BidSummary]
    answer: str | None
    trace_url: str


# ── Shared runner state ───────────────────────────────────────

# All runner-scoped state lives in module globals. ``register()``
# populates these at lifespan startup. Tests construct a fresh module
# (via the shell-modules by-path loader) so state never bleeds across
# tests.
_specialists: list[BiddableAgent] = []
_executor: Any = None


# ── Test seams ────────────────────────────────────────────────


def _build_specialists(context: ShellContext) -> list[BiddableAgent]:
    """Construct the four specialists against a shared LLM client.

    Tests patch this to return :class:`BiddableAgent` instances paired
    with ``FixedBidGenerator`` + ``MockLLMClient``-backed agents so
    ``register()`` can run without a real provider.
    """
    client = context.build_client()
    specialists: list[BiddableAgent] = []
    # The per-specialist emitter is replaced via ``.bind()`` inside
    # ``Bidding.run``'s winning-agent execution. The placeholder keeps
    # ``ReActAgent.__init__`` satisfied without attaching to a real run.
    placeholder_emitter = InMemoryEmitter(trace_id="auction-routing-placeholder")
    for spec in SPECIALIST_SPECS:
        agent = ReActAgent(
            name=spec.name,
            llm_client=client,
            emitter=placeholder_emitter,
            system_prompt=spec.system_prompt,
            tools=[],
            max_iterations=_AGENT_MAX_ITERATIONS,
        )
        bid_generator: BidGenerator = _GroundedCostBidGenerator(
            llm_client=client,
            agent_description=spec.agent_description,
            out_of_scope=spec.out_of_scope,
            base_rate=spec.base_rate,
        )
        specialists.append(BiddableAgent(agent=agent, bid_generator=bid_generator))
    return specialists


# ── Helpers ───────────────────────────────────────────────────


def _bid_summaries(result_bids: list[Any]) -> list[_BidSummary]:
    """Convert SDK :class:`Bid` objects into the response envelope subset."""
    return [
        _BidSummary(
            agent_name=bid.agent_name,
            confidence=bid.confidence,
            capabilities=list(bid.capabilities),
            estimated_cost=bid.estimated_cost,
            reasoning=bid.reasoning,
        )
        for bid in result_bids
    ]


# ── Route handlers ────────────────────────────────────────────


async def _handle_request(request: _HandleRequest) -> _HandleResponse:
    """Drive the bidding auction for one incoming request.

    There is no HITL branch — the auction always allocates to the
    highest-confidence bid (with ``LowestCost`` breaking strict ties), so
    every successful response carries ``outcome == "specialist_answered"``
    and a non-null ``winner``.
    """
    request_text = request.request_text

    # Accumulator for the outer envelope; populated inside the factory.
    outcome: dict[str, Any] = {
        "winner": None,
        "bids": [],
        "answer": None,
    }

    async def _factory(emitter: EventEmitter, run_id: str) -> None:
        del run_id  # observatory-side; surfaced via ``run_id`` return below.
        bidding = Bidding(
            participants=_specialists,
            emitter=emitter,
            allocation_strategy=HighestConfidence(tiebreaker=LowestCost()),
            min_bid_threshold=None,
        )
        result = await bidding.run(request_text)

        outcome["bids"] = _bid_summaries(result.all_bids)

        # ``Bidding`` with ``min_bid_threshold=None`` always allocates
        # when at least one bid is present and the allocation strategy
        # returns a winner. ``HighestConfidence`` returns ``None`` only
        # on an empty bid list, which cannot happen here because the
        # specialist roster is non-empty by construction.
        assert result.allocated
        assert result.winning_bid is not None
        outcome["winner"] = result.winning_bid.agent_name
        # ``BiddingResult.execution_result`` is ``Any`` at the SDK
        # level, but our specialists are ``ReActAgent``s whose
        # ``AgentResult.output`` is ``str | None``. Passing through
        # directly preserves both states; the FastAPI JSON encoder
        # serialises ``None`` as ``null``.
        outcome["answer"] = result.execution_result

    run_id, _ = await _executor.execute(_factory, metadata={"runner": RUNNER_SLUG})

    return _HandleResponse(
        run_id=run_id,
        outcome="specialist_answered",
        winner=outcome["winner"],
        bids=outcome["bids"],
        answer=outcome["answer"],
        trace_url=f"/api/observatory/runs/{run_id}",
    )


# ── Registration ──────────────────────────────────────────────


def register(app: FastAPI, context: ShellContext) -> None:
    """Mount the auction-routing runner onto *app*.

    Builds the specialist roster and registers the single
    ``/runners/auction-routing/handle`` route. The runner has no env-var
    configuration — calibrated bidding always allocates.
    """
    global _specialists, _executor

    _specialists = _build_specialists(context)
    _executor = context.executor

    app.post("/runners/auction-routing/handle", response_model=_HandleResponse)(_handle_request)
