"""Comparative-judgment routing primitive.

A single LLM acts as a judge that ranks all candidate agents in one call,
counter-balancing the self-overclaim bias of independent self-rated bids.

The shape mirrors :mod:`nanitics.composition.multi_agent.bidding` so adopters
can swap ``Bidding`` → ``JudgeRouter`` at the call site without rebuilding
agents (``BiddableAgent`` is reused; the ``bid_generator`` field is unused
by ``JudgeRouter`` — that's fine, the type stays shared so adopters can flip
primitives without rebuilding agents).
"""

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel, ConfigDict

from nanitics.composition.multi_agent.bidding import (
    DEFAULT_CALIBRATED_JUDGE_PROMPT_TEMPLATE,
    BiddableAgent,
    _validate_judge_prompt_template,
)
from nanitics.core.agents.base import Agent
from nanitics.infrastructure.llm.instrumented import InstrumentedLLMClient
from nanitics.infrastructure.llm.protocol import LLMClient, Message
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import (
    JudgeAllocatedEvent,
    JudgeRankingEvent,
    JudgeRoutingCompleteEvent,
    JudgeRoutingStartEvent,
)
from nanitics.safety.cancellation import CancellationToken

# --- Models ---


class RankedCandidate(BaseModel):
    """A single ranked candidate produced by the judge.

    Same shape as :class:`~nanitics.composition.multi_agent.bidding.Bid` so
    callers swapping ``Bidding`` → ``JudgeRouter`` see a near-identical
    surface.
    """

    model_config = ConfigDict(frozen=True)

    agent_name: str
    confidence: float
    capabilities: list[str]
    estimated_cost: float | None = None
    reasoning: str


class JudgeRouterResult(BaseModel):
    """Outcome of a judge-routed allocation.

    Mirrors :class:`~nanitics.composition.multi_agent.bidding.BiddingResult`
    so adopters can swap the call site.
    """

    model_config = ConfigDict(frozen=True)

    winner: RankedCandidate | None
    ranking: list[RankedCandidate]
    execution_result: Any
    allocated: bool
    judge_error: str | None = None
    execution_error: str | None = None


# --- Schemas ---


class _RankedCandidateSchema(BaseModel):
    agent_name: str
    confidence: float
    capabilities: list[str]
    estimated_cost: float | None = None
    reasoning: str


class _JudgeRankingSchema(BaseModel):
    ranking: list[_RankedCandidateSchema]


# --- Controller ---


class JudgeRouter:
    """Centralised comparative-judgment router.

    A single LLM call produces a full ranking of candidate agents — the
    judge sees every candidate together and discriminates comparatively,
    avoiding the self-overclaim bias inherent to per-agent independent
    bidding.

    The judge LLM is wrapped with
    :class:`~nanitics.infrastructure.llm.instrumented.InstrumentedLLMClient`
    using ``label="judge"`` so judge-phase spend rolls into the run's
    summary alongside the winning agent's calls (parallel to
    :class:`~nanitics.composition.multi_agent.bidding.LLMBidGenerator`'s
    ``label="bid"``).

    Args:
        participants: Agents (paired with bid generators for type uniformity
            with :class:`Bidding`; the bid generator is unused here).
        judge_llm: LLM client used for the single comparative-ranking call.
        emitter: Run-scoped event emitter.
        prompt_template: Optional custom judge prompt template. Defaults to
            :data:`DEFAULT_CALIBRATED_JUDGE_PROMPT_TEMPLATE`. Templates must
            include ``{participants}`` and ``{task}`` placeholders;
            otherwise :class:`ValueError` is raised at construction time.
        min_confidence_threshold: When set, winners below this threshold
            are rejected (``allocated=False``, ``rejection_reason="below_threshold"``).
        cancellation_token: Optional cancellation signal.
    """

    def __init__(
        self,
        *,
        participants: list[BiddableAgent],
        judge_llm: LLMClient,
        emitter: EventEmitter,
        prompt_template: str | None = None,
        min_confidence_threshold: float | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> None:
        resolved_template = DEFAULT_CALIBRATED_JUDGE_PROMPT_TEMPLATE if prompt_template is None else prompt_template
        _validate_judge_prompt_template(resolved_template)
        self._participants = participants
        self._judge_llm = judge_llm
        self._emitter = emitter
        self._prompt_template = resolved_template
        self._min_confidence_threshold = min_confidence_threshold
        self._cancellation_token = cancellation_token

    async def run(self, task: str) -> JudgeRouterResult:
        """Run the judge-routed allocation.

        One LLM call produces the full ranking; the top-ranked candidate
        executes the task subject to the optional confidence threshold.
        """
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
            output_schema=_JudgeRankingSchema,
        )
        parsed = cast(_JudgeRankingSchema, result.parsed)

        ranking: list[RankedCandidate] = [
            RankedCandidate(
                agent_name=item.agent_name,
                confidence=max(0.0, min(1.0, item.confidence)),
                capabilities=item.capabilities,
                estimated_cost=item.estimated_cost,
                reasoning=item.reasoning,
            )
            for item in parsed.ranking
        ]

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
            winning_agent: Agent | None = None
            for p in self._participants:
                if p.agent.name == winner.agent_name:
                    winning_agent = p.agent
                    break
            # winner.agent_name is in known_names — invariant from the branch
            # above, so winning_agent is non-None.
            assert winning_agent is not None
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
