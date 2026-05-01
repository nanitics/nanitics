from __future__ import annotations

import asyncio
import string
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict

from nanitics.composition.multi_agent.broadcast import AgentFailure
from nanitics.core.agents.base import Agent
from nanitics.infrastructure.llm.instrumented import InstrumentedLLMClient
from nanitics.infrastructure.llm.protocol import LLMClient, Message
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import (
    BidAllocatedEvent,
    BiddingCompleteEvent,
    BiddingStartEvent,
    BidReceivedEvent,
)
from nanitics.safety.cancellation import CancellationToken

# --- Calibration anchors ---


DEFAULT_CALIBRATED_BID_PROMPT_TEMPLATE = """You are evaluating whether agent '{agent_name}' is suitable for a task.

Agent description: {agent_description}

Task: {task}

Calibration anchors — pick the confidence band that matches your role for THIS task:
- 0.9 = uniquely positioned: this task falls squarely inside the agent's stated \
expertise and no closer specialist is plausible.
- 0.7 = capable: the agent can handle the task, but a closer specialist may exist.
- 0.4 = adjacent: the agent has tangentially related expertise; another agent is \
probably better.
- 0.0 = out of scope: the task is outside the agent's described scope.

Self-overclaim is a known failure mode — anchor against the bands above rather than rating in isolation.

Rate your confidence (0.0-1.0), list relevant capabilities, estimate cost if possible, and explain your reasoning."""


_REQUIRED_BID_TEMPLATE_PLACEHOLDERS = ("agent_name", "agent_description", "task")


DEFAULT_CALIBRATED_JUDGE_PROMPT_TEMPLATE = """You are a routing judge. Compare the candidate agents below and rank \
them by suitability for the task — a single comparative voice across all \
candidates, not independent self-assessments.

Candidates:
{participants}

Task: {task}

Calibration anchors — assign each candidate the confidence band that matches \
its fit relative to the others:
- 0.9 = uniquely positioned: this task falls squarely inside the candidate's \
stated expertise and no closer specialist is plausible among the candidates.
- 0.7 = capable: the candidate can handle the task, but a closer specialist \
exists in the list.
- 0.4 = adjacent: the candidate has tangentially related expertise; another \
candidate is clearly better.
- 0.0 = out of scope: the task is outside the candidate's described scope.

Self-overclaim is the failure mode you are guarding against — only one candidate \
should typically reach 0.9 for a given task. Prefer comparative discrimination \
over uniformly high scores.

Return the full ranking, ordered best-first. For each candidate, include \
relevant capabilities, an optional cost estimate, and reasoning that references \
the other candidates where useful."""


_REQUIRED_JUDGE_TEMPLATE_PLACEHOLDERS = ("participants", "task")


def _validate_judge_prompt_template(template: str) -> None:
    """Validate a custom judge prompt template at construction time."""
    formatter = string.Formatter()
    referenced = {
        field_name for _literal, field_name, _format_spec, _conversion in formatter.parse(template) if field_name
    }
    missing = [key for key in _REQUIRED_JUDGE_TEMPLATE_PLACEHOLDERS if key not in referenced]
    if missing:
        raise ValueError(
            f"prompt_template is missing required placeholder(s) {missing}. "
            f"Required placeholders: {list(_REQUIRED_JUDGE_TEMPLATE_PLACEHOLDERS)}."
        )


def _validate_bid_prompt_template(template: str) -> None:
    """Validate a custom bid prompt template at construction time.

    Surfaces missing required placeholders as ``ValueError`` from the
    constructor so callers see the failure at the boundary, not at the
    first ``generate`` call.
    """
    formatter = string.Formatter()
    referenced = {
        field_name for _literal, field_name, _format_spec, _conversion in formatter.parse(template) if field_name
    }
    missing = [key for key in _REQUIRED_BID_TEMPLATE_PLACEHOLDERS if key not in referenced]
    if missing:
        raise ValueError(
            f"bid_prompt_template is missing required placeholder(s) {missing}. "
            f"Required placeholders: {list(_REQUIRED_BID_TEMPLATE_PLACEHOLDERS)}."
        )


# --- Models ---


class Bid(BaseModel):
    """A bid submitted by an agent for a task.

    Attributes:
        agent_name: Name of the bidding agent.
        confidence: Self-assessed confidence for the task (0.0–1.0).
        capabilities: Relevant capabilities for this task.
        estimated_cost: Optional estimated execution cost.
        reasoning: Explanation of the bid.
        metadata: Arbitrary metadata.
    """

    model_config = ConfigDict(frozen=True)

    agent_name: str
    confidence: float
    capabilities: list[str]
    estimated_cost: float | None = None
    reasoning: str
    metadata: dict[str, Any] = {}


class _BidSchema(BaseModel):
    """Schema used for structured LLM output when generating bids."""

    confidence: float
    capabilities: list[str]
    estimated_cost: float | None = None
    reasoning: str


@dataclass
class BiddableAgent:
    """Pairs an agent with its bid generator for use in auctions.

    Attributes:
        agent: The agent that will execute the task if it wins.
        bid_generator: Strategy for generating this agent's bid.
    """

    agent: Agent
    bid_generator: BidGenerator


class BiddingResult(BaseModel):
    """Outcome of a bidding auction.

    Attributes:
        winning_bid: The winning bid, or ``None`` if no agent was allocated.
        all_bids: All bids received from participants.
        execution_result: Output from the winning agent's run, or ``None``.
        allocated: Whether a winner was selected and executed.
        bid_failures: Failures from bid generation.
        execution_error: Error message if the winning agent failed during execution.
    """

    model_config = ConfigDict(frozen=True)

    winning_bid: Bid | None
    all_bids: list[Bid]
    execution_result: Any
    allocated: bool
    bid_failures: list[AgentFailure] = []
    execution_error: str | None = None


# --- Protocols ---


@runtime_checkable
class BidGenerator(Protocol):
    """Protocol for generating bids on behalf of an agent.

    ``emitter`` is keyword-only and required — the calling primitive
    (typically :class:`Bidding`) passes its own run-scoped emitter so
    any LLM or tool calls made inside ``generate`` are traced under the
    caller's run. Custom LLM-using generators should wrap their client
    with :class:`~nanitics.infrastructure.llm.instrumented.InstrumentedLLMClient`
    using the passed emitter; see :class:`LLMBidGenerator` for the
    canonical pattern.
    """

    async def generate(
        self,
        agent_name: str,
        task: str,
        *,
        emitter: EventEmitter,
    ) -> Bid: ...


@runtime_checkable
class AllocationStrategy(Protocol):
    """Protocol for selecting a winning bid from a list of bids."""

    def select(self, bids: list[Bid]) -> Bid | None: ...


# --- Bid Generators ---


class LLMBidGenerator:
    """Generate bids using an LLM to assess agent suitability.

    The LLM evaluates the agent's description against the task and
    produces structured confidence, capabilities, cost, and reasoning.

    Each :meth:`generate` call emits one ``LLMRequestEvent`` and one
    ``LLMResponseEvent`` — both labelled ``"bid"`` — through the emitter
    passed by the caller. This lets bid-phase LLM spend roll up into the
    caller's Observatory run alongside the winning agent's calls, so the
    run's ``summary.total_input_tokens`` / ``summary.total_output_tokens``
    reflect the full per-run cost. The wrapping is per-call: the
    generator itself is emitter-agnostic and safe to share across
    concurrent :class:`Bidding` invocations.

    Args:
        llm_client: LLM client for bid generation.
        agent_description: Description of the agent's expertise and tools.
        bid_prompt_template: Optional custom prompt template. When ``None``
            (default), the legacy uncalibrated wording is used so existing
            adopters see no behavioural change. When set, the template is
            rendered with ``.format(agent_name=..., agent_description=...,
            task=...)``. Pass :data:`DEFAULT_CALIBRATED_BID_PROMPT_TEMPLATE`
            to enable the four-tier calibration anchors that counter
            self-overclaim. Templates missing any required placeholder
            raise :class:`ValueError` from the constructor.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        agent_description: str,
        *,
        bid_prompt_template: str | None = None,
    ) -> None:
        if bid_prompt_template is not None:
            _validate_bid_prompt_template(bid_prompt_template)
        self._llm_client = llm_client
        self._agent_description = agent_description
        self._bid_prompt_template = bid_prompt_template

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
        if self._bid_prompt_template is None:
            prompt = (
                f"You are evaluating whether agent '{agent_name}' is suitable for a task.\n\n"
                f"Agent description: {self._agent_description}\n\n"
                f"Task: {task}\n\n"
                "Rate your confidence (0.0-1.0), list relevant capabilities, "
                "estimate cost if possible, and explain your reasoning."
            )
        else:
            prompt = self._bid_prompt_template.format(
                agent_name=agent_name,
                agent_description=self._agent_description,
                task=task,
            )
        result = await instrumented.generate(
            system_prompt="You are a bid evaluator.",
            messages=[Message(role="user", content=prompt)],
            output_schema=_BidSchema,
        )
        parsed: _BidSchema = cast(_BidSchema, result.parsed)
        return Bid(
            agent_name=agent_name,
            confidence=max(0.0, min(1.0, parsed.confidence)),
            capabilities=parsed.capabilities,
            estimated_cost=parsed.estimated_cost,
            reasoning=parsed.reasoning,
        )


class FixedBidGenerator:
    """Generate bids with predetermined values.

    Useful when agent capabilities are known in advance and don't
    need LLM assessment.

    Args:
        confidence: Fixed confidence score (0.0–1.0).
        capabilities: Fixed list of capabilities.
        estimated_cost: Fixed estimated cost.
    """

    def __init__(
        self,
        confidence: float,
        capabilities: list[str] | None = None,
        estimated_cost: float | None = None,
    ) -> None:
        self._confidence = confidence
        self._capabilities = capabilities or []
        self._estimated_cost = estimated_cost

    async def generate(
        self,
        agent_name: str,
        task: str,
        *,
        emitter: EventEmitter,
    ) -> Bid:
        # ``emitter`` is accepted for :class:`BidGenerator` protocol parity.
        # Fixed bids are deterministic state with no work to trace, so no
        # event is emitted.
        del emitter
        return Bid(
            agent_name=agent_name,
            confidence=self._confidence,
            capabilities=self._capabilities,
            estimated_cost=self._estimated_cost,
            reasoning="Fixed bid",
        )


# --- Allocation Strategies ---


class HighestConfidence:
    """Select the bid with the highest confidence score.

    Args:
        tiebreaker: Optional secondary :class:`AllocationStrategy` used to
            break strict ties at the top confidence value. When ``None``
            (default), the first-listed bid wins on a tie — preserves the
            legacy behaviour for backward-compatible re-runs. A common
            chain that disambiguates calibrated bids deterministically is
            ``HighestConfidence(tiebreaker=LowestCost())``; tiebreakers
            themselves can take a ``tiebreaker`` to chain further (e.g.
            ``LowestCost`` → capability-count → first). When the supplied
            tiebreaker returns ``None`` on the tied subset (for example,
            :class:`LowestCost` invoked on bids that all lack
            ``estimated_cost``), this strategy falls through to the first
            tied bid so callers always get a winner when bids exist.
    """

    def __init__(self, *, tiebreaker: AllocationStrategy | None = None) -> None:
        self._tiebreaker = tiebreaker

    def select(self, bids: list[Bid]) -> Bid | None:
        if not bids:
            return None
        top_confidence = max(b.confidence for b in bids)
        tied = [b for b in bids if b.confidence == top_confidence]
        if len(tied) == 1 or self._tiebreaker is None:
            return tied[0]
        resolved = self._tiebreaker.select(tied)
        if resolved is None:
            return tied[0]
        return resolved


class LowestCost:
    """Select the bid with the lowest estimated cost.

    Bids without an ``estimated_cost`` are excluded from consideration.

    Args:
        tiebreaker: Optional secondary :class:`AllocationStrategy` used to
            break strict ties at the lowest cost value. When ``None``
            (default), the first-listed bid among the tied-cheapest bids
            wins. Symmetric with :class:`HighestConfidence` so callers can
            compose multi-stage chains such as
            ``HighestConfidence(tiebreaker=LowestCost(tiebreaker=...))``.
    """

    def __init__(self, *, tiebreaker: AllocationStrategy | None = None) -> None:
        self._tiebreaker = tiebreaker

    def select(self, bids: list[Bid]) -> Bid | None:
        costed = [b for b in bids if b.estimated_cost is not None]
        if not costed:
            return None
        cheapest_cost = min(b.estimated_cost for b in costed if b.estimated_cost is not None)
        tied = [b for b in costed if b.estimated_cost == cheapest_cost]
        if len(tied) == 1 or self._tiebreaker is None:
            return tied[0]
        resolved = self._tiebreaker.select(tied)
        if resolved is None:
            return tied[0]
        return resolved


class WeightedScore:
    """Select the bid with the highest weighted score across dimensions.

    Normalizes values across all bids before applying weights. Supported
    weight keys: ``"confidence"`` (higher is better), ``"cost"`` (lower
    is better, automatically inverted), ``"capabilities"`` (more is better).

    Args:
        weights: Mapping of dimension name to weight value.
    """

    def __init__(self, weights: dict[str, float]) -> None:
        self._weights = weights

    def select(self, bids: list[Bid]) -> Bid | None:
        if not bids:
            return None

        # Collect raw values for normalization
        confidences = [b.confidence for b in bids]
        costs = [b.estimated_cost for b in bids if b.estimated_cost is not None]
        cap_counts = [len(b.capabilities) for b in bids]

        def _normalize(val: float, min_v: float, max_v: float) -> float:
            if max_v == min_v:
                return 1.0
            return (val - min_v) / (max_v - min_v)

        conf_min, conf_max = min(confidences), max(confidences)
        cost_min = min(costs) if costs else 0.0
        cost_max = max(costs) if costs else 0.0
        cap_min, cap_max = min(cap_counts), max(cap_counts)

        best_bid = bids[0]
        best_score = float("-inf")

        for bid in bids:
            score = 0.0

            if "confidence" in self._weights:
                score += self._weights["confidence"] * _normalize(bid.confidence, conf_min, conf_max)

            if "cost" in self._weights and bid.estimated_cost is not None and costs:
                # Invert: lower cost is better
                normalized_cost = _normalize(bid.estimated_cost, cost_min, cost_max)
                score += self._weights["cost"] * (1.0 - normalized_cost)

            if "capabilities" in self._weights:
                score += self._weights["capabilities"] * _normalize(len(bid.capabilities), cap_min, cap_max)

            if score > best_score:
                best_score = score
                best_bid = bid

        return best_bid


# --- Bidding Controller ---


class Bidding:
    """Competitive auction for task allocation among agents.

    Each participant generates a bid, an allocation strategy selects a
    winner, and the winning agent executes the task.

    Args:
        participants: Agents paired with their bid generators.
        emitter: Event emitter for bidding events.
        allocation_strategy: Strategy for selecting the winner.
            Defaults to ``HighestConfidence``.
        min_bid_threshold: Minimum confidence to accept a winner.
            Winners below this threshold are rejected.
        cancellation_token: Cancellation signal.
    """

    def __init__(
        self,
        *,
        participants: list[BiddableAgent],
        emitter: EventEmitter,
        allocation_strategy: AllocationStrategy | None = None,
        min_bid_threshold: float | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> None:
        self._participants = participants
        self._emitter = emitter
        self._allocation_strategy: AllocationStrategy = (
            allocation_strategy if allocation_strategy is not None else HighestConfidence()
        )
        self._min_bid_threshold = min_bid_threshold
        self._cancellation_token = cancellation_token

    async def run(self, task: str) -> BiddingResult:
        """Run the bidding auction.

        Collects bids from all participants in parallel, selects a winner
        via the allocation strategy, and executes the task with the
        winning agent.

        Args:
            task: The task to bid on and execute.

        Returns:
            A ``BiddingResult`` with the winning bid and execution output.
        """
        self._emitter.emit(
            BiddingStartEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                task=task,
                participant_names=[p.agent.name for p in self._participants],
            )
        )

        bid_failures: list[AgentFailure] = []

        async def _generate_bid(participant: BiddableAgent) -> Bid | AgentFailure:
            try:
                bid = await participant.bid_generator.generate(
                    participant.agent.name,
                    task,
                    emitter=self._emitter,
                )
                self._emitter.emit(
                    BidReceivedEvent(
                        trace_id=self._emitter.trace_id,
                        span_id=self._emitter.span_id,
                        parent_span_id=self._emitter.parent_span_id,
                        agent_name=bid.agent_name,
                        confidence=bid.confidence,
                        reasoning=bid.reasoning,
                        estimated_cost=bid.estimated_cost,
                    )
                )
                return bid
            except Exception as exc:
                failure = AgentFailure(
                    agent_name=participant.agent.name,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                self._emitter.emit(
                    BidReceivedEvent(
                        trace_id=self._emitter.trace_id,
                        span_id=self._emitter.span_id,
                        parent_span_id=self._emitter.parent_span_id,
                        agent_name=participant.agent.name,
                        confidence=0.0,
                        reasoning="",
                        error=str(exc),
                    )
                )
                return failure

        bid_results = await asyncio.gather(*(_generate_bid(p) for p in self._participants))
        bids = [b for b in bid_results if isinstance(b, Bid)]
        bid_failures = [b for b in bid_results if isinstance(b, AgentFailure)]

        winner = self._allocation_strategy.select(bids)

        rejection_reason: str | None = None
        if winner is not None and self._min_bid_threshold is not None:
            if winner.confidence < self._min_bid_threshold:
                rejection_reason = "below_threshold"
                winner = None
        if not bids:
            rejection_reason = "no_bids"

        self._emitter.emit(
            BidAllocatedEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                winner=winner.agent_name if winner else None,
                confidence=winner.confidence if winner else None,
                total_bids=len(bids),
                rejection_reason=rejection_reason,
            )
        )

        execution_result: Any = None
        execution_error: str | None = None
        allocated = winner is not None

        if winner is not None:
            # Find the matching agent
            winning_agent: Agent | None = None
            for p in self._participants:
                if p.agent.name == winner.agent_name:
                    winning_agent = p.agent
                    break

            if winning_agent is not None:
                try:
                    agent_result = await winning_agent.bind(self._emitter).run(task)
                    execution_result = agent_result.output
                except Exception as exc:
                    execution_result = None
                    execution_error = str(exc)
                    winner = winner.model_copy(update={"metadata": {**winner.metadata, "execution_error": str(exc)}})

        self._emitter.emit(
            BiddingCompleteEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                winner=winner.agent_name if winner else None,
                total_participants=len(self._participants),
                allocated=allocated,
            )
        )

        return BiddingResult(
            winning_bid=winner,
            all_bids=bids,
            execution_result=execution_result,
            allocated=allocated,
            bid_failures=bid_failures,
            execution_error=execution_error,
        )
