from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import (
    ConsensusAgreementEvent,
    ConsensusCompleteEvent,
    ConsensusStartEvent,
    ConsensusVoteEvent,
)
from nanitics.safety.cancellation import CancellationToken
from nanitics.strategies.agents.base import Agent

# --- Models ---


class ConsensusResponse(BaseModel):
    """An individual agent's response in a consensus round.

    Attributes:
        agent_name: Name of the responding agent.
        output: The agent's response content.
        round: Round number this response came from.
        steps: Number of agent steps taken.
        termination_reason: Why the agent stopped.
    """

    model_config = ConfigDict(frozen=True)

    agent_name: str
    output: Any
    round: int
    steps: int
    termination_reason: str


class ConsensusAggregation(BaseModel):
    """Result of aggregating consensus responses.

    Attributes:
        result: The aggregated result value.
        agreement_level: Degree of agreement (0.0–1.0).
        vote_distribution: Count per unique response (stringified).
        strategy: Name of the aggregation strategy used.
    """

    model_config = ConfigDict(frozen=True)

    result: Any
    agreement_level: float
    vote_distribution: dict[str, int]
    strategy: str


class ConsensusResult(BaseModel):
    """Outcome of a consensus run.

    Attributes:
        aggregation: The aggregated consensus outcome.
        responses: All responses across all rounds.
        rounds_completed: Total rounds completed.
        termination_reason: ``"single_round"``, ``"agreement_reached"``,
            or ``"max_rounds"``.
        agents_participated: Number of unique agents that participated.
    """

    model_config = ConfigDict(frozen=True)

    aggregation: ConsensusAggregation
    responses: list[ConsensusResponse]
    rounds_completed: int
    termination_reason: str
    agents_participated: int


class DeliberationConfig(BaseModel):
    """Configuration for multi-round deliberation in consensus.

    Attributes:
        max_rounds: Maximum deliberation rounds.
        agreement_threshold: Agreement level to stop early (0.0–1.0).
        agreement_fn: Custom function measuring agreement across responses.
            Defaults to string-equality grouping.
        fallback_strategy: Aggregation strategy used when max_rounds is
            reached without agreement. Defaults to ``MajorityVoting``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    max_rounds: int = 3
    agreement_threshold: float = 1.0
    agreement_fn: Callable[[list[ConsensusResponse]], float] | None = None
    fallback_strategy: Any | None = None  # AggregationStrategy, typed as Any to avoid forward ref


# --- Protocols ---


@runtime_checkable
class AggregationStrategy(Protocol):
    """Protocol for aggregating consensus responses into a collective decision."""

    async def aggregate(self, responses: list[ConsensusResponse]) -> ConsensusAggregation: ...


# --- Aggregation Strategies ---


class MajorityVoting:
    """Select the response held by the largest group.

    Groups responses by equality and picks the most popular. Agreement
    level is the fraction of responses in the winning group.

    Args:
        eq_fn: Custom equality function for grouping responses.
            Defaults to string comparison.
    """

    def __init__(
        self,
        eq_fn: Callable[[Any, Any], bool] | None = None,
    ) -> None:
        self._eq_fn = eq_fn or (lambda a, b: str(a) == str(b))

    async def aggregate(self, responses: list[ConsensusResponse]) -> ConsensusAggregation:
        groups: list[tuple[Any, list[ConsensusResponse]]] = []
        for resp in responses:
            placed = False
            for representative, members in groups:
                if self._eq_fn(resp.output, representative):
                    members.append(resp)
                    placed = True
                    break
            if not placed:
                groups.append((resp.output, [resp]))

        best_group = max(groups, key=lambda g: len(g[1]))
        total = len(responses)

        vote_distribution: dict[str, int] = {}
        for representative, members in groups:
            vote_distribution[str(representative)] = len(members)

        return ConsensusAggregation(
            result=best_group[0],
            agreement_level=len(best_group[1]) / total,
            vote_distribution=vote_distribution,
            strategy="MajorityVoting",
        )


class WeightedVoting:
    """Select the response group with the highest total weight.

    Like ``MajorityVoting``, but each response is weighted by a user-defined
    function. The group with the highest cumulative weight wins.

    Args:
        weight_fn: Function computing a weight for each response.
        eq_fn: Custom equality function for grouping responses.
            Defaults to string comparison.
    """

    def __init__(
        self,
        weight_fn: Callable[[ConsensusResponse], float],
        eq_fn: Callable[[Any, Any], bool] | None = None,
    ) -> None:
        self._weight_fn = weight_fn
        self._eq_fn = eq_fn or (lambda a, b: str(a) == str(b))

    async def aggregate(self, responses: list[ConsensusResponse]) -> ConsensusAggregation:
        groups: list[tuple[Any, list[ConsensusResponse], float]] = []
        for resp in responses:
            weight = self._weight_fn(resp)
            placed = False
            for i, (representative, members, total_weight) in enumerate(groups):
                if self._eq_fn(resp.output, representative):
                    members.append(resp)
                    groups[i] = (representative, members, total_weight + weight)
                    placed = True
                    break
            if not placed:
                groups.append((resp.output, [resp], weight))

        best_group = max(groups, key=lambda g: g[2])
        total_weight = sum(g[2] for g in groups)

        vote_distribution: dict[str, int] = {}
        for representative, members, _ in groups:
            vote_distribution[str(representative)] = len(members)

        return ConsensusAggregation(
            result=best_group[0],
            agreement_level=best_group[2] / total_weight if total_weight > 0 else 0.0,
            vote_distribution=vote_distribution,
            strategy="WeightedVoting",
        )


class BestOfN:
    """Select the single highest-scoring response.

    Scores each response individually and picks the best. Supports
    both synchronous and asynchronous scorer functions.

    Args:
        scorer: Function computing a score for each response.
    """

    def __init__(
        self,
        scorer: Callable[[ConsensusResponse], float | Awaitable[float]],
    ) -> None:
        self._scorer = scorer

    async def aggregate(self, responses: list[ConsensusResponse]) -> ConsensusAggregation:
        scored: list[tuple[ConsensusResponse, float]] = []
        for resp in responses:
            result = self._scorer(resp)
            if inspect.isawaitable(result):
                score = await result
            else:
                score = result
            scored.append((resp, score))

        best = max(scored, key=lambda s: s[1])

        return ConsensusAggregation(
            result=best[0].output,
            agreement_level=1.0,
            vote_distribution={best[0].agent_name: 1},
            strategy="BestOfN",
        )


# --- Peer Response Formatting ---


def _format_peer_responses(responses: list[ConsensusResponse], exclude: str) -> str:
    parts = [f"[{resp.agent_name}]: {resp.output}" for resp in responses if resp.agent_name != exclude]
    return "\n".join(parts)


# --- Default Agreement Function ---


def _default_agreement(responses: list[ConsensusResponse]) -> float:
    if not responses:
        return 0.0
    groups: dict[str, int] = {}
    for resp in responses:
        key = str(resp.output)
        groups[key] = groups.get(key, 0) + 1
    return max(groups.values()) / len(responses)


# --- Consensus Controller ---


class Consensus:
    """Gather independent responses and aggregate into a collective decision.

    In single-round mode, all agents respond in parallel and responses
    are aggregated. With deliberation, agents see peer responses and
    revise their answers across multiple rounds until agreement converges
    or the round limit is reached.

    Args:
        agents: At least 2 agents to participate.
        emitter: Event emitter for consensus events.
        aggregation_strategy: Strategy for aggregating responses.
            Defaults to ``MajorityVoting``.
        deliberation: Configuration for multi-round deliberation.
            When ``None``, runs a single round.
        cancellation_token: Cancellation signal.

    Raises:
        ValueError: If fewer than 2 agents are provided.
    """

    def __init__(
        self,
        *,
        agents: Sequence[Agent],
        emitter: EventEmitter,
        aggregation_strategy: AggregationStrategy | None = None,
        deliberation: DeliberationConfig | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> None:
        if len(agents) < 2:
            raise ValueError("Consensus requires at least 2 agents")
        self._agents = agents
        self._emitter = emitter
        self._aggregation_strategy = aggregation_strategy or MajorityVoting()
        self._deliberation = deliberation
        self._cancellation_token = cancellation_token

    async def run(self, task: str) -> ConsensusResult:
        """Execute the consensus process.

        Runs either a single round or multi-round deliberation depending
        on whether a ``DeliberationConfig`` was provided.

        Args:
            task: The task or question to reach consensus on.

        Returns:
            A ``ConsensusResult`` with the aggregated outcome and all responses.
        """
        strategy_name = type(self._aggregation_strategy).__name__

        self._emitter.emit(
            ConsensusStartEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                task=task,
                agent_names=[a.name for a in self._agents],
                strategy=strategy_name,
                deliberation_enabled=self._deliberation is not None,
            )
        )

        if self._deliberation is None:
            return await self._run_single_round(task, strategy_name)
        return await self._run_deliberation(task, strategy_name)

    async def _run_single_round(self, task: str, strategy_name: str) -> ConsensusResult:
        responses = await self._collect_responses(task, round_num=1)

        aggregation = await self._aggregation_strategy.aggregate(responses)

        self._emitter.emit(
            ConsensusCompleteEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                strategy=strategy_name,
                rounds_completed=1,
                final_agreement=aggregation.agreement_level,
                agents_participated=len(responses),
                termination_reason="single_round",
            )
        )

        return ConsensusResult(
            aggregation=aggregation,
            responses=responses,
            rounds_completed=1,
            termination_reason="single_round",
            agents_participated=len(responses),
        )

    async def _run_deliberation(self, task: str, strategy_name: str) -> ConsensusResult:
        config = self._deliberation
        assert config is not None
        agreement_fn = config.agreement_fn or _default_agreement

        all_responses: list[ConsensusResponse] = []

        # Round 1: independent generation
        current_responses = await self._collect_responses(task, round_num=1)
        all_responses.extend(current_responses)

        agreement = agreement_fn(current_responses)
        self._emitter.emit(
            ConsensusAgreementEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                round=1,
                agreement_level=agreement,
                converged=agreement >= config.agreement_threshold,
            )
        )

        if agreement >= config.agreement_threshold:
            aggregation = await self._aggregation_strategy.aggregate(current_responses)
            return self._finalize(
                aggregation=aggregation,
                all_responses=all_responses,
                rounds_completed=1,
                termination_reason="agreement_reached",
                strategy_name=strategy_name,
            )

        # Deliberation rounds
        for round_num in range(2, config.max_rounds + 1):
            tasks = []
            for agent in self._agents:
                peer_context = _format_peer_responses(current_responses, agent.name)
                deliberation_task = (
                    f"Original task: {task}\n\nOther agents' responses:\n{peer_context}\n\n"
                    f"Considering all perspectives, provide your revised answer."
                )
                tasks.append(self._run_agent(agent, deliberation_task, round_num))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            revised_responses = [result for result in results if isinstance(result, ConsensusResponse)]

            current_responses = revised_responses
            all_responses.extend(current_responses)

            agreement = agreement_fn(current_responses)
            self._emitter.emit(
                ConsensusAgreementEvent(
                    trace_id=self._emitter.trace_id,
                    span_id=self._emitter.span_id,
                    parent_span_id=self._emitter.parent_span_id,
                    round=round_num,
                    agreement_level=agreement,
                    converged=agreement >= config.agreement_threshold,
                )
            )

            if agreement >= config.agreement_threshold:
                aggregation = await self._aggregation_strategy.aggregate(current_responses)
                return self._finalize(
                    aggregation=aggregation,
                    all_responses=all_responses,
                    rounds_completed=round_num,
                    termination_reason="agreement_reached",
                    strategy_name=strategy_name,
                )

        # Max rounds reached — use fallback strategy
        fallback = config.fallback_strategy or MajorityVoting()
        aggregation = await fallback.aggregate(current_responses)
        return self._finalize(
            aggregation=aggregation,
            all_responses=all_responses,
            rounds_completed=config.max_rounds,
            termination_reason="max_rounds",
            strategy_name=strategy_name,
        )

    async def _collect_responses(self, task: str, round_num: int) -> list[ConsensusResponse]:
        tasks = [self._run_agent(agent, task, round_num) for agent in self._agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [result for result in results if isinstance(result, ConsensusResponse)]

    async def _run_agent(self, agent: Agent, task: str, round_num: int) -> ConsensusResponse:
        try:
            result = await agent.bind(self._emitter).run(task)
            response = ConsensusResponse(
                agent_name=agent.name,
                output=result.output or "",
                round=round_num,
                steps=result.total_steps,
                termination_reason=result.termination_reason,
            )
            self._emitter.emit(
                ConsensusVoteEvent(
                    trace_id=self._emitter.trace_id,
                    span_id=self._emitter.span_id,
                    parent_span_id=self._emitter.parent_span_id,
                    agent_name=agent.name,
                    output=str(response.output),
                    round=round_num,
                )
            )
            return response
        except Exception as exc:
            self._emitter.emit(
                ConsensusVoteEvent(
                    trace_id=self._emitter.trace_id,
                    span_id=self._emitter.span_id,
                    parent_span_id=self._emitter.parent_span_id,
                    agent_name=agent.name,
                    output="",
                    round=round_num,
                    error=str(exc),
                )
            )
            raise

    def _finalize(
        self,
        *,
        aggregation: ConsensusAggregation,
        all_responses: list[ConsensusResponse],
        rounds_completed: int,
        termination_reason: str,
        strategy_name: str,
    ) -> ConsensusResult:
        self._emitter.emit(
            ConsensusCompleteEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                strategy=strategy_name,
                rounds_completed=rounds_completed,
                final_agreement=aggregation.agreement_level,
                agents_participated=len({r.agent_name for r in all_responses}),
                termination_reason=termination_reason,
            )
        )
        return ConsensusResult(
            aggregation=aggregation,
            responses=all_responses,
            rounds_completed=rounds_completed,
            termination_reason=termination_reason,
            agents_participated=len({r.agent_name for r in all_responses}),
        )
