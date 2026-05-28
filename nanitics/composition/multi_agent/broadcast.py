from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict

from nanitics.infrastructure.llm.protocol import LLMClient, Message
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import (
    BroadcastCompleteEvent,
    BroadcastResponseEvent,
    BroadcastStartEvent,
)
from nanitics.safety.cancellation import CancellationToken
from nanitics.strategies.agents.base import Agent

# --- Models ---


@dataclass(frozen=True)
class AgentFailure:
    """Captures a single agent failure during coordination."""

    agent_name: str
    error_type: str
    error_message: str


class BroadcastResponse(BaseModel):
    """Captures a single agent's response to a broadcast task.

    Attributes:
        agent_name: Name of the responding agent.
        output: The agent's output.
        steps: Number of steps the agent took.
        termination_reason: Why the agent stopped.
    """

    model_config = ConfigDict(frozen=True)

    agent_name: str
    output: Any
    steps: int
    termination_reason: str


class BroadcastResult(BaseModel):
    """Result of a broadcast execution.

    Attributes:
        responses: All successful agent responses.
        failures: Agent failures that occurred during execution.
        aggregated_output: Result from the response strategy aggregation.
        response_strategy: Name of the strategy class used.
        agents_participated: Number of agents that received the task.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    responses: list[BroadcastResponse]
    failures: list[AgentFailure] = []
    aggregated_output: Any
    response_strategy: str
    agents_participated: int


# --- Protocols ---


@runtime_checkable
class ResponseStrategy(Protocol):
    """Protocol for aggregating broadcast responses into a single output."""

    async def aggregate(self, responses: list[BroadcastResponse]) -> Any: ...


@runtime_checkable
class EligibilityFilter(Protocol):
    """Protocol for filtering which agents participate in a broadcast."""

    async def filter(self, agents: Sequence[Agent], task: str) -> list[Agent]: ...


# --- Response Strategies ---


class CollectAll:
    """Collects all response outputs into a list."""

    async def aggregate(self, responses: list[BroadcastResponse]) -> Any:
        return [r.output for r in responses]


class SelectBest:
    """Selects the response with the highest score from a user-defined scorer.

    Args:
        scorer: Sync or async callable that scores a ``BroadcastResponse``.
            Higher scores are better.
    """

    def __init__(self, scorer: Callable[[BroadcastResponse], float | Awaitable[float]]) -> None:
        self._scorer = scorer

    async def aggregate(self, responses: list[BroadcastResponse]) -> Any:
        if not responses:
            return None
        best_response = responses[0]
        raw = self._scorer(best_response)
        best_score: float = cast(float, await raw if asyncio.iscoroutine(raw) else raw)
        for response in responses[1:]:
            raw = self._scorer(response)
            score: float = cast(float, await raw if asyncio.iscoroutine(raw) else raw)
            if score > best_score:
                best_score = score
                best_response = response
        return best_response.output


class MergeResponses:
    """Uses an LLM to synthesize all responses into a unified answer.

    Args:
        llm_client: LLM client for generating the synthesis.
        merge_prompt: Custom prompt template. Must contain a ``{responses}``
            placeholder. Defaults to a built-in synthesis prompt.
    """

    _DEFAULT_PROMPT = (
        "Synthesize the following responses into a single unified answer. "
        "Preserve unique insights from each response.\n\n{responses}"
    )

    def __init__(self, llm_client: LLMClient, merge_prompt: str | None = None) -> None:
        self._llm_client = llm_client
        self._merge_prompt = merge_prompt or self._DEFAULT_PROMPT

    async def aggregate(self, responses: list[BroadcastResponse]) -> Any:
        if not responses:
            return None
        formatted = "\n\n".join(f"--- Response from {r.agent_name} ---\n{r.output}" for r in responses)
        prompt_text = self._merge_prompt.format(responses=formatted)
        result = await self._llm_client.generate(
            system_prompt="You are a response synthesizer.",
            messages=[Message(role="user", content=prompt_text)],
        )
        return result.content


class FilterResponses:
    """Keeps only responses matching a predicate.

    Args:
        predicate: Sync or async callable that returns True to keep a response.
    """

    def __init__(self, predicate: Callable[[BroadcastResponse], bool | Awaitable[bool]]) -> None:
        self._predicate = predicate

    async def aggregate(self, responses: list[BroadcastResponse]) -> Any:
        kept: list[Any] = []
        for response in responses:
            result = self._predicate(response)
            matches = await result if asyncio.iscoroutine(result) else result
            if matches:
                kept.append(response.output)
        return kept


# --- Eligibility Filters ---


class AllEligible:
    """Default eligibility filter that includes all agents."""

    async def filter(self, agents: list[Agent], task: str) -> list[Agent]:
        return list(agents)


class CapabilityFilter:
    """Selects agents whose capabilities overlap with a required set.

    Args:
        capabilities: Mapping of agent name to list of capability strings.
        required: Capabilities that an agent must have at least one of
            to be eligible.
    """

    def __init__(self, capabilities: dict[str, list[str]], required: list[str]) -> None:
        self._capabilities = capabilities
        self._required = set(required)

    async def filter(self, agents: Sequence[Agent], task: str) -> list[Agent]:
        return [a for a in agents if self._required & set(self._capabilities.get(a.name, []))]


# --- Broadcast Controller ---


class Broadcast:
    """Sends a task to multiple agents in parallel and aggregates their responses.

    Agents are filtered for eligibility, executed concurrently, and their
    responses aggregated via the configured ``ResponseStrategy``. Agents
    that raise exceptions are excluded from results.

    Args:
        agents: Agents to broadcast to.
        emitter: Event emitter for broadcast tracing.
        response_strategy: How to aggregate responses. Defaults to
            ``CollectAll``.
        eligibility_filter: Which agents participate. Defaults to
            ``AllEligible``.
        cancellation_token: Cancellation signal.
        thread_keys: Optional mapping from agent name to thread key.
            Each participating agent named in the mapping carries its
            own conversation thread across broadcast runs. Agents not
            in the mapping run stateless. Each agent must be configured
            with a :class:`~nanitics.composition.threads.ThreadStore`
            for the prefix to be persisted.

    Raises:
        ValueError: If ``thread_keys`` references an agent name not
            in ``agents``.
    """

    def __init__(
        self,
        *,
        agents: Sequence[Agent],
        emitter: EventEmitter,
        response_strategy: ResponseStrategy | None = None,
        eligibility_filter: EligibilityFilter | None = None,
        cancellation_token: CancellationToken | None = None,
        thread_keys: dict[str, str] | None = None,
    ) -> None:
        if thread_keys is not None:
            agent_names = {a.name for a in agents}
            unknown = set(thread_keys) - agent_names
            if unknown:
                raise ValueError(
                    f"thread_keys references agents not in this Broadcast: {sorted(unknown)}. "
                    f"Known agents: {sorted(agent_names)}."
                )
        self._agents = agents
        self._emitter = emitter
        self._response_strategy: ResponseStrategy = response_strategy if response_strategy is not None else CollectAll()
        self._eligibility_filter = eligibility_filter if eligibility_filter is not None else AllEligible()
        self._cancellation_token = cancellation_token
        self._thread_keys = thread_keys or {}

    async def run(self, task: str) -> BroadcastResult:
        strategy_name = type(self._response_strategy).__name__
        eligible = await self._eligibility_filter.filter(list(self._agents), task)

        if not eligible:
            return BroadcastResult(
                responses=[],
                aggregated_output=None,
                response_strategy=strategy_name,
                agents_participated=0,
            )

        self._emitter.emit(
            BroadcastStartEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                task=task,
                agent_names=[a.name for a in eligible],
                response_strategy=strategy_name,
            )
        )

        failures: list[AgentFailure] = []

        async def _run_agent(agent: Agent) -> BroadcastResponse | AgentFailure:
            try:
                result = await agent.bind(self._emitter).run(task, thread_key=self._thread_keys.get(agent.name))
                response = BroadcastResponse(
                    agent_name=agent.name,
                    output=result.output,
                    steps=result.total_steps,
                    termination_reason=result.termination_reason,
                )
                self._emitter.emit(
                    BroadcastResponseEvent(
                        trace_id=self._emitter.trace_id,
                        span_id=self._emitter.span_id,
                        parent_span_id=self._emitter.parent_span_id,
                        agent_name=agent.name,
                        output=str(result.output),
                        steps=result.total_steps,
                    )
                )
                return response
            except Exception as exc:
                self._emitter.emit(
                    BroadcastResponseEvent(
                        trace_id=self._emitter.trace_id,
                        span_id=self._emitter.span_id,
                        parent_span_id=self._emitter.parent_span_id,
                        agent_name=agent.name,
                        output="",
                        steps=0,
                        error=str(exc),
                    )
                )
                return AgentFailure(
                    agent_name=agent.name,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )

        results = await asyncio.gather(*(_run_agent(a) for a in eligible))
        responses = [r for r in results if isinstance(r, BroadcastResponse)]
        failures = [r for r in results if isinstance(r, AgentFailure)]

        aggregated = await self._response_strategy.aggregate(responses)

        aggregated_str = str(aggregated) if aggregated is not None else ""

        self._emitter.emit(
            BroadcastCompleteEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                total_agents=len(eligible),
                responses_collected=len(responses),
                response_strategy=strategy_name,
                aggregated_output=aggregated_str,
                failures=len(failures),
            )
        )

        return BroadcastResult(
            responses=responses,
            failures=failures,
            aggregated_output=aggregated,
            response_strategy=strategy_name,
            agents_participated=len(eligible),
        )
