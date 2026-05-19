from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from nanitics.capabilities.memory.shared import SharedMemory
from nanitics.capabilities.memory.shared_tools import create_shared_memory_tools
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import (
    BlackboardCompleteEvent,
    BlackboardRoundEntry,
    BlackboardRoundEvent,
    BlackboardStartEvent,
    SharedMemoryRetractEvent,
    SharedMemorySupersededEvent,
    SharedMemoryWriteEvent,
    TraceEvent,
)
from nanitics.strategies.agents.base import Agent


@dataclass
class _RoundCollector:
    """Mutable per-round accounting for the blackboard contribution listener.

    A single listener, registered once for the lifetime of ``Blackboard.run``,
    reads and writes through this object. Each round resets the fields and
    toggles ``active``; outside an active window the listener is a no-op.
    Using a shared collector avoids registering fresh closures per round,
    which would alias loop-variable state across rounds.
    """

    active: bool = False
    contributions: int = 0
    entries: list[BlackboardRoundEntry] = field(default_factory=list)
    agent_authors: dict[str, int] = field(default_factory=dict)

    def reset(self) -> None:
        self.contributions = 0
        self.entries = []
        self.agent_authors = {}
        self.active = True

    def record_write(self, event: SharedMemoryWriteEvent) -> None:
        self.contributions += 1
        self.agent_authors[event.author] = self.agent_authors.get(event.author, 0) + 1
        self.entries.append(
            BlackboardRoundEntry(
                operation="write",
                author=event.author,
                content=event.content,
                scope=event.scope,
                entry_id=event.entry_id,
            )
        )

    def record_supersede(self, event: SharedMemorySupersededEvent) -> None:
        self.contributions += 1
        self.agent_authors[event.author] = self.agent_authors.get(event.author, 0) + 1
        self.entries.append(
            BlackboardRoundEntry(
                operation="supersede",
                author=event.author,
                content=event.content,
                scope=event.scope,
                entry_id=event.new_entry_id,
                original_entry_id=event.original_entry_id,
            )
        )

    def record_retract(self, event: SharedMemoryRetractEvent) -> None:
        self.contributions += 1
        self.agent_authors[event.author] = self.agent_authors.get(event.author, 0) + 1
        self.entries.append(
            BlackboardRoundEntry(
                operation="retract",
                author=event.author,
                content="",
                scope=event.scope,
                entry_id=event.entry_id,
                retract_reason=event.reason,
            )
        )


# --- Models ---


class BlackboardState(BaseModel):
    """Snapshot of blackboard state passed to control strategies and termination conditions.

    Attributes:
        round_number: Current round number (1-indexed).
        round_contributions: Shared memory writes in the current round.
        total_contributions: Cumulative contributions across all rounds.
    """

    model_config = ConfigDict(frozen=True)

    round_number: int
    round_contributions: int
    total_contributions: int


class BlackboardResult(BaseModel):
    """Outcome of a blackboard coordination run.

    Attributes:
        entries: All shared memory entries (including inactive/retracted).
        rounds_completed: Total rounds executed.
        termination_reason: Class name of the termination condition that fired,
            or ``"max_rounds"`` if the round limit was reached.
        agent_contributions: Number of contributions per agent name.
    """

    model_config = ConfigDict(frozen=True)

    entries: list[Any]
    rounds_completed: int
    termination_reason: str
    agent_contributions: dict[str, int]


# --- Protocols ---


@runtime_checkable
class ControlStrategy(Protocol):
    """Protocol for blackboard control strategies.

    Determines which agents run each round and whether they run in
    parallel or sequentially.
    """

    @property
    def parallel(self) -> bool: ...

    def select(self, agents: Sequence[Agent], state: BlackboardState) -> Sequence[Agent]: ...


@runtime_checkable
class TerminationCondition(Protocol):
    """Protocol for blackboard termination conditions.

    Checked after each round to decide whether to stop iteration.
    """

    async def should_terminate(self, state: BlackboardState, shared_memory: SharedMemory) -> bool: ...


# --- Control Strategies ---


class ScheduledControl:
    """Run all agents sequentially in order every round."""

    @property
    def parallel(self) -> bool:
        return False

    def select(self, agents: Sequence[Agent], state: BlackboardState) -> Sequence[Agent]:
        return agents


class PrioritizedControl:
    """Run all agents sequentially, ordered by priority (highest first).

    Args:
        priorities: Mapping of agent name to priority value.
            Agents not in the map get priority 0.
    """

    def __init__(self, priorities: dict[str, int]) -> None:
        self._priorities = priorities

    @property
    def parallel(self) -> bool:
        return False

    def select(self, agents: Sequence[Agent], state: BlackboardState) -> Sequence[Agent]:
        return sorted(agents, key=lambda a: self._priorities.get(a.name, 0), reverse=True)


class OpportunisticControl:
    """Run all agents in parallel every round."""

    @property
    def parallel(self) -> bool:
        return True

    def select(self, agents: Sequence[Agent], state: BlackboardState) -> Sequence[Agent]:
        return agents


# --- Termination Conditions ---


class NoNewContributions:
    """Terminate when a round produces zero shared memory writes."""

    async def should_terminate(self, state: BlackboardState, shared_memory: SharedMemory) -> bool:
        return state.round_contributions == 0


class MaxRoundsTermination:
    """Terminate after a fixed number of rounds.

    Args:
        max_rounds: Stop when ``round_number >= max_rounds``.
    """

    def __init__(self, max_rounds: int) -> None:
        if max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        self._max_rounds = max_rounds

    async def should_terminate(self, state: BlackboardState, shared_memory: SharedMemory) -> bool:
        return state.round_number >= self._max_rounds


class BlackboardPredicateTermination:
    """Terminate based on a custom async predicate.

    Args:
        predicate: Async callable receiving ``(BlackboardState, SharedMemory)``
            and returning a truthy value to terminate.
    """

    def __init__(self, predicate: Callable[[BlackboardState, SharedMemory], Any]) -> None:
        self._predicate = predicate

    async def should_terminate(self, state: BlackboardState, shared_memory: SharedMemory) -> bool:
        return bool(await self._predicate(state, shared_memory))


class BlackboardCompositeTermination:
    """Combine multiple termination conditions.

    Args:
        conditions: List of termination conditions to evaluate.
        mode: ``"any"`` (default) terminates when any condition fires;
            ``"all"`` terminates only when all conditions fire.
    """

    def __init__(
        self,
        conditions: list[TerminationCondition],
        mode: str = "any",
    ) -> None:
        self._conditions = conditions
        self._mode = mode

    async def should_terminate(self, state: BlackboardState, shared_memory: SharedMemory) -> bool:
        results = [await c.should_terminate(state, shared_memory) for c in self._conditions]
        if self._mode == "any":
            return any(results)
        return all(results)


# --- Blackboard Controller ---


class Blackboard:
    """Coordinate agents through a shared memory space.

    Agents read from and write to a ``SharedMemory`` instance. A control
    strategy determines which agents run each round, and a termination
    condition decides when to stop iterating.

    Args:
        shared_memory: The shared memory instance agents interact with.
        agents: Agents participating in the blackboard.
        emitter: Event emitter for blackboard events.
        control: Strategy controlling agent selection and parallelism.
            Defaults to ``ScheduledControl`` (sequential, all agents).
        termination: Condition for stopping iteration.
            Defaults to ``NoNewContributions``.
        max_rounds: Hard upper bound on rounds regardless of termination.
    """

    def __init__(
        self,
        *,
        shared_memory: SharedMemory,
        agents: Sequence[Agent],
        emitter: EventEmitter,
        control: ControlStrategy | None = None,
        termination: TerminationCondition | None = None,
        max_rounds: int = 10,
    ) -> None:
        if not agents:
            raise ValueError("agents must not be empty")
        non_capable = [a.name for a in agents if not a.supports_dynamic_tools]
        if non_capable:
            raise ValueError(
                f"All Blackboard agents must support dynamic tools (reactive tool loop). "
                f"These agents do not: {', '.join(non_capable)}. "
                f"Use ReActAgent or another agent type with supports_dynamic_tools=True."
            )
        self._shared_memory = shared_memory
        self._agents = agents
        self._emitter = emitter
        self._control: ControlStrategy = control if control is not None else ScheduledControl()
        self._termination: TerminationCondition = termination if termination is not None else NoNewContributions()
        self._max_rounds = max_rounds

    async def run(self, task: str) -> BlackboardResult:
        """Execute the blackboard coordination loop.

        Injects shared memory tools into participating agents before the
        loop and removes them afterward.  Agents that don't support tools
        have their output auto-captured as shared memory contributions.

        Args:
            task: The task description passed to each agent.

        Returns:
            A ``BlackboardResult`` with all entries and contribution metrics.
        """
        agent_contributions: dict[str, int] = {a.name: 0 for a in self._agents}
        total_contributions = 0
        round_number = 0
        termination_reason = "max_rounds"

        # --- Inject shared memory tools into agents ---
        injected_tools: dict[str, list[str]] = {}
        for agent in self._agents:
            tools = create_shared_memory_tools(self._shared_memory, agent.name)
            added = agent.add_tools(tools)
            injected_tools[agent.name] = added

        self._emitter.emit(
            BlackboardStartEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                task=task,
                agent_names=[a.name for a in self._agents],
                control_strategy=type(self._control).__name__,
                max_rounds=self._max_rounds,
            )
        )

        collector = _RoundCollector()

        def _contribution_listener(event: TraceEvent) -> None:
            if not collector.active:
                return
            if isinstance(event, SharedMemoryWriteEvent):
                collector.record_write(event)
            elif isinstance(event, SharedMemorySupersededEvent):
                collector.record_supersede(event)
            elif isinstance(event, SharedMemoryRetractEvent):
                collector.record_retract(event)

        self._emitter.add_listener(_contribution_listener, internal=True)

        try:
            for round_number in range(1, self._max_rounds + 1):
                collector.reset()

                state = BlackboardState(
                    round_number=round_number,
                    round_contributions=0,
                    total_contributions=total_contributions,
                )
                selected = self._control.select(list(self._agents), state)
                activated_names = [a.name for a in selected]

                try:
                    if self._control.parallel:
                        # Each bound handle is constructed inside its own
                        # task via bind-and-run so the child emitter's
                        # span-stack ContextVar is set in the task that
                        # reads it.
                        async def _bind_and_run(a: Any, emitter: EventEmitter) -> Any:
                            return await a.bind(emitter).run(task)

                        results = await asyncio.gather(*(_bind_and_run(a, self._emitter) for a in selected))
                        agent_results: dict[str, Any] = dict(zip([a.name for a in selected], results, strict=True))
                    else:
                        agent_results = {}
                        for agent in selected:
                            result = await agent.bind(self._emitter).run(task)
                            agent_results[agent.name] = result
                finally:
                    collector.active = False

                round_contributions = collector.contributions
                round_entries = collector.entries
                round_agent_authors = collector.agent_authors

                total_contributions += round_contributions
                for author, count in round_agent_authors.items():
                    if author in agent_contributions:
                        agent_contributions[author] += count

                state = BlackboardState(
                    round_number=round_number,
                    round_contributions=round_contributions,
                    total_contributions=total_contributions,
                )

                self._emitter.emit(
                    BlackboardRoundEvent(
                        trace_id=self._emitter.trace_id,
                        span_id=self._emitter.span_id,
                        parent_span_id=self._emitter.parent_span_id,
                        round_number=round_number,
                        agents_activated=activated_names,
                        contributions=round_contributions,
                        total_contributions=total_contributions,
                        round_entries=round_entries,
                    )
                )

                if await self._termination.should_terminate(state, self._shared_memory):
                    termination_reason = type(self._termination).__name__
                    if isinstance(self._termination, BlackboardCompositeTermination):
                        for cond in self._termination._conditions:
                            if await cond.should_terminate(state, self._shared_memory):
                                termination_reason = type(cond).__name__
                                break
                    break
        finally:
            # --- Clean up injected tools ---
            for agent in self._agents:
                names = injected_tools.get(agent.name, [])
                if names:
                    agent.remove_tools(names)

        entries = await self._shared_memory.read(include_inactive=True)

        self._emitter.emit(
            BlackboardCompleteEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                rounds_completed=round_number,
                termination_reason=termination_reason,
                total_contributions=total_contributions,
                agent_contributions=agent_contributions,
            )
        )

        return BlackboardResult(
            entries=entries,
            rounds_completed=round_number,
            termination_reason=termination_reason,
            agent_contributions=agent_contributions,
        )
