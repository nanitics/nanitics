"""Per-invocation agent binding — ``BoundAgent`` and ``RunContext``.

Returned by :meth:`Agent.bind` as a non-mutating handle that scopes a
child emitter to a single ``run`` call. The underlying ``Agent`` is never
mutated, so one instance can be shared safely across concurrent asyncio
tasks — each task obtains its own ``BoundAgent`` whose child emitter is
constructed in that task (so ``InMemoryEmitter``'s span-stack
``ContextVar`` is set in the right task context).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nanitics.infrastructure.observability.emitter import EventEmitter

if TYPE_CHECKING:
    from nanitics.strategies.agents.base import Agent, AgentInput, AgentResult


@dataclass(frozen=True)
class RunContext:
    """Per-invocation context threaded through :meth:`Agent._run_with_context`.

    Carries the child emitter for the run. Held by a :class:`BoundAgent`
    and installed onto per-task ``ContextVar`` slots for the duration of
    the call so methods reading ``self._emitter`` observe the bound
    emitter without any shared-state mutation on the agent.
    """

    emitter: EventEmitter


class BoundAgent:
    """A per-invocation binding of an ``Agent`` to a parent trace context.

    Returned by :meth:`Agent.bind`. Holds a child emitter constructed in
    the calling task and drives the agent's reasoning loop via
    :meth:`run`. The underlying ``Agent`` is never mutated — multiple
    ``BoundAgent`` handles for the same agent can run concurrently in
    independent tasks without interference.

    A ``BoundAgent`` is single-task: construct and call ``run`` in the
    same task. Sharing one handle across tasks is undefined.
    """

    __slots__ = ("_agent", "_ctx")

    def __init__(self, agent: Agent, ctx: RunContext) -> None:
        self._agent = agent
        self._ctx = ctx

    @property
    def agent(self) -> Agent:
        """The underlying (unmutated) agent."""
        return self._agent

    @property
    def emitter(self) -> EventEmitter:
        """The child emitter this handle runs the agent under."""
        return self._ctx.emitter

    async def run(self, input: AgentInput) -> AgentResult:
        """Run the agent under this handle's child emitter."""
        return await self._agent._run_with_context(input, self._ctx)
