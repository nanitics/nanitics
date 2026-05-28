from __future__ import annotations

import threading
from typing import Any

from pydantic import BaseModel, ConfigDict

from nanitics.composition.threads.store import ThreadStore
from nanitics.infrastructure.errors import AgentError
from nanitics.infrastructure.llm.protocol import LLMClient, ToolSchema
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import (
    PeerConsultationEvent,
    PeerNetworkCompleteEvent,
    PeerNetworkStartEvent,
)
from nanitics.safety.cancellation import CancellationToken
from nanitics.strategies.agents.base import AgentResult
from nanitics.strategies.agents.react import ReActAgent
from nanitics.strategies.tools.protocol import Tool, ToolResult


class PeerSpec(BaseModel):
    """Specification for a peer agent in a ``PeerNetwork``.

    Attributes:
        name: Unique peer name. Used as the identifier in
            ``consult_<name>`` tools.
        description: What this peer does. Shown to other peers so they
            know when to consult it.
        llm_client: LLM client for this peer.
        system_prompt: Base system prompt (augmented with peer info at
            network creation).
        tools: This peer's own tools (consultation tools are added
            automatically).
        max_iterations: Max agent loop iterations per consultation.
        allowed_peers: Structural consultation graph for this peer. ``None``
            (the default) means "all other peers in the network except
            self" — i.e., a fully-connected graph minus self-loops.
            An explicit list names exactly the peers this one can consult
            (a ``consult_<name>`` tool is injected for each). An empty
            list declares a leaf consultant that cannot consult anyone.
            ``PeerNetwork.__init__`` validates the list: including
            ``name`` (self-reference) raises ``ValueError``; naming a
            peer not present in the network also raises ``ValueError``.
            ``consult_<self>`` is never injected under any configuration.
        thread_key: Opaque key identifying this peer's conversation
            thread. When set, every consultation of this peer — whether
            entry-point or via ``consult_<name>`` — forwards the key,
            so the peer accumulates its prior turns across the
            network's lifetime. Per-peer-identity is the default
            scoping; per-pair or per-network scoping is deferred to a
            follow-up if real consumers report a need. The peer's
            underlying ``ReActAgent`` must be wired with a
            :class:`~nanitics.composition.threads.ThreadStore` for the
            prefix to be persisted, which the network does by accepting
            an optional ``thread_store`` constructor argument.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    llm_client: LLMClient
    system_prompt: str
    tools: list[Tool]
    max_iterations: int = 10
    allowed_peers: list[str] | None = None
    thread_key: str | None = None


class PeerBudgetExceededError(AgentError):
    """Raised when the shared peer consultation budget is exhausted.

    Attributes:
        invocations_used: Total invocations consumed.
        max_invocations: Budget limit.
    """

    invocations_used: int
    max_invocations: int

    def __init__(
        self,
        *,
        invocations_used: int,
        max_invocations: int,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> None:
        super().__init__(
            f"Peer consultation budget exhausted ({invocations_used}/{max_invocations} used). "
            "You cannot consult any more peers — produce your final answer now.",
            trace_id=trace_id,
            span_id=span_id,
        )
        self.invocations_used = invocations_used
        self.max_invocations = max_invocations


class InvocationBudget:
    """Thread-safe counter that tracks shared consultation budget.

    Args:
        max_invocations: Maximum allowed consultations across the network.
    """

    def __init__(self, max_invocations: int) -> None:
        if max_invocations <= 0:
            raise ValueError("max_invocations must be positive")
        self._max = max_invocations
        self._used = 0
        self._lock = threading.Lock()

    @property
    def remaining(self) -> int:
        with self._lock:
            return self._max - self._used

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    def consume(self) -> int:
        """Consume one invocation. Returns the consultation number (1-indexed).

        Raises PeerBudgetExceededError if budget is exhausted.
        """
        with self._lock:
            if self._used >= self._max:
                raise PeerBudgetExceededError(
                    invocations_used=self._used,
                    max_invocations=self._max,
                )
            self._used += 1
            return self._used


class PeerTool:
    """Tool that lets one peer consult another in the network.

    Exposes a ``consult_<peer_name>`` tool with a single ``message``
    parameter. Consumes one invocation from the shared budget per call.
    """

    def __init__(
        self,
        *,
        peer_name: str,
        peer_description: str,
        caller_name: str,
        registry: dict[str, ReActAgent],
        budget: InvocationBudget,
        emitter: EventEmitter,
        consulted: set[str],
        thread_key: str | None = None,
    ) -> None:
        self._peer_name = peer_name
        self._peer_description = peer_description
        self._caller_name = caller_name
        self._registry = registry
        self._budget = budget
        self._emitter = emitter
        self._consulted = consulted
        self._thread_key = thread_key

    @property
    def schema(self) -> ToolSchema:
        """Tool schema with a single ``message`` string parameter."""
        return ToolSchema(
            name=f"consult_{self._peer_name}",
            description=self._peer_description,
            parameters={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The question or request to send to this peer.",
                    },
                },
                "required": ["message"],
            },
        )

    async def execute(self, **params: Any) -> ToolResult:
        """Consult the peer agent with the given message.

        Consumes one invocation from the shared budget. Emits a
        ``PeerConsultationEvent``. Returns a ``ToolResult`` with an
        error message if the budget is exhausted.
        """
        message: str = params["message"]

        try:
            consultation_number = self._budget.consume()
        except PeerBudgetExceededError as exc:
            return ToolResult(content=str(exc))

        agent = self._registry[self._peer_name]
        self._consulted.add(self._peer_name)

        self._emitter.emit(
            PeerConsultationEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                from_agent=self._caller_name,
                to_agent=self._peer_name,
                message=message,
                consultation_number=consultation_number,
                remaining_budget=self._budget.remaining,
            )
        )

        result = await agent.bind(self._emitter).run(message, thread_key=self._thread_key)

        return ToolResult(
            content=result.output or "",
            metadata={
                "total_steps": result.total_steps,
                "termination_reason": result.termination_reason,
            },
        )


def _augment_system_prompt(
    base_prompt: str,
    peers: list[tuple[str, str]],
) -> str:
    if not peers:
        # Leaf consultant — no roster to render; prompt stays clean rather
        # than advertising an empty "Available Peers" block.
        return base_prompt
    lines = ["\n\n## Available Peers\n"]
    lines.append("You can consult the following peers using the consult tools:\n")
    for name, description in peers:
        lines.append(f"- **{name}**: {description}")
    lines.append("\nUse these when you need expertise outside your domain. Peers can also consult you.")
    lines.append("You have a shared consultation budget — use it wisely.")
    return base_prompt + "\n".join(lines)


def _resolve_allowed_peers(
    spec: PeerSpec,
    all_peer_names: set[str],
) -> list[str]:
    """Resolve ``spec.allowed_peers`` into the concrete peer-name list.

    Default (``None``) expands to "every peer name except ``spec.name``".
    An explicit list is validated — self-reference and unknown peer
    names raise ``ValueError``. An empty list is valid and yields an
    empty result (leaf consultant).
    """
    if spec.allowed_peers is None:
        return sorted(name for name in all_peer_names if name != spec.name)

    for candidate in spec.allowed_peers:
        if candidate == spec.name:
            raise ValueError(
                f"PeerSpec for {spec.name!r} cannot list itself in allowed_peers; self-consultation is not supported."
            )
        if candidate not in all_peer_names:
            raise ValueError(
                f"PeerSpec for {spec.name!r} lists unknown peer {candidate!r} in "
                f"allowed_peers; known peers are {sorted(all_peer_names)}."
            )
    return list(spec.allowed_peers)


class PeerNetwork:
    """Peer-to-peer consultation network with shared invocation budget.

    Creates ``ReActAgent`` instances for each peer, augmenting their
    system prompts with peer descriptions and adding ``consult_<peer>``
    tools. The consultation graph is declared structurally per peer via
    ``PeerSpec.allowed_peers`` — defaulting to "all other peers in the
    network minus self" when unset. ``consult_<self>`` is never
    injected on any peer, under any configuration. All peers share a
    single invocation budget that prevents runaway recursion.

    Args:
        peers: Peer specifications. Names must be unique.
        emitter: Event emitter for network tracing.
        max_invocations: Total consultation budget shared across all peers.
        cancellation_token: Cancellation signal propagated to all peer agents.
        thread_store: Shared :class:`~nanitics.composition.threads.ThreadStore`
            wired into every peer ``ReActAgent`` so peers with a
            ``thread_key`` accumulate behavioral continuity across
            consultations. When ``None``, peer ``thread_key`` values
            are accepted but no prefix is persisted. Per-peer-identity
            is the default scoping — each peer carries its own thread,
            regardless of who called it.

    Raises:
        ValueError: If peer names are not unique, or if any ``PeerSpec``'s
            ``allowed_peers`` lists the peer's own name (self-reference)
            or a name not present in the network (unknown peer).
    """

    def __init__(
        self,
        peers: list[PeerSpec],
        emitter: EventEmitter,
        max_invocations: int = 50,
        cancellation_token: CancellationToken | None = None,
        thread_store: ThreadStore | None = None,
    ) -> None:
        names = [p.name for p in peers]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate peer names: {names}")

        self._emitter = emitter
        self._max_invocations = max_invocations
        self._registry: dict[str, ReActAgent] = {}
        self._budget = InvocationBudget(max_invocations)
        self._peer_specs = {p.name: p for p in peers}
        self._consulted: set[str] = set()

        all_peer_names = set(names)
        peer_descriptions = {p.name: p.description for p in peers}

        # Validate every spec's allowed_peers up front so a construction
        # error never leaves a partially-wired network behind.
        resolved_allowed: dict[str, list[str]] = {
            spec.name: _resolve_allowed_peers(spec, all_peer_names) for spec in peers
        }

        for spec in peers:
            allowed = resolved_allowed[spec.name]

            peer_tools: list[Tool] = [
                PeerTool(
                    peer_name=other_name,
                    peer_description=peer_descriptions[other_name],
                    caller_name=spec.name,
                    registry=self._registry,
                    budget=self._budget,
                    emitter=emitter,
                    consulted=self._consulted,
                    thread_key=self._peer_specs[other_name].thread_key,
                )
                for other_name in allowed
            ]

            augmented_prompt = _augment_system_prompt(
                spec.system_prompt,
                [(name, peer_descriptions[name]) for name in allowed],
            )

            agent = ReActAgent(
                name=spec.name,
                llm_client=spec.llm_client,
                emitter=emitter,
                system_prompt=augmented_prompt,
                tools=[*spec.tools, *peer_tools],
                max_iterations=spec.max_iterations,
                cancellation_token=cancellation_token,
                thread_store=thread_store,
            )

            self._registry[spec.name] = agent

    async def run(self, agent_name: str, task: str, *, thread_key: str | None = None) -> AgentResult:
        """Start execution from a specific peer agent.

        Args:
            agent_name: Name of the entry-point agent.
            task: Task to execute.
            thread_key: Optional override for the entry agent's thread
                key. When ``None`` (the default) the entry peer's
                ``PeerSpec.thread_key`` is used, so per-peer-identity
                accumulation continues across repeated network runs.
                Pass an explicit value to override on a per-network-run
                basis (e.g., per-session keys layered over per-peer
                identity).

        Returns:
            The entry agent's ``AgentResult``.

        Raises:
            ValueError: If ``agent_name`` is not in the network.
        """
        if agent_name not in self._registry:
            raise ValueError(
                f"Agent '{agent_name}' not found in peer network. Available: {list(self._registry.keys())}"
            )

        effective_key = thread_key if thread_key is not None else self._peer_specs[agent_name].thread_key

        self._emitter.emit(
            PeerNetworkStartEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                task=task,
                entry_agent=agent_name,
                peer_names=list(self._registry.keys()),
                peer_descriptions={name: spec.description for name, spec in self._peer_specs.items()},
                max_invocations=self._max_invocations,
            )
        )

        result = await self._registry[agent_name].bind(self._emitter).run(task, thread_key=effective_key)

        self._emitter.emit(
            PeerNetworkCompleteEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                entry_agent=agent_name,
                total_consultations=self._budget.used,
                invocations_used=self._budget.used,
                agents_consulted=sorted(self._consulted),
                termination_reason=result.termination_reason,
            )
        )

        return result
