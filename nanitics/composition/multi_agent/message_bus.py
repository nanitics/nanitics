from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from nanitics.core.agents.base import Agent, AgentResult
from nanitics.core.agents.context import ContextContent
from nanitics.core.tools.context import ToolContext
from nanitics.core.tools.function_tool import FunctionTool, tool
from nanitics.infrastructure.llm.protocol import Message
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import (
    MessageBusCompleteEvent,
    MessageBusStartEvent,
    MessageDeliveredEvent,
    MessagePublishedEvent,
)
from nanitics.safety.cancellation import CancellationToken

# --- Data Models ---


class BusMessage(BaseModel):
    """A message on the bus, published to a topic by an agent or the system.

    Attributes:
        message_id: Unique identifier (auto-generated UUID).
        topic: Topic this message is published to.
        content: Message content.
        author: Name of the publishing agent or ``"system"`` for seed messages.
        depth: Chain depth. Seed messages are 0; published responses are
            parent depth + 1.
        parent_message_id: ID of the message that triggered this one.
        metadata: Arbitrary metadata.
        timestamp: When the message was created (UTC).
    """

    model_config = ConfigDict(frozen=True)

    message_id: str = Field(default_factory=lambda: str(uuid4()))
    topic: str
    content: str
    author: str
    depth: int = 0
    parent_message_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


@runtime_checkable
class MessageFilter(Protocol):
    """Protocol for filtering which messages trigger a subscription."""

    async def match(self, message: BusMessage) -> bool: ...


class TopicSubscription(BaseModel):
    """Maps an agent to the topics it subscribes to, with an optional filter.

    Attributes:
        agent: Subscribing agent.
        topics: Topics this agent listens to.
        filter: Optional filter to further narrow which messages trigger
            this subscription.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    agent: Agent = Field(exclude=True)
    topics: list[str]
    filter: MessageFilter | None = Field(default=None, exclude=True)


class AgentExecution(BaseModel):
    """Records a single agent's processing of a bus message.

    Attributes:
        agent_name: Which agent ran.
        trigger_message_id: Message that triggered this execution.
        output: Agent's output.
        published_messages: Messages published during this execution.
        steps: Number of agent loop steps taken.
        termination_reason: Why the agent stopped.
    """

    model_config = ConfigDict(frozen=True)

    agent_name: str
    trigger_message_id: str
    output: Any
    published_messages: list[BusMessage]
    steps: int
    termination_reason: str


@dataclass(frozen=True)
class FailedExecution:
    """Captures a subscriber failure during message bus processing."""

    agent_name: str
    trigger_message_id: str
    error_type: str
    error_message: str


class MessageBusResult(BaseModel):
    """Result of a message bus execution.

    Attributes:
        messages: All messages processed.
        executions: All agent executions.
        failed_executions: Subscriber failures that occurred during processing.
        total_messages: Total message count.
        total_executions: Total execution count.
        termination_reason: Why processing ended (``"quiescence"``,
            ``"max_messages"``, ``"cancelled"``, or termination class name).
    """

    model_config = ConfigDict(frozen=True)

    messages: list[BusMessage]
    executions: list[AgentExecution]
    failed_executions: list[FailedExecution] = []
    total_messages: int
    total_executions: int
    termination_reason: str


# --- Termination Conditions ---


class BusState(BaseModel):
    """Snapshot of bus state, passed to termination conditions.

    Attributes:
        total_messages: Messages processed so far.
        total_executions: Agent executions so far.
        max_depth_reached: Deepest message chain depth.
        message_log: All messages.
        execution_log: All executions.
    """

    model_config = ConfigDict(frozen=True)

    total_messages: int
    total_executions: int
    max_depth_reached: int
    message_log: list[BusMessage]
    execution_log: list[AgentExecution]


@runtime_checkable
class BusTerminationCondition(Protocol):
    """Protocol for deciding when the message bus should stop processing."""

    async def should_terminate(self, state: BusState) -> bool: ...


class MaxMessagesTermination:
    """Terminates when total messages reaches a threshold.

    Args:
        max_messages: Maximum number of messages before termination.
    """

    def __init__(self, max_messages: int) -> None:
        if max_messages <= 0:
            raise ValueError("max_messages must be positive")
        self._max_messages = max_messages

    async def should_terminate(self, state: BusState) -> bool:
        return state.total_messages >= self._max_messages


class MaxExecutionsTermination:
    """Terminates when total agent executions reaches a threshold.

    Args:
        max_executions: Maximum number of agent executions before termination.
    """

    def __init__(self, max_executions: int) -> None:
        if max_executions <= 0:
            raise ValueError("max_executions must be positive")
        self._max_executions = max_executions

    async def should_terminate(self, state: BusState) -> bool:
        return state.total_executions >= self._max_executions


class BusPredicateTermination:
    """Terminates when a custom async predicate returns True.

    Args:
        predicate: Async callable that receives ``BusState`` and returns
            True to terminate.
    """

    def __init__(self, predicate: Callable[[BusState], Awaitable[bool]]) -> None:
        self._predicate = predicate

    async def should_terminate(self, state: BusState) -> bool:
        return await self._predicate(state)


class BusCompositeTermination:
    """Combines multiple termination conditions.

    Args:
        conditions: List of termination conditions to evaluate.
        mode: ``"any"`` (default) terminates when any condition fires;
            ``"all"`` requires all conditions to fire.
    """

    def __init__(
        self,
        conditions: list[BusTerminationCondition],
        mode: str = "any",
    ) -> None:
        self._conditions = conditions
        self._mode = mode

    async def should_terminate(self, state: BusState) -> bool:
        results = [await c.should_terminate(state) for c in self._conditions]
        if self._mode == "any":
            return any(results)
        return all(results)


# --- Publishing Tools ---


def create_bus_tools(
    outbox: list[BusMessage],
    agent_name: str,
    emitter: EventEmitter | None = None,
) -> list[FunctionTool]:
    """Create the ``publish_message`` tool for an agent on the bus.

    Published messages are appended to the outbox and emitted as
    ``MessagePublishedEvent``s. The bus controller reads the outbox
    after the agent completes.

    Args:
        outbox: Mutable list where published messages are collected.
        agent_name: Name of the publishing agent.
        emitter: Event emitter for publish events.
    """

    @tool(
        name="publish_message",
        description=(
            "Publish a message to a topic on the message bus. "
            "Other agents subscribed to that topic will receive and react to your message. "
            "Use this to share findings, ask questions, or trigger downstream work."
        ),
    )
    async def publish_message(
        topic: str,
        content: str,
        context: ToolContext,
    ) -> str:
        msg = BusMessage(
            topic=topic,
            content=content,
            author=agent_name,
        )
        outbox.append(msg)
        em = context.emitter if context is not None else emitter
        if em is not None:
            em.emit(
                MessagePublishedEvent(
                    trace_id=em.trace_id,
                    span_id=em.span_id,
                    parent_span_id=em.parent_span_id,
                    message_id=msg.message_id,
                    topic=topic,
                    author=agent_name,
                    content=content,
                    depth=msg.depth,
                )
            )
        return f"Published message {msg.message_id} to topic '{topic}'."

    return [publish_message]


# --- Context Provider ---


class MessageHistoryProvider:
    """Context provider that injects recent bus messages into an agent's context.

    Shows messages from subscribed topics, grouped by topic with relative
    timestamps.

    Args:
        message_log: Shared message log from the bus.
        subscribed_topics: Topics this agent subscribes to.
        max_messages: Maximum number of recent messages to include.
    """

    def __init__(
        self,
        message_log: list[BusMessage],
        subscribed_topics: list[str],
        max_messages: int = 20,
    ) -> None:
        self._message_log = message_log
        self._subscribed_topics = subscribed_topics
        self._max_messages = max_messages

    async def provide(self, messages: list[Message]) -> ContextContent | None:
        relevant = [m for m in self._message_log if m.topic in self._subscribed_topics]
        if not relevant:
            return None

        recent = relevant[-self._max_messages :]

        by_topic: dict[str, list[BusMessage]] = {}
        for m in recent:
            by_topic.setdefault(m.topic, []).append(m)

        lines = ["## Message Bus — Recent Messages", ""]
        now = datetime.now(UTC)
        for topic, msgs in by_topic.items():
            lines.append(f"### Topic: {topic}")
            for m in msgs:
                delta = now - m.timestamp
                seconds = int(delta.total_seconds())
                if seconds < 60:
                    ago = f"{seconds}s ago"
                elif seconds < 3600:
                    ago = f"{seconds // 60}m ago"
                else:
                    ago = f"{seconds // 3600}h ago"
                lines.append(f"[{m.author}, {ago}] {m.content[:200]}")
            lines.append("")

        return ContextContent(
            provider_name="MessageHistoryProvider",
            content="\n".join(lines),
            priority=5,
            protected=False,
        )


# --- System Prompt Contributor ---


class MessageBusContributor:
    """System prompt contributor that describes the bus to a participating agent.

    Adds a section listing subscribed and available topics, the
    ``publish_message`` tool, and usage guidelines.

    Args:
        subscribed_topics: Topics this agent subscribes to.
        all_topics: All topics available on the bus.
    """

    def __init__(
        self,
        subscribed_topics: list[str],
        all_topics: list[str],
    ) -> None:
        self._subscribed_topics = subscribed_topics
        self._all_topics = all_topics

    def system_prompt_section(self) -> tuple[str, str] | None:
        subscribed = ", ".join(self._subscribed_topics)
        available = ", ".join(self._all_topics)
        return (
            "message_bus",
            (
                "You are participating in a message bus with other agents.\n\n"
                f"**Your subscribed topics:** {subscribed}\n"
                f"**All available topics:** {available}\n\n"
                "Use the `publish_message` tool to send messages to any topic. "
                "Other agents subscribed to that topic will receive your message and may react.\n\n"
                "**Guidelines:**\n"
                "- Publish when you have findings, questions, or requests for other agents\n"
                "- Your task is complete when you have processed the incoming message and "
                "published any relevant outputs\n"
                "- Do not publish unless you have meaningful content to share"
            ),
        )


# --- MessageBus Controller ---


class MessageBus:
    """Topic-based publish-subscribe communication layer for agents.

    Agents subscribe to topics. Seed messages initiate processing; each
    message triggers subscribed agents concurrently. Agents can publish
    new messages that continue the reactive chain.

    Processing ends when the queue is empty (quiescence), a termination
    condition fires, the ``max_messages`` safety cap is reached, or
    cancellation is requested.

    By default, a subscriber whose ``agent.name`` equals the message
    ``author`` does not receive its own publish; set
    ``allow_self_delivery=True`` to restore broadcast-to-self for
    legitimate reactive patterns (polling loops, self-reflection chains,
    echo topologies). Self-delivery suppression filters at the routing
    layer, so ``MessageDeliveredEvent`` is never emitted for the
    suppressed delivery.

    Args:
        subscriptions: Agent-to-topic mappings.
        emitter: Event emitter for bus tracing.
        termination: Optional condition for when to stop processing.
        max_messages: Hard safety cap on total messages processed.
        max_depth: Maximum message chain depth.
        cancellation_token: Cancellation signal.
        allow_self_delivery: When ``False`` (default), a subscriber
            whose ``agent.name`` equals the message ``author`` is
            excluded from delivery for that message. When ``True``, the
            subscriber is included and may re-execute on its own
            publish.
    """

    def __init__(
        self,
        *,
        subscriptions: list[TopicSubscription],
        emitter: EventEmitter,
        termination: BusTerminationCondition | None = None,
        max_messages: int = 100,
        max_depth: int = 10,
        cancellation_token: CancellationToken | None = None,
        allow_self_delivery: bool = False,
    ) -> None:
        self._subscriptions = subscriptions
        self._emitter = emitter
        self._termination = termination
        self._max_messages = max_messages
        self._max_depth = max_depth
        self._cancellation_token = cancellation_token
        self._allow_self_delivery = allow_self_delivery

        # Build routing table: topic -> list of (subscription,) entries
        self._routing: dict[str, list[TopicSubscription]] = {}
        for sub in subscriptions:
            for topic in sub.topics:
                self._routing.setdefault(topic, []).append(sub)

        self._all_topics = sorted(self._routing.keys())

    async def run(self, seed_messages: list[BusMessage]) -> MessageBusResult:
        """Process seed messages through the bus until termination.

        Args:
            seed_messages: Initial messages to start processing. Must not
                be empty.

        Returns:
            ``MessageBusResult`` with all messages, executions, and the
            termination reason.

        Raises:
            ValueError: If ``seed_messages`` is empty.
        """
        if not seed_messages:
            raise ValueError("seed_messages must not be empty")

        # Build subscription map for event: topic -> agent names
        subscription_map: dict[str, list[str]] = {}
        for sub in self._subscriptions:
            for topic in sub.topics:
                subscription_map.setdefault(topic, []).append(sub.agent.name)

        with self._emitter.span("message_bus"):
            self._emitter.emit(
                MessageBusStartEvent(
                    trace_id=self._emitter.trace_id,
                    span_id=self._emitter.span_id,
                    parent_span_id=self._emitter.parent_span_id,
                    seed_topics=list({m.topic for m in seed_messages}),
                    seed_count=len(seed_messages),
                    subscriber_count=len(self._subscriptions),
                    subscriptions=subscription_map,
                    max_messages=self._max_messages,
                    max_depth=self._max_depth,
                )
            )

            message_log: list[BusMessage] = []
            execution_log: list[AgentExecution] = []
            failed_execution_log: list[FailedExecution] = []
            agent_execution_counts: dict[str, int] = {}
            max_depth_reached = 0
            termination_reason = "quiescence"

            queue: list[BusMessage] = list(seed_messages)

            while queue:
                message = queue.pop(0)
                message_log.append(message)
                max_depth_reached = max(max_depth_reached, message.depth)

                # Find subscribers for this topic
                subs = self._routing.get(message.topic, [])
                if not subs:
                    continue

                # Apply filters and collect matching subscribers
                matching_subs: list[TopicSubscription] = []
                for sub in subs:
                    if sub.filter is not None:
                        if not await sub.filter.match(message):
                            continue
                    # Self-delivery suppression: a subscriber does not receive
                    # its own publish unless explicitly opted in. The filter
                    # runs per-subscriber-per-message so other subscribers on
                    # the same topic are unaffected.
                    if not self._allow_self_delivery and sub.agent.name == message.author:
                        continue
                    matching_subs.append(sub)

                if not matching_subs:
                    continue

                # Execute matching subscribers concurrently
                results = await asyncio.gather(
                    *(self._run_subscriber(sub, message, message_log) for sub in matching_subs)
                )

                # Partition results into executions and failures
                for result_item in results:
                    if isinstance(result_item, FailedExecution):
                        failed_execution_log.append(result_item)
                        continue
                    execution_log.append(result_item)
                    agent_execution_counts[result_item.agent_name] = (
                        agent_execution_counts.get(result_item.agent_name, 0) + 1
                    )
                    queue.extend(
                        pub_msg for pub_msg in result_item.published_messages if pub_msg.depth <= self._max_depth
                    )

                # Check max_messages safety bound
                if len(message_log) >= self._max_messages:
                    termination_reason = "max_messages"
                    break

                # Check cancellation
                if self._cancellation_token is not None and self._cancellation_token.is_cancelled:
                    termination_reason = "cancelled"
                    break

                # Check termination condition
                if self._termination is not None:
                    state = BusState(
                        total_messages=len(message_log),
                        total_executions=len(execution_log),
                        max_depth_reached=max_depth_reached,
                        message_log=message_log,
                        execution_log=execution_log,
                    )
                    if await self._termination.should_terminate(state):
                        termination_reason = type(self._termination).__name__
                        if isinstance(self._termination, BusCompositeTermination):
                            for cond in self._termination._conditions:
                                if await cond.should_terminate(state):
                                    termination_reason = type(cond).__name__
                                    break
                        break

            self._emitter.emit(
                MessageBusCompleteEvent(
                    trace_id=self._emitter.trace_id,
                    span_id=self._emitter.span_id,
                    parent_span_id=self._emitter.parent_span_id,
                    total_messages=len(message_log),
                    total_executions=len(execution_log),
                    failed_executions=len(failed_execution_log),
                    max_depth_reached=max_depth_reached,
                    termination_reason=termination_reason,
                    agent_execution_counts=agent_execution_counts,
                )
            )

            return MessageBusResult(
                messages=message_log,
                executions=execution_log,
                failed_executions=failed_execution_log,
                total_messages=len(message_log),
                total_executions=len(execution_log),
                termination_reason=termination_reason,
            )

    async def _run_subscriber(
        self,
        sub: TopicSubscription,
        message: BusMessage,
        message_log: list[BusMessage],
    ) -> AgentExecution | FailedExecution:
        agent = sub.agent
        outbox: list[BusMessage] = []

        # Create per-execution tools and providers
        bus_tools = create_bus_tools(outbox, agent.name, self._emitter)
        history_provider = MessageHistoryProvider(
            message_log=message_log,
            subscribed_topics=sub.topics,
        )
        contributor = MessageBusContributor(
            subscribed_topics=sub.topics,
            all_topics=self._all_topics,
        )

        # Temporarily inject tools, context provider, and system prompt section
        registry = agent._tool_registry  # type: ignore[attr-defined]
        for t in bus_tools:
            registry.register(t)

        original_providers = agent._context_providers
        if original_providers is None:
            agent._context_providers = [history_provider]
        else:
            agent._context_providers = [*original_providers, history_provider]

        original_prompt = agent._system_prompt
        section = contributor.system_prompt_section()
        if section is not None:
            agent._system_prompt = original_prompt + "\n\n## " + section[0] + "\n" + section[1]

        try:
            # Per-subscriber context-provider and prompt injection above
            # still mutates the shared agent; concurrent subscribers
            # racing on that state are a known hazard (not yet addressed).
            # The bind call itself is non-mutating under the new contract.
            result: AgentResult = await agent.bind(self._emitter).run(message.content)

            # Set depth and parent on published messages
            published = [
                msg.model_copy(
                    update={
                        "depth": message.depth + 1,
                        "parent_message_id": message.message_id,
                    }
                )
                for msg in outbox
            ]

            execution = AgentExecution(
                agent_name=agent.name,
                trigger_message_id=message.message_id,
                output=result.output,
                published_messages=published,
                steps=result.total_steps,
                termination_reason=result.termination_reason,
            )

            self._emitter.emit(
                MessageDeliveredEvent(
                    trace_id=self._emitter.trace_id,
                    span_id=self._emitter.span_id,
                    parent_span_id=self._emitter.parent_span_id,
                    message_id=message.message_id,
                    topic=message.topic,
                    agent_name=agent.name,
                    output=result.output or "",
                    steps=result.total_steps,
                    messages_published=len(published),
                )
            )

            return execution

        except Exception as exc:
            self._emitter.emit(
                MessageDeliveredEvent(
                    trace_id=self._emitter.trace_id,
                    span_id=self._emitter.span_id,
                    parent_span_id=self._emitter.parent_span_id,
                    message_id=message.message_id,
                    topic=message.topic,
                    agent_name=agent.name,
                    output="",
                    steps=0,
                    messages_published=0,
                    error=str(exc),
                )
            )
            return FailedExecution(
                agent_name=agent.name,
                trigger_message_id=message.message_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

        finally:
            # Restore agent state
            for t in bus_tools:
                registry._tools.pop(t.schema.name, None)
            agent._context_providers = original_providers
            agent._system_prompt = original_prompt
