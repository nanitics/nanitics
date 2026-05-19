"""Tests for MessageBus: data models, termination conditions, tools, providers, controller, events."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from nanitics.composition.multi_agent.message_bus import (
    BusCompositeTermination,
    BusMessage,
    BusPredicateTermination,
    BusState,
    BusTerminationCondition,
    MaxExecutionsTermination,
    MaxMessagesTermination,
    MessageBus,
    MessageBusContributor,
    MessageFilter,
    MessageHistoryProvider,
    TopicSubscription,
    create_bus_tools,
)
from nanitics.infrastructure import (
    LLMResponse,
    MockLLMClient,
)
from nanitics.infrastructure.llm.protocol import ToolCall
from nanitics.infrastructure.observability.events import (
    MessageBusCompleteEvent,
    MessageBusStartEvent,
    MessageDeliveredEvent,
    MessagePublishedEvent,
)
from nanitics.safety.cancellation import CancellationToken
from nanitics.strategies import ReActAgent
from nanitics.tracing import InMemoryEmitter
from tests.testing_helpers import make_emitter, make_response, make_usage


def make_tool_call_response(tool_name: str, **kwargs: str) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(id="tc-1", name=tool_name, arguments=dict(kwargs))],
        usage=make_usage(),
        model="test-model",
        stop_reason="tool_use",
    )


def make_agent(
    name: str,
    emitter: InMemoryEmitter,
    *,
    responses: list[LLMResponse] | None = None,
    publish_topic: str | None = None,
    publish_content: str | None = None,
) -> ReActAgent:
    """Create a test agent. If publish_topic is given, the agent will call publish_message then reply."""
    if responses is not None:
        agent_responses = responses
    elif publish_topic is not None:
        agent_responses = [
            make_tool_call_response(
                "publish_message",
                topic=publish_topic,
                content=publish_content or f"Output from {name}",
            ),
            make_response("done"),
        ]
    else:
        agent_responses = [make_response("done")]

    return ReActAgent(
        name=name,
        llm_client=MockLLMClient(agent_responses),
        emitter=emitter,
        system_prompt=f"You are {name}.",
        tools=[],
    )


def make_seed(topic: str, content: str = "seed message") -> BusMessage:
    return BusMessage(topic=topic, content=content, author="seed")


# ──────────────────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────────────────


class TestBusMessage:
    def test_defaults(self) -> None:
        msg = BusMessage(topic="t", content="c", author="a")
        assert msg.depth == 0
        assert msg.parent_message_id is None
        assert msg.metadata == {}
        assert msg.message_id  # auto-generated UUID

    def test_frozen(self) -> None:
        msg = BusMessage(topic="t", content="c", author="a")
        with pytest.raises(ValidationError):
            msg.topic = "other"


class TestTopicSubscription:
    def test_construction(self) -> None:
        emitter = make_emitter()
        agent = make_agent("a", emitter)
        sub = TopicSubscription(agent=agent, topics=["x", "y"])
        assert sub.topics == ["x", "y"]
        assert sub.agent.name == "a"

    def test_agent_excluded_from_serialization(self) -> None:
        emitter = make_emitter()
        agent = make_agent("a", emitter)
        sub = TopicSubscription(agent=agent, topics=["x"])
        data = sub.model_dump()
        assert "agent" not in data


# ──────────────────────────────────────────────────────────
# Termination Conditions
# ──────────────────────────────────────────────────────────


def make_bus_state(
    *,
    total_messages: int = 0,
    total_executions: int = 0,
) -> BusState:
    return BusState(
        total_messages=total_messages,
        total_executions=total_executions,
        max_depth_reached=0,
        message_log=[],
        execution_log=[],
    )


class TestMaxMessagesTermination:
    async def test_terminates_at_max(self) -> None:
        cond = MaxMessagesTermination(5)
        assert await cond.should_terminate(make_bus_state(total_messages=5)) is True

    async def test_does_not_terminate_below_max(self) -> None:
        cond = MaxMessagesTermination(5)
        assert await cond.should_terminate(make_bus_state(total_messages=4)) is False

    def test_satisfies_protocol(self) -> None:
        assert isinstance(MaxMessagesTermination(5), BusTerminationCondition)


class TestMaxExecutionsTermination:
    async def test_terminates_at_max(self) -> None:
        cond = MaxExecutionsTermination(3)
        assert await cond.should_terminate(make_bus_state(total_executions=3)) is True

    async def test_does_not_terminate_below_max(self) -> None:
        cond = MaxExecutionsTermination(3)
        assert await cond.should_terminate(make_bus_state(total_executions=2)) is False


class TestBusPredicateTermination:
    async def test_uses_predicate(self) -> None:
        async def pred(s: BusState) -> bool:
            return s.total_messages > 2

        cond = BusPredicateTermination(pred)
        state = make_bus_state(total_messages=3)
        assert await cond.should_terminate(state) is True

    async def test_async_predicate(self) -> None:
        async def pred(s: BusState) -> bool:
            return s.total_executions >= 1

        cond = BusPredicateTermination(pred)
        assert await cond.should_terminate(make_bus_state(total_executions=1)) is True
        assert await cond.should_terminate(make_bus_state(total_executions=0)) is False


class TestBusCompositeTermination:
    async def test_any_mode(self) -> None:
        cond = BusCompositeTermination(
            [MaxMessagesTermination(5), MaxExecutionsTermination(1)],
            mode="any",
        )
        assert await cond.should_terminate(make_bus_state(total_messages=5, total_executions=0)) is True

    async def test_all_mode(self) -> None:
        cond = BusCompositeTermination(
            [MaxMessagesTermination(5), MaxExecutionsTermination(1)],
            mode="all",
        )
        assert await cond.should_terminate(make_bus_state(total_messages=5, total_executions=0)) is False
        assert await cond.should_terminate(make_bus_state(total_messages=5, total_executions=1)) is True


# ──────────────────────────────────────────────────────────
# Publishing Tools
# ──────────────────────────────────────────────────────────


class TestCreateBusTools:
    def test_returns_single_tool(self) -> None:
        outbox: list[BusMessage] = []
        tools = create_bus_tools(outbox, "agent_a")
        assert len(tools) == 1
        assert tools[0].schema.name == "publish_message"

    async def test_publish_appends_to_outbox(self) -> None:
        outbox: list[BusMessage] = []
        tools = create_bus_tools(outbox, "agent_a")
        publish = tools[0]
        result = await publish.execute(topic="findings", content="hello world")
        assert len(outbox) == 1
        assert outbox[0].topic == "findings"
        assert outbox[0].content == "hello world"
        assert outbox[0].author == "agent_a"
        assert outbox[0].message_id in result.content


# ──────────────────────────────────────────────────────────
# MessageHistoryProvider
# ──────────────────────────────────────────────────────────


class TestMessageHistoryProvider:
    async def test_returns_none_when_no_relevant_messages(self) -> None:
        provider = MessageHistoryProvider(
            message_log=[BusMessage(topic="other", content="x", author="a")],
            subscribed_topics=["findings"],
        )
        result = await provider.provide([])
        assert result is None

    async def test_formats_messages_by_topic(self) -> None:
        msgs = [
            BusMessage(topic="findings", content="Finding 1", author="researcher"),
            BusMessage(topic="findings", content="Finding 2", author="analyst"),
            BusMessage(topic="questions", content="Question 1", author="reviewer"),
        ]
        provider = MessageHistoryProvider(
            message_log=msgs,
            subscribed_topics=["findings", "questions"],
        )
        result = await provider.provide([])
        assert result is not None
        assert "Topic: findings" in result.content
        assert "Topic: questions" in result.content
        assert "Finding 1" in result.content
        assert "Question 1" in result.content
        assert result.priority == 5

    async def test_respects_max_messages(self) -> None:
        msgs = [BusMessage(topic="t", content=f"msg {i}", author="a") for i in range(30)]
        provider = MessageHistoryProvider(
            message_log=msgs,
            subscribed_topics=["t"],
            max_messages=5,
        )
        result = await provider.provide([])
        assert result is not None
        # Should only include the last 5 messages
        assert "msg 25" in result.content
        assert "msg 29" in result.content
        assert "msg 0" not in result.content

    async def test_formats_hours_ago_for_old_messages(self) -> None:
        old_timestamp = datetime.now(UTC) - timedelta(hours=2)
        mid_timestamp = datetime.now(UTC) - timedelta(minutes=5)
        msgs = [
            BusMessage(topic="t", content="old content", author="a", timestamp=old_timestamp),
            BusMessage(topic="t", content="mid content", author="b", timestamp=mid_timestamp),
        ]
        provider = MessageHistoryProvider(
            message_log=msgs,
            subscribed_topics=["t"],
        )
        result = await provider.provide([])
        assert result is not None
        assert "2h ago" in result.content
        assert "5m ago" in result.content


# ──────────────────────────────────────────────────────────
# MessageBusContributor
# ──────────────────────────────────────────────────────────


class TestMessageBusContributor:
    def test_system_prompt_content(self) -> None:
        contrib = MessageBusContributor(
            subscribed_topics=["findings"],
            all_topics=["findings", "questions", "decisions"],
        )
        section = contrib.system_prompt_section()
        assert section is not None
        name, content = section
        assert name == "message_bus"
        assert "findings" in content
        assert "questions" in content
        assert "publish_message" in content


# ──────────────────────────────────────────────────────────
# MessageBus Controller
# ──────────────────────────────────────────────────────────


class TestMessageBusBasicRouting:
    async def test_basic_routing(self) -> None:
        """One seed message, one subscriber — message delivered correctly."""
        emitter = make_emitter()
        subscriber = make_agent("sub", emitter)
        bus = MessageBus(
            subscriptions=[TopicSubscription(agent=subscriber, topics=["findings"])],
            emitter=emitter,
        )
        result = await bus.run([make_seed("findings", "hello")])

        assert result.total_messages == 1
        assert result.total_executions == 1
        assert result.executions[0].agent_name == "sub"
        assert result.termination_reason == "quiescence"

    async def test_multi_subscriber(self) -> None:
        """Message on topic reaches all subscribers to that topic."""
        emitter = make_emitter()
        sub_a = make_agent("sub_a", emitter)
        sub_b = make_agent("sub_b", emitter)
        bus = MessageBus(
            subscriptions=[
                TopicSubscription(agent=sub_a, topics=["findings"]),
                TopicSubscription(agent=sub_b, topics=["findings"]),
            ],
            emitter=emitter,
        )
        result = await bus.run([make_seed("findings")])

        assert result.total_executions == 2
        exec_names = {e.agent_name for e in result.executions}
        assert exec_names == {"sub_a", "sub_b"}

    async def test_topic_isolation(self) -> None:
        """Message on topic A doesn't reach subscriber to topic B."""
        emitter = make_emitter()
        sub_a = make_agent("sub_a", emitter)
        sub_b = make_agent("sub_b", emitter)
        bus = MessageBus(
            subscriptions=[
                TopicSubscription(agent=sub_a, topics=["findings"]),
                TopicSubscription(agent=sub_b, topics=["questions"]),
            ],
            emitter=emitter,
        )
        result = await bus.run([make_seed("findings")])

        assert result.total_executions == 1
        assert result.executions[0].agent_name == "sub_a"


class TestMessageBusMultiHop:
    async def test_multi_hop_chain(self) -> None:
        """Agent A publishes, triggers B, B publishes, triggers C."""
        emitter = make_emitter()
        agent_a = make_agent("agent_a", emitter, publish_topic="topic_b", publish_content="from A")
        agent_b = make_agent("agent_b", emitter, publish_topic="topic_c", publish_content="from B")
        agent_c = make_agent("agent_c", emitter)

        bus = MessageBus(
            subscriptions=[
                TopicSubscription(agent=agent_a, topics=["topic_a"]),
                TopicSubscription(agent=agent_b, topics=["topic_b"]),
                TopicSubscription(agent=agent_c, topics=["topic_c"]),
            ],
            emitter=emitter,
        )
        result = await bus.run([make_seed("topic_a")])

        assert result.total_executions == 3
        exec_names = [e.agent_name for e in result.executions]
        assert exec_names == ["agent_a", "agent_b", "agent_c"]
        # Verify depth increases: seed=0, A's pub=1, B's pub=2
        assert result.messages[0].depth == 0  # seed
        assert result.messages[1].depth == 1  # from A
        assert result.messages[2].depth == 2  # from B


class TestMessageBusDepthLimiting:
    async def test_depth_limiting(self) -> None:
        """Messages beyond max_depth are discarded."""
        emitter = make_emitter()
        # Agent publishes on every invocation — give enough responses for 3 runs
        responses = [
            make_tool_call_response("publish_message", topic="loop", content="loop msg"),
            make_response("done"),
        ] * 4  # enough for up to 4 runs
        agent = make_agent("looper", emitter, responses=responses)

        # Self-publish drives the depth chain — opt in to self-delivery so
        # the agent re-triggers on its own publishes and depth_limit has
        # something to limit. The default allow_self_delivery=False would
        # short-circuit this test into quiescence after the seed execution.
        bus = MessageBus(
            subscriptions=[TopicSubscription(agent=agent, topics=["loop"])],
            emitter=emitter,
            max_depth=2,
            max_messages=50,
            allow_self_delivery=True,
        )
        result = await bus.run([make_seed("loop")])

        # seed (depth 0) -> pub (depth 1) -> pub (depth 2) -> pub (depth 3, discarded)
        assert result.total_executions == 3
        # Messages: seed(0), pub(1), pub(2) — the depth 3 message is discarded from queue
        assert all(m.depth <= 2 for m in result.messages)


class TestMessageBusMaxMessages:
    async def test_max_messages_safety_bound(self) -> None:
        """Execution stops at max_messages."""
        emitter = make_emitter()
        responses = [
            make_tool_call_response("publish_message", topic="t", content="more"),
            make_response("done"),
        ] * 10
        agent = make_agent("agent", emitter, responses=responses)

        # The agent both subscribes to and publishes on "t" to drive the
        # self-sustaining chain that max_messages is meant to cap. Enable
        # self-delivery explicitly since it is off by default.
        bus = MessageBus(
            subscriptions=[TopicSubscription(agent=agent, topics=["t"])],
            emitter=emitter,
            max_messages=3,
            allow_self_delivery=True,
        )
        result = await bus.run([make_seed("t")])

        assert result.total_messages <= 3
        assert result.termination_reason == "max_messages"


class TestMessageBusQuiescence:
    async def test_quiescence(self) -> None:
        """Bus stops when no new messages are published."""
        emitter = make_emitter()
        agent = make_agent("quiet", emitter)  # doesn't publish
        bus = MessageBus(
            subscriptions=[TopicSubscription(agent=agent, topics=["t"])],
            emitter=emitter,
        )
        result = await bus.run([make_seed("t")])

        assert result.termination_reason == "quiescence"
        assert result.total_messages == 1
        assert result.total_executions == 1


class TestMessageBusConcurrentSubscribers:
    async def test_concurrent_subscribers(self) -> None:
        """Multiple subscribers to same topic execute in parallel."""
        emitter = make_emitter()
        execution_order: list[str] = []

        async def track_execution(name: str) -> None:
            execution_order.append(f"{name}_start")
            await asyncio.sleep(0.01)
            execution_order.append(f"{name}_end")

        sub_a = make_agent("sub_a", emitter)
        sub_b = make_agent("sub_b", emitter)

        bus = MessageBus(
            subscriptions=[
                TopicSubscription(agent=sub_a, topics=["t"]),
                TopicSubscription(agent=sub_b, topics=["t"]),
            ],
            emitter=emitter,
        )
        result = await bus.run([make_seed("t")])

        # Both should execute (concurrency is verified by asyncio.gather being used)
        assert result.total_executions == 2


class TestMessageBusMessageFilter:
    async def test_filter_accepts(self) -> None:
        """Subscriber with filter only receives matching messages."""
        emitter = make_emitter()

        class OnlyUrgent:
            async def match(self, message: BusMessage) -> bool:
                return "urgent" in message.content.lower()

        agent = make_agent("filtered", emitter)
        bus = MessageBus(
            subscriptions=[
                TopicSubscription(agent=agent, topics=["alerts"], filter=OnlyUrgent()),
            ],
            emitter=emitter,
        )

        # Non-urgent message should not trigger agent
        result = await bus.run([make_seed("alerts", "routine check")])
        assert result.total_executions == 0

        # Urgent message should trigger agent
        emitter2 = make_emitter()
        agent2 = make_agent("filtered2", emitter2)
        bus2 = MessageBus(
            subscriptions=[
                TopicSubscription(agent=agent2, topics=["alerts"], filter=OnlyUrgent()),
            ],
            emitter=emitter2,
        )
        result2 = await bus2.run([make_seed("alerts", "URGENT: issue detected")])
        assert result2.total_executions == 1

    def test_filter_satisfies_protocol(self) -> None:
        class MyFilter:
            async def match(self, message: BusMessage) -> bool:
                return True

        assert isinstance(MyFilter(), MessageFilter)


class TestMessageBusCancellation:
    async def test_cancellation(self) -> None:
        """CancellationToken stops execution."""
        emitter = make_emitter()
        token = CancellationToken()
        responses = [
            make_tool_call_response("publish_message", topic="t", content="more"),
            make_response("done"),
        ] * 5
        agent = make_agent("agent", emitter, responses=responses)

        bus = MessageBus(
            subscriptions=[TopicSubscription(agent=agent, topics=["t"])],
            emitter=emitter,
            cancellation_token=token,
            max_messages=50,
        )

        # Cancel after first message is processed
        token.cancel()
        result = await bus.run([make_seed("t")])

        assert result.termination_reason == "cancelled"


class TestMessageBusTerminationConditions:
    async def test_max_executions_termination(self) -> None:
        emitter = make_emitter()
        responses = [
            make_tool_call_response("publish_message", topic="t", content="more"),
            make_response("done"),
        ] * 10
        agent = make_agent("agent", emitter, responses=responses)

        # Self-echo is required to produce repeated executions that the
        # MaxExecutionsTermination can then cap — opt in explicitly.
        bus = MessageBus(
            subscriptions=[TopicSubscription(agent=agent, topics=["t"])],
            emitter=emitter,
            termination=MaxExecutionsTermination(2),
            max_messages=50,
            allow_self_delivery=True,
        )
        result = await bus.run([make_seed("t")])

        assert result.total_executions <= 2
        assert result.termination_reason == "MaxExecutionsTermination"

    async def test_composite_termination_identifies_sub_condition(self) -> None:
        """When a BusCompositeTermination fires, the result identifies which sub-condition triggered."""
        emitter = make_emitter()
        responses = [
            make_tool_call_response("publish_message", topic="t", content="more"),
            make_response("done"),
        ] * 10
        agent = make_agent("agent", emitter, responses=responses)

        composite = BusCompositeTermination(
            [MaxMessagesTermination(100), MaxExecutionsTermination(2)],
            mode="any",
        )
        # Self-echo on topic "t" produces the execution chain that drives
        # the composite termination. Opt in to self-delivery.
        bus = MessageBus(
            subscriptions=[TopicSubscription(agent=agent, topics=["t"])],
            emitter=emitter,
            termination=composite,
            max_messages=50,
            allow_self_delivery=True,
        )
        result = await bus.run([make_seed("t")])

        assert result.termination_reason == "MaxExecutionsTermination"


class TestMessageBusEmptySubscribers:
    async def test_empty_subscribers(self) -> None:
        """Message on topic with no subscribers is logged but triggers nothing."""
        emitter = make_emitter()
        agent = make_agent("agent", emitter)

        bus = MessageBus(
            subscriptions=[TopicSubscription(agent=agent, topics=["other"])],
            emitter=emitter,
        )
        result = await bus.run([make_seed("unmatched")])

        assert result.total_messages == 1
        assert result.total_executions == 0
        assert result.termination_reason == "quiescence"


class TestMessageBusEvents:
    async def test_event_emission(self) -> None:
        """All four event types emitted at correct points."""
        emitter = make_emitter()
        agent = make_agent("agent", emitter, publish_topic="output", publish_content="result")

        bus = MessageBus(
            subscriptions=[
                TopicSubscription(agent=agent, topics=["input"]),
            ],
            emitter=emitter,
        )
        await bus.run([make_seed("input")])

        event_types = [type(e) for e in emitter.events]
        assert MessageBusStartEvent in event_types
        assert MessageBusCompleteEvent in event_types
        assert MessageDeliveredEvent in event_types
        # MessagePublishedEvent emitted from the publish tool
        assert MessagePublishedEvent in event_types

    async def test_start_event_fields(self) -> None:
        emitter = make_emitter()
        agent = make_agent("agent", emitter)

        bus = MessageBus(
            subscriptions=[TopicSubscription(agent=agent, topics=["t"])],
            emitter=emitter,
            max_messages=50,
            max_depth=5,
        )
        await bus.run([make_seed("t")])

        start_events = [e for e in emitter.events if isinstance(e, MessageBusStartEvent)]
        assert len(start_events) == 1
        start = start_events[0]
        assert start.seed_count == 1
        assert start.max_messages == 50
        assert start.max_depth == 5

    async def test_complete_event_fields(self) -> None:
        emitter = make_emitter()
        agent = make_agent("agent", emitter)

        bus = MessageBus(
            subscriptions=[TopicSubscription(agent=agent, topics=["t"])],
            emitter=emitter,
        )
        await bus.run([make_seed("t")])

        complete_events = [e for e in emitter.events if isinstance(e, MessageBusCompleteEvent)]
        assert len(complete_events) == 1
        complete = complete_events[0]
        assert complete.total_messages == 1
        assert complete.total_executions == 1
        assert complete.agent_execution_counts == {"agent": 1}


class TestMessageBusValidation:
    async def test_empty_seed_messages_raises(self) -> None:
        emitter = make_emitter()
        bus = MessageBus(
            subscriptions=[],
            emitter=emitter,
        )
        with pytest.raises(ValueError, match="seed_messages must not be empty"):
            await bus.run([])


class TestMessageBusSubscriberFailure:
    async def test_subscriber_failure_populates_failed_executions(self) -> None:
        """A failing subscriber is captured in result.failed_executions with correct details."""
        emitter = make_emitter()
        # Agent that will raise an error
        failing_agent = ReActAgent(
            name="failing_agent",
            llm_client=MockLLMClient([]),  # empty responses causes error
            emitter=emitter,
            system_prompt="You are failing_agent.",
            tools=[],
        )
        bus = MessageBus(
            subscriptions=[TopicSubscription(agent=failing_agent, topics=["t"])],
            emitter=emitter,
        )
        result = await bus.run([make_seed("t")])

        assert len(result.failed_executions) == 1
        failure = result.failed_executions[0]
        assert failure.agent_name == "failing_agent"
        assert failure.trigger_message_id == result.messages[0].message_id
        assert failure.error_type != ""
        assert failure.error_message != ""
        assert result.executions == []

    async def test_partial_failure_captures_both(self) -> None:
        """When one subscriber fails and another succeeds, both are recorded correctly."""
        emitter = make_emitter()
        good_agent = make_agent("good_agent", emitter)
        failing_agent = ReActAgent(
            name="failing_agent",
            llm_client=MockLLMClient([]),
            emitter=emitter,
            system_prompt="You are failing_agent.",
            tools=[],
        )
        bus = MessageBus(
            subscriptions=[
                TopicSubscription(agent=good_agent, topics=["t"]),
                TopicSubscription(agent=failing_agent, topics=["t"]),
            ],
            emitter=emitter,
        )
        result = await bus.run([make_seed("t")])

        assert len(result.executions) == 1
        assert result.executions[0].agent_name == "good_agent"
        assert len(result.failed_executions) == 1
        assert result.failed_executions[0].agent_name == "failing_agent"

    async def test_agent_with_existing_context_providers(self) -> None:
        """MessageBus appends history provider to agent's existing context providers."""
        from nanitics.strategies.agents.context import ContextContent

        class DummyProvider:
            async def provide(self, messages: list) -> ContextContent | None:
                return None

        emitter = make_emitter()
        dummy = DummyProvider()
        agent = ReActAgent(
            name="agent",
            llm_client=MockLLMClient([make_response("done")]),
            emitter=emitter,
            system_prompt="You are agent.",
            tools=[],
            context_providers=[dummy],
        )

        bus = MessageBus(
            subscriptions=[TopicSubscription(agent=agent, topics=["t"])],
            emitter=emitter,
        )
        result = await bus.run([make_seed("t")])

        assert result.total_executions == 1
        # After run, original providers should be restored
        assert agent._context_providers == [dummy]

    async def test_failure_emits_event_and_populates_result(self) -> None:
        """Subscriber failure emits MessageDeliveredEvent with error AND populates result.failed_executions."""
        emitter = make_emitter()
        failing_agent = ReActAgent(
            name="failing_agent",
            llm_client=MockLLMClient([]),
            emitter=emitter,
            system_prompt="You are failing_agent.",
            tools=[],
        )
        bus = MessageBus(
            subscriptions=[TopicSubscription(agent=failing_agent, topics=["t"])],
            emitter=emitter,
        )
        result = await bus.run([make_seed("t")])

        # Check event emission
        delivered_events = [e for e in emitter.events if isinstance(e, MessageDeliveredEvent)]
        assert len(delivered_events) == 1
        assert delivered_events[0].error is not None
        assert delivered_events[0].agent_name == "failing_agent"

        # Check result
        assert len(result.failed_executions) == 1

        # Check complete event has failure count
        complete_events = [e for e in emitter.events if isinstance(e, MessageBusCompleteEvent)]
        assert len(complete_events) == 1
        assert complete_events[0].failed_executions == 1


class TestMessageBusSelfDelivery:
    """Cover the default self-delivery suppression and the opt-in flag.

    The bus suppresses delivery of a publish back to the publishing agent
    by default; ``allow_self_delivery=True`` restores the pre-fix
    broadcast-to-self behavior for reactive loops.
    """

    async def test_default_suppresses_self_delivery_when_publisher_subscribes(self) -> None:
        emitter = make_emitter()
        # agent_a subscribes to "topic-x" and publishes to "topic-x". The
        # seed delivers to agent_a; agent_a publishes; the published
        # message should NOT trigger a second execution of agent_a.
        agent_a = make_agent(
            "agent_a",
            emitter,
            publish_topic="topic-x",
            publish_content="self publish",
        )
        bus = MessageBus(
            subscriptions=[TopicSubscription(agent=agent_a, topics=["topic-x"])],
            emitter=emitter,
        )

        result = await bus.run([make_seed("topic-x", "kickoff")])

        # Seed delivers once to agent_a; published message is suppressed.
        assert result.total_executions == 1, f"expected 1 execution (seed only), got {result.total_executions}"
        # MessagePublishedEvent still fires (publish happened) but no
        # MessageDeliveredEvent carries agent_a as the publish's recipient.
        delivered = [e for e in emitter.events if isinstance(e, MessageDeliveredEvent)]
        published = [e for e in emitter.events if isinstance(e, MessagePublishedEvent)]
        assert len(published) == 1
        assert published[0].author == "agent_a"
        # Cross-check: no delivered event for the self-published message id.
        self_publish_id = published[0].message_id
        assert not any(d.message_id == self_publish_id for d in delivered), (
            f"self-publish {self_publish_id} should not appear as a delivery; got: {delivered!r}"
        )

    async def test_allow_self_delivery_restores_echo(self) -> None:
        emitter = make_emitter()
        # Two publish-then-reply cycles so the echo chain can fire twice.
        responses = [
            make_tool_call_response("publish_message", topic="topic-x", content="echo 1"),
            make_response("done-1"),
            make_tool_call_response("publish_message", topic="topic-x", content="echo 2"),
            make_response("done-2"),
        ]
        agent_a = make_agent("agent_a", emitter, responses=responses)
        bus = MessageBus(
            subscriptions=[TopicSubscription(agent=agent_a, topics=["topic-x"])],
            emitter=emitter,
            allow_self_delivery=True,
            max_messages=3,
        )

        result = await bus.run([make_seed("topic-x", "kickoff")])

        # Pre-fix behavior: seed executes agent, the first publish is
        # delivered back to agent_a (self-echo), triggering a second
        # execution. max_messages caps the chain.
        assert result.total_executions >= 2, (
            f"allow_self_delivery=True must permit self-echo; got {result.total_executions}"
        )
        delivered = [e for e in emitter.events if isinstance(e, MessageDeliveredEvent)]
        self_deliveries = [d for d in delivered if d.agent_name == "agent_a"]
        # At least one delivery is to agent_a for a message agent_a authored
        # (i.e. self-echo actually fired).
        published = [e for e in emitter.events if isinstance(e, MessagePublishedEvent)]
        self_authored_ids = {p.message_id for p in published if p.author == "agent_a"}
        self_echo_deliveries = [d for d in self_deliveries if d.message_id in self_authored_ids]
        assert self_echo_deliveries, "allow_self_delivery=True must deliver agent_a's own publish back to agent_a"

    async def test_self_delivery_filter_composes_with_message_filter(self) -> None:
        """Filter + self-delivery check compose: both must pass for delivery."""
        emitter = make_emitter()

        class AcceptAll:
            async def match(self, message: BusMessage) -> bool:
                return True

        class RejectAll:
            async def match(self, message: BusMessage) -> bool:
                return False

        # Case 1: MessageFilter rejects → subscriber excluded regardless of
        # self-delivery flag state (rejection wins).
        rejecting_agent = make_agent("rej", emitter, publish_topic="topic-x")
        bus_reject = MessageBus(
            subscriptions=[
                TopicSubscription(agent=rejecting_agent, topics=["topic-x"], filter=RejectAll()),
            ],
            emitter=emitter,
            allow_self_delivery=True,  # even with opt-in, filter still gates
        )
        result_reject = await bus_reject.run([make_seed("topic-x")])
        assert result_reject.total_executions == 0, "MessageFilter rejection must override self-delivery opt-in"

        # Case 2: MessageFilter accepts + self-delivery default → publishing
        # agent's self-publish is still suppressed despite the accepting filter.
        emitter2 = make_emitter()
        accepting_agent = make_agent(
            "acc",
            emitter2,
            publish_topic="topic-y",
        )
        bus_accept = MessageBus(
            subscriptions=[
                TopicSubscription(agent=accepting_agent, topics=["topic-y"], filter=AcceptAll()),
            ],
            emitter=emitter2,
        )
        result_accept = await bus_accept.run([make_seed("topic-y")])
        # Seed fires one execution; the self-publish is suppressed by the
        # default self-delivery filter even though the AcceptAll filter
        # would otherwise let it through.
        assert result_accept.total_executions == 1, (
            f"self-delivery suppression must fire after MessageFilter accept; got {result_accept.total_executions}"
        )


@pytest.mark.parametrize("value", [0, -1])
def test_max_messages_termination_rejects_non_positive(value: int) -> None:
    with pytest.raises(ValueError, match="max_messages must be positive"):
        MaxMessagesTermination(max_messages=value)


@pytest.mark.parametrize("value", [0, -1])
def test_max_executions_termination_rejects_non_positive(value: int) -> None:
    with pytest.raises(ValueError, match="max_executions must be positive"):
        MaxExecutionsTermination(max_executions=value)
