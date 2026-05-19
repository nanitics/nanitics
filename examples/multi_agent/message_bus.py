"""MessageBus: topic-based publish-subscribe agent coordination.

Demonstrates MessageBus — agents subscribe to topics, seed messages trigger
processing, agents can publish new messages to trigger downstream agents.
Covers basic routing, multi-hop reactive chains, message filtering, and
termination conditions.

Related guide: docs/guides/multi-agent-foundations.md
"""

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics import (
    MockLLMClient,
    ReActAgent,
    ToolCall,
)
from nanitics.infrastructure import (
    MessageBusCompleteEvent,
    MessageBusStartEvent,
    MessageDeliveredEvent,
    MessagePublishedEvent,
)
from nanitics.specialized import (
    BusMessage,
    MaxMessagesTermination,
    MessageBus,
    MessageBusResult,
    TopicSubscription,
)


async def main() -> None:
    # --- Section 1: Basic Routing ---
    # Simplest bus: one agent subscribes to one topic, one seed message.
    # Inspect result fields and verify event emission.

    print("--- Section 1: Basic Routing ---")

    emitter = make_emitter("bus-s1")
    analyst = ReActAgent(
        name="analyst",
        llm_client=MockLLMClient([make_response("Revenue is up 15% this quarter.")]),
        emitter=emitter,
        system_prompt="You are a financial analyst.",
        tools=[],
    )

    subscriptions = [TopicSubscription(agent=analyst, topics=["raw-data"])]
    bus = MessageBus(subscriptions=subscriptions, emitter=emitter)

    seed = BusMessage(topic="raw-data", content="Q4 revenue data: $2.3M", author="system")
    result: MessageBusResult = await bus.run(seed_messages=[seed])

    # Result inspection
    assert result.total_messages == 1, f"Expected 1 message, got {result.total_messages}"
    assert result.total_executions == 1, f"Expected 1 execution, got {result.total_executions}"
    assert result.termination_reason == "quiescence"
    assert result.executions[0].agent_name == "analyst"
    assert result.executions[0].output == "Revenue is up 15% this quarter."
    assert result.executions[0].published_messages == []
    assert len(result.failed_executions) == 0

    # Event verification
    events = emitter.events
    start_events = [e for e in events if isinstance(e, MessageBusStartEvent)]
    delivered_events = [e for e in events if isinstance(e, MessageDeliveredEvent)]
    complete_events = [e for e in events if isinstance(e, MessageBusCompleteEvent)]

    assert len(start_events) == 1
    start = start_events[0]
    assert start.seed_topics == ["raw-data"]
    assert start.subscriber_count == 1

    assert len(delivered_events) == 1
    assert delivered_events[0].agent_name == "analyst"

    assert len(complete_events) == 1
    complete = complete_events[0]
    assert complete.total_messages == 1
    assert complete.termination_reason == "quiescence"

    print(f"  Seed: topic={seed.topic}, content={seed.content!r}")
    print(f"  Agent output: {result.executions[0].output!r}")
    print(f"  Termination: {result.termination_reason}")
    print(f"  Events: {len(start_events)} start, {len(delivered_events)} delivered, {len(complete_events)} complete")
    print("✓ Single agent processes seed message, bus reaches quiescence")

    # --- Section 2: Multi-Hop Reactive Chain ---
    # Agent publishes a message that triggers a downstream agent.
    # Demonstrates depth tracking and parent message linking.

    print("\n--- Section 2: Multi-Hop Reactive Chain ---")

    emitter = make_emitter("bus-s2")

    # Analyst subscribes to "raw-data", publishes to "findings"
    analyst = ReActAgent(
        name="analyst",
        llm_client=MockLLMClient(
            [
                make_response(
                    "Let me publish my analysis.",
                    tool_calls=[
                        ToolCall(
                            id="tc-1",
                            name="publish_message",
                            arguments={
                                "topic": "findings",
                                "content": "Analysis: Revenue up 15%, driven by new product line.",
                            },
                        )
                    ],
                    stop_reason="tool_use",
                ),
                make_response("Analysis complete. Published findings."),
            ]
        ),
        emitter=emitter,
        system_prompt="You are a financial analyst. Publish findings when you discover insights.",
        tools=[],
    )

    # Reporter subscribes to "findings"
    reporter = ReActAgent(
        name="reporter",
        llm_client=MockLLMClient(
            [
                make_response("Report: Q4 showed 15% revenue growth led by new product expansion."),
            ]
        ),
        emitter=emitter,
        system_prompt="You write executive reports from analyst findings.",
        tools=[],
    )

    subscriptions = [
        TopicSubscription(agent=analyst, topics=["raw-data"]),
        TopicSubscription(agent=reporter, topics=["findings"]),
    ]
    bus = MessageBus(subscriptions=subscriptions, emitter=emitter)

    seed = BusMessage(topic="raw-data", content="Q4 revenue data: $2.3M", author="system")
    result = await bus.run(seed_messages=[seed])

    assert result.total_messages == 2, f"Expected 2 messages, got {result.total_messages}"
    assert result.total_executions == 2, f"Expected 2 executions, got {result.total_executions}"
    assert result.termination_reason == "quiescence"

    # Analyst execution: published one message
    analyst_exec = next(e for e in result.executions if e.agent_name == "analyst")
    assert len(analyst_exec.published_messages) == 1
    published = analyst_exec.published_messages[0]
    assert published.topic == "findings"
    assert published.depth == 1
    assert published.parent_message_id == seed.message_id

    # Reporter execution: triggered by published message
    reporter_exec = next(e for e in result.executions if e.agent_name == "reporter")
    assert reporter_exec.output == "Report: Q4 showed 15% revenue growth led by new product expansion."

    # Message chain: seed (depth=0) → published (depth=1)
    assert result.messages[0].depth == 0
    assert result.messages[1].depth == 1

    # Verify publish event
    publish_events = [e for e in emitter.events if isinstance(e, MessagePublishedEvent)]
    assert len(publish_events) == 1
    assert publish_events[0].topic == "findings"
    assert publish_events[0].author == "analyst"

    print(f"  Chain: seed({seed.topic}) → analyst → publish({published.topic}) → reporter")
    print(f"  Analyst output: {analyst_exec.output!r}")
    print(f"  Reporter output: {reporter_exec.output!r}")
    print("  Depth tracking: seed=0, published=1")
    print(
        "  Parent linking: published.parent_message_id == seed.message_id: "
        f"{published.parent_message_id == seed.message_id}"
    )
    print("✓ Multi-hop chain: analyst publishes findings that trigger reporter")

    # --- Section 3: Message Filtering ---
    # MessageFilter narrows which messages trigger a subscriber beyond topic matching.
    # Only messages containing "urgent" reach the monitor agent.

    print("\n--- Section 3: Message Filtering ---")

    class UrgentFilter:
        """Only match messages containing 'urgent'."""

        async def match(self, message: BusMessage) -> bool:
            return "urgent" in message.content.lower()

    emitter = make_emitter("bus-s3")
    monitor = ReActAgent(
        name="monitor",
        llm_client=MockLLMClient(
            [
                make_response("Alert acknowledged: handling urgent issue."),
            ]
        ),
        emitter=emitter,
        system_prompt="You monitor alerts and respond to urgent issues.",
        tools=[],
    )

    subscriptions = [
        TopicSubscription(agent=monitor, topics=["alerts"], filter=UrgentFilter()),
    ]
    bus = MessageBus(subscriptions=subscriptions, emitter=emitter)

    routine_msg = BusMessage(topic="alerts", content="Routine check: all systems normal", author="system")
    urgent_msg = BusMessage(topic="alerts", content="URGENT: service degradation detected", author="system")
    result = await bus.run(seed_messages=[routine_msg, urgent_msg])

    assert result.total_messages == 2, f"Expected 2 messages, got {result.total_messages}"
    assert result.total_executions == 1, f"Expected 1 execution (filtered), got {result.total_executions}"
    assert result.executions[0].agent_name == "monitor"
    assert result.executions[0].trigger_message_id == urgent_msg.message_id

    print(f"  Message 1: {routine_msg.content!r} → filtered out")
    print(f"  Message 2: {urgent_msg.content!r} → triggered monitor")
    print(f"  Agent output: {result.executions[0].output!r}")
    print("✓ UrgentFilter selectively triggers monitor only for urgent messages")

    # --- Section 4: Termination Conditions ---
    # MaxMessagesTermination caps a self-sustaining reactive loop.
    # Without it, the echo agent would publish back to "loop" indefinitely.

    print("\n--- Section 4: Termination Conditions ---")

    emitter = make_emitter("bus-s4")

    # Echo agent: subscribes to "loop", publishes back to "loop", creating a chain.
    # Needs enough mock responses for 3 iterations (tool call + final text each).
    echo_responses = []
    for i in range(5):
        echo_responses.extend(
            [
                make_response(
                    f"Echoing message {i + 1}.",
                    tool_calls=[
                        ToolCall(
                            id=f"tc-echo-{i}",
                            name="publish_message",
                            arguments={"topic": "loop", "content": f"Echo {i + 1}: still going"},
                        )
                    ],
                    stop_reason="tool_use",
                ),
                make_response(f"Echo {i + 1} complete."),
            ]
        )

    echo = ReActAgent(
        name="echo",
        llm_client=MockLLMClient(echo_responses),
        emitter=emitter,
        system_prompt="You echo messages back to the loop topic.",
        tools=[],
    )

    subscriptions = [TopicSubscription(agent=echo, topics=["loop"])]
    # The echo agent both subscribes to and publishes to "loop". By default
    # the bus suppresses delivery of an agent's own publish back to itself;
    # this section demonstrates a legitimate broadcast-to-self loop, so
    # opt in with allow_self_delivery=True to drive the chain.
    bus = MessageBus(
        subscriptions=subscriptions,
        emitter=emitter,
        termination=MaxMessagesTermination(max_messages=3),
        allow_self_delivery=True,
    )

    seed = BusMessage(topic="loop", content="Start the loop", author="system")
    result = await bus.run(seed_messages=[seed])

    assert result.total_messages == 3, f"Expected 3 messages (capped), got {result.total_messages}"
    assert result.termination_reason == "MaxMessagesTermination"

    print(f"  Messages processed: {result.total_messages} (capped at 3)")
    print(f"  Termination: {result.termination_reason}")
    print(f"  Executions: {result.total_executions}")
    print("  Other conditions: MaxExecutionsTermination, BusPredicateTermination, BusCompositeTermination")
    print("✓ MaxMessagesTermination stops the self-sustaining echo loop at 3 messages")


if __name__ == "__main__":
    asyncio.run(main())
