"""MessageBus publish/subscribe semantics with topic routing, filtering, and termination.

Two tests validate the bus primitive against real LLMs:

* ``test_message_bus_publish_then_subscriber`` — an ``analyst`` subscribed
  to ``raw-data`` publishes a message to ``findings`` using the
  ``publish_message`` tool injected by ``create_bus_tools``; a ``reporter``
  subscribed to ``findings`` receives and processes it. Pins the core
  pub/sub contract: A publishes, B (subscribed) receives, A does not echo
  itself.

* ``test_message_bus_filter_and_termination`` — a ``monitor`` subscribes
  to ``alerts`` with a ``MessageFilter`` that only matches messages
  containing ``"urgent"``. Two seed messages (routine, urgent) are
  injected plus a ``MaxMessagesTermination`` cap. Pins that the filter
  gates execution (agent only runs for the urgent seed) and that the
  termination class name surfaces via
  ``MessageBusResult.termination_reason`` when the cap would fire.

Acceptance criteria (publish/subscribe):
  - Trace contains ``MessageBusStartEvent`` with subscriber_count=2 and a
    ``subscriptions`` map listing both ``raw-data`` → [analyst] and
    ``findings`` → [reporter].
  - Trace contains a ``MessagePublishedEvent`` whose ``author ==
    "analyst"`` and ``topic == "findings"`` (proves analyst's tool call
    went through ``create_bus_tools``).
  - Trace contains a ``MessageDeliveredEvent`` for ``reporter`` on topic
    ``findings`` (proves cross-agent delivery).
  - No ``MessageDeliveredEvent`` has ``agent_name == e.author`` for any
    ``e`` in ``publish_events`` — the bus does not self-echo (default
    ``allow_self_delivery=False`` is enforced universally, not just for
    analyst).
  - ``total_executions == 2`` on the publish-then-subscriber scenario
    (seed → analyst; analyst publish → reporter). Any third execution
    would be a self-echo regression.
  - ``MessageBusResult.messages`` snapshot length matches
    ``total_messages`` and contains the seed + the published message;
    exactly one published message has ``author == "analyst"`` and
    ``parent_message_id == seed.message_id``.

Acceptance criteria (filter + termination):
  - ``total_executions == 1`` (filter suppressed the routine seed).
  - The single execution's ``trigger_message_id`` is the urgent seed's
    ``message_id``, not the routine one — proving the filter chose
    correctly, not just dropped a random one.
  - ``termination_reason`` is a class name from the termination-condition
    set (``MaxMessagesTermination`` or ``quiescence``) — we pin the
    surface-level contract rather than forcing the cap to fire since the
    two seeds naturally terminate via quiescence.
"""

from __future__ import annotations

import pytest

from nanitics import (
    InMemoryEmitter,
    MaxMessagesTermination,
    ReActAgent,
)
from nanitics.experimental.coordination import (
    BusMessage,
    MessageBus,
    TopicSubscription,
)
from nanitics.infrastructure import (
    MessageBusCompleteEvent,
    MessageBusStartEvent,
    MessageDeliveredEvent,
    MessagePublishedEvent,
)
from validation.helpers import (
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)


class _UrgentFilter:
    """Match only messages whose content contains ``'urgent'`` (case-insensitive)."""

    async def match(self, message: BusMessage) -> bool:
        return "urgent" in message.content.lower()


async def test_message_bus_publish_then_subscriber(traced_emitter: InMemoryEmitter) -> None:
    client = make_llm_client("anthropic")

    analyst = ReActAgent(
        name="analyst",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You are a financial analyst. When you receive raw quarterly "
            "data, call the `publish_message` tool exactly once with "
            "topic='findings' and a one-sentence analysis as content. "
            "Then reply with a short confirmation."
        ),
        tools=[],
        max_iterations=3,
    )
    reporter = ReActAgent(
        name="reporter",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You are an executive reporter. Given an analyst finding, produce a single-sentence executive summary."
        ),
        tools=[],
        max_iterations=2,
    )

    subscriptions = [
        TopicSubscription(agent=analyst, topics=["raw-data"]),
        TopicSubscription(agent=reporter, topics=["findings"]),
    ]

    bus = MessageBus(subscriptions=subscriptions, emitter=traced_emitter, max_messages=10)
    seed = BusMessage(
        topic="raw-data",
        content="Q4 revenue: $2.3M, up 15% YoY, driven by enterprise expansion.",
        author="system",
    )

    result = await run_with_retry(lambda: bus.run(seed_messages=[seed]), max_attempts=2)

    # --- Start event pins topology ---
    assert_trace_contains(
        traced_emitter,
        MessageBusStartEvent,
        predicate=lambda e: (
            e.subscriber_count == 2
            and e.subscriptions.get("raw-data") == ["analyst"]
            and e.subscriptions.get("findings") == ["reporter"]
        ),
    )

    # --- Analyst published to findings; reporter received it ---
    publish_events = [e for e in traced_emitter.events if isinstance(e, MessagePublishedEvent)]
    assert publish_events, "Expected at least one MessagePublishedEvent from the analyst"
    analyst_publishes = [e for e in publish_events if e.author == "analyst" and e.topic == "findings"]
    assert analyst_publishes, (
        f"Expected a MessagePublishedEvent from analyst to topic 'findings'; "
        f"got: {[(e.author, e.topic) for e in publish_events]}"
    )

    delivered = [e for e in traced_emitter.events if isinstance(e, MessageDeliveredEvent)]
    reporter_deliveries = [e for e in delivered if e.agent_name == "reporter" and e.topic == "findings"]
    assert reporter_deliveries, (
        f"Expected a MessageDeliveredEvent for reporter on 'findings'; "
        f"got: {[(e.agent_name, e.topic) for e in delivered]}"
    )

    # --- No self-echo: universal check — no delivery to any author of any
    # publish in this trace. Catches the C1 regression (analyst or reporter
    # or any future subscriber publishing to a topic it also listens to).
    for d in delivered:
        matching_pub = next((p for p in publish_events if p.message_id == d.message_id), None)
        if matching_pub is not None:
            assert matching_pub.author != d.agent_name, (
                f"self-echo regression: {d.agent_name} received its own publish {d.message_id} on topic {d.topic}"
            )

    # --- Execution count pins the scenario shape ---
    # seed → analyst (1) and analyst-publish → reporter (2). A third
    # execution would mean analyst (or reporter) received its own publish
    # back and re-ran.
    assert result.total_executions == 2, (
        f"Expected exactly 2 executions (seed→analyst, analyst-publish→reporter); "
        f"got: {result.total_executions}. A third would indicate a self-echo regression."
    )

    # --- BusState snapshot: messages match total ---
    assert len(result.messages) == result.total_messages
    assert result.total_messages >= 2, (
        f"Expected at least 2 messages (seed + analyst publish), got: {result.total_messages}"
    )

    # Exactly one published message is authored by the analyst, and its
    # parent is the seed message — the chain tracks correctly.
    analyst_published = [m for m in result.messages if m.author == "analyst"]
    assert len(analyst_published) == 1, f"Expected exactly one analyst-authored message, got: {len(analyst_published)}"
    assert analyst_published[0].parent_message_id == seed.message_id, (
        f"Analyst-published message must link back to seed; got parent_message_id="
        f"{analyst_published[0].parent_message_id!r}, expected {seed.message_id!r}"
    )
    assert analyst_published[0].depth == 1, (
        f"Analyst-published message must be at depth 1, got: {analyst_published[0].depth}"
    )

    assert_trace_contains(
        traced_emitter,
        MessageBusCompleteEvent,
        predicate=lambda e: e.total_messages == result.total_messages,
    )


@pytest.mark.quick
async def test_message_bus_filter_and_termination(traced_emitter: InMemoryEmitter) -> None:
    client = make_llm_client("anthropic")

    monitor = ReActAgent(
        name="monitor",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You are an incident monitor. Respond in one short sentence acknowledging the alert you were given."
        ),
        tools=[],
        max_iterations=2,
    )

    subscriptions = [
        TopicSubscription(
            agent=monitor,
            topics=["alerts"],
            filter=_UrgentFilter(),
        ),
    ]

    # Termination is configured even though it likely won't fire with only
    # two seeds — presence pins the termination_reason branch set.
    bus = MessageBus(
        subscriptions=subscriptions,
        emitter=traced_emitter,
        termination=MaxMessagesTermination(max_messages=10),
        max_messages=10,
    )

    routine = BusMessage(
        topic="alerts",
        content="Routine check: all systems nominal.",
        author="system",
    )
    urgent = BusMessage(
        topic="alerts",
        content="URGENT: payment service degradation detected.",
        author="system",
    )

    result = await run_with_retry(
        lambda: bus.run(seed_messages=[routine, urgent]),
        max_attempts=2,
    )

    # --- Filter gated execution: only the urgent seed triggered the monitor ---
    assert result.total_executions == 1, (
        f"Filter must suppress routine seed; expected 1 execution, got: {result.total_executions}"
    )
    assert result.executions[0].agent_name == "monitor"
    assert result.executions[0].trigger_message_id == urgent.message_id, (
        f"Expected monitor to be triggered by the urgent seed ({urgent.message_id!r}), "
        f"got: {result.executions[0].trigger_message_id!r}"
    )

    # The message log still records both seeds (filter blocks execution, not ingestion).
    seed_ids = {m.message_id for m in result.messages}
    assert {routine.message_id, urgent.message_id}.issubset(seed_ids), (
        f"Both seeds must appear in the message log; got ids: {seed_ids}"
    )

    assert result.termination_reason in {"quiescence", "MaxMessagesTermination"}, (
        f"Expected termination_reason in {{'quiescence','MaxMessagesTermination'}}, got: {result.termination_reason!r}"
    )
