"""Supervisor retry/accept and max-retry exhaustion paths on a real-LLM agent.

Two tests exercise the ``Supervisor`` primitive end-to-end against a real
``ReActAgent``:

* ``test_supervisor_retry_then_accept`` — one retry, then accept. The
  trigger's first invocation returns ``RETRY`` with feedback; the second
  invocation accepts. Pins the feedback-injection contract (the retry
  task carries ``"Feedback from review"`` into the agent's messages) and
  the two-event trace shape (``retry`` then ``accept``).
* ``test_supervisor_max_retry_exhaustion`` — always retry. The trigger
  unconditionally returns ``RETRY``; the supervisor runs
  ``max_retries + 1`` attempts, emits ``max_retries + 1`` ``retry``
  events (no terminal accept), and returns ``accepted=False``.

Acceptance criteria (retry-then-accept):
  - ``supervision_result.accepted is True`` after retry.
  - ``supervision_result.total_attempts == 2``.
  - Exactly one intervention recorded with ``action == RETRY`` and
    ``trigger_name == "detail_check"``.
  - Trace contains a ``SupervisionEvent`` with ``action == "retry"``,
    ``supervised_agent == "analyst"``, and ``attempt == 1``.
  - Trace contains a ``SupervisionEvent`` with ``action == "accept"``,
    ``supervised_agent == "analyst"``, and ``attempt == 2``.
  - The final agent result's messages contain the supervisor feedback
    string (proves the retry task was actually re-issued to the worker
    with the feedback appended).

Acceptance criteria (max-retry exhaustion):
  - ``supervision_result.accepted is False``.
  - ``supervision_result.total_attempts == max_retries + 1``.
  - Trace contains ``max_retries + 1`` ``SupervisionEvent`` instances
    with ``action == "retry"`` and no terminal ``"accept"``.
"""

from __future__ import annotations

from nanitics import (
    AgentResult,
    InMemoryEmitter,
    PredicateTrigger,
    ReActAgent,
    SupervisionAction,
    SupervisionDecision,
    Supervisor,
)
from nanitics.infrastructure import SupervisionEvent
from validation.helpers import (
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)

_RETRY_FEEDBACK = "Include at least one specific data point or concrete example to support your analysis."


def _make_analyst(client, emitter: InMemoryEmitter) -> ReActAgent:
    return ReActAgent(
        name="analyst",
        llm_client=client,
        emitter=emitter,
        system_prompt=(
            "You are a concise analyst. Produce a single-paragraph analysis of the topic the user provides."
        ),
        tools=[],
        max_iterations=3,
    )


async def test_supervisor_retry_then_accept(traced_emitter: InMemoryEmitter) -> None:
    client = make_llm_client("anthropic")

    # Closure state captured here. A factory is used so each outer
    # ``run_with_retry`` attempt starts from a fresh counter — otherwise
    # a transient provider error on the second supervised run would cause
    # the outer retry's first predicate invocation to fall through to
    # ``None`` and the test would fail the ``total_attempts == 2`` check
    # while pointing at the SDK instead of the provider flake.
    def make_supervise_call():
        call_count = 0

        def retry_once(result: AgentResult, task: str) -> SupervisionDecision | None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return SupervisionDecision(
                    action=SupervisionAction.RETRY,
                    feedback=_RETRY_FEEDBACK,
                    trigger_name="detail_check",
                )
            return None

        trigger = PredicateTrigger(name="detail_check", predicate=retry_once)
        agent = _make_analyst(client, traced_emitter)
        supervisor = Supervisor(
            triggers=[trigger],
            emitter=traced_emitter,
            max_retries=2,
        )
        return supervisor.supervise(
            agent,
            "Analyze the impact of remote work on software engineering productivity.",
        )

    supervision_result = await run_with_retry(make_supervise_call, max_attempts=2)

    # --- Supervision-result invariants ---
    assert supervision_result.accepted is True, (
        f"Expected supervision_result.accepted=True after retry, got: {supervision_result.accepted}"
    )
    assert supervision_result.total_attempts == 2, (
        f"Expected total_attempts=2, got: {supervision_result.total_attempts}"
    )
    assert len(supervision_result.interventions) == 1, (
        f"Expected exactly one intervention, got: {len(supervision_result.interventions)}"
    )
    intervention = supervision_result.interventions[0]
    assert intervention.action == SupervisionAction.RETRY, (
        f"Expected intervention action=RETRY, got: {intervention.action}"
    )
    assert intervention.trigger_name == "detail_check", (
        f"Expected intervention trigger_name='detail_check', got: {intervention.trigger_name!r}"
    )

    # --- Trace-shape invariants ---
    assert_trace_contains(
        traced_emitter,
        SupervisionEvent,
        predicate=lambda e: e.action == "retry" and e.supervised_agent == "analyst" and e.attempt == 1,
    )
    assert_trace_contains(
        traced_emitter,
        SupervisionEvent,
        predicate=lambda e: e.action == "accept" and e.supervised_agent == "analyst" and e.attempt == 2,
    )

    # --- Feedback propagation ---
    # The supervisor's retry path appends
    # "\n\n## Feedback from review\n{decision.feedback}" to the original
    # task. That retry task becomes the user message fed to the worker on
    # the second attempt, so it must appear verbatim somewhere in the
    # final AgentResult's message history. This is the distinguishing
    # assertion for supervisor-driven retry: the feedback actually reached
    # the worker.
    serialized_messages = "\n".join(str(getattr(m, "content", "")) for m in supervision_result.result.messages)
    assert "Feedback from review" in serialized_messages, (
        "Expected the retry task's 'Feedback from review' header to appear in the "
        f"worker's message history; got messages: {serialized_messages!r}"
    )
    assert _RETRY_FEEDBACK in serialized_messages, (
        "Expected the supervisor's feedback string to appear in the worker's "
        f"message history; got messages: {serialized_messages!r}"
    )


async def test_supervisor_max_retry_exhaustion(traced_emitter: InMemoryEmitter) -> None:
    """Exhaust ``max_retries`` to cover the non-accept exit branch."""
    client = make_llm_client("anthropic")
    max_retries = 1

    def always_retry(result: AgentResult, task: str) -> SupervisionDecision | None:
        return SupervisionDecision(
            action=SupervisionAction.RETRY,
            feedback="Keep refining.",
            trigger_name="never_happy",
        )

    trigger = PredicateTrigger(name="never_happy", predicate=always_retry)
    agent = _make_analyst(client, traced_emitter)
    supervisor = Supervisor(
        triggers=[trigger],
        emitter=traced_emitter,
        max_retries=max_retries,
    )

    supervision_result = await run_with_retry(
        lambda: supervisor.supervise(
            agent,
            "Analyze the impact of remote work on software engineering productivity.",
        ),
        max_attempts=2,
    )

    expected_attempts = max_retries + 1

    assert supervision_result.accepted is False, (
        f"Expected supervision_result.accepted=False after exhaustion, got: {supervision_result.accepted}"
    )
    assert supervision_result.total_attempts == expected_attempts, (
        f"Expected total_attempts={expected_attempts}, got: {supervision_result.total_attempts}"
    )

    retry_events = [e for e in traced_emitter.events if isinstance(e, SupervisionEvent) and e.action == "retry"]
    assert len(retry_events) == expected_attempts, (
        f"Expected {expected_attempts} SupervisionEvent(action='retry') instances, got: {len(retry_events)}"
    )
    accept_events = [e for e in traced_emitter.events if isinstance(e, SupervisionEvent) and e.action == "accept"]
    assert not accept_events, f"Expected no terminal accept SupervisionEvent on exhaustion, got: {len(accept_events)}"
