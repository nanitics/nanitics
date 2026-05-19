"""Context management under pressure: real LLM crosses the threshold, policy fires.

Validates the :class:`ContextManager` pipeline: when the running message history
crosses ``context_limit * threshold``, the truncation and summarization policies
fire and messages are actually dropped or compressed before the next LLM call.

Two scenarios cover the two strategy branches independently, because the
pipeline runs truncation first and only falls through to summarization if
truncation did not free enough budget
(``nanitics/capabilities/context/manager.py:196-258``).

Scenario 1 — truncation: a tight budget with ``TruncationPolicy(preserve_first=
True, preserve_recent=2)`` and no ``SummarizationPolicy``. Verifies that
truncation drops middle messages, preserves the first and last-two groups, and
fits under the post-management budget.

Scenario 2 — summarization: a tight budget with ``truncation=None`` and a real
LLM-backed ``SummarizationPolicy``. This forces the summarization branch at
``manager.py:235-258`` — otherwise truncation alone would satisfy the budget
and the summarization branch would never fire.

Prior messages are seeded via :class:`ReActAgent`'s ``initial_messages`` parameter.
The synthetic priming crosses the threshold deterministically under
:class:`EstimateTokenCounter`.

Note: context management is per-LLM-call ephemeral — ``result.messages`` holds
the full persistent conversation history regardless of management. Reduction is
verified via the event payload.

Acceptance criteria — truncation scenario:
  - Trace contains a ``ContextTruncationEvent`` with
    ``messages_after < messages_before``.
  - ``preserve_first=True`` contract: the first primed message
    (``original_index == 0``) is not in ``removed_messages``.
  - ``preserve_recent=2`` contract: neither of the last two primed
    messages (``original_index`` in the top two) is in
    ``removed_messages``.
  - Budget-fit invariant: ``tokens_after <= context_limit -
    reserve_tokens``.
  - The agent's final answer identifies a theme and addresses the final
    question.

Acceptance criteria — summarization scenario:
  - Trace contains a ``ContextSummarizationEvent`` with
    ``messages_summarized > 0``.
  - ``summary_tokens < original_tokens`` — the summary is actually a
    reduction, not a no-op.
  - ``summary_text`` is non-empty — the summary carries content, not a
    placeholder.
"""

from __future__ import annotations

from nanitics.context import (
    ContextManager,
    SummarizationPolicy,
    TruncationPolicy,
)
from nanitics.infrastructure import ContextSummarizationEvent, ContextTruncationEvent
from nanitics.strategies import ReActAgent
from nanitics.tracing import (
    InMemoryEmitter,
    Message,
)
from validation.helpers import (
    assert_result_satisfies,
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)

_PRIMED_MESSAGE_COUNT = 36
_SYNTHETIC_TEXT = (
    "We discussed the architecture of distributed systems. The conversation "
    "covered consensus protocols, partition tolerance, and replication strategies. "
    "Many trade-offs between consistency and availability were weighed carefully."
)

_CONTEXT_LIMIT = 2000
_RESERVE_TOKENS = 200


def _prime_messages() -> list[Message]:
    msgs: list[Message] = []
    for i in range(_PRIMED_MESSAGE_COUNT // 2):
        msgs.append(Message(role="user", content=f"Turn {i}: {_SYNTHETIC_TEXT}"))
        msgs.append(Message(role="assistant", content=f"Ack {i}: {_SYNTHETIC_TEXT}"))
    return msgs


async def test_context_truncation_preserves_first_and_recent(traced_emitter: InMemoryEmitter) -> None:
    llm = make_llm_client("anthropic")
    context_manager = ContextManager(
        context_limit=_CONTEXT_LIMIT,
        reserve_tokens=_RESERVE_TOKENS,
        threshold=0.8,
        truncation=TruncationPolicy(preserve_first=True, preserve_recent=2),
        # No summarization — isolates the truncation branch.
    )

    primed = _prime_messages()
    agent = ReActAgent(
        name="context-truncation-agent",
        llm_client=llm,
        emitter=traced_emitter,
        system_prompt="You are a helpful assistant reflecting on a long prior conversation.",
        tools=[],
        context_manager=context_manager,
        initial_messages=primed,
        max_iterations=2,
    )

    result = await run_with_retry(
        lambda: agent.run(
            "Summarize our conversation so far and then answer this final question: What was the main theme?"
        ),
        max_attempts=2,
    )

    # A truncation event must have fired, and it must represent a real reduction.
    truncation_event = assert_trace_contains(
        traced_emitter,
        ContextTruncationEvent,
        predicate=lambda e: e.messages_after < e.messages_before,
    )

    # preserve_first contract: the first primed message must never be removed.
    removed_indices = {r.original_index for r in truncation_event.removed_messages}
    assert 0 not in removed_indices, (
        f"preserve_first=True violated: original_index 0 appears in removed_messages. "
        f"Removed indices: {sorted(removed_indices)}"
    )

    # preserve_recent=2 contract: the last two messages in the list passed to
    # prepare() must never be removed. ``original_index`` is indexed against
    # the full ``messages`` argument of prepare() (initial_messages + current
    # turn messages), so we derive the last-two indices from the event's own
    # ``messages_before`` value rather than len(primed).
    total_before = truncation_event.messages_before
    last_two = {total_before - 1, total_before - 2}
    overlap = last_two & removed_indices
    assert not overlap, (
        f"preserve_recent=2 violated: last-two indices {sorted(last_two)} (of "
        f"messages_before={total_before}) intersect removed indices "
        f"{sorted(removed_indices)} at {sorted(overlap)}."
    )

    # Budget-fit invariant: tokens_after must fit under the configured budget.
    budget = _CONTEXT_LIMIT - _RESERVE_TOKENS
    assert truncation_event.tokens_after <= budget, (
        f"Post-truncation message tokens ({truncation_event.tokens_after}) exceed the "
        f"configured budget ({budget} = context_limit {_CONTEXT_LIMIT} - reserve "
        f"{_RESERVE_TOKENS})."
    )

    await assert_result_satisfies(
        result.output or "",
        "The output identifies a conversation theme and answers the final question.",
    )


async def test_context_summarization_compresses(traced_emitter: InMemoryEmitter) -> None:
    llm = make_llm_client("anthropic")
    summarizer_llm = make_llm_client("anthropic")
    context_manager = ContextManager(
        context_limit=_CONTEXT_LIMIT,
        reserve_tokens=_RESERVE_TOKENS,
        threshold=0.8,
        truncation=None,  # force the summarization branch
        summarization=SummarizationPolicy(llm_client=summarizer_llm),
    )

    primed = _prime_messages()
    agent = ReActAgent(
        name="context-summarization-agent",
        llm_client=llm,
        emitter=traced_emitter,
        system_prompt="You are a helpful assistant reflecting on a long prior conversation.",
        tools=[],
        context_manager=context_manager,
        initial_messages=primed,
        max_iterations=2,
    )

    await run_with_retry(
        lambda: agent.run(
            "Summarize our conversation so far and then answer this final question: What was the main theme?"
        ),
        max_attempts=2,
    )

    # Summarization fired with non-empty payload.
    summarization_event = assert_trace_contains(
        traced_emitter,
        ContextSummarizationEvent,
        predicate=lambda e: e.messages_summarized > 0,
    )

    # Real reduction: summary tokens strictly less than original tokens.
    assert summarization_event.summary_tokens < summarization_event.original_tokens, (
        f"Expected summary_tokens < original_tokens (actual compression), got "
        f"summary_tokens={summarization_event.summary_tokens}, "
        f"original_tokens={summarization_event.original_tokens}."
    )

    # Summary carries content, not a placeholder/empty string.
    assert summarization_event.summary_text.strip(), (
        f"Expected non-empty summary_text, got: {summarization_event.summary_text!r}"
    )
