"""Behavioral continuity round-trip across two ``Agent.run`` calls.

Validates the load-bearing semantic claim from
``temp/sdk-thread-identity/design-rationale.md`` §4: a real LLM, given a
``thread_key``-keyed run on top of a populated
:class:`~nanitics.composition.threads.InMemoryThreadStore`, treats the
replayed prior assistant turn as ITS OWN prior turn — not as injected
context — and revises it in place. A wrapped-in-``<nanitics:context>``
implementation would tend to produce a generic "I don't see a previous
draft" response; an unwrapped replay produces a true revision.

Acceptance criteria:
  - Run 1 ``AgentStartEvent`` reports ``thread_key="t1"`` and
    ``replayed_message_count == 0`` (empty store).
  - Run 1's :class:`AgentResult` carries ``thread_key == "t1"`` and a
    non-empty ``output`` — the initial draft.
  - The store advances: the first run's user input and assistant
    response are persisted under ``"t1"``.
  - Run 2 ``AgentStartEvent`` reports ``thread_key="t1"`` and a
    ``replayed_message_count`` matching the post-run-1 store length.
  - Run 2 ``LLMRequestEvent`` carries the prior assistant turn as an
    unwrapped ``assistant``-role message — no ``<nanitics:context>``
    envelope around it.
  - Run 2's output references the shop name from the run-1 draft and
    contains no "no prior draft" disclaimer — proving the model
    treated the replay as its own prior turn. The check is
    deterministic on the output text rather than an LLM-as-judge call,
    which has proven flaky (the judge sees only ``result2.output`` and
    has no view into run 1, so it cannot reliably evaluate whether a
    valid revision happened — see commit history for the prior judge
    prompt and its false-negative pattern).
"""

from __future__ import annotations

import pytest

from nanitics.composition import InMemoryThreadStore
from nanitics.infrastructure import AgentStartEvent, LLMRequestEvent
from nanitics.strategies import ReActAgent
from nanitics.tracing import InMemoryEmitter
from validation.helpers import (
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)

_DISCLAIMER_PHRASES = (
    "i don't have",
    "i do not have",
    "no record",
    "no prior draft",
    "no previous draft",
    "i can't see",
    "i cannot see",
    "haven't seen",
    "have not seen",
)

_THREAD_KEY = "behavioral-continuity-t1"


@pytest.mark.quick
async def test_thread_identity_revises_prior_draft(traced_emitter: InMemoryEmitter) -> None:
    store = InMemoryThreadStore()
    agent = ReActAgent(
        name="drafter",
        llm_client=make_llm_client("anthropic"),
        emitter=traced_emitter,
        system_prompt=(
            "You are a copywriter. Write short, concrete drafts. "
            "When asked to revise a previous draft, treat your prior "
            "assistant turn as your own work and rewrite it directly — "
            "do not say you have no record of it."
        ),
        tools=[],
        thread_store=store,
        max_iterations=2,
    )

    # --- Run 1 ---
    result1 = await run_with_retry(
        lambda: agent.run(
            "Draft a one-sentence pitch for a coffee shop named 'Bean There'. Mention the shop name in the draft.",
            thread_key=_THREAD_KEY,
        ),
        max_attempts=2,
    )

    assert result1.thread_key == _THREAD_KEY
    assert result1.output, "Run 1 must produce an initial draft."

    assert_trace_contains(
        traced_emitter,
        AgentStartEvent,
        predicate=lambda e: e.thread_key == _THREAD_KEY and e.replayed_message_count == 0,
    )

    # The store advanced: the first run's user input and assistant turn are
    # persisted under the thread key.
    after_run1 = await store.load(_THREAD_KEY)
    assert len(after_run1) >= 2, f"Expected store to advance after run 1; got {len(after_run1)} message(s)."
    roles = [m.role for m in after_run1]
    assert "assistant" in roles, f"Expected an assistant message in the thread; got roles={roles!r}"

    # --- Run 2: revision against the same thread ---
    result2 = await run_with_retry(
        lambda: agent.run(
            "Revise the previous draft to emphasize community. Keep it one sentence.",
            thread_key=_THREAD_KEY,
        ),
        max_attempts=2,
    )

    assert result2.thread_key == _THREAD_KEY
    assert result2.output, "Run 2 must produce a revised draft."

    # The start event for run 2 reports the replay length.
    start_events = [e for e in traced_emitter.events if isinstance(e, AgentStartEvent)]
    assert len(start_events) >= 2
    assert start_events[-1].thread_key == _THREAD_KEY
    assert start_events[-1].replayed_message_count == len(after_run1)

    # Run 2's LLMRequestEvent must include the prior assistant turn as an
    # unwrapped ``assistant``-role message. A wrapped implementation would
    # have spliced it into a ``user`` message inside ``<nanitics:context>``.
    # Walk events in order, find the first LLMRequestEvent after the second
    # AgentStartEvent.
    second_start_idx = next(
        i for i, e in enumerate(traced_emitter.events) if isinstance(e, AgentStartEvent) and e is start_events[-1]
    )
    run2_first_request = next(e for e in traced_emitter.events[second_start_idx:] if isinstance(e, LLMRequestEvent))
    assistant_replays = [m for m in run2_first_request.messages if m.get("role") == "assistant" and m.get("content")]
    assert assistant_replays, (
        "Expected the replayed prior assistant turn to appear as an unwrapped "
        "assistant-role message in run 2's first LLM request."
    )
    for m in assistant_replays:
        content = m.get("content") or ""
        assert "<nanitics:context" not in content, (
            "Replayed thread messages must bypass the <nanitics:context> wrapper; "
            f"found wrapper bytes in assistant content: {content!r}"
        )

    # Deterministic continuity check: the revised output continues the
    # same pitch (mentions the shop name from run 1) and contains no
    # "no prior draft" disclaimer. A model that ignored the replayed
    # prior turn would either fall back to a disclaimer or drop the
    # shop name entirely. The structural proof of replay (above) plus
    # this output-level check together cover the load-bearing semantic
    # claim without the false-negative tax of an LLM-as-judge call.
    output_lower = str(result2.output or "").lower()
    assert "bean there" in output_lower, (
        f"Run 2 output should reference the shop name from run 1; got: {result2.output!r}"
    )
    for phrase in _DISCLAIMER_PHRASES:
        assert phrase not in output_lower, (
            f"Run 2 output contains a 'no prior draft' disclaimer phrase ({phrase!r}); got: {result2.output!r}"
        )
