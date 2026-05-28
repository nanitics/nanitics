"""Behavioral continuity through multi-agent constructs.

Validates that the ``thread_key`` plumbing on ``AgentTool`` actually
carries a delegate's prior turns into a subsequent delegation by the
same coordinator. A coordinator iterates a drafter→critic→drafter loop
twice; the second drafter delegation hits the same drafter thread, so
the drafter sees its first draft as ITS OWN prior assistant turn (not
as injected context) and revises it.

This is the multi-agent analogue of ``thread_identity_replay.py``: it
proves that the per-construct ``thread_key`` shape is wired correctly
end-to-end through ``AgentTool.execute → Agent.run → ThreadStore`` and
that the substrate distinction (unwrapped replay vs. injected context)
holds across the delegation boundary.

Acceptance criteria:
  - The drafter's ``InMemoryThreadStore`` accumulates messages across
    the two delegations under the same thread key (post-loop length
    strictly greater than post-first-delegation length).
  - The trace contains at least two ``AgentStartEvent`` rows for the
    drafter where ``thread_key`` matches the AgentTool's configured
    key. The second drafter start reports
    ``replayed_message_count > 0``.
  - The drafter's second-draft output preserves the shop name from
    its first draft and contains no "no prior draft" disclaimer. The
    check is deterministic on the output text rather than an
    LLM-as-judge call, which has proven flaky (the judge sees only the
    second draft and cannot reliably evaluate whether a valid revision
    happened — see commit history for the prior judge prompt and its
    false-negative pattern).
"""

from __future__ import annotations

import pytest

from nanitics.composition import AgentTool, InMemoryThreadStore
from nanitics.infrastructure import AgentStartEvent
from nanitics.strategies import ReActAgent
from nanitics.tracing import InMemoryEmitter
from validation.helpers import (
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

_DRAFTER_KEY = "drafter-multi-agent-t1"


@pytest.mark.quick
async def test_multi_agent_thread_continuity(traced_emitter: InMemoryEmitter) -> None:
    thread_store = InMemoryThreadStore()

    drafter = ReActAgent(
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
        thread_store=thread_store,
        max_iterations=2,
    )

    critic = ReActAgent(
        name="critic",
        llm_client=make_llm_client("anthropic"),
        emitter=traced_emitter,
        system_prompt=(
            "You are an editor. Read a one-sentence pitch and reply with "
            "one concrete, actionable revision instruction (e.g. 'emphasize "
            "community' or 'make it shorter'). Keep your reply to one sentence."
        ),
        tools=[],
        max_iterations=2,
    )

    draft_tool = AgentTool(
        agent=drafter,
        emitter=traced_emitter,
        description="Drafts and revises one-sentence pitches for a coffee shop.",
        thread_key=_DRAFTER_KEY,
    )

    # First delegation — produces the initial draft.
    first_delegation = await run_with_retry(
        lambda: draft_tool.execute(
            task=(
                "Draft a one-sentence pitch for a coffee shop named 'Bean There'. Mention the shop name in the draft."
            ),
        ),
        max_attempts=2,
    )
    assert first_delegation.content, "Drafter must produce an initial draft."
    after_first = await thread_store.load(_DRAFTER_KEY)
    assert len(after_first) >= 2, (
        f"Expected drafter thread to advance after first delegation; got {len(after_first)} message(s)."
    )

    # Critic produces revision feedback (the critic does not share the drafter's thread).
    critique = await run_with_retry(
        lambda: critic.bind(traced_emitter).run(f"Pitch to review: {first_delegation.content}"),
        max_attempts=2,
    )
    assert critique.output, "Critic must produce feedback."

    # Second delegation — same drafter, same thread_key. The drafter
    # should see its first draft as its own prior assistant turn.
    second_delegation = await run_with_retry(
        lambda: draft_tool.execute(
            task=(
                f"Revise your previous draft. Editor feedback: {critique.output}. "
                "Keep it one sentence and keep mentioning the shop name."
            ),
        ),
        max_attempts=2,
    )
    assert second_delegation.content, "Drafter must produce a revised draft."

    after_second = await thread_store.load(_DRAFTER_KEY)
    assert len(after_second) > len(after_first), (
        f"Expected the drafter thread to grow across delegations; first={len(after_first)} second={len(after_second)}."
    )

    # The drafter's AgentStartEvents must carry the thread_key, and the
    # second one must report a non-zero replayed_message_count.
    drafter_starts = [e for e in traced_emitter.events if isinstance(e, AgentStartEvent) and e.agent_name == "drafter"]
    assert len(drafter_starts) >= 2, f"Expected at least 2 drafter AgentStartEvents; got {len(drafter_starts)}."
    for start in drafter_starts:
        assert start.thread_key == _DRAFTER_KEY, (
            f"Expected drafter start events to carry thread_key={_DRAFTER_KEY!r}; got {start.thread_key!r}."
        )
    assert drafter_starts[-1].replayed_message_count > 0, (
        "Expected the second drafter start to report a non-zero "
        "replayed_message_count — proving the thread prefix was loaded."
    )

    # Deterministic continuity check: the revised draft references the
    # shop name from the first delegation and contains no "no prior
    # draft" disclaimer. The structural trace assertions above already
    # prove the replay happened; this output-level check confirms the
    # model treated it as its own prior turn rather than falling back
    # to a disclaimer or producing an unrelated draft.
    output_lower = str(second_delegation.content or "").lower()
    assert "bean there" in output_lower, (
        "Second drafter delegation should reference the shop name from the first "
        f"delegation; got: {second_delegation.content!r}"
    )
    for phrase in _DISCLAIMER_PHRASES:
        assert phrase not in output_lower, (
            f"Second drafter delegation contains a 'no prior draft' disclaimer phrase "
            f"({phrase!r}); got: {second_delegation.content!r}"
        )
