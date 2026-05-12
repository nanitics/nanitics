"""Map-reduce: three Python-async questions fanned out to real agents.

Validates the :class:`MapReduce` workflow splitting a list of questions,
mapping each to a real :class:`ReActAgent`, and reducing the answers into
one combined string. The map-phase assertions pin the primitive's
distinguishing property: **each of the three distinct input items
produced its own shard output** — not a single output replayed three
times or a broken splitter feeding the whole list into one map call.

Acceptance criteria:
  - ``result.metadata["total_items"] == 3`` and ``total_steps_executed == 3``.
  - Exactly three ``WorkflowStepCompleteEvent`` events (one per mapped
    item; the reducer emits none).
  - The three events' ``step_index`` values are exactly ``{0, 1, 2}``
    — proves ordered fan-out with no collapsed/lost shards.
  - Each shard's ``step_output`` is non-empty.
  - Each shard's ``step_output`` individually answers *its own*
    question (shard-0 → asyncio, shard-1 → coroutine, shard-2 →
    event loop) — proves the splitter fed three distinct inputs to
    three distinct map invocations.
  - Final output answers all three Python async questions.
"""

from __future__ import annotations

import pytest

from nanitics import (
    AgentStep,
    InMemoryEmitter,
    ReActAgent,
    StepResult,
)
from nanitics.experimental import MapReduce
from nanitics.infrastructure import WorkflowStepCompleteEvent
from validation.helpers import (
    assert_result_satisfies,
    make_llm_client,
    run_with_retry,
)

QUESTIONS = [
    "What is asyncio?",
    "What is a coroutine?",
    "What is an event loop?",
]

# Per-shard acceptance criteria — indexed by step_index.
SHARD_CRITERIA = [
    "The output primarily answers 'What is asyncio?' — describing Python's asyncio library or module.",
    "The output primarily answers 'What is a coroutine?' — describing coroutines as a concept or construct.",
    "The output primarily answers 'What is an event loop?' — describing the event loop concept.",
]


def _combine(results: list[StepResult]) -> str:
    return "\n\n".join(str(r.output) for r in results if r.output is not None)


@pytest.mark.quick
async def test_map_reduce_python_async_qa(traced_emitter: InMemoryEmitter) -> None:
    answering_agent = ReActAgent(
        name="python-answerer",
        llm_client=make_llm_client("anthropic"),
        emitter=traced_emitter,
        system_prompt="Answer the user's Python question concisely in 1-2 sentences.",
        tools=[],
        max_iterations=2,
    )

    workflow = MapReduce(
        name="python-qa",
        step=AgentStep(answering_agent),
        splitter=lambda x: list(x),
        reducer=_combine,
        emitter=traced_emitter,
    )

    result = await run_with_retry(
        lambda: workflow.execute(QUESTIONS),
        max_attempts=2,
    )

    # --- Item / step counts ---
    assert result.metadata["total_items"] == 3, f"Expected 3 items, got: {result.metadata['total_items']}"
    assert result.metadata["total_steps_executed"] == 3, (
        f"Expected 3 steps, got: {result.metadata['total_steps_executed']}"
    )

    # --- Exactly three step-complete events (one per mapped item) ---
    step_events = [e for e in traced_emitter.events if isinstance(e, WorkflowStepCompleteEvent)]
    assert len(step_events) == 3, f"Expected exactly 3 WorkflowStepCompleteEvent, got: {len(step_events)}"

    # --- step_index covers {0, 1, 2} — no collapsed or lost shards ---
    indices = {e.step_index for e in step_events}
    assert indices == {0, 1, 2}, f"Expected step_index set == {{0, 1, 2}}, got: {indices}"

    # --- Each shard output is non-empty ---
    events_by_index = {e.step_index: e for e in step_events}
    for idx in (0, 1, 2):
        shard_output = events_by_index[idx].step_output
        assert shard_output, f"Expected truthy step_output for shard {idx}, got: {shard_output!r}"
        assert shard_output.strip(), f"Expected non-empty step_output for shard {idx}, got: {shard_output!r}"

    # --- Per-shard content check: each shard answers ITS OWN question.
    # This is the distinguishing assertion — a splitter that fed the whole
    # list into every map call, or that replayed one item three times,
    # cannot satisfy all three shard criteria simultaneously.
    for idx, criterion in enumerate(SHARD_CRITERIA):
        shard_output = events_by_index[idx].step_output or ""
        await assert_result_satisfies(shard_output, criterion)

    # --- Reduced final output covers all three topics ---
    await assert_result_satisfies(
        str(result.output or ""),
        "The output answers all three Python async questions (asyncio, coroutine, event loop).",
    )
