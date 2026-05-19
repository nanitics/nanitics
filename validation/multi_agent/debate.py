"""Debate between two real debaters resolved by a real-LLM judge.

A pro debater and a con debater each argue opposing positions over
``max_rounds=2`` rounds. A ``JudgeResolution`` with a third real agent
as judge evaluates the transcript and produces a free-form verdict. The
shared emitter captures the full event sequence: ``DebateStartEvent``,
one ``DebateArgumentEvent`` per argument, a ``DebateResolutionEvent``,
and ``DebateCompleteEvent``.

Acceptance criteria:
  - Trace contains a ``DebateStartEvent`` whose ``positions`` map pins
    pro/con positions and ``resolution_strategy == "JudgeResolution"``.
  - Trace contains exactly four ``DebateArgumentEvent`` instances
    (two debaters across two rounds), with two events at ``round == 1``
    and two at ``round == 2`` — proving round 2 actually executed.
  - Trace contains a ``DebateResolutionEvent`` with non-empty
    ``reasoning`` matching ``result.resolution.reasoning``.
  - Trace contains a ``DebateCompleteEvent`` with
    ``rounds_completed == 2``, ``total_arguments == 4``, and
    ``termination_reason == "max_rounds"``.
  - ``result.rounds_completed == 2``.
  - Round 1 transcript is ordered pro → con.
  - Synthesis evaluates both positions and produces a coherent verdict.
  - Round 2 argument from con engages with pro's round 1 argument
    (inter-round context sharing).
"""

from __future__ import annotations

import pytest

from nanitics import (
    InMemoryEmitter,
    ReActAgent,
)
from nanitics.infrastructure import (
    DebateArgumentEvent,
    DebateCompleteEvent,
    DebateResolutionEvent,
    DebateStartEvent,
)
from nanitics.specialized import (
    Debate,
    Debater,
    JudgeResolution,
)
from validation.helpers import (
    assert_result_satisfies,
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)


@pytest.mark.quick
async def test_debate_two_debaters_and_judge(traced_emitter: InMemoryEmitter) -> None:
    client = make_llm_client("anthropic")

    pro_agent = ReActAgent(
        name="pro",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You argue for prioritizing safety in AI development. Make concise, "
            "focused points — two or three sentences per round."
        ),
        tools=[],
        max_iterations=2,
    )
    con_agent = ReActAgent(
        name="con",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You argue for prioritizing speed in AI development. Make concise, "
            "focused points — two or three sentences per round."
        ),
        tools=[],
        max_iterations=2,
    )
    judge_agent = ReActAgent(
        name="judge",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You are an impartial debate judge. Read the transcript and produce "
            "a short verdict paragraph that evaluates both sides."
        ),
        tools=[],
        max_iterations=2,
    )

    debate = Debate(
        debaters=[
            Debater(agent=pro_agent, position="for prioritizing safety"),
            Debater(agent=con_agent, position="for prioritizing speed"),
        ],
        emitter=traced_emitter,
        resolution=JudgeResolution(judge=judge_agent),
        max_rounds=2,
    )

    result = await run_with_retry(
        lambda: debate.run("Should AI development prioritize safety over speed?"),
        max_attempts=2,
    )

    # --- Start-event payload pins wiring ---
    assert_trace_contains(
        traced_emitter,
        DebateStartEvent,
        predicate=lambda e: (
            e.resolution_strategy == "JudgeResolution"
            and e.positions
            == {
                "pro": "for prioritizing safety",
                "con": "for prioritizing speed",
            }
            and e.max_rounds == 2
        ),
    )

    # --- Argument events: exactly 4 (2 debaters x 2 rounds), per-round breakdown ---
    argument_events = [e for e in traced_emitter.events if isinstance(e, DebateArgumentEvent)]
    assert len(argument_events) == 4, (
        f"Expected exactly 4 DebateArgumentEvent instances (2 debaters x 2 rounds), got: {len(argument_events)}"
    )
    round_1_events = [e for e in argument_events if e.round == 1]
    round_2_events = [e for e in argument_events if e.round == 2]
    assert len(round_1_events) == 2, f"Expected 2 round-1 argument events, got: {len(round_1_events)}"
    assert len(round_2_events) == 2, (
        f"Expected 2 round-2 argument events (proves multi-round execution), got: {len(round_2_events)}"
    )

    # --- Resolution event payload ---
    resolution_event = assert_trace_contains(
        traced_emitter,
        DebateResolutionEvent,
        predicate=lambda e: (
            bool(e.reasoning) and e.reasoning == result.resolution.reasoning and e.rounds_completed == 2
        ),
    )
    assert resolution_event.reasoning, "DebateResolutionEvent.reasoning must be non-empty"

    # --- Complete event payload ---
    assert_trace_contains(
        traced_emitter,
        DebateCompleteEvent,
        predicate=lambda e: e.rounds_completed == 2 and e.total_arguments == 4 and e.termination_reason == "max_rounds",
    )

    # --- Result invariants ---
    assert result.rounds_completed == 2, f"Expected result.rounds_completed == 2, got: {result.rounds_completed}"
    assert len(result.transcript) == 4, f"Expected transcript of length 4, got: {len(result.transcript)}"

    # --- Transcript ordering: round 1 pro → con ---
    assert result.transcript[0].agent_name == "pro", (
        f"Expected round 1 transcript[0] to be 'pro', got: {result.transcript[0].agent_name!r}"
    )
    assert result.transcript[0].round == 1, (
        f"Expected round 1 transcript[0] round to be 1, got: {result.transcript[0].round}"
    )
    assert result.transcript[1].agent_name == "con", (
        f"Expected round 1 transcript[1] to be 'con', got: {result.transcript[1].agent_name!r}"
    )
    assert result.transcript[1].round == 1, (
        f"Expected round 1 transcript[1] round to be 1, got: {result.transcript[1].round}"
    )

    # --- Fuzzy verdict ---
    await assert_result_satisfies(
        result.resolution.synthesis,
        ("The output evaluates arguments for both safety and speed in AI development and produces a coherent verdict."),
    )

    # --- Inter-round context sharing: round-2 con engages with pro's round-1 argument ---
    pro_round_1 = next(e for e in round_1_events if e.agent_name == "pro")
    con_round_2 = next(e for e in round_2_events if e.agent_name == "con")
    await assert_result_satisfies(
        (f"Pro's round 1 argument:\n{pro_round_1.argument}\n\nCon's round 2 argument:\n{con_round_2.argument}"),
        (
            "Con's round 2 argument addresses, rebuts, or otherwise engages with "
            "specific claims, points, or framing from pro's round 1 argument — "
            "rather than restating an opening case in isolation."
        ),
    )
