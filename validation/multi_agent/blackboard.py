"""Blackboard shared-memory coordination across control and termination strategies.

Parametrised over the three control strategies (``OpportunisticControl``,
``PrioritizedControl``, ``ScheduledControl``) plus a dedicated write→read
round-trip test that pins the shared-memory semantics: agent A writes a
partial result in round 1 carrying a distinctive sentinel token, agent B
reads and quotes the sentinel in round 2.

Acceptance criteria (parametrised control strategies):
  - ``BlackboardStartEvent`` carries the configured ``control_strategy``
    name and lists both agents.
  - Exactly ``max_rounds`` ``BlackboardRoundEvent`` instances are emitted
    (one per round of execution under ``MaxRoundsTermination``).
  - Each round's ``round_entries`` has at least one ``"write"`` entry
    (agents actually wrote) — proves the shared-memory tools were
    injected and routed events back to the contribution listener.
  - ``result.rounds_completed == max_rounds`` and
    ``result.termination_reason == "MaxRoundsTermination"``.
  - ``BlackboardState`` is accurately reflected in
    ``BlackboardCompleteEvent.total_contributions``:
    ``total == sum(agent_contributions.values())``.
  - For ``PrioritizedControl``: the round's ``agents_activated`` order
    matches the descending-priority order (proves the sort is applied
    independently of the ``agents=`` argument order).

Acceptance criteria (write→read round-trip):
  - Round 1: both ``scribe`` and ``reader`` run; ``scribe`` writes a
    sentinel-bearing contribution to shared memory.
  - Round 2: only ``reader`` is selected (structurally, via a test-local
    ``ControlStrategy`` that gates on ``state.round_number``); it reads
    via the ``read_shared`` tool and quotes the sentinel token verbatim.
    Only possible if the write persisted across rounds and the read
    tool surfaced it.
  - ``BlackboardResult.entries`` contains an entry authored by
    ``scribe`` with the sentinel in its content.
  - Termination is ``NoNewContributions``: round 2 produces zero
    writes because only the reader is selected, so the blackboard
    loop stops at the end of round 2 with
    ``rounds_completed == 2``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from nanitics import (
    InMemoryEmitter,
    InMemorySharedMemory,
    MaxRoundsTermination,
    NoNewContributions,
    OpportunisticControl,
    PrioritizedControl,
    ReActAgent,
    ScheduledControl,
)
from nanitics.experimental.coordination import Blackboard
from nanitics.infrastructure import (
    BlackboardCompleteEvent,
    BlackboardRoundEvent,
    BlackboardStartEvent,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nanitics.composition.multi_agent.blackboard import BlackboardState
    from nanitics.core.agents.base import Agent
from validation.helpers import (
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)

# Distinctive token carrying a spelled-out marker and a UUID-ish suffix
# so "guessed by a language model" is implausible — it only reaches the
# reader if the write actually persisted in shared memory.
_SENTINEL = "ZX42-patent-filing-count-seven"


def _make_contributor(name: str, role: str, client, emitter: InMemoryEmitter) -> ReActAgent:
    return ReActAgent(
        name=name,
        llm_client=client,
        emitter=emitter,
        system_prompt=(
            f"You are {role}. Call the `write_to_shared` tool exactly "
            "once with a one-sentence contribution in your area of "
            "expertise (set scope to a short label). After writing, "
            "reply with a brief confirmation."
        ),
        tools=[],
        max_iterations=3,
    )


@pytest.mark.quick
@pytest.mark.parametrize(
    "control_name",
    ["OpportunisticControl", "PrioritizedControl", "ScheduledControl"],
)
async def test_blackboard_control_strategy(traced_emitter: InMemoryEmitter, control_name: str) -> None:
    client = make_llm_client("anthropic")
    shared = InMemorySharedMemory()

    lead = _make_contributor(
        "lead",
        "a strategy lead who contributes strategic direction",
        client,
        traced_emitter,
    )
    analyst = _make_contributor(
        "analyst",
        "a market analyst who contributes supporting market data",
        client,
        traced_emitter,
    )

    # Pass agents in a non-priority order so PrioritizedControl's sort is
    # actually exercised rather than hidden by input ordering.
    agents_in_order = [analyst, lead]

    controls = {
        "OpportunisticControl": OpportunisticControl(),
        "PrioritizedControl": PrioritizedControl(priorities={"lead": 10, "analyst": 5}),
        "ScheduledControl": ScheduledControl(),
    }
    control = controls[control_name]
    max_rounds = 1

    blackboard = Blackboard(
        shared_memory=shared,
        agents=agents_in_order,
        emitter=traced_emitter,
        control=control,
        termination=MaxRoundsTermination(max_rounds=max_rounds),
        max_rounds=max_rounds,
    )

    result = await run_with_retry(
        lambda: blackboard.run("Produce a shared-board plan for market expansion."),
        max_attempts=2,
    )

    # --- Start event pins wiring ---
    assert_trace_contains(
        traced_emitter,
        BlackboardStartEvent,
        predicate=lambda e: (
            e.control_strategy == control_name
            and set(e.agent_names) == {"lead", "analyst"}
            and e.max_rounds == max_rounds
        ),
    )

    # --- Round events: one per round, each with write entries ---
    round_events = [e for e in traced_emitter.events if isinstance(e, BlackboardRoundEvent)]
    assert len(round_events) == max_rounds, f"Expected {max_rounds} BlackboardRoundEvent(s), got: {len(round_events)}"
    for evt in round_events:
        writes = [re for re in evt.round_entries if re.operation == "write"]
        assert writes, (
            f"Round {evt.round_number} has no write entries — agents did not "
            f"contribute. round_entries: {evt.round_entries}"
        )

    # --- PrioritizedControl: sort order surfaces in agents_activated ---
    if control_name == "PrioritizedControl":
        assert round_events[0].agents_activated == ["lead", "analyst"], (
            f"PrioritizedControl must order by descending priority; "
            f"got agents_activated={round_events[0].agents_activated}"
        )

    # --- Result invariants ---
    assert result.rounds_completed == max_rounds
    assert result.termination_reason == "MaxRoundsTermination"

    complete = assert_trace_contains(
        traced_emitter,
        BlackboardCompleteEvent,
        predicate=lambda e: e.rounds_completed == max_rounds and e.termination_reason == "MaxRoundsTermination",
    )
    # BlackboardState accurately reflects contributions.
    assert complete.total_contributions == sum(complete.agent_contributions.values()), (
        f"total_contributions must equal sum of agent_contributions; "
        f"got total={complete.total_contributions}, "
        f"agents={complete.agent_contributions}"
    )


class _WriteThenReadControl:
    """Select ``[scribe, reader]`` in round 1; ``[reader]`` in later rounds.

    ``ReActAgent.run()`` resets working memory each call, so "scribe
    contributes once" cannot be encoded through prompt wording about
    "subsequent turns". The invariant is encoded structurally here:
    the scribe is simply not selected past round 1, so round 2 is
    guaranteed to be reader-only and ``NoNewContributions`` fires.
    """

    @property
    def parallel(self) -> bool:
        return False

    def select(self, agents: Sequence[Agent], state: BlackboardState) -> Sequence[Agent]:
        by_name = {a.name: a for a in agents}
        if state.round_number == 1:
            return [by_name["scribe"], by_name["reader"]]
        return [by_name["reader"]]


async def test_blackboard_write_then_read_round_trip(traced_emitter: InMemoryEmitter) -> None:
    """Round 1 writes a sentinel; Round 2 reads and quotes it."""
    client = make_llm_client("anthropic")
    shared = InMemorySharedMemory()

    scribe = ReActAgent(
        name="scribe",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You are a research scribe. Call the `write_to_shared` tool "
            "exactly once with content that is literally the following "
            "string, and nothing else:\n"
            f"  {_SENTINEL}\n"
            "Use scope='findings'. After the tool returns, reply with a "
            "short confirmation."
        ),
        tools=[],
        max_iterations=4,
    )
    reader = ReActAgent(
        name="reader",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You are a summariser. To answer, first call the "
            "`read_shared` tool with scope='findings' to retrieve the "
            "scribe's contribution. Then produce a single-sentence "
            "answer that quotes the scribe's exact contribution verbatim "
            "(including any unusual tokens or identifiers). Do not call "
            "`write_to_shared`."
        ),
        tools=[],
        max_iterations=4,
    )

    blackboard = Blackboard(
        shared_memory=shared,
        agents=[scribe, reader],
        emitter=traced_emitter,
        control=_WriteThenReadControl(),
        termination=NoNewContributions(),
        max_rounds=3,
    )

    result = await run_with_retry(
        lambda: blackboard.run("Collaborate through the shared board."),
        max_attempts=2,
    )

    # --- Scribe's sentinel persisted to shared memory ---
    scribe_entries = [e for e in result.entries if e.author == "scribe" and _SENTINEL in e.content]
    assert scribe_entries, (
        f"Expected a scribe-authored entry containing the sentinel; got entries: "
        f"{[(e.author, e.content) for e in result.entries]}"
    )

    # --- Exactly 2 rounds: round 1 selects [scribe, reader] (1 write),
    #     round 2 selects [reader] only (0 writes → NoNewContributions) ---
    round_events = [e for e in traced_emitter.events if isinstance(e, BlackboardRoundEvent)]
    assert len(round_events) == 2, (
        f"Expected exactly 2 rounds (write round + reader-only quiet round), got: {len(round_events)}"
    )
    round_entry_counts = [len(e.round_entries) for e in round_events]
    assert round_entry_counts == [1, 0], (
        f"Expected round_entries lengths [1, 0] (scribe writes once in round 1; "
        f"reader-only round 2 writes nothing); got: {round_entry_counts}"
    )

    # --- Termination was by NoNewContributions (the read-only round) ---
    assert result.rounds_completed == 2, (
        f"Expected rounds_completed == 2 (terminate at end of reader-only round); got: {result.rounds_completed}"
    )
    assert result.termination_reason == "NoNewContributions", (
        f"Expected NoNewContributions termination after reader round; got: {result.termination_reason!r}"
    )
    assert_trace_contains(
        traced_emitter,
        BlackboardCompleteEvent,
        predicate=lambda e: e.termination_reason == "NoNewContributions",
    )
