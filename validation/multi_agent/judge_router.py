"""JudgeRouter against a real provider.

Two parametrised tests exercise ``JudgeRouter`` end-to-end against a real
LLM. They assert that the comparative-judgment path correctly
discriminates a domain-mismatched specialist pool and that the
calibration-anchor template body actually reaches the judge LLM.

Acceptance criteria:

* ``test_judge_router_comparative_winner`` — given four specialists where
  only ``billing-specialist`` is the natural fit for an "invoice amount
  wrong" task, the judge ranks ``billing-specialist`` first with
  ``confidence >= 0.7``. Pins that the comparative-judgment primitive
  produces an honest ranking under a real provider, that the winning
  agent actually executes, and that the trace carries one start, four
  rankings, one allocated, one complete.

* ``test_judge_router_carries_calibrated_template`` — when constructed
  with the explicit
  :data:`DEFAULT_CALIBRATED_JUDGE_PROMPT_TEMPLATE`, the prompt body
  observed by the judge LLM contains all four calibration anchors and
  the substituted ``{participants}`` / ``{task}`` slots. Pins that the
  template injection point survives end-to-end.

Cost ceiling per parametrisation: 5k input + 2.5k output tokens.
Total per script run: 20k input + 10k output tokens.
"""

from __future__ import annotations

from typing import Any

import pytest

from nanitics import (
    DEFAULT_CALIBRATED_JUDGE_PROMPT_TEMPLATE,
    BiddableAgent,
    InMemoryEmitter,
    JudgeRouter,
    ReActAgent,
)
from nanitics.infrastructure import (
    JudgeAllocatedEvent,
    JudgeRankingEvent,
    JudgeRoutingCompleteEvent,
    JudgeRoutingStartEvent,
    LLMRequestEvent,
)
from nanitics.specialized import FixedBidGenerator
from validation.helpers import (
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)

_PARTICIPANT_SPECS = [
    (
        "billing-specialist",
        "Billing specialist for invoices, refunds, and payment disputes.",
    ),
    (
        "technical-support",
        "Technical support for software bugs, outages, and platform errors.",
    ),
    (
        "account-manager",
        "Account manager for subscription tier changes and account ownership.",
    ),
    (
        "data-scientist",
        "Data scientist for churn analysis, statistical modelling, and ML.",
    ),
]


def _make_participants(client: Any, emitter: InMemoryEmitter) -> list[BiddableAgent]:
    participants: list[BiddableAgent] = []
    for name, description in _PARTICIPANT_SPECS:
        agent = ReActAgent(
            name=name,
            llm_client=client,
            emitter=emitter,
            system_prompt=f"{description} Answer concisely in one sentence.",
            tools=[],
            max_iterations=2,
        )
        # bid_generator is unused by JudgeRouter — the type stays shared so
        # adopters can swap Bidding ↔ JudgeRouter without rebuilding agents.
        participants.append(
            BiddableAgent(agent=agent, bid_generator=FixedBidGenerator(confidence=0.0)),
        )
    return participants


@pytest.mark.quick
async def test_judge_router_comparative_winner(traced_emitter: InMemoryEmitter) -> None:
    """Real-provider judge ranks billing-specialist first for an invoice dispute."""
    client = make_llm_client("anthropic")
    participants = _make_participants(client, traced_emitter)

    router = JudgeRouter(
        participants=participants,
        judge_llm=client,
        emitter=traced_emitter,
    )

    task = "My invoice shows the wrong amount — please correct it."
    result = await run_with_retry(lambda: router.run(task), max_attempts=2)

    # --- Result invariants ---
    assert result.allocated is True, f"Expected allocation to succeed, got: {result.allocated}"
    assert result.judge_error is None, f"Unexpected judge_error: {result.judge_error!r}"
    assert result.winner is not None
    assert result.winner.agent_name == "billing-specialist", (
        f"Expected billing-specialist to win invoice dispute; got "
        f"{result.winner.agent_name!r} with ranking: "
        f"{[(c.agent_name, c.confidence) for c in result.ranking]}"
    )
    assert result.winner.confidence >= 0.7, (
        f"Top-ranked confidence below 0.7 — calibration anchors not biting. Got: {result.winner.confidence}"
    )
    assert len(result.ranking) == len(_PARTICIPANT_SPECS), (
        f"Expected {len(_PARTICIPANT_SPECS)} ranked candidates, got: {len(result.ranking)}"
    )
    expected_names = {name for name, _ in _PARTICIPANT_SPECS}
    assert {c.agent_name for c in result.ranking} == expected_names, (
        f"Ranking names did not match participant set; got: {[c.agent_name for c in result.ranking]}"
    )
    for candidate in result.ranking:
        assert 0.0 <= candidate.confidence <= 1.0, (
            f"Confidence out of [0,1] for {candidate.agent_name!r}: {candidate.confidence}"
        )
    assert result.execution_result, f"Winning agent must execute; got execution_result={result.execution_result!r}"
    assert result.execution_error is None

    # --- Trace invariants: start → N rankings → allocated → complete ---
    assert_trace_contains(
        traced_emitter,
        JudgeRoutingStartEvent,
        predicate=lambda e: set(e.participant_names) == expected_names,
    )

    ranking_events = [e for e in traced_emitter.events if isinstance(e, JudgeRankingEvent)]
    assert len(ranking_events) == len(_PARTICIPANT_SPECS), (
        f"Expected {len(_PARTICIPANT_SPECS)} JudgeRankingEvent instances, got: {len(ranking_events)}"
    )
    assert [e.rank for e in ranking_events] == list(range(len(_PARTICIPANT_SPECS))), (
        f"Ranking events not emitted in 0..N-1 order; got: {[e.rank for e in ranking_events]}"
    )

    assert_trace_contains(
        traced_emitter,
        JudgeAllocatedEvent,
        predicate=lambda e: e.winner == "billing-specialist" and e.total_candidates == len(_PARTICIPANT_SPECS),
    )
    assert_trace_contains(
        traced_emitter,
        JudgeRoutingCompleteEvent,
        predicate=lambda e: e.winner == "billing-specialist" and e.allocated is True,
    )


async def test_judge_router_carries_calibrated_template(
    traced_emitter: InMemoryEmitter,
) -> None:
    """Calibrated template body and substituted slots reach the judge LLM."""
    client = make_llm_client("anthropic")
    participants = _make_participants(client, traced_emitter)

    router = JudgeRouter(
        participants=participants,
        judge_llm=client,
        emitter=traced_emitter,
        prompt_template=DEFAULT_CALIBRATED_JUDGE_PROMPT_TEMPLATE,
    )

    task = "My invoice shows the wrong amount — please correct it."
    await run_with_retry(lambda: router.run(task), max_attempts=2)

    # The InstrumentedLLMClient(label="judge") wraps the judge call and
    # emits an LLMRequestEvent carrying the rendered prompt body. Find the
    # judge-labelled request and assert the calibration anchors and slot
    # substitutions are present.
    judge_requests = [e for e in traced_emitter.events if isinstance(e, LLMRequestEvent) and e.label == "judge"]
    assert len(judge_requests) >= 1, f"Expected at least one judge-labelled LLMRequestEvent; got: {len(judge_requests)}"
    rendered = judge_requests[0].messages[0]["content"]

    for anchor in (
        "0.9 = uniquely positioned",
        "0.7 = capable",
        "0.4 = adjacent",
        "0.0 = out of scope",
    ):
        assert anchor in rendered, f"Calibration anchor not present in rendered judge prompt: {anchor!r}"

    # Participant block and task slot substituted.
    for name, _ in _PARTICIPANT_SPECS:
        assert name in rendered, f"Participant {name!r} missing from rendered prompt"
    assert task in rendered, "Task not substituted into rendered prompt"
