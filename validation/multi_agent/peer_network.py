"""PeerNetwork discovery, cross-peer consultation, and budget exhaustion.

Two tests exercise the peer-network primitive against real LLMs:

* ``test_peer_network_basic_consultation`` — two peers (``researcher`` as
  entry, ``analyst`` as specialist). The researcher is instructed to
  call ``consult_analyst`` before answering, so a cross-peer
  consultation must appear in the trace. Pins peer discovery and the
  ``PeerConsultationEvent`` wiring.

* ``test_peer_network_budget_exhausted`` — ``max_invocations=1``. The
  entry peer is instructed to consult *two* other peers in sequence;
  the second attempt must be rejected because the shared budget is
  exhausted. Pins the ``PeerBudgetExceededError`` path by asserting
  only one ``PeerConsultationEvent`` fires and the budget's ``used``
  counter surfaces via ``PeerNetworkCompleteEvent``. The consultation
  graph is declared structurally via ``allowed_peers`` — coordinator
  can consult both experts; experts are leaf consultants. The test
  also pins capability-layer evidence: no ``consult_*``
  ``ToolInvokeEvent`` originates from either expert's span, proving
  the reentrancy path is closed at the tool-belt layer rather than by
  prompt instruction.

Acceptance criteria (basic):
  - ``PeerNetworkStartEvent`` lists both peers and the entry agent.
  - Trace contains a ``PeerConsultationEvent`` with
    ``from_agent == "researcher"``, ``to_agent == "analyst"``, and
    ``consultation_number == 1`` (the cross-peer call actually ran).
  - ``PeerNetworkCompleteEvent`` reports
    ``total_consultations == 1`` and ``"analyst" in agents_consulted``.
  - ``result.output`` is non-empty (researcher produced a final answer
    after consulting).

Acceptance criteria (budget exhausted):
  - ``max_invocations=1`` and the scenario asks for two consultations.
  - Exactly one ``PeerConsultationEvent`` fires (budget cap enforced).
  - ``PeerNetworkCompleteEvent.invocations_used == 1`` and
    ``total_consultations == 1`` — proves budget surfaces via events.
  - ``result.output`` is non-empty — the agent handled the budget-exceeded
    tool result and produced a partial answer, not a crash.
  - Zero ``ToolInvokeEvent`` with ``tool_name`` starting with
    ``consult_`` originates from either expert's span — the capability
    layer (tool-belt, not prompt) closes the reentrancy path.
"""

from __future__ import annotations

import pytest

from nanitics.infrastructure import (
    AgentStartEvent,
    PeerConsultationEvent,
    PeerNetworkCompleteEvent,
    PeerNetworkStartEvent,
    ToolInvokeEvent,
)
from nanitics.specialized import (
    PeerNetwork,
    PeerSpec,
)
from nanitics.tracing import InMemoryEmitter
from validation.helpers import (
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)


async def test_peer_network_basic_consultation(traced_emitter: InMemoryEmitter) -> None:
    client = make_llm_client("anthropic")

    network = PeerNetwork(
        peers=[
            PeerSpec(
                name="researcher",
                description="Synthesises market research from specialist input.",
                llm_client=client,
                system_prompt=(
                    "You are a market researcher. Before answering the user, "
                    "call `consult_analyst` exactly once with a focused "
                    "question about key risks. After receiving the analyst's "
                    "response, write a two-sentence synthesis that explicitly "
                    "incorporates the analyst's input."
                ),
                tools=[],
                max_iterations=4,
            ),
            PeerSpec(
                name="analyst",
                description="Risk analyst — assesses quantitative risks.",
                llm_client=client,
                system_prompt=("You are a risk analyst. Given a question, produce a two-sentence risk assessment."),
                tools=[],
                max_iterations=2,
                allowed_peers=[],  # Leaf consultant — no downstream consultation.
            ),
        ],
        emitter=traced_emitter,
        max_invocations=5,
    )

    result = await run_with_retry(
        lambda: network.run("researcher", "Should we expand into the Southeast Asian market?"),
        max_attempts=2,
    )

    assert result.output, f"Expected non-empty final output; got: {result.output!r}"

    assert_trace_contains(
        traced_emitter,
        PeerNetworkStartEvent,
        predicate=lambda e: e.entry_agent == "researcher" and set(e.peer_names) == {"researcher", "analyst"},
    )

    assert_trace_contains(
        traced_emitter,
        PeerConsultationEvent,
        predicate=lambda e: e.from_agent == "researcher" and e.to_agent == "analyst" and e.consultation_number == 1,
    )

    assert_trace_contains(
        traced_emitter,
        PeerNetworkCompleteEvent,
        predicate=lambda e: e.total_consultations == 1 and "analyst" in e.agents_consulted,
    )


@pytest.mark.quick
async def test_peer_network_budget_exhausted(traced_emitter: InMemoryEmitter) -> None:
    """Force the ``PeerBudgetExceededError`` path via ``max_invocations=1``."""
    client = make_llm_client("anthropic")

    network = PeerNetwork(
        peers=[
            PeerSpec(
                name="coordinator",
                description="Coordinates expert input for project evaluation.",
                llm_client=client,
                system_prompt=(
                    "You coordinate expert assessments. To answer the user, "
                    "first call `consult_expert_a` for technical feasibility, "
                    "then call `consult_expert_b` for market demand. "
                    "After both consultations (or after exhausting your "
                    "consultation budget), write a final two-sentence "
                    "summary. If you cannot consult a peer because the "
                    "budget is exhausted, acknowledge that in your summary "
                    "and still produce a final answer — do not retry."
                ),
                tools=[],
                max_iterations=6,
                allowed_peers=["expert_a", "expert_b"],
            ),
            PeerSpec(
                name="expert_a",
                description="Technical feasibility expert.",
                llm_client=client,
                system_prompt=("You assess technical feasibility in one concise sentence."),
                tools=[],
                max_iterations=2,
                allowed_peers=[],  # Leaf consultant — no reentrant consultation path.
            ),
            PeerSpec(
                name="expert_b",
                description="Market demand analyst.",
                llm_client=client,
                system_prompt=("You assess market demand in one concise sentence."),
                tools=[],
                max_iterations=2,
                allowed_peers=[],  # Leaf consultant — no reentrant consultation path.
            ),
        ],
        emitter=traced_emitter,
        max_invocations=1,  # --- Forces budget exhaustion on second call ---
    )

    proposal = (
        "Evaluate this product proposal: a subscription-based home hydroponics "
        "kit targeting urban apartment renters in Tokyo and Seoul, priced at "
        "¥8,000/month with mobile-app-controlled nutrient dosing. Consult both "
        "experts before summarising — do not rely on your own knowledge."
    )
    result = await run_with_retry(
        lambda: network.run("coordinator", proposal),
        max_attempts=2,
    )

    assert result.output, (
        f"Coordinator must still produce a final answer even after budget exhaustion; got: {result.output!r}"
    )

    consultation_events = [e for e in traced_emitter.events if isinstance(e, PeerConsultationEvent)]
    assert len(consultation_events) == 1, (
        f"Expected exactly one PeerConsultationEvent (budget=1), got: "
        f"{len(consultation_events)} — {[(e.from_agent, e.to_agent) for e in consultation_events]}"
    )

    assert_trace_contains(
        traced_emitter,
        PeerNetworkCompleteEvent,
        predicate=lambda e: e.invocations_used == 1 and e.total_consultations == 1,
    )

    # --- Capability-layer evidence: under the restricted graph
    # (experts declared as leaf consultants with allowed_peers=[]), no
    # consult_* tool exists on either expert, so no ToolInvokeEvent for
    # a consult_* tool should originate from an expert's span. This is
    # the trace-level proof that the W1 fix closed the reentrancy path
    # at the capability layer, not via prompt instruction.
    expert_start_spans = {
        e.span_id
        for e in traced_emitter.events
        if isinstance(e, AgentStartEvent) and e.agent_name in {"expert_a", "expert_b"}
    }
    offgraph_invokes = [
        e
        for e in traced_emitter.events
        if isinstance(e, ToolInvokeEvent)
        and e.tool_name.startswith("consult_")
        and e.parent_span_id in expert_start_spans
    ]
    assert not offgraph_invokes, (
        f"Experts must not invoke consult_* tools under the restricted graph; got: "
        f"{[(e.tool_name, e.parent_span_id) for e in offgraph_invokes]}"
    )
