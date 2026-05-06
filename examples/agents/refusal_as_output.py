"""Refusal-as-output: typed refusal artifacts via deterministic routing + rationale agent + assembly.

Demonstrates the application-layer composition where an upstream record that
the agent should not handle (out-of-scope, missing prerequisite, non-canonical
category) is captured as a typed `RefusalToDraft` rather than as natural-language
prose. Three parts: deterministic routing onto a closed `RefusalTriggerKind`
typology (no LLM); a `ReasoningAgent` with `output_schema=RefusalRationaleDraft`
that produces the rationale; an assembly step that composes the typed artifact.

The pattern is application-layer composition on top of `ReasoningAgent` — no
SDK primitive is added. It is the pre-pattern (typed-output-then-deterministic-dispatch)
applied to the refusal case: the LLM provides judgement on the rationale, the
application provides routing and assembly.

Related guide: docs/guides/agent-types.md
"""

import asyncio
from dataclasses import dataclass
from typing import Literal, get_args

from pydantic import BaseModel, Field

from examples.helpers import make_emitter, make_response
from nanitics import MockLLMClient, ReasoningAgent

# --- Closed-vocabulary `Literal` typology (the C12 shape) ---

RefusalTriggerKind = Literal[
    "out_of_scope",
    "missing_prerequisite",
    "non_canonical",
    "other",
]
"""Closed set of reasons the agent should not handle a record."""

REFUSAL_TRIGGER_KINDS: frozenset[str] = frozenset(get_args(RefusalTriggerKind))
"""Membership-check helper for application-side validation outside the agent path."""


# --- Typed artifacts ---


class RefusalRationaleDraft(BaseModel):
    """Structured output of the rationale-generating ReasoningAgent."""

    kind: RefusalTriggerKind = Field(description="The closed-vocabulary refusal trigger; must match the routed kind.")
    summary: str = Field(description="One-sentence summary of why this record cannot be handled.")
    suggested_next_step: str = Field(
        description="What the upstream system should do instead — re-route, request more data, escalate."
    )


class RefusalToDraft(BaseModel):
    """Assembled refusal artifact: routed kind + agent rationale + record metadata."""

    record_id: str
    kind: RefusalTriggerKind
    rationale: RefusalRationaleDraft
    routed_by: Literal["deterministic"] = "deterministic"


@dataclass(frozen=True)
class UpstreamRecord:
    id: str
    payload: str
    has_required_metadata: bool


# --- Part 1: Deterministic upstream-record routing (no LLM) ---


def route_record(record: UpstreamRecord) -> RefusalTriggerKind | None:
    """Pure routing: returns None when the record is in scope.

    A record is out-of-scope, lacks prerequisite metadata, etc. — the routing
    is a fixed function of the record fields and never calls the LLM.
    """
    if not record.has_required_metadata:
        return "missing_prerequisite"
    if "[OUT-OF-SCOPE]" in record.payload:
        return "out_of_scope"
    if "[NON-CANONICAL]" in record.payload:
        return "non_canonical"
    return None


# --- Part 3: Assembly into the typed artifact ---


def assemble_refusal(
    *,
    record_id: str,
    kind: RefusalTriggerKind,
    rationale: RefusalRationaleDraft,
) -> RefusalToDraft:
    """Pure assembly: combines deterministic routing output with the agent rationale."""
    return RefusalToDraft(record_id=record_id, kind=kind, rationale=rationale)


async def main() -> None:
    # --- Section 1: In-scope record — routing skips the refusal pipeline ---
    print("--- Section 1: In-scope record — routing skips the refusal pipeline ---")

    in_scope = UpstreamRecord(
        id="rec-001",
        payload="Standard support question about pricing tiers.",
        has_required_metadata=True,
    )

    # Deterministic routing returns None — no agent call needed.
    kind = route_record(in_scope)
    assert kind is None, "In-scope record should bypass the refusal pipeline."

    # No LLM client constructed for this path — the pre-pattern's defining property:
    # the LLM is invoked only when judgement is needed.
    print(f"  Record {in_scope.id}: routing returned None → in-scope, no refusal artifact")
    print("✓ Deterministic routing avoids the LLM for in-scope records")

    # --- Section 2: Out-of-scope record — full three-part pipeline ---
    print("\n--- Section 2: Out-of-scope record — routing → rationale agent → assembly ---")

    out_of_scope = UpstreamRecord(
        id="rec-002",
        payload="[OUT-OF-SCOPE] Question about a third-party product unrelated to our SDK.",
        has_required_metadata=True,
    )

    kind = route_record(out_of_scope)
    assert kind == "out_of_scope", "Routing identifies the closed-vocabulary kind."
    assert kind in REFUSAL_TRIGGER_KINDS, "The routed kind is in the closed vocabulary."

    client = MockLLMClient(
        responses=[
            make_response(
                '{"kind": "out_of_scope", '
                '"summary": "The record references a third-party product outside this SDK\'s scope.", '
                '"suggested_next_step": "Re-route to the third-party vendor\'s support channel."}'
            ),
        ]
    )
    rationale_agent = ReasoningAgent(
        name="refusal-rationale",
        llm_client=client,
        emitter=make_emitter("refusal-as-output"),
        system_prompt=(
            "Produce a typed RefusalRationaleDraft explaining why the upstream record "
            "cannot be handled by this agent. Match the routed `kind` exactly."
        ),
        output_schema=RefusalRationaleDraft,
    )

    result = await rationale_agent.run(out_of_scope.payload)
    assert result.parsed is not None
    rationale = result.parsed
    assert isinstance(rationale, RefusalRationaleDraft)
    assert rationale.kind == kind, "The rationale's kind matches the routed kind."

    refusal = assemble_refusal(record_id=out_of_scope.id, kind=kind, rationale=rationale)
    assert isinstance(refusal, RefusalToDraft)
    assert refusal.record_id == "rec-002"
    assert refusal.kind == "out_of_scope"
    assert refusal.routed_by == "deterministic"

    # Defining property: one LLM call across the whole refusal pipeline (rationale only).
    assert len(client.calls) == 1, "Routing and assembly are pure Python; only the rationale calls the LLM."

    print(f"  Record {refusal.record_id}: kind={refusal.kind}")
    print(f"  Rationale: {refusal.rationale.summary}")
    print(f"  Suggested next step: {refusal.rationale.suggested_next_step}")
    print(f"  LLM calls: {len(client.calls)} (rationale only — routing and assembly are pure)")
    print("✓ Three-part composition: deterministic routing → rationale agent → typed assembly")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
