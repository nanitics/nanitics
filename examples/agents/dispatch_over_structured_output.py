"""Pre-pattern: single agent emits typed structured output, then deterministic Python dispatches.

Demonstrates the pre-pattern check that comes before any multi-agent or
workflow primitive. One LLM-driven `ReasoningAgent` with an `output_schema`
produces a typed `DispatchDecision`. Plain Python `await`s a pure dispatcher
that consumes the typed output and routes to one of several outcomes — no
second LLM call, no shared state, no concurrency primitive needed.

If your second stage is a pure function of typed input, the agent provides
judgment and the dispatcher provides routing. No multi-agent pattern is
warranted.

Related guide: docs/guides/multi-agent-foundations.md
"""

import asyncio
from typing import Literal

from pydantic import BaseModel, Field

from examples.helpers import make_emitter, make_response
from nanitics import MockLLMClient, ReasoningAgent


class DispatchDecision(BaseModel):
    """Typed output of the upstream agent; consumed by the deterministic dispatcher."""

    kind: Literal["bug_report", "feature_request", "billing_question", "other"] = Field(
        description="The category the agent decided this input falls into."
    )
    summary: str = Field(description="A one-line summary of the input.")
    priority: Literal["low", "medium", "high"] = Field(description="The priority level for the dispatched outcome.")


# Pure async follow-on functions, one per `kind`. No LLM, no I/O, no shared state.


async def _handle_bug_report(decision: DispatchDecision) -> str:
    return f"Bug filed [{decision.priority}]: {decision.summary}"


async def _handle_feature_request(decision: DispatchDecision) -> str:
    return f"Feature request logged [{decision.priority}]: {decision.summary}"


async def _handle_billing_question(decision: DispatchDecision) -> str:
    return f"Billing ticket opened [{decision.priority}]: {decision.summary}"


async def _handle_other(decision: DispatchDecision) -> str:
    return f"Triage queue [{decision.priority}]: {decision.summary}"


async def dispatch(decision: DispatchDecision) -> str:
    """Deterministic Python dispatcher — pattern-matches on the typed `kind` field.

    No LLM call, no shared state. The exhaustive `match` over the `Literal[...]`
    arms is verified by `mypy` strict mode.
    """
    match decision.kind:
        case "bug_report":
            return await _handle_bug_report(decision)
        case "feature_request":
            return await _handle_feature_request(decision)
        case "billing_question":
            return await _handle_billing_question(decision)
        case "other":
            return await _handle_other(decision)


async def main() -> None:
    # --- Section 1: The pre-pattern, end-to-end ---
    print("--- Section 1: The pre-pattern, end-to-end ---")

    client = MockLLMClient(
        responses=[
            make_response(
                '{"kind": "bug_report", "summary": "Login button unresponsive on iOS Safari", "priority": "high"}'
            ),
            make_response(
                '{"kind": "feature_request", "summary": "Add dark mode to settings page", "priority": "medium"}'
            ),
        ]
    )
    emitter = make_emitter("dispatch-pre-pattern")

    agent = ReasoningAgent(
        name="triage-classifier",
        llm_client=client,
        emitter=emitter,
        system_prompt=(
            "Classify the support message into a DispatchDecision. "
            "Pick the most appropriate `kind` and assign a priority."
        ),
        output_schema=DispatchDecision,
    )

    result = await agent.run("When I tap Login on iOS Safari nothing happens — this is blocking customers.")

    # The agent produces typed structured output; `.parsed` is the DispatchDecision.
    assert result.parsed is not None
    decision = result.parsed
    assert isinstance(decision, DispatchDecision)
    assert decision.kind == "bug_report"
    assert decision.priority == "high"

    # Deterministic Python consumes the typed output and dispatches.
    outcome = await dispatch(decision)
    assert outcome == "Bug filed [high]: Login button unresponsive on iOS Safari"

    # The defining property of the pre-pattern: one LLM call across the whole pipeline.
    assert len(client.calls) == 1

    print(f"  Decision: kind={decision.kind}, priority={decision.priority}")
    print(f"  Outcome: {outcome}")
    print(f"  LLM calls: {len(client.calls)} (agent only — dispatcher is pure Python)")
    print("✓ One LLM call, then deterministic dispatch")

    # --- Section 2: Same dispatcher, different `kind` ---
    print("\n--- Section 2: Same dispatcher, different `kind` ---")

    result = await agent.run("Could you add a dark mode option to the settings?")

    assert result.parsed is not None
    decision = result.parsed
    assert decision.kind == "feature_request"
    assert decision.priority == "medium"

    outcome = await dispatch(decision)
    assert outcome == "Feature request logged [medium]: Add dark mode to settings page"

    # Cumulative across both runs: one LLM call per agent run, no second-stage calls.
    assert len(client.calls) == 2

    print(f"  Decision: kind={decision.kind}, priority={decision.priority}")
    print(f"  Outcome: {outcome}")
    print(f"  Cumulative LLM calls: {len(client.calls)} (one per input — dispatch added zero)")
    print("✓ Same dispatcher routes a second input with no extra LLM cost")

    # --- Section 3: Why this is the pre-pattern, not multi-agent ---
    print("\n--- Section 3: Why this is the pre-pattern, not multi-agent ---")
    print("  A multi-agent variant would have added:")
    print("    - A second LLM call (another agent reasoning over the first agent's output)")
    print("    - A `ContextTransferStrategy` choice (Raw/Trajectory/Summary)")
    print("    - A `DelegationEvent` or `HandoffEvent` in the trace")
    print("  A workflow variant would have added:")
    print("    - A `Sequential`/`Parallel`/`DAG` primitive with `Step`/`StepResult` plumbing")
    print("    - A workflow event tree above the agent's own events")
    print("  Here, the second stage is a pure function of typed input — neither is warranted.")
    print("✓ Pre-pattern check before reaching for multi-agent or workflow primitives")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
