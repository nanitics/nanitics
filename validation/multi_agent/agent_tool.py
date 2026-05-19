"""Agent-as-tool delegation between two real agents.

A specialist ``ReActAgent`` is wrapped as an ``AgentTool`` and handed to a
caller ``ReActAgent``. The caller's LLM picks the delegate; the delegate's
LLM produces an answer using its inner tool; the caller synthesizes a
final response. The ``DelegationEvent`` ties the two agents together in
the trace. A second test case exercises the error-propagation path:
when the caller asks about an internal project codename absent from the
registry, the specialist's "no record" string propagates to the caller's
final output via the tool-result channel.

The scenario uses an internal project-codename registry as its fact
source: parametric LLM knowledge cannot answer from it, so delegation is
the *capable* path to a correct answer rather than a *commanded* one.
The prompts describe the agents' roles and the available collaboration
pattern abstractly — they do not issue case-level imperatives about
tool calls.

Acceptance criteria (happy path):
  - Trace contains a ``DelegationEvent`` with ``caller_agent="coordinator"``,
    ``delegate_agent="researcher"``, and ``"Nimbus" in task`` (the
    coordinator forwarded the codename of interest — not an empty string
    or a canned prompt fragment).
  - Trace contains a ``ToolInvokeEvent`` for ``lookup_codename`` (the
    specialist actually used its inner tool).
  - The ``lookup_codename`` ``ToolInvokeEvent``'s ``parent_span_id``
    matches an ``AgentStartEvent`` with ``agent_name == "researcher"``
    (the inner tool was invoked under the delegate's span, not the
    caller's — hierarchical trace attribution).
  - Trace contains a ``ToolResultEvent`` whose ``tool_name`` matches the
    ``AgentTool`` and whose ``result`` contains ``"Snowflake"`` (the
    specialist's output actually flowed through ``RawOutputTransfer``
    into the caller's tool-result channel, not just the final synthesis).
  - Final output identifies Project Nimbus as the Q4 2025 data-warehouse
    migration to Snowflake (or equivalent phrasing).

Acceptance criteria (error-propagation path):
  - Inner ``ToolResultEvent`` for ``lookup_codename`` carries the verbatim
    ``"No record for codename"`` string — proves the tool layer surfaces
    the absent-data signal deterministically.
  - Outer ``ToolResultEvent`` for the ``AgentTool`` is successful and its
    ``result`` satisfies an LLM-judge check that the absent-data surface
    propagates to the caller (the specialist may paraphrase the inner
    string under ``RawOutputTransfer``; the contract is that the
    no-record semantics survive the transfer, not a specific phrasing).
  - Caller's final output reflects that no record was found.
"""

from __future__ import annotations

import pytest

from nanitics.composition import AgentTool
from nanitics.infrastructure import (
    AgentStartEvent,
    DelegationEvent,
    ToolInvokeEvent,
    ToolResultEvent,
)
from nanitics.strategies import (
    ReActAgent,
    tool,
)
from nanitics.tracing import InMemoryEmitter
from validation.helpers import (
    assert_result_satisfies,
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)

_PROJECT_REGISTRY = {
    "Nimbus": "Q4 2025 data-warehouse migration to Snowflake; owner: Platform team.",
    "Falcon": "Internal OAuth2 identity broker for cross-tenant SSO; owner: Security team.",
    "Lighthouse": "Customer-facing status page rebuild on Next.js; owner: Frontend team.",
    "Mercator": "Geographic-routing layer for regional compliance; owner: Infrastructure team.",
}

_AGENT_TOOL_NAME = "researcher"


@tool("lookup_codename", "Look up the canonical description of an internal project codename.")
async def lookup_codename(codename: str) -> str:
    # The LLM sometimes phrases the codename as "Project <Name>"; strip the
    # conventional "Project " prefix before the registry lookup so natural
    # phrasing resolves to the bare registry key. The "No record" branch
    # below is preserved verbatim for genuinely unknown codenames.
    lookup_key = codename[len("Project ") :] if codename.startswith("Project ") else codename
    description = _PROJECT_REGISTRY.get(lookup_key)
    if description is None:
        return f"No record for codename {lookup_key!r} in the internal project registry."
    return description


def _build_caller_and_specialist(client, emitter: InMemoryEmitter) -> tuple[ReActAgent, ReActAgent]:
    specialist = ReActAgent(
        name=_AGENT_TOOL_NAME,
        llm_client=client,
        emitter=emitter,
        system_prompt=(
            "You are a research specialist with access to an internal "
            "project-codename registry. When the user asks about an internal "
            "project by codename, use your lookup_codename tool to retrieve "
            "the canonical description, then report what the tool returns. "
            "If the tool reports no record exists, say so clearly."
        ),
        tools=[lookup_codename],
        max_iterations=3,
    )

    agent_tool = AgentTool(
        agent=specialist,
        emitter=emitter,
        description="Delegate internal project-codename lookups to a research specialist.",
        caller_name="coordinator",
    )

    caller = ReActAgent(
        name="coordinator",
        llm_client=client,
        emitter=emitter,
        system_prompt=(
            "You are a general-purpose assistant. When a user asks about an "
            "internal project, tool, or system referenced by a codename, the "
            "research specialist has access to the internal project registry "
            "— forward the lookup to them via the researcher tool and report "
            "what they find."
        ),
        tools=[agent_tool],
        max_iterations=4,
    )
    return caller, specialist


@pytest.mark.quick
async def test_agent_tool_delegation(traced_emitter: InMemoryEmitter) -> None:
    client = make_llm_client("anthropic")
    caller, _specialist = _build_caller_and_specialist(client, traced_emitter)

    result = await run_with_retry(
        lambda: caller.run("Can you look up Project Nimbus and tell me what it's about?"),
        max_attempts=2,
    )

    # --- DelegationEvent: identity + task-payload fidelity ---
    assert_trace_contains(
        traced_emitter,
        DelegationEvent,
        predicate=lambda e: (
            e.caller_agent == "coordinator" and e.delegate_agent == _AGENT_TOOL_NAME and "Nimbus" in e.task
        ),
    )

    # --- Inner ToolInvokeEvent on the specialist's tool ---
    inner_invoke = assert_trace_contains(
        traced_emitter,
        ToolInvokeEvent,
        predicate=lambda e: e.tool_name == "lookup_codename",
    )

    # --- Hierarchical trace attribution: the lookup_codename invocation
    # sits under the researcher's span, proving nested-event attribution
    # via span lineage rather than by name collision.
    researcher_start_spans = {
        e.span_id for e in traced_emitter.events if isinstance(e, AgentStartEvent) and e.agent_name == _AGENT_TOOL_NAME
    }
    assert researcher_start_spans, (
        "Expected at least one AgentStartEvent for the researcher; "
        f"found agent_name values: "
        f"{[getattr(e, 'agent_name', None) for e in traced_emitter.events if isinstance(e, AgentStartEvent)]}"
    )
    assert inner_invoke.parent_span_id in researcher_start_spans, (
        "Expected lookup_codename ToolInvokeEvent.parent_span_id to match a "
        f"researcher AgentStartEvent.span_id; got parent_span_id="
        f"{inner_invoke.parent_span_id!r}, researcher spans={researcher_start_spans}"
    )

    # --- ToolResultEvent for the AgentTool itself: the specialist's
    # output flows back to the caller via the tool-result channel.
    # Under RawOutputTransfer the delegate's result.output becomes the
    # tool content, so "Snowflake" (a distinctive registry-value
    # substring) appears in the ToolResultEvent.result.
    assert_trace_contains(
        traced_emitter,
        ToolResultEvent,
        predicate=lambda e: (
            e.tool_name == _AGENT_TOOL_NAME and e.success is True and (e.result is not None and "Snowflake" in e.result)
        ),
    )

    # --- Final-output synthesis ---
    await assert_result_satisfies(
        result.output or "",
        (
            "The output identifies Project Nimbus as the Q4 2025 data-warehouse "
            "migration to Snowflake (or equivalent phrasing that captures the "
            "data-warehouse migration and Snowflake)."
        ),
    )


@pytest.mark.quick
async def test_agent_tool_error_propagation(traced_emitter: InMemoryEmitter) -> None:
    """Specialist's "no record" surface propagates to the caller's final answer."""
    client = make_llm_client("anthropic")
    caller, _specialist = _build_caller_and_specialist(client, traced_emitter)

    result = await run_with_retry(
        lambda: caller.run("Can you look up Project Atlantis and tell me who owns it?"),
        max_attempts=2,
    )

    # Delegation with the unknown codename in the forwarded task.
    assert_trace_contains(
        traced_emitter,
        DelegationEvent,
        predicate=lambda e: (
            e.caller_agent == "coordinator" and e.delegate_agent == _AGENT_TOOL_NAME and "Atlantis" in e.task
        ),
    )

    # Two-channel check for the absent-data surface:
    #   (1) Inner channel — deterministic. The lookup_codename tool produces
    #       a fixed "No record for codename" string for unknown keys; this
    #       proves the absent-data surface is generated correctly at the
    #       tool layer regardless of how the specialist paraphrases it.
    #   (2) Outer channel — paraphrase-tolerant judge. The specialist's
    #       free-form synthesis flows through RawOutputTransfer into the
    #       outer ToolResultEvent.result; the contract being validated
    #       is that the absent-data semantics survive that transfer, not
    #       any specific phrasing.
    assert_trace_contains(
        traced_emitter,
        ToolResultEvent,
        predicate=lambda e: (
            e.tool_name == "lookup_codename" and e.result is not None and "No record for codename" in e.result
        ),
    )

    outer_result = assert_trace_contains(
        traced_emitter,
        ToolResultEvent,
        predicate=lambda e: e.tool_name == _AGENT_TOOL_NAME and e.success is True and e.result is not None,
    )
    assert outer_result.result is not None
    await assert_result_satisfies(
        outer_result.result,
        (
            "Indicates that no record was found for the codename in question. "
            "The original wording 'No record for codename' may have been "
            "paraphrased; what matters is that the absent-data surface reached "
            "the caller."
        ),
    )

    # Caller's final output reflects the no-data outcome.
    await assert_result_satisfies(
        result.output or "",
        (
            "The output indicates there is no record for Project Atlantis in "
            "the internal registry (or equivalent phrasing that no record was "
            "found). It does not invent an owner team or description."
        ),
    )
