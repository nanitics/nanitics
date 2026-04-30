"""Broadcast fan-out across real agents with response-strategy and eligibility-filter coverage.

Three real ``ReActAgent`` analysts (``optimist``, ``skeptic``, ``pragmatist``)
are broadcast the same business question. The script is parametrised over
``CollectAll`` and ``SelectBest`` to pin the two core aggregation paths,
and adds separate tests for ``FilterResponses``, ``MergeResponses``, and
``CapabilityFilter`` eligibility — each with a distinguishing assertion
that would catch a regression in its specific path.

Acceptance criteria (parametrised CollectAll/SelectBest):
  - ``BroadcastStartEvent`` lists all three agent names and carries the
    configured ``response_strategy`` name.
  - Every eligible agent emitted an ``AgentStartEvent`` (proves they
    actually ran, not just that events were fabricated).
  - ``BroadcastCompleteEvent`` has ``total_agents == 3`` and
    ``responses_collected == 3``.
  - For ``CollectAll``: ``aggregated_output`` is a list of length 3
    containing each agent's output string.
  - For ``SelectBest`` (longest-output scorer): ``aggregated_output`` is
    exactly one of the three agent outputs (proves the scorer chose an
    actual input, not a synthesized string).

Acceptance criteria (FilterResponses):
  - ``aggregated_output`` is a list and every surviving entry satisfies
    the predicate (case-insensitive ``"risk"`` substring).
  - All three agents still ran (filter applies post-response).

Acceptance criteria (MergeResponses with real LLM):
  - ``aggregated_output`` is a non-empty string (LLM synthesis path).
  - Final string mentions all three perspectives (fuzzy LLM-judge check).

Acceptance criteria (CapabilityFilter eligibility):
  - Exactly the agents with the required capability emitted
    ``AgentStartEvent``; the excluded agent did not run.
  - ``BroadcastStartEvent.agent_names`` matches the eligible set (no
    entry for the filtered-out agent).
  - ``result.agents_participated`` equals the eligible count.
"""

from __future__ import annotations

import pytest

from nanitics import (
    AllEligible,
    Broadcast,
    CapabilityFilter,
    CollectAll,
    FilterResponses,
    InMemoryEmitter,
    MergeResponses,
    ReActAgent,
    SelectBest,
)
from nanitics.infrastructure import (
    AgentStartEvent,
    BroadcastCompleteEvent,
    BroadcastStartEvent,
)
from validation.helpers import (
    assert_result_satisfies,
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)

_TASK = "Should a mid-size SaaS company expand into the European market next year?"

_OPTIMIST_PROMPT = (
    "You are an optimistic growth strategist. In two sentences, argue in "
    "favor of European expansion, highlighting one concrete upside."
)
_SKEPTIC_PROMPT = (
    "You are a risk-focused analyst. In two sentences, argue against "
    "immediate European expansion, highlighting one concrete risk."
)
_PRAGMATIST_PROMPT = (
    "You are a pragmatic operator. In two sentences, propose a balanced, phased approach to European expansion."
)


def _make_agents(emitter: InMemoryEmitter, client) -> list[ReActAgent]:
    return [
        ReActAgent(
            name="optimist",
            llm_client=client,
            emitter=emitter,
            system_prompt=_OPTIMIST_PROMPT,
            tools=[],
            max_iterations=2,
        ),
        ReActAgent(
            name="skeptic",
            llm_client=client,
            emitter=emitter,
            system_prompt=_SKEPTIC_PROMPT,
            tools=[],
            max_iterations=2,
        ),
        ReActAgent(
            name="pragmatist",
            llm_client=client,
            emitter=emitter,
            system_prompt=_PRAGMATIST_PROMPT,
            tools=[],
            max_iterations=2,
        ),
    ]


@pytest.mark.quick
@pytest.mark.parametrize("strategy_name", ["CollectAll", "SelectBest"])
async def test_broadcast_strategy(traced_emitter: InMemoryEmitter, strategy_name: str) -> None:
    client = make_llm_client("anthropic")
    agents = _make_agents(traced_emitter, client)

    if strategy_name == "CollectAll":
        strategy = CollectAll()
    else:
        # Longest-output scorer — distinct from "first response" or
        # "random"; catches regressions that would otherwise return a
        # constant winner.
        strategy = SelectBest(scorer=lambda r: len(str(r.output)))

    broadcast = Broadcast(
        agents=agents,
        emitter=traced_emitter,
        response_strategy=strategy,
        eligibility_filter=AllEligible(),
    )

    result = await run_with_retry(lambda: broadcast.run(_TASK), max_attempts=2)

    assert result.response_strategy == strategy_name
    assert result.agents_participated == 3
    assert len(result.responses) == 3

    assert_trace_contains(
        traced_emitter,
        BroadcastStartEvent,
        predicate=lambda e: (
            set(e.agent_names) == {"optimist", "skeptic", "pragmatist"} and e.response_strategy == strategy_name
        ),
    )

    for name in ("optimist", "skeptic", "pragmatist"):
        assert_trace_contains(
            traced_emitter,
            AgentStartEvent,
            predicate=lambda e, n=name: e.agent_name == n,
        )

    assert_trace_contains(
        traced_emitter,
        BroadcastCompleteEvent,
        predicate=lambda e: e.total_agents == 3 and e.responses_collected == 3,
    )

    if strategy_name == "CollectAll":
        assert isinstance(result.aggregated_output, list), (
            f"Expected list for CollectAll, got: {type(result.aggregated_output).__name__}"
        )
        assert len(result.aggregated_output) == 3
        response_outputs = {str(r.output) for r in result.responses}
        for entry in result.aggregated_output:
            assert str(entry) in response_outputs, f"CollectAll entry {entry!r} is not one of the agent responses."
    else:
        # Winner must be exactly one of the agent outputs.
        response_outputs = {str(r.output) for r in result.responses}
        assert str(result.aggregated_output) in response_outputs, (
            f"SelectBest winner {result.aggregated_output!r} is not among inputs {sorted(response_outputs)!r}"
        )
        # And by construction (longest scorer), it must be the longest.
        longest = max(response_outputs, key=len)
        assert str(result.aggregated_output) == longest, (
            f"SelectBest with longest-output scorer should return the longest "
            f"response; got {result.aggregated_output!r}, expected {longest!r}"
        )


async def test_broadcast_filter_responses(traced_emitter: InMemoryEmitter) -> None:
    client = make_llm_client("anthropic")
    # Prompt skeptic and pragmatist toward the "risk" predicate; optimist
    # is free to ignore it. The predicate is robust to whether optimist
    # also mentions risk — we only assert surviving entries match.
    agents = _make_agents(traced_emitter, client)

    broadcast = Broadcast(
        agents=agents,
        emitter=traced_emitter,
        response_strategy=FilterResponses(predicate=lambda r: "risk" in str(r.output).lower()),
    )

    result = await run_with_retry(lambda: broadcast.run(_TASK), max_attempts=2)

    assert result.response_strategy == "FilterResponses"
    assert len(result.responses) == 3, f"Filter must not drop pre-aggregation responses, got: {len(result.responses)}"
    assert isinstance(result.aggregated_output, list)
    for entry in result.aggregated_output:
        assert "risk" in str(entry).lower(), f"FilterResponses kept an entry that fails predicate: {entry!r}"


async def test_broadcast_merge_responses(traced_emitter: InMemoryEmitter) -> None:
    client = make_llm_client("anthropic")
    agents = _make_agents(traced_emitter, client)

    broadcast = Broadcast(
        agents=agents,
        emitter=traced_emitter,
        response_strategy=MergeResponses(llm_client=client),
    )

    result = await run_with_retry(lambda: broadcast.run(_TASK), max_attempts=2)

    assert result.response_strategy == "MergeResponses"
    assert isinstance(result.aggregated_output, str), (
        f"Expected merged output to be a string, got: {type(result.aggregated_output).__name__}"
    )
    assert result.aggregated_output, f"Expected non-empty merged string, got: {result.aggregated_output!r}"

    await assert_result_satisfies(
        result.aggregated_output,
        (
            "The text is a synthesis that reflects at least three distinct "
            "perspectives on European market expansion — including both a "
            "growth/opportunity angle and a risk/caution angle — rather than "
            "restating a single viewpoint."
        ),
    )


async def test_broadcast_capability_filter(traced_emitter: InMemoryEmitter) -> None:
    client = make_llm_client("anthropic")
    agents = _make_agents(traced_emitter, client)

    capabilities = {
        "optimist": ["growth", "market-analysis"],
        "skeptic": ["risk-assessment"],
        "pragmatist": ["market-analysis", "risk-assessment"],
    }

    broadcast = Broadcast(
        agents=agents,
        emitter=traced_emitter,
        eligibility_filter=CapabilityFilter(
            capabilities=capabilities,
            required=["risk-assessment"],
        ),
    )

    result = await run_with_retry(lambda: broadcast.run(_TASK), max_attempts=2)

    assert result.agents_participated == 2, (
        f"Expected 2 eligible agents (skeptic+pragmatist), got: {result.agents_participated}"
    )
    eligible_names = {r.agent_name for r in result.responses}
    assert eligible_names == {"skeptic", "pragmatist"}, f"Expected {{'skeptic','pragmatist'}}, got: {eligible_names}"

    assert_trace_contains(
        traced_emitter,
        BroadcastStartEvent,
        predicate=lambda e: set(e.agent_names) == {"skeptic", "pragmatist"},
    )

    # Only eligible agents ran.
    start_event_names = {e.agent_name for e in traced_emitter.events if isinstance(e, AgentStartEvent)}
    assert "optimist" not in start_event_names, (
        f"Excluded agent 'optimist' must not have run; AgentStartEvent names: {start_event_names}"
    )
    assert {"skeptic", "pragmatist"}.issubset(start_event_names), (
        f"Eligible agents must have run; AgentStartEvent names: {start_event_names}"
    )
