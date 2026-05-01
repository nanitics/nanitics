"""Broadcast: fan-out the same task to multiple agents in parallel.

Demonstrates Broadcast — sending a task to multiple agents concurrently
and aggregating their responses with different strategies. Covers CollectAll
(default), SelectBest, MergeResponses (LLM synthesis), FilterResponses,
capability-based eligibility filtering, and failure handling.

Related guide: docs/guides/multi-agent-foundations.md
"""

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics import (
    AgentFailure,
    Broadcast,
    BroadcastResponse,
    BroadcastResult,
    CapabilityFilter,
    CollectAll,
    FilterResponses,
    MergeResponses,
    MockLLMClient,
    ReActAgent,
    SelectBest,
)
from nanitics.infrastructure import (
    BroadcastCompleteEvent,
    BroadcastResponseEvent,
    BroadcastStartEvent,
)

TASK = "Should we expand into the European market?"


def make_agents(emitter, optimist_text: str, skeptic_text: str, pragmatist_text: str):
    """Create three analyst agents with distinct perspectives."""
    return [
        ReActAgent(
            name="optimist",
            llm_client=MockLLMClient([make_response(optimist_text)]),
            emitter=emitter,
            system_prompt="You provide optimistic analysis.",
            tools=[],
        ),
        ReActAgent(
            name="skeptic",
            llm_client=MockLLMClient([make_response(skeptic_text)]),
            emitter=emitter,
            system_prompt="You provide skeptical analysis.",
            tools=[],
        ),
        ReActAgent(
            name="pragmatist",
            llm_client=MockLLMClient([make_response(pragmatist_text)]),
            emitter=emitter,
            system_prompt="You provide balanced, pragmatic analysis.",
            tools=[],
        ),
    ]


async def main() -> None:
    # --- Section 1: Basic Broadcast (CollectAll) ---
    # CollectAll is the simplest response strategy — it collects all outputs
    # into a list without filtering or scoring. It's also the default when no
    # strategy is specified. Here we pass it explicitly.

    emitter = make_emitter("broadcast-s1")
    agents = make_agents(
        emitter,
        optimist_text="Europe offers massive growth potential with 450M consumers.",
        skeptic_text="Regulatory complexity in the EU will slow us down significantly.",
        pragmatist_text="Europe is viable but requires phased entry starting with the UK and Germany.",
    )

    broadcast = Broadcast(agents=agents, emitter=emitter, response_strategy=CollectAll())
    result: BroadcastResult = await broadcast.run(TASK)

    assert result.agents_participated == 3
    assert len(result.responses) == 3
    assert result.response_strategy == "CollectAll"
    assert isinstance(result.aggregated_output, list)
    assert len(result.aggregated_output) == 3
    assert len(result.failures) == 0
    for resp in result.responses:
        assert isinstance(resp, BroadcastResponse)
        assert resp.agent_name in ("optimist", "skeptic", "pragmatist")
        assert resp.output
        assert resp.steps >= 1
        assert resp.termination_reason

    # Verify events
    events = emitter.events
    start_events = [e for e in events if isinstance(e, BroadcastStartEvent)]
    response_events = [e for e in events if isinstance(e, BroadcastResponseEvent)]
    complete_events = [e for e in events if isinstance(e, BroadcastCompleteEvent)]
    assert len(start_events) == 1
    assert set(start_events[0].agent_names) == {"optimist", "skeptic", "pragmatist"}
    assert start_events[0].task == TASK
    assert len(response_events) == 3
    assert len(complete_events) == 1
    assert complete_events[0].total_agents == 3
    assert complete_events[0].responses_collected == 3
    assert complete_events[0].failures == 0

    print("--- Section 1: Basic Broadcast (CollectAll) ---")
    for resp in result.responses:
        print(f"  {resp.agent_name}: {resp.output}")
    print(f"  Agents participated: {result.agents_participated}")
    print(f"  Strategy: {result.response_strategy}")
    print("✓ All 3 agents responded, events verified")

    # --- Section 2: SelectBest Strategy ---
    # Picks the response with the highest score. Here: longest response wins.

    emitter = make_emitter("broadcast-s2")
    agents = make_agents(
        emitter,
        optimist_text="Go for it!",
        skeptic_text="Too risky.",
        pragmatist_text="A phased approach targeting Germany first would balance opportunity and risk effectively.",
    )

    broadcast = Broadcast(
        agents=agents,
        emitter=emitter,
        response_strategy=SelectBest(scorer=lambda r: len(str(r.output))),
    )
    result = await broadcast.run(TASK)

    assert result.response_strategy == "SelectBest"
    assert isinstance(result.aggregated_output, str)
    assert len(result.responses) == 3
    # The pragmatist's response is the longest
    assert "phased approach" in result.aggregated_output

    print("\n--- Section 2: SelectBest Strategy ---")
    print("  Scorer: length of response (longer = better)")
    for resp in result.responses:
        print(f"  {resp.agent_name} ({len(str(resp.output))} chars): {resp.output}")
    print(f"  Selected: {result.aggregated_output}")
    print("✓ SelectBest picked the longest response")

    # --- Section 3: MergeResponses Strategy ---
    # Uses an LLM to synthesize all responses into a single unified answer.
    # Unlike the other strategies, this costs one additional LLM call.

    emitter = make_emitter("broadcast-s3")
    agents = make_agents(
        emitter,
        optimist_text="Europe offers massive growth potential with 450M consumers.",
        skeptic_text="Regulatory complexity in the EU will slow us down significantly.",
        pragmatist_text="Europe is viable but requires phased entry starting with the UK and Germany.",
    )

    # The synthesizer LLM produces a single merged answer from all responses
    synthesizer_client = MockLLMClient(
        [
            make_response(
                "Synthesis: Europe presents significant growth potential (450M consumers) "
                "but regulatory complexity poses risks. A phased entry starting with "
                "the UK and Germany balances opportunity against regulatory burden."
            ),
        ]
    )

    broadcast = Broadcast(
        agents=agents,
        emitter=emitter,
        response_strategy=MergeResponses(llm_client=synthesizer_client),
    )
    result = await broadcast.run(TASK)

    assert result.response_strategy == "MergeResponses"
    assert isinstance(result.aggregated_output, str)
    assert len(result.responses) == 3
    assert "phased entry" in result.aggregated_output

    # The synthesizer received all three agent responses as input
    assert len(synthesizer_client.calls) == 1
    synthesizer_input = synthesizer_client.calls[0]["messages"][0].content
    assert "optimist" in synthesizer_input
    assert "skeptic" in synthesizer_input
    assert "pragmatist" in synthesizer_input

    print("\n--- Section 3: MergeResponses Strategy ---")
    print("  All 3 responses sent to synthesizer LLM")
    print(f"  Synthesizer input ({len(synthesizer_input)} chars):")
    for line in synthesizer_input.split("\n")[:6]:
        print(f"    {line}")
    print(f"  Merged output: {result.aggregated_output}")
    print("✓ MergeResponses synthesized 3 perspectives into a single answer")

    # --- Section 4: FilterResponses Strategy ---
    # Keeps only responses containing a keyword. Here: "risk".

    emitter = make_emitter("broadcast-s4")
    agents = make_agents(
        emitter,
        optimist_text="Europe offers incredible growth potential.",
        skeptic_text="The regulatory risk in the EU is substantial and could delay launch.",
        pragmatist_text="Balancing risk and opportunity, a staged rollout minimizes downside.",
    )

    broadcast = Broadcast(
        agents=agents,
        emitter=emitter,
        response_strategy=FilterResponses(predicate=lambda r: "risk" in str(r.output).lower()),
    )
    result = await broadcast.run(TASK)

    assert result.response_strategy == "FilterResponses"
    assert isinstance(result.aggregated_output, list)
    assert len(result.aggregated_output) == 2  # skeptic + pragmatist mention "risk"
    assert all("risk" in str(o).lower() for o in result.aggregated_output)
    assert len(result.responses) == 3  # all 3 still ran

    print("\n--- Section 4: FilterResponses Strategy ---")
    print("  Predicate: response must contain 'risk'")
    for resp in result.responses:
        passed = "risk" in str(resp.output).lower()
        print(f"  {resp.agent_name}: {'PASS' if passed else 'FILTERED'} — {resp.output}")
    print(f"  Kept {len(result.aggregated_output)} of {len(result.responses)} responses")
    print("✓ FilterResponses kept only responses mentioning 'risk'")

    # --- Section 5: Capability-Based Eligibility ---
    # CapabilityFilter restricts which agents receive the task.
    # Only agents with at least one required capability participate.

    emitter = make_emitter("broadcast-s5")
    agents = make_agents(
        emitter,
        optimist_text="Growth opportunity is enormous.",
        skeptic_text="Risk assessment: high regulatory burden.",
        pragmatist_text="Risk-adjusted returns favor phased entry.",
    )

    broadcast = Broadcast(
        agents=agents,
        emitter=emitter,
        eligibility_filter=CapabilityFilter(
            capabilities={
                "optimist": ["market-analysis", "growth"],
                "skeptic": ["risk-assessment"],
                "pragmatist": ["market-analysis", "risk-assessment"],
            },
            required=["risk-assessment"],
        ),
    )
    result = await broadcast.run(TASK)

    assert result.agents_participated == 2
    assert len(result.responses) == 2
    assert result.response_strategy == "CollectAll"
    responded_names = {r.agent_name for r in result.responses}
    assert "optimist" not in responded_names
    assert "skeptic" in responded_names
    assert "pragmatist" in responded_names

    print("\n--- Section 5: Capability-Based Eligibility ---")
    print("  Required capability: 'risk-assessment'")
    print("  optimist capabilities: ['market-analysis', 'growth'] → EXCLUDED")
    print("  skeptic capabilities: ['risk-assessment'] → ELIGIBLE")
    print("  pragmatist capabilities: ['market-analysis', 'risk-assessment'] → ELIGIBLE")
    for resp in result.responses:
        print(f"  {resp.agent_name}: {resp.output}")
    print(f"  {result.agents_participated} of 3 agents participated")
    print("✓ CapabilityFilter excluded agents without required capability")

    # --- Section 6: Failure Handling ---
    # One agent fails (empty mock → LLMProviderError). Broadcast completes
    # with partial results and captures the failure.

    emitter = make_emitter("broadcast-s6")
    agents = [
        ReActAgent(
            name="optimist",
            llm_client=MockLLMClient([make_response("Europe is a great opportunity.")]),
            emitter=emitter,
            system_prompt="You provide optimistic analysis.",
            tools=[],
        ),
        ReActAgent(
            name="skeptic",
            llm_client=MockLLMClient([]),  # empty → raises LLMProviderError
            emitter=emitter,
            system_prompt="You provide skeptical analysis.",
            tools=[],
        ),
        ReActAgent(
            name="pragmatist",
            llm_client=MockLLMClient([make_response("Proceed with caution and a staged plan.")]),
            emitter=emitter,
            system_prompt="You provide balanced, pragmatic analysis.",
            tools=[],
        ),
    ]

    broadcast = Broadcast(agents=agents, emitter=emitter)
    result = await broadcast.run(TASK)

    assert result.agents_participated == 3  # all 3 started
    assert len(result.responses) == 2  # 2 succeeded
    assert len(result.failures) == 1
    failure = result.failures[0]
    assert isinstance(failure, AgentFailure)
    assert failure.agent_name == "skeptic"
    assert failure.error_type  # e.g., "LLMProviderError"
    assert failure.error_message
    assert isinstance(result.aggregated_output, list)
    assert len(result.aggregated_output) == 2

    print("\n--- Section 6: Failure Handling ---")
    print(f"  Agents started: {result.agents_participated}")
    print(f"  Successful responses: {len(result.responses)}")
    for resp in result.responses:
        print(f"    {resp.agent_name}: {resp.output}")
    print(f"  Failures: {len(result.failures)}")
    print(f"    {failure.agent_name}: {failure.error_type} — {failure.error_message}")
    print("✓ Broadcast completed with partial results despite agent failure")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
