"""Debate: structured adversarial reasoning between agents.

Demonstrates Debate — agents argue assigned positions across rounds,
then a resolution strategy evaluates the transcript to produce a verdict.
Covers JudgeResolution, LLMJudgeResolution with criteria, custom resolution
strategies, multi-party debate, and event verification.

Related guide: docs/guides/multi-agent-coordination.md
"""

import asyncio
import json

from examples.helpers import make_emitter, make_response
from nanitics.infrastructure import (
    DebateArgumentEvent,
    DebateCompleteEvent,
    DebateResolutionEvent,
    DebateStartEvent,
    MockLLMClient,
)
from nanitics.specialized import (
    Argument,
    Debate,
    Debater,
    DebateResolution,
    DebateResult,
    JudgeResolution,
    LLMJudgeResolution,
    ResolutionStrategy,
)
from nanitics.strategies import ReActAgent


async def main() -> None:
    # --- Section 1: Basic Debate with JudgeResolution ---
    # Two debaters argue opposite positions over 2 rounds. A separate
    # judge agent evaluates the transcript and produces a free-form verdict.
    # JudgeResolution always returns winner=None (the judge doesn't produce
    # structured output — its response becomes both reasoning and synthesis).

    emitter = make_emitter("debate-s1")

    pro_agent = ReActAgent(
        name="pro",
        llm_client=MockLLMClient(
            [
                make_response("Safety must come first. Unchecked AI development risks catastrophic failures."),
                make_response("Speed sacrifices quality. History shows rushed technology causes harm."),
            ]
        ),
        emitter=emitter,
        system_prompt="You argue for prioritizing safety.",
        tools=[],
    )
    con_agent = ReActAgent(
        name="con",
        llm_client=MockLLMClient(
            [
                make_response("Speed is essential. Competitors who move faster capture the market."),
                make_response("Over-cautious approaches stifle innovation. Iterative deployment handles risk."),
            ]
        ),
        emitter=emitter,
        system_prompt="You argue for prioritizing speed.",
        tools=[],
    )
    judge_agent = ReActAgent(
        name="judge",
        llm_client=MockLLMClient(
            [
                make_response("Both sides made compelling points. Safety advocates showed stronger evidence."),
            ]
        ),
        emitter=emitter,
        system_prompt="You are an impartial debate judge.",
        tools=[],
    )

    debate = Debate(
        debaters=[
            Debater(agent=pro_agent, position="for prioritizing safety"),
            Debater(agent=con_agent, position="for prioritizing speed"),
        ],
        emitter=emitter,
        resolution=JudgeResolution(judge=judge_agent),
        max_rounds=2,
    )
    result: DebateResult = await debate.run("Should AI development prioritize safety over speed?")

    # Result structure
    assert result.rounds_completed == 2
    assert result.termination_reason == "max_rounds"

    # Transcript: 2 debaters × 2 rounds = 4 arguments in chronological order
    assert len(result.transcript) == 4
    assert all(isinstance(arg, Argument) for arg in result.transcript)

    # Round 1: pro then con
    assert result.transcript[0].round == 1
    assert result.transcript[0].agent_name == "pro"
    assert result.transcript[0].position == "for prioritizing safety"
    assert "Safety" in result.transcript[0].content
    assert result.transcript[1].round == 1
    assert result.transcript[1].agent_name == "con"
    assert result.transcript[1].position == "for prioritizing speed"

    # Round 2: pro then con
    assert result.transcript[2].round == 2
    assert result.transcript[2].agent_name == "pro"
    assert result.transcript[3].round == 2
    assert result.transcript[3].agent_name == "con"

    # JudgeResolution: winner is always None, reasoning/synthesis are the judge's output
    assert result.resolution.winner is None
    assert "compelling" in result.resolution.reasoning
    assert result.resolution.synthesis == result.resolution.reasoning  # same for JudgeResolution

    print("--- Section 1: Basic Debate with JudgeResolution ---")
    print("  Topic: Should AI development prioritize safety over speed?")
    print(f"  Rounds completed: {result.rounds_completed}")
    for arg in result.transcript:
        print(f"  Round {arg.round} [{arg.agent_name}]: {arg.content[:60]}...")
    print(f"  Judge verdict: {result.resolution.reasoning[:60]}...")
    print(f"  Winner: {result.resolution.winner} (JudgeResolution always returns None)")
    print("✓ Full debate lifecycle verified")

    # --- Section 2: LLMJudgeResolution with Structured Output ---
    # Uses an LLM with output_schema to produce a typed verdict with
    # winner, reasoning, and synthesis — unlike JudgeResolution which
    # always returns winner=None.

    emitter = make_emitter("debate-s2")

    pro_agent = ReActAgent(
        name="pro",
        llm_client=MockLLMClient(
            [
                make_response("Remote work increases productivity and reduces commute time."),
            ]
        ),
        emitter=emitter,
        system_prompt="You argue for remote work.",
        tools=[],
    )
    con_agent = ReActAgent(
        name="con",
        llm_client=MockLLMClient(
            [
                make_response("Office work fosters collaboration and builds team culture."),
            ]
        ),
        emitter=emitter,
        system_prompt="You argue for office work.",
        tools=[],
    )

    verdict_json = json.dumps(
        {
            "winner": "pro",
            "reasoning": "The pro side presented stronger evidence on productivity gains.",
            "synthesis": "Remote work wins on productivity; office work excels at culture building.",
        }
    )
    judge_client = MockLLMClient([make_response(verdict_json)])

    debate = Debate(
        debaters=[
            Debater(agent=pro_agent, position="for remote work"),
            Debater(agent=con_agent, position="for office work"),
        ],
        emitter=emitter,
        resolution=LLMJudgeResolution(llm_client=judge_client),
        max_rounds=1,
    )
    result = await debate.run("Is remote work better than office work?")

    assert result.resolution.winner == "pro"
    assert "productivity" in result.resolution.reasoning
    assert "Remote work wins" in result.resolution.synthesis

    print("\n--- Section 2: LLMJudgeResolution with Structured Output ---")
    print(f"  Winner: {result.resolution.winner}")
    print(f"  Reasoning: {result.resolution.reasoning}")
    print(f"  Synthesis: {result.resolution.synthesis}")
    print("✓ Structured verdict with winner parsed correctly")

    # --- Section 3: Custom Evaluation Criteria ---
    # LLMJudgeResolution accepts a criteria parameter that guides the
    # judge's evaluation. We verify the criteria string is included in
    # the prompt sent to the LLM.

    emitter = make_emitter("debate-s3")

    pro_agent = ReActAgent(
        name="pro",
        llm_client=MockLLMClient(
            [
                make_response("Studies show 70% of remote workers report higher output."),
            ]
        ),
        emitter=emitter,
        system_prompt="You argue with data.",
        tools=[],
    )
    con_agent = ReActAgent(
        name="con",
        llm_client=MockLLMClient(
            [
                make_response("In-office teams ship features 2x faster due to real-time collaboration."),
            ]
        ),
        emitter=emitter,
        system_prompt="You argue with data.",
        tools=[],
    )

    criteria = "Focus on empirical evidence and data"
    verdict_json = json.dumps(
        {
            "winner": "con",
            "reasoning": "Both cited data, but the con side's claims were more specific.",
            "synthesis": "Empirical evidence supports both sides with different metrics.",
        }
    )
    judge_client = MockLLMClient([make_response(verdict_json)])

    debate = Debate(
        debaters=[
            Debater(agent=pro_agent, position="for remote work"),
            Debater(agent=con_agent, position="for office work"),
        ],
        emitter=emitter,
        resolution=LLMJudgeResolution(llm_client=judge_client, criteria=criteria),
        max_rounds=1,
    )
    result = await debate.run("Is remote work better than office work?")

    # Verify criteria was passed to the LLM in the user message
    assert len(judge_client.calls) == 1
    user_message = judge_client.calls[0]["messages"][0].content
    assert criteria in user_message

    assert result.resolution.winner == "con"

    print("\n--- Section 3: Custom Evaluation Criteria ---")
    print(f"  Criteria: {criteria}")
    print("  Criteria found in LLM prompt: True")
    print(f"  Winner: {result.resolution.winner}")
    print("✓ Custom criteria passed to judge LLM")

    # --- Section 4: Custom Resolution Strategy ---
    # The ResolutionStrategy is a Protocol — any class with
    # async def resolve(transcript, task) -> DebateResolution works.
    # Here we implement a strategy that picks the debater who wrote
    # the most total content.

    class LongestArgumentResolution:
        """Picks the debater who produced the most total content."""

        async def resolve(self, transcript: list[Argument], task: str) -> DebateResolution:
            totals: dict[str, int] = {}
            positions: dict[str, str] = {}
            for arg in transcript:
                totals[arg.agent_name] = totals.get(arg.agent_name, 0) + len(arg.content)
                positions[arg.agent_name] = arg.position
            winner = max(totals, key=totals.get)  # type: ignore[arg-type]
            return DebateResolution(
                winner=winner,
                reasoning=f"{winner} wrote {totals[winner]} characters total.",
                synthesis=f"Winner by volume: {winner} ({positions[winner]}).",
            )

    # Verify Protocol conformance
    assert isinstance(LongestArgumentResolution(), ResolutionStrategy)

    emitter = make_emitter("debate-s4")

    short_agent = ReActAgent(
        name="brief",
        llm_client=MockLLMClient([make_response("Keep it simple.")]),
        emitter=emitter,
        system_prompt="You are brief.",
        tools=[],
    )
    verbose_agent = ReActAgent(
        name="verbose",
        llm_client=MockLLMClient(
            [
                make_response(
                    "Let me elaborate extensively on why this approach has significant "
                    "merit when considering the full context of the situation at hand."
                ),
            ]
        ),
        emitter=emitter,
        system_prompt="You are verbose.",
        tools=[],
    )

    debate = Debate(
        debaters=[
            Debater(agent=short_agent, position="for brevity"),
            Debater(agent=verbose_agent, position="for detail"),
        ],
        emitter=emitter,
        resolution=LongestArgumentResolution(),
        max_rounds=1,
    )
    result = await debate.run("Should documentation be brief or detailed?")

    assert result.resolution.winner == "verbose"

    print("\n--- Section 4: Custom Resolution Strategy ---")
    print("  Strategy: LongestArgumentResolution (Protocol-based)")
    print("  Satisfies ResolutionStrategy Protocol: True")
    print(f"  Winner: {result.resolution.winner}")
    print(f"  Reasoning: {result.resolution.reasoning}")
    print("✓ Custom resolution strategy works via Protocol")

    # --- Section 5: Multi-Party Debate (3 Debaters) ---
    # Debate supports more than 2 participants. Each debater argues
    # their own position across all rounds.

    emitter = make_emitter("debate-s5")

    rewrite_agent = ReActAgent(
        name="rewriter",
        llm_client=MockLLMClient(
            [
                make_response("A clean rewrite eliminates accumulated technical debt entirely."),
                make_response("Rewriting lets us adopt modern architecture from the start."),
            ]
        ),
        emitter=emitter,
        system_prompt="You argue for rewriting from scratch.",
        tools=[],
    )
    refactor_agent = ReActAgent(
        name="refactorer",
        llm_client=MockLLMClient(
            [
                make_response("Incremental refactoring delivers value continuously without big-bang risk."),
                make_response("Refactoring preserves working features while improving code quality."),
            ]
        ),
        emitter=emitter,
        system_prompt="You argue for incremental refactoring.",
        tools=[],
    )
    leave_agent = ReActAgent(
        name="pragmatist",
        llm_client=MockLLMClient(
            [
                make_response("If it works, don't fix it. Focus engineering effort on new features."),
                make_response("Technical debt is only a problem when it blocks progress. Ours doesn't."),
            ]
        ),
        emitter=emitter,
        system_prompt="You argue for leaving it alone.",
        tools=[],
    )

    judge_agent = ReActAgent(
        name="judge",
        llm_client=MockLLMClient(
            [
                make_response("The refactoring approach balances risk and improvement best."),
            ]
        ),
        emitter=emitter,
        system_prompt="You are an impartial judge.",
        tools=[],
    )

    debate = Debate(
        debaters=[
            Debater(agent=rewrite_agent, position="rewrite from scratch"),
            Debater(agent=refactor_agent, position="incremental refactoring"),
            Debater(agent=leave_agent, position="leave it alone"),
        ],
        emitter=emitter,
        resolution=JudgeResolution(judge=judge_agent),
        max_rounds=2,
    )
    result = await debate.run("What's the best approach to technical debt?")

    # 3 debaters × 2 rounds = 6 arguments
    assert len(result.transcript) == 6
    assert result.rounds_completed == 2

    # Each round has 3 arguments
    round_1 = [a for a in result.transcript if a.round == 1]
    round_2 = [a for a in result.transcript if a.round == 2]
    assert len(round_1) == 3
    assert len(round_2) == 3

    # All three positions appear
    positions = {a.position for a in result.transcript}
    assert positions == {"rewrite from scratch", "incremental refactoring", "leave it alone"}

    print("\n--- Section 5: Multi-Party Debate (3 Debaters) ---")
    print("  Topic: What's the best approach to technical debt?")
    debater_names = sorted({arg.agent_name for arg in result.transcript})
    print(f"  Debaters: {debater_names}")
    print(f"  Total arguments: {len(result.transcript)} (3 debaters × 2 rounds)")
    for arg in result.transcript:
        print(f"  Round {arg.round} [{arg.agent_name}]: {arg.content[:50]}...")
    print("✓ Multi-party debate with 3 positions verified")

    # --- Section 6: Event Verification ---
    # Verifies the complete event stream: order, types, and data fields.

    emitter = make_emitter("debate-s6")

    pro_agent = ReActAgent(
        name="pro",
        llm_client=MockLLMClient(
            [
                make_response("Opening argument for the proposal."),
                make_response("Rebuttal strengthening the proposal."),
            ]
        ),
        emitter=emitter,
        system_prompt="You argue for the proposal.",
        tools=[],
    )
    con_agent = ReActAgent(
        name="con",
        llm_client=MockLLMClient(
            [
                make_response("Opening argument against the proposal."),
                make_response("Rebuttal against the proposal."),
            ]
        ),
        emitter=emitter,
        system_prompt="You argue against the proposal.",
        tools=[],
    )
    judge_agent = ReActAgent(
        name="judge",
        llm_client=MockLLMClient(
            [
                make_response("The pro side presented a more coherent case."),
            ]
        ),
        emitter=emitter,
        system_prompt="You are an impartial judge.",
        tools=[],
    )

    debate = Debate(
        debaters=[
            Debater(agent=pro_agent, position="for the proposal"),
            Debater(agent=con_agent, position="against the proposal"),
        ],
        emitter=emitter,
        resolution=JudgeResolution(judge=judge_agent),
        max_rounds=2,
    )
    task = "Should we adopt the proposal?"
    result = await debate.run(task)

    # Filter debate events (ignore agent-internal events like AgentStartEvent)
    start_events = [e for e in emitter.events if isinstance(e, DebateStartEvent)]
    argument_events = [e for e in emitter.events if isinstance(e, DebateArgumentEvent)]
    resolution_events = [e for e in emitter.events if isinstance(e, DebateResolutionEvent)]
    complete_events = [e for e in emitter.events if isinstance(e, DebateCompleteEvent)]

    # 1. DebateStartEvent
    assert len(start_events) == 1
    start = start_events[0]
    assert start.task == task
    assert start.debater_names == ["pro", "con"]
    assert start.positions == {"pro": "for the proposal", "con": "against the proposal"}
    assert start.max_rounds == 2
    assert start.resolution_strategy == "JudgeResolution"

    # 2. DebateArgumentEvent — 4 total (2 debaters × 2 rounds)
    assert len(argument_events) == 4
    assert argument_events[0].round == 1
    assert argument_events[0].agent_name == "pro"
    assert argument_events[0].position == "for the proposal"
    assert "Opening" in argument_events[0].argument
    assert argument_events[1].round == 1
    assert argument_events[1].agent_name == "con"
    assert argument_events[2].round == 2
    assert argument_events[2].agent_name == "pro"
    assert argument_events[3].round == 2
    assert argument_events[3].agent_name == "con"

    # 3. DebateResolutionEvent
    assert len(resolution_events) == 1
    assert resolution_events[0].winner is None  # JudgeResolution
    assert "coherent" in resolution_events[0].reasoning
    assert resolution_events[0].rounds_completed == 2

    # 4. DebateCompleteEvent
    assert len(complete_events) == 1
    assert complete_events[0].winner is None
    assert complete_events[0].rounds_completed == 2
    assert complete_events[0].total_arguments == 4
    assert complete_events[0].termination_reason == "max_rounds"

    print("\n--- Section 6: Event Verification ---")
    print(f"  DebateStartEvent: task={start.task}, strategy={start.resolution_strategy}")
    print(f"  DebateArgumentEvent: {len(argument_events)} events")
    for ae in argument_events:
        print(f"    Round {ae.round} [{ae.agent_name}]: {ae.argument[:40]}...")
    print(
        f"  DebateResolutionEvent: winner={resolution_events[0].winner}, rounds={resolution_events[0].rounds_completed}"
    )
    print(f"  DebateCompleteEvent: total_arguments={complete_events[0].total_arguments}")
    print("✓ All event types verified in correct order")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
