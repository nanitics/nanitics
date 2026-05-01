"""Evaluation: quality gates, revision loops, and composite evaluation pipelines.

Demonstrates the SDK's evaluation system — how evaluators assess agent output,
trigger revision loops, and how to compose different evaluator types. Covers
ProgrammaticEvaluator, LLMEvaluator, CompositeEvaluator, and custom evaluators
via the OutputEvaluator protocol.

Related guide: docs/guides/evaluation.md
"""

import asyncio
import json

from examples.helpers import make_emitter, make_response
from nanitics import (
    CompositeEvaluator,
    EvaluationCheck,
    EvaluationContext,
    EvaluationResult,
    EvaluationVerdict,
    LLMEvaluator,
    MockLLMClient,
    OutputEvaluator,
    ProgrammaticEvaluator,
    ReActAgent,
)
from nanitics.infrastructure import (
    EvaluationEvent,
    EvaluationRevisionEvent,
)


async def main() -> None:
    # --- Section 1: ProgrammaticEvaluator — Accept on First Try ---
    print("--- Section 1: ProgrammaticEvaluator — Accept on First Try ---")

    # Define checks: the output must be at least 20 characters and contain "analysis"
    evaluator = ProgrammaticEvaluator(
        checks=[
            EvaluationCheck(
                name="minimum_length",
                check=lambda output: len(output) >= 20,
                feedback="Output must be at least 20 characters.",
            ),
            EvaluationCheck(
                name="contains_analysis",
                check=lambda output: "analysis" in output.lower(),
                feedback="Output must include an analysis section.",
            ),
        ],
        max_revisions=1,
    )

    client = MockLLMClient(
        responses=[
            make_response("Detailed analysis of the competitive landscape and key trends."),
        ]
    )
    emitter = make_emitter("eval-s1")

    agent = ReActAgent(
        name="analyst",
        llm_client=client,
        emitter=emitter,
        system_prompt="Analyze the given topic.",
        tools=[],
        output_evaluator=evaluator,
    )

    result = await agent.run("Analyze the market")

    assert result.termination_reason == "complete"
    assert result.output == "Detailed analysis of the competitive landscape and key trends."

    # Evaluation event emitted with accept verdict
    eval_events = [e for e in emitter.events if isinstance(e, EvaluationEvent)]
    assert len(eval_events) == 1
    assert eval_events[0].verdict == "accept"
    assert eval_events[0].score == 1.0
    assert eval_events[0].evaluator_name == "programmatic"

    print(f"  Output: {result.output}")
    print(f"  Verdict: {eval_events[0].verdict}, Score: {eval_events[0].score}")
    print("✓ All checks passed — output accepted on first try")

    # --- Section 2: ProgrammaticEvaluator — Revision Loop ---
    print("\n--- Section 2: ProgrammaticEvaluator — Revision Loop ---")

    # Same checks, but now the first response fails
    evaluator = ProgrammaticEvaluator(
        checks=[
            EvaluationCheck(
                name="minimum_length",
                check=lambda output: len(output) >= 20,
                feedback="Output must be at least 20 characters.",
            ),
            EvaluationCheck(
                name="contains_analysis",
                check=lambda output: "analysis" in output.lower(),
                feedback="Output must include an analysis section.",
            ),
        ],
        max_revisions=2,
    )

    client = MockLLMClient(
        responses=[
            make_response("Market looks good."),  # Fails: too short, no "analysis"
            make_response("Detailed analysis: the market shows strong growth with emerging trends."),  # Passes
        ]
    )
    emitter = make_emitter("eval-s2")

    agent = ReActAgent(
        name="analyst",
        llm_client=client,
        emitter=emitter,
        system_prompt="Analyze the given topic.",
        tools=[],
        output_evaluator=evaluator,
        max_iterations=10,
    )

    result = await agent.run("Analyze the market")

    assert result.output == "Detailed analysis: the market shows strong growth with emerging trends."
    assert result.termination_reason == "complete"
    assert result.total_steps == 2

    # Inspect evaluation events: first revise, then accept
    eval_events = [e for e in emitter.events if isinstance(e, EvaluationEvent)]
    assert len(eval_events) == 2
    assert eval_events[0].verdict == "revise"
    assert eval_events[0].revision_attempt == 0
    assert eval_events[1].verdict == "accept"
    assert eval_events[1].revision_attempt == 1

    # Revision event carries the feedback injected into the conversation
    revision_events = [e for e in emitter.events if isinstance(e, EvaluationRevisionEvent)]
    assert len(revision_events) == 1
    assert "minimum_length" in revision_events[0].feedback
    assert "contains_analysis" in revision_events[0].feedback
    assert revision_events[0].max_revisions == 2

    print(f"  First attempt: {result.messages[1].content}")
    print(f"  Feedback: {revision_events[0].feedback}")
    print(f"  Revised output: {result.output}")
    print(f"  Steps: {result.total_steps}")
    print("✓ Evaluator rejected first attempt, revision succeeded")

    # --- Section 3: Evaluation Failure — Exhausted Revisions ---
    print("\n--- Section 3: Evaluation Failure — Exhausted Revisions ---")

    evaluator = ProgrammaticEvaluator(
        checks=[
            EvaluationCheck(
                name="impossible_check",
                check=lambda _: False,  # Always fails
                feedback="This requirement can never be met.",
            ),
        ],
        max_revisions=1,
    )

    client = MockLLMClient(
        responses=[
            make_response("First attempt at analysis."),
            make_response("Second attempt at analysis."),
        ]
    )
    emitter = make_emitter("eval-s3")

    agent = ReActAgent(
        name="analyst",
        llm_client=client,
        emitter=emitter,
        system_prompt="Analyze the topic.",
        tools=[],
        output_evaluator=evaluator,
        max_iterations=10,
    )

    result = await agent.run("Analyze something")

    # After exhausting revisions, the last output is returned with evaluation_failed
    assert result.termination_reason == "evaluation_failed"
    assert result.output == "Second attempt at analysis."
    assert result.total_steps == 2

    print(f"  Termination: {result.termination_reason}")
    print(f"  Last output: {result.output}")
    print("✓ Revision budget exhausted — returns last attempt with evaluation_failed")

    # --- Section 4: LLMEvaluator — Accept and Revise ---
    print("\n--- Section 4: LLMEvaluator — Accept and Revise ---")

    # 4a: Score above threshold → accept
    print("  4a: Accept (score above threshold)")

    # LLMEvaluator uses a separate LLM client for evaluation.
    # The evaluator's LLM must return JSON matching: {"score": float, "reasoning": str, "issues": [str]}
    agent_client = MockLLMClient(
        responses=[
            make_response("The market is growing at 15% annually with strong demand in AI and cloud."),
        ]
    )
    evaluator_client = MockLLMClient(
        responses=[
            make_response('{"score": 0.85, "reasoning": "Clear and comprehensive answer", "issues": []}'),
        ]
    )

    evaluator = LLMEvaluator(
        llm_client=evaluator_client,
        criteria="Provide specific data points and clear reasoning.",
        score_threshold=0.7,
    )
    emitter = make_emitter("eval-s4a")

    agent = ReActAgent(
        name="analyst",
        llm_client=agent_client,
        emitter=emitter,
        system_prompt="Analyze the given topic with data.",
        tools=[],
        output_evaluator=evaluator,
    )

    result = await agent.run("Analyze market growth")

    assert result.termination_reason == "complete"

    eval_events = [e for e in emitter.events if isinstance(e, EvaluationEvent)]
    assert len(eval_events) == 1
    assert eval_events[0].verdict == "accept"
    assert eval_events[0].score == 0.85
    assert eval_events[0].evaluator_name == "llm"

    print(f"    Verdict: {eval_events[0].verdict}, Score: {eval_events[0].score}")

    # 4b: Score below threshold → revise → then accept
    print("  4b: Revise then accept")

    agent_client = MockLLMClient(
        responses=[
            make_response("Market is growing."),  # First attempt — too brief
            make_response("The market is growing at 15% annually, driven by AI adoption and cloud migration."),
        ]
    )
    evaluator_client = MockLLMClient(
        responses=[
            make_response(
                '{"score": 0.4, "reasoning": "Missing key details", "issues": ["No data points", "Too brief"]}'
            ),
            make_response('{"score": 0.85, "reasoning": "Much improved with specifics", "issues": []}'),
        ]
    )

    evaluator = LLMEvaluator(
        llm_client=evaluator_client,
        criteria="Provide specific data points and clear reasoning.",
        score_threshold=0.7,
        max_revisions=1,
    )
    emitter = make_emitter("eval-s4b")

    agent = ReActAgent(
        name="analyst",
        llm_client=agent_client,
        emitter=emitter,
        system_prompt="Analyze the given topic with data.",
        tools=[],
        output_evaluator=evaluator,
        max_iterations=10,
    )

    result = await agent.run("Analyze market growth")

    assert result.output == "The market is growing at 15% annually, driven by AI adoption and cloud migration."
    assert result.termination_reason == "complete"

    print(f"    Revised output: {result.output}")
    print("✓ LLMEvaluator scored output, triggered revision, then accepted")

    # --- Section 5: LLMEvaluator — Reject Threshold ---
    print("\n--- Section 5: LLMEvaluator — Reject Threshold ---")

    # When score falls below reject_threshold, output is rejected immediately — no revision
    agent_client = MockLLMClient(
        responses=[
            make_response("I like pizza."),  # Completely off-topic
        ]
    )
    evaluator_client = MockLLMClient(
        responses=[
            make_response(
                '{"score": 0.1, "reasoning": "Completely off-topic", "issues": ["Does not address the question"]}'
            ),
        ]
    )

    evaluator = LLMEvaluator(
        llm_client=evaluator_client,
        criteria="Answer must address market analysis.",
        score_threshold=0.7,
        reject_threshold=0.3,
        max_revisions=2,  # Has budget, but reject bypasses it
    )
    emitter = make_emitter("eval-s5")

    agent = ReActAgent(
        name="analyst",
        llm_client=agent_client,
        emitter=emitter,
        system_prompt="Analyze market trends.",
        tools=[],
        output_evaluator=evaluator,
    )

    result = await agent.run("Analyze the market")

    assert result.termination_reason == "evaluation_failed"

    eval_events = [e for e in emitter.events if isinstance(e, EvaluationEvent)]
    assert len(eval_events) == 1
    assert eval_events[0].verdict == "reject"
    assert eval_events[0].score == 0.1

    # No revision attempted — reject is immediate
    revision_events = [e for e in emitter.events if isinstance(e, EvaluationRevisionEvent)]
    assert len(revision_events) == 0

    print(f"  Verdict: {eval_events[0].verdict}, Score: {eval_events[0].score}")
    print("✓ Score below reject_threshold — immediate rejection, no revision")

    # --- Section 6: CompositeEvaluator — Layered Evaluation ---
    print("\n--- Section 6: CompositeEvaluator — Layered Evaluation ---")

    # 6a: Short-circuit on programmatic failure — LLM evaluator never runs
    print("  6a: Short-circuit on programmatic failure")

    programmatic = ProgrammaticEvaluator(
        checks=[
            EvaluationCheck(
                name="minimum_length",
                check=lambda output: len(output) >= 50,
                feedback="Output must be at least 50 characters.",
            ),
        ],
    )

    evaluator_client = MockLLMClient(responses=[])  # No responses — should never be called

    llm_eval = LLMEvaluator(
        llm_client=evaluator_client,
        criteria="Thorough analysis with data points.",
        score_threshold=0.7,
    )

    composite = CompositeEvaluator(
        evaluators=[programmatic, llm_eval],  # Cheap check first
        max_revisions=0,  # No revisions — just evaluate
    )

    agent_client = MockLLMClient(
        responses=[
            make_response("Too short."),  # Fails programmatic check
        ]
    )
    emitter = make_emitter("eval-s6a")

    agent = ReActAgent(
        name="analyst",
        llm_client=agent_client,
        emitter=emitter,
        system_prompt="Analyze the topic.",
        tools=[],
        output_evaluator=composite,
    )

    result = await agent.run("Analyze")

    assert result.termination_reason == "evaluation_failed"

    eval_events = [e for e in emitter.events if isinstance(e, EvaluationEvent)]
    assert len(eval_events) == 1
    assert eval_events[0].evaluator_name == "programmatic"
    assert eval_events[0].verdict == "revise"

    # The LLM evaluator was never called — short-circuit saved the cost
    assert len(evaluator_client.calls) == 0

    print(f"    Failed on: {eval_events[0].evaluator_name}")
    print(f"    LLM evaluator calls: {len(evaluator_client.calls)}")

    # 6b: Programmatic passes, LLM evaluates
    print("  6b: Programmatic passes, LLM evaluates")

    programmatic = ProgrammaticEvaluator(
        checks=[
            EvaluationCheck(
                name="minimum_length",
                check=lambda output: len(output) >= 50,
                feedback="Output must be at least 50 characters.",
            ),
        ],
    )

    evaluator_client = MockLLMClient(
        responses=[
            make_response('{"score": 0.9, "reasoning": "Excellent analysis", "issues": []}'),
        ]
    )

    llm_eval = LLMEvaluator(
        llm_client=evaluator_client,
        criteria="Thorough analysis with data points.",
        score_threshold=0.7,
    )

    composite = CompositeEvaluator(
        evaluators=[programmatic, llm_eval],
        max_revisions=0,
    )

    agent_client = MockLLMClient(
        responses=[
            make_response(
                "Comprehensive analysis: the market shows 20% growth driven by AI innovation and cloud adoption trends."
            ),
        ]
    )
    emitter = make_emitter("eval-s6b")

    agent = ReActAgent(
        name="analyst",
        llm_client=agent_client,
        emitter=emitter,
        system_prompt="Analyze the topic.",
        tools=[],
        output_evaluator=composite,
    )

    result = await agent.run("Analyze")

    assert result.termination_reason == "complete"
    assert len(evaluator_client.calls) == 1  # LLM evaluator was invoked

    print(f"    LLM evaluator calls: {len(evaluator_client.calls)}")
    print("✓ CompositeEvaluator: cheap checks first, expensive LLM only when needed")

    # --- Section 7: Custom Evaluator — OutputEvaluator Protocol ---
    print("\n--- Section 7: Custom Evaluator — OutputEvaluator Protocol ---")

    # Any object with max_revisions property + async evaluate() method satisfies OutputEvaluator
    class JsonFormatEvaluator:
        """Custom evaluator that checks if output is valid JSON."""

        @property
        def max_revisions(self) -> int:
            return 1

        async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
            try:
                json.loads(output)
                return EvaluationResult(
                    verdict=EvaluationVerdict.ACCEPT,
                    score=1.0,
                    evaluator_name="json_format",
                )
            except json.JSONDecodeError as e:
                return EvaluationResult(
                    verdict=EvaluationVerdict.REVISE,
                    score=0.0,
                    feedback=f"Output must be valid JSON. Parse error: {e}",
                    evaluator_name="json_format",
                )

    # Verify it satisfies the protocol
    assert isinstance(JsonFormatEvaluator(), OutputEvaluator)

    client = MockLLMClient(
        responses=[
            make_response('{"market": "growing", "rate": "15%", "drivers": ["AI", "cloud"]}'),
        ]
    )
    emitter = make_emitter("eval-s7")

    agent = ReActAgent(
        name="json-analyst",
        llm_client=client,
        emitter=emitter,
        system_prompt="Return analysis as JSON.",
        tools=[],
        output_evaluator=JsonFormatEvaluator(),
    )

    result = await agent.run("Analyze market")

    assert result.termination_reason == "complete"

    eval_events = [e for e in emitter.events if isinstance(e, EvaluationEvent)]
    assert len(eval_events) == 1
    assert eval_events[0].evaluator_name == "json_format"
    assert eval_events[0].verdict == "accept"

    print(f"  Evaluator: {eval_events[0].evaluator_name}")
    print(f"  Verdict: {eval_events[0].verdict}")
    print("✓ Custom evaluator via OutputEvaluator protocol")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
