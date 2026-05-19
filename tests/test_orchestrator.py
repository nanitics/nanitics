import pytest
from pydantic import BaseModel

from nanitics import (
    InMemoryEmitter,
    MockLLMClient,
    ReActAgent,
    ToolCall,
)
from nanitics.composition.multi_agent.agent_tool import AgentTool
from nanitics.composition.multi_agent.orchestrator import (
    create_orchestrator,
    orchestrator_prompt_section,
)
from nanitics.infrastructure.observability.events import DelegationEvent, EvaluationEvent
from nanitics.patterns import FinalOutputStrategy
from nanitics.strategies.agents.evaluation import (
    EvaluationContext,
    EvaluationResult,
    EvaluationVerdict,
)
from nanitics.strategies.agents.reasoning import ReasoningAgent
from tests.testing_helpers import make_emitter, make_response


def make_specialist(name: str, description: str, response: str, emitter: InMemoryEmitter) -> AgentTool:
    agent = ReasoningAgent(
        name=name,
        llm_client=MockLLMClient([make_response(response)]),
        emitter=emitter,
        system_prompt=f"You are {name}.",
    )
    return AgentTool(agent=agent, emitter=emitter, description=description)


# ── orchestrator_prompt_section Tests ──────────────────────


class TestOrchestratorPromptSection:
    def test_returns_orchestration_tuple(self) -> None:
        emitter = make_emitter()
        specialists = [
            make_specialist("researcher", "Researches topics", "result", emitter),
            make_specialist("writer", "Writes content", "result", emitter),
        ]
        name, content = orchestrator_prompt_section(specialists)
        assert name == "Orchestration"
        assert isinstance(content, str)

    def test_includes_specialist_listing(self) -> None:
        emitter = make_emitter()
        specialists = [
            make_specialist("researcher", "Researches topics in depth", "result", emitter),
            make_specialist("writer", "Writes polished articles", "result", emitter),
        ]
        _, content = orchestrator_prompt_section(specialists)
        assert "**researcher**" in content
        assert "Researches topics in depth" in content
        assert "**writer**" in content
        assert "Writes polished articles" in content

    def test_includes_strategy_instructions(self) -> None:
        emitter = make_emitter()
        specialists = [make_specialist("a", "does a", "r", emitter)]
        _, content = orchestrator_prompt_section(specialists)
        assert "Analyze the task" in content
        assert "Break complex tasks" in content
        assert "Delegate" in content
        # The combined "assess / re-delegate if insufficient" clause.
        assert "assess completeness and quality" in content
        assert "re-delegate with refined instructions" in content
        # The positive-principle rewrite of step 6 — preserve substance
        # rather than compress. No "when you see X, do Y" wording.
        assert "Combine the specialists' findings" in content
        assert "preserving their substance" in content

    def test_no_case_branching_language(self) -> None:
        """The strategy prompt states principles, not per-case instructions."""
        emitter = make_emitter()
        specialists = [make_specialist("a", "does a", "r", emitter)]
        _, content = orchestrator_prompt_section(specialists)
        # Guard against the case-specific-instruction anti-pattern:
        # "when you see X, do Y" or a single-vs-multiple branch.
        lowered = content.lower()
        assert "when you see" not in lowered
        assert "if there is only one" not in lowered
        assert "when there is only one" not in lowered


# ── create_orchestrator Tests ──────────────────────────────


class TestCreateOrchestrator:
    def test_returns_react_agent(self) -> None:
        emitter = make_emitter()
        specialists = [make_specialist("s1", "desc", "r", emitter)]
        agent = create_orchestrator(
            name="orch",
            llm_client=MockLLMClient([make_response()]),
            emitter=emitter,
            specialists=specialists,
        )
        assert isinstance(agent, ReActAgent)

    def test_agent_has_specialists_as_tools(self) -> None:
        emitter = make_emitter()
        s1 = make_specialist("researcher", "Researches", "r", emitter)
        s2 = make_specialist("writer", "Writes", "r", emitter)
        agent = create_orchestrator(
            name="orch",
            llm_client=MockLLMClient([make_response()]),
            emitter=emitter,
            specialists=[s1, s2],
        )
        schemas = agent._tool_registry.list_schemas()
        schema_names = {s.name for s in schemas}
        assert "researcher" in schema_names
        assert "writer" in schema_names

    def test_generated_prompt_contains_specialist_info(self) -> None:
        emitter = make_emitter()
        s1 = make_specialist("researcher", "Deep research", "r", emitter)
        agent = create_orchestrator(
            name="orch",
            llm_client=MockLLMClient([make_response()]),
            emitter=emitter,
            specialists=[s1],
        )
        assert "orchestrating agent" in agent._system_prompt
        assert "researcher" in agent._system_prompt
        assert "Deep research" in agent._system_prompt

    def test_custom_system_prompt_override(self) -> None:
        emitter = make_emitter()
        s1 = make_specialist("s1", "desc", "r", emitter)
        agent = create_orchestrator(
            name="orch",
            llm_client=MockLLMClient([make_response()]),
            emitter=emitter,
            specialists=[s1],
            system_prompt="Custom prompt.",
        )
        assert agent._system_prompt.startswith("Custom prompt.")

    def test_parameters_pass_through(self) -> None:
        emitter = make_emitter()
        s1 = make_specialist("s1", "desc", "r", emitter)
        agent = create_orchestrator(
            name="custom-orch",
            llm_client=MockLLMClient([make_response()]),
            emitter=emitter,
            specialists=[s1],
            max_iterations=25,
        )
        assert agent.name == "custom-orch"
        assert agent._limiter._max_iterations == 25


# ── Integration Test ───────────────────────────────────────


class TestOrchestratorIntegration:
    async def test_orchestrator_delegates_and_synthesizes(self) -> None:
        """Orchestrator delegates to specialists and synthesizes result."""
        emitter = make_emitter()

        s1 = make_specialist("researcher", "Researches topics", "AI is transformative", emitter)
        s2 = make_specialist("writer", "Writes articles", "A well-written article", emitter)

        orchestrator_client = MockLLMClient(
            [
                # Step 1: orchestrator delegates to researcher
                make_response(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="tc1",
                            name="researcher",
                            arguments={"task": "Research AI trends"},
                        )
                    ],
                ),
                # Step 2: orchestrator delegates to writer
                make_response(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="tc2",
                            name="writer",
                            arguments={"task": "Write about AI trends"},
                        )
                    ],
                ),
                # Step 3: orchestrator synthesizes
                make_response(content="Here is the comprehensive article about AI trends."),
            ]
        )

        orch = create_orchestrator(
            name="orchestrator",
            llm_client=orchestrator_client,
            emitter=emitter,
            specialists=[s1, s2],
        )

        result = await orch.run("Write an article about AI trends")

        assert result.output == "Here is the comprehensive article about AI trends."
        assert result.total_steps == 3

        # Verify delegation events carry the orchestrator name as caller
        delegation_events = [e for e in emitter.events if isinstance(e, DelegationEvent)]
        assert len(delegation_events) == 2
        assert all(e.caller_agent == "orchestrator" for e in delegation_events)


# ── FinalOutputStrategy.RELAY_LAST Tests ───────────────────


class TestRelayLastStrategy:
    async def test_relay_last_single_specialist_verbatim(self) -> None:
        """RELAY_LAST returns the last tool_result content verbatim."""
        emitter = make_emitter()

        long_article = (
            "A substantial specialist article with concrete detail. "
            "This is the deliverable the caller wants returned exactly as produced — "
            "not summarised, not paraphrased, not compressed into a meta-description."
        )

        writer = make_specialist("writer", "Writes articles", long_article, emitter)

        orchestrator_client = MockLLMClient(
            [
                # Step 1: delegate to writer.
                make_response(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="tc1",
                            name="writer",
                            arguments={"task": "Write the article"},
                        )
                    ],
                ),
                # Step 2: coordinator would synthesise — under RELAY_LAST
                # this text is discarded in favour of the tool_result.
                make_response(content="Short meta-description."),
            ]
        )

        orch = create_orchestrator(
            name="orchestrator",
            llm_client=orchestrator_client,
            emitter=emitter,
            specialists=[writer],
            final_output_strategy=FinalOutputStrategy.RELAY_LAST,
        )

        result = await orch.run("Write the article")

        assert result.output == long_article
        assert "Short meta-description" not in (result.output or "")

    async def test_relay_last_multi_specialist_returns_most_recent(self) -> None:
        """RELAY_LAST returns the *last* specialist tool_result, not the first."""
        emitter = make_emitter()

        researcher_output = "Research findings about the topic."
        writer_output = "Polished article drawing on the research findings."

        researcher = make_specialist("researcher", "Researches", researcher_output, emitter)
        writer = make_specialist("writer", "Writes", writer_output, emitter)

        orchestrator_client = MockLLMClient(
            [
                make_response(
                    content=None,
                    tool_calls=[
                        ToolCall(id="tc1", name="researcher", arguments={"task": "Research"}),
                    ],
                ),
                make_response(
                    content=None,
                    tool_calls=[
                        ToolCall(id="tc2", name="writer", arguments={"task": "Write"}),
                    ],
                ),
                make_response(content="Coordinator synthesis text."),
            ]
        )

        orch = create_orchestrator(
            name="orchestrator",
            llm_client=orchestrator_client,
            emitter=emitter,
            specialists=[researcher, writer],
            final_output_strategy=FinalOutputStrategy.RELAY_LAST,
        )

        result = await orch.run("Write an article")

        assert result.output == writer_output

    async def test_relay_last_fallback_when_no_tool_result(self) -> None:
        """RELAY_LAST falls back to assistant text if the coordinator never delegated."""
        emitter = make_emitter()

        s1 = make_specialist("researcher", "Researches", "result", emitter)

        # Coordinator answers directly without delegating — there is no
        # tool_result to relay. RELAY_LAST falls back to the assistant text.
        orchestrator_client = MockLLMClient([make_response(content="Direct answer without delegation.")])

        orch = create_orchestrator(
            name="orchestrator",
            llm_client=orchestrator_client,
            emitter=emitter,
            specialists=[s1],
            final_output_strategy=FinalOutputStrategy.RELAY_LAST,
        )

        result = await orch.run("Answer directly")

        assert result.output == "Direct answer without delegation."

    def test_relay_last_rejects_output_schema_at_construction(self) -> None:
        """RELAY_LAST + output_schema is incompatible — rejected at construction."""
        emitter = make_emitter()
        s1 = make_specialist("s1", "desc", "r", emitter)

        class StructuredOutput(BaseModel):
            summary: str

        with pytest.raises(ValueError, match="RELAY_LAST is incompatible with output_schema"):
            create_orchestrator(
                name="orch",
                llm_client=MockLLMClient([make_response()]),
                emitter=emitter,
                specialists=[s1],
                output_schema=StructuredOutput,
                final_output_strategy=FinalOutputStrategy.RELAY_LAST,
            )

    def test_synthesize_with_output_schema_is_allowed(self) -> None:
        """Default SYNTHESIZE + output_schema remains a valid combination."""
        emitter = make_emitter()
        s1 = make_specialist("s1", "desc", "r", emitter)

        class StructuredOutput(BaseModel):
            summary: str

        # No exception: SYNTHESIZE + output_schema is unchanged behaviour.
        agent = create_orchestrator(
            name="orch",
            llm_client=MockLLMClient([make_response()]),
            emitter=emitter,
            specialists=[s1],
            output_schema=StructuredOutput,
        )
        assert isinstance(agent, ReActAgent)

    async def test_relay_last_evaluator_revise_downgrades_to_skipped(self) -> None:
        """Under RELAY_LAST, evaluator REVISE is non-actionable — marks evaluation_skipped."""
        emitter = make_emitter()

        writer_output = "The specialist's deliverable."
        writer = make_specialist("writer", "Writes", writer_output, emitter)

        class ReviseEvaluator:
            @property
            def max_revisions(self) -> int:
                return 2

            async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
                return EvaluationResult(
                    verdict=EvaluationVerdict.REVISE,
                    feedback="Please expand.",
                    evaluator_name="relay-test-eval",
                )

        orchestrator_client = MockLLMClient(
            [
                make_response(
                    content=None,
                    tool_calls=[
                        ToolCall(id="tc1", name="writer", arguments={"task": "Write"}),
                    ],
                ),
                make_response(content="Coordinator synthesis."),
            ]
        )

        orch = create_orchestrator(
            name="orchestrator",
            llm_client=orchestrator_client,
            emitter=emitter,
            specialists=[writer],
            final_output_strategy=FinalOutputStrategy.RELAY_LAST,
            output_evaluator=ReviseEvaluator(),
        )

        result = await orch.run("Write")

        # Output is the relayed specialist content, not revised.
        assert result.output == writer_output
        # Termination is downgraded because REVISE was non-actionable.
        assert result.termination_reason == "evaluation_skipped"
        # An EvaluationEvent is emitted for the relayed-content check.
        eval_events = [e for e in emitter.events if isinstance(e, EvaluationEvent)]
        assert any(e.evaluator_name == "relay-test-eval" and e.verdict == "revise" for e in eval_events)

    async def test_relay_last_evaluator_accept_keeps_complete(self) -> None:
        """Under RELAY_LAST, evaluator ACCEPT preserves termination_reason='complete'."""
        emitter = make_emitter()

        writer_output = "The specialist's deliverable."
        writer = make_specialist("writer", "Writes", writer_output, emitter)

        class AcceptEvaluator:
            @property
            def max_revisions(self) -> int:
                return 2

            async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
                return EvaluationResult(
                    verdict=EvaluationVerdict.ACCEPT,
                    evaluator_name="relay-test-eval",
                )

        orchestrator_client = MockLLMClient(
            [
                make_response(
                    content=None,
                    tool_calls=[
                        ToolCall(id="tc1", name="writer", arguments={"task": "Write"}),
                    ],
                ),
                make_response(content="Coordinator synthesis."),
            ]
        )

        orch = create_orchestrator(
            name="orchestrator",
            llm_client=orchestrator_client,
            emitter=emitter,
            specialists=[writer],
            final_output_strategy=FinalOutputStrategy.RELAY_LAST,
            output_evaluator=AcceptEvaluator(),
        )

        result = await orch.run("Write")

        assert result.output == writer_output
        assert result.termination_reason == "complete"

    async def test_relay_last_evaluator_reject_marks_failed(self) -> None:
        """Under RELAY_LAST, evaluator REJECT marks termination_reason='evaluation_failed'."""
        emitter = make_emitter()

        writer_output = "The specialist's deliverable."
        writer = make_specialist("writer", "Writes", writer_output, emitter)

        class RejectEvaluator:
            @property
            def max_revisions(self) -> int:
                return 2

            async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
                return EvaluationResult(
                    verdict=EvaluationVerdict.REJECT,
                    feedback="Not acceptable.",
                    evaluator_name="relay-test-eval",
                )

        orchestrator_client = MockLLMClient(
            [
                make_response(
                    content=None,
                    tool_calls=[
                        ToolCall(id="tc1", name="writer", arguments={"task": "Write"}),
                    ],
                ),
                make_response(content="Coordinator synthesis."),
            ]
        )

        orch = create_orchestrator(
            name="orchestrator",
            llm_client=orchestrator_client,
            emitter=emitter,
            specialists=[writer],
            final_output_strategy=FinalOutputStrategy.RELAY_LAST,
            output_evaluator=RejectEvaluator(),
        )

        result = await orch.run("Write")

        assert result.output == writer_output
        assert result.termination_reason == "evaluation_failed"

    async def test_relay_last_evaluator_error_marks_skipped(self) -> None:
        """Under RELAY_LAST, EVALUATOR_ERROR marks termination_reason='evaluation_skipped'."""
        emitter = make_emitter()

        writer_output = "The specialist's deliverable."
        writer = make_specialist("writer", "Writes", writer_output, emitter)

        class ErrorEvaluator:
            @property
            def max_revisions(self) -> int:
                return 2

            async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
                return EvaluationResult(
                    verdict=EvaluationVerdict.EVALUATOR_ERROR,
                    evaluator_name="relay-test-eval",
                    error_detail="boom",
                )

        orchestrator_client = MockLLMClient(
            [
                make_response(
                    content=None,
                    tool_calls=[
                        ToolCall(id="tc1", name="writer", arguments={"task": "Write"}),
                    ],
                ),
                make_response(content="Coordinator synthesis."),
            ]
        )

        orch = create_orchestrator(
            name="orchestrator",
            llm_client=orchestrator_client,
            emitter=emitter,
            specialists=[writer],
            final_output_strategy=FinalOutputStrategy.RELAY_LAST,
            output_evaluator=ErrorEvaluator(),
        )

        result = await orch.run("Write")

        assert result.output == writer_output
        assert result.termination_reason == "evaluation_skipped"

    def test_final_output_strategy_importable_from_patterns(self) -> None:
        """FinalOutputStrategy is part of the ``nanitics.patterns`` public API."""
        from nanitics.patterns import FinalOutputStrategy as PatternsEnum

        assert PatternsEnum.SYNTHESIZE.value == "synthesize"
        assert PatternsEnum.RELAY_LAST.value == "relay_last"
