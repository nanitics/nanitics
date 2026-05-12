from nanitics import (
    InMemoryEmitter,
    MockEmbeddingClient,
    MockLLMClient,
    ReasoningAgent,
)
from nanitics.capabilities.memory.episodic import (
    InMemoryEpisodeStore,
    OutcomeType,
)
from nanitics.core.agents.evaluation import (
    EvaluationContext,
    EvaluationResult,
    EvaluationVerdict,
    OutputEvaluator,
)
from nanitics.experimental import ReflexionAgent
from nanitics.infrastructure.observability.events import (
    AgentCompleteEvent,
    AgentStartEvent,
    EvaluationEvent,
    LLMRequestEvent,
    LLMResponseEvent,
    ReflectionGeneratedEvent,
)
from nanitics.safety.cancellation import CancellationToken
from tests.testing_helpers import make_emitter, make_response, make_usage


class _AcceptEvaluator:
    @property
    def max_revisions(self) -> int:
        return 2

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        return EvaluationResult(
            verdict=EvaluationVerdict.ACCEPT,
            evaluator_name="test-evaluator",
        )


class _RejectEvaluator:
    @property
    def max_revisions(self) -> int:
        return 1

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        return EvaluationResult(
            verdict=EvaluationVerdict.REJECT,
            feedback="Not acceptable",
            evaluator_name="test-evaluator",
        )


class _RejectThenAcceptEvaluator:
    def __init__(self) -> None:
        self._call_count = 0

    @property
    def max_revisions(self) -> int:
        return 3

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        self._call_count += 1
        if self._call_count == 1:
            return EvaluationResult(
                verdict=EvaluationVerdict.REJECT,
                feedback="Please improve",
                evaluator_name="test-evaluator",
            )
        return EvaluationResult(
            verdict=EvaluationVerdict.ACCEPT,
            evaluator_name="test-evaluator",
        )


def _make_inner_agent(
    llm_client: MockLLMClient,
    outer_emitter: InMemoryEmitter,
    cancellation_token: CancellationToken | None = None,
) -> ReasoningAgent:
    # Distinct inner emitter: proves end-to-end that ReflexionAgent rebinds the
    # inner agent's emitter to the outer Reflexion emitter on every attempt.
    return ReasoningAgent(
        name="inner-agent",
        llm_client=llm_client,
        emitter=InMemoryEmitter(trace_id="inner-unused-bound-by-reflexion"),
        system_prompt="You are an assistant.",
        cancellation_token=cancellation_token,
    )


def _make_reflexion_agent(
    *,
    inner_agent: ReasoningAgent,
    evaluator: OutputEvaluator,
    episode_store: InMemoryEpisodeStore,
    emitter: InMemoryEmitter,
    max_attempts: int = 3,
    cancellation_token: CancellationToken | None = None,
    reflection_llm: MockLLMClient | None = None,
) -> ReflexionAgent:
    return ReflexionAgent(
        name="reflexion-agent",
        llm_client=reflection_llm or MockLLMClient(responses=[]),
        emitter=emitter,
        system_prompt="You are a reflective agent.",
        inner_agent=inner_agent,
        evaluator=evaluator,
        episode_store=episode_store,
        max_attempts=max_attempts,
        cancellation_token=cancellation_token,
    )


# ──────────────────────────────────────────────────────────
# Single-attempt success
# ──────────────────────────────────────────────────────────


class TestSingleAttemptSuccess:
    async def test_returns_result_on_accept(self) -> None:
        emitter = make_emitter()
        inner_llm = MockLLMClient(responses=[make_response("good answer")])
        inner = _make_inner_agent(inner_llm, emitter)
        store = InMemoryEpisodeStore(embedding_client=MockEmbeddingClient())

        agent = _make_reflexion_agent(
            inner_agent=inner,
            evaluator=_AcceptEvaluator(),
            episode_store=store,
            emitter=emitter,
        )

        result = await agent.run("What is 2+2?")

        assert result.output == "good answer"
        assert result.termination_reason == "complete"
        assert result.total_steps == 1

    async def test_no_reflection_on_success(self) -> None:
        emitter = make_emitter()
        inner_llm = MockLLMClient(responses=[make_response("good answer")])
        inner = _make_inner_agent(inner_llm, emitter)
        store = InMemoryEpisodeStore(embedding_client=MockEmbeddingClient())

        agent = _make_reflexion_agent(
            inner_agent=inner,
            evaluator=_AcceptEvaluator(),
            episode_store=store,
            emitter=emitter,
        )

        await agent.run("What is 2+2?")

        reflection_events = [e for e in emitter.events if isinstance(e, ReflectionGeneratedEvent)]
        assert len(reflection_events) == 0

    async def test_success_episode_stored(self) -> None:
        emitter = make_emitter()
        inner_llm = MockLLMClient(responses=[make_response("good answer")])
        inner = _make_inner_agent(inner_llm, emitter)
        store = InMemoryEpisodeStore(embedding_client=MockEmbeddingClient())

        agent = _make_reflexion_agent(
            inner_agent=inner,
            evaluator=_AcceptEvaluator(),
            episode_store=store,
            emitter=emitter,
        )

        await agent.run("What is 2+2?")

        assert await store.count() == 1
        episodes = await store.recall("What is 2+2?")
        assert episodes[0].episode.outcome == OutcomeType.SUCCESS
        assert episodes[0].episode.reflection is None


# ──────────────────────────────────────────────────────────
# Retry on failure
# ──────────────────────────────────────────────────────────


class TestRetryOnFailure:
    async def test_retry_produces_reflection_then_succeeds(self) -> None:
        emitter = make_emitter()
        inner_llm = MockLLMClient(
            responses=[
                make_response("bad answer"),
                make_response("good answer"),
            ]
        )
        inner = _make_inner_agent(inner_llm, emitter)
        store = InMemoryEpisodeStore(embedding_client=MockEmbeddingClient())
        reflection_llm = MockLLMClient(responses=[make_response("Try a different approach")])

        agent = _make_reflexion_agent(
            inner_agent=inner,
            evaluator=_RejectThenAcceptEvaluator(),
            episode_store=store,
            emitter=emitter,
            reflection_llm=reflection_llm,
        )

        result = await agent.run("What is 2+2?")

        assert result.output == "good answer"
        assert result.termination_reason == "complete"
        assert result.total_steps == 2

    async def test_reflection_generated_event_emitted(self) -> None:
        emitter = make_emitter()
        inner_llm = MockLLMClient(
            responses=[
                make_response("bad answer"),
                make_response("good answer"),
            ]
        )
        inner = _make_inner_agent(inner_llm, emitter)
        store = InMemoryEpisodeStore(embedding_client=MockEmbeddingClient())
        reflection_llm = MockLLMClient(responses=[make_response("Try a different approach")])

        agent = _make_reflexion_agent(
            inner_agent=inner,
            evaluator=_RejectThenAcceptEvaluator(),
            episode_store=store,
            emitter=emitter,
            reflection_llm=reflection_llm,
        )

        await agent.run("What is 2+2?")

        reflection_events = [e for e in emitter.events if isinstance(e, ReflectionGeneratedEvent)]
        assert len(reflection_events) == 1
        assert reflection_events[0].attempt_number == 1
        assert reflection_events[0].max_attempts == 3
        assert reflection_events[0].reflection_text == "Try a different approach"
        assert reflection_events[0].evaluation_feedback == "Please improve"

    async def test_reflection_emits_llm_trace_events(self) -> None:
        emitter = make_emitter()
        inner_llm = MockLLMClient(
            responses=[
                make_response("bad answer"),
                make_response("good answer"),
            ]
        )
        inner = _make_inner_agent(inner_llm, emitter)
        store = InMemoryEpisodeStore(embedding_client=MockEmbeddingClient())
        reflection_llm = MockLLMClient(responses=[make_response("Try a different approach")])

        agent = _make_reflexion_agent(
            inner_agent=inner,
            evaluator=_RejectThenAcceptEvaluator(),
            episode_store=store,
            emitter=emitter,
            reflection_llm=reflection_llm,
        )

        await agent.run("What is 2+2?")

        request_events = [e for e in emitter.events if isinstance(e, LLMRequestEvent) and e.label == "reflection"]
        response_events = [e for e in emitter.events if isinstance(e, LLMResponseEvent) and e.label == "reflection"]
        assert len(request_events) == 1
        assert len(response_events) == 1

    async def test_reflection_stored_in_episode(self) -> None:
        emitter = make_emitter()
        inner_llm = MockLLMClient(
            responses=[
                make_response("bad answer"),
                make_response("good answer"),
            ]
        )
        inner = _make_inner_agent(inner_llm, emitter)
        store = InMemoryEpisodeStore(embedding_client=MockEmbeddingClient())
        reflection_llm = MockLLMClient(responses=[make_response("Try a different approach")])

        agent = _make_reflexion_agent(
            inner_agent=inner,
            evaluator=_RejectThenAcceptEvaluator(),
            episode_store=store,
            emitter=emitter,
            reflection_llm=reflection_llm,
        )

        await agent.run("What is 2+2?")

        assert await store.count() == 2
        episodes = await store.recall("What is 2+2?", limit=10)
        reflections = [e.episode.reflection for e in episodes if e.episode.reflection]
        assert "Try a different approach" in reflections

    async def test_failed_attempt_episode_has_failure_outcome(self) -> None:
        emitter = make_emitter()
        inner_llm = MockLLMClient(
            responses=[
                make_response("bad answer"),
                make_response("good answer"),
            ]
        )
        inner = _make_inner_agent(inner_llm, emitter)
        store = InMemoryEpisodeStore(embedding_client=MockEmbeddingClient())
        reflection_llm = MockLLMClient(responses=[make_response("Try a different approach")])

        agent = _make_reflexion_agent(
            inner_agent=inner,
            evaluator=_RejectThenAcceptEvaluator(),
            episode_store=store,
            emitter=emitter,
            reflection_llm=reflection_llm,
        )

        await agent.run("What is 2+2?")

        episodes = await store.recall("What is 2+2?", limit=10)
        failed = [e.episode for e in episodes if e.episode.reflection is not None]
        assert len(failed) == 1
        assert failed[0].outcome == OutcomeType.FAILURE
        assert failed[0].reflection == "Try a different approach"

    async def test_failed_attempt_episode_carries_verbatim_evaluator_feedback(self) -> None:
        """The per-attempt failure episode preserves the evaluator's verbatim feedback.

        This is the SDK-fix observable: ``EpisodicMemoryProvider.provide()``
        renders this verbatim string into the recalled context block ahead
        of the LLM-generated reflection paraphrase.
        """
        emitter = make_emitter()
        inner_llm = MockLLMClient(
            responses=[
                make_response("bad answer"),
                make_response("good answer"),
            ]
        )
        inner = _make_inner_agent(inner_llm, emitter)
        store = InMemoryEpisodeStore(embedding_client=MockEmbeddingClient())
        reflection_llm = MockLLMClient(responses=[make_response("Try a different approach")])

        agent = _make_reflexion_agent(
            inner_agent=inner,
            evaluator=_RejectThenAcceptEvaluator(),
            episode_store=store,
            emitter=emitter,
            reflection_llm=reflection_llm,
        )

        await agent.run("What is 2+2?")

        episodes = await store.recall("What is 2+2?", limit=10)
        failed = [e.episode for e in episodes if e.episode.outcome == OutcomeType.FAILURE]
        assert len(failed) == 1
        # _RejectThenAcceptEvaluator emits feedback="Please improve" on the rejection
        assert failed[0].evaluator_feedback == "Please improve"

    async def test_accepted_attempt_episode_has_no_evaluator_feedback(self) -> None:
        """Accepted attempts have no rejection feedback to forward."""
        emitter = make_emitter()
        inner_llm = MockLLMClient(responses=[make_response("good answer")])
        inner = _make_inner_agent(inner_llm, emitter)
        store = InMemoryEpisodeStore(embedding_client=MockEmbeddingClient())

        agent = _make_reflexion_agent(
            inner_agent=inner,
            evaluator=_AcceptEvaluator(),
            episode_store=store,
            emitter=emitter,
        )

        await agent.run("What is 2+2?")

        episodes = await store.recall("What is 2+2?", limit=10)
        assert len(episodes) == 1
        assert episodes[0].episode.evaluator_feedback is None

    async def test_final_failure_episode_carries_verbatim_evaluator_feedback(self) -> None:
        """The final-failure (max-attempts-exhausted) episode also carries verbatim feedback."""
        emitter = make_emitter()
        inner_llm = MockLLMClient(
            responses=[
                make_response("bad 1"),
                make_response("bad 2"),
            ]
        )
        inner = _make_inner_agent(inner_llm, emitter)
        store = InMemoryEpisodeStore(embedding_client=MockEmbeddingClient())
        reflection_llm = MockLLMClient(responses=[make_response("Reflection 1")])

        agent = _make_reflexion_agent(
            inner_agent=inner,
            evaluator=_RejectEvaluator(),
            episode_store=store,
            emitter=emitter,
            max_attempts=2,
            reflection_llm=reflection_llm,
        )

        await agent.run("Hard task")

        episodes = await store.recall("Hard task", limit=10)
        # _RejectEvaluator emits feedback="Not acceptable"
        assert all(e.episode.evaluator_feedback == "Not acceptable" for e in episodes)


# ──────────────────────────────────────────────────────────
# Max attempts exhausted
# ──────────────────────────────────────────────────────────


class TestMaxAttemptsExhausted:
    async def test_returns_evaluation_failed(self) -> None:
        emitter = make_emitter()
        inner_llm = MockLLMClient(
            responses=[
                make_response("bad 1"),
                make_response("bad 2"),
                make_response("bad 3"),
            ]
        )
        inner = _make_inner_agent(inner_llm, emitter)
        store = InMemoryEpisodeStore(embedding_client=MockEmbeddingClient())
        reflection_llm = MockLLMClient(
            responses=[
                make_response("Reflection 1"),
                make_response("Reflection 2"),
            ]
        )

        agent = _make_reflexion_agent(
            inner_agent=inner,
            evaluator=_RejectEvaluator(),
            episode_store=store,
            emitter=emitter,
            max_attempts=3,
            reflection_llm=reflection_llm,
        )

        result = await agent.run("Impossible task")

        assert result.output == "bad 3"
        assert result.termination_reason == "evaluation_failed"
        assert result.total_steps == 3

    async def test_episodes_stored_for_each_attempt(self) -> None:
        emitter = make_emitter()
        inner_llm = MockLLMClient(
            responses=[
                make_response("bad 1"),
                make_response("bad 2"),
                make_response("bad 3"),
            ]
        )
        inner = _make_inner_agent(inner_llm, emitter)
        store = InMemoryEpisodeStore(embedding_client=MockEmbeddingClient())
        reflection_llm = MockLLMClient(
            responses=[
                make_response("Reflection 1"),
                make_response("Reflection 2"),
            ]
        )

        agent = _make_reflexion_agent(
            inner_agent=inner,
            evaluator=_RejectEvaluator(),
            episode_store=store,
            emitter=emitter,
            max_attempts=3,
            reflection_llm=reflection_llm,
        )

        await agent.run("Impossible task")

        assert await store.count() == 3

    async def test_each_failed_attempt_episode_has_failure_outcome(self) -> None:
        emitter = make_emitter()
        inner_llm = MockLLMClient(
            responses=[
                make_response("bad 1"),
                make_response("bad 2"),
                make_response("bad 3"),
            ]
        )
        inner = _make_inner_agent(inner_llm, emitter)
        store = InMemoryEpisodeStore(embedding_client=MockEmbeddingClient())
        reflection_llm = MockLLMClient(
            responses=[
                make_response("Reflection 1"),
                make_response("Reflection 2"),
            ]
        )

        agent = _make_reflexion_agent(
            inner_agent=inner,
            evaluator=_RejectEvaluator(),
            episode_store=store,
            emitter=emitter,
            max_attempts=3,
            reflection_llm=reflection_llm,
        )

        await agent.run("Impossible task")

        episodes = await store.recall("Impossible task", limit=10)
        assert len(episodes) == 3
        assert all(e.episode.outcome == OutcomeType.FAILURE for e in episodes)

    async def test_final_attempt_has_no_reflection(self) -> None:
        emitter = make_emitter()
        inner_llm = MockLLMClient(
            responses=[
                make_response("bad 1"),
                make_response("bad 2"),
            ]
        )
        inner = _make_inner_agent(inner_llm, emitter)
        store = InMemoryEpisodeStore(embedding_client=MockEmbeddingClient())
        reflection_llm = MockLLMClient(responses=[make_response("Reflection 1")])

        agent = _make_reflexion_agent(
            inner_agent=inner,
            evaluator=_RejectEvaluator(),
            episode_store=store,
            emitter=emitter,
            max_attempts=2,
            reflection_llm=reflection_llm,
        )

        await agent.run("Hard task")

        reflection_events = [e for e in emitter.events if isinstance(e, ReflectionGeneratedEvent)]
        # Only 1 reflection: for attempt 1. Final attempt (2) has no reflection.
        assert len(reflection_events) == 1
        assert reflection_events[0].attempt_number == 1


# ──────────────────────────────────────────────────────────
# Reflection quality
# ──────────────────────────────────────────────────────────


class TestReflectionQuality:
    async def test_reflection_receives_correct_context(self) -> None:
        emitter = make_emitter()
        inner_llm = MockLLMClient(
            responses=[
                make_response("bad answer"),
                make_response("good answer"),
            ]
        )
        inner = _make_inner_agent(inner_llm, emitter)
        store = InMemoryEpisodeStore(embedding_client=MockEmbeddingClient())
        reflection_llm = MockLLMClient(responses=[make_response("Better approach needed")])

        agent = _make_reflexion_agent(
            inner_agent=inner,
            evaluator=_RejectThenAcceptEvaluator(),
            episode_store=store,
            emitter=emitter,
            reflection_llm=reflection_llm,
        )

        await agent.run("Solve the puzzle")

        # Verify the reflection LLM received the right context
        assert len(reflection_llm.calls) == 1
        call = reflection_llm.calls[0]
        user_msg = call["messages"][0].content
        assert "Solve the puzzle" in user_msg
        assert "bad answer" in user_msg
        assert "Please improve" in user_msg


# ──────────────────────────────────────────────────────────
# Cancellation
# ──────────────────────────────────────────────────────────


class TestCancellation:
    async def test_cancellation_stops_early(self) -> None:
        emitter = make_emitter()
        token = CancellationToken()
        inner_llm = MockLLMClient(
            responses=[
                make_response("attempt 1"),
                make_response("attempt 2"),
                make_response("attempt 3"),
            ]
        )
        inner = _make_inner_agent(inner_llm, emitter, cancellation_token=token)
        store = InMemoryEpisodeStore(embedding_client=MockEmbeddingClient())
        reflection_llm = MockLLMClient(
            responses=[
                make_response("Reflection 1"),
                make_response("Reflection 2"),
            ]
        )

        # Cancel after first attempt evaluation
        call_count = 0
        original_evaluate = _RejectEvaluator.evaluate

        class _CancellingEvaluator(_RejectEvaluator):
            async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
                nonlocal call_count
                call_count += 1
                result = await original_evaluate(self, output, context)
                if call_count >= 1:
                    token.cancel()
                return result

        agent = _make_reflexion_agent(
            inner_agent=inner,
            evaluator=_CancellingEvaluator(),
            episode_store=store,
            emitter=emitter,
            max_attempts=3,
            cancellation_token=token,
            reflection_llm=reflection_llm,
        )

        result = await agent.run("Task")

        # Should have stopped before reaching max_attempts=3
        assert result.total_steps < 3


# ──────────────────────────────────────────────────────────
# Usage aggregation
# ──────────────────────────────────────────────────────────


class TestUsageAggregation:
    async def test_usages_aggregated_across_attempts_and_reflections(self) -> None:
        emitter = make_emitter()
        inner_usage = make_usage(input_tokens=100, output_tokens=50)
        reflection_usage = make_usage(input_tokens=20, output_tokens=10)

        inner_llm = MockLLMClient(
            responses=[
                make_response("bad", usage=inner_usage),
                make_response("good", usage=inner_usage),
            ]
        )
        inner = _make_inner_agent(inner_llm, emitter)
        store = InMemoryEpisodeStore(embedding_client=MockEmbeddingClient())
        reflection_llm = MockLLMClient(responses=[make_response("Reflect", usage=reflection_usage)])

        agent = _make_reflexion_agent(
            inner_agent=inner,
            evaluator=_RejectThenAcceptEvaluator(),
            episode_store=store,
            emitter=emitter,
            reflection_llm=reflection_llm,
        )

        result = await agent.run("Task")

        # 2 inner agent calls (100+100=200 input) + 1 reflection (20 input)
        assert result.usage.input_tokens == 220
        assert result.usage.output_tokens == 110


# ──────────────────────────────────────────────────────────
# Events
# ──────────────────────────────────────────────────────────


class TestEvents:
    async def test_evaluation_events_emitted_per_attempt(self) -> None:
        emitter = make_emitter()
        inner_llm = MockLLMClient(
            responses=[
                make_response("bad"),
                make_response("good"),
            ]
        )
        inner = _make_inner_agent(inner_llm, emitter)
        store = InMemoryEpisodeStore(embedding_client=MockEmbeddingClient())
        reflection_llm = MockLLMClient(responses=[make_response("Reflect")])

        agent = _make_reflexion_agent(
            inner_agent=inner,
            evaluator=_RejectThenAcceptEvaluator(),
            episode_store=store,
            emitter=emitter,
            reflection_llm=reflection_llm,
        )

        await agent.run("Task")

        eval_events = [e for e in emitter.events if isinstance(e, EvaluationEvent)]
        assert len(eval_events) == 2
        assert eval_events[0].verdict == "reject"
        assert eval_events[1].verdict == "accept"

    async def test_agent_lifecycle_events(self) -> None:
        emitter = make_emitter()
        inner_llm = MockLLMClient(responses=[make_response("answer")])
        inner = _make_inner_agent(inner_llm, emitter)
        store = InMemoryEpisodeStore(embedding_client=MockEmbeddingClient())

        agent = _make_reflexion_agent(
            inner_agent=inner,
            evaluator=_AcceptEvaluator(),
            episode_store=store,
            emitter=emitter,
        )

        await agent.run("Task")

        # ReflexionAgent emits its own start/complete events
        start_events = [e for e in emitter.events if isinstance(e, AgentStartEvent)]
        complete_events = [e for e in emitter.events if isinstance(e, AgentCompleteEvent)]

        # Inner agent also emits start/complete, so we should have at least 2 of each
        reflexion_starts = [e for e in start_events if e.agent_name == "reflexion-agent"]
        reflexion_completes = [e for e in complete_events if e.agent_name == "reflexion-agent"]
        assert len(reflexion_starts) == 1
        assert len(reflexion_completes) == 1


# ──────────────────────────────────────────────────────────
# Different inner agent types
# ──────────────────────────────────────────────────────────


class TestDifferentInnerAgents:
    async def test_with_reasoning_agent(self) -> None:
        emitter = make_emitter()
        inner_llm = MockLLMClient(responses=[make_response("reasoning answer")])
        inner = ReasoningAgent(
            name="reasoning-inner",
            llm_client=inner_llm,
            emitter=emitter,
            system_prompt="Reason carefully.",
        )
        store = InMemoryEpisodeStore(embedding_client=MockEmbeddingClient())

        agent = _make_reflexion_agent(
            inner_agent=inner,
            evaluator=_AcceptEvaluator(),
            episode_store=store,
            emitter=emitter,
        )

        result = await agent.run("What is 2+2?")

        assert result.output == "reasoning answer"
        assert result.termination_reason == "complete"


# ──────────────────────────────────────────────────────────
# Parsed propagation
# ──────────────────────────────────────────────────────────


class TestParsedPropagation:
    async def test_parsed_propagated_from_inner_agent(self) -> None:
        """ReflexionAgent should carry through the inner agent's parsed output."""
        from pydantic import BaseModel

        class Answer(BaseModel):
            value: int

        emitter = make_emitter()
        inner_llm = MockLLMClient(responses=[make_response('{"value": 42}')])
        inner = ReasoningAgent(
            name="inner",
            llm_client=inner_llm,
            emitter=emitter,
            system_prompt="Answer.",
            output_schema=Answer,
        )
        store = InMemoryEpisodeStore(embedding_client=MockEmbeddingClient())

        agent = _make_reflexion_agent(
            inner_agent=inner,
            evaluator=_AcceptEvaluator(),
            episode_store=store,
            emitter=emitter,
        )

        result = await agent.run("What is the answer?")

        assert result.parsed is not None
        assert isinstance(result.parsed, Answer)
        assert result.parsed.value == 42


# ──────────────────────────────────────────────────────────
# Evaluator Error
# ──────────────────────────────────────────────────────────


class TestEvaluatorError:
    async def test_evaluator_error_returns_evaluation_skipped(self) -> None:
        """EVALUATOR_ERROR verdict → evaluation_skipped with episode stored."""

        class _EvaluatorErrorEvaluator:
            @property
            def max_revisions(self) -> int:
                return 2

            async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
                return EvaluationResult(
                    verdict=EvaluationVerdict.EVALUATOR_ERROR,
                    feedback="evaluator crashed",
                    evaluator_name="test-evaluator",
                )

        emitter = make_emitter()
        inner_llm = MockLLMClient(responses=[make_response("some answer")])
        inner = _make_inner_agent(inner_llm, emitter)
        store = InMemoryEpisodeStore(embedding_client=MockEmbeddingClient())

        agent = _make_reflexion_agent(
            inner_agent=inner,
            evaluator=_EvaluatorErrorEvaluator(),
            episode_store=store,
            emitter=emitter,
        )

        result = await agent.run("Task")

        assert result.output == "some answer"
        assert result.termination_reason == "evaluation_skipped"
        assert result.total_steps == 1
        assert await store.count() == 1

    async def test_evaluator_error_episode_has_partial_outcome(self) -> None:
        """EVALUATOR_ERROR verdict → episode stored with PARTIAL outcome."""

        class _EvaluatorErrorEvaluator:
            @property
            def max_revisions(self) -> int:
                return 2

            async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
                return EvaluationResult(
                    verdict=EvaluationVerdict.EVALUATOR_ERROR,
                    feedback="evaluator crashed",
                    evaluator_name="test-evaluator",
                )

        emitter = make_emitter()
        inner_llm = MockLLMClient(responses=[make_response("some answer")])
        inner = _make_inner_agent(inner_llm, emitter)
        store = InMemoryEpisodeStore(embedding_client=MockEmbeddingClient())

        agent = _make_reflexion_agent(
            inner_agent=inner,
            evaluator=_EvaluatorErrorEvaluator(),
            episode_store=store,
            emitter=emitter,
        )

        await agent.run("Task")

        episodes = await store.recall("Task", limit=10)
        assert len(episodes) == 1
        assert episodes[0].episode.outcome == OutcomeType.PARTIAL


# ──────────────────────────────────────────────────────────
# Reflection with Tool Names
# ──────────────────────────────────────────────────────────


class TestReflectionToolNames:
    async def test_reflection_includes_tool_names_from_inner_agent(self) -> None:
        """When inner agent uses tools, reflection prompt includes tool names."""
        from nanitics import LLMResponse, ReActAgent, ToolCall, Usage, tool

        @tool(name="calculator", description="Calculate math")
        async def calc_tool(expression: str) -> str:
            return "42"

        # ReAct agent: tool call response → tool result → final answer
        tc = ToolCall(id="tc1", name="calculator", arguments={"expression": "6*7"})
        inner_responses = [
            LLMResponse(
                content="Let me calculate",
                tool_calls=[tc],
                usage=Usage(input_tokens=10, output_tokens=10),
                model="test",
                stop_reason="tool_use",
            ),
            LLMResponse(
                content="The answer is 42",
                tool_calls=[],
                usage=Usage(input_tokens=10, output_tokens=10),
                model="test",
                stop_reason="end_turn",
            ),
        ]
        inner_llm = MockLLMClient(responses=inner_responses)

        emitter = make_emitter()
        inner = ReActAgent(
            name="inner-react",
            llm_client=inner_llm,
            emitter=emitter,
            system_prompt="You are an assistant.",
            tools=[calc_tool],
        )
        store = InMemoryEpisodeStore(embedding_client=MockEmbeddingClient())

        # Rejection then acceptance — reflection LLM captures tool names
        reflection_llm = MockLLMClient(responses=[make_response("Use a different approach")])

        agent = ReflexionAgent(
            name="reflexion-agent",
            llm_client=reflection_llm,
            emitter=emitter,
            system_prompt="You are reflective.",
            inner_agent=inner,
            evaluator=_RejectThenAcceptEvaluator(),
            episode_store=store,
            max_attempts=2,
        )

        # Need responses for second attempt too
        inner_llm._responses.append(
            LLMResponse(
                content="Better answer: 42",
                tool_calls=[],
                usage=Usage(input_tokens=10, output_tokens=10),
                model="test",
                stop_reason="end_turn",
            )
        )

        await agent.run("What is 6*7?")

        # Verify reflection LLM received tool names
        assert len(reflection_llm.calls) == 1
        reflection_input = reflection_llm.calls[0]["messages"][0].content
        assert "calculator" in reflection_input


# ──────────────────────────────────────────────────────────
# Inner emitter binding
# ──────────────────────────────────────────────────────────


class TestInnerEmitterBinding:
    """Verifies that ReflexionAgent rebinds the inner agent's emitter to the
    outer Reflexion emitter on every attempt, so inner-agent events share the
    outer ``trace_id`` and chain via ``parent_span_id`` through the
    ``attempt-<N>`` span.
    """

    async def test_inner_llm_events_reach_outer_emitter(self) -> None:
        outer_emitter = make_emitter()
        inner_constructor_emitter = InMemoryEmitter(trace_id="inner-unused-bound-by-reflexion")

        inner_llm = MockLLMClient(
            responses=[
                make_response("bad answer"),
                make_response("good answer"),
            ]
        )
        inner = ReasoningAgent(
            name="inner-agent",
            llm_client=inner_llm,
            emitter=inner_constructor_emitter,
            system_prompt="You are an assistant.",
        )
        store = InMemoryEpisodeStore(embedding_client=MockEmbeddingClient())
        reflection_llm = MockLLMClient(responses=[make_response("Try a different approach")])

        agent = _make_reflexion_agent(
            inner_agent=inner,
            evaluator=_RejectThenAcceptEvaluator(),
            episode_store=store,
            emitter=outer_emitter,
            reflection_llm=reflection_llm,
        )

        await agent.run("What is 2+2?")

        # 1. Inner LLM request events reach the outer emitter.
        inner_requests = [e for e in outer_emitter.events if isinstance(e, LLMRequestEvent) and e.label != "reflection"]
        assert len(inner_requests) >= 1, "inner-agent LLM request events must appear in the outer emitter"

        # 1a. An attempt-1 span exists in the outer trace, and inner LLM requests
        # are nested (non-null parent_span_id). The child emitter's root span is
        # synthetic (not emitted as SpanStartEvent), so we assert structurally:
        # the outer trace contains the attempt-1 span, and inner events land
        # under some span rather than at the root.
        from nanitics.infrastructure.observability.events import SpanStartEvent

        attempt_spans = [e for e in outer_emitter.events if isinstance(e, SpanStartEvent) and e.name == "attempt-1"]
        assert len(attempt_spans) == 1, "outer trace must contain exactly one attempt-1 span"
        assert all(e.parent_span_id is not None for e in inner_requests), (
            "inner LLM requests must be nested under some span, not at the root"
        )

        # 2. All outer-emitter events share a single trace_id.
        trace_ids = {e.trace_id for e in outer_emitter.events}
        assert trace_ids == {outer_emitter.trace_id}, (
            f"expected single trace_id {outer_emitter.trace_id!r}, got {trace_ids!r}"
        )

        # 3. The inner agent's constructor emitter received no events — the inner
        # agent emitted only to the child emitter installed by ReflexionAgent.bind().
        assert inner_constructor_emitter.events == [], (
            "inner agent's constructor-time emitter must be unused after bind()"
        )
