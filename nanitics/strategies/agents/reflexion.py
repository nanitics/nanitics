from __future__ import annotations

from typing import TYPE_CHECKING

from nanitics.infrastructure.llm.instrumented import InstrumentedLLMClient
from nanitics.infrastructure.llm.protocol import LLMClient, Message
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import (
    EvaluationEvent,
    ReflectionGeneratedEvent,
    ToolInfo,
    Usage,
)
from nanitics.safety.cancellation import CancellationToken
from nanitics.strategies.agents.evaluation import (
    EvaluationContext,
    EvaluationVerdict,
    OutputEvaluator,
)

from .base import Agent, AgentInput, AgentResult, _input_to_text

if TYPE_CHECKING:
    from nanitics.capabilities.memory.episodic import EpisodeStore

_REFLECTION_SYSTEM_PROMPT = (
    "Analyze a failed attempt at a task. Produce a reflection that will help on the next attempt.\n\n"
    "Focus on:\n"
    "- What specific approach was taken\n"
    "- Why it failed (based on the evaluation feedback)\n"
    "- What concrete alternative to try next\n"
    "- What assumptions were wrong\n\n"
    "Be specific and actionable. Avoid generic advice."
)


class ReflexionAgent(Agent):
    """Agent that wraps an inner agent with a retry-and-reflect loop.

    After each attempt, the evaluator checks the output. On failure, the
    agent generates a reflection (analysis of what went wrong), stores it
    as an episode in the ``episode_store``, and retries. This enables
    cross-run learning: future runs can recall past episodes to avoid
    repeating mistakes.

    The inner agent is rebound to this agent's emitter at the start of each
    attempt via :meth:`Agent.bind`, so inner-agent events share the outer
    ``trace_id`` and nest under the ``attempt-<N>`` span. The emitter passed
    to the inner agent's constructor is therefore transient — it is
    overwritten by ``bind()`` before each attempt runs. Inner-agent owned
    resources (tool registry, context providers, output evaluator) are
    rebound with it.

    Args:
        name: Identifies the agent in events and traces.
        llm_client: Language model used for generating reflections.
        emitter: Event emitter for observability.
        system_prompt: Base system prompt text.
        inner_agent: The agent that performs the actual task. Can be any
            agent type.
        evaluator: Judges whether the inner agent's output is acceptable.
        episode_store: Stores episodes (successes and failures with
            reflections) for cross-run learning.
        max_attempts: Maximum retry attempts before returning the last
            result with ``termination_reason="evaluation_failed"``.
        cancellation_token: External cancellation signal.
    """

    def __init__(
        self,
        *,
        name: str,
        llm_client: LLMClient,
        emitter: EventEmitter,
        system_prompt: str,
        inner_agent: Agent,
        evaluator: OutputEvaluator,
        episode_store: EpisodeStore,
        max_attempts: int = 3,
        cancellation_token: CancellationToken | None = None,
    ) -> None:
        super().__init__(
            name=name,
            llm_client=llm_client,
            emitter=emitter,
            system_prompt=system_prompt,
            cancellation_token=cancellation_token,
        )
        self._inner_agent = inner_agent
        self._evaluator = evaluator
        self._episode_store = episode_store
        self._max_attempts = max_attempts

    def _agent_type(self) -> str:
        return "reflexion"

    def _active_capabilities(self) -> list[str]:
        caps = super()._active_capabilities()
        caps.append("episodic_memory")
        return caps

    def _get_tools_available(self) -> list[str]:
        return []

    def _get_tool_schemas(self) -> list[ToolInfo]:
        return []

    async def _execute(self, input: AgentInput, *, thread_key: str | None = None) -> AgentResult:
        from nanitics.capabilities.memory.episodic import OutcomeType, extract_episode

        usages: list[Usage] = []
        result: AgentResult | None = None
        attempts_made = 0

        for attempt in range(1, self._max_attempts + 1):
            if self._is_cancelled:
                self._emit_safety_cancellation(step_number=attempts_made if attempts_made > 0 else None)
                break

            attempts_made = attempt

            with self._emitter.span(f"attempt-{attempt}"):
                result = await self._inner_agent.bind(self._emitter).run(input)
                usages.append(result.usage)

                eval_context = EvaluationContext(
                    messages=result.messages,
                    task_input=input,
                )
                eval_result = await self._evaluator.evaluate(result.output or "", eval_context)
                self._emitter.emit(
                    EvaluationEvent(
                        trace_id=self._emitter.trace_id,
                        span_id=self._emitter.span_id,
                        parent_span_id=self._emitter.parent_span_id,
                        evaluator_name=eval_result.evaluator_name,
                        verdict=eval_result.verdict.value,
                        score=eval_result.score,
                        feedback=eval_result.feedback,
                        revision_attempt=attempt,
                    )
                )

                if eval_result.verdict == EvaluationVerdict.ACCEPT:
                    episode = extract_episode(
                        task_input=input,
                        result=result,
                        metadata={"attempt": attempt, "agent": self._inner_agent.name},
                    )
                    await self._episode_store.record(episode)
                    return AgentResult(
                        output=result.output,
                        parsed=result.parsed,
                        total_steps=attempt,
                        termination_reason="complete",
                        messages=result.messages,
                        usage=self._aggregate_usage(usages),
                    )

                if eval_result.verdict == EvaluationVerdict.EVALUATOR_ERROR:
                    episode = extract_episode(
                        task_input=input,
                        result=result,
                        outcome=OutcomeType.PARTIAL,
                        metadata={"attempt": attempt, "agent": self._inner_agent.name},
                    )
                    await self._episode_store.record(episode)
                    return AgentResult(
                        output=result.output,
                        parsed=result.parsed,
                        total_steps=attempt,
                        termination_reason="evaluation_skipped",
                        messages=result.messages,
                        usage=self._aggregate_usage(usages),
                    )

                if attempt < self._max_attempts:
                    reflection, reflection_usage = await self._generate_reflection(input, result, eval_result.feedback)
                    usages.append(reflection_usage)
                    episode = extract_episode(
                        task_input=input,
                        result=result,
                        outcome=OutcomeType.FAILURE,
                        reflection=reflection,
                        evaluator_feedback=eval_result.feedback,
                        metadata={"attempt": attempt, "agent": self._inner_agent.name},
                    )
                    episode_id = await self._episode_store.record(episode)
                    self._emitter.emit(
                        ReflectionGeneratedEvent(
                            trace_id=self._emitter.trace_id,
                            span_id=self._emitter.span_id,
                            parent_span_id=self._emitter.parent_span_id,
                            attempt_number=attempt,
                            max_attempts=self._max_attempts,
                            reflection_text=reflection,
                            evaluation_feedback=eval_result.feedback,
                            episode_id=episode_id,
                        )
                    )
                else:
                    episode = extract_episode(
                        task_input=input,
                        result=result,
                        outcome=OutcomeType.FAILURE,
                        evaluator_feedback=eval_result.feedback,
                        metadata={"attempt": attempt, "agent": self._inner_agent.name},
                    )
                    await self._episode_store.record(episode)

        assert result is not None
        return AgentResult(
            output=result.output,
            parsed=result.parsed,
            total_steps=attempts_made,
            termination_reason="evaluation_failed",
            messages=result.messages,
            usage=self._aggregate_usage(usages),
        )

    async def _generate_reflection(
        self, input: AgentInput, result: AgentResult, feedback: str | None
    ) -> tuple[str, Usage]:
        tool_names: list[str] = []
        for msg in result.messages:
            if msg.role == "assistant" and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.name not in tool_names:
                        tool_names.append(tc.name)

        tools_str = ", ".join(tool_names) if tool_names else "None"

        task_text = _input_to_text(input)
        user_content = (
            f"Task: {task_text}\n\n"
            f"Attempt result: {result.output or 'No output produced'}\n\n"
            f"Evaluation feedback: {feedback or 'No feedback provided'}\n\n"
            f"Termination reason: {result.termination_reason}\n\n"
            f"Tools used: {tools_str}\n\n"
            f"What went wrong and what should be tried differently?"
        )

        # InstrumentedLLMClient is request-scoped — rebuilt with the current
        # emitter each call rather than held across bind() so concurrent
        # binds do not share an emitter.
        instrumented = InstrumentedLLMClient(self._llm_client, emitter=self._emitter, label="reflection")
        response = await instrumented.generate(
            system_prompt=_REFLECTION_SYSTEM_PROMPT,
            messages=[Message(role="user", content=user_content)],
        )

        return response.content or "", response.usage
