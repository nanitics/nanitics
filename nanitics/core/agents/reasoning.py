from __future__ import annotations

from pydantic import BaseModel

from nanitics.core.agents.context import ContextManagement, ContextProvider
from nanitics.core.agents.evaluation import (
    EvaluationResult,
    EvaluationVerdict,
    OutputEvaluator,
)
from nanitics.core.prompts.builder import SystemPromptContributor
from nanitics.infrastructure.llm.protocol import LLMClient, Message
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.safety.cancellation import CancellationToken

from .base import Agent, AgentInput, AgentResult
from .errors import ErrorHandling


class ReasoningAgent(Agent):
    """Agent that produces a response in a single LLM call.

    Unlike tool-using agents, ``ReasoningAgent`` does not enter a
    reasoning loop. It sends the task to the LLM once and returns the
    response. When ``output_schema`` is provided, the LLM is constrained
    to produce structured JSON matching the given Pydantic model.

    If an ``output_evaluator`` is attached, the agent will re-prompt the
    LLM for revisions on ``REVISE`` verdicts, making it multi-step
    despite being fundamentally a single-call agent.

    Args:
        name: Identifies the agent in events and traces.
        llm_client: Language model to use.
        emitter: Event emitter for observability.
        system_prompt: Base system prompt text.
        output_schema: Pydantic model for structured output. When
            provided, the LLM is constrained to produce JSON matching
            this schema.
        cancellation_token: External cancellation signal.
        error_handler: Error recovery strategy.
        context_manager: Context window management.
        context_providers: Inject context before each LLM call.
        output_evaluator: Quality gate for the output.
        prompt_contributors: Additional system prompt sections.
        streaming: Enable token-level streaming via ``LLMTokenEvent``.
    """

    def __init__(
        self,
        *,
        name: str,
        llm_client: LLMClient,
        emitter: EventEmitter,
        system_prompt: str,
        output_schema: type[BaseModel] | None = None,
        cancellation_token: CancellationToken | None = None,
        error_handler: ErrorHandling | None = None,
        context_manager: ContextManagement | None = None,
        context_providers: list[ContextProvider] | None = None,
        output_evaluator: OutputEvaluator | None = None,
        prompt_contributors: list[SystemPromptContributor] | None = None,
        streaming: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            llm_client=llm_client,
            emitter=emitter,
            system_prompt=system_prompt,
            cancellation_token=cancellation_token,
            error_handler=error_handler,
            context_manager=context_manager,
            context_providers=context_providers,
            output_evaluator=output_evaluator,
            prompt_contributors=prompt_contributors,
            streaming=streaming,
        )
        self._output_schema = output_schema

    def _agent_type(self) -> str:
        return "reasoning"

    async def _execute(self, input: AgentInput) -> AgentResult:
        messages = [Message(role="user", content=input)]
        usages = []

        response = await self._call_llm(messages, output_schema=self._output_schema)
        usages.append(response.usage)

        output = response.content
        messages.append(Message(role="assistant", content=output))
        step_count = 1
        termination_reason = "complete"

        self._emit_step(
            step_count,
            thought=response.reasoning_text,
            artifact=response.parsed.model_dump() if response.parsed else None,
        )

        if self._output_evaluator is not None:
            revision_count = 0
            max_revisions = self._output_evaluator.max_revisions

            if self._is_truncated(response):
                eval_result = EvaluationResult(
                    verdict=EvaluationVerdict.REVISE,
                    score=None,
                    feedback=self._TRUNCATION_FEEDBACK,
                    evaluator_name="truncation",
                )
                self._emit_truncation_events(revision_count, max_revisions)
            else:
                eval_result = await self._evaluate_output(
                    output or "",
                    input,
                    messages,
                    revision_count,
                )

            while eval_result.verdict == EvaluationVerdict.REVISE and revision_count < max_revisions:
                messages.append(Message(role="user", content=eval_result.feedback or ""))
                if eval_result.evaluator_name != "truncation":
                    self._emit_evaluation_revision(
                        eval_result.feedback or "",
                        revision_count,
                        max_revisions,
                    )
                revision_count += 1

                response = await self._call_llm(messages, output_schema=self._output_schema)
                usages.append(response.usage)
                output = response.content
                messages.append(Message(role="assistant", content=output))
                step_count += 1
                self._emit_step(
                    step_count,
                    thought=response.reasoning_text,
                    artifact=response.parsed.model_dump() if response.parsed else None,
                )

                if self._is_truncated(response):
                    eval_result = EvaluationResult(
                        verdict=EvaluationVerdict.REVISE,
                        score=None,
                        feedback=self._TRUNCATION_FEEDBACK,
                        evaluator_name="truncation",
                    )
                    self._emit_truncation_events(revision_count, max_revisions)
                else:
                    eval_result = await self._evaluate_output(
                        output or "",
                        input,
                        messages,
                        revision_count,
                    )

            if eval_result.verdict == EvaluationVerdict.EVALUATOR_ERROR:
                termination_reason = "evaluation_skipped"
            elif eval_result.verdict != EvaluationVerdict.ACCEPT:
                termination_reason = "evaluation_failed"
                self._emit_evaluation_exhausted(
                    evaluator_name=eval_result.evaluator_name,
                    verdict=eval_result.verdict.value,
                    revision_count=revision_count,
                    max_revisions=max_revisions,
                    feedback=eval_result.feedback,
                )

        return AgentResult(
            output=output,
            parsed=response.parsed,
            total_steps=step_count,
            termination_reason=termination_reason,
            messages=messages,
            usage=self._aggregate_usage(usages),
        )
