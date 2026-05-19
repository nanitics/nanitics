from __future__ import annotations

import time
from asyncio import sleep  # module-local seam — see ``await sleep(wait)`` below
from collections.abc import Callable
from typing import cast

from pydantic import BaseModel, ConfigDict

from nanitics.capabilities.evaluation.protocol import (
    EvaluationContext,
    EvaluationResult,
    EvaluationVerdict,
)
from nanitics.infrastructure.errors import LLMError, LLMRateLimitError
from nanitics.infrastructure.llm.protocol import LLMClient, LLMResponse, Message
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import (
    LLMRequestEvent,
    LLMResponseEvent,
)
from nanitics.strategies.agents.base import AgentInput, _input_to_text


class _EvaluationResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    score: float
    reasoning: str
    issues: list[str]


_EVALUATION_SYSTEM_PROMPT = """You are an evaluator assessing the quality of an AI assistant's output.

You will be given:
1. The original task/question
2. The assistant's output
3. Evaluation criteria

Your job is to identify weaknesses, errors, and areas where the output fails to meet the criteria.
Be critical and thorough — it is better to flag a potential issue than to miss one.

Score the output from 0.0 (completely inadequate) to 1.0 (excellent). Be strict with scoring."""


def _build_evaluation_prompt(
    output: str,
    task_input: AgentInput,
    criteria: str,
    context: EvaluationContext | None = None,
) -> str:
    task_text = _input_to_text(task_input)
    prompt = f"""## Original Task
{task_text}

## Assistant's Output
{output}

## Evaluation Criteria
{criteria}

Evaluate the output against the criteria. Identify specific issues and assign a score."""

    if context is not None and context.depth is not None:
        exploration_section = f"""\n\n## Exploration Context
This output was produced at depth {context.depth} of {context.max_depth} in a tree search."""
        if context.trajectory_length is not None:
            exploration_section += f"\nThe trajectory from root to this node is {context.trajectory_length} steps."
        if context.total_nodes_explored is not None:
            exploration_section += f"\n{context.total_nodes_explored} nodes have been explored so far."
        exploration_section += (
            "\n\nConsider whether the depth of reasoning is appropriate for the complexity of the task."
        )
        prompt += exploration_section

    return prompt


class LLMEvaluator:
    """LLM-based evaluator that assesses output quality against criteria.

    Sends the original task, agent output, and evaluation criteria to an LLM.
    The LLM returns a score (0.0–1.0), reasoning, and specific issues.
    Returns ACCEPT if the score meets the threshold, REVISE otherwise.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        criteria: str,
        score_threshold: float = 0.7,
        reject_threshold: float | None = None,
        max_revisions: int = 1,
        emitter: EventEmitter | None = None,
        *,
        emitter_provider: Callable[[], EventEmitter | None] | None = None,
    ) -> None:
        """Initialize the LLM evaluator.

        Args:
            llm_client: LLM client for evaluation calls. Can be a different
                (cheaper) model than the agent's model.
            criteria: Natural language description of quality criteria.
            score_threshold: Minimum score to accept output. Scores below
                this trigger revision.
            reject_threshold: Score below which output is rejected outright.
                When set, scores below this produce REJECT instead of REVISE.
                Must be less than score_threshold. Default None disables rejection.
            max_revisions: Maximum revision attempts before the agent gives up.
            emitter: Optional event emitter for trace observability. When set,
                emits LLMRequestEvent and LLMResponseEvent around evaluation calls.
        """
        if max_revisions < 0:
            raise ValueError(f"max_revisions must be non-negative, got {max_revisions}")
        if reject_threshold is not None and reject_threshold >= score_threshold:
            raise ValueError(
                f"reject_threshold ({reject_threshold}) must be less than score_threshold ({score_threshold})"
            )
        self._llm_client = llm_client
        self._criteria = criteria
        self._score_threshold = score_threshold
        self._reject_threshold = reject_threshold
        self._max_revisions = max_revisions
        self._static_emitter = emitter
        self._emitter_provider: Callable[[], EventEmitter | None] | None = emitter_provider

    @property
    def _emitter(self) -> EventEmitter | None:
        """Emitter used for trace events.

        Resolves through ``emitter_provider`` when set (so the evaluator
        follows its owning agent's per-task bound emitter); otherwise
        the static emitter passed at construction.
        """
        if self._emitter_provider is not None:
            return self._emitter_provider()
        return self._static_emitter

    @property
    def max_revisions(self) -> int:
        return self._max_revisions

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        prompt = _build_evaluation_prompt(
            output=output,
            task_input=context.task_input,
            criteria=self._criteria,
            context=context,
        )
        messages = [Message(role="user", content=prompt)]

        try:
            response = await self._instrumented_generate(messages)
        except LLMRateLimitError as e:
            wait = min(e.retry_after or 2.0, 30.0)
            # Uses module-local ``sleep`` (patched by ``_patch_retry_sleep`` in
            # tests/conftest.py) — not ``asyncio.sleep``, which would patch the
            # asyncio module globally and break ``await asyncio.sleep(0)`` yields.
            await sleep(wait)
            try:
                response = await self._instrumented_generate(messages)
            except LLMError as retry_err:
                detail = f"{type(retry_err).__name__}: {retry_err.message}"
                return EvaluationResult(
                    verdict=EvaluationVerdict.EVALUATOR_ERROR,
                    score=None,
                    feedback="Evaluation failed: evaluator LLM call failed.",
                    evaluator_name="llm",
                    error_detail=detail,
                )
        except LLMError as e:
            detail = f"{type(e).__name__}: {e.message}"
            return EvaluationResult(
                verdict=EvaluationVerdict.EVALUATOR_ERROR,
                score=None,
                feedback="Evaluation failed: evaluator LLM call failed.",
                evaluator_name="llm",
                error_detail=detail,
            )

        parsed: _EvaluationResponse | None = cast(_EvaluationResponse | None, response.parsed)
        if parsed is None:
            return EvaluationResult(
                verdict=EvaluationVerdict.EVALUATOR_ERROR,
                score=None,
                feedback="Evaluation failed: could not parse evaluator response.",
                evaluator_name="llm",
                error_detail="Response did not match expected schema",
            )

        if parsed.score >= self._score_threshold:
            return EvaluationResult(
                verdict=EvaluationVerdict.ACCEPT,
                score=parsed.score,
                feedback=None,
                evaluator_name="llm",
            )

        feedback_parts = [parsed.reasoning]
        if parsed.issues:
            feedback_parts.append("Issues:")
            feedback_parts.extend(f"- {issue}" for issue in parsed.issues)
        feedback = "\n".join(feedback_parts)

        if self._reject_threshold is not None and parsed.score < self._reject_threshold:
            return EvaluationResult(
                verdict=EvaluationVerdict.REJECT,
                score=parsed.score,
                feedback=feedback,
                evaluator_name="llm",
            )

        return EvaluationResult(
            verdict=EvaluationVerdict.REVISE,
            score=parsed.score,
            feedback=feedback,
            evaluator_name="llm",
        )

    async def _instrumented_generate(self, messages: list[Message]) -> LLMResponse:
        """Call the LLM, emitting request/response events when an emitter is set."""
        emitter = self._emitter
        if emitter:
            emitter.emit(
                LLMRequestEvent(
                    trace_id=emitter.trace_id,
                    span_id=emitter.span_id,
                    parent_span_id=emitter.parent_span_id,
                    model_name=self._llm_client.model or "",
                    system_prompt=_EVALUATION_SYSTEM_PROMPT,
                    messages=[m.model_dump() for m in messages],
                    output_schema=_EvaluationResponse.model_json_schema(),
                    label="evaluator",
                )
            )

        start = time.perf_counter()
        response = await self._llm_client.generate(
            system_prompt=_EVALUATION_SYSTEM_PROMPT,
            messages=messages,
            output_schema=_EvaluationResponse,
        )
        duration_ms = (time.perf_counter() - start) * 1000

        emitter = self._emitter
        if emitter:
            emitter.emit(
                LLMResponseEvent(
                    trace_id=emitter.trace_id,
                    span_id=emitter.span_id,
                    parent_span_id=emitter.parent_span_id,
                    model_name=response.model,
                    content=response.content,
                    usage=response.usage,
                    duration_ms=duration_ms,
                    label="evaluator",
                )
            )

        return response
