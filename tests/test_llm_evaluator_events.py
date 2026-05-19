"""Tests for LLMEvaluator event emission."""

from __future__ import annotations

from nanitics.capabilities.evaluation.llm_evaluator import LLMEvaluator
from nanitics.capabilities.evaluation.protocol import EvaluationContext
from nanitics.infrastructure import MockLLMClient
from nanitics.infrastructure.errors import LLMRateLimitError
from nanitics.infrastructure.llm.protocol import LLMResponse, Message
from nanitics.infrastructure.observability.events import (
    LLMRequestEvent,
    LLMResponseEvent,
    Usage,
)
from nanitics.tracing import InMemoryEmitter

USAGE = Usage(input_tokens=100, output_tokens=50)

EVAL_RESPONSE = LLMResponse(
    content='{"score": 0.9, "reasoning": "Good output.", "issues": []}',
    usage=USAGE,
    model="mock",
    stop_reason="end_turn",
)


def _make_context() -> EvaluationContext:
    return EvaluationContext(
        messages=[Message(role="user", content="test input")],
        task_input="test input",
    )


class TestLLMEvaluatorEvents:
    async def test_emits_request_and_response_events(self) -> None:
        """When emitter is provided, evaluate() emits LLMRequestEvent and LLMResponseEvent."""
        mock_llm = MockLLMClient(responses=[EVAL_RESPONSE])
        emitter = InMemoryEmitter(trace_id="test-trace")

        evaluator = LLMEvaluator(mock_llm, criteria="Be good.", emitter=emitter)
        await evaluator.evaluate("some output", _make_context())

        request_events = [e for e in emitter.events if isinstance(e, LLMRequestEvent)]
        response_events = [e for e in emitter.events if isinstance(e, LLMResponseEvent)]
        assert len(request_events) == 1
        assert len(response_events) == 1
        assert request_events[0].label == "evaluator"
        assert response_events[0].label == "evaluator"
        assert response_events[0].model_name == "mock"

    async def test_no_events_when_emitter_is_none(self) -> None:
        """When emitter is None (default), no events are emitted."""
        mock_llm = MockLLMClient(responses=[EVAL_RESPONSE])

        evaluator = LLMEvaluator(mock_llm, criteria="Be good.")
        result = await evaluator.evaluate("some output", _make_context())

        # No crash, result still valid
        assert result.score == 0.9

    async def test_emits_events_on_rate_limit_retry(self) -> None:
        """On rate limit retry, both the initial and retry LLM calls emit events."""

        call_count = 0

        async def rate_limit_then_succeed(*, output_schema=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise LLMRateLimitError("Rate limited", retry_after=0.01)
            response = EVAL_RESPONSE
            if output_schema is not None and response.content is not None:
                parsed = output_schema.model_validate_json(response.content)
                response = response.model_copy(update={"parsed": parsed})
            return response

        mock_llm = MockLLMClient(responses=[])
        mock_llm.generate = rate_limit_then_succeed  # type: ignore[method-assign]

        emitter = InMemoryEmitter(trace_id="test-trace")
        evaluator = LLMEvaluator(mock_llm, criteria="Be good.", emitter=emitter)
        result = await evaluator.evaluate("some output", _make_context())

        request_events = [e for e in emitter.events if isinstance(e, LLMRequestEvent)]
        response_events = [e for e in emitter.events if isinstance(e, LLMResponseEvent)]
        # First call emits request (then raises), second call emits request + response
        assert len(request_events) == 2
        assert len(response_events) == 1
        assert result.score == 0.9
