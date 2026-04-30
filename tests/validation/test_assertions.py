"""Unit tests for `validation.helpers.assertions`.

Covers trace matching with and without predicates plus the LLM-as-judge
helper against scripted mock judges.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from nanitics.infrastructure.llm.mock import MockLLMClient
from nanitics.infrastructure.llm.protocol import LLMResponse
from nanitics.infrastructure.observability.emitter import InMemoryEmitter
from nanitics.infrastructure.observability.events import AgentStepEvent, ToolInvokeEvent, Usage
from validation.helpers.assertions import assert_result_satisfies, assert_trace_contains


def _step_event(emitter: InMemoryEmitter, step: int, thought: str = "t") -> AgentStepEvent:
    event = AgentStepEvent(
        trace_id=emitter.trace_id,
        span_id=emitter.span_id,
        agent_name="a",
        step_number=step,
        thought=thought,
    )
    emitter.emit(event)
    return event


def _judge_response(verdict: dict[str, object]) -> LLMResponse:
    return LLMResponse(
        content=json.dumps(verdict),
        usage=Usage(input_tokens=1, output_tokens=1),
        model="mock",
        stop_reason="end_turn",
    )


def test_assert_trace_contains_returns_first_match() -> None:
    emitter = InMemoryEmitter(trace_id="t")
    first = _step_event(emitter, 1, thought="first")
    _step_event(emitter, 2, thought="second")
    got = assert_trace_contains(emitter, AgentStepEvent)
    assert got is first


def test_assert_trace_contains_applies_predicate() -> None:
    emitter = InMemoryEmitter(trace_id="t")
    _step_event(emitter, 1, thought="first")
    target = _step_event(emitter, 2, thought="second")
    got = assert_trace_contains(emitter, AgentStepEvent, lambda e: e.step_number == 2)
    assert got is target


def test_assert_trace_contains_raises_when_type_missing() -> None:
    emitter = InMemoryEmitter(trace_id="t")
    _step_event(emitter, 1)
    with pytest.raises(AssertionError, match="ToolInvokeEvent"):
        assert_trace_contains(emitter, ToolInvokeEvent)


def test_assert_trace_contains_raises_with_empty_summary() -> None:
    emitter = InMemoryEmitter(trace_id="t")
    with pytest.raises(AssertionError, match=r"\(none\)"):
        assert_trace_contains(emitter, AgentStepEvent)


def test_assert_trace_contains_raises_when_predicate_rejects_all() -> None:
    emitter = InMemoryEmitter(trace_id="t")
    _step_event(emitter, 1)
    with pytest.raises(AssertionError, match="none satisfied the predicate"):
        assert_trace_contains(emitter, AgentStepEvent, lambda e: e.step_number == 99)


async def test_assert_result_satisfies_passes() -> None:
    judge = MockLLMClient([_judge_response({"pass": True, "reason": "good"})])
    await assert_result_satisfies("ok", "criterion", judge=judge)


async def test_assert_result_satisfies_fails_with_reason() -> None:
    judge = MockLLMClient([_judge_response({"pass": False, "reason": "missing x"})])
    with pytest.raises(AssertionError, match="Judge failed: missing x"):
        await assert_result_satisfies("bad", "criterion", judge=judge)


async def test_assert_result_satisfies_default_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = MockLLMClient([_judge_response({"pass": True, "reason": "ok"})])

    def fake_make(provider: str, *, model: str | None = None) -> MockLLMClient:
        assert provider == "anthropic"
        return stub

    monkeypatch.setattr("validation.helpers.assertions.make_llm_client", fake_make, raising=False)
    # The import inside `assert_result_satisfies` is deferred; patch where it is looked up.
    import validation.helpers.llm as llm_mod

    monkeypatch.setattr(llm_mod, "make_llm_client", fake_make)
    await assert_result_satisfies("out", "crit")


async def test_assert_result_satisfies_malformed_raises_runtime_error() -> None:
    bad = LLMResponse(
        content="not json",
        usage=Usage(input_tokens=1, output_tokens=1),
        model="mock",
        stop_reason="end_turn",
    )
    # Bypass the structured-output parsing by using a stub that returns content
    # directly without parsed data.

    class RawJudge:
        model = "mock"

        async def generate(self, **kwargs: object) -> LLMResponse:
            return bad

    with pytest.raises(RuntimeError, match="malformed response"):
        await assert_result_satisfies("out", "crit", judge=RawJudge())


async def test_assert_result_satisfies_no_content_raises_runtime_error() -> None:
    empty = LLMResponse(
        content=None,
        usage=Usage(input_tokens=1, output_tokens=1),
        model="mock",
        stop_reason="end_turn",
    )

    class EmptyJudge:
        model = "mock"

        async def generate(self, **kwargs: object) -> LLMResponse:
            return empty

    with pytest.raises(RuntimeError, match="malformed response"):
        await assert_result_satisfies("out", "crit", judge=EmptyJudge())  # type: ignore[arg-type]


async def test_assert_result_satisfies_uses_parsed_field() -> None:
    # When the judge client exposes `parsed` as the structured verdict, the
    # helper consumes it directly without re-parsing content.
    from validation.helpers.assertions import JudgeVerdict

    verdict_obj: BaseModel = JudgeVerdict.model_validate({"pass": True, "reason": "ok"})
    response = LLMResponse(
        content="ignored",
        usage=Usage(input_tokens=1, output_tokens=1),
        model="mock",
        stop_reason="end_turn",
    ).model_copy(update={"parsed": verdict_obj})

    class ParsedJudge:
        model = "mock"

        async def generate(self, **kwargs: object) -> LLMResponse:
            return response

    await assert_result_satisfies("out", "crit", judge=ParsedJudge())
