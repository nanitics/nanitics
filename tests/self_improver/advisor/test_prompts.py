"""Unit tests for :mod:`self_improver.advisor._prompts`.

Covers the purity of the builder, rubric filtering by target dimension,
and deterministic trace rendering.
"""

from __future__ import annotations

from pathlib import Path

from self_improver.advisor._prompts import build_specialist_system_prompt
from self_improver.advisor.proposal import ProposalCategory, ProposalSeverity, RubricSource
from self_improver.advisor.rubric import Rubric

from nanitics.infrastructure.observability.events import (
    AgentStartEvent,
    LLMResponseEvent,
)
from tests.testing_helpers import make_usage


def _rubric(
    rubric_id: str,
    *,
    target_dimension: str = "prompts",
    body: str = "Body content.",
    severity: ProposalSeverity = ProposalSeverity.WARNING,
    category: ProposalCategory = ProposalCategory.PROMPTS,
    source: RubricSource = RubricSource.BUILTIN,
) -> Rubric:
    return Rubric(
        id=rubric_id,
        severity=severity,
        category=category,
        target_dimension=target_dimension,
        body=body,
        source=source,
        path=Path(f"/fake/{rubric_id}.md"),
    )


def _agent_start_event() -> AgentStartEvent:
    return AgentStartEvent(
        trace_id="t1",
        span_id="s1",
        agent_name="demo",
        task_input="Do the thing.",
        tools_available=[],
    )


def _llm_response_event() -> LLMResponseEvent:
    return LLMResponseEvent(
        trace_id="t1",
        span_id="s1",
        model_name="fake-model",
        content="hi",
        usage=make_usage(),
        duration_ms=10.0,
    )


class TestBuildSpecialistSystemPrompt:
    def test_returns_string(self) -> None:
        prompt = build_specialist_system_prompt("prompts", [], [])
        assert isinstance(prompt, str)

    def test_includes_rubrics_only_for_matching_target_dimension(self) -> None:
        in_scope = _rubric(
            "prompts-scope-a",
            target_dimension="prompts",
            body="Scope A body.",
        )
        out_of_scope = _rubric(
            "tooldesc-b",
            target_dimension="tool_descriptions",
            body="Out of scope body.",
            category=ProposalCategory.TOOL_DESCRIPTIONS,
        )
        prompt = build_specialist_system_prompt("prompts", [in_scope, out_of_scope], [])
        assert "prompts-scope-a" in prompt
        assert "Scope A body." in prompt
        assert "tooldesc-b" not in prompt
        assert "Out of scope body." not in prompt

    def test_no_rubrics_still_renders_rubric_section_placeholder(self) -> None:
        prompt = build_specialist_system_prompt("prompts", [], [])
        assert "No rubrics loaded" in prompt

    def test_rubric_block_is_deterministic(self) -> None:
        rubrics = [
            _rubric("prompts-b", body="B body."),
            _rubric("prompts-a", body="A body."),
        ]
        first = build_specialist_system_prompt("prompts", rubrics, [])
        second = build_specialist_system_prompt("prompts", list(reversed(rubrics)), [])
        assert first == second
        a_idx = first.index("prompts-a")
        b_idx = first.index("prompts-b")
        assert a_idx < b_idx

    def test_trace_block_includes_every_event_with_index(self) -> None:
        events = [_agent_start_event(), _llm_response_event()]
        prompt = build_specialist_system_prompt("prompts", [], events)
        assert "[0] agent.start" in prompt
        assert "[1] llm.response" in prompt

    def test_trace_block_is_deterministic(self) -> None:
        events = [_agent_start_event(), _llm_response_event()]
        first = build_specialist_system_prompt("prompts", [], events)
        second = build_specialist_system_prompt("prompts", [], list(events))
        assert first == second

    def test_empty_trace_renders_explicit_placeholder(self) -> None:
        prompt = build_specialist_system_prompt("prompts", [], [])
        assert "trace contains no events" in prompt

    def test_role_mentions_target_dimension(self) -> None:
        prompt = build_specialist_system_prompt("tool_descriptions", [], [])
        assert "tool_descriptions" in prompt

    def test_includes_task_framing(self) -> None:
        prompt = build_specialist_system_prompt("prompts", [], [])
        assert "SpecialistProposals" in prompt
        assert "proposals" in prompt

    def test_role_instructs_not_to_invent_categories(self) -> None:
        prompt = build_specialist_system_prompt("prompts", [], [])
        assert "do not emit categories outside" in prompt


class TestPurity:
    def test_no_side_effects_on_rubric_list(self) -> None:
        rubrics = [_rubric("p-a"), _rubric("p-b")]
        original = list(rubrics)
        build_specialist_system_prompt("prompts", rubrics, [])
        assert rubrics == original

    def test_no_side_effects_on_trace_list(self) -> None:
        events = [_agent_start_event(), _llm_response_event()]
        original = list(events)
        build_specialist_system_prompt("prompts", [], events)
        assert events == original
