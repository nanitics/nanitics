"""Unit tests for :mod:`self_improver.advisor._specialists`.

Covers the structured-output contract (``SpecialistProposals`` round-trip),
the ``build_specialist`` factory (agent name, output schema, rubric
filtering), and ``build_all_specialists`` (launch-dimension partitioning
with silent exclusion for deferred dimensions).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from self_improver.advisor._specialists import (
    SpecialistProposals,
    build_all_specialists,
    build_specialist,
    run_specialist,
)
from self_improver.advisor.proposal import (
    EvidenceReference,
    Proposal,
    ProposalCategory,
    ProposalSeverity,
    RubricSource,
)
from self_improver.advisor.rubric import Rubric

from nanitics.infrastructure.llm.mock import MockLLMClient
from nanitics.infrastructure.llm.protocol import LLMResponse
from nanitics.infrastructure.observability.emitter import InMemoryEmitter
from nanitics.strategies.agents.reasoning import ReasoningAgent
from tests.testing_helpers import make_usage


def _rubric(
    rubric_id: str,
    *,
    target_dimension: str = "prompts",
    category: ProposalCategory = ProposalCategory.PROMPTS,
    severity: ProposalSeverity = ProposalSeverity.WARNING,
    source: RubricSource = RubricSource.BUILTIN,
) -> Rubric:
    return Rubric(
        id=rubric_id,
        severity=severity,
        category=category,
        target_dimension=target_dimension,
        body=f"Body for {rubric_id}.",
        source=source,
        path=Path(f"/fake/{rubric_id}.md"),
    )


def _proposal(
    rubric_id: str,
    *,
    target_dimension: str = "prompts",
    category: ProposalCategory = ProposalCategory.PROMPTS,
    severity: ProposalSeverity = ProposalSeverity.WARNING,
    source: RubricSource = RubricSource.BUILTIN,
) -> Proposal:
    return Proposal(
        rubric_id=rubric_id,
        rubric_source=source,
        severity=severity,
        category=category,
        target_dimension=target_dimension,
        headline="Headline text",
        detail="Detail paragraph explaining the concern.",
        evidence=[
            EvidenceReference(event_index=0, event_type="agent.start", excerpt="excerpt"),
        ],
        suggested_action="Do the thing.",
        ranking_score=0.5,
    )


class TestSpecialistProposalsModel:
    def test_round_trip_with_valid_proposals(self) -> None:
        original = SpecialistProposals(
            proposals=[
                _proposal("prompts-a"),
                _proposal("prompts-b", severity=ProposalSeverity.CRITICAL),
            ]
        )
        payload = original.model_dump_json()
        rebuilt = SpecialistProposals.model_validate_json(payload)
        assert rebuilt == original

    def test_rejects_non_list_proposals(self) -> None:
        with pytest.raises(ValidationError):
            SpecialistProposals.model_validate({"proposals": "not a list"})

    def test_rejects_invalid_proposal_body(self) -> None:
        with pytest.raises(ValidationError):
            SpecialistProposals.model_validate(
                {"proposals": [{"rubric_id": ""}]}  # ranking_score missing, headline missing, etc.
            )

    def test_empty_proposals_list_is_valid(self) -> None:
        sp = SpecialistProposals(proposals=[])
        assert sp.proposals == []

    def test_model_is_frozen(self) -> None:
        sp = SpecialistProposals(proposals=[])
        with pytest.raises(ValidationError):
            sp.proposals = [_proposal("x")]  # type: ignore[misc]


class TestBuildSpecialist:
    def test_agent_has_expected_name(self) -> None:
        emitter = InMemoryEmitter(trace_id="t")
        agent = build_specialist(
            target_dimension="prompts",
            rubrics=[_rubric("prompts-a")],
            trace_events=[],
            llm_client=MockLLMClient([]),
            emitter=emitter,
        )
        assert agent.name == "advisor-prompts"

    def test_agent_is_reasoning_agent_with_output_schema(self) -> None:
        emitter = InMemoryEmitter(trace_id="t")
        agent = build_specialist(
            target_dimension="tool_descriptions",
            rubrics=[],
            trace_events=[],
            llm_client=MockLLMClient([]),
            emitter=emitter,
        )
        assert isinstance(agent, ReasoningAgent)
        assert agent._output_schema is SpecialistProposals

    def test_system_prompt_contains_only_matching_rubrics(self) -> None:
        emitter = InMemoryEmitter(trace_id="t")
        rubrics = [
            _rubric("prompts-match", target_dimension="prompts"),
            _rubric(
                "tooldesc-nope",
                target_dimension="tool_descriptions",
                category=ProposalCategory.TOOL_DESCRIPTIONS,
            ),
        ]
        agent = build_specialist(
            target_dimension="prompts",
            rubrics=rubrics,
            trace_events=[],
            llm_client=MockLLMClient([]),
            emitter=emitter,
        )
        assert "prompts-match" in agent._system_prompt
        assert "Body for prompts-match." in agent._system_prompt
        assert "tooldesc-nope" not in agent._system_prompt

    async def test_end_to_end_specialist_call_produces_valid_proposals(self) -> None:
        emitter = InMemoryEmitter(trace_id="t")
        proposal = _proposal("prompts-ambiguous-scope", target_dimension="prompts")
        canned = SpecialistProposals(proposals=[proposal])
        mock_client = MockLLMClient(
            [
                LLMResponse(
                    content=canned.model_dump_json(),
                    tool_calls=[],
                    usage=make_usage(),
                    model="fake",
                    stop_reason="end_turn",
                    parsed=None,
                )
            ]
        )
        agent = build_specialist(
            target_dimension="prompts",
            rubrics=[_rubric("prompts-ambiguous-scope", target_dimension="prompts")],
            trace_events=[],
            llm_client=mock_client,
            emitter=emitter,
        )
        result = await agent.run("Emit proposals.")
        parsed: SpecialistProposals = result.parsed
        assert isinstance(parsed, SpecialistProposals)
        assert len(parsed.proposals) == 1
        assert parsed.proposals[0].rubric_id == "prompts-ambiguous-scope"
        # The specialist's proposal carries the target dimension matching
        # its specialist scope.
        assert parsed.proposals[0].target_dimension == "prompts"


class TestBuildAllSpecialists:
    def test_produces_three_specialists_for_launch_dimensions(self) -> None:
        emitter = InMemoryEmitter(trace_id="t")
        agents = build_all_specialists(
            rubrics=[],
            trace_events=[],
            llm_client=MockLLMClient([]),
            emitter=emitter,
        )
        assert len(agents) == 3
        names = [a.name for a in agents]
        assert names == [
            "advisor-prompts",
            "advisor-tool_descriptions",
            "advisor-coordination_patterns",
        ]

    def test_rubrics_partitioned_by_target_dimension(self) -> None:
        emitter = InMemoryEmitter(trace_id="t")
        rubrics = [
            _rubric("p-1", target_dimension="prompts"),
            _rubric(
                "t-1",
                target_dimension="tool_descriptions",
                category=ProposalCategory.TOOL_DESCRIPTIONS,
            ),
            _rubric(
                "c-1",
                target_dimension="coordination_patterns",
                category=ProposalCategory.COORDINATION_PATTERNS,
            ),
        ]
        agents = build_all_specialists(
            rubrics=rubrics,
            trace_events=[],
            llm_client=MockLLMClient([]),
            emitter=emitter,
        )
        by_name = {a.name: a for a in agents}
        assert "p-1" in by_name["advisor-prompts"]._system_prompt
        assert "t-1" not in by_name["advisor-prompts"]._system_prompt
        assert "t-1" in by_name["advisor-tool_descriptions"]._system_prompt
        assert "c-1" in by_name["advisor-coordination_patterns"]._system_prompt

    def test_deferred_dimensions_silently_excluded(self) -> None:
        emitter = InMemoryEmitter(trace_id="t")
        rubrics = [
            _rubric(
                "agent-strat",
                target_dimension="agent_strategy",
                category=ProposalCategory.AGENT_STRATEGY,
            ),
            _rubric(
                "iter-budget",
                target_dimension="iteration_budgets",
                category=ProposalCategory.ITERATION_BUDGETS,
            ),
        ]
        # No warnings, no exceptions — silent exclusion.
        agents = build_all_specialists(
            rubrics=rubrics,
            trace_events=[],
            llm_client=MockLLMClient([]),
            emitter=emitter,
        )
        assert len(agents) == 3
        # No specialist's system prompt contains the deferred rubrics.
        for agent in agents:
            assert "agent-strat" not in agent._system_prompt
            assert "iter-budget" not in agent._system_prompt

    async def test_run_specialist_returns_proposals_from_parsed_payload(self) -> None:
        """``run_specialist`` unwraps ``SpecialistProposals.proposals`` into a list."""
        emitter = InMemoryEmitter(trace_id="t")
        canned = SpecialistProposals(proposals=[_proposal("prompts-scope", target_dimension="prompts")])
        mock_client = MockLLMClient(
            [
                LLMResponse(
                    content=canned.model_dump_json(),
                    tool_calls=[],
                    usage=make_usage(),
                    model="fake",
                    stop_reason="end_turn",
                    parsed=None,
                )
            ]
        )
        proposals = await run_specialist(
            target_dimension="prompts",
            rubrics=[_rubric("prompts-scope", target_dimension="prompts")],
            trace_events=[],
            llm_client=mock_client,
            emitter=emitter,
        )
        assert len(proposals) == 1
        assert proposals[0].rubric_id == "prompts-scope"

    async def test_run_specialist_raises_when_parsed_payload_missing(self) -> None:
        """A completed specialist without a parsed payload is a contract violation.

        The advisor wires ``output_schema=SpecialistProposals`` on the
        specialist agent; if the agent framework returns without populating
        ``result.parsed``, that is a bug in the pipeline, not a normal
        outcome — propagate as a :class:`RuntimeError` rather than
        silently returning an empty proposal list that masks the failure.
        """
        emitter = InMemoryEmitter(trace_id="t")
        # Content ``None`` skips the MockLLMClient's output_schema parsing,
        # so ``response.parsed`` remains ``None`` and the specialist's
        # ``result.parsed`` is also ``None``.
        mock_client = MockLLMClient(
            [
                LLMResponse(
                    content=None,
                    tool_calls=[],
                    usage=make_usage(),
                    model="fake",
                    stop_reason="end_turn",
                    parsed=None,
                )
            ]
        )
        with pytest.raises(RuntimeError, match="advisor-prompts"):
            await run_specialist(
                target_dimension="prompts",
                rubrics=[_rubric("prompts-scope", target_dimension="prompts")],
                trace_events=[],
                llm_client=mock_client,
                emitter=emitter,
            )

    async def test_specialist_produces_proposal_matching_its_dimension(self) -> None:
        """A specialist invoked with canned output returns proposals whose
        ``rubric_id`` originates from the specialist's target dimension."""
        emitter = InMemoryEmitter(trace_id="t")
        rubric = _rubric(
            "tooldesc-missing",
            target_dimension="tool_descriptions",
            category=ProposalCategory.TOOL_DESCRIPTIONS,
        )
        proposal = _proposal(
            "tooldesc-missing",
            target_dimension="tool_descriptions",
            category=ProposalCategory.TOOL_DESCRIPTIONS,
        )
        canned = SpecialistProposals(proposals=[proposal])
        mock_client = MockLLMClient(
            [
                LLMResponse(
                    content=canned.model_dump_json(),
                    tool_calls=[],
                    usage=make_usage(),
                    model="fake",
                    stop_reason="end_turn",
                    parsed=None,
                )
            ]
        )
        agents = build_all_specialists(
            rubrics=[rubric],
            trace_events=[],
            llm_client=mock_client,
            emitter=emitter,
        )
        tooldesc_agent = next(a for a in agents if a.name == "advisor-tool_descriptions")
        result = await tooldesc_agent.run("Emit proposals.")
        assert isinstance(result.parsed, SpecialistProposals)
        # MockLLMClient returns the same canned response regardless of prompt,
        # but the test confirms the schema is correctly wired end-to-end.
        # The canned rubric_id comes from the tool_descriptions dimension.
        assert result.parsed.proposals[0].rubric_id == "tooldesc-missing"
        # Confirm the mock was called with the specialist's system prompt,
        # which only includes the tool_descriptions rubric.
        assert "tooldesc-missing" in mock_client.calls[0]["system_prompt"]
        # Serialize the mock call back to JSON to confirm the structured-output
        # round-trip worked through MockLLMClient.
        dumped = json.loads(canned.model_dump_json())
        assert dumped["proposals"][0]["target_dimension"] == "tool_descriptions"
