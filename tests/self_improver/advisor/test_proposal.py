"""Unit tests for :mod:`self_improver.advisor.proposal`.

Covers instantiation, serialization round-trip, enum strictness, frozen
semantics, and ``RubricSource`` label enforcement.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from self_improver.advisor import (
    EvidenceReference,
    Proposal,
    ProposalCategory,
    ProposalSeverity,
    RubricSource,
)


def _minimal_proposal(**overrides: object) -> Proposal:
    defaults: dict[str, object] = {
        "rubric_id": "prompts-ambiguous-exit-criteria",
        "rubric_source": RubricSource.BUILTIN,
        "severity": ProposalSeverity.WARNING,
        "category": ProposalCategory.PROMPTS,
        "target_dimension": "prompts",
        "headline": "Ambiguous exit criteria",
        "detail": "The coordinator prompt does not define a termination condition.",
        "suggested_action": "Add an explicit 'stop when' clause to the prompt.",
        "ranking_score": 0.5,
    }
    defaults.update(overrides)
    return Proposal(**defaults)  # type: ignore[arg-type]


class TestEvidenceReference:
    def test_minimal_instantiation(self) -> None:
        ref = EvidenceReference(event_index=0, event_type="agent.start", excerpt="system: ...")
        assert ref.event_index == 0
        assert ref.event_type == "agent.start"
        assert ref.excerpt == "system: ..."

    def test_frozen(self) -> None:
        ref = EvidenceReference(event_index=0, event_type="agent.start", excerpt="system: ...")
        with pytest.raises(ValidationError):
            ref.event_index = 1  # type: ignore[misc]

    def test_negative_event_index_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvidenceReference(event_index=-1, event_type="agent.start", excerpt="x")

    def test_empty_event_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvidenceReference(event_index=0, event_type="", excerpt="x")

    def test_empty_excerpt_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvidenceReference(event_index=0, event_type="agent.start", excerpt="")

    def test_serialization_round_trip(self) -> None:
        original = EvidenceReference(event_index=3, event_type="tool.call", excerpt="search(query='x')")
        restored = EvidenceReference.model_validate(original.model_dump())
        assert restored == original


class TestProposalEnums:
    def test_severity_values(self) -> None:
        assert ProposalSeverity.CRITICAL.value == "critical"
        assert ProposalSeverity.WARNING.value == "warning"
        assert ProposalSeverity.OBSERVATION.value == "observation"

    def test_category_values(self) -> None:
        # The advisor-specific and run-analyst-taxonomy categories both live
        # in the enum; absence of either set would break schema/rubric parity.
        advisor_specific = {
            ProposalCategory.PROMPTS.value,
            ProposalCategory.TOOL_DESCRIPTIONS.value,
            ProposalCategory.COORDINATION_PATTERNS.value,
            ProposalCategory.AGENT_STRATEGY.value,
            ProposalCategory.ITERATION_BUDGETS.value,
        }
        run_analyst_taxonomy = {
            ProposalCategory.APPLICATION_LOGIC.value,
            ProposalCategory.CONFIGURATION.value,
            ProposalCategory.SDK.value,
            ProposalCategory.EVALUATION.value,
            ProposalCategory.OBSERVABILITY.value,
        }
        assert advisor_specific.issubset({c.value for c in ProposalCategory})
        assert run_analyst_taxonomy.issubset({c.value for c in ProposalCategory})

    def test_rubric_source_values(self) -> None:
        assert {s.value for s in RubricSource} == {"builtin", "custom"}


class TestProposal:
    def test_minimal_instantiation(self) -> None:
        proposal = _minimal_proposal()
        assert proposal.rubric_id == "prompts-ambiguous-exit-criteria"
        assert proposal.rubric_source is RubricSource.BUILTIN
        assert proposal.severity is ProposalSeverity.WARNING
        assert proposal.category is ProposalCategory.PROMPTS
        assert proposal.target_dimension == "prompts"
        assert proposal.evidence == []
        assert proposal.ranking_score == 0.5

    def test_frozen(self) -> None:
        proposal = _minimal_proposal()
        with pytest.raises(ValidationError):
            proposal.headline = "mutated"  # type: ignore[misc]

    def test_serialization_round_trip(self) -> None:
        evidence = [EvidenceReference(event_index=2, event_type="tool.call", excerpt="x")]
        original = _minimal_proposal(
            rubric_source=RubricSource.CUSTOM,
            severity=ProposalSeverity.CRITICAL,
            evidence=evidence,
            ranking_score=0.82,
        )
        restored = Proposal.model_validate(original.model_dump())
        assert restored == original
        assert restored.rubric_source is RubricSource.CUSTOM
        assert restored.evidence == evidence

    def test_rubric_source_required(self) -> None:
        # A Proposal without an explicit ``rubric_source`` must fail — the
        # label is load-bearing for attribution and the loader is the only
        # authority that sets it.
        with pytest.raises(ValidationError) as excinfo:
            Proposal(  # type: ignore[call-arg]
                rubric_id="x",
                severity=ProposalSeverity.WARNING,
                category=ProposalCategory.PROMPTS,
                target_dimension="prompts",
                headline="h",
                detail="d",
                suggested_action="a",
                ranking_score=0.5,
            )
        assert "rubric_source" in str(excinfo.value)

    def test_invalid_severity_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _minimal_proposal(severity="catastrophic")

    def test_invalid_category_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _minimal_proposal(category="mystery")

    def test_invalid_rubric_source_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _minimal_proposal(rubric_source="vendor")

    def test_ranking_score_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _minimal_proposal(ranking_score=1.1)
        with pytest.raises(ValidationError):
            _minimal_proposal(ranking_score=-0.01)

    def test_empty_strings_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _minimal_proposal(headline="")
        with pytest.raises(ValidationError):
            _minimal_proposal(detail="")
        with pytest.raises(ValidationError):
            _minimal_proposal(suggested_action="")
        with pytest.raises(ValidationError):
            _minimal_proposal(target_dimension="")
        with pytest.raises(ValidationError):
            _minimal_proposal(rubric_id="")
