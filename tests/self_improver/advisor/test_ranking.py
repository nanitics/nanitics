"""Unit tests for :mod:`self_improver.advisor.ranking`.

Covers empty input, single-proposal input, severity ordering, within-severity
score ordering, ASCII tie-break, and the full interaction of all three rules.
"""

from __future__ import annotations

import pytest
from self_improver.advisor import (
    Proposal,
    ProposalCategory,
    ProposalSeverity,
    RubricSource,
    rank_proposals,
)


def _proposal(
    rubric_id: str,
    severity: ProposalSeverity,
    ranking_score: float,
    *,
    rubric_source: RubricSource = RubricSource.BUILTIN,
    category: ProposalCategory = ProposalCategory.PROMPTS,
    target_dimension: str = "prompts",
) -> Proposal:
    return Proposal(
        rubric_id=rubric_id,
        rubric_source=rubric_source,
        severity=severity,
        category=category,
        target_dimension=target_dimension,
        headline="h",
        detail="d",
        suggested_action="a",
        ranking_score=ranking_score,
    )


class TestRankProposals:
    def test_empty_input_returns_empty_list(self) -> None:
        assert rank_proposals([]) == []

    def test_generator_input_accepted(self) -> None:
        proposals = (_proposal("a", ProposalSeverity.WARNING, 0.5) for _ in range(1))
        result = rank_proposals(proposals)
        assert len(result) == 1
        assert result[0].rubric_id == "a"

    def test_single_proposal(self) -> None:
        proposal = _proposal("only", ProposalSeverity.WARNING, 0.5)
        assert rank_proposals([proposal]) == [proposal]

    def test_severity_order_is_primary(self) -> None:
        observation = _proposal("obs", ProposalSeverity.OBSERVATION, 1.0)
        warning = _proposal("warn", ProposalSeverity.WARNING, 0.1)
        critical = _proposal("crit", ProposalSeverity.CRITICAL, 0.0)
        ranked = rank_proposals([observation, warning, critical])
        assert [p.severity for p in ranked] == [
            ProposalSeverity.CRITICAL,
            ProposalSeverity.WARNING,
            ProposalSeverity.OBSERVATION,
        ]

    def test_within_severity_score_descending(self) -> None:
        low = _proposal("low", ProposalSeverity.WARNING, 0.2)
        mid = _proposal("mid", ProposalSeverity.WARNING, 0.5)
        high = _proposal("high", ProposalSeverity.WARNING, 0.9)
        ranked = rank_proposals([low, mid, high])
        assert [p.rubric_id for p in ranked] == ["high", "mid", "low"]

    def test_rubric_id_tie_break_ascii(self) -> None:
        a = _proposal("aaa", ProposalSeverity.CRITICAL, 0.5)
        b = _proposal("bbb", ProposalSeverity.CRITICAL, 0.5)
        z = _proposal("zzz", ProposalSeverity.CRITICAL, 0.5)
        # Input order deliberately inverted — ranking must not reflect it.
        ranked = rank_proposals([z, b, a])
        assert [p.rubric_id for p in ranked] == ["aaa", "bbb", "zzz"]

    def test_all_three_rules_together(self) -> None:
        proposals = [
            _proposal("warn-low", ProposalSeverity.WARNING, 0.2),
            _proposal("crit-tie-b", ProposalSeverity.CRITICAL, 0.5),
            _proposal("obs-hi", ProposalSeverity.OBSERVATION, 0.95),
            _proposal("crit-high", ProposalSeverity.CRITICAL, 0.9),
            _proposal("crit-tie-a", ProposalSeverity.CRITICAL, 0.5),
            _proposal("warn-high", ProposalSeverity.WARNING, 0.8),
        ]
        ranked = rank_proposals(proposals)
        assert [p.rubric_id for p in ranked] == [
            "crit-high",
            "crit-tie-a",
            "crit-tie-b",
            "warn-high",
            "warn-low",
            "obs-hi",
        ]

    def test_custom_source_does_not_affect_ordering(self) -> None:
        builtin = _proposal(
            "aaa-builtin",
            ProposalSeverity.CRITICAL,
            0.5,
            rubric_source=RubricSource.BUILTIN,
        )
        custom = _proposal(
            "bbb-custom",
            ProposalSeverity.CRITICAL,
            0.5,
            rubric_source=RubricSource.CUSTOM,
        )
        # Under identical severity and score, pure ASCII tie-break wins —
        # the source label is attribution, not an ordering input.
        ranked = rank_proposals([custom, builtin])
        assert [p.rubric_id for p in ranked] == ["aaa-builtin", "bbb-custom"]

    def test_deterministic_across_runs(self) -> None:
        proposals = [
            _proposal("a", ProposalSeverity.WARNING, 0.5),
            _proposal("b", ProposalSeverity.CRITICAL, 0.3),
            _proposal("c", ProposalSeverity.OBSERVATION, 0.9),
            _proposal("d", ProposalSeverity.WARNING, 0.5),
            _proposal("e", ProposalSeverity.CRITICAL, 0.3),
        ]
        first = rank_proposals(proposals)
        second = rank_proposals(proposals)
        third = rank_proposals(list(reversed(proposals)))
        assert first == second == third

    def test_does_not_mutate_input(self) -> None:
        original = [
            _proposal("z", ProposalSeverity.OBSERVATION, 0.1),
            _proposal("a", ProposalSeverity.CRITICAL, 0.9),
        ]
        snapshot = list(original)
        rank_proposals(original)
        assert original == snapshot

    @pytest.mark.parametrize(
        "severity",
        [ProposalSeverity.CRITICAL, ProposalSeverity.WARNING, ProposalSeverity.OBSERVATION],
    )
    def test_every_severity_sortable_alone(self, severity: ProposalSeverity) -> None:
        proposals = [
            _proposal("b", severity, 0.5),
            _proposal("a", severity, 0.5),
        ]
        ranked = rank_proposals(proposals)
        assert [p.rubric_id for p in ranked] == ["a", "b"]
