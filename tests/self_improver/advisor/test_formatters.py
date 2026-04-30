"""Unit tests for :mod:`self_improver.advisor.formatters`.

Covers both shipping formatters:

- :class:`JSONFormatter` round-trip (render → ``model_validate_json``).
- :class:`MarkdownFormatter` deterministic snapshot, severity grouping,
  empty-severity handling, and structural-proposal handling.
- :class:`OutputFormatter` protocol conformance for a custom renderer.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from self_improver.advisor.analyze import AdvisorReport
from self_improver.advisor.formatters import (
    JSONFormatter,
    MarkdownFormatter,
    OutputFormatter,
)
from self_improver.advisor.proposal import (
    EvidenceReference,
    Proposal,
    ProposalCategory,
    ProposalSeverity,
    RubricSource,
)

from nanitics.infrastructure.observability.events import Usage


def _proposal(
    rubric_id: str,
    *,
    severity: ProposalSeverity = ProposalSeverity.WARNING,
    category: ProposalCategory = ProposalCategory.PROMPTS,
    source: RubricSource = RubricSource.BUILTIN,
    target_dimension: str = "prompts",
    evidence: list[EvidenceReference] | None = None,
    headline: str | None = None,
    detail: str = "Detail paragraph.",
    suggested_action: str = "Do something.",
    ranking_score: float = 0.5,
) -> Proposal:
    return Proposal(
        rubric_id=rubric_id,
        rubric_source=source,
        severity=severity,
        category=category,
        target_dimension=target_dimension,
        headline=headline or f"Headline for {rubric_id}",
        detail=detail,
        evidence=evidence
        if evidence is not None
        else [
            EvidenceReference(event_index=0, event_type="agent.start", excerpt="task"),
        ],
        suggested_action=suggested_action,
        ranking_score=ranking_score,
    )


def _report(
    proposals: list[Proposal] | None = None,
    *,
    usage: Usage | None = None,
    rubric_counts: dict[RubricSource, int] | None = None,
    trace_id: str = "trace-42",
) -> AdvisorReport:
    """Build a fixed, deterministic :class:`AdvisorReport` for snapshot tests."""
    ordered = proposals if proposals is not None else [_proposal("builtin-proposals")]
    return AdvisorReport(
        trace_id=trace_id,
        generated_at=datetime(2026, 4, 16, 12, 0, 0, tzinfo=UTC),
        proposals=ordered,
        usage=usage
        or Usage(
            input_tokens=120,
            output_tokens=45,
        ),
        rubric_counts=rubric_counts or {RubricSource.BUILTIN: len(ordered)},
        target_dimensions_analyzed=["prompts", "tool_descriptions", "coordination_patterns"],
    )


class TestJSONFormatter:
    def test_render_is_indented_json(self) -> None:
        report = _report()
        rendered = JSONFormatter().render(report)
        assert rendered.startswith("{")
        # Indented output has newlines and two-space indent markers.
        assert "\n  " in rendered

    def test_round_trip_reconstructs_report(self) -> None:
        report = _report(
            [
                _proposal(
                    "high-critical",
                    severity=ProposalSeverity.CRITICAL,
                    ranking_score=0.9,
                ),
                _proposal("low-observation", severity=ProposalSeverity.OBSERVATION),
            ]
        )
        rendered = JSONFormatter().render(report)
        reconstructed = AdvisorReport.model_validate_json(rendered)
        assert reconstructed == report

    def test_render_is_deterministic(self) -> None:
        report = _report()
        formatter = JSONFormatter()
        assert formatter.render(report) == formatter.render(report)


class TestMarkdownFormatter:
    def test_snapshot_matches_expected_layout(self) -> None:
        """Byte-level snapshot so layout drift surfaces immediately."""
        report = _report(
            [
                _proposal(
                    "critical-alpha",
                    severity=ProposalSeverity.CRITICAL,
                    category=ProposalCategory.PROMPTS,
                    headline="Critical alpha headline",
                    ranking_score=0.9,
                ),
                _proposal(
                    "warning-beta",
                    severity=ProposalSeverity.WARNING,
                    category=ProposalCategory.TOOL_DESCRIPTIONS,
                    target_dimension="tool_descriptions",
                    headline="Warning beta headline",
                    evidence=[],
                    ranking_score=0.5,
                ),
            ],
            rubric_counts={RubricSource.BUILTIN: 1, RubricSource.CUSTOM: 1},
        )
        # Flip the second proposal's source to CUSTOM so the footer counts
        # reflect one of each — achieved by rebuilding with rubric_counts
        # already set above and the per-proposal source controlled here.
        proposals = [
            report.proposals[0],
            _proposal(
                "warning-beta",
                severity=ProposalSeverity.WARNING,
                category=ProposalCategory.TOOL_DESCRIPTIONS,
                target_dimension="tool_descriptions",
                source=RubricSource.CUSTOM,
                headline="Warning beta headline",
                evidence=[],
                ranking_score=0.5,
            ),
        ]
        report = _report(
            proposals,
            rubric_counts={RubricSource.BUILTIN: 1, RubricSource.CUSTOM: 1},
        )
        rendered = MarkdownFormatter().render(report)
        expected = (
            "# Advisor Report\n\n"
            "- **Trace:** trace-42\n"
            "- **Generated at:** 2026-04-16T12:00:00Z\n"
            "- **Proposals:** 2 (1 critical, 1 warning, 0 observation)\n\n"
            "## Critical\n\n"
            "### Critical alpha headline\n\n"
            "- **Rubric:** `critical-alpha` (builtin)\n"
            "- **Category:** prompts\n"
            "- **Target:** prompts\n\n"
            "Detail paragraph.\n\n"
            "#### Evidence\n"
            '- Event 0 (`agent.start`) — "task"\n\n'
            "#### Suggested action\n"
            "Do something.\n\n"
            "---\n\n"
            "## Warning\n\n"
            "### Warning beta headline\n\n"
            "- **Rubric:** `warning-beta` (custom)\n"
            "- **Category:** tool-descriptions\n"
            "- **Target:** tool_descriptions\n\n"
            "Detail paragraph.\n\n"
            "#### Evidence\n"
            "- _No evidence cited (structural proposal)._\n\n"
            "#### Suggested action\n"
            "Do something.\n\n"
            "---\n\n"
            "## Observation\n\n"
            "_No proposals at this severity._\n\n"
            "## Usage\n\n"
            "- Input tokens: 120\n"
            "- Output tokens: 45\n"
            "- Rubric counts: builtin=1, custom=1\n"
        )
        assert rendered == expected

    def test_severity_grouping_order(self) -> None:
        """Critical → warning → observation, regardless of input order."""
        proposals = [
            _proposal("obs-1", severity=ProposalSeverity.OBSERVATION),
            _proposal("crit-1", severity=ProposalSeverity.CRITICAL),
            _proposal("warn-1", severity=ProposalSeverity.WARNING),
        ]
        rendered = MarkdownFormatter().render(_report(proposals))
        critical_idx = rendered.index("## Critical")
        warning_idx = rendered.index("## Warning")
        observation_idx = rendered.index("## Observation")
        assert critical_idx < warning_idx < observation_idx

    def test_structural_proposal_renders_explicit_no_evidence_line(self) -> None:
        proposal = _proposal("no-evidence", evidence=[])
        rendered = MarkdownFormatter().render(_report([proposal]))
        assert "- _No evidence cited (structural proposal)._" in rendered

    def test_empty_severity_renders_placeholder(self) -> None:
        rendered = MarkdownFormatter().render(_report(proposals=[]))
        # All three severity sections present, each with the placeholder.
        for heading in ("## Critical", "## Warning", "## Observation"):
            assert heading in rendered
        assert rendered.count("_No proposals at this severity._") == 3

    def test_render_is_deterministic(self) -> None:
        report = _report()
        formatter = MarkdownFormatter()
        assert formatter.render(report) == formatter.render(report)


class TestOutputFormatterProtocol:
    def test_shipping_formatters_satisfy_protocol(self) -> None:
        assert isinstance(JSONFormatter(), OutputFormatter)
        assert isinstance(MarkdownFormatter(), OutputFormatter)

    def test_custom_formatter_satisfies_protocol(self) -> None:
        class UpperCaseFormatter:
            def render(self, report: AdvisorReport) -> str:
                return JSONFormatter().render(report).upper()

        formatter = UpperCaseFormatter()
        assert isinstance(formatter, OutputFormatter)
        assert "TRACE-42" in formatter.render(_report())

    def test_non_renderable_object_fails_protocol_check(self) -> None:
        class NotAFormatter:
            pass

        assert not isinstance(NotAFormatter(), OutputFormatter)

    def test_protocol_render_signature_is_callable(self) -> None:
        """Sanity check that the protocol method is genuinely abstract."""

        class Concrete:
            def render(self, report: AdvisorReport) -> str:
                return ""

        formatter: OutputFormatter = Concrete()
        # Mypy-wise, we already satisfy; runtime-wise, no execution needed.
        # But exercising the call at runtime silences any accidental
        # regression where the protocol declares extra methods.
        assert formatter.render(_report()) == ""


def test_markdown_formatter_structural_proposal_does_not_collapse_section() -> None:
    """Structural proposal (empty evidence) still emits a proper Evidence header."""
    proposal = _proposal(
        "structural-proposal",
        evidence=[],
        severity=ProposalSeverity.CRITICAL,
    )
    rendered = MarkdownFormatter().render(_report([proposal]))
    # Evidence header is still present; body line is the explicit placeholder.
    idx_evidence = rendered.index("#### Evidence")
    # Ensure placeholder appears after the header.
    placeholder_idx = rendered.index("- _No evidence cited", idx_evidence)
    assert placeholder_idx > idx_evidence


def test_protocol_is_runtime_checkable_for_duck_types() -> None:
    """A plain object with a ``render`` method satisfies the protocol."""

    class Duck:
        def render(self, report: AdvisorReport) -> str:
            return ""

    assert isinstance(Duck(), OutputFormatter)


def test_usage_footer_no_crash_on_zero_counts() -> None:
    """Report with zero proposals still renders a valid footer."""
    rendered = MarkdownFormatter().render(
        _report(proposals=[], rubric_counts={}),
    )
    assert "builtin=0, custom=0" in rendered


# Ensure pytest-asyncio does not try to run synchronous tests as async;
# the formatter module is pure/sync and has no async entry points.
@pytest.mark.parametrize("formatter", [JSONFormatter(), MarkdownFormatter()])
def test_shipping_formatters_render_type_is_str(formatter: OutputFormatter) -> None:
    assert isinstance(formatter.render(_report()), str)
