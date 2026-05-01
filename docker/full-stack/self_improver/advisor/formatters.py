"""Output formatters for :class:`AdvisorReport`.

Ships two default implementations — :class:`JSONFormatter` and
:class:`MarkdownFormatter` — behind a :class:`OutputFormatter` protocol.

Both shipping formatters are deterministic: identical
:class:`AdvisorReport` in, byte-identical string out. The Markdown layout
is a fixed template rather than a free-form renderer so snapshot tests
can compare output as strings.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from self_improver.advisor.analyze import AdvisorReport
from self_improver.advisor.proposal import Proposal, ProposalSeverity, RubricSource

_SEVERITY_HEADINGS: tuple[tuple[ProposalSeverity, str], ...] = (
    (ProposalSeverity.CRITICAL, "Critical"),
    (ProposalSeverity.WARNING, "Warning"),
    (ProposalSeverity.OBSERVATION, "Observation"),
)

_EMPTY_SEVERITY_BODY = "_No proposals at this severity._"
_STRUCTURAL_EVIDENCE_LINE = "- _No evidence cited (structural proposal)._"


@runtime_checkable
class OutputFormatter(Protocol):
    """Render an :class:`AdvisorReport` to a single output string.

    Duck-typed: any class implementing a ``render(report) -> str`` method
    satisfies this protocol at runtime via :func:`isinstance`. Adopters
    shipping post-launch renderers (PR comments, Jira, etc.) implement
    this protocol directly.
    """

    def render(self, report: AdvisorReport) -> str:
        """Render ``report`` to a string."""
        ...


class JSONFormatter:
    """Render an :class:`AdvisorReport` as indented JSON.

    The output is the direct ``model_dump_json(indent=2)`` of the report —
    a lossless serialization that round-trips through
    :meth:`AdvisorReport.model_validate_json`.
    """

    def render(self, report: AdvisorReport) -> str:
        """Return ``report`` as an indented JSON document."""
        return report.model_dump_json(indent=2)


class MarkdownFormatter:
    """Render an :class:`AdvisorReport` as a Markdown document.

    The layout is deterministic and stable — severity sections are always
    emitted in ``critical → warning → observation`` order, even when empty,
    so snapshot tests can compare byte-equal output.
    """

    def render(self, report: AdvisorReport) -> str:
        """Return ``report`` as a Markdown document."""
        parts: list[str] = [
            _render_header(report),
            *(
                _render_severity_section(severity, heading, report.proposals)
                for severity, heading in _SEVERITY_HEADINGS
            ),
            _render_usage_footer(report),
        ]
        # Separator matches the human-readable cue between sections; the
        # trailing newline keeps terminals from running reports together
        # when concatenated.
        return "\n\n".join(parts) + "\n"


def _render_header(report: AdvisorReport) -> str:
    """Render the report header block."""
    critical = sum(1 for p in report.proposals if p.severity == ProposalSeverity.CRITICAL)
    warning = sum(1 for p in report.proposals if p.severity == ProposalSeverity.WARNING)
    observation = sum(1 for p in report.proposals if p.severity == ProposalSeverity.OBSERVATION)
    generated_at_iso = report.generated_at.isoformat().replace("+00:00", "Z")
    return (
        "# Advisor Report\n\n"
        f"- **Trace:** {report.trace_id}\n"
        f"- **Generated at:** {generated_at_iso}\n"
        f"- **Proposals:** {len(report.proposals)} "
        f"({critical} critical, {warning} warning, {observation} observation)"
    )


def _render_severity_section(
    severity: ProposalSeverity,
    heading: str,
    proposals: list[Proposal],
) -> str:
    """Render a single severity section — header plus grouped proposals."""
    scoped = [p for p in proposals if p.severity == severity]
    lines = [f"## {heading}"]
    if not scoped:
        lines.append(_EMPTY_SEVERITY_BODY)
        return "\n\n".join(lines)
    rendered_proposals = [_render_proposal(p) for p in scoped]
    return "\n\n".join([lines[0], *rendered_proposals])


def _render_proposal(proposal: Proposal) -> str:
    """Render a single proposal block.

    The body includes the headline as an H3, a metadata list, the detail
    paragraph, an evidence subsection, a suggested-action subsection, and
    a horizontal-rule separator so consecutive proposals within the same
    severity are visually distinguishable.
    """
    evidence_block = _render_evidence_block(proposal)
    return (
        f"### {proposal.headline}\n\n"
        f"- **Rubric:** `{proposal.rubric_id}` ({proposal.rubric_source.value})\n"
        f"- **Category:** {proposal.category.value}\n"
        f"- **Target:** {proposal.target_dimension}\n\n"
        f"{proposal.detail}\n\n"
        f"#### Evidence\n{evidence_block}\n\n"
        f"#### Suggested action\n{proposal.suggested_action}\n\n"
        "---"
    )


def _render_evidence_block(proposal: Proposal) -> str:
    """Render a proposal's evidence list as bullet points."""
    if not proposal.evidence:
        # Structural proposals may have zero evidence — render an explicit
        # line rather than an empty section so the Markdown structure
        # stays stable for snapshot tests.
        return _STRUCTURAL_EVIDENCE_LINE
    return "\n".join(f'- Event {ref.event_index} (`{ref.event_type}`) — "{ref.excerpt}"' for ref in proposal.evidence)


def _render_usage_footer(report: AdvisorReport) -> str:
    """Render the usage footer with token totals and rubric counts."""
    builtin = report.rubric_counts.get(RubricSource.BUILTIN, 0)
    custom = report.rubric_counts.get(RubricSource.CUSTOM, 0)
    return (
        "## Usage\n\n"
        f"- Input tokens: {report.usage.input_tokens}\n"
        f"- Output tokens: {report.usage.output_tokens}\n"
        f"- Rubric counts: builtin={builtin}, custom={custom}"
    )


__all__ = [
    "JSONFormatter",
    "MarkdownFormatter",
    "OutputFormatter",
]
