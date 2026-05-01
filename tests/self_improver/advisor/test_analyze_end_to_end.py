"""End-to-end mock-driven integration test for the advisor runtime.

Drives :func:`self_improver.advisor.analyze` with a :class:`MockLLMClient`
configured so each of the three launch specialists receives one
scripted :class:`SpecialistProposals` response. Specialists run in
parallel via :func:`asyncio.gather`; each does a single LLM call and
contributes proposals to the aggregated, ranked report. The test
exercises the full path from a :class:`~pathlib.Path` fixture trace
through to rendered JSON and Markdown strings, and asserts
deterministic (byte-equal) re-rendering.

Assertions cover:

- Proposal ordering: critical → warning → observation, then ``ranking_score``
  descending, then ``rubric_id`` ASCII-ascending as a tie-break.
- ``rubric_source`` attribution: a builtin-only run produces only
  :attr:`RubricSource.BUILTIN` proposals; a composite run with a custom
  rubric path also yields proposals carrying :attr:`RubricSource.CUSTOM`
  for rubric ids from that path.
- Usage surfacing: the Markdown footer reports non-zero ``input_tokens``
  and ``output_tokens`` drawn from the mock's canned responses.
"""

from __future__ import annotations

from pathlib import Path

from self_improver.advisor._specialists import SpecialistProposals
from self_improver.advisor.analyze import AdvisorReport, analyze
from self_improver.advisor.formatters import JSONFormatter, MarkdownFormatter
from self_improver.advisor.proposal import (
    EvidenceReference,
    Proposal,
    ProposalCategory,
    ProposalSeverity,
    RubricSource,
)

from nanitics.infrastructure.llm.mock import MockLLMClient
from nanitics.infrastructure.llm.protocol import LLMResponse
from tests.testing_helpers import make_usage

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MINIMAL_TRACE = FIXTURES_DIR / "analyze_minimal_trace.json"


# --- Helpers --------------------------------------------------------------


def _proposal(
    rubric_id: str,
    *,
    severity: ProposalSeverity,
    category: ProposalCategory,
    target_dimension: str,
    source: RubricSource = RubricSource.BUILTIN,
    ranking_score: float = 0.5,
    headline: str | None = None,
) -> Proposal:
    return Proposal(
        rubric_id=rubric_id,
        rubric_source=source,
        severity=severity,
        category=category,
        target_dimension=target_dimension,
        headline=headline or f"Headline for {rubric_id}",
        detail=f"Detail paragraph for {rubric_id}.",
        evidence=[
            EvidenceReference(
                event_index=0,
                event_type="agent.start",
                excerpt="Do the thing.",
            ),
        ],
        suggested_action=f"Act on {rubric_id}.",
        ranking_score=ranking_score,
    )


def _specialist_response(*proposals: Proposal) -> LLMResponse:
    payload = SpecialistProposals(proposals=list(proposals)).model_dump_json()
    return LLMResponse(
        content=payload,
        tool_calls=[],
        usage=make_usage(input_tokens=4, output_tokens=9),
        model="mock",
        stop_reason="end_turn",
    )


def _full_delegation_script(
    prompts_proposals: list[Proposal],
    tool_desc_proposals: list[Proposal],
    coord_proposals: list[Proposal],
) -> list[LLMResponse]:
    """Scripted parallel-specialist flow.

    The advisor runs the three launch specialists in parallel via
    :func:`asyncio.gather`; each does exactly one LLM call. The
    :class:`MockLLMClient` consumes responses sequentially, so any
    specialist may receive any response in the list — the test relies
    on the aggregate ranked output rather than per-specialist
    attribution.
    """
    return [
        _specialist_response(*prompts_proposals),
        _specialist_response(*tool_desc_proposals),
        _specialist_response(*coord_proposals),
    ]


def _write_custom_rubric(tmp_path: Path) -> Path:
    custom_dir = tmp_path / "custom-rubrics"
    custom_dir.mkdir()
    (custom_dir / "prompts-custom-adopter-rule.md").write_text(
        "---\n"
        "id: prompts-custom-adopter-rule\n"
        "severity: observation\n"
        "category: prompts\n"
        "target_dimension: prompts\n"
        "---\n\n"
        "# Custom adopter rule\n\n"
        "Body for the adopter's custom rubric.\n",
        encoding="utf-8",
    )
    return custom_dir


# --- The composite integration test --------------------------------------


class TestAdvisorEndToEnd:
    """One composite test driving the whole advisor from a fixture trace.

    Each sub-test shares the same scripted coordinator flow shape but
    varies the specialist proposals and (optionally) the adopter rubric
    set so the assertions exercise distinct edges of the public contract.
    """

    async def test_composite_flow_renders_deterministic_outputs(self) -> None:
        """Builtin-only composite run end-to-end.

        Drives :func:`analyze` against the fixture trace with a
        :class:`MockLLMClient` that produces proposals spanning every
        severity and every launch target dimension so the ordering and
        rendering assertions see a realistic report.

        Asserts:

        - The report is an :class:`AdvisorReport`.
        - Proposal order is ``critical → warning → observation`` and,
          within a severity, descending ``ranking_score`` with ASCII-
          ascending ``rubric_id`` as the tie-break.
        - Every proposal carries :attr:`RubricSource.BUILTIN` since no
          custom rubric path was supplied.
        - Rendered Markdown and JSON are byte-equal on repeat invocations
          (determinism).
        - The Markdown footer surfaces non-zero ``Input tokens`` and
          ``Output tokens`` sourced from the mock's canned responses.
        """
        # Prompts specialist: two proposals at different severities and
        # scores so the ordering test bites.
        prompts_proposals = [
            _proposal(
                "prompts-missing-exit-criteria",
                severity=ProposalSeverity.CRITICAL,
                category=ProposalCategory.PROMPTS,
                target_dimension="prompts",
                ranking_score=0.95,
            ),
            _proposal(
                "prompts-role-ambiguous",
                severity=ProposalSeverity.WARNING,
                category=ProposalCategory.PROMPTS,
                target_dimension="prompts",
                ranking_score=0.6,
            ),
        ]
        # Tool-descriptions specialist: one proposal at CRITICAL with a
        # lower score than the prompts CRITICAL, so within-severity
        # score ordering is testable.
        tool_desc_proposals = [
            _proposal(
                "tool-descriptions-missing-docstring",
                severity=ProposalSeverity.CRITICAL,
                category=ProposalCategory.TOOL_DESCRIPTIONS,
                target_dimension="tool_descriptions",
                ranking_score=0.85,
            ),
        ]
        # Coordination specialist: one WARNING and one OBSERVATION so the
        # severity bucket boundary is exercised.
        coord_proposals = [
            _proposal(
                "coordination-patterns-no-termination",
                severity=ProposalSeverity.WARNING,
                category=ProposalCategory.COORDINATION_PATTERNS,
                target_dimension="coordination_patterns",
                ranking_score=0.7,
            ),
            _proposal(
                "coordination-patterns-delegation-depth",
                severity=ProposalSeverity.OBSERVATION,
                category=ProposalCategory.COORDINATION_PATTERNS,
                target_dimension="coordination_patterns",
                ranking_score=0.4,
            ),
        ]

        client = MockLLMClient(
            _full_delegation_script(
                prompts_proposals,
                tool_desc_proposals,
                coord_proposals,
            )
        )

        report = await analyze(MINIMAL_TRACE, llm_client=client)

        # --- Report shape ---
        assert isinstance(report, AdvisorReport)
        assert report.trace_id == "advisor-core-phase2-fixture"
        assert report.target_dimensions_analyzed == [
            "prompts",
            "tool_descriptions",
            "coordination_patterns",
        ]
        assert len(report.proposals) == 5

        # --- Proposal ordering ---
        # Critical bucket: prompts (0.95) then tool-descriptions (0.85).
        # Warning bucket:  coordination (0.7) then prompts (0.6).
        # Observation bucket: coordination (0.4).
        assert [p.rubric_id for p in report.proposals] == [
            "prompts-missing-exit-criteria",
            "tool-descriptions-missing-docstring",
            "coordination-patterns-no-termination",
            "prompts-role-ambiguous",
            "coordination-patterns-delegation-depth",
        ]

        # --- Rubric source attribution ---
        assert all(p.rubric_source == RubricSource.BUILTIN for p in report.proposals)
        assert report.rubric_counts == {RubricSource.BUILTIN: 5}

        # --- Deterministic JSON and Markdown rendering ---
        json_first = JSONFormatter().render(report)
        json_second = JSONFormatter().render(report)
        md_first = MarkdownFormatter().render(report)
        md_second = MarkdownFormatter().render(report)
        assert json_first == json_second
        assert md_first == md_second

        # JSON round-trip matches the original report (extra safety
        # alongside the dedicated JSON round-trip test in
        # ``test_formatters.py``).
        reconstructed = AdvisorReport.model_validate_json(json_first)
        assert reconstructed == report

        # --- Usage surfacing in the Markdown footer ---
        assert "## Usage" in md_first
        assert f"Input tokens: {report.usage.input_tokens}" in md_first
        assert f"Output tokens: {report.usage.output_tokens}" in md_first
        assert report.usage.input_tokens > 0
        assert report.usage.output_tokens > 0

    async def test_composite_flow_attributes_custom_rubric_source(
        self,
        tmp_path: Path,
    ) -> None:
        """Composite run with an adopter custom rubric path.

        The prompts specialist returns one builtin and one custom-sourced
        proposal; the other two specialists return builtin-only proposals.
        Asserts that proposals derived from the custom rubric path carry
        :attr:`RubricSource.CUSTOM` and that the report's ``rubric_counts``
        distinguishes builtin and custom totals.
        """
        custom_dir = _write_custom_rubric(tmp_path)

        prompts_proposals = [
            _proposal(
                "prompts-missing-exit-criteria",
                severity=ProposalSeverity.CRITICAL,
                category=ProposalCategory.PROMPTS,
                target_dimension="prompts",
                ranking_score=0.9,
            ),
            _proposal(
                "prompts-custom-adopter-rule",
                severity=ProposalSeverity.OBSERVATION,
                category=ProposalCategory.PROMPTS,
                target_dimension="prompts",
                source=RubricSource.CUSTOM,
                ranking_score=0.3,
            ),
        ]
        tool_desc_proposals = [
            _proposal(
                "tool-descriptions-missing-docstring",
                severity=ProposalSeverity.WARNING,
                category=ProposalCategory.TOOL_DESCRIPTIONS,
                target_dimension="tool_descriptions",
                ranking_score=0.55,
            ),
        ]
        coord_proposals = [
            _proposal(
                "coordination-patterns-no-termination",
                severity=ProposalSeverity.WARNING,
                category=ProposalCategory.COORDINATION_PATTERNS,
                target_dimension="coordination_patterns",
                ranking_score=0.65,
            ),
        ]

        client = MockLLMClient(
            _full_delegation_script(
                prompts_proposals,
                tool_desc_proposals,
                coord_proposals,
            )
        )

        report = await analyze(
            MINIMAL_TRACE,
            llm_client=client,
            rubrics=[custom_dir],
        )

        # Custom-sourced proposals are isolated to the adopter's rubric id.
        custom_proposals = [p for p in report.proposals if p.rubric_source == RubricSource.CUSTOM]
        assert [p.rubric_id for p in custom_proposals] == ["prompts-custom-adopter-rule"]

        # Builtin counts still reflect the three builtin proposals that fired.
        assert report.rubric_counts == {RubricSource.BUILTIN: 3, RubricSource.CUSTOM: 1}

        # Rendered outputs still deterministic on repeat.
        md_first = MarkdownFormatter().render(report)
        md_second = MarkdownFormatter().render(report)
        assert md_first == md_second
        # The custom proposal's rubric source is surfaced in the Markdown
        # body so an adopter inspecting the report can tell which rubric
        # fired against their own rule.
        assert "`prompts-custom-adopter-rule` (custom)" in md_first
