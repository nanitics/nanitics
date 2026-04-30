"""Unit tests for :mod:`self_improver.advisor.analyze`.

Covers:

- ``analyze()`` returns an :class:`AdvisorReport` with proposals from every
  launch target dimension when the specialists emit canned output.
- ``analyze()`` accepts a pre-loaded ``list[TraceEvent]`` and skips the
  adapter.
- ``rubrics=None`` loads builtins only; ``rubrics=[...]`` merges custom
  rubrics and the resulting proposals carry ``rubric_source=CUSTOM``.
- A caller-supplied ``emitter`` receives all the events from the run.
- ``write_report`` stages both JSON and Markdown files atomically.
- Every shipping builtin rubric id fires in at least one ``analyze()``
  invocation across the mocked fixture set.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from self_improver.advisor._specialists import SpecialistProposals
from self_improver.advisor.analyze import AdvisorReport, analyze, write_report
from self_improver.advisor.proposal import (
    EvidenceReference,
    Proposal,
    ProposalCategory,
    ProposalSeverity,
    RubricSource,
)
from self_improver.advisor.rubric import load_rubrics
from self_improver.advisor.trace_adapter import load_trace

from nanitics.infrastructure.llm.mock import MockLLMClient
from nanitics.infrastructure.llm.protocol import LLMResponse, Message
from nanitics.infrastructure.observability.emitter import InMemoryEmitter
from nanitics.infrastructure.observability.events import Usage
from tests.testing_helpers import make_usage

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MINIMAL_TRACE = FIXTURES_DIR / "analyze_minimal_trace.json"


# --- Helpers --------------------------------------------------------------


def _proposal(
    rubric_id: str,
    *,
    severity: ProposalSeverity = ProposalSeverity.WARNING,
    category: ProposalCategory = ProposalCategory.PROMPTS,
    target_dimension: str = "prompts",
    source: RubricSource = RubricSource.BUILTIN,
    ranking_score: float = 0.5,
) -> Proposal:
    return Proposal(
        rubric_id=rubric_id,
        rubric_source=source,
        severity=severity,
        category=category,
        target_dimension=target_dimension,
        headline=f"Headline for {rubric_id}",
        detail="Detail paragraph.",
        evidence=[EvidenceReference(event_index=0, event_type="agent.start", excerpt="task")],
        suggested_action="Do something.",
        ranking_score=ranking_score,
    )


def _specialist_response(
    *proposals: Proposal,
    usage: Usage | None = None,
) -> LLMResponse:
    payload = SpecialistProposals(proposals=list(proposals)).model_dump_json()
    return LLMResponse(
        content=payload,
        tool_calls=[],
        usage=usage or make_usage(input_tokens=3, output_tokens=7),
        model="mock",
        stop_reason="end_turn",
    )


def _builtin_per_dimension_proposals() -> tuple[Proposal, Proposal, Proposal]:
    """One proposal per launch target dimension, each tied to a builtin rubric."""
    return (
        _proposal(
            "prompts-missing-exit-criteria",
            severity=ProposalSeverity.CRITICAL,
            category=ProposalCategory.PROMPTS,
            target_dimension="prompts",
            ranking_score=0.9,
        ),
        _proposal(
            "tool-descriptions-missing-docstring",
            severity=ProposalSeverity.CRITICAL,
            category=ProposalCategory.TOOL_DESCRIPTIONS,
            target_dimension="tool_descriptions",
            ranking_score=0.8,
        ),
        _proposal(
            "coordination-patterns-no-termination",
            severity=ProposalSeverity.CRITICAL,
            category=ProposalCategory.COORDINATION_PATTERNS,
            target_dimension="coordination_patterns",
            ranking_score=0.7,
        ),
    )


def _per_specialist_script(
    prompts_proposals: list[Proposal],
    tool_desc_proposals: list[Proposal],
    coord_proposals: list[Proposal],
) -> list[Callable[[list[Message]], LLMResponse]]:
    """Route each specialist call to its own proposal bundle.

    With parallel dispatch the three specialists hit the mock in an order
    that depends on the event loop, so per-index responses would assign
    the wrong bundle. Matching by ``target_dimension`` in the system
    prompt keeps the test independent of scheduling order.
    """
    by_dimension = {
        "prompts": _specialist_response(*prompts_proposals),
        "tool_descriptions": _specialist_response(*tool_desc_proposals),
        "coordination_patterns": _specialist_response(*coord_proposals),
    }

    def _route(messages: list[Message]) -> LLMResponse:  # pragma: no cover - closure, exercised via mock
        raise AssertionError("unused closure")

    # MockLLMClient only exposes messages in the callable; the system
    # prompt is stored on the ``calls`` list. A simpler routing pattern
    # works when we return one callable per call index that looks at
    # ``mock_client.calls`` directly; but we instead build a callable
    # closure over the mock client so we can read the most recent call.
    # The helper below returns three callables, each aware of which
    # dimension it matches.
    dimensions_seen: list[str] = []

    def _make_router(mock_client_holder: list[MockLLMClient]) -> Callable[[list[Message]], LLMResponse]:
        def router(_messages: list[Message]) -> LLMResponse:
            mock_client = mock_client_holder[0]
            system_prompt = mock_client.calls[-1]["system_prompt"]
            for dimension in ("prompts", "tool_descriptions", "coordination_patterns"):
                role_marker = f"advisor specialist for the '{dimension}'"
                if role_marker in system_prompt and dimension not in dimensions_seen:
                    dimensions_seen.append(dimension)
                    return by_dimension[dimension]
            raise AssertionError(
                f"Could not route specialist call to a dimension; system prompt preview: {system_prompt[:160]}"
            )

        return router

    # We can't know the mock client until caller constructs it — return
    # the routers plus a holder that the caller patches in.
    _router = _route  # unused placeholder to silence linter
    del _router
    holder: list[MockLLMClient] = []
    router = _make_router(holder)
    return holder, [router, router, router]  # type: ignore[return-value]


def _build_mock(
    prompts_proposals: list[Proposal],
    tool_desc_proposals: list[Proposal],
    coord_proposals: list[Proposal],
) -> MockLLMClient:
    """Build a mock that routes each specialist call by target dimension."""
    holder, routers = _per_specialist_script(prompts_proposals, tool_desc_proposals, coord_proposals)
    mock = MockLLMClient(routers)
    holder.append(mock)
    return mock


def _custom_rubric_dir(tmp_path: Path) -> Path:
    """Write a single custom rubric targeting ``prompts`` into ``tmp_path``."""
    custom_dir = tmp_path / "custom-rubrics"
    custom_dir.mkdir()
    (custom_dir / "prompts-custom-rule.md").write_text(
        "---\n"
        "id: prompts-custom-rule\n"
        "severity: observation\n"
        "category: prompts\n"
        "target_dimension: prompts\n"
        "---\n\n"
        "# Custom rule\n\n"
        "Body for the custom rubric.\n",
        encoding="utf-8",
    )
    return custom_dir


# --- Tests ----------------------------------------------------------------


class TestAnalyzeReturnsAdvisorReport:
    async def test_returns_report_with_all_dimensions(self) -> None:
        prompts, tool_desc, coord = _builtin_per_dimension_proposals()
        client = _build_mock([prompts], [tool_desc], [coord])

        report = await analyze(MINIMAL_TRACE, llm_client=client)

        assert isinstance(report, AdvisorReport)
        assert report.trace_id == "advisor-core-phase2-fixture"
        assert report.target_dimensions_analyzed == [
            "prompts",
            "tool_descriptions",
            "coordination_patterns",
        ]
        assert [p.rubric_id for p in report.proposals] == [
            "prompts-missing-exit-criteria",
            "tool-descriptions-missing-docstring",
            "coordination-patterns-no-termination",
        ]

    async def test_usage_is_aggregated(self) -> None:
        prompts, tool_desc, coord = _builtin_per_dimension_proposals()
        client = _build_mock([prompts], [tool_desc], [coord])

        report = await analyze(MINIMAL_TRACE, llm_client=client)

        # Three specialist LLM calls, each at 3 input / 7 output.
        assert report.usage.input_tokens == 3 * 3
        assert report.usage.output_tokens == 3 * 7

    async def test_generated_at_is_utc(self) -> None:
        prompts, tool_desc, coord = _builtin_per_dimension_proposals()
        client = _build_mock([prompts], [tool_desc], [coord])
        before = datetime.now(UTC)

        report = await analyze(MINIMAL_TRACE, llm_client=client)

        after = datetime.now(UTC)
        assert before <= report.generated_at <= after
        assert report.generated_at.tzinfo is not None

    async def test_rubric_counts_reflect_sources(self) -> None:
        prompts, tool_desc, coord = _builtin_per_dimension_proposals()
        client = _build_mock([prompts], [tool_desc], [coord])

        report = await analyze(MINIMAL_TRACE, llm_client=client)

        assert report.rubric_counts == {RubricSource.BUILTIN: 3}


class TestAnalyzeAcceptsPreLoadedEvents:
    async def test_list_input_skips_adapter(self) -> None:
        events = load_trace(MINIMAL_TRACE)
        prompts, tool_desc, coord = _builtin_per_dimension_proposals()
        client = _build_mock([prompts], [tool_desc], [coord])

        report = await analyze(events, llm_client=client)

        assert report.trace_id == events[0].trace_id

    async def test_list_input_accepts_dumped_dicts(self) -> None:
        events = load_trace(MINIMAL_TRACE)
        dumped = [event.model_dump(mode="json") for event in events]
        prompts, tool_desc, coord = _builtin_per_dimension_proposals()
        client = _build_mock([prompts], [tool_desc], [coord])

        report = await analyze(dumped, llm_client=client)  # type: ignore[arg-type]

        assert len(report.proposals) == 3

    async def test_empty_list_raises(self) -> None:
        client = MockLLMClient([])
        with pytest.raises(ValueError, match="at least one event"):
            await analyze([], llm_client=client)


class TestAnalyzeRubricLoading:
    async def test_custom_rubrics_are_merged(self, tmp_path: Path) -> None:
        custom_dir = _custom_rubric_dir(tmp_path)
        builtin_prompt = _proposal(
            "prompts-missing-exit-criteria",
            severity=ProposalSeverity.CRITICAL,
            ranking_score=0.9,
        )
        custom_prompt = _proposal(
            "prompts-custom-rule",
            severity=ProposalSeverity.OBSERVATION,
            source=RubricSource.CUSTOM,
            ranking_score=0.3,
        )
        _, tool_desc, coord = _builtin_per_dimension_proposals()
        client = _build_mock([builtin_prompt, custom_prompt], [tool_desc], [coord])

        report = await analyze(
            MINIMAL_TRACE,
            llm_client=client,
            rubrics=[custom_dir],
        )

        assert report.rubric_counts == {RubricSource.BUILTIN: 3, RubricSource.CUSTOM: 1}
        custom_proposals = [p for p in report.proposals if p.rubric_source == RubricSource.CUSTOM]
        assert [p.rubric_id for p in custom_proposals] == ["prompts-custom-rule"]

    async def test_rubrics_none_loads_builtins_only(self) -> None:
        prompts, tool_desc, coord = _builtin_per_dimension_proposals()
        client = _build_mock([prompts], [tool_desc], [coord])

        report = await analyze(MINIMAL_TRACE, llm_client=client, rubrics=None)

        assert all(p.rubric_source == RubricSource.BUILTIN for p in report.proposals)
        assert all(p.rubric_id.split("-", 1)[0] in {"prompts", "tool", "coordination"} for p in report.proposals)


class TestAnalyzeEmitterHandling:
    async def test_caller_supplied_emitter_receives_events(self) -> None:
        prompts, tool_desc, coord = _builtin_per_dimension_proposals()
        client = _build_mock([prompts], [tool_desc], [coord])
        external_emitter = InMemoryEmitter(trace_id="external-trace")

        await analyze(MINIMAL_TRACE, llm_client=client, emitter=external_emitter)

        assert external_emitter.events, "external emitter captured no events"

    async def test_no_emitter_constructs_internal_one(self) -> None:
        prompts, tool_desc, coord = _builtin_per_dimension_proposals()
        client = _build_mock([prompts], [tool_desc], [coord])

        report = await analyze(MINIMAL_TRACE, llm_client=client)

        assert report.usage.input_tokens > 0


class TestWriteReport:
    async def test_writes_both_formats(self, tmp_path: Path) -> None:
        prompts, tool_desc, coord = _builtin_per_dimension_proposals()
        client = _build_mock([prompts], [tool_desc], [coord])
        report = await analyze(MINIMAL_TRACE, llm_client=client)

        json_path = tmp_path / "report.json"
        md_path = tmp_path / "report.md"
        write_report(report, json_path=json_path, markdown_path=md_path)

        assert json_path.is_file()
        assert md_path.is_file()
        reconstructed = AdvisorReport.model_validate_json(json_path.read_text(encoding="utf-8"))
        assert reconstructed == report
        assert "# Advisor Report" in md_path.read_text(encoding="utf-8")

    async def test_writes_only_json_when_markdown_omitted(self, tmp_path: Path) -> None:
        prompts, tool_desc, coord = _builtin_per_dimension_proposals()
        client = _build_mock([prompts], [tool_desc], [coord])
        report = await analyze(MINIMAL_TRACE, llm_client=client)

        json_path = tmp_path / "report.json"
        write_report(report, json_path=json_path, markdown_path=None)

        assert json_path.is_file()
        assert not (tmp_path / "report.md").exists()

    async def test_writes_only_markdown_when_json_omitted(self, tmp_path: Path) -> None:
        prompts, tool_desc, coord = _builtin_per_dimension_proposals()
        client = _build_mock([prompts], [tool_desc], [coord])
        report = await analyze(MINIMAL_TRACE, llm_client=client)

        md_path = tmp_path / "report.md"
        write_report(report, json_path=None, markdown_path=md_path)

        assert md_path.is_file()
        assert not (tmp_path / "report.json").exists()

    async def test_write_is_atomic_no_stale_tmp_files(self, tmp_path: Path) -> None:
        prompts, tool_desc, coord = _builtin_per_dimension_proposals()
        client = _build_mock([prompts], [tool_desc], [coord])
        report = await analyze(MINIMAL_TRACE, llm_client=client)

        write_report(
            report,
            json_path=tmp_path / "report.json",
            markdown_path=tmp_path / "report.md",
        )

        leftovers = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
        assert leftovers == []

    async def test_write_creates_parent_directory(self, tmp_path: Path) -> None:
        prompts, tool_desc, coord = _builtin_per_dimension_proposals()
        client = _build_mock([prompts], [tool_desc], [coord])
        report = await analyze(MINIMAL_TRACE, llm_client=client)

        nested = tmp_path / "a" / "b" / "c"
        write_report(report, json_path=nested / "report.json")

        assert (nested / "report.json").is_file()

    async def test_write_report_noop_when_both_paths_none(self, tmp_path: Path) -> None:
        prompts, tool_desc, coord = _builtin_per_dimension_proposals()
        client = _build_mock([prompts], [tool_desc], [coord])
        report = await analyze(MINIMAL_TRACE, llm_client=client)

        write_report(report, json_path=None, markdown_path=None)

        assert list(tmp_path.iterdir()) == []


class TestEveryBuiltinRubricIsExercisedByMockFixture:
    async def test_every_builtin_rubric_id_appears_in_some_analyze_call(self) -> None:
        builtins = [r for r in load_rubrics() if r.source == RubricSource.BUILTIN]
        assert builtins, "expected shipping builtin rubrics"

        per_dim: dict[str, list] = {}
        for r in builtins:
            per_dim.setdefault(r.target_dimension, []).append(r)

        fired_ids: set[str] = set()
        for dim_rubrics in per_dim.values():
            scripted: dict[str, list[Proposal]] = {
                "prompts": [],
                "tool_descriptions": [],
                "coordination_patterns": [],
            }
            for rubric in dim_rubrics:
                scripted[rubric.target_dimension].append(
                    _proposal(
                        rubric.id,
                        severity=rubric.severity,
                        category=ProposalCategory(rubric.category.value),
                        target_dimension=rubric.target_dimension,
                        source=RubricSource.BUILTIN,
                    )
                )
            client = _build_mock(
                scripted["prompts"],
                scripted["tool_descriptions"],
                scripted["coordination_patterns"],
            )
            report = await analyze(MINIMAL_TRACE, llm_client=client)
            fired_ids.update(p.rubric_id for p in report.proposals)

        expected = {r.id for r in builtins}
        assert expected <= fired_ids, f"rubrics not covered: {expected - fired_ids}"


class TestTraceIdNormalization:
    async def test_path_derives_trace_id_from_envelope(self) -> None:
        prompts, tool_desc, coord = _builtin_per_dimension_proposals()
        client = _build_mock([prompts], [tool_desc], [coord])
        payload = json.loads(MINIMAL_TRACE.read_text(encoding="utf-8"))

        report = await analyze(MINIMAL_TRACE, llm_client=client)

        assert report.trace_id == payload["trace_id"]


def test_minimal_fixture_matches_canonical_envelope() -> None:
    events = load_trace(MINIMAL_TRACE)
    assert events, "fixture produced no events"
    payload = json.loads(MINIMAL_TRACE.read_text(encoding="utf-8"))
    assert set(payload) >= {"trace_id", "events"}
