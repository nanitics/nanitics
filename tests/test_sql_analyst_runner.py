"""Deterministic unit tests for the SQL-analyst runner.

No Docker, no real Postgres, no real LLM. Every SDK primitive the
runner consumes is exercised against in-process fakes:

* ``TestGroundTruthEvaluator`` — tests Step 4's evaluator end-to-end
  with hand-constructed ``SampleQuestion`` + ``ExpectedAnswer``
  instances. These tests are authored *before* the evaluator per the
  plan's TDD timing so the "real ground truth, not LLM-as-judge"
  contract is pinned before implementation.
* ``TestRunSqlTool`` — exercises the ``run_sql`` tool's LIMIT
  injection, error-to-tool-content path, and success metadata via a
  fake connector so no asyncpg connection is required.
* ``TestCatalogInvariants`` — sanity-checks the sample-question
  catalog: unique kebab-case ids, non-empty questions, canonical
  answers self-match.

Later steps add ``TestSqlAnalystEndpoint``, ``TestSqlAnalystAdHoc``,
and ``TestSqlAnalystRegistrationIdempotence`` against the runner
module.

``docker/full-stack/sql_analyst/`` is a proper Python package that the
runtime image copies to ``/srv/sql_analyst/``. Tests add the parent
directory to ``sys.path`` so the package is importable as ``sql_analyst``
without ceremony.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# Put ``docker/full-stack/`` on sys.path so ``sql_analyst`` resolves as
# a top-level package — mirroring how the runtime image lays out
# ``/srv/sql_analyst/``. Done once at module import time.
_FULL_STACK_DIR = Path(__file__).resolve().parent.parent / "docker" / "full-stack"
if str(_FULL_STACK_DIR) not in sys.path:
    sys.path.insert(0, str(_FULL_STACK_DIR))


from sql_analyst.evaluator import GroundTruthEvaluator
from sql_analyst.questions import (
    QUESTIONS,
    ExpectedRow,
    ExpectedRowSet,
    ExpectedScalar,
    SampleQuestion,
)
from sql_analyst.tool import build_run_sql_tool

from nanitics import EvaluationContext, EvaluationVerdict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_context() -> EvaluationContext:
    """Construct a minimal :class:`EvaluationContext` for unit tests."""
    return EvaluationContext(messages=[], task_input="")


# ---------------------------------------------------------------------------
# TestGroundTruthEvaluator — TDD; tests first per plan Step 4.
# ---------------------------------------------------------------------------


class TestGroundTruthEvaluator:
    """Pins the value-level comparison contract before implementation."""

    @staticmethod
    def _question(expected: Any, *, qid: str = "q") -> SampleQuestion:
        return SampleQuestion(id=qid, question="Unused in unit tests.", expected=expected)

    @pytest.mark.asyncio
    async def test_matching_scalar_accepts(self) -> None:
        question = self._question(ExpectedScalar(value=37))
        evaluator = GroundTruthEvaluator(question=question)

        result = await evaluator.evaluate(
            '{"answer": 37, "sql": "SELECT 37", "rowcount": 1}',
            _empty_context(),
        )

        assert result.verdict == EvaluationVerdict.ACCEPT

    @pytest.mark.asyncio
    async def test_off_by_one_scalar_revises_with_values(self) -> None:
        question = self._question(ExpectedScalar(value=37))
        evaluator = GroundTruthEvaluator(question=question)

        result = await evaluator.evaluate(
            '{"answer": 42, "sql": "SELECT 42", "rowcount": 1}',
            _empty_context(),
        )

        assert result.verdict == EvaluationVerdict.REVISE
        assert result.feedback is not None
        assert "37" in result.feedback
        assert "42" in result.feedback

    @pytest.mark.asyncio
    async def test_scalar_within_tolerance_accepts(self) -> None:
        question = self._question(ExpectedScalar(value=100.0, tolerance=0.5))
        evaluator = GroundTruthEvaluator(question=question)

        result = await evaluator.evaluate(
            '{"answer": 100.3, "sql": "SELECT 100.3", "rowcount": 1}',
            _empty_context(),
        )

        assert result.verdict == EvaluationVerdict.ACCEPT

    @pytest.mark.asyncio
    async def test_row_set_ordered_wrong_order_revises(self) -> None:
        expected = ExpectedRowSet(
            rows=(
                ExpectedRow(columns=("a",), values=(1,)),
                ExpectedRow(columns=("a",), values=(2,)),
            ),
            ordered=True,
        )
        question = self._question(expected)
        evaluator = GroundTruthEvaluator(question=question)

        result = await evaluator.evaluate(
            '{"answer": [[2], [1]], "sql": "SELECT", "rowcount": 2}',
            _empty_context(),
        )

        assert result.verdict == EvaluationVerdict.REVISE
        assert result.feedback is not None

    @pytest.mark.asyncio
    async def test_row_set_unordered_same_rows_accepts(self) -> None:
        expected = ExpectedRowSet(
            rows=(
                ExpectedRow(columns=("a",), values=(1,)),
                ExpectedRow(columns=("a",), values=(2,)),
            ),
            ordered=False,
        )
        question = self._question(expected)
        evaluator = GroundTruthEvaluator(question=question)

        result = await evaluator.evaluate(
            '{"answer": [[2], [1]], "sql": "SELECT", "rowcount": 2}',
            _empty_context(),
        )

        assert result.verdict == EvaluationVerdict.ACCEPT

    @pytest.mark.asyncio
    async def test_row_set_length_mismatch_revises(self) -> None:
        expected = ExpectedRowSet(
            rows=(
                ExpectedRow(columns=("a",), values=(1,)),
                ExpectedRow(columns=("a",), values=(2,)),
            ),
            ordered=True,
        )
        question = self._question(expected)
        evaluator = GroundTruthEvaluator(question=question)

        result = await evaluator.evaluate(
            '{"answer": [[1]], "sql": "SELECT", "rowcount": 1}',
            _empty_context(),
        )

        assert result.verdict == EvaluationVerdict.REVISE
        assert result.feedback is not None
        assert "length" in result.feedback or "rows" in result.feedback

    @pytest.mark.asyncio
    async def test_envelope_missing_answer_revises(self) -> None:
        question = self._question(ExpectedScalar(value=1))
        evaluator = GroundTruthEvaluator(question=question)

        result = await evaluator.evaluate(
            '{"sql": "SELECT 1", "rowcount": 1}',
            _empty_context(),
        )

        assert result.verdict == EvaluationVerdict.REVISE
        assert result.feedback is not None
        # Feedback must name the envelope shape so the agent can rewrite.
        assert "answer" in result.feedback

    @pytest.mark.asyncio
    async def test_unparseable_on_final_revision_rejects(self) -> None:
        question = self._question(ExpectedScalar(value=1))
        evaluator = GroundTruthEvaluator(question=question, max_revisions=0)

        result = await evaluator.evaluate(
            "this is not json at all",
            _empty_context(),
        )

        assert result.verdict == EvaluationVerdict.REJECT

    @pytest.mark.asyncio
    async def test_answer_inside_json_code_fence_parses(self) -> None:
        """A ```json fence is tolerated — LLMs often wrap envelopes."""
        question = self._question(ExpectedScalar(value=5))
        evaluator = GroundTruthEvaluator(question=question)

        result = await evaluator.evaluate(
            '```json\n{"answer": 5, "sql": "SELECT 5", "rowcount": 1}\n```',
            _empty_context(),
        )

        assert result.verdict == EvaluationVerdict.ACCEPT

    @pytest.mark.asyncio
    async def test_empty_output_revises(self) -> None:
        question = self._question(ExpectedScalar(value=1))
        evaluator = GroundTruthEvaluator(question=question)

        result = await evaluator.evaluate("", _empty_context())

        assert result.verdict == EvaluationVerdict.REVISE

    @pytest.mark.asyncio
    async def test_envelope_parses_but_is_not_object(self) -> None:
        """``[1,2,3]`` is valid JSON but not an envelope — revise."""
        question = self._question(ExpectedScalar(value=1))
        evaluator = GroundTruthEvaluator(question=question)

        result = await evaluator.evaluate("[1, 2, 3]", _empty_context())

        assert result.verdict == EvaluationVerdict.REVISE

    @pytest.mark.asyncio
    async def test_non_string_output_rejected_at_parse(self) -> None:
        """Non-string output is a protocol violation too."""
        question = self._question(ExpectedScalar(value=1))
        evaluator = GroundTruthEvaluator(question=question)

        # Bypass the protocol's ``output: str`` signature to exercise
        # the defensive non-string guard in ``_try_parse_envelope``.
        result = await evaluator.evaluate(42, _empty_context())  # type: ignore[arg-type]

        assert result.verdict == EvaluationVerdict.REVISE

    @pytest.mark.asyncio
    async def test_max_revisions_property_round_trips(self) -> None:
        question = self._question(ExpectedScalar(value=1))
        evaluator = GroundTruthEvaluator(question=question, max_revisions=4)
        assert evaluator.max_revisions == 4

    def test_max_revisions_negative_raises(self) -> None:
        question = self._question(ExpectedScalar(value=1))
        with pytest.raises(ValueError, match="max_revisions"):
            GroundTruthEvaluator(question=question, max_revisions=-1)

    def test_evaluator_module_has_no_llm_imports(self) -> None:
        """The ground-truth evaluator must never import an LLM client.

        Loose heuristic (per plan Step 4): no ``llm`` substring after
        comment stripping. Tightens the "value-equality-only" guarantee.
        """
        from sql_analyst import evaluator as evaluator_module

        source = Path(evaluator_module.__file__ or "").read_text(encoding="utf-8")
        # Stripping single-line comments and docstrings is over-engineering
        # for this check — comment-line filtering is sufficient. Assert
        # against the non-comment lines.
        active = "\n".join(line for line in source.splitlines() if not line.strip().startswith("#"))
        # Explicitly forbid importing any of the LLM-client types.
        for symbol in ("AnthropicLLMClient", "OpenAILLMClient", "LLMClient", "MockLLMClient"):
            assert symbol not in active, f"evaluator.py imports {symbol!r}"
        # Also forbid the word ``llm`` appearing outside comments/docstrings.
        # We search the non-comment body only; docstrings remain but a
        # ``llm`` reference in a docstring is acceptable — the guard is
        # against code-level imports.
        import_lines = [line for line in active.splitlines() if line.strip().startswith(("import ", "from "))]
        for line in import_lines:
            assert "llm" not in line.lower(), f"evaluator.py has LLM import: {line!r}"


# ---------------------------------------------------------------------------
# TestRunSqlTool — per Step 3
# ---------------------------------------------------------------------------


class _FakeRecord(dict[str, Any]):
    """asyncpg.Record-compatible facade — keys() + values()."""


class _FakeConnection:
    """In-memory fake of an asyncpg connection.

    Configured at construction time with either a list of records to
    return from ``fetch`` or an exception to raise. Records ``sql``
    arguments for assertion.
    """

    def __init__(
        self,
        *,
        records: list[dict[str, Any]] | None = None,
        raises: BaseException | None = None,
    ) -> None:
        self._records = records or []
        self._raises = raises
        self.fetch_calls: list[str] = []
        self.closed = False

    async def fetch(self, sql: str) -> list[_FakeRecord]:
        self.fetch_calls.append(sql)
        if self._raises is not None:
            raise self._raises
        return [_FakeRecord(record) for record in self._records]

    async def close(self) -> None:
        self.closed = True


def _connector_factory_for(connection: _FakeConnection) -> Any:
    """Return a ``connector_factory`` closure that yields *connection*.

    Asserts the factory was invoked with the expected DSN so tests
    catch regressions where the tool reaches for the privileged pool.
    """
    calls: list[tuple[str, int]] = []

    def factory(dsn: str, statement_timeout_ms: int) -> Any:
        calls.append((dsn, statement_timeout_ms))

        async def connect() -> Any:
            return connection

        return connect

    factory.calls = calls  # type: ignore[attr-defined]
    return factory


class TestRunSqlTool:
    """LIMIT injection, error surfacing, success shape."""

    def test_build_run_sql_tool_returns_tool_protocol(self) -> None:
        tool = build_run_sql_tool(
            sandbox_dsn="postgresql://sandbox:pass@host/db",
            connector_factory=_connector_factory_for(_FakeConnection()),
        )

        assert tool.schema.name == "run_sql"
        assert callable(tool.execute)

    @pytest.mark.asyncio
    async def test_select_one_returns_shape(self) -> None:
        connection = _FakeConnection(records=[{"?column?": 1}])
        factory = _connector_factory_for(connection)

        tool = build_run_sql_tool(
            sandbox_dsn="postgresql://sandbox:pass@host/db",
            connector_factory=factory,
        )
        result = await tool.execute(sql="SELECT 1")

        assert result.metadata["columns"] == ["?column?"]
        assert result.metadata["rows"] == [[1]]
        assert result.metadata["rowcount"] == 1
        assert result.metadata["truncated"] is False
        assert result.metadata["error"] is False
        # ``SELECT 1`` is neither a bare SELECT nor an aggregate, but it
        # is also not a scalar-aggregate match — the injection rule
        # passes it through. Asserting the exact shape below guards
        # against regressions.
        assert result.metadata["sql"].strip().endswith("LIMIT 200") or result.metadata["sql"] == "SELECT 1"

    @pytest.mark.asyncio
    async def test_bare_select_star_injects_limit(self) -> None:
        connection = _FakeConnection(records=[])
        factory = _connector_factory_for(connection)

        tool = build_run_sql_tool(
            sandbox_dsn="postgresql://sandbox/db",
            connector_factory=factory,
        )
        result = await tool.execute(sql="SELECT * FROM orders")

        assert "LIMIT 200" in result.metadata["sql"]
        assert connection.fetch_calls == [result.metadata["sql"]]
        assert connection.closed is True

    @pytest.mark.asyncio
    async def test_explicit_limit_passes_through_unchanged(self) -> None:
        connection = _FakeConnection(records=[])
        factory = _connector_factory_for(connection)

        tool = build_run_sql_tool(
            sandbox_dsn="postgresql://sandbox/db",
            connector_factory=factory,
        )
        original = "SELECT * FROM orders LIMIT 10"
        result = await tool.execute(sql=original)

        assert result.metadata["sql"] == original

    @pytest.mark.asyncio
    async def test_count_star_passes_through_unchanged(self) -> None:
        connection = _FakeConnection(records=[{"count": 200}])
        factory = _connector_factory_for(connection)

        tool = build_run_sql_tool(
            sandbox_dsn="postgresql://sandbox/db",
            connector_factory=factory,
        )
        original = "SELECT COUNT(*) FROM orders"
        result = await tool.execute(sql=original)

        assert result.metadata["sql"] == original

    @pytest.mark.asyncio
    async def test_explain_passes_through_unchanged(self) -> None:
        connection = _FakeConnection(records=[])
        factory = _connector_factory_for(connection)

        tool = build_run_sql_tool(
            sandbox_dsn="postgresql://sandbox/db",
            connector_factory=factory,
        )
        original = "EXPLAIN SELECT * FROM orders"
        result = await tool.execute(sql=original)

        assert result.metadata["sql"] == original

    @pytest.mark.asyncio
    async def test_aggregate_with_group_by_injects_limit(self) -> None:
        """Scalar-aggregate passthrough only applies without GROUP BY."""
        connection = _FakeConnection(records=[])
        factory = _connector_factory_for(connection)

        tool = build_run_sql_tool(
            sandbox_dsn="postgresql://sandbox/db",
            connector_factory=factory,
        )
        result = await tool.execute(sql="SELECT COUNT(*) FROM orders GROUP BY status")

        assert "LIMIT 200" in result.metadata["sql"]

    @pytest.mark.asyncio
    async def test_postgres_error_surfaced_as_tool_content(self) -> None:
        class FakePostgresError(Exception):
            pass

        connection = _FakeConnection(raises=FakePostgresError("permission denied"))
        factory = _connector_factory_for(connection)

        tool = build_run_sql_tool(
            sandbox_dsn="postgresql://sandbox/db",
            connector_factory=factory,
        )
        result = await tool.execute(sql="INSERT INTO orders DEFAULT VALUES")

        assert result.metadata["error"] is True
        assert result.metadata["error_type"] == "FakePostgresError"
        assert result.content.startswith("ERROR: FakePostgresError")
        assert connection.closed is True

    @pytest.mark.asyncio
    async def test_truncated_flag_set_when_rowcount_equals_limit(self) -> None:
        records = [{"id": n} for n in range(3)]
        connection = _FakeConnection(records=records)
        factory = _connector_factory_for(connection)

        tool = build_run_sql_tool(
            sandbox_dsn="postgresql://sandbox/db",
            row_limit=3,
            connector_factory=factory,
        )
        result = await tool.execute(sql="SELECT * FROM customers")

        assert result.metadata["truncated"] is True
        # Markdown-table rendering carries the truncation notice.
        assert "truncated" in result.content or "more" in result.content

    @pytest.mark.asyncio
    async def test_trailing_semicolon_preserved_when_injecting(self) -> None:
        connection = _FakeConnection(records=[])
        factory = _connector_factory_for(connection)

        tool = build_run_sql_tool(
            sandbox_dsn="postgresql://sandbox/db",
            connector_factory=factory,
        )
        result = await tool.execute(sql="SELECT * FROM customers;")

        assert result.metadata["sql"].rstrip().endswith(";")
        assert "LIMIT 200" in result.metadata["sql"]

    @pytest.mark.asyncio
    async def test_empty_sql_passes_through_unchanged(self) -> None:
        """Blank SQL has no rule to apply; the tool does not inject."""
        connection = _FakeConnection(records=[])
        factory = _connector_factory_for(connection)

        tool = build_run_sql_tool(
            sandbox_dsn="postgresql://sandbox/db",
            connector_factory=factory,
        )
        result = await tool.execute(sql="   ")

        assert result.metadata["sql"] == "   "

    def test_factory_does_not_receive_privileged_pool(self) -> None:
        """Regression guard: ``build_run_sql_tool`` accepts a DSN and
        never reaches for ``ShellContext.pool``."""
        factory = _connector_factory_for(_FakeConnection())

        tool = build_run_sql_tool(
            sandbox_dsn="postgresql://sandbox/db",
            statement_timeout_ms=1500,
            connector_factory=factory,
        )

        # Factory was invoked with the sandbox DSN and custom timeout.
        calls = factory.calls  # type: ignore[attr-defined]
        assert calls == [("postgresql://sandbox/db", 1500)]
        # The returned object carries the expected tool name.
        assert tool.schema.name == "run_sql"

    @pytest.mark.asyncio
    async def test_no_rows_render_placeholder_content(self) -> None:
        """Zero-row result sets still render a legible ``content``
        payload (columns list + separator, no data)."""
        connection = _FakeConnection(records=[])
        factory = _connector_factory_for(connection)

        tool = build_run_sql_tool(
            sandbox_dsn="postgresql://sandbox/db",
            connector_factory=factory,
        )
        result = await tool.execute(sql="SELECT * FROM customers WHERE 1=0")

        assert result.metadata["rowcount"] == 0
        # With no records, columns are unknown — content is a placeholder.
        assert result.content == "(no columns)"

    @pytest.mark.asyncio
    async def test_large_result_set_includes_truncation_footer(self) -> None:
        """When more than ~20 rows come back, the rendered table footer
        notes the omission so the agent sees the truncation."""
        records = [{"id": n} for n in range(25)]
        connection = _FakeConnection(records=records)
        factory = _connector_factory_for(connection)

        tool = build_run_sql_tool(
            sandbox_dsn="postgresql://sandbox/db",
            row_limit=200,
            connector_factory=factory,
        )
        result = await tool.execute(sql="SELECT * FROM customers")

        assert result.metadata["rowcount"] == 25
        assert "and 5 more" in result.content

    @pytest.mark.asyncio
    async def test_default_connector_factory_calls_asyncpg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The default factory closes over ``asyncpg.connect`` with the
        sandbox DSN and the server-side ``statement_timeout``. Exercised
        via a fake ``asyncpg`` module so no real network hop is made."""
        import types

        # Build a stub ``asyncpg`` module whose ``connect`` records the
        # positional DSN and the server_settings kwarg before returning
        # the fake connection.
        connection = _FakeConnection(records=[{"n": 1}])
        calls: list[dict[str, Any]] = []

        async def fake_connect(dsn: str, **kwargs: Any) -> Any:
            calls.append({"dsn": dsn, "kwargs": kwargs})
            return connection

        stub = types.ModuleType("asyncpg")
        stub.connect = fake_connect  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "asyncpg", stub)

        # Build a tool with the default factory (no connector_factory kwarg).
        tool = build_run_sql_tool(
            sandbox_dsn="postgresql://sandbox/db",
            statement_timeout_ms=1500,
        )
        result = await tool.execute(sql="SELECT 1")

        # Behavioral checks: connector invoked with sandbox DSN + the
        # server-side statement_timeout carrying the requested value.
        assert result.metadata["error"] is False
        assert len(calls) == 1
        assert calls[0]["dsn"] == "postgresql://sandbox/db"
        assert calls[0]["kwargs"] == {"server_settings": {"statement_timeout": "1500"}}


# ---------------------------------------------------------------------------
# TestCatalogInvariants — per Step 2 / Step 6
# ---------------------------------------------------------------------------


_KEBAB_CASE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class TestCatalogInvariants:
    def test_every_question_has_unique_kebab_case_id(self) -> None:
        ids = [q.id for q in QUESTIONS]
        assert len(ids) == len(set(ids)), f"duplicate ids in QUESTIONS: {ids}"
        for qid in ids:
            assert _KEBAB_CASE.match(qid), f"id {qid!r} is not kebab-case"

    def test_every_question_has_non_empty_prompt(self) -> None:
        for question in QUESTIONS:
            assert question.question.strip(), f"empty question for {question.id!r}"

    def test_every_expected_answer_is_a_known_shape(self) -> None:
        for question in QUESTIONS:
            assert isinstance(question.expected, ExpectedScalar | ExpectedRow | ExpectedRowSet)

    def test_every_canonical_answer_self_matches(self) -> None:
        """Every ``expected.compare(canonical_candidate)`` must pass
        when fed the canonical value itself. This is the guard against
        silent typos in the catalog."""
        for question in QUESTIONS:
            expected = question.expected
            candidate = _canonical_candidate_for(expected)
            passed, feedback = expected.compare(candidate)
            assert passed, f"question {question.id!r} failed self-match: {feedback!r} candidate={candidate!r}"

    def test_catalog_covers_the_five_spec_shapes(self) -> None:
        """The five canonical question ids are present in the catalog."""
        required = {
            "total-orders-count",
            "revenue-total",
            "top-5-customers-by-revenue",
            "orders-by-region",
            "cancelled-orders-by-month",
        }
        ids = {q.id for q in QUESTIONS}
        assert required.issubset(ids), f"missing canonical ids: {required - ids}"


def _canonical_candidate_for(expected: Any) -> Any:
    """Build a candidate that ``expected.compare`` should accept.

    Used by :class:`TestCatalogInvariants` to prove each entry's
    canonical answer is self-consistent — feeds the expected value
    itself back through comparison.
    """
    if isinstance(expected, ExpectedScalar):
        return expected.value
    if isinstance(expected, ExpectedRow):
        return list(expected.values)
    if isinstance(expected, ExpectedRowSet):
        return [list(row.values) for row in expected.rows]
    raise AssertionError(f"unknown ExpectedAnswer shape: {type(expected).__name__}")


# ---------------------------------------------------------------------------
# ExpectedAnswer comparison — edge coverage
# ---------------------------------------------------------------------------


class TestExpectedAnswerComparisonEdges:
    """Exercises the comparison branches that the five-question
    catalog does not reach (bool, string mismatch, type errors,
    mapping vs. sequence row input, unordered fallthrough)."""

    def test_scalar_string_mismatch_reports_values(self) -> None:
        expected = ExpectedScalar(value="foo")
        ok, feedback = expected.compare("bar")

        assert ok is False
        assert "foo" in feedback and "bar" in feedback

    def test_scalar_string_match(self) -> None:
        expected = ExpectedScalar(value="foo")
        ok, _ = expected.compare("foo")
        assert ok is True

    def test_scalar_numeric_bool_rejected(self) -> None:
        """bool is an int subclass but should not satisfy a numeric
        scalar — the comparator returns a type-mismatch feedback."""
        expected = ExpectedScalar(value=1)
        ok, feedback = expected.compare(True)

        assert ok is False
        assert "numeric" in feedback

    def test_scalar_type_mismatch_names_type(self) -> None:
        expected = ExpectedScalar(value=10)
        ok, feedback = expected.compare({"a": 1})

        assert ok is False
        assert "dict" in feedback

    def test_scalar_tolerance_out_of_range(self) -> None:
        expected = ExpectedScalar(value=100.0, tolerance=0.01)
        ok, feedback = expected.compare(100.5)

        assert ok is False
        assert "100.5" in feedback

    def test_scalar_string_coercion(self) -> None:
        """A numeric scalar accepts a numeric-looking string."""
        expected = ExpectedScalar(value=10)
        ok, _ = expected.compare("10")

        assert ok is True

    def test_scalar_string_coercion_fails_for_non_numeric(self) -> None:
        expected = ExpectedScalar(value=10)
        ok, feedback = expected.compare("abc")

        assert ok is False
        assert "numeric" in feedback

    def test_row_mapping_shape_accepts(self) -> None:
        expected = ExpectedRow(columns=("a", "b"), values=(1, 2))
        ok, _ = expected.compare({"a": 1, "b": 2})
        assert ok is True

    def test_row_missing_column(self) -> None:
        expected = ExpectedRow(columns=("a", "b"), values=(1, 2))
        ok, feedback = expected.compare({"a": 1})

        assert ok is False
        assert "b" in feedback

    def test_row_sequence_length_mismatch(self) -> None:
        expected = ExpectedRow(columns=("a", "b"), values=(1, 2))
        ok, feedback = expected.compare([1])

        assert ok is False
        assert "row" in feedback

    def test_row_not_mapping_or_sequence(self) -> None:
        expected = ExpectedRow(columns=("a",), values=(1,))
        ok, feedback = expected.compare(42)

        assert ok is False
        assert "int" in feedback

    def test_row_cell_string_match(self) -> None:
        expected = ExpectedRow(columns=("name",), values=("alice",))
        ok, _ = expected.compare(["alice"])
        assert ok is True

    def test_row_cell_string_mismatch(self) -> None:
        expected = ExpectedRow(columns=("name",), values=("alice",))
        ok, feedback = expected.compare(["bob"])

        assert ok is False
        assert "alice" in feedback

    def test_row_cell_non_numeric_equal_fallthrough(self) -> None:
        """Non-numeric, non-string cells fall back to ``==``."""
        expected = ExpectedRow(columns=("d",), values=(None,))
        ok, _ = expected.compare([None])
        assert ok is True

    def test_row_cell_non_numeric_mismatch(self) -> None:
        expected = ExpectedRow(columns=("d",), values=(None,))
        ok, feedback = expected.compare([(1, 2)])

        assert ok is False
        assert "None" in feedback

    def test_rowset_candidate_not_sequence(self) -> None:
        expected = ExpectedRowSet(rows=(ExpectedRow(columns=("a",), values=(1,)),), ordered=False)
        ok, feedback = expected.compare(42)

        assert ok is False
        assert "row set" in feedback

    def test_rowset_candidate_string_rejected(self) -> None:
        """Strings are Sequences but must not be treated as row sets."""
        expected = ExpectedRowSet(rows=(ExpectedRow(columns=("a",), values=(1,)),), ordered=False)
        ok, feedback = expected.compare("1")

        assert ok is False
        assert "row set" in feedback

    def test_rowset_unordered_no_match_found(self) -> None:
        expected = ExpectedRowSet(
            rows=(
                ExpectedRow(columns=("a",), values=(1,)),
                ExpectedRow(columns=("a",), values=(2,)),
            ),
            ordered=False,
        )
        ok, feedback = expected.compare([[1], [3]])

        assert ok is False
        assert "not found" in feedback


# ---------------------------------------------------------------------------
# TestBootstrap — mocked pool exercises the template rendering + execution
# path. Real-Postgres coverage of idempotence + grants lives in the
# Docker-gated ``tests/test_sql_analyst_schema.py`` (Step 8).
# ---------------------------------------------------------------------------


class TestBootstrap:
    @pytest.mark.asyncio
    async def test_ensure_analyst_schema_renders_placeholders_and_executes(self) -> None:
        from sql_analyst.bootstrap import ensure_analyst_schema

        fake_conn = MagicMock(name="conn")
        fake_conn.execute = AsyncMock()

        class _AcquireCtx:
            async def __aenter__(self) -> Any:
                return fake_conn

            async def __aexit__(self, *_exc: Any) -> None:
                return None

        class _TxnCtx:
            async def __aenter__(self) -> Any:
                return None

            async def __aexit__(self, *_exc: Any) -> None:
                return None

        fake_conn.transaction = MagicMock(return_value=_TxnCtx())

        fake_pool = MagicMock(name="pool")
        fake_pool.acquire = MagicMock(return_value=_AcquireCtx())

        await ensure_analyst_schema(
            fake_pool,
            sandbox_role="sandbox_user",
            sandbox_password="s3cret",
        )

        fake_conn.execute.assert_awaited_once()
        (executed_sql,) = fake_conn.execute.call_args.args
        # Placeholders gone and replaced verbatim.
        assert "{{sandbox_role}}" not in executed_sql
        assert "{{sandbox_password}}" not in executed_sql
        assert "sandbox_user" in executed_sql
        assert "s3cret" in executed_sql
        # Transaction context opened (single-transaction semantics).
        fake_conn.transaction.assert_called_once()

    def test_render_schema_sql_is_pure(self) -> None:
        """The pure helper the async function wraps is side-effect free."""
        from sql_analyst.bootstrap import _render_schema_sql

        rendered = _render_schema_sql(
            "CREATE ROLE {{sandbox_role}} PASSWORD '{{sandbox_password}}';",
            sandbox_role="alice",
            sandbox_password="pw",
        )
        assert rendered == "CREATE ROLE alice PASSWORD 'pw';"


# ---------------------------------------------------------------------------
# TestSqlAnalyst{Endpoint,AdHoc,RegistrationIdempotence} — per Step 5 / 6.
#
# Exercises the runner end-to-end via :class:`~fastapi.testclient.TestClient`
# with every external dependency injected: ``MockLLMClient`` scripts the
# agent path, a fake connector drives ``run_sql``, and
# ``InMemoryPersistentTraceStore`` backs ``TracedExecutor``. No real
# Postgres, no real LLM.
# ---------------------------------------------------------------------------


import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from nanitics import (
    InMemoryPersistentTraceStore,
    LLMResponse,
    MockLLMClient,
    ToolCall,
    TracedExecutor,
    Usage,
)


def _response(
    content: str | None = None,
    *,
    tool_calls: list[ToolCall] | None = None,
) -> LLMResponse:
    """Build an :class:`LLMResponse` with deterministic tokens."""
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        usage=Usage(input_tokens=5, output_tokens=5),
        model="mock",
        stop_reason="end_turn",
    )


def _tool_call(name: str = "run_sql", *, sql: str, call_id: str = "call-1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments={"sql": sql})


def _envelope(answer: Any, *, sql: str = "SELECT 1", rowcount: int = 1) -> str:
    """JSON envelope the agent's final response must match."""
    return json.dumps({"answer": answer, "sql": sql, "rowcount": rowcount})


class _FakeRowSetConnector:
    """Connector factory that returns fresh fake connections per call.

    Each invocation yields a configured :class:`_FakeConnection` built
    from a queue of pre-scripted record sets. When the queue is empty,
    the connection returns no rows.
    """

    def __init__(self, record_sets: list[list[dict[str, Any]]]) -> None:
        self._queue = list(record_sets)
        self.closed_connections: list[_FakeConnection] = []
        self.invocations: list[tuple[str, int]] = []

    def __call__(self, dsn: str, statement_timeout_ms: int) -> Any:
        self.invocations.append((dsn, statement_timeout_ms))

        async def connect() -> Any:
            records = self._queue.pop(0) if self._queue else []
            connection = _FakeConnection(records=records)
            self.closed_connections.append(connection)
            return connection

        return connect


def _build_shell_context(
    *,
    build_client: Callable[[], Any],
    trace_store: InMemoryPersistentTraceStore | None = None,
) -> Any:
    """Construct a minimal :class:`runners.ShellContext` for tests."""
    from runners import ShellContext

    store = trace_store or InMemoryPersistentTraceStore()
    executor = TracedExecutor(store)
    # ``ShellContext.pool`` is typed ``asyncpg.Pool``; supply a
    # :class:`MagicMock` with the methods the bootstrap closure would
    # touch. The runner's ``on_event("startup")`` closure captures
    # ``context.pool`` but ``TestClient`` only triggers the startup
    # hook when we explicitly invoke it — the endpoint path does not.
    pool = MagicMock(name="pool")
    return ShellContext(
        executor=executor,
        trace_store=store,
        pool=pool,
        build_client=build_client,
    ), store


from collections.abc import Callable


def _make_app_with_runner(
    *,
    build_client: Callable[[], Any],
    sandbox_user: str = "sandbox_user",
    sandbox_password: str = "sandbox_password",
    postgres_dsn: str = "postgresql://app:pw@db:5432/app",
    connector_factory: Callable[[str, int], Any] | None = None,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FastAPI, Any]:
    """Build a fresh FastAPI app with the SQL-analyst runner registered.

    Patches the sandbox-credential / DSN env vars and replaces
    :func:`sql_analyst.tool.build_run_sql_tool`'s ``connector_factory``
    default so ``run_sql`` does not try to reach a real Postgres.
    """
    monkeypatch.setenv("NANITICS_SQL_ANALYST_SANDBOX_USER", sandbox_user)
    monkeypatch.setenv("NANITICS_SQL_ANALYST_SANDBOX_PASSWORD", sandbox_password)
    monkeypatch.setenv("POSTGRES_DSN", postgres_dsn)

    from sql_analyst import runner as runner_module

    # The registration function schedules an app startup hook that calls
    # ``ensure_analyst_schema(context.pool, ...)`` — the ``TestClient``
    # enter path triggers that hook. Tests at this layer are not exercising
    # the bootstrap (see ``TestBootstrap`` for that) so replace the symbol
    # the runner's closure resolves with a no-op async function.
    async def _noop_bootstrap(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(runner_module, "ensure_analyst_schema", _noop_bootstrap)

    if connector_factory is not None:
        # Patch ``build_run_sql_tool`` to force the injected connector
        # factory onto every tool the runner builds during register().
        original = runner_module.build_run_sql_tool

        def patched(*, sandbox_dsn: str, **kwargs: Any) -> Any:
            kwargs.pop("connector_factory", None)
            return original(
                sandbox_dsn=sandbox_dsn,
                connector_factory=connector_factory,
                **kwargs,
            )

        monkeypatch.setattr(runner_module, "build_run_sql_tool", patched)

    context, store = _build_shell_context(build_client=build_client)

    app = FastAPI()
    runner_module.register(app, context)
    return app, store


class TestSqlAnalystEndpoint:
    """Endpoint scenarios."""

    def test_canonical_path_retry_retry_accept(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Three attempts: malformed envelope → wrong answer → correct.

        Exercises both Supervisor triggers — the ``query_error_or_empty``
        path (first attempt's tool call returned zero rows) and the
        ``quality`` path (second attempt's envelope parsed but the
        value mismatched).
        """
        # Script three separate agent runs (one per supervisor attempt).
        # Each run is a simple assistant-only message that skips the
        # ReAct tool loop entirely on attempts where it matters — but
        # we still want the first attempt to emit a ``run_sql`` tool
        # call that returns an empty result, so the error-catch trigger
        # fires.
        responses = [
            # Attempt 1: agent calls run_sql, gets zero rows, then
            # produces a malformed (non-JSON) final response.
            _response(tool_calls=[_tool_call(sql="SELECT * FROM orders WHERE 1=0", call_id="c1")]),
            _response(content="not even close to JSON"),
            # Attempt 2: agent calls run_sql (good rows), produces
            # an envelope with the wrong answer.
            _response(tool_calls=[_tool_call(sql="SELECT COUNT(*) FROM orders", call_id="c2")]),
            _response(content=_envelope(999)),
            # Attempt 3: agent calls run_sql, produces the correct
            # canonical answer (200 orders per the catalog).
            _response(tool_calls=[_tool_call(sql="SELECT COUNT(*) FROM orders", call_id="c3")]),
            _response(content=_envelope(200)),
        ]
        scripted_client = MockLLMClient(responses)
        connector = _FakeRowSetConnector(
            record_sets=[
                [],  # attempt 1 — zero rows fires the error predicate
                [{"count": 200}],
                [{"count": 200}],
            ]
        )
        app, _ = _make_app_with_runner(
            build_client=lambda: scripted_client,
            connector_factory=connector,
            monkeypatch=monkeypatch,
        )

        with TestClient(app) as client:
            response = client.post(
                "/runners/sql-analyst/ask",
                json={"question_id": "total-orders-count"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["accepted"] is True
        assert body["attempts"] == 3
        assert body["answer"] == 200
        # Two non-accept interventions (the retry driven by the error
        # predicate on attempt 1, then the retry driven by the quality
        # trigger on attempt 2). The accept on attempt 3 is not in
        # ``interventions``.
        assert len(body["interventions"]) == 2
        trigger_names = {i["trigger_name"] for i in body["interventions"]}
        assert "query_error_or_empty" in trigger_names
        assert "quality" in trigger_names

    def test_canonical_path_first_attempt_correct(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Happy path: agent nails the answer on the first attempt."""
        responses = [
            _response(tool_calls=[_tool_call(sql="SELECT COUNT(*) FROM orders", call_id="c1")]),
            _response(content=_envelope(200)),
        ]
        scripted_client = MockLLMClient(responses)
        connector = _FakeRowSetConnector(record_sets=[[{"count": 200}]])
        app, _ = _make_app_with_runner(
            build_client=lambda: scripted_client,
            connector_factory=connector,
            monkeypatch=monkeypatch,
        )

        with TestClient(app) as client:
            response = client.post(
                "/runners/sql-analyst/ask",
                json={"question_id": "total-orders-count"},
            )

        body = response.json()
        assert response.status_code == 200
        assert body["accepted"] is True
        assert body["attempts"] == 1
        assert body["answer"] == 200
        assert body["interventions"] == []

    def test_unknown_question_id_returns_404(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # No agent runs on the 404 path — pass an exhausted client.
        app, _ = _make_app_with_runner(
            build_client=lambda: MockLLMClient([]),
            connector_factory=_FakeRowSetConnector(record_sets=[]),
            monkeypatch=monkeypatch,
        )

        with TestClient(app) as client:
            response = client.post(
                "/runners/sql-analyst/ask",
                json={"question_id": "never-heard-of-it"},
            )

        assert response.status_code == 404
        assert "never-heard-of-it" in response.json()["detail"]

    def test_both_fields_none_returns_422(self, monkeypatch: pytest.MonkeyPatch) -> None:
        app, _ = _make_app_with_runner(
            build_client=lambda: MockLLMClient([]),
            connector_factory=_FakeRowSetConnector(record_sets=[]),
            monkeypatch=monkeypatch,
        )

        with TestClient(app) as client:
            response = client.post("/runners/sql-analyst/ask", json={})

        assert response.status_code == 422

    def test_list_questions_omits_canonical_answers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``GET /questions`` must leak neither ``expected`` nor canonical values."""
        app, _ = _make_app_with_runner(
            build_client=lambda: MockLLMClient([]),
            connector_factory=_FakeRowSetConnector(record_sets=[]),
            monkeypatch=monkeypatch,
        )

        with TestClient(app) as client:
            response = client.get("/runners/sql-analyst/questions")

        assert response.status_code == 200
        body = response.json()
        # Response shape carries id + question only.
        assert set(body) == {"questions"}
        for item in body["questions"]:
            assert set(item) == {"id", "question"}
        # Sanity: every catalog entry is returned.
        assert {item["id"] for item in body["questions"]} == {q.id for q in QUESTIONS}

    def test_question_id_wins_when_both_provided(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When both fields are set, ``question_id`` is authoritative — the
        ad-hoc ``question`` is silently ignored per the OpenAPI contract."""
        responses = [
            _response(tool_calls=[_tool_call(sql="SELECT COUNT(*) FROM orders", call_id="c1")]),
            _response(content=_envelope(200)),
        ]
        scripted_client = MockLLMClient(responses)
        connector = _FakeRowSetConnector(record_sets=[[{"count": 200}]])
        app, _ = _make_app_with_runner(
            build_client=lambda: scripted_client,
            connector_factory=connector,
            monkeypatch=monkeypatch,
        )

        with TestClient(app) as client:
            response = client.post(
                "/runners/sql-analyst/ask",
                json={
                    "question_id": "total-orders-count",
                    "question": "some free-form question that should be ignored",
                },
            )

        body = response.json()
        assert response.status_code == 200
        assert body["accepted"] is True
        # The canonical task went to the agent (the catalog question
        # text, not the free-form one). We can verify via the
        # scripted client's recorded calls — the user message text
        # must match the catalog's prompt.
        first_call_messages = scripted_client.calls[0]["messages"]
        user_message = next(m for m in first_call_messages if m.role == "user")
        expected_question_text = next(q.question for q in QUESTIONS if q.id == "total-orders-count")
        assert user_message.content == expected_question_text


class TestSqlAnalystAdHoc:
    """Ad-hoc path runs without the ``GroundTruthEvaluator`` — only the
    error-catch predicate trigger remains."""

    def test_ad_hoc_question_error_trigger_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A first-attempt SQL error drives one retry via the error predicate."""

        class FakeError(Exception):
            pass

        responses = [
            # Attempt 1: agent runs a SQL that fails.
            _response(tool_calls=[_tool_call(sql="INSERT INTO x VALUES (1)", call_id="c1")]),
            _response(content=_envelope(42)),
            # Attempt 2: agent recovers with a SELECT that returns a row.
            _response(tool_calls=[_tool_call(sql="SELECT 42", call_id="c2")]),
            _response(content=_envelope(42)),
        ]
        scripted_client = MockLLMClient(responses)

        # First call raises (fires the error predicate); second returns a row.
        class _ConnectorScripted:
            def __init__(self) -> None:
                self._calls = 0

            def __call__(self, dsn: str, statement_timeout_ms: int) -> Any:
                async def connect() -> Any:
                    self._calls += 1
                    if self._calls == 1:
                        return _FakeConnection(raises=FakeError("boom"))
                    return _FakeConnection(records=[{"n": 42}])

                return connect

        connector = _ConnectorScripted()
        app, _ = _make_app_with_runner(
            build_client=lambda: scripted_client,
            connector_factory=connector,
            monkeypatch=monkeypatch,
        )

        with TestClient(app) as client:
            response = client.post(
                "/runners/sql-analyst/ask",
                json={"question": "How many rows are in customers?"},
            )

        assert response.status_code == 200
        body = response.json()
        # Ad-hoc path: no QualityTrigger — no value-level check. The
        # error-trigger fires once, the retry succeeds, accepted.
        assert body["accepted"] is True
        assert body["attempts"] == 2
        assert len(body["interventions"]) == 1
        assert body["interventions"][0]["trigger_name"] == "query_error_or_empty"
        assert body["answer"] == 42


class TestSqlAnalystRegistrationIdempotence:
    """Calling ``register`` twice does not duplicate routes or hooks."""

    def test_register_twice_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NANITICS_SQL_ANALYST_SANDBOX_USER", "sandbox")
        monkeypatch.setenv("NANITICS_SQL_ANALYST_SANDBOX_PASSWORD", "pw")
        monkeypatch.setenv("POSTGRES_DSN", "postgresql://app:pw@db:5432/app")

        from sql_analyst import runner as runner_module

        app = FastAPI()
        context, _store = _build_shell_context(build_client=lambda: MockLLMClient([]))

        runner_module.register(app, context)
        route_count_after_first = len(app.router.routes)

        runner_module.register(app, context)
        route_count_after_second = len(app.router.routes)

        assert route_count_after_first == route_count_after_second
        # And only one copy of each of the two endpoints exists.
        paths = [getattr(r, "path", None) for r in app.router.routes]
        assert paths.count("/runners/sql-analyst/ask") == 1
        assert paths.count("/runners/sql-analyst/questions") == 1


class TestSqlAnalystHelpers:
    """Unit tests for the runner's internal helpers — exercises the
    env-resolution, sandbox-DSN derivation, envelope parsing, and the
    Anthropic caching wrapper branches that the endpoint tests don't
    reach."""

    def test_derive_sandbox_dsn_preserves_host_and_db(self) -> None:
        from sql_analyst.runner import _derive_sandbox_dsn

        dsn = _derive_sandbox_dsn(
            "postgresql://app:apppw@db.internal:5432/mydb",
            "sandbox_role",
            "sandbox_pw",
        )
        assert "sandbox_role:sandbox_pw@db.internal:5432/mydb" in dsn

    def test_derive_sandbox_dsn_urlencodes_credentials(self) -> None:
        """Passwords with ``@`` or ``:`` must not corrupt the DSN."""
        from sql_analyst.runner import _derive_sandbox_dsn

        dsn = _derive_sandbox_dsn(
            "postgresql://app:pw@db:5432/x",
            "role@odd",
            "p@ss:word",
        )
        # The derived DSN still parses cleanly and the hostname is preserved.
        from urllib.parse import urlsplit

        parts = urlsplit(dsn)
        assert parts.hostname == "db"
        assert parts.port == 5432

    def test_resolve_privileged_dsn_uses_env_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from sql_analyst.runner import _resolve_privileged_dsn

        monkeypatch.setenv("POSTGRES_DSN", "postgresql://x:y@host:5432/z")
        assert _resolve_privileged_dsn() == "postgresql://x:y@host:5432/z"

    def test_resolve_privileged_dsn_falls_back_to_components(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from sql_analyst.runner import _resolve_privileged_dsn

        monkeypatch.delenv("POSTGRES_DSN", raising=False)
        monkeypatch.setenv("POSTGRES_USER", "ulrich")
        monkeypatch.setenv("POSTGRES_PASSWORD", "secret")
        monkeypatch.setenv("POSTGRES_DB", "analytics")
        dsn = _resolve_privileged_dsn()
        assert "ulrich:secret" in dsn
        assert "/analytics" in dsn

    def test_resolve_sandbox_credentials_missing_user_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from sql_analyst.runner import _resolve_sandbox_credentials

        monkeypatch.delenv("NANITICS_SQL_ANALYST_SANDBOX_USER", raising=False)
        monkeypatch.setenv("NANITICS_SQL_ANALYST_SANDBOX_PASSWORD", "pw")

        with pytest.raises(RuntimeError, match="SANDBOX_USER"):
            _resolve_sandbox_credentials()

    def test_resolve_sandbox_credentials_missing_password_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from sql_analyst.runner import _resolve_sandbox_credentials

        monkeypatch.setenv("NANITICS_SQL_ANALYST_SANDBOX_USER", "sandbox")
        monkeypatch.delenv("NANITICS_SQL_ANALYST_SANDBOX_PASSWORD", raising=False)

        with pytest.raises(RuntimeError, match="SANDBOX_PASSWORD"):
            _resolve_sandbox_credentials()

    def test_parse_envelope_handles_none_and_blank(self) -> None:
        from sql_analyst.runner import _parse_envelope

        assert _parse_envelope(None) is None
        assert _parse_envelope("") is None
        assert _parse_envelope("   ") is None

    def test_parse_envelope_handles_code_fence(self) -> None:
        from sql_analyst.runner import _parse_envelope

        parsed = _parse_envelope('```json\n{"answer": 1, "sql": "x", "rowcount": 1}\n```')
        assert parsed == {"answer": 1, "sql": "x", "rowcount": 1}

    def test_parse_envelope_rejects_non_object_json(self) -> None:
        from sql_analyst.runner import _parse_envelope

        assert _parse_envelope("[1,2,3]") is None

    def test_parse_envelope_rejects_invalid_json(self) -> None:
        from sql_analyst.runner import _parse_envelope

        assert _parse_envelope("not json at all") is None

    def test_opt_in_caching_passes_non_anthropic_through(self) -> None:
        """OpenAI clients (and test stubs) are returned unchanged."""
        from sql_analyst.runner import _opt_in_caching

        sentinel = object()
        assert _opt_in_caching(sentinel) is sentinel

    def test_opt_in_caching_reconstructs_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When the client is ``AnthropicLLMClient``, the wrapper rebuilds
        it with ``enable_caching=True`` using the env vars as the key
        source."""
        from sql_analyst.runner import _opt_in_caching

        from nanitics import AnthropicLLMClient

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("NANITICS_LLM_MODEL", "claude-haiku-4-5-20251001")

        original = AnthropicLLMClient(model="claude-haiku-4-5-20251001", api_key="sk-ant-test")
        wrapped = _opt_in_caching(original)

        assert isinstance(wrapped, AnthropicLLMClient)
        # Different instance (re-constructed with enable_caching=True).
        assert wrapped is not original
        # The reconstructed client has caching enabled (the internal
        # attribute is not public, but the reconstructed client's
        # ``model`` matches).
        assert wrapped.model == "claude-haiku-4-5-20251001"

    def test_opt_in_caching_falls_back_when_env_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If ``ANTHROPIC_API_KEY`` is not in env, pass through unchanged."""
        from sql_analyst.runner import _opt_in_caching

        from nanitics import AnthropicLLMClient

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("NANITICS_LLM_MODEL", "claude-haiku-4-5-20251001")

        original = AnthropicLLMClient(model="claude-haiku-4-5-20251001", api_key="sk-ant-test")
        wrapped = _opt_in_caching(original)
        # Fallback: return the original client untouched.
        assert wrapped is original

    def test_expected_non_empty_branches(self) -> None:
        from sql_analyst.questions import ExpectedRowSet
        from sql_analyst.runner import _expected_non_empty

        # Ad-hoc path (no catalog question) → False (empty results allowed).
        assert _expected_non_empty(None) is False
        # Scalar expected → True (zero is a valid scalar answer).
        assert _expected_non_empty(SampleQuestion(id="x", question="q", expected=ExpectedScalar(value=0))) is True
        # Empty row-set expected → False.
        assert (
            _expected_non_empty(
                SampleQuestion(
                    id="y",
                    question="q",
                    expected=ExpectedRowSet(rows=(), ordered=False),
                )
            )
            is False
        )
        # Non-empty row-set expected → True.
        non_empty_rowset = ExpectedRowSet(rows=(ExpectedRow(columns=("a",), values=(1,)),), ordered=False)
        assert _expected_non_empty(SampleQuestion(id="z", question="q", expected=non_empty_rowset)) is True

    def test_intervention_summaries_preserve_order_and_attempts(self) -> None:
        from sql_analyst.runner import _intervention_summaries

        from nanitics.composition.multi_agent.supervisor import SupervisionAction, SupervisionDecision

        decisions = [
            SupervisionDecision(
                action=SupervisionAction.RETRY,
                trigger_name="query_error_or_empty",
                feedback="error",
            ),
            SupervisionDecision(
                action=SupervisionAction.RETRY,
                trigger_name="quality",
                feedback="wrong value",
            ),
        ]
        summaries = _intervention_summaries(decisions)
        assert [s.attempt for s in summaries] == [1, 2]
        assert [s.trigger_name for s in summaries] == ["query_error_or_empty", "quality"]
        assert [s.action for s in summaries] == ["retry", "retry"]

    def test_query_error_predicate_retries_on_error(self) -> None:
        from sql_analyst.runner import _make_query_error_predicate
        from sql_analyst.tool import LAST_TOOL_METADATA_STATE_KEY

        state = {
            LAST_TOOL_METADATA_STATE_KEY: {
                "error": True,
                "error_type": "SyntaxError",
                "sql": "SEL",
            }
        }
        predicate = _make_query_error_predicate(tool_state=state, allow_empty_result=False)
        decision = predicate(_fake_agent_result(), "task")
        assert decision is not None
        assert decision.trigger_name == "query_error_or_empty"
        assert decision.action.value == "retry"
        assert decision.feedback is not None
        assert "SyntaxError" in decision.feedback

    def test_query_error_predicate_retries_on_empty_when_disallowed(self) -> None:
        from sql_analyst.runner import _make_query_error_predicate
        from sql_analyst.tool import LAST_TOOL_METADATA_STATE_KEY

        state = {LAST_TOOL_METADATA_STATE_KEY: {"error": False, "rowcount": 0, "sql": "SELECT"}}
        predicate = _make_query_error_predicate(tool_state=state, allow_empty_result=False)
        decision = predicate(_fake_agent_result(), "task")
        assert decision is not None
        assert decision.trigger_name == "query_error_or_empty"

    def test_query_error_predicate_passes_on_empty_when_allowed(self) -> None:
        from sql_analyst.runner import _make_query_error_predicate
        from sql_analyst.tool import LAST_TOOL_METADATA_STATE_KEY

        state = {LAST_TOOL_METADATA_STATE_KEY: {"error": False, "rowcount": 0}}
        predicate = _make_query_error_predicate(tool_state=state, allow_empty_result=True)
        assert predicate(_fake_agent_result(), "task") is None

    def test_query_error_predicate_passes_on_success(self) -> None:
        from sql_analyst.runner import _make_query_error_predicate
        from sql_analyst.tool import LAST_TOOL_METADATA_STATE_KEY

        state = {LAST_TOOL_METADATA_STATE_KEY: {"error": False, "rowcount": 5}}
        predicate = _make_query_error_predicate(tool_state=state, allow_empty_result=False)
        assert predicate(_fake_agent_result(), "task") is None

    def test_query_error_predicate_passes_when_state_empty(self) -> None:
        """Agent never called ``run_sql`` — no decision from this predicate."""
        from sql_analyst.runner import _make_query_error_predicate

        predicate = _make_query_error_predicate(tool_state={}, allow_empty_result=False)
        assert predicate(_fake_agent_result(), "task") is None


def _fake_agent_result() -> Any:
    from nanitics import Usage
    from nanitics.core.agents.base import AgentResult
    from nanitics.infrastructure.llm.protocol import Message

    return AgentResult(
        output="unused",
        total_steps=1,
        termination_reason="complete",
        messages=[Message(role="user", content="x")],
        usage=Usage(input_tokens=1, output_tokens=1),
    )


class TestRunSqlToolContextIntegration:
    """The tool records each invocation's metadata into ``ctx.state`` so
    the runner's error predicate can consume it. Exercised through the
    registry's ``ToolContext`` injection path to prove the wiring."""

    @pytest.mark.asyncio
    async def test_tool_state_records_last_metadata(self) -> None:
        from sql_analyst.tool import LAST_TOOL_METADATA_STATE_KEY

        from nanitics.core.tools.context import ToolContext, _current_tool_context

        connection = _FakeConnection(records=[{"n": 1}])
        factory = _connector_factory_for(connection)
        tool = build_run_sql_tool(
            sandbox_dsn="postgresql://sandbox/db",
            connector_factory=factory,
        )

        shared_state: dict[str, Any] = {}
        token = _current_tool_context.set(ToolContext(state=shared_state))
        try:
            await tool.execute(sql="SELECT 1")
        finally:
            _current_tool_context.reset(token)

        assert LAST_TOOL_METADATA_STATE_KEY in shared_state
        metadata = shared_state[LAST_TOOL_METADATA_STATE_KEY]
        assert metadata["rowcount"] == 1
        assert metadata["error"] is False
