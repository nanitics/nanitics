"""Sample-question catalog with canonical ground-truth answers.

The catalog is the ``[x] here is the known-good answer for this
question`` ground truth the :class:`GroundTruthEvaluator` compares
against. Every canonical answer was hand-computed from the deterministic
seed data in ``schema.sql`` — re-applying the schema leaves row counts
unchanged, so the canonical values are stable across container
rebuilds.

How to add a new sample question
--------------------------------

Append a single :class:`SampleQuestion` entry to :data:`QUESTIONS` with
a kebab-case id, a natural-language ``question`` string, and an
``expected`` :class:`ExpectedAnswer`. Then run
``pytest tests/test_sql_analyst_runner.py::TestCatalogInvariants`` to
guard against typos. No changes to :mod:`evaluator`, :mod:`runner`, or
:mod:`tool` are required — the whole point of the catalog pattern is
that additions are one line of Python.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any


def _as_float(value: Any) -> float | None:
    """Coerce a scalar candidate to ``float`` when possible.

    Accepts ``int`` / ``float`` / ``Decimal`` / numeric-looking strings.
    Returns ``None`` when no safe coercion exists — the caller treats
    that as a type mismatch.
    """
    if isinstance(value, bool):  # bool is an int subclass; reject it here
        return None
    if isinstance(value, int | float | Decimal):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


@dataclass(frozen=True)
class ExpectedScalar:
    """Canonical scalar answer for a question.

    Attributes:
        value: The expected scalar value — ``int``, ``float``, or ``str``.
        tolerance: Optional absolute tolerance for numeric comparisons.
            When set, ``abs(candidate - value) <= tolerance`` counts as a
            match. Ignored for string values.
    """

    value: int | float | str
    tolerance: float | None = None

    def compare(self, candidate: Any) -> tuple[bool, str]:
        """Compare *candidate* against the expected scalar."""
        if isinstance(self.value, str):
            if candidate == self.value:
                return True, ""
            return False, f"expected scalar {self.value!r}, got {candidate!r}"

        candidate_num = _as_float(candidate)
        if candidate_num is None:
            return False, (f"expected numeric scalar {self.value!r}, got {type(candidate).__name__}: {candidate!r}")
        expected_num = float(self.value)
        if self.tolerance is not None:
            if abs(candidate_num - expected_num) <= self.tolerance:
                return True, ""
            return False, (f"expected scalar {expected_num} (±{self.tolerance}), got {candidate_num}")
        if candidate_num == expected_num:
            return True, ""
        return False, f"expected scalar {expected_num}, got {candidate_num}"


@dataclass(frozen=True)
class ExpectedRow:
    """Canonical row answer — a single fixed-column tuple.

    Attributes:
        columns: The column names, in order.
        values: The expected values, in the same order as ``columns``.
    """

    columns: tuple[str, ...]
    values: tuple[Any, ...]

    def compare(self, candidate: Any) -> tuple[bool, str]:
        """Compare *candidate* against the expected row.

        Accepts either a mapping keyed by column name, or a sequence
        positionally aligned with ``columns``.
        """
        mapping = self._coerce_to_mapping(candidate)
        if mapping is None:
            return False, (
                f"expected row with columns {list(self.columns)}, got {type(candidate).__name__}: {candidate!r}"
            )
        for column, expected in zip(self.columns, self.values, strict=True):
            if column not in mapping:
                return False, f"expected row to carry column {column!r}; got {sorted(mapping)}"
            ok, detail = self._compare_cell(expected, mapping[column])
            if not ok:
                return False, f"column {column!r}: {detail}"
        return True, ""

    def _coerce_to_mapping(self, candidate: Any) -> dict[str, Any] | None:
        if isinstance(candidate, Mapping):
            return dict(candidate)
        if isinstance(candidate, Sequence) and not isinstance(candidate, str | bytes):
            if len(candidate) != len(self.columns):
                return None
            return dict(zip(self.columns, candidate, strict=True))
        return None

    @staticmethod
    def _compare_cell(expected: Any, candidate: Any) -> tuple[bool, str]:
        """Cell-level comparison — scalar equality with numeric coercion."""
        if isinstance(expected, str):
            if candidate == expected:
                return True, ""
            return False, f"expected {expected!r}, got {candidate!r}"
        expected_num = _as_float(expected)
        candidate_num = _as_float(candidate)
        if expected_num is None or candidate_num is None:
            if candidate == expected:
                return True, ""
            return False, f"expected {expected!r}, got {candidate!r}"
        if expected_num == candidate_num:
            return True, ""
        return False, f"expected {expected_num}, got {candidate_num}"


@dataclass(frozen=True)
class ExpectedRowSet:
    """Canonical row-set answer.

    Attributes:
        rows: The expected rows. Each :class:`ExpectedRow` shares its
            column ordering with the others.
        ordered: When ``True``, ``candidate`` rows are compared
            positionally. When ``False``, set equality applies.
    """

    rows: tuple[ExpectedRow, ...]
    ordered: bool = False

    def compare(self, candidate: Any) -> tuple[bool, str]:
        """Compare *candidate* against the expected row-set."""
        if not isinstance(candidate, Sequence) or isinstance(candidate, str | bytes):
            return False, (
                f"expected row set of length {len(self.rows)}, got {type(candidate).__name__}: {candidate!r}"
            )
        if len(candidate) != len(self.rows):
            return False, (f"expected row set of length {len(self.rows)}, got {len(candidate)} rows")
        if self.ordered:
            for index, (expected_row, actual) in enumerate(zip(self.rows, candidate, strict=True)):
                ok, detail = expected_row.compare(actual)
                if not ok:
                    return False, f"row {index}: {detail}"
            return True, ""

        # Unordered: every expected row must find a unique match in the
        # candidate set. Use index consumption to avoid double-matching.
        remaining = list(range(len(candidate)))
        for expected_index, expected_row in enumerate(self.rows):
            match_index: int | None = None
            for idx in remaining:
                ok, _ = expected_row.compare(candidate[idx])
                if ok:
                    match_index = idx
                    break
            if match_index is None:
                return False, (
                    f"expected row {expected_index} "
                    f"{list(zip(expected_row.columns, expected_row.values, strict=True))} "
                    f"was not found in candidate row set"
                )
            remaining.remove(match_index)
        return True, ""


# The three shapes the evaluator knows how to compare.
ExpectedAnswer = ExpectedScalar | ExpectedRow | ExpectedRowSet


@dataclass(frozen=True)
class SampleQuestion:
    """One canonical question with a hand-computed ground-truth answer.

    Attributes:
        id: kebab-case identifier, unique within :data:`QUESTIONS`.
        question: natural-language prompt shown to the agent.
        expected: canonical answer compared to the agent's output.
    """

    id: str
    question: str
    expected: ExpectedAnswer


# The initial five canonical questions. Every canonical answer was
# computed against the deterministic seed data in ``schema.sql``.
QUESTIONS: list[SampleQuestion] = [
    SampleQuestion(
        id="total-orders-count",
        question="How many orders are there in total? Respond with the scalar count.",
        expected=ExpectedScalar(value=200),
    ),
    SampleQuestion(
        id="revenue-total",
        question=(
            "What is the total revenue across all order items? Compute "
            "SUM(quantity * unit_price_at_order) and respond with the scalar sum."
        ),
        expected=ExpectedScalar(value=71125.00, tolerance=0.01),
    ),
    SampleQuestion(
        id="top-5-customers-by-revenue",
        question=(
            "Return the top 5 customers by total revenue. Revenue is "
            "SUM(quantity * unit_price_at_order) across all order items in "
            "each customer's orders. Respond with a list of rows with columns "
            "`customer_id` and `revenue`, ordered by revenue descending with "
            "customer_id ascending as a tiebreaker. Limit to 5 rows."
        ),
        expected=ExpectedRowSet(
            rows=(
                ExpectedRow(columns=("customer_id", "revenue"), values=(30, 3000.00)),
                ExpectedRow(columns=("customer_id", "revenue"), values=(20, 2875.00)),
                ExpectedRow(columns=("customer_id", "revenue"), values=(50, 2875.00)),
                ExpectedRow(columns=("customer_id", "revenue"), values=(10, 2750.00)),
                ExpectedRow(columns=("customer_id", "revenue"), values=(40, 2750.00)),
            ),
            ordered=True,
        ),
    ),
    SampleQuestion(
        id="orders-by-region",
        question=(
            "For each region, how many orders were placed by customers in that "
            "region? Join `orders` to `customers` to `regions`. Respond with "
            "a list of rows with columns `region` (the region name) and "
            "`order_count`. Order is not significant."
        ),
        expected=ExpectedRowSet(
            rows=(
                ExpectedRow(columns=("region", "order_count"), values=("North America", 40)),
                ExpectedRow(columns=("region", "order_count"), values=("Europe", 40)),
                ExpectedRow(columns=("region", "order_count"), values=("Asia Pacific", 40)),
                ExpectedRow(columns=("region", "order_count"), values=("Latin America", 40)),
                ExpectedRow(columns=("region", "order_count"), values=("Middle East", 40)),
            ),
            ordered=False,
        ),
    ),
    SampleQuestion(
        id="top-customer-per-region",
        question=(
            "For each region, which customer had the highest total revenue "
            "from non-cancelled orders (status 'pending', 'shipped', or "
            "'delivered')? Revenue is SUM(quantity * unit_price_at_order) "
            "across qualifying order_items rows joined to those orders. "
            "Return one row per region with columns `region_name` (the region "
            "name), `customer_name` (the customer name), and `total_revenue` "
            "(the numeric sum). Order by region_name ascending."
        ),
        expected=ExpectedRowSet(
            rows=(
                ExpectedRow(
                    columns=("region_name", "customer_name", "total_revenue"),
                    values=("Asia Pacific", "Customer 023", 1275),
                ),
                ExpectedRow(
                    columns=("region_name", "customer_name", "total_revenue"),
                    values=("Europe", "Customer 027", 1050),
                ),
                ExpectedRow(
                    columns=("region_name", "customer_name", "total_revenue"),
                    values=("Latin America", "Customer 029", 2300),
                ),
                ExpectedRow(
                    columns=("region_name", "customer_name", "total_revenue"),
                    values=("Middle East", "Customer 025", 2375),
                ),
                ExpectedRow(
                    columns=("region_name", "customer_name", "total_revenue"),
                    values=("North America", "Customer 021", 375),
                ),
            ),
            ordered=True,
        ),
    ),
    SampleQuestion(
        id="cancelled-orders-by-month",
        question=(
            "For cancelled orders only, how many were placed per calendar "
            "month? Bucket `order_date` by month (use `date_trunc('month', "
            "order_date)`). Respond with a list of rows with columns `month` "
            "(ISO date string `YYYY-MM-01`) and `count`, ordered by month "
            "ascending."
        ),
        expected=ExpectedRowSet(
            rows=(
                ExpectedRow(columns=("month", "count"), values=("2024-01-01", 7)),
                ExpectedRow(columns=("month", "count"), values=("2024-02-01", 8)),
                ExpectedRow(columns=("month", "count"), values=("2024-03-01", 7)),
                ExpectedRow(columns=("month", "count"), values=("2024-04-01", 8)),
                ExpectedRow(columns=("month", "count"), values=("2024-05-01", 8)),
                ExpectedRow(columns=("month", "count"), values=("2024-06-01", 7)),
                ExpectedRow(columns=("month", "count"), values=("2024-07-01", 5)),
            ),
            ordered=True,
        ),
    ),
]
