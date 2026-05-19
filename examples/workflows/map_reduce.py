"""MapReduce: split, map concurrently, reduce.

Demonstrates ``MapReduce`` — the orchestration workflow that splits input
into items, applies a step to each concurrently, and reduces results.
Covers structural splitting, concurrency control, failure policies,
and async splitter/reducer support.

Related guide: docs/guides/orchestration.md
"""

import asyncio

from examples.helpers import make_emitter
from nanitics.composition import (
    FailurePolicy,
    FunctionStep,
    StepResult,
)
from nanitics.specialized import MapReduce


async def main() -> None:
    # --- Section 1: Basic Split-Map-Reduce ---
    print("--- Section 1: Basic Split-Map-Reduce ---")

    # The fundamental three-phase flow:
    # 1. Splitter breaks input into items
    # 2. Step runs on each item concurrently
    # 3. Reducer combines all results into final output

    async def double(n: int) -> int:
        return n * 2

    emitter = make_emitter("mapreduce-s1")

    workflow = MapReduce(
        name="double-and-sum",
        step=FunctionStep(name="double", fn=double),
        splitter=lambda x: x,  # Input is already a list
        reducer=lambda results: sum(r.output for r in results),
        emitter=emitter,
    )

    result = await workflow.execute([1, 2, 3, 4, 5])

    assert result.output == 30  # 2 + 4 + 6 + 8 + 10
    assert result.metadata["total_items"] == 5
    assert result.metadata["total_steps_executed"] == 5

    print("  Input: [1, 2, 3, 4, 5]")
    print(f"  Output: {result.output}")
    print("✓ Split → double each → sum = 30")

    # --- Section 2: Structural Splitting ---
    print("\n--- Section 2: Structural Splitting ---")

    # Splitter extracts items from structured input.
    # Reducer reassembles into structure.
    # The step itself stays generic — domain logic lives in splitter/reducer.

    report = {"title": "Q3 Report", "sections": ["Revenue", "Costs", "Outlook"]}

    async def analyze_section(section: str) -> str:
        return f"{section} analysis complete"

    # Reducer captures `report` via closure to access the original title
    def reassemble(results: list[StepResult]) -> dict:
        return {
            "title": report["title"],
            "analyses": [r.output for r in results],
            "section_count": len(results),
        }

    emitter = make_emitter("mapreduce-s2")

    workflow = MapReduce(
        name="report-analysis",
        step=FunctionStep(name="analyze", fn=analyze_section),
        splitter=lambda r: r["sections"],
        reducer=reassemble,
        emitter=emitter,
    )

    result = await workflow.execute(report)

    assert result.output["title"] == "Q3 Report"
    assert result.output["section_count"] == 3
    assert result.output["analyses"] == [
        "Revenue analysis complete",
        "Costs analysis complete",
        "Outlook analysis complete",
    ]

    print(f"  Title: {result.output['title']}")
    print(f"  Analyses: {result.output['analyses']}")
    print("✓ Splitter extracted sections, reducer reassembled report structure")

    # --- Section 3: Max Concurrency ---
    print("\n--- Section 3: Max Concurrency ---")

    # max_concurrency limits how many items run simultaneously.
    # It affects parallelism, not results — all items are still processed.

    async def times_ten(n: int) -> int:
        return n * 10

    emitter = make_emitter("mapreduce-s3")

    workflow = MapReduce(
        name="scaled-values",
        step=FunctionStep(name="times_ten", fn=times_ten),
        splitter=lambda x: x,
        reducer=lambda results: sorted(r.output for r in results),
        max_concurrency=2,
        emitter=emitter,
    )

    result = await workflow.execute([1, 2, 3, 4, 5, 6])

    assert result.output == [10, 20, 30, 40, 50, 60]
    assert result.metadata["total_items"] == 6
    assert result.metadata["total_steps_executed"] == 6

    print(f"  Output: {result.output}")
    print(f"  Items processed: {result.metadata['total_items']}")
    print("✓ All 6 items processed with max_concurrency=2")

    # --- Section 4: ALL_OR_NOTHING Failure ---
    print("\n--- Section 4: ALL_OR_NOTHING Failure ---")

    # Default failure policy: if any item fails, the exception propagates.
    # Same semantics as Parallel's ALL_OR_NOTHING.

    async def process_or_fail(item: str) -> str:
        if item == "FAIL":
            raise ValueError("processing failed")
        return item.upper()

    emitter = make_emitter("mapreduce-s4")

    workflow = MapReduce(
        name="strict-processing",
        step=FunctionStep(name="process", fn=process_or_fail),
        splitter=lambda x: x,
        reducer=lambda results: [r.output for r in results],
        emitter=emitter,
    )

    raised = False
    try:
        await workflow.execute(["alpha", "beta", "FAIL", "delta"])
    except ValueError as exc:
        raised = True
        assert "processing failed" in str(exc)
        print(f"  Caught: {exc}")

    assert raised, "Expected ValueError from failing item"
    print("✓ ALL_OR_NOTHING propagated the item failure")

    # --- Section 5: BEST_EFFORT Failure ---
    print("\n--- Section 5: BEST_EFFORT Failure ---")

    # BEST_EFFORT collects partial results from successful items.
    # Failed item indices are tracked in metadata.

    async def double_or_fail(n: int) -> int:
        if n == -1:
            raise ValueError("invalid item")
        return n * 2

    emitter = make_emitter("mapreduce-s5")

    workflow = MapReduce(
        name="resilient-processing",
        step=FunctionStep(name="double", fn=double_or_fail),
        splitter=lambda x: x,
        reducer=lambda results: sum(r.output for r in results),
        failure_policy=FailurePolicy.BEST_EFFORT,
        emitter=emitter,
    )

    result = await workflow.execute([10, 20, -1, 40, -1])

    # Reducer receives only successful results: 20 + 40 + 80 = 140
    assert result.output == 140
    assert result.metadata["total_items"] == 5
    assert result.metadata["total_steps_executed"] == 3
    assert sorted(result.metadata["failed_items"]) == [2, 4]

    print(f"  Output: {result.output}")
    print(f"  Failed items at indices: {result.metadata['failed_items']}")
    print("✓ BEST_EFFORT returned partial results with failure tracking")

    # --- Section 6: Async Splitter and Reducer ---
    print("\n--- Section 6: Async Splitter and Reducer ---")

    # Both splitter and reducer can be async functions.
    # MapReduce detects and awaits them transparently.

    async def split_words(text: str) -> list[str]:
        return text.split()

    async def word_length(word: str) -> int:
        return len(word)

    async def combine_stats(results: list[StepResult]) -> dict:
        lengths = [r.output for r in results]
        return {
            "word_count": len(lengths),
            "total_chars": sum(lengths),
            "avg_length": sum(lengths) / len(lengths),
        }

    emitter = make_emitter("mapreduce-s6")

    workflow = MapReduce(
        name="word-stats",
        step=FunctionStep(name="length", fn=word_length),
        splitter=split_words,
        reducer=combine_stats,
        emitter=emitter,
    )

    result = await workflow.execute("hello world foo bar")

    assert result.output["word_count"] == 4
    assert result.output["total_chars"] == 16  # 5 + 5 + 3 + 3
    assert result.output["avg_length"] == 4.0

    print(f"  Stats: {result.output}")
    print("✓ Async splitter and reducer handled transparently")

    print("\n✅ All sections passed!")


if __name__ == "__main__":
    asyncio.run(main())
