"""ErrorHandler correction prompt drives real-LLM retry of a flaky tool.

A ``@tool``-decorated function fails on its first invocation and succeeds
on the second. The ``ValueError`` raised inside the tool is wrapped by
the ``ToolRegistry`` as ``ToolExecutionError``, which
``classify_error`` maps to ``CORRECTABLE``. With
``ErrorHandler.default()`` installed, the agent sees a correction prompt
in place of the failed tool result, retries the tool, and produces a
final answer — the ``ErrorCorrectionEvent`` is the load-bearing trace
artifact.

The flaky-counter state lives on a per-attempt container so
``run_with_retry`` can re-enter the scenario from a clean slate and the
emitter can be cleared between attempts. This keeps the scenario
provider-flake-tolerant while preserving first-call-fails semantics.

Acceptance criteria:
  - Agent terminated with ``complete`` (recovered, not bailed).
  - ``result.total_steps >= 2`` (the load-bearing step count is the
    tool-invocation count below; total_steps is a weak shape check).
  - Trace contains an ``ErrorCorrectionEvent`` whose ``error_type ==
    "ToolExecutionError"`` (proves registry wrapping), whose
    ``correction_prompt`` references ``"search"`` and the failure (proves
    the ``ToolExecutionError`` branch of ``format_correction_prompt``
    fired — not the generic fallback), with ``attempt == 1`` and
    ``max_attempts == 3`` (proves ``ErrorHandler.default()`` budgeting
    reached the event).
  - Trace contains at least two ``ToolInvokeEvent`` for ``search``
    (the LLM retried the tool rather than switching strategies).
  - The first ``ToolResultEvent`` for ``search`` is a failure
    (``success=False`` with ``error`` populated) and a later
    ``ToolResultEvent`` for ``search`` is a success (``success=True``
    with ``result`` containing the seeded success string) — pins the
    failed→succeeded boundary in the trace.
  - The ``ErrorCorrectionEvent`` is emitted *before* the second
    ``ToolInvokeEvent`` for ``search`` — pins the causal order
    (correction → retry), not just co-occurrence.
  - Output contains the substring "Python 3.13" — a literal
    verification the judge is not needed for.
  - Output acknowledges that a search was performed for the user and
    reports that results were obtained — judged against the original
    user prompt so "for the user" has grounding. The "after a retry"
    claim is verified by the trace assertions above, not the judge.
"""

from __future__ import annotations

from dataclasses import dataclass

from nanitics.errors import ErrorHandler
from nanitics.infrastructure import (
    ErrorCorrectionEvent,
    ToolInvokeEvent,
    ToolResultEvent,
)
from nanitics.strategies import (
    ReActAgent,
    tool,
)
from nanitics.tracing import InMemoryEmitter
from validation.helpers import (
    assert_result_satisfies,
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)


@dataclass
class _FlakyCounter:
    """Per-attempt mutable state for the flaky tool.

    Module-level mutable state leaks across ``run_with_retry`` attempts.
    A dataclass rebuilt per attempt is the cleanest way to keep the
    "first call fails, second call succeeds" contract deterministic
    while still letting the rest of the scenario be retry-wrapped.
    """

    count: int = 0


# The tool reads its per-attempt counter through this mutable reference.
# Each attempt swaps the container for a fresh one (see the test body).
_counter: list[_FlakyCounter] = [_FlakyCounter()]


@tool("search", "Search for articles on a topic. Requires a 'query' parameter.")
async def flaky_search(query: str) -> str:
    _counter[0].count += 1
    if _counter[0].count == 1:
        raise ValueError("Connection timeout — please retry with the same query.")
    return f"Found 3 articles about '{query}'."


async def test_error_handler_correction_drives_retry(traced_emitter: InMemoryEmitter) -> None:
    client = make_llm_client("anthropic")

    async def _run() -> object:
        # Reset per-attempt state: the flaky counter must be clean, and
        # the emitter must not carry events from a prior attempt — otherwise
        # trace assertions could satisfy against the earlier (possibly
        # failed) run rather than the current one.
        _counter[0] = _FlakyCounter()
        traced_emitter.events.clear()

        agent = ReActAgent(
            name="self-correcting",
            llm_client=client,
            emitter=traced_emitter,
            system_prompt=(
                "You are a research assistant. When a tool fails with a recoverable "
                "error, retry the same tool with the same arguments once before "
                "giving up."
            ),
            tools=[flaky_search],
            error_handler=ErrorHandler.default(),
            max_iterations=5,
        )
        return await agent.run("Search for articles about Python 3.13. If the search fails, retry it.")

    result = await run_with_retry(_run, max_attempts=2)

    # --- Result-shape invariants ---
    assert result.termination_reason == "complete", (
        f"Expected termination_reason='complete', got: {result.termination_reason!r}"
    )
    assert result.total_steps >= 2, f"Expected total_steps >= 2, got: {result.total_steps}"

    # --- Trace-shape invariants ---
    # Pin the correction event's payload. A regression that emitted the
    # event from the wrong branch (e.g. registry stopped wrapping and the
    # raw ValueError surfaced as error_type='ValueError'; or
    # format_correction_prompt broke silently and returned an empty
    # correction_prompt) must fail here.
    correction_event = assert_trace_contains(
        traced_emitter,
        ErrorCorrectionEvent,
        predicate=lambda e: (
            e.error_type == "ToolExecutionError"
            and "search" in e.correction_prompt
            and "failed" in e.correction_prompt.lower()
            and e.attempt == 1
            and e.max_attempts == 3
        ),
    )

    # The tool invocation count is the load-bearing "retry actually
    # happened" signal.
    search_invocations = [
        e for e in traced_emitter.events if isinstance(e, ToolInvokeEvent) and e.tool_name == "search"
    ]
    assert len(search_invocations) >= 2, (
        f"Expected at least 2 ToolInvokeEvent for 'search' (initial + retry), got: {len(search_invocations)}"
    )

    # Pin the failed → succeeded boundary in the trace. The first
    # tool.result for search must be a failure; a later one must be a
    # success carrying the seeded success string.
    search_results = [e for e in traced_emitter.events if isinstance(e, ToolResultEvent) and e.tool_name == "search"]
    assert len(search_results) >= 2, (
        f"Expected at least 2 ToolResultEvent for 'search' (failure + success), got: {len(search_results)}"
    )
    assert search_results[0].success is False, (
        f"Expected first 'search' ToolResultEvent to be a failure, got: success={search_results[0].success!r}"
    )
    assert search_results[0].error is not None, (
        f"Expected first 'search' ToolResultEvent to carry an error payload, got: error={search_results[0].error!r}"
    )
    successful_after = [r for r in search_results[1:] if r.success and r.result and "Found 3 articles" in r.result]
    assert successful_after, (
        "Expected a successful 'search' ToolResultEvent after the initial failure "
        f"carrying the seeded success string. Got: {[(r.success, r.result) for r in search_results]}"
    )

    # Pin the causal order: correction must precede the retry invocation,
    # not merely co-occur. A broken implementation that emitted a
    # spurious ErrorCorrectionEvent after both tool calls would fail here.
    correction_index = traced_emitter.events.index(correction_event)
    second_invoke_index = traced_emitter.events.index(search_invocations[1])
    assert correction_index < second_invoke_index, (
        f"Expected ErrorCorrectionEvent (index {correction_index}) to precede the "
        f"second 'search' ToolInvokeEvent (index {second_invoke_index})."
    )

    # --- Fuzzy output ---
    # Split into a literal-substring check and a narrowed judge call.
    # "After a retry" is pinned by the trace assertions above; here we
    # confirm the output surfaces the search result to the user. The judge
    # receives the original user prompt so "for the user" has grounding.
    assert "Python 3.13" in (result.output or ""), f"Expected 'Python 3.13' in output, got: {result.output!r}"
    user_prompt_text = "Search for articles about Python 3.13. If the search fails, retry it."
    await assert_result_satisfies(
        result.output or "",
        "The output acknowledges that a search was performed for the user and reports that results were obtained.",
        user_prompt=user_prompt_text,
    )
