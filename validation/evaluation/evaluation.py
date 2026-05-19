"""LLMEvaluator, ProgrammaticEvaluator, and CompositeEvaluator against real LLMs.

Three evaluator classes are exercised directly (not via an agent) so the
verdict logic can be pinned without confounding it with agent loop
behaviour. Real Anthropic calls flow through ``LLMEvaluator`` and the
``LLMRequestEvent`` / ``LLMResponseEvent`` pair with ``label="evaluator"``
is the load-bearing trace artifact proving the evaluator routed its
request through the instrumentation path.

Cases (parametrized per section):

  1. ``ProgrammaticEvaluator`` — pass: output satisfies every check;
     fail: output violates at least one check. Result verdict pinned.

  2. ``LLMEvaluator`` — pass: real LLM scores high-quality output above
     threshold → ACCEPT; fail: real LLM scores clearly-inadequate
     output below threshold → REVISE (or REJECT if a reject_threshold
     is configured, but here we pin the REVISE contract since
     threshold alone drives the decision).

  3. ``CompositeEvaluator`` — composes a programmatic and an LLM
     evaluator in that order. Pass only when both pass; fails when
     either the programmatic or the LLM evaluator rejects. The short-
     circuit contract is pinned by counting evaluator events.

Acceptance criteria:
  - Programmatic pass case: verdict ``ACCEPT``, score 1.0,
    ``evaluator_name == "programmatic"``.
  - Programmatic fail case: verdict ``REVISE``, score 0.0, feedback
    naming the failing check.
  - LLM pass case: verdict ``ACCEPT``, score >= threshold; trace
    contains an ``LLMRequestEvent`` and ``LLMResponseEvent`` labelled
    ``"evaluator"``.
  - LLM fail case: verdict ``REVISE`` (the threshold is exceeded from
    above), score < threshold; same trace invariants as the pass case.
  - Composite pass case: verdict ``ACCEPT`` from the LLM evaluator
    (the second evaluator's result is the one returned when both
    pass).
  - Composite fail-on-programmatic case: verdict ``REVISE`` from the
    programmatic evaluator; the LLM evaluator must NOT have been
    called (pinned by zero evaluator-labelled LLMRequestEvents in the
    trace).
  - Composite fail-on-llm case: verdict ``REVISE`` from the LLM
    evaluator after the programmatic accepts.
"""

from __future__ import annotations

import pytest

from nanitics.evaluation import (
    CompositeEvaluator,
    EvaluationCheck,
    EvaluationContext,
    EvaluationVerdict,
    LLMEvaluator,
    ProgrammaticEvaluator,
)
from nanitics.infrastructure import LLMRequestEvent, LLMResponseEvent
from nanitics.tracing import InMemoryEmitter
from validation.helpers import make_llm_client, run_with_retry

_ANALYSIS_CRITERIA = (
    "The output must be a clear, substantive analysis that contains at "
    "least one concrete data point (a number, percentage, or named fact) "
    "and explicitly addresses the original task. Empty, off-topic, or "
    "trivially short answers must score below 0.3.\n"
    "\n"
    "Scoring anchors (apply these strictly):\n"
    "- If the output contains at least one concrete data point AND "
    "addresses the task, score >= 0.75.\n"
    "- Do NOT deduct for missing source citations, time period, "
    "geographic scope, competitive dynamics, or forward-looking "
    "commentary — none of these are required by the criteria above. "
    "Score only against the criteria as written."
)


def _programmatic_checks() -> list[EvaluationCheck]:
    return [
        EvaluationCheck(
            name="minimum_length",
            check=lambda output: len(output) >= 40,
            feedback="Output must be at least 40 characters.",
        ),
        EvaluationCheck(
            name="contains_analysis",
            check=lambda output: "analysis" in output.lower(),
            feedback="Output must include the word 'analysis'.",
        ),
    ]


def _context() -> EvaluationContext:
    return EvaluationContext(messages=[], task_input="Analyze the market.")


# ---------------------------------------------------------------------------
# ProgrammaticEvaluator — pass and fail cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("output", "expected_verdict"),
    [
        pytest.param(
            "Detailed analysis of market trends with 15% growth drivers across the sector.",
            EvaluationVerdict.ACCEPT,
            id="pass_all_checks",
        ),
        pytest.param(
            "too short",
            EvaluationVerdict.REVISE,
            id="fail_length_and_keyword",
        ),
    ],
)
async def test_programmatic_evaluator_verdict(
    traced_emitter: InMemoryEmitter,
    output: str,
    expected_verdict: EvaluationVerdict,
) -> None:
    # The emitter is unused here but the fixture's finaliser still writes
    # a (trivial) trace — keeping per-script parity with the rest of the
    # validation suite.
    _ = traced_emitter

    evaluator = ProgrammaticEvaluator(checks=_programmatic_checks())
    result = await evaluator.evaluate(output, _context())

    assert result.verdict == expected_verdict, f"Expected verdict {expected_verdict}, got {result.verdict}"
    assert result.evaluator_name == "programmatic"
    if expected_verdict == EvaluationVerdict.ACCEPT:
        assert result.score == 1.0
    else:
        assert result.score == 0.0
        # Feedback must name at least one failing check — proves the branch,
        # not a generic fallback.
        assert "minimum_length" in (result.feedback or "") or "contains_analysis" in (result.feedback or "")


# ---------------------------------------------------------------------------
# LLMEvaluator — pass and fail against a real Anthropic call
# ---------------------------------------------------------------------------


_LLM_PASS_OUTPUT = (
    "Analysis: the cloud infrastructure market grew approximately 22% year-over-year, "
    "driven by enterprise AI workloads. Key vendors (AWS, Azure, GCP) captured over "
    "60% of net-new spend, with generative AI services contributing the largest share "
    "of incremental revenue."
)

_LLM_FAIL_OUTPUT = "Idk. Markets are fine I guess."


@pytest.mark.parametrize(
    ("output", "expected_verdict"),
    [
        pytest.param(_LLM_PASS_OUTPUT, EvaluationVerdict.ACCEPT, id="real_llm_pass"),
        pytest.param(_LLM_FAIL_OUTPUT, EvaluationVerdict.REVISE, id="real_llm_fail"),
    ],
)
async def test_llm_evaluator_verdict(
    traced_emitter: InMemoryEmitter,
    output: str,
    expected_verdict: EvaluationVerdict,
) -> None:
    judge_client = make_llm_client("anthropic")
    evaluator = LLMEvaluator(
        llm_client=judge_client,
        criteria=_ANALYSIS_CRITERIA,
        score_threshold=0.7,
        emitter=traced_emitter,
    )

    async def _run() -> object:
        traced_emitter.events.clear()
        return await evaluator.evaluate(output, _context())

    result = await run_with_retry(_run, max_attempts=3)

    assert result.verdict == expected_verdict, (
        f"Expected verdict {expected_verdict}, got {result.verdict} "
        f"(score={result.score!r}, feedback={result.feedback!r})"
    )
    assert result.evaluator_name == "llm"
    assert result.score is not None
    if expected_verdict == EvaluationVerdict.ACCEPT:
        assert result.score >= 0.7, f"Accepted output must score >= 0.7, got {result.score}"
    else:
        assert result.score < 0.7, f"Non-accepted output must score < 0.7, got {result.score}"

    # Trace pins: the evaluator labels its request/response events so we can
    # distinguish them from other LLM calls and confirm a real provider
    # round-trip happened.
    evaluator_requests = [e for e in traced_emitter.events if isinstance(e, LLMRequestEvent) and e.label == "evaluator"]
    evaluator_responses = [
        e for e in traced_emitter.events if isinstance(e, LLMResponseEvent) and e.label == "evaluator"
    ]
    assert evaluator_requests, "Expected at least one evaluator-labelled LLMRequestEvent."
    assert evaluator_responses, "Expected at least one evaluator-labelled LLMResponseEvent."


# ---------------------------------------------------------------------------
# CompositeEvaluator — short-circuit and pass-through
# ---------------------------------------------------------------------------


async def test_composite_passes_when_all_pass(traced_emitter: InMemoryEmitter) -> None:
    """Both evaluators accept — composite returns the last (LLM) result."""
    judge_client = make_llm_client("anthropic")
    programmatic = ProgrammaticEvaluator(checks=_programmatic_checks())
    llm_eval = LLMEvaluator(
        llm_client=judge_client,
        criteria=_ANALYSIS_CRITERIA,
        score_threshold=0.7,
        emitter=traced_emitter,
    )
    composite = CompositeEvaluator(evaluators=[programmatic, llm_eval])

    async def _run() -> object:
        traced_emitter.events.clear()
        return await composite.evaluate(_LLM_PASS_OUTPUT, _context())

    result = await run_with_retry(_run, max_attempts=3)

    assert result.verdict == EvaluationVerdict.ACCEPT, (
        f"Expected composite ACCEPT when both pass, got {result.verdict} (evaluator_name={result.evaluator_name!r})"
    )
    # Returned result is from the last evaluator — the LLM one.
    assert result.evaluator_name == "llm"

    # The LLM evaluator DID run — pinned by at least one evaluator-labelled
    # request in the trace.
    evaluator_requests = [e for e in traced_emitter.events if isinstance(e, LLMRequestEvent) and e.label == "evaluator"]
    assert evaluator_requests, "Expected the LLM evaluator to have been called."


async def test_composite_short_circuits_on_programmatic_failure(traced_emitter: InMemoryEmitter) -> None:
    """Programmatic fails — LLM evaluator must not be called (cost-saving contract)."""
    # A judge client is constructed but must not be invoked — the assertion
    # below proves the short-circuit.
    judge_client = make_llm_client("anthropic")
    programmatic = ProgrammaticEvaluator(checks=_programmatic_checks())
    llm_eval = LLMEvaluator(
        llm_client=judge_client,
        criteria=_ANALYSIS_CRITERIA,
        score_threshold=0.7,
        emitter=traced_emitter,
    )
    composite = CompositeEvaluator(evaluators=[programmatic, llm_eval])

    result = await composite.evaluate("too short", _context())

    assert result.verdict == EvaluationVerdict.REVISE, (
        f"Expected composite REVISE when programmatic fails, got {result.verdict}"
    )
    assert result.evaluator_name == "programmatic"

    # Zero evaluator-labelled LLM requests — the LLM evaluator was not
    # reached because the programmatic one short-circuited.
    evaluator_requests = [e for e in traced_emitter.events if isinstance(e, LLMRequestEvent) and e.label == "evaluator"]
    assert not evaluator_requests, (
        f"Expected the LLM evaluator to be short-circuited, but found "
        f"{len(evaluator_requests)} evaluator-labelled LLMRequestEvent(s)."
    )


async def test_composite_fails_when_llm_rejects(traced_emitter: InMemoryEmitter) -> None:
    """Programmatic passes, LLM rejects — composite returns the LLM's REVISE verdict."""
    judge_client = make_llm_client("anthropic")
    # Craft an output that passes the programmatic checks (long enough, has
    # "analysis") but is too vacuous for the LLM evaluator to accept.
    borderline_output = (
        "Analysis: things happened and some numbers were involved; overall it was "
        "reasonable but there is not much to say beyond that."
    )
    programmatic = ProgrammaticEvaluator(checks=_programmatic_checks())
    llm_eval = LLMEvaluator(
        llm_client=judge_client,
        criteria=_ANALYSIS_CRITERIA,
        score_threshold=0.7,
        emitter=traced_emitter,
    )
    composite = CompositeEvaluator(evaluators=[programmatic, llm_eval])

    async def _run() -> object:
        traced_emitter.events.clear()
        return await composite.evaluate(borderline_output, _context())

    result = await run_with_retry(_run, max_attempts=3)

    assert result.verdict == EvaluationVerdict.REVISE, (
        f"Expected composite REVISE when LLM rejects, got {result.verdict} (evaluator_name={result.evaluator_name!r})"
    )
    assert result.evaluator_name == "llm"
