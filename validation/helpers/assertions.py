"""Assertion helpers for validation scripts.

- :func:`assert_trace_contains`: filter :class:`EventEmitter` events by type/predicate.
- :func:`assert_result_satisfies`: LLM-as-judge binary assertion.

Both raise :class:`AssertionError` on failure so they integrate with pytest.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel, Field, ValidationError

from nanitics import LLMClient, Message, TraceEvent

if TYPE_CHECKING:
    from nanitics import InMemoryEmitter

E = TypeVar("E", bound=TraceEvent)


def assert_trace_contains(
    emitter: InMemoryEmitter,
    event_type: type[E],
    predicate: Callable[[E], bool] | None = None,
) -> E:
    """Return the first event of ``event_type`` (optionally matching ``predicate``).

    Args:
        emitter: The in-memory emitter whose ``events`` list is scanned.
        event_type: A :class:`TraceEvent` subclass to match via ``isinstance``.
        predicate: Optional additional filter over type-matching events.

    Returns:
        The first matching event. Validation scripts often read fields off
        the matched event, so we return it rather than ``None``.

    Raises:
        AssertionError: If no event matches. The message lists the event
            type counts so the author can see whether the event is missing
            or the predicate failed.
    """
    matched_type: list[E] = [e for e in emitter.events if isinstance(e, event_type)]
    if not matched_type:
        counts = Counter(type(e).__name__ for e in emitter.events)
        summary = ", ".join(f"{name}={n}" for name, n in sorted(counts.items())) or "(none)"
        raise AssertionError(f"Expected at least one {event_type.__name__} event; found types: {summary}")
    if predicate is None:
        return matched_type[0]
    for event in matched_type:
        if predicate(event):
            return event
    raise AssertionError(f"Found {len(matched_type)} {event_type.__name__} events but none satisfied the predicate.")


class JudgeVerdict(BaseModel):
    """Schema for the LLM-as-judge's binary verdict."""

    pass_: bool = Field(alias="pass")
    reason: str


_JUDGE_SYSTEM_PROMPT = (
    "You are a strict binary judge. Your task is to evaluate whether a "
    "produced output satisfies a stated criterion.\n\n"
    "Return JSON in the exact shape:\n"
    '{"pass": <boolean>, "reason": "<one or two sentences>"}\n\n'
    "Do not return any other text. Judge against the criterion as stated. "
    "Do not impose requirements the criterion does not state. If the "
    "criterion is ambiguous, judge against the narrower reading that is "
    "consistent with the output — a criterion is satisfied when the output "
    "meets what it asks for, not when the output exceeds what a stricter "
    "version might ask for.\n\n"
    "If the output does not meet the criterion as stated, return pass=false. "
    "If it does, return pass=true with a one-line acknowledgement.\n\n"
    'Example of "satisfies":\n'
    '  Criterion: "The output tells the user what happened and reports the '
    'resulting value."\n'
    '  Output: "I ran the calculation; the result is 42."\n'
    "  Verdict: pass=true. The output narrates what happened (ran the "
    "calculation) and reports the resulting value (42). The absence of "
    "showing the steps, or justifying the 42, is not a reason to fail — "
    "the criterion did not ask for those."
)


def _format_judge_user_prompt(criteria: str, output: str, user_prompt: str | None) -> str:
    if user_prompt:
        return f"User's original request:\n{user_prompt}\n\nCriterion:\n{criteria}\n\nOutput to evaluate:\n{output}"
    return f"Criterion:\n{criteria}\n\nOutput to evaluate:\n{output}"


async def assert_result_satisfies(
    output: str,
    criteria: str,
    *,
    user_prompt: str | None = None,
    judge: LLMClient | None = None,
) -> None:
    """LLM-as-judge binary assertion: does ``output`` satisfy ``criteria``?

    The default judge is constructed via
    ``make_llm_client(DEFAULT_JUDGE_PROVIDER, model=DEFAULT_JUDGE_MODEL)``
    (cheap, fast Haiku). Pass ``judge`` explicitly to override.

    Pass ``user_prompt`` whenever the criterion references context that
    only resolves against the original prompt — phrases like "the user's
    request", "the user's question", or "what the user asked". The judge
    otherwise sees only ``(criterion, output)`` and must invent a reading
    for such phrases. Supplying ``user_prompt`` gives the judge the
    grounding to evaluate the criterion as written rather than guess at
    it. When the criterion is fully self-contained (e.g. "reports 720 as
    the factorial of 6"), leave ``user_prompt`` as ``None``.

    Raises:
        AssertionError: On ``pass=false``. Message format:
            ``"Judge failed: {reason}"``.
        RuntimeError: If the judge returns malformed JSON — this is an
            infrastructure error, not a judgment failure.
    """
    if judge is None:
        # Local import to avoid a circular import at module load.
        from validation.helpers.llm import (
            DEFAULT_JUDGE_MODEL,
            DEFAULT_JUDGE_PROVIDER,
            make_llm_client,
        )

        judge = make_llm_client(DEFAULT_JUDGE_PROVIDER, model=DEFAULT_JUDGE_MODEL)

    response = await judge.generate(
        system_prompt=_JUDGE_SYSTEM_PROMPT,
        messages=[Message(role="user", content=_format_judge_user_prompt(criteria, output, user_prompt))],
        output_schema=JudgeVerdict,
    )

    verdict: JudgeVerdict | None = None
    if isinstance(response.parsed, JudgeVerdict):
        verdict = response.parsed
    elif response.content:
        try:
            verdict = JudgeVerdict.model_validate(json.loads(response.content))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise RuntimeError(f"Judge returned malformed response: {response.content}") from exc

    if verdict is None:
        raise RuntimeError(f"Judge returned malformed response: {response.content!r}")

    if not verdict.pass_:
        raise AssertionError(f"Judge failed: {verdict.reason}")
