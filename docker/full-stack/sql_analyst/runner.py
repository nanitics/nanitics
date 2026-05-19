"""Self-healing SQL analyst runner.

Composes the runner from SDK primitives:

- :class:`~nanitics.ReActAgent` writes SQL against the bundled schema
  using the :func:`~sql_analyst.tool.build_run_sql_tool` tool.
- :class:`~nanitics.Supervisor` observes each attempt, retries with
  targeted feedback when the last ``run_sql`` call errored or returned
  zero rows (:class:`~nanitics.PredicateTrigger`), and on the canonical
  path gates on a :class:`~sql_analyst.evaluator.GroundTruthEvaluator`
  wrapped by :class:`~nanitics.QualityTrigger`.
- Every invocation is wrapped by ``context.executor.execute(...)`` so
  the supervise-rewrite-evaluate loop surfaces in the Observatory UI.

Two endpoints are mounted under ``/runners/sql-analyst``:

- ``POST /runners/sql-analyst/ask`` drives either the canonical path
  (by ``question_id``) or the ad-hoc path (by free-form ``question``).
- ``GET /runners/sql-analyst/questions`` returns the sample-question
  catalog (ids + question strings only — canonical answers stay
  server-side so the ground-truth gate isn't trivially bypassable).
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from nanitics.composition import (
    PredicateTrigger,
    QualityTrigger,
    SupervisionAction,
    SupervisionDecision,
    Supervisor,
)
from nanitics.infrastructure import AnthropicLLMClient
from nanitics.strategies import ReActAgent
from nanitics.strategies.agents.base import AgentResult
from sql_analyst.bootstrap import ensure_analyst_schema
from sql_analyst.evaluator import GroundTruthEvaluator
from sql_analyst.questions import QUESTIONS, SampleQuestion
from sql_analyst.tool import LAST_TOOL_METADATA_STATE_KEY, build_run_sql_tool

if TYPE_CHECKING:
    from runners import ShellContext

RUNNER_SLUG = "sql-analyst"
RUNNER_TITLE = "Self-healing SQL analyst"
RUNNER_DESCRIPTION = (
    "Writes SQL against a bundled analytical schema, self-corrects "
    "under a Supervisor + programmatic ground-truth evaluator."
)

_SANDBOX_USER_ENV = "NANITICS_SQL_ANALYST_SANDBOX_USER"
_SANDBOX_PASSWORD_ENV = "NANITICS_SQL_ANALYST_SANDBOX_PASSWORD"
_POSTGRES_DSN_ENV = "POSTGRES_DSN"
_ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"
_LLM_MODEL_ENV = "NANITICS_LLM_MODEL"

# Marker attribute on the FastAPI app so ``register`` is idempotent —
# a second call on the same app does not duplicate routes or stack
# multiple schema-bootstrap startup hooks.
_REGISTRATION_MARKER = "_sql_analyst_registered"

MAX_AGENT_ITERATIONS = 10
"""Max ReAct iterations per agent run.

The agent must discover the schema via information_schema before writing
its answer query, which costs 2–3 tool calls before the first attempt.
Ten iterations keeps the trace bounded while leaving headroom.
"""

SUPERVISOR_MAX_RETRIES = 3
"""Supervisor retry budget."""

AD_HOC_MAX_RETRIES = 1
"""Ad-hoc path retries once on error; value-level evaluation is skipped."""


SYSTEM_PROMPT = """\
You are a SQL analyst. You have access to a PostgreSQL database \
through the ``run_sql`` tool. The schema is not documented here — \
discover it by querying the information schema before writing your \
answer query.

Discovery hints
---------------

Useful queries to explore the schema:

    SELECT table_name FROM information_schema.tables
      WHERE table_schema = 'public' AND table_type = 'BASE TABLE';

    SELECT column_name, data_type
      FROM information_schema.columns
      WHERE table_name = '<table>';

You can only SELECT — writes are denied at the database-role level.

Response protocol
-----------------

Your final response must be a single JSON object matching this schema:

    {"answer": <value>, "sql": <your final answer SQL>, "rowcount": <int>}

Use only that JSON, no prose before or after, no code fences. The \
``answer`` field is the value the question asks for:

- A scalar for scalar questions (e.g., ``37`` or ``1234.50``).
- A list of row arrays for row-set questions (e.g.,
  ``[[12, 42.50], [15, 38.00]]``). Column order must match the order
  named in the question.

Tools
-----

You have one tool, ``run_sql(sql)``. Call it to execute a query and \
receive the columns, rows, and rowcount. On error the result's \
``content`` starts with ``ERROR:`` — read the error and rewrite your \
query. Do not call ``run_sql`` with anything other than a single \
``SELECT`` — writes raise a permission error.
"""


# ── Request / response models ──────────────────────────────


class AskRequest(BaseModel):
    """Request body for ``POST /runners/sql-analyst/ask``.

    Exactly one of ``question_id`` (canonical path, full supervisor) or
    ``question`` (ad-hoc path, error-trigger only) must be set. When
    both are set, ``question_id`` wins and ``question`` is ignored.
    """

    model_config = ConfigDict(extra="forbid")

    question_id: str | None = None
    question: str | None = None


class InterventionSummary(BaseModel):
    """Summary of one supervisor intervention, surfaced in the response."""

    model_config = ConfigDict(frozen=True)

    action: str
    trigger_name: str
    feedback: str | None = None
    attempt: int


class AskResponse(BaseModel):
    """Response body for ``POST /runners/sql-analyst/ask``."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    accepted: bool
    attempts: int
    final_sql: str | None = None
    answer: Any = None
    rowcount: int | None = None
    interventions: list[InterventionSummary]


class QuestionListItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    question: str


class QuestionListResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    questions: list[QuestionListItem]


# ── Helpers ────────────────────────────────────────────────


def _derive_sandbox_dsn(privileged_dsn: str, user: str, password: str) -> str:
    """Rebuild a DSN with a different user/password, same host/port/db.

    The privileged DSN wired into ``POSTGRES_DSN`` carries the app's
    credentials. The sandbox role connects to the same database on the
    same host — only the credentials change.
    """
    from urllib.parse import quote, urlsplit, urlunsplit

    parts = urlsplit(privileged_dsn)
    userinfo = f"{quote(user, safe='')}:{quote(password, safe='')}"
    netloc = parts.hostname or ""
    if parts.port is not None:
        netloc = f"{netloc}:{parts.port}"
    netloc = f"{userinfo}@{netloc}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _resolve_privileged_dsn() -> str:
    """Resolve the privileged Postgres DSN from env (same scheme as the app shell)."""
    dsn = os.environ.get(_POSTGRES_DSN_ENV)
    if dsn:
        return dsn
    user = os.environ.get("POSTGRES_USER", "nanitics")
    password = os.environ.get("POSTGRES_PASSWORD", "nanitics-local")
    db = os.environ.get("POSTGRES_DB", "nanitics")
    return f"postgresql://{user}:{password}@postgres:5432/{db}"


def _resolve_sandbox_credentials() -> tuple[str, str]:
    """Read sandbox user + password from env. Missing values raise."""
    user = os.environ.get(_SANDBOX_USER_ENV)
    password = os.environ.get(_SANDBOX_PASSWORD_ENV)
    if not user:
        raise RuntimeError(f"{_SANDBOX_USER_ENV} is required for the SQL-analyst runner.")
    if not password:
        raise RuntimeError(f"{_SANDBOX_PASSWORD_ENV} is required for the SQL-analyst runner.")
    return user, password


def _opt_in_caching(client: Any) -> Any:
    """Reconstruct an ``AnthropicLLMClient`` with ``enable_caching=True``.

    ``AnthropicLLMClient.api_key`` is not a public attribute, so the
    wrapper reads ``ANTHROPIC_API_KEY`` from env — the same variable the
    app-shell factory required at startup. Non-Anthropic clients pass
    through unchanged.
    """
    if not isinstance(client, AnthropicLLMClient):
        return client
    model = client.model or os.environ.get(_LLM_MODEL_ENV)
    api_key = os.environ.get(_ANTHROPIC_KEY_ENV)
    if not model or not api_key:
        # Cannot reconstruct — fall back to the client we were given.
        return client
    return AnthropicLLMClient(model=model, api_key=api_key, enable_caching=True)


def _make_query_error_predicate(
    *,
    tool_state: dict[str, Any],
    allow_empty_result: bool,
) -> Any:
    """Build the predicate for the ``query_error_or_empty`` trigger.

    Args:
        tool_state: The per-run tool state dict the ``run_sql`` tool
            writes the latest metadata into. The predicate reads it
            after every agent run.
        allow_empty_result: When ``True``, a zero-row result is *not*
            a retry trigger (used for questions whose canonical answer
            is zero rows). When ``False``, the predicate retries on
            empty results too.
    """

    def predicate(result: AgentResult, task: str) -> SupervisionDecision | None:
        metadata = tool_state.get(LAST_TOOL_METADATA_STATE_KEY)
        if metadata is None:
            # The agent never invoked ``run_sql`` this attempt; let
            # the next trigger (or the accept path) decide.
            return None
        if metadata.get("error") is True:
            return SupervisionDecision(
                action=SupervisionAction.RETRY,
                feedback=(
                    "The last SQL you ran raised "
                    f"{metadata.get('error_type', 'an error')}. "
                    "Read the tool's ERROR output and rewrite the query "
                    "so it parses, runs under the sandbox role, and "
                    "returns rows."
                ),
                trigger_name="query_error_or_empty",
            )
        if not allow_empty_result and metadata.get("rowcount") == 0:
            return SupervisionDecision(
                action=SupervisionAction.RETRY,
                feedback=(
                    "The last query returned zero rows but the question "
                    "expects at least one row. Check your filters and "
                    "JOIN conditions and rewrite the query."
                ),
                trigger_name="query_error_or_empty",
            )
        return None

    return predicate


def _expected_non_empty(question: SampleQuestion | None) -> bool:
    """Whether the expected answer for *question* is at least one row.

    Scalar answers are always "non-empty" (a count of zero is a valid
    scalar). Only zero-row row-sets and the ad-hoc path (no catalog
    question) allow the empty-result branch of the predicate to quiet.
    """
    if question is None:
        return False
    expected = question.expected
    from sql_analyst.questions import ExpectedRowSet

    if isinstance(expected, ExpectedRowSet):
        return len(expected.rows) > 0
    return True


def _parse_envelope(output: str | None) -> dict[str, Any] | None:
    """Mirror the evaluator's envelope parser for the endpoint's benefit."""
    if not isinstance(output, str) or not output.strip():
        return None
    text = output.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _intervention_summaries(
    interventions: list[SupervisionDecision],
) -> list[InterventionSummary]:
    """Map :class:`SupervisionDecision` to :class:`InterventionSummary`.

    ``attempt`` is derived from position — the Supervisor records one
    intervention per non-accept decision, in order.
    """
    return [
        InterventionSummary(
            action=decision.action.value,
            trigger_name=decision.trigger_name,
            feedback=decision.feedback,
            attempt=index + 1,
        )
        for index, decision in enumerate(interventions)
    ]


# ── Registration ───────────────────────────────────────────


def register(app: FastAPI, context: ShellContext) -> None:
    """Mount the SQL-analyst runner onto *app*.

    Idempotent: a second call on the same app is a no-op (no duplicate
    routes, no stacked startup hooks). The app-shell lifespan invokes
    ``register`` exactly once, but the idempotence guard makes the
    runner safe to re-register from tests.
    """
    if getattr(app.state, _REGISTRATION_MARKER, False):
        return
    app.state.__setattr__(_REGISTRATION_MARKER, True)

    sandbox_user, sandbox_password = _resolve_sandbox_credentials()
    privileged_dsn = _resolve_privileged_dsn()
    sandbox_dsn = _derive_sandbox_dsn(privileged_dsn, sandbox_user, sandbox_password)

    # Schedule the schema bootstrap on app startup. The closure captures
    # ``context.pool`` and the sandbox credentials; the coroutine runs
    # exactly once after registration and before the first request.
    @app.on_event("startup")
    async def _bootstrap_analyst_schema() -> None:
        await ensure_analyst_schema(
            context.pool,
            sandbox_role=sandbox_user,
            sandbox_password=sandbox_password,
        )

    run_sql_tool = build_run_sql_tool(sandbox_dsn=sandbox_dsn)

    @app.post("/runners/sql-analyst/ask", response_model=AskResponse)
    async def ask(request: AskRequest) -> AskResponse:
        if request.question_id is None and request.question is None:
            raise HTTPException(
                status_code=422,
                detail="exactly one of question_id or question must be set",
            )

        sample_question: SampleQuestion | None = None
        if request.question_id is not None:
            for candidate in QUESTIONS:
                if candidate.id == request.question_id:
                    sample_question = candidate
                    break
            if sample_question is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"unknown question_id: {request.question_id!r}",
                )
            task = sample_question.question
            max_retries = SUPERVISOR_MAX_RETRIES
        else:
            # Ad-hoc path: request.question is non-None (checked above).
            task = request.question or ""
            max_retries = AD_HOC_MAX_RETRIES

        client = _opt_in_caching(context.build_client())

        async def _supervised_run(emitter: Any, _run_id: str) -> Any:
            # The per-run tool state holds the latest ``run_sql``
            # metadata so the error-catch predicate can read it
            # without re-parsing tool-result message content. Seeded
            # with a ``None`` placeholder so the dict is truthy — the
            # SDK's :class:`ToolRegistry` constructor evaluates
            # ``tool_state or {}`` and would otherwise substitute a
            # fresh empty dict, breaking the shared-reference contract.
            tool_state: dict[str, Any] = {LAST_TOOL_METADATA_STATE_KEY: None}
            agent = ReActAgent(
                name="sql-analyst",
                llm_client=client,
                emitter=emitter,
                system_prompt=SYSTEM_PROMPT,
                tools=[run_sql_tool],
                max_iterations=MAX_AGENT_ITERATIONS,
                tool_state=tool_state,
            )
            error_predicate = _make_query_error_predicate(
                tool_state=tool_state,
                allow_empty_result=not _expected_non_empty(sample_question),
            )
            triggers: list[Any] = [
                PredicateTrigger(
                    name="query_error_or_empty",
                    predicate=error_predicate,
                ),
            ]
            if sample_question is not None:
                triggers.append(QualityTrigger(GroundTruthEvaluator(sample_question)))

            supervisor = Supervisor(
                triggers=triggers,
                emitter=emitter,
                max_retries=max_retries,
            )
            return await supervisor.supervise(agent, task)

        run_id, supervised = await context.executor.execute(
            _supervised_run,
            metadata={
                "runner": "sql-analyst",
                "question_id": request.question_id,
                "ad_hoc": request.question_id is None,
            },
        )

        envelope = _parse_envelope(supervised.result.output)
        answer = envelope.get("answer") if envelope else None
        final_sql = envelope.get("sql") if envelope else None
        rowcount = envelope.get("rowcount") if envelope else None

        return AskResponse(
            run_id=run_id,
            accepted=supervised.accepted,
            attempts=supervised.total_attempts,
            final_sql=final_sql if isinstance(final_sql, str) else None,
            answer=answer,
            rowcount=rowcount if isinstance(rowcount, int) else None,
            interventions=_intervention_summaries(supervised.interventions),
        )

    @app.get("/runners/sql-analyst/questions", response_model=QuestionListResponse)
    async def list_questions() -> QuestionListResponse:
        return QuestionListResponse(
            questions=[QuestionListItem(id=q.id, question=q.question) for q in QUESTIONS],
        )
