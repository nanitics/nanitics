"""Retrospective self-improver showcase runner.

One of three showcase runners inside ``docker/full-stack/``. The runner
slug is ``self-improver`` and the endpoint is mounted at
``/runners/self-improver``. See the package ``README.md`` for the
runner-facing narrative.

Two runs per invocation:

1. The **task run** — a deliberately-imperfect single-shot
   :class:`~nanitics.ReActAgent` with two file-read tools over a small
   bundled corpus, wrapped in
   ``context.executor.execute(...)`` so its trace lands in the shared
   :class:`~nanitics.PersistentTraceStore`. If the caller passes
   ``trace_id`` in the request body, the task phase is skipped and the
   referenced trace is critiqued directly.
2. The **critic run** — :func:`self_improver.advisor.analyze` called against
   the task trace's events, also wrapped in an
   ``executor.execute(...)`` so the critic's own specialist fan-out
   surfaces in Observatory alongside the task run. The response body
   returns both trace ids plus the full :class:`~self_improver.advisor.AdvisorReport`.

The critic-side reader uses :func:`nanitics.trace_events_from_stored`
to turn the store's ``list[StoredTraceEvent]`` rows back into the typed
``list[TraceEvent]`` ingested by ``analyze()``.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from nanitics import (
    AnthropicLLMClient,
    LLMClient,
    ReActAgent,
    tool,
    trace_events_from_stored,
)
from nanitics.infrastructure.observability.emitter import EventEmitter
from self_improver.advisor import analyze as advisor_analyze
from self_improver.advisor.analyze import AdvisorReport

if TYPE_CHECKING:
    from runners import ShellContext

# ── Module constants ──────────────────────────────────────────

RUNNER_SLUG = "self-improver"
RUNNER_TITLE = "Retrospective self-improver"
RUNNER_DESCRIPTION = (
    "A task agent runs end-to-end; a critic reads the trace via the SDK's "
    "trace API and emits ranked improvement proposals."
)
RUNNER_PREFIX = "/runners/self-improver"

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
"""Absolute path to the bundled markdown corpus directory."""

DEFAULT_TASK_INPUT = (
    "Summarise how observability works in the Nanitics SDK. Cover event "
    "emission, storage, redaction, and how an external consumer reads a "
    "trace after a run completes."
)
"""Canned task input the task agent receives when the request body omits both
``task_input`` and ``trace_id``."""

TASK_ITERATION_CAP = 6
"""Maximum iterations the task agent is allowed.

Deliberately tight for a multi-doc synthesis task; the advisor's
``iteration-budgets`` rubric is expected to flag this."""

TASK_AGENT_NAME = "self-improver-task"
TASK_SYSTEM_PROMPT = "You are a research assistant. Answer the user's question using the tools available to you."
"""Deliberately thin system prompt — no guidance on decomposition, on
reading multiple docs before answering, on citation style. The advisor's
``prompts`` rubric is expected to flag this."""

_ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"
_LLM_MODEL_ENV = "NANITICS_LLM_MODEL"


# ── Corpus tools ──────────────────────────────────────────────


@tool(
    name="list_bundled_docs",
    description="Lists available documents.",
)
async def list_bundled_docs() -> str:
    """Return a newline-separated manifest of corpus filenames.

    Deliberately returns filenames only, without per-file summaries — the
    advisor's ``tool-descriptions`` rubric is expected to flag the
    uselessly thin description paired with a manifest-only return.
    """
    if not CORPUS_DIR.is_dir():
        raise FileNotFoundError(f"corpus directory missing: {CORPUS_DIR}")
    names = sorted(p.name for p in CORPUS_DIR.iterdir() if p.is_file() and p.suffix == ".md")
    return "\n".join(names)


@tool(
    name="read_bundled_doc",
    description="Reads a document by filename.",
)
async def read_bundled_doc(filename: str) -> str:
    """Return the text of one corpus file.

    The ``filename`` argument must resolve to a path inside ``CORPUS_DIR``.
    Any attempt to escape the corpus (``../etc/passwd``, absolute paths,
    symlinks pointing outside) raises :class:`ValueError` with the
    offending input in the message — callers see a clear failure rather
    than a silent read of some unrelated file.
    """
    candidate = (CORPUS_DIR / filename).resolve()
    try:
        candidate.relative_to(CORPUS_DIR.resolve())
    except ValueError as exc:
        raise ValueError(f"refusing to read outside corpus: {filename!r}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"corpus file not found: {filename!r}")
    return candidate.read_text(encoding="utf-8")


# ── Request / response models ─────────────────────────────────


class RunRequest(BaseModel):
    """Request body for ``POST /runners/self-improver/run``.

    Both fields optional. When ``trace_id`` is set, the task phase is
    skipped and the referenced trace is critiqued directly. When absent,
    ``task_input`` (or :data:`DEFAULT_TASK_INPUT` if also absent) drives
    the bundled task agent.
    """

    model_config = ConfigDict(extra="forbid")

    task_input: str | None = Field(default=None)
    trace_id: str | None = Field(default=None)


class RunResponse(BaseModel):
    """Response body for ``POST /runners/self-improver/run``."""

    model_config = ConfigDict(frozen=True)

    task_run_id: str | None
    critic_run_id: str
    task_trace_id: str
    report: AdvisorReport


# ── Factories ─────────────────────────────────────────────────


def build_task_agent(llm_client: LLMClient, emitter: EventEmitter) -> ReActAgent:
    """Construct the deliberately-thin task agent.

    Args:
        llm_client: Shared LLM client for the task run. The endpoint
            passes the plain (non-caching) client — a 6-iteration single
            session does not amortise cache-write cost.
        emitter: Event emitter provided by :class:`TracedExecutor` so
            events land under the task run's fresh trace id.

    Returns:
        A :class:`ReActAgent` with the two corpus tools, the thin system
        prompt, and :data:`TASK_ITERATION_CAP` iterations.
    """
    return ReActAgent(
        name=TASK_AGENT_NAME,
        llm_client=llm_client,
        emitter=emitter,
        system_prompt=TASK_SYSTEM_PROMPT,
        tools=[list_bundled_docs, read_bundled_doc],
        max_iterations=TASK_ITERATION_CAP,
    )


def _build_caching_client(base_client: LLMClient) -> LLMClient:
    """Return a caching-enabled client for the critic, when possible.

    The critic's three specialists share a long trace prefix, so cache
    writes amortise. When ``base_client`` is not an
    :class:`AnthropicLLMClient` (e.g., OpenAI at launch), caching is not
    shipped — return the original client. When it is Anthropic but the
    env vars needed to reconstruct are missing, return the original
    client; this is a defensive fallback documented in the README.
    """
    if not isinstance(base_client, AnthropicLLMClient):
        return base_client
    model = base_client.model or os.environ.get(_LLM_MODEL_ENV)
    api_key = os.environ.get(_ANTHROPIC_KEY_ENV)
    if not model or not api_key:
        return base_client
    return AnthropicLLMClient(model=model, api_key=api_key, enable_caching=True)


# ── Orchestration ─────────────────────────────────────────────


async def _run_task_phase(
    context: ShellContext,
    task_input: str,
) -> tuple[str, str]:
    """Run the task agent under ``context.executor``.

    :class:`TracedExecutor` generates ``run_id`` and ``trace_id`` as two
    fresh UUIDs — the run record ties them together, and events are
    indexed by ``trace_id``. ``executor.execute`` only returns the
    ``run_id``, so the factory captures the ``trace_id`` off the
    emitter and the caller reads it out of a mutable accumulator.

    Returns:
        ``(task_run_id, task_trace_id)``.
    """
    task_client = context.build_client()
    captured: dict[str, str] = {}

    async def _factory(emitter: EventEmitter, _run_id: str) -> Any:
        captured["trace_id"] = emitter.trace_id
        agent = build_task_agent(task_client, emitter)
        return await agent.run(task_input)

    run_id, _ = await context.executor.execute(
        _factory,
        metadata={"runner": RUNNER_SLUG, "phase": "task"},
    )
    return run_id, captured["trace_id"]


async def _run_critic_phase(
    context: ShellContext,
    task_trace_id: str,
) -> tuple[str, AdvisorReport]:
    """Load the task trace, convert to typed events, run the advisor.

    Raises:
        HTTPException: 404 when ``task_trace_id`` has no stored events.
    """
    stored_events = await context.trace_store.get_span_tree(task_trace_id)
    if not stored_events:
        raise HTTPException(
            status_code=404,
            detail={"error": "trace_not_found", "trace_id": task_trace_id},
        )
    typed_events = trace_events_from_stored(stored_events)

    critic_client = _build_caching_client(context.build_client())

    async def _factory(emitter: EventEmitter, _run_id: str) -> AdvisorReport:
        return await advisor_analyze(
            typed_events,
            llm_client=critic_client,
            emitter=emitter,
        )

    critic_run_id, report = await context.executor.execute(
        _factory,
        metadata={
            "runner": RUNNER_SLUG,
            "phase": "critic",
            "task_trace_id": task_trace_id,
        },
    )
    return critic_run_id, report


async def run_task_and_critique(
    context: ShellContext,
    request: RunRequest,
) -> RunResponse:
    """Drive task + critic through the two-run orchestration.

    1. Task phase — skipped when ``request.trace_id`` is provided. When
       driven, the bundled task agent runs under a fresh
       :class:`TracedExecutor` run with its events persisted to the
       shared trace store.
    2. Critic phase — reads the task trace via
       :meth:`PersistentTraceStore.get_span_tree`, converts the stored
       rows to typed events via :func:`trace_events_from_stored`, and
       hands them to :func:`self_improver.advisor.analyze` wrapped in a
       second :class:`TracedExecutor` run.

    Any advisor or task-agent exception propagates — the showcase
    endpoint surfaces failures rather than masking them.
    """
    if request.trace_id is not None:
        task_run_id: str | None = None
        task_trace_id = request.trace_id
    else:
        task_input = request.task_input or DEFAULT_TASK_INPUT
        task_run_id, task_trace_id = await _run_task_phase(context, task_input)

    critic_run_id, report = await _run_critic_phase(context, task_trace_id)

    return RunResponse(
        task_run_id=task_run_id,
        critic_run_id=critic_run_id,
        task_trace_id=task_trace_id,
        report=report,
    )


# ── Router and registration ───────────────────────────────────


def build_runner_router(
    handler: Callable[[RunRequest], Awaitable[RunResponse]],
) -> APIRouter:
    """Construct the runner's ``APIRouter`` with the ``/run`` POST route.

    Args:
        handler: The orchestration coroutine bound to the active
            :class:`ShellContext`. The router is free of global state;
            the closure holds the context reference.

    Returns:
        A :class:`fastapi.APIRouter` with one route.
    """
    router = APIRouter()

    @router.post("/run", response_model=RunResponse)
    async def run(request: RunRequest) -> RunResponse:
        return await handler(request)

    return router


def register(app: FastAPI, context: ShellContext) -> None:
    """Mount the self-improver runner onto *app*.

    Closes over *context* so the handler can reach the shared
    :class:`TracedExecutor`, :class:`PersistentTraceStore`, and the
    ``build_client`` factory without module-level globals.
    """

    async def _handler(request: RunRequest) -> RunResponse:
        return await run_task_and_critique(context, request)

    app.include_router(build_runner_router(_handler), prefix=RUNNER_PREFIX)
