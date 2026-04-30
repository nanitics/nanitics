"""Runner-registration seam for the full-stack compose.

Each showcase runner appends one :class:`RunnerRegistration` to
:data:`REGISTRATIONS`. The shell iterates the list at lifespan startup
and calls each registration's ``register(app, context)`` callable. No
plugin discovery, no config file, no base class — a list in a Python
file.

Current registrations:

- ``sql-analyst``
- ``auction-routing``
- ``judge-routing``
- ``self-improver``
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncpg
    from fastapi import FastAPI

    from nanitics import LLMClient, PersistentTraceStore, TracedExecutor


@dataclass(frozen=True)
class ShellContext:
    """Shared infrastructure handed to each runner at registration time.

    Runners never construct their own ``TracedExecutor``,
    ``PersistentTraceStore``, or asyncpg pool. All shared infrastructure
    comes from this context so every runner writes traces to the same
    store and shares the same executor.

    Attributes:
        executor: The shared :class:`TracedExecutor` backing every run.
        trace_store: The shared :class:`PersistentTraceStore` (Postgres).
        pool: The shared :class:`asyncpg.Pool` for runner-owned DB use.
        build_client: Factory that returns a fresh :class:`LLMClient` —
            runners that need to opt in to caching or swap providers
            call this to obtain their own client instance.
    """

    executor: TracedExecutor
    trace_store: PersistentTraceStore
    pool: asyncpg.Pool
    build_client: Callable[[], LLMClient]


@dataclass(frozen=True)
class RunnerRegistration:
    """One showcase runner entry.

    Conventions:

    - ``slug`` is kebab-case (e.g., ``sql-analyst``) and matches the
      package the runner lives in (``sql_analyst/``, with the
      underscore-to-hyphen convention).
    - ``register`` mounts routes under ``/runners/<slug>`` unless there
      is a good reason not to; sub-routers are fine.
    - ``register`` is called exactly once per process lifetime at
      lifespan startup and should be idempotent-safe.

    Attributes:
        slug: kebab-case identifier; keys the ``/runners`` index page.
        title: short human label shown on ``/runners``.
        description: one-sentence summary shown on ``/runners``.
        register: Callable invoked during lifespan startup to attach
            the runner's routes to the FastAPI app.
    """

    slug: str
    title: str
    description: str
    register: Callable[[FastAPI, ShellContext], None]


REGISTRATIONS: list[RunnerRegistration] = []

# The ``sql_analyst`` package lives alongside this file. Its
# ``runner.register`` callable is imported here and wrapped in the one
# :class:`RunnerRegistration` entry the seam expects.
from sql_analyst.runner import RUNNER_DESCRIPTION as _SQL_ANALYST_DESCRIPTION  # noqa: E402
from sql_analyst.runner import RUNNER_SLUG as _SQL_ANALYST_SLUG  # noqa: E402
from sql_analyst.runner import RUNNER_TITLE as _SQL_ANALYST_TITLE  # noqa: E402
from sql_analyst.runner import register as _sql_analyst_register  # noqa: E402

REGISTRATIONS.append(
    RunnerRegistration(
        slug=_SQL_ANALYST_SLUG,
        title=_SQL_ANALYST_TITLE,
        description=_SQL_ANALYST_DESCRIPTION,
        register=_sql_analyst_register,
    )
)

# ``auction-routing``. Four specialists bid on each incoming request;
# no-confident-bid branches to HITL.
from auction_routing.runner import RUNNER_DESCRIPTION as _AUCTION_DESCRIPTION  # noqa: E402
from auction_routing.runner import RUNNER_SLUG as _AUCTION_SLUG  # noqa: E402
from auction_routing.runner import RUNNER_TITLE as _AUCTION_TITLE  # noqa: E402
from auction_routing.runner import register as _register_auction_routing  # noqa: E402

REGISTRATIONS.append(
    RunnerRegistration(
        slug=_AUCTION_SLUG,
        title=_AUCTION_TITLE,
        description=_AUCTION_DESCRIPTION,
        register=_register_auction_routing,
    )
)

# ``judge-routing``. Four tool-using specialists are routed by a single
# comparative-judgment LLM call; the winning specialist answers using
# in-memory billing/technical/account/policy fixtures.
from judge_routing.runner import RUNNER_DESCRIPTION as _JUDGE_DESCRIPTION  # noqa: E402
from judge_routing.runner import RUNNER_SLUG as _JUDGE_SLUG  # noqa: E402
from judge_routing.runner import RUNNER_TITLE as _JUDGE_TITLE  # noqa: E402
from judge_routing.runner import register as _register_judge_routing  # noqa: E402

REGISTRATIONS.append(
    RunnerRegistration(
        slug=_JUDGE_SLUG,
        title=_JUDGE_TITLE,
        description=_JUDGE_DESCRIPTION,
        register=_register_judge_routing,
    )
)

# ``self-improver``. A deliberately-imperfect task agent runs end-to-end;
# a critic reads its trace via the SDK's trace API and emits ranked
# improvement proposals.
from self_improver.runner import RUNNER_DESCRIPTION as _SELF_IMPROVER_DESCRIPTION  # noqa: E402
from self_improver.runner import RUNNER_SLUG as _SELF_IMPROVER_SLUG  # noqa: E402
from self_improver.runner import RUNNER_TITLE as _SELF_IMPROVER_TITLE  # noqa: E402
from self_improver.runner import register as _register_self_improver  # noqa: E402

REGISTRATIONS.append(
    RunnerRegistration(
        slug=_SELF_IMPROVER_SLUG,
        title=_SELF_IMPROVER_TITLE,
        description=_SELF_IMPROVER_DESCRIPTION,
        register=_register_self_improver,
    )
)
