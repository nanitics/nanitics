"""Test configuration for the unit-test suite.

Default `addopts` in `pyproject.toml` excludes `docker`-marked tests; run
`just check docker=true` (or `just ci`) to include them. Real-service
testing lives in `validation/`, not here.
"""

from collections.abc import Iterator
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture(autouse=True)
def _patch_retry_sleep() -> Iterator[None]:
    """Patch retry/evaluator sleeps to a no-op for unit tests.

    The retry module and the LLM evaluator both sleep during exponential-backoff
    recovery. For deterministic unit tests these sleeps should be instant —
    production semantics are covered by the explicit ``test_retry.py``
    delay-sequence assertions, which use their own scoped ``patch`` contexts
    and shadow this autouse.

    Targets each module's local ``sleep`` binding (created by
    ``from asyncio import sleep``) rather than ``asyncio.sleep`` globally —
    a global patch would replace ``asyncio.sleep`` for every module and break
    the ``await asyncio.sleep(0)`` event-loop yield idiom used across the
    suite. If the seam breaks (e.g. a refactor switches to
    ``import asyncio``), ``mock.patch`` raises ``AttributeError`` at setup
    with the missing attribute path — clear enough without a bespoke check.
    """
    sleep_targets = [
        "nanitics.capabilities.errors.retry.sleep",
        "nanitics.capabilities.evaluation.llm_evaluator.sleep",
    ]
    patchers = [patch(target, new_callable=AsyncMock) for target in sleep_targets]
    for p in patchers:
        p.start()
    try:
        yield
    finally:
        for p in reversed(patchers):
            p.stop()
