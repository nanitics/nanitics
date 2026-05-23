"""Unit tests for ``run_cancellable`` and ``RunCancelled``."""

from __future__ import annotations

import asyncio

import pytest

from nanitics.infrastructure.errors import NaniticsError
from nanitics.safety.cancellable_dispatch import RunCancelled, run_cancellable
from nanitics.safety.cancellation import CancellationToken


class TestRunCancellable:
    async def test_inner_completes_first_returns_value(self) -> None:
        token = CancellationToken()

        async def _work() -> int:
            return 42

        result = await run_cancellable(_work(), token)
        assert result == 42

    async def test_token_fires_during_await_raises_run_cancelled(self) -> None:
        token = CancellationToken()
        inner_was_cancelled = asyncio.Event()

        async def _slow() -> None:
            try:
                await asyncio.Event().wait()  # never resolves
            except asyncio.CancelledError:
                inner_was_cancelled.set()
                raise

        async def _cancel_soon() -> None:
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            token.cancel()

        cancel_task = asyncio.create_task(_cancel_soon())
        with pytest.raises(RunCancelled):
            await run_cancellable(_slow(), token, tool_name="slow", step_number=3)
        await cancel_task
        assert inner_was_cancelled.is_set()

    async def test_token_already_cancelled_raises_run_cancelled(self) -> None:
        token = CancellationToken()
        token.cancel()
        ran = False

        async def _work() -> int:
            nonlocal ran
            ran = True
            return 1

        with pytest.raises(RunCancelled):
            await run_cancellable(_work(), token)
        # The pre-cancelled fast path skips execution entirely.
        assert ran is False

    async def test_none_token_passes_through_without_extra_task(self) -> None:
        # When ``token`` is None there should be no extra task scheduled
        # beyond the current task.
        tasks_before = len(asyncio.all_tasks())

        async def _work() -> str:
            return "hi"

        result = await run_cancellable(_work(), None)
        assert result == "hi"
        tasks_after = len(asyncio.all_tasks())
        # No leak — the current task is the only one accounted for.
        assert tasks_after <= tasks_before

    async def test_inner_raises_non_cancelled_exception_propagates(self) -> None:
        token = CancellationToken()

        async def _broken() -> None:
            raise ValueError("nope")

        with pytest.raises(ValueError, match="nope"):
            await run_cancellable(_broken(), token)

    async def test_run_cancelled_carries_tool_name_and_step(self) -> None:
        token = CancellationToken()
        token.cancel()

        async def _work() -> None:
            await asyncio.Event().wait()

        with pytest.raises(RunCancelled) as excinfo:
            await run_cancellable(_work(), token, tool_name="t", step_number=7)
        assert excinfo.value.tool_name == "t"
        assert excinfo.value.step_number == 7

    async def test_run_cancelled_is_nanitics_error(self) -> None:
        assert issubclass(RunCancelled, NaniticsError)

    async def test_outer_cancellation_propagates(self) -> None:
        # If the caller of ``run_cancellable`` is itself cancelled, the
        # helper must not swallow that — it must re-raise CancelledError
        # after draining its children.
        token = CancellationToken()

        async def _slow() -> None:
            await asyncio.Event().wait()

        outer = asyncio.create_task(run_cancellable(_slow(), token))
        await asyncio.sleep(0)
        outer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await outer
