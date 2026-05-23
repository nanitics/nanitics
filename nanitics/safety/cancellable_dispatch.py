"""Cancellable-await helper used by every agent's tool-dispatch path.

A single primitive — ``run_cancellable`` — wraps an arbitrary awaitable
and races it against a ``CancellationToken``. If the token wins, the
inner task is cancelled and ``RunCancelled`` is raised. The helper is
opaque: it has no knowledge of ``Tool``, ``ToolCall``, ``MCPTool``, or
any tool-shaped abstraction. Agents call it around their dispatch
awaits so an in-flight call is interrupted the moment the token fires.

``RunCancelled`` is a control-flow signal between this helper and the
caller (an agent loop). Agents catch it, emit a single cancellation
event, and return a normal ``AgentResult`` with
``termination_reason="cancelled"``. ``RunCancelled`` is **not** raised
out of ``Agent.run()`` for tool-cancellation; the structured
``AgentResult`` is the public contract. See the Phase
``design-rationale.md`` §3.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Coroutine
from typing import Any, TypeVar

from nanitics.infrastructure.errors import NaniticsError
from nanitics.safety.cancellation import CancellationToken

_T = TypeVar("_T")


class RunCancelled(NaniticsError):
    """Raised by ``run_cancellable`` when the token wins the race.

    Attributes:
        tool_name: Optional tool name populated by the caller for
            observability — the helper itself does not infer it.
        step_number: Optional agent step number populated by the
            caller, for the safety event the agent emits when it
            catches this exception.
    """

    def __init__(
        self,
        message: str = "Run was cancelled",
        *,
        tool_name: str | None = None,
        step_number: int | None = None,
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.step_number = step_number


async def run_cancellable(
    coro: Coroutine[Any, Any, _T],
    token: CancellationToken | None,
    *,
    tool_name: str | None = None,
    step_number: int | None = None,
) -> _T:
    """Await ``coro`` racing against ``token``.

    Outcomes:

    * ``token is None`` — pure pass-through await; no task is scheduled
      and no extra overhead is incurred.
    * ``token`` already cancelled on entry — the inner task is
      scheduled and immediately cancelled, then ``RunCancelled`` is
      raised. (The inner coroutine may begin executing for a single
      step before cancellation lands; the helper does not guarantee
      zero-execution.)
    * Token fires during the await — the inner task is cancelled, its
      ``CancelledError`` is awaited and suppressed, and
      ``RunCancelled`` is raised carrying ``tool_name`` / ``step_number``.
    * Inner task raises any other exception — propagate unchanged.
    * Inner task completes — return its value.
    """
    if token is None:
        return await coro

    if token.is_cancelled:
        # Pre-cancelled fast path: do not run the coroutine. Close it
        # to release any allocated resources, then raise.
        coro.close()
        raise RunCancelled(tool_name=tool_name, step_number=step_number)

    inner_task: asyncio.Task[_T] = asyncio.ensure_future(coro)
    waiter: asyncio.Task[None] = asyncio.ensure_future(token.wait_async())

    try:
        done, _pending = await asyncio.wait(
            {inner_task, waiter},
            return_when=asyncio.FIRST_COMPLETED,
        )
    except BaseException:
        # The await itself was cancelled (e.g. an outer ``run_cancellable``
        # cancelled us). Cancel both children, drain their exceptions,
        # then re-raise the original.
        inner_task.cancel()
        waiter.cancel()
        await asyncio.gather(inner_task, waiter, return_exceptions=True)
        raise

    if inner_task in done:
        # Inner task finished first — cancel the waiter and surface the
        # inner result/exception.
        waiter.cancel()
        # Await the waiter cancellation so the underlying asyncio.Event
        # waiter is fully unwound before this coroutine returns.
        with contextlib.suppress(BaseException):
            await waiter
        return inner_task.result()

    # Token won the race. Cancel the inner task, drain it, and raise
    # RunCancelled. ``inner_task.result()`` is intentionally NOT called
    # — we suppress its CancelledError as the proximate consequence of
    # the cancellation we requested.
    inner_task.cancel()
    # If the inner coroutine completed concurrently with our cancel and
    # raised a different exception during teardown, we still surface
    # ``RunCancelled`` — the cancellation was the primary event.
    with contextlib.suppress(BaseException):
        await inner_task
    raise RunCancelled(tool_name=tool_name, step_number=step_number)
