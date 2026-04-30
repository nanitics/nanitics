from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from nanitics.collaboration.protocol import HumanInputRequest, HumanInputResponse
from nanitics.infrastructure.errors import ApprovalTimeoutError

if TYPE_CHECKING:
    from nanitics.collaboration.hitl_store import HitlRequestStore


class AsyncHumanInputProvider:
    """Suspends on an asyncio.Future until resolved via an external call.

    Designed for HTTP API integration: ``request_input`` blocks while
    a FastAPI endpoint (or similar) calls ``resolve`` with the human's
    response.

    When an optional ``store`` is provided, requests and responses are
    persisted so they survive application restarts.

    Fail-closed timeout. When a constructor-level ``default_timeout`` or
    per-call ``timeout`` is set, ``request_input`` raises
    :class:`ApprovalTimeoutError` on expiry and removes the pending entry
    so a subsequent ``resolve`` for the same ``request_id`` returns
    ``False``. The wrapped tool never executes on timeout — the library
    encodes no "timeout-as-reject" policy; adopters who want that catch
    the error and decide.

    Args:
        store: Optional backing store. Requests are persisted on
            ``request_input`` and responses on ``resolve``.
        default_timeout: Optional seconds to wait for each pending
            request before raising :class:`ApprovalTimeoutError`. When
            ``None`` (default), ``request_input`` waits indefinitely
            unless the per-call ``timeout`` is set.
    """

    def __init__(
        self,
        *,
        store: HitlRequestStore | None = None,
        default_timeout: float | None = None,
    ) -> None:
        if default_timeout is not None and default_timeout <= 0:
            raise ValueError(
                f"default_timeout must be positive, got {default_timeout}",
            )
        self._pending: dict[str, tuple[HumanInputRequest, asyncio.Future[HumanInputResponse]]] = {}
        self._store = store
        self._default_timeout = default_timeout

    async def request_input(
        self,
        request: HumanInputRequest,
        *,
        timeout: float | None = None,
    ) -> HumanInputResponse:
        """Register the request and suspend until ``resolve`` is called.

        Args:
            request: The human-input request to register.
            timeout: Per-call timeout in seconds. Overrides the
                constructor-level ``default_timeout`` when provided. When
                both are ``None``, waits indefinitely.

        Raises:
            ValueError: If a pending request with the same ``request_id``
                is already registered, or if ``timeout`` is non-positive.
            ApprovalTimeoutError: If an effective timeout expires before
                ``resolve`` is called for this ``request_id``.
        """
        if timeout is not None and timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")
        if request.request_id in self._pending:
            raise ValueError(
                f"AsyncHumanInputProvider already has a pending request for request_id={request.request_id!r}"
            )
        if self._store:
            await self._store.save_request(request)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[HumanInputResponse] = loop.create_future()
        self._pending[request.request_id] = (request, future)

        effective_timeout = timeout if timeout is not None else self._default_timeout
        if effective_timeout is None:
            return await future
        try:
            return await asyncio.wait_for(future, timeout=effective_timeout)
        except TimeoutError as exc:
            # ``asyncio.wait_for`` cancels the inner future before re-raising,
            # so it is already done here; we only need to drop the pending
            # entry so ``resolve`` for the same ``request_id`` returns False.
            self._pending.pop(request.request_id, None)
            raise ApprovalTimeoutError(
                f"HITL request {request.request_id!r} not answered within {effective_timeout}s",
            ) from exc

    async def resolve(self, request_id: str, response: HumanInputResponse) -> bool:
        """Deliver a response to the waiting ``request_input`` call.

        Returns:
            ``True`` if the response was delivered, ``False`` if the
            request is unknown or already resolved.
        """
        entry = self._pending.pop(request_id, None)
        if entry is None or entry[1].done():
            return False
        if self._store:
            await self._store.save_response(request_id, response)
        entry[1].set_result(response)
        return True

    def get_pending(self, run_id: str | None = None) -> list[HumanInputRequest]:
        """Return requests awaiting a response, optionally filtered by ``run_id``."""
        return [
            req for req, fut in self._pending.values() if not fut.done() and (run_id is None or req.run_id == run_id)
        ]
