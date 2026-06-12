from __future__ import annotations

from typing import Protocol, runtime_checkable

from nanitics.collaboration.protocol import HumanInputRequest, HumanInputResponse
from nanitics.infrastructure.errors import NaniticsError


class DuplicateHitlRequestError(NaniticsError):
    """A HITL request with the given ``request_id`` already exists in the store.

    Raised by :meth:`HitlRequestStore.save_request` when called with a
    ``request_id`` that was already persisted. Surfaces the same failure
    mode uniformly across every backend so misuse (e.g. re-saving on resume)
    cannot hide in the in-memory path while failing under Postgres.

    Attributes:
        request_id: The offending ``request_id`` that already exists.
    """

    request_id: str

    def __init__(
        self,
        request_id: str,
        *,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> None:
        super().__init__(
            f"HITL request with request_id={request_id!r} already exists",
            trace_id=trace_id,
            span_id=span_id,
        )
        self.request_id = request_id


class DuplicateHitlResponseError(NaniticsError):
    """A HITL response for the given ``request_id`` already exists in the store.

    Raised by :meth:`HitlRequestStore.save_response` when called with a
    ``request_id`` that already has a recorded response. Mirrors
    :class:`DuplicateHitlRequestError` on the request side: a re-save during
    resume re-execution (a worker re-driving ``ResumeService.resume`` after a
    crash) is the expected trigger and is swallowed by the resume path. The
    typed error surfaces the same failure mode uniformly across every backend,
    so a re-save cannot silently overwrite in one store while raising a raw
    driver exception in another.

    Attributes:
        request_id: The ``request_id`` whose response already exists.
    """

    request_id: str

    def __init__(
        self,
        request_id: str,
        *,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> None:
        super().__init__(
            f"HITL response for request_id={request_id!r} already exists",
            trace_id=trace_id,
            span_id=span_id,
        )
        self.request_id = request_id


@runtime_checkable
class HitlRequestStore(Protocol):
    """Protocol for persisting HITL requests and responses.

    Implementations store pending requests so an external system (API, UI)
    can display them, and store responses so the durable provider can
    retrieve them on resume.
    """

    async def save_request(self, request: HumanInputRequest) -> None:
        """Persist a new HITL request.

        Raises:
            DuplicateHitlRequestError: If ``request.request_id`` already exists
                in the store. Implementations must surface this uniformly so
                misuse does not hide in one backend while failing in another.
        """
        ...

    async def save_response(self, request_id: str, response: HumanInputResponse) -> None:
        """Store a human's response to a request.

        Raises:
            DuplicateHitlResponseError: If a response for ``request_id`` was
                already persisted. Implementations must surface this uniformly
                (mirroring :meth:`save_request`) so a re-save on resume cannot
                silently overwrite in one backend while failing in another.
        """
        ...

    async def get_response(self, request_id: str) -> HumanInputResponse | None:
        """Retrieve a response by request ID, or None if not yet responded."""
        ...

    async def get_pending_requests(self, run_id: str) -> list[HumanInputRequest]:
        """Return all requests for a run that have no response yet."""
        ...


class InMemoryHitlRequestStore:
    """In-memory implementation of HitlRequestStore for testing.

    Stores requests and responses in dictionaries. Pending requests
    are those without a matching response for the given run.

    Args:
        run_id: Default run ID applied to requests saved without one set,
            so ``get_pending_requests`` can find them.
    """

    def __init__(self, run_id: str | None = None) -> None:
        self._run_id = run_id
        self._requests: dict[str, HumanInputRequest] = {}
        self._responses: dict[str, HumanInputResponse] = {}

    async def save_request(self, request: HumanInputRequest) -> None:
        if request.request_id in self._requests:
            raise DuplicateHitlRequestError(request.request_id)
        if request.run_id is None and self._run_id is not None:
            request = request.model_copy(update={"run_id": self._run_id})
        self._requests[request.request_id] = request

    async def save_response(self, request_id: str, response: HumanInputResponse) -> None:
        if request_id in self._responses:
            raise DuplicateHitlResponseError(request_id)
        self._responses[request_id] = response

    async def get_response(self, request_id: str) -> HumanInputResponse | None:
        return self._responses.get(request_id)

    async def get_pending_requests(self, run_id: str) -> list[HumanInputRequest]:
        responded_ids = set(self._responses.keys())
        return [req for req in self._requests.values() if req.request_id not in responded_ids and req.run_id == run_id]
