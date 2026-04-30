from __future__ import annotations

import contextlib

from nanitics.collaboration.hitl_store import (
    DuplicateHitlRequestError,
    HitlRequestStore,
)
from nanitics.collaboration.protocol import (
    HumanInputRequest,
    HumanInputResponse,
)
from nanitics.composition.durability.models import SuspensionInfo
from nanitics.composition.durability.suspension import SuspendExecution
from nanitics.infrastructure.errors import (
    ApprovalUnavailableError,
    HumanInputProviderError,
)


class DurableHumanInputProvider:
    """HumanInputProvider that routes HITL requests through a persistent store.

    The provider holds no in-process state beyond the store reference — the
    store is the single source of truth. On every call to ``request_input``:

    - If a response for ``request.request_id`` is already in the store,
      it is returned immediately (the resume path).
    - Otherwise, the request is persisted and ``SuspendExecution`` is raised.
      A :class:`DuplicateHitlRequestError` on re-save is expected during
      resume re-execution (the request was persisted on the first run) and
      is swallowed — the store is the idempotent gate.

    Because the provider is stateless, any two provider instances sharing
    the same store behave identically — the defining durability property.

    Fail-closed guarantee. Any backend exception from the store that is not
    a known control-flow signal (``DuplicateHitlRequestError``,
    ``SuspendExecution``, or an already-typed ``HumanInputProviderError``)
    is re-raised as :class:`ApprovalUnavailableError` with the original
    exception preserved as ``__cause__``. The caller (e.g.
    ``ApprovalWrappedTool``) never reaches the wrapped tool's ``execute``
    on a store failure — the approval simply cannot be obtained, and the
    tool never runs.
    """

    def __init__(self, request_store: HitlRequestStore) -> None:
        self._request_store = request_store

    async def request_input(self, request: HumanInputRequest) -> HumanInputResponse:
        """Return the stored response, or persist the request and suspend.

        Request identity is caller-provided (derived deterministically from
        ``run_id`` and ``tool_call_id``/``step_name``) and stable across
        suspend/resume — so the re-executed tool call asks about the same
        ``request_id``, finds the stored response, and resumes cleanly.

        Raises:
            ApprovalUnavailableError: The store backend raised an unexpected
                exception on ``get_response`` or ``save_request``. The
                original exception is preserved as ``__cause__``.
            SuspendExecution: No response is yet recorded for this request;
                the caller should unwind so the outer durability layer can
                persist a checkpoint and return control to the host.
        """
        try:
            existing = await self._request_store.get_response(request.request_id)
        except (HumanInputProviderError, SuspendExecution):
            raise
        except Exception as exc:
            raise ApprovalUnavailableError(
                f"HITL store failed on get_response for request_id={request.request_id!r}",
            ) from exc
        if existing is not None:
            return existing

        # A duplicate save is expected on re-execution after partial suspend —
        # the request was already persisted on the first run and no response
        # has been recorded yet. Swallow it and fall through to suspend again.
        try:
            with contextlib.suppress(DuplicateHitlRequestError):
                await self._request_store.save_request(request)
        except (HumanInputProviderError, SuspendExecution):
            raise
        except Exception as exc:
            raise ApprovalUnavailableError(
                f"HITL store failed on save_request for request_id={request.request_id!r}",
            ) from exc

        raise SuspendExecution(
            suspension_info=SuspensionInfo(
                suspension_id=request.request_id,
                request_id=request.request_id,
                request_type=request.request_type.value,
                prompt=request.prompt,
                agent_name=request.agent_name,
            ),
        )
