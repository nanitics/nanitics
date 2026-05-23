"""Cooperative cancellation primitive.

``CancellationToken`` is the single primitive callers use to signal an
agent to stop. The token supports two waiters:

* ``is_cancelled`` — synchronous boolean, safe to poll from any thread.
* ``wait_async()`` — coroutine that resolves the instant ``cancel()`` is
  called, even when ``cancel()`` fires from a different thread.

The two surfaces share state: setting the token sets both the
``threading.Event`` (mirrored for synchronous readers and thread-side
``cancel()`` callers) and the lazily-bound ``asyncio.Event`` (created on
the first ``wait_async()`` call so the token can be allocated long before
any event loop exists).

A given token instance is bound to **one** event loop. The binding
happens lazily on the first ``wait_async()`` call. Calling
``wait_async()`` from a different loop raises ``RuntimeError`` — supporting
multi-loop reuse would require ``run_coroutine_threadsafe`` plumbing and
ill-defined semantics for "cancelled in loop A, awaited in loop B". The
single-loop rule matches every production use case (one run = one loop).
"""

import asyncio
import threading


class CancellationToken:
    """Thread-safe cooperative cancellation signal.

    Allows external code (API timeout handlers, UI cancel buttons,
    orchestrators) to signal an agent to stop gracefully. The agent
    checks ``is_cancelled`` between steps and awaits ``wait_async()``
    inside the cancellable-dispatch helper that wraps every tool call,
    so an in-flight tool call is interrupted as soon as ``cancel()``
    fires. Cancellation is irreversible.

    Threading.
        ``cancel()`` is safe from any thread. From the loop's own
        thread it sets the asyncio event directly; from any other
        thread it schedules ``asyncio.Event.set()`` via
        ``loop.call_soon_threadsafe`` against the recorded bound loop.

    Loop binding.
        A single token instance is bound to one event loop on the
        first ``wait_async()`` call. ``wait_async()`` from a different
        loop raises ``RuntimeError``.
    """

    def __init__(self) -> None:
        self._thread_event = threading.Event()
        self._asyncio_event: asyncio.Event | None = None
        self._bound_loop: asyncio.AbstractEventLoop | None = None
        # Guards lazy binding from concurrent ``wait_async`` callers on
        # the same loop. The asyncio side is single-threaded once bound,
        # but pre-binding two awaiters may race the first call.
        self._bind_lock = threading.Lock()

    @property
    def is_cancelled(self) -> bool:
        return self._thread_event.is_set()

    def cancel(self) -> None:
        """Signal cancellation. Idempotent and safe from any thread.

        Sets the synchronous ``threading.Event`` immediately so
        ``is_cancelled`` readers observe the change without waiting on
        the loop. If a loop has been bound (via a prior
        ``wait_async()``), schedules the asyncio event's ``set()`` on
        that loop — using ``call_soon_threadsafe`` so cross-thread
        cancellers behave correctly.
        """
        self._thread_event.set()
        loop = self._bound_loop
        event = self._asyncio_event
        if loop is None or event is None:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            event.set()
        else:
            loop.call_soon_threadsafe(event.set)

    async def wait_async(self) -> None:
        """Resolve when the token is cancelled.

        On first call, binds the token to the running event loop and
        creates the underlying ``asyncio.Event`` in that loop. If the
        token is already cancelled, returns immediately. Calling from
        a different loop than the bound one raises ``RuntimeError``.
        """
        running = asyncio.get_running_loop()
        with self._bind_lock:
            if self._bound_loop is None:
                self._bound_loop = running
                self._asyncio_event = asyncio.Event()
                if self._thread_event.is_set():
                    self._asyncio_event.set()
            elif self._bound_loop is not running:
                raise RuntimeError("CancellationToken is bound to a different event loop")
        event = self._asyncio_event
        assert event is not None  # bound implies event exists
        if self._thread_event.is_set():
            return
        await event.wait()
