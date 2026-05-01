import threading


class CancellationToken:
    """Thread-safe cooperative cancellation signal.

    Allows external code (API timeout handlers, UI cancel buttons) to
    signal an agent to stop gracefully. The agent checks ``is_cancelled``
    between steps and exits when set. Cancellation is irreversible.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()
