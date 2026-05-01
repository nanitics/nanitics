from collections.abc import Sequence
from typing import Self

from nanitics.safety.sandbox.protocol import ExecutionResult


class MockSandbox:
    """Test sandbox that returns predefined responses in sequence.

    Provide a sequence of ``ExecutionResult`` objects at construction.
    Each ``execute()`` call returns the next response. Raises ``IndexError``
    if ``execute()`` is called more times than there are configured responses.

    Args:
        responses: Sequence of results to return on successive ``execute()`` calls.
    """

    def __init__(self, responses: Sequence[ExecutionResult]) -> None:
        self._responses = list(responses)
        self._index = 0

    async def start(self) -> None:
        pass

    async def execute(self, code: str) -> ExecutionResult:
        if self._index >= len(self._responses):
            raise IndexError(
                f"MockSandbox exhausted: {len(self._responses)} responses configured, "
                f"but execute() called {self._index + 1} times"
            )
        result = self._responses[self._index]
        self._index += 1
        return result

    async def reset(self) -> None:
        pass

    async def cleanup(self) -> None:
        pass

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.cleanup()
