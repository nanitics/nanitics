from nanitics.infrastructure.errors import AgentIterationLimitError, AgentToolCallLimitError


class IterationLimiter:
    """Tracks and enforces a maximum number of agent steps.

    Each call to ``step()`` increments the counter. When the counter
    exceeds ``max_iterations``, ``AgentIterationLimitError`` is raised.

    Args:
        max_iterations: Maximum allowed steps. Must be at least 1.

    Raises:
        ValueError: If ``max_iterations`` is less than 1.
    """

    def __init__(self, max_iterations: int) -> None:
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be at least 1, got {max_iterations}")
        self._max_iterations = max_iterations
        self._current_iteration = 0

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    @property
    def current_iteration(self) -> int:
        return self._current_iteration

    @property
    def remaining(self) -> int:
        return self._max_iterations - self._current_iteration

    def step(self) -> None:
        self._current_iteration += 1
        if self._current_iteration > self._max_iterations:
            raise AgentIterationLimitError(
                f"Agent exceeded iteration limit of {self._max_iterations}",
                iteration_count=self._current_iteration,
                iteration_limit=self._max_iterations,
            )

    def reset(self) -> None:
        self._current_iteration = 0

    def restore(self, count: int) -> None:
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        if count > self._max_iterations:
            raise ValueError(f"count ({count}) exceeds max_iterations ({self._max_iterations})")
        self._current_iteration = count


class ToolCallLimiter:
    """Tracks and enforces a maximum number of tool calls across all agent steps.

    Unlike ``IterationLimiter`` which increments by 1 per step,
    ``step(count)`` accepts the batch size since one LLM response can
    request multiple tool calls.

    Args:
        max_tool_calls: Maximum allowed tool calls. Must be at least 1.

    Raises:
        ValueError: If ``max_tool_calls`` is less than 1.
    """

    def __init__(self, max_tool_calls: int) -> None:
        if max_tool_calls < 1:
            raise ValueError(f"max_tool_calls must be at least 1, got {max_tool_calls}")
        self._max_tool_calls = max_tool_calls
        self._current_tool_calls = 0

    @property
    def max_tool_calls(self) -> int:
        return self._max_tool_calls

    @property
    def current_tool_calls(self) -> int:
        return self._current_tool_calls

    @property
    def remaining(self) -> int:
        return self._max_tool_calls - self._current_tool_calls

    def step(self, count: int) -> None:
        if count == 0:
            return
        self._current_tool_calls += count
        if self._current_tool_calls > self._max_tool_calls:
            raise AgentToolCallLimitError(
                f"Agent exceeded tool call limit of {self._max_tool_calls}",
                tool_call_count=self._current_tool_calls,
                tool_call_limit=self._max_tool_calls,
            )

    def reset(self) -> None:
        self._current_tool_calls = 0

    def restore(self, count: int) -> None:
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        if count > self._max_tool_calls:
            raise ValueError(f"count ({count}) exceeds max_tool_calls ({self._max_tool_calls})")
        self._current_tool_calls = count
