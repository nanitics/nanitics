import pytest

from nanitics.infrastructure.errors import AgentIterationLimitError, AgentToolCallLimitError
from nanitics.safety.iteration_limits import IterationLimiter, ToolCallLimiter


class TestIterationLimiterConstruction:
    def test_valid_limit(self):
        limiter = IterationLimiter(max_iterations=10)
        assert limiter.max_iterations == 10
        assert limiter.current_iteration == 0
        assert limiter.remaining == 10

    def test_limit_of_one(self):
        limiter = IterationLimiter(max_iterations=1)
        assert limiter.max_iterations == 1

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            IterationLimiter(max_iterations=0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            IterationLimiter(max_iterations=-5)


class TestIterationLimiterStep:
    def test_allows_exactly_max_iterations(self):
        limiter = IterationLimiter(max_iterations=3)
        limiter.step()  # 1
        limiter.step()  # 2
        limiter.step()  # 3
        assert limiter.current_iteration == 3
        assert limiter.remaining == 0

    def test_raises_on_exceeding_limit(self):
        limiter = IterationLimiter(max_iterations=2)
        limiter.step()
        limiter.step()
        with pytest.raises(AgentIterationLimitError) as exc_info:
            limiter.step()
        assert exc_info.value.iteration_count == 3
        assert exc_info.value.iteration_limit == 2

    def test_single_step_limit(self):
        limiter = IterationLimiter(max_iterations=1)
        limiter.step()
        with pytest.raises(AgentIterationLimitError):
            limiter.step()

    def test_remaining_decrements(self):
        limiter = IterationLimiter(max_iterations=3)
        assert limiter.remaining == 3
        limiter.step()
        assert limiter.remaining == 2
        limiter.step()
        assert limiter.remaining == 1
        limiter.step()
        assert limiter.remaining == 0


class TestIterationLimiterReset:
    def test_reset_restores_initial_state(self):
        limiter = IterationLimiter(max_iterations=3)
        limiter.step()
        limiter.step()
        limiter.reset()
        assert limiter.current_iteration == 0
        assert limiter.remaining == 3

    def test_multiple_cycles_with_reset(self):
        limiter = IterationLimiter(max_iterations=2)
        limiter.step()
        limiter.step()
        with pytest.raises(AgentIterationLimitError):
            limiter.step()

        limiter.reset()
        limiter.step()
        limiter.step()
        with pytest.raises(AgentIterationLimitError):
            limiter.step()


class TestIterationLimiterRestore:
    def test_restore_sets_iteration_count(self):
        limiter = IterationLimiter(max_iterations=5)
        limiter.restore(3)
        assert limiter.current_iteration == 3
        assert limiter.remaining == 2

    def test_step_after_restore_respects_limit(self):
        limiter = IterationLimiter(max_iterations=3)
        limiter.restore(2)
        limiter.step()  # 3 — at limit
        with pytest.raises(AgentIterationLimitError):
            limiter.step()  # 4 — exceeds

    def test_restore_to_max_leaves_zero_remaining(self):
        limiter = IterationLimiter(max_iterations=3)
        limiter.restore(3)
        assert limiter.remaining == 0
        with pytest.raises(AgentIterationLimitError):
            limiter.step()

    def test_restore_negative_raises(self):
        limiter = IterationLimiter(max_iterations=5)
        with pytest.raises(ValueError, match="non-negative"):
            limiter.restore(-1)

    def test_restore_beyond_max_raises(self):
        limiter = IterationLimiter(max_iterations=5)
        with pytest.raises(ValueError, match="exceeds max_iterations"):
            limiter.restore(6)

    def test_restore_zero_equivalent_to_reset(self):
        limiter = IterationLimiter(max_iterations=5)
        limiter.step()
        limiter.step()
        limiter.restore(0)
        assert limiter.current_iteration == 0
        assert limiter.remaining == 5


# ──────────────────────────────────────────────────────────
# ToolCallLimiter Tests
# ──────────────────────────────────────────────────────────


class TestToolCallLimiterConstruction:
    def test_valid_limit(self):
        limiter = ToolCallLimiter(max_tool_calls=6)
        assert limiter.max_tool_calls == 6
        assert limiter.current_tool_calls == 0
        assert limiter.remaining == 6

    def test_limit_of_one(self):
        limiter = ToolCallLimiter(max_tool_calls=1)
        assert limiter.max_tool_calls == 1

    def test_zero_limit_allows_construction(self):
        limiter = ToolCallLimiter(max_tool_calls=0)
        assert limiter.max_tool_calls == 0
        assert limiter.current_tool_calls == 0
        assert limiter.remaining == 0

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            ToolCallLimiter(max_tool_calls=-3)


class TestToolCallLimiterStep:
    def test_single_calls_up_to_limit(self):
        limiter = ToolCallLimiter(max_tool_calls=3)
        limiter.step(1)
        limiter.step(1)
        limiter.step(1)
        assert limiter.current_tool_calls == 3
        assert limiter.remaining == 0

    def test_batch_counting(self):
        limiter = ToolCallLimiter(max_tool_calls=6)
        limiter.step(3)
        assert limiter.current_tool_calls == 3
        assert limiter.remaining == 3
        limiter.step(3)
        assert limiter.current_tool_calls == 6
        assert limiter.remaining == 0

    def test_raises_on_exceeding_limit(self):
        limiter = ToolCallLimiter(max_tool_calls=6)
        limiter.step(3)
        with pytest.raises(AgentToolCallLimitError) as exc_info:
            limiter.step(4)
        assert exc_info.value.tool_call_count == 7
        assert exc_info.value.tool_call_limit == 6

    def test_exactly_at_limit_does_not_raise(self):
        limiter = ToolCallLimiter(max_tool_calls=6)
        limiter.step(6)
        assert limiter.current_tool_calls == 6

    def test_step_zero_is_noop(self):
        limiter = ToolCallLimiter(max_tool_calls=3)
        limiter.step(0)
        assert limiter.current_tool_calls == 0
        assert limiter.remaining == 3

    def test_step_zero_at_limit_does_not_raise(self):
        limiter = ToolCallLimiter(max_tool_calls=1)
        limiter.step(1)
        limiter.step(0)  # Should not raise
        assert limiter.current_tool_calls == 1

    def test_remaining_decrements_by_batch_size(self):
        limiter = ToolCallLimiter(max_tool_calls=10)
        assert limiter.remaining == 10
        limiter.step(3)
        assert limiter.remaining == 7
        limiter.step(5)
        assert limiter.remaining == 2

    def test_single_step_limit(self):
        limiter = ToolCallLimiter(max_tool_calls=1)
        limiter.step(1)
        with pytest.raises(AgentToolCallLimitError):
            limiter.step(1)

    def test_zero_limit_rejects_first_tool_call(self):
        limiter = ToolCallLimiter(max_tool_calls=0)
        with pytest.raises(AgentToolCallLimitError) as exc_info:
            limiter.step(1)
        assert exc_info.value.tool_call_count == 1
        assert exc_info.value.tool_call_limit == 0

    def test_zero_limit_step_zero_is_noop(self):
        limiter = ToolCallLimiter(max_tool_calls=0)
        limiter.step(0)
        assert limiter.current_tool_calls == 0


class TestToolCallLimiterReset:
    def test_reset_restores_initial_state(self):
        limiter = ToolCallLimiter(max_tool_calls=6)
        limiter.step(4)
        limiter.reset()
        assert limiter.current_tool_calls == 0
        assert limiter.remaining == 6

    def test_multiple_cycles_with_reset(self):
        limiter = ToolCallLimiter(max_tool_calls=3)
        limiter.step(3)
        with pytest.raises(AgentToolCallLimitError):
            limiter.step(1)

        limiter.reset()
        limiter.step(2)
        limiter.step(1)
        with pytest.raises(AgentToolCallLimitError):
            limiter.step(1)


class TestToolCallLimiterRestore:
    def test_restore_sets_count(self):
        limiter = ToolCallLimiter(max_tool_calls=6)
        limiter.restore(4)
        assert limiter.current_tool_calls == 4
        assert limiter.remaining == 2

    def test_step_after_restore_respects_limit(self):
        limiter = ToolCallLimiter(max_tool_calls=6)
        limiter.restore(5)
        limiter.step(1)  # 6 — at limit
        with pytest.raises(AgentToolCallLimitError):
            limiter.step(1)  # 7 — exceeds

    def test_restore_to_max_leaves_zero_remaining(self):
        limiter = ToolCallLimiter(max_tool_calls=6)
        limiter.restore(6)
        assert limiter.remaining == 0
        with pytest.raises(AgentToolCallLimitError):
            limiter.step(1)

    def test_restore_negative_raises(self):
        limiter = ToolCallLimiter(max_tool_calls=6)
        with pytest.raises(ValueError, match="non-negative"):
            limiter.restore(-1)

    def test_restore_beyond_max_raises(self):
        limiter = ToolCallLimiter(max_tool_calls=6)
        with pytest.raises(ValueError, match="exceeds max_tool_calls"):
            limiter.restore(7)

    def test_restore_zero_equivalent_to_reset(self):
        limiter = ToolCallLimiter(max_tool_calls=6)
        limiter.step(3)
        limiter.restore(0)
        assert limiter.current_tool_calls == 0
        assert limiter.remaining == 6
