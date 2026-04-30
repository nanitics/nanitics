import pytest

from nanitics.safety.sandbox.mock import MockSandbox
from nanitics.safety.sandbox.protocol import ExecutionResult, Sandbox


def _ok(stdout: str = "", return_value: str | None = None) -> ExecutionResult:
    return ExecutionResult(
        stdout=stdout,
        stderr="",
        return_value=return_value,
        success=True,
        duration_ms=1.0,
    )


def _err(error: str) -> ExecutionResult:
    return ExecutionResult(
        stdout="",
        stderr="",
        success=False,
        error=error,
        duration_ms=1.0,
    )


class TestMockSandbox:
    async def test_returns_responses_in_order(self) -> None:
        responses = [_ok("first"), _ok("second"), _ok("third")]
        sandbox = MockSandbox(responses)
        await sandbox.start()

        r1 = await sandbox.execute("code1")
        assert r1.stdout == "first"

        r2 = await sandbox.execute("code2")
        assert r2.stdout == "second"

        r3 = await sandbox.execute("code3")
        assert r3.stdout == "third"

    async def test_exhausted_raises_index_error(self) -> None:
        sandbox = MockSandbox([_ok("only")])
        r = await sandbox.execute("first")
        assert r.stdout == "only"

        with pytest.raises(IndexError, match="MockSandbox exhausted"):
            await sandbox.execute("second")

    async def test_lifecycle_methods_are_noop(self) -> None:
        sandbox = MockSandbox([])
        await sandbox.start()
        await sandbox.reset()
        await sandbox.cleanup()

    async def test_async_context_manager(self) -> None:
        async with MockSandbox([_ok("hi")]) as sandbox:
            result = await sandbox.execute("code")
            assert result.stdout == "hi"

    async def test_error_responses(self) -> None:
        sandbox = MockSandbox([_err("NameError")])
        result = await sandbox.execute("bad code")
        assert result.success is False
        assert result.error == "NameError"

    def test_satisfies_sandbox_protocol(self) -> None:
        sandbox = MockSandbox([])
        assert isinstance(sandbox, Sandbox)
