"""Sandbox: SandboxConfig security boundaries, ExecutionResult, and MockSandbox lifecycle.

Demonstrates the sandbox system as a safety primitive — SandboxConfig defaults and
customization, ExecutionResult for success and failure cases, MockSandbox sequential
responses and exhaustion, and the sandbox lifecycle (start, execute, reset, cleanup).

Related guide: docs/guides/safety.md
"""

import asyncio

from pydantic import ValidationError

from nanitics import (
    ExecutionResult,
    MockSandbox,
    SandboxConfig,
)


async def main() -> None:
    # --- Section 1: SandboxConfig — Security Boundaries ---
    print("--- Section 1: SandboxConfig — Security Boundaries ---")

    config = SandboxConfig()

    # Secure defaults
    assert config.image == "python:3.13-slim"
    assert config.timeout == 30.0
    assert config.memory_limit_mb == 256
    assert config.cpu_count == 1.0
    assert config.network_access is False
    assert config.working_directory == "/sandbox"
    assert config.environment == {}

    # Custom config
    custom = SandboxConfig(
        timeout=60.0,
        memory_limit_mb=512,
        network_access=True,
        environment={"API_KEY": "test-key"},
    )
    assert custom.network_access is True
    assert custom.memory_limit_mb == 512
    assert custom.environment == {"API_KEY": "test-key"}

    # Frozen — immutable after creation
    try:
        config.timeout = 999.0  # type: ignore[misc]
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass

    print(
        f"  Default: network_access={config.network_access}, "
        f"timeout={config.timeout}s, memory={config.memory_limit_mb}MB"
    )
    print(
        f"  Custom: network_access={custom.network_access}, "
        f"timeout={custom.timeout}s, memory={custom.memory_limit_mb}MB"
    )
    print("✓ Secure defaults (network disabled, limited resources); frozen after creation")

    # --- Section 2: ExecutionResult — Success and Failure ---
    print("\n--- Section 2: ExecutionResult — Success and Failure ---")

    # Successful execution
    success = ExecutionResult(
        stdout="42\n",
        stderr="",
        return_value="42",
        success=True,
        error=None,
        duration_ms=5.2,
    )
    assert success.success is True
    assert success.stdout == "42\n"
    assert success.return_value == "42"
    assert success.error is None
    assert success.duration_ms == 5.2

    # Failed execution
    failure = ExecutionResult(
        stdout="",
        stderr="Traceback (most recent call last):\n  NameError: name 'x' is not defined",
        return_value=None,
        success=False,
        error="NameError: name 'x' is not defined",
        duration_ms=1.0,
    )
    assert failure.success is False
    assert failure.error == "NameError: name 'x' is not defined"
    assert failure.return_value is None

    # Frozen — immutable after creation
    try:
        success.stdout = "modified"  # type: ignore[misc]
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass

    print(f"  Success: stdout={success.stdout!r}, return_value={success.return_value!r}")
    print(f"  Failure: error={failure.error!r}")
    print("✓ ExecutionResult captures stdout, stderr, return_value, error; frozen after creation")

    # --- Section 3: MockSandbox — Deterministic Testing ---
    print("\n--- Section 3: MockSandbox — Deterministic Testing ---")

    responses = [
        ExecutionResult(stdout="4\n", stderr="", success=True, duration_ms=1.0),
        ExecutionResult(stdout="hello\n", stderr="", success=True, duration_ms=2.0),
    ]

    # Sequential responses
    sandbox = MockSandbox(responses=responses)
    result1 = await sandbox.execute("print(2 + 2)")
    result2 = await sandbox.execute("print('hello')")

    assert result1.stdout == "4\n"
    assert result2.stdout == "hello\n"

    # Exhaustion raises IndexError
    try:
        await sandbox.execute("print('too many')")
        assert False, "Should have raised IndexError"
    except IndexError:
        pass

    # Async context manager
    async with MockSandbox(responses=responses) as ctx_sandbox:
        r = await ctx_sandbox.execute("print(2 + 2)")
        assert r.stdout == "4\n"

    print(f"  Response 1: {result1.stdout!r}")
    print(f"  Response 2: {result2.stdout!r}")
    print("  3rd call: IndexError (exhausted)")
    print("✓ MockSandbox returns responses in order; IndexError on exhaustion; supports async with")

    # --- Section 4: MockSandbox — Lifecycle Methods ---
    print("\n--- Section 4: MockSandbox — Lifecycle Methods ---")

    responses = [
        ExecutionResult(stdout="first\n", stderr="", success=True, duration_ms=1.0),
        ExecutionResult(stdout="second\n", stderr="", success=True, duration_ms=1.0),
    ]

    sandbox = MockSandbox(responses=responses)

    # Full lifecycle: start → execute → reset → execute → cleanup
    await sandbox.start()  # No-op on MockSandbox
    r1 = await sandbox.execute("print('first')")
    assert r1.stdout == "first\n"

    await sandbox.reset()  # No-op — does NOT reset response index
    r2 = await sandbox.execute("print('second')")
    assert r2.stdout == "second\n"

    await sandbox.cleanup()  # No-op on MockSandbox

    print(f"  start() → execute() → {r1.stdout!r}")
    print(f"  reset() → execute() → {r2.stdout!r}")
    print("  cleanup() — all lifecycle methods are no-ops on MockSandbox")
    print("✓ Lifecycle methods (start, reset, cleanup) are safe no-ops; reset does not rewind responses")


if __name__ == "__main__":
    asyncio.run(main())
