"""Integration tests for DockerSandbox — requires Docker daemon."""

import pytest

import docker as docker_lib
from nanitics.safety.sandbox.docker import DockerSandbox
from nanitics.safety.sandbox.protocol import Sandbox, SandboxConfig


def _docker_available() -> bool:
    try:
        client = docker_lib.from_env()  # type: ignore[attr-defined]
        client.ping()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(not _docker_available(), reason="Docker daemon not available"),
]


class TestDockerSandboxLifecycle:
    async def test_start_and_cleanup(self) -> None:
        sandbox = DockerSandbox()
        await sandbox.start()
        await sandbox.cleanup()

    async def test_start_is_idempotent(self) -> None:
        async with DockerSandbox() as sandbox:
            await sandbox.start()  # Second start is a no-op

    async def test_async_context_manager(self) -> None:
        async with DockerSandbox() as sandbox:
            result = await sandbox.execute("print('hello')")
            assert result.success is True
            assert result.stdout.strip() == "hello"

    def test_satisfies_sandbox_protocol(self) -> None:
        sandbox = DockerSandbox()
        assert isinstance(sandbox, Sandbox)


class TestDockerSandboxExecution:
    async def test_simple_print(self) -> None:
        async with DockerSandbox() as sandbox:
            result = await sandbox.execute("print('hello world')")
            assert result.success is True
            assert result.stdout.strip() == "hello world"
            assert result.stderr == ""
            assert result.error is None

    async def test_return_value_capture(self) -> None:
        async with DockerSandbox() as sandbox:
            result = await sandbox.execute("2 + 3")
            assert result.success is True
            assert result.return_value == "5"

    async def test_multiline_code(self) -> None:
        async with DockerSandbox() as sandbox:
            code = """
def greet(name):
    return f"Hello, {name}!"

result = greet("World")
print(result)
"""
            result = await sandbox.execute(code)
            assert result.success is True
            assert "Hello, World!" in result.stdout

    async def test_stateful_session(self) -> None:
        """Variables persist across execute() calls."""
        async with DockerSandbox() as sandbox:
            await sandbox.execute("x = 42")
            result = await sandbox.execute("x * 2")
            assert result.return_value == "84"

    async def test_import_stdlib(self) -> None:
        async with DockerSandbox() as sandbox:
            result = await sandbox.execute("import json; json.dumps({'a': 1})")
            assert result.success is True
            assert result.return_value == "'{\"a\": 1}'"

    async def test_syntax_error(self) -> None:
        async with DockerSandbox() as sandbox:
            result = await sandbox.execute("def f(:")
            assert result.success is False
            assert result.error is not None
            assert "SyntaxError" in result.error

    async def test_runtime_error(self) -> None:
        async with DockerSandbox() as sandbox:
            result = await sandbox.execute("1 / 0")
            assert result.success is False
            assert result.error is not None
            assert "ZeroDivisionError" in result.error

    async def test_name_error(self) -> None:
        async with DockerSandbox() as sandbox:
            result = await sandbox.execute("print(undefined_var)")
            assert result.success is False
            assert "NameError" in (result.error or "")

    async def test_stderr_capture(self) -> None:
        async with DockerSandbox() as sandbox:
            result = await sandbox.execute("import sys; print('err', file=sys.stderr)")
            assert result.success is True
            assert "err" in result.stderr

    async def test_duration_is_positive(self) -> None:
        async with DockerSandbox() as sandbox:
            result = await sandbox.execute("1 + 1")
            assert result.duration_ms > 0


class TestDockerSandboxReset:
    async def test_reset_clears_namespace(self) -> None:
        async with DockerSandbox() as sandbox:
            await sandbox.execute("x = 42")
            await sandbox.reset()
            result = await sandbox.execute("x")
            assert result.success is False
            assert "NameError" in (result.error or "")

    async def test_reset_preserves_sandbox(self) -> None:
        """After reset, new code can still execute."""
        async with DockerSandbox() as sandbox:
            await sandbox.reset()
            result = await sandbox.execute("1 + 1")
            assert result.success is True
            assert result.return_value == "2"


class TestDockerSandboxResourceLimits:
    async def test_custom_config(self) -> None:
        config = SandboxConfig(memory_limit_mb=128, cpu_count=0.5)
        async with DockerSandbox(config) as sandbox:
            result = await sandbox.execute("print('constrained')")
            assert result.success is True
            assert result.stdout.strip() == "constrained"

    async def test_timeout_enforcement(self) -> None:
        config = SandboxConfig(timeout=3.0)
        async with DockerSandbox(config) as sandbox:
            result = await sandbox.execute("import time; time.sleep(10)")
            assert result.success is False
            assert result.error is not None
            assert "timed out" in result.error.lower()


class TestDockerSandboxToolBridge:
    async def test_tool_call_roundtrip(self) -> None:
        async def dispatcher(name: str, args: dict) -> str:
            if name == "add":
                return str(args["a"] + args["b"])
            raise ValueError(f"Unknown tool: {name}")

        async with DockerSandbox(tool_dispatcher=dispatcher) as sandbox:
            code = """result = __call_tool__("add", {"a": 3, "b": 4})
print(result)"""
            result = await sandbox.execute(code)
            assert result.success is True
            assert result.stdout.strip() == "7"

    async def test_tool_call_error_propagates(self) -> None:
        async def dispatcher(name: str, args: dict) -> str:
            raise RuntimeError("Tool failed")

        async with DockerSandbox(tool_dispatcher=dispatcher) as sandbox:
            code = """result = __call_tool__("broken", {})"""
            result = await sandbox.execute(code)
            assert result.success is False
            assert "Tool failed" in (result.error or "")

    async def test_no_dispatcher_returns_error(self) -> None:
        async with DockerSandbox() as sandbox:
            code = """result = __call_tool__("anything", {})"""
            result = await sandbox.execute(code)
            assert result.success is False
            assert "No tool dispatcher" in (result.error or "")

    async def test_multiple_tool_calls_in_one_execution(self) -> None:
        call_count = 0

        async def dispatcher(name: str, args: dict) -> str:
            nonlocal call_count
            call_count += 1
            return f"result_{call_count}"

        async with DockerSandbox(tool_dispatcher=dispatcher) as sandbox:
            code = """
a = __call_tool__("first", {})
b = __call_tool__("second", {})
print(f"{a},{b}")
"""
            result = await sandbox.execute(code)
            assert result.success is True
            assert result.stdout.strip() == "result_1,result_2"
            assert call_count == 2


class TestDockerSandboxSecurity:
    async def test_read_only_filesystem(self) -> None:
        """Container filesystem is read-only outside tmpfs mounts."""
        async with DockerSandbox() as sandbox:
            code = """
try:
    with open("/etc/test_file", "w") as f:
        f.write("test")
    print("WRITABLE")
except OSError:
    print("READ_ONLY")
"""
            result = await sandbox.execute(code)
            assert result.success is True
            assert "READ_ONLY" in result.stdout

    async def test_working_directory_is_writable(self) -> None:
        """Working directory (tmpfs) is writable."""
        async with DockerSandbox() as sandbox:
            code = """
with open("test_file.txt", "w") as f:
    f.write("hello")
with open("test_file.txt") as f:
    print(f.read())
"""
            result = await sandbox.execute(code)
            assert result.success is True
            assert "hello" in result.stdout

    async def test_network_access_false_blocks_dns(self) -> None:
        """When network_access=False, DNS resolution fails."""
        config = SandboxConfig(network_access=False)
        async with DockerSandbox(config) as sandbox:
            code = """
import socket
try:
    socket.getaddrinfo("example.com", 80)
    print("RESOLVED")
except socket.gaierror:
    print("DNS_BLOCKED")
"""
            result = await sandbox.execute(code)
            assert result.success is True
            assert "DNS_BLOCKED" in result.stdout

    async def test_network_access_true_allows_dns(self) -> None:
        """When network_access=True, DNS resolution works."""
        config = SandboxConfig(network_access=True)
        async with DockerSandbox(config) as sandbox:
            code = """
import socket
try:
    socket.getaddrinfo("example.com", 80)
    print("RESOLVED")
except socket.gaierror:
    print("DNS_BLOCKED")
"""
            result = await sandbox.execute(code)
            assert result.success is True
            assert "RESOLVED" in result.stdout

    async def test_tool_bridge_works_without_network_access(self) -> None:
        """Tool bridge still works when network_access=False."""

        async def dispatcher(name: str, args: dict) -> str:
            return str(args.get("x", 0) * 2)

        config = SandboxConfig(network_access=False)
        async with DockerSandbox(config, tool_dispatcher=dispatcher) as sandbox:
            code = 'result = __call_tool__("double", {"x": 21})\nprint(result)'
            result = await sandbox.execute(code)
            assert result.success is True
            assert result.stdout.strip() == "42"
