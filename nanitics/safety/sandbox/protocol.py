from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


class SandboxConfig(BaseModel):
    """Configuration for a sandboxed code execution environment.

    Attributes:
        image: Docker image to use for the container.
        timeout: Maximum execution time per ``execute()`` call in seconds.
        memory_limit_mb: Container memory limit in megabytes.
        cpu_count: CPU allocation for the container.
        network_access: Whether the container can access the network.
        working_directory: Working directory inside the container.
        environment: Environment variables for the container.
    """

    model_config = ConfigDict(frozen=True)

    image: str = "python:3.13-slim"
    timeout: float = 30.0
    memory_limit_mb: int = 256
    cpu_count: float = 1.0
    network_access: bool = False
    working_directory: str = "/sandbox"
    environment: dict[str, str] = {}


class ExecutionResult(BaseModel):
    """Result of executing code in a sandbox.

    Attributes:
        stdout: Standard output from the executed code.
        stderr: Standard error output.
        return_value: The value of the last expression, if any.
        success: Whether execution completed without error.
        error: Error message if execution failed.
        duration_ms: Execution time in milliseconds.
    """

    model_config = ConfigDict(frozen=True)

    stdout: str
    stderr: str
    return_value: str | None = None
    success: bool
    error: str | None = None
    duration_ms: float


@runtime_checkable
class Sandbox(Protocol):
    """Protocol for isolated code execution environments.

    Implementations manage a sandboxed environment where LLM-generated
    code can run safely. State persists across ``execute()`` calls until
    ``reset()`` is called. Use as an async context manager for automatic cleanup.
    """

    async def start(self) -> None:
        """Initialize the sandbox environment."""
        ...

    async def execute(self, code: str) -> ExecutionResult:
        """Execute code and return the result."""
        ...

    async def reset(self) -> None:
        """Clear execution state while keeping the environment alive."""
        ...

    async def cleanup(self) -> None:
        """Tear down the environment and release all resources."""
        ...

    async def __aenter__(self) -> Sandbox:
        """Start the sandbox and return it for use as an async context manager."""
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Clean up the sandbox on context manager exit."""
        ...
