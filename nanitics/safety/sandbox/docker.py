"""Docker-based sandbox for isolated code execution.

Requires the ``docker`` package (``pip install nanitics[code_execution]``).
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.resources
import json
import os
import tempfile
from collections.abc import Awaitable, Callable
from typing import Any, Self

from nanitics.safety.sandbox.protocol import ExecutionResult, SandboxConfig

_RUNNER_PORT = 9999
_PID_LIMIT = 64
_CONTAINER_LABEL = "nanitics-sandbox"


class DockerSandbox:
    """Sandbox implementation backed by a Docker container.

    The container runs a persistent Python process (the runner script)
    that executes code sent over TCP.  State persists across ``execute()``
    calls until ``reset()`` is called.

    Security relies on container-level isolation: read-only root filesystem,
    ``no-new-privileges``, PID limits, memory/CPU limits, and a minimal
    base image.  The container uses Docker's default bridge network for
    host ↔ container TCP communication.

    Honest limits.
        Blocks: host filesystem access outside the bind-mounted runner
        script; outbound network by default (no allow-list, DNS stubbed
        to loopback, ``NET_RAW`` dropped); privilege escalation
        (``no-new-privileges``); resource exhaustion (PID limit, memory
        and CPU caps enforced by ``mem_limit`` and ``nano_cpus``).
        Does not block: a determined escape exploiting a Docker daemon
        CVE; side-channel or timing attacks against the host kernel; data
        exfiltration through any network destination an adopter chooses
        to allow-list. ``DockerSandbox`` is the right tool for untrusted
        LLM-generated code in a development or low-consequence context;
        running untrusted code against high-value production state
        requires stronger isolation the SDK does not ship. See
        ``docs/guides/security.md`` for the posture this pairs with.
    """

    def __init__(
        self,
        config: SandboxConfig | None = None,
        *,
        tool_dispatcher: Callable[[str, dict[str, Any]], Awaitable[str]] | None = None,
    ) -> None:
        self._config = config or SandboxConfig()
        self._tool_dispatcher = tool_dispatcher
        self._container: Any = None
        self._client: Any = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._runner_tmp: str | None = None
        self._started = False

    async def start(self) -> None:
        if self._started:
            return

        import docker

        self._client = docker.from_env()  # type: ignore[attr-defined]

        # Write runner script to a host temp file for bind-mounting.
        # This avoids put_archive issues with read-only root filesystems.
        runner_source = _get_runner_source()
        fd, self._runner_tmp = tempfile.mkstemp(suffix=".py")
        os.write(fd, runner_source.encode("utf-8"))
        os.close(fd)

        host_config = self._build_host_config()

        timeout_arg = str(int(self._config.timeout))
        self._container = await asyncio.to_thread(
            self._client.containers.create,
            image=self._config.image,
            command=["python3", "/opt/runner/_runner.py", timeout_arg],
            detach=True,
            auto_remove=True,
            labels={_CONTAINER_LABEL: "true"},
            ports={f"{_RUNNER_PORT}/tcp": 0},
            volumes={
                self._runner_tmp: {"bind": "/opt/runner/_runner.py", "mode": "ro"},
            },
            **host_config,
        )

        await asyncio.to_thread(self._container.start)

        # Establish TCP connection to the runner
        await self._connect()
        self._started = True

    async def execute(self, code: str) -> ExecutionResult:
        if not self._started:
            raise RuntimeError("Sandbox not started — call start() first")

        await self._send({"type": "execute", "code": code})

        # Read messages until we get execution_result, handling tool_calls
        while True:
            msg = await self._recv()
            msg_type = msg.get("type")

            if msg_type == "execution_result":
                return ExecutionResult(
                    stdout=msg["stdout"],
                    stderr=msg["stderr"],
                    return_value=msg.get("return_value"),
                    success=msg["success"],
                    error=msg.get("error"),
                    duration_ms=msg["duration_ms"],
                )
            if msg_type == "tool_call":
                await self._handle_tool_call(msg)
            else:
                raise RuntimeError(f"Unexpected message type: {msg_type}")

    async def reset(self) -> None:
        if not self._started:
            return
        await self._send({"type": "reset"})
        msg = await self._recv()
        if msg.get("type") != "ready":
            raise RuntimeError(f"Expected ready after reset, got {msg.get('type')}")

    async def cleanup(self) -> None:
        if self._writer:
            self._writer.close()
            with contextlib.suppress(OSError, ConnectionError):
                await self._writer.wait_closed()
            self._writer = None
            self._reader = None

        if self._container:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self._container.stop, timeout=5)
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self._container.remove, force=True)
            self._container = None

        if self._client:
            await asyncio.to_thread(self._client.close)
            self._client = None

        if self._runner_tmp:
            with contextlib.suppress(OSError):
                os.unlink(self._runner_tmp)
            self._runner_tmp = None

        self._started = False

    @staticmethod
    async def cleanup_orphans() -> None:
        """Remove all lingering sandbox containers from prior crashed sessions.

        Finds all containers with the ``nanitics-sandbox`` label regardless
        of state and force-removes them.  Safe to call at application startup.
        """
        import docker

        client = await asyncio.to_thread(docker.from_env)  # type: ignore[attr-defined]
        try:
            containers = await asyncio.to_thread(
                client.containers.list,
                all=True,
                filters={"label": _CONTAINER_LABEL},
            )
            for container in containers:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(container.remove, force=True)
        finally:
            await asyncio.to_thread(client.close)

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.cleanup()

    # -- Private helpers --

    def _build_host_config(self) -> dict[str, Any]:
        """Build container creation kwargs for resource limits and security."""
        cfg = self._config
        tmpfs: dict[str, str] = {
            cfg.working_directory: f"size={cfg.memory_limit_mb}m,mode=1777",
            "/tmp": "size=64m,mode=1777",
        }
        host_config: dict[str, Any] = {
            "mem_limit": f"{cfg.memory_limit_mb}m",
            "nano_cpus": int(cfg.cpu_count * 1e9),
            "pids_limit": _PID_LIMIT,
            "security_opt": ["no-new-privileges"],
            "read_only": True,
            "tmpfs": tmpfs,
            "working_dir": cfg.working_directory,
            "environment": cfg.environment,
        }

        if not cfg.network_access:
            # Block outbound internet while keeping the default bridge for
            # host ↔ container TCP communication (tool bridge).
            # Docker internal networks don't support port publishing, so we
            # stay on the default bridge but restrict DNS and raw sockets.
            host_config["dns"] = ["127.0.0.1"]
            host_config["dns_search"] = [""]
            host_config["cap_drop"] = ["NET_RAW"]

        return host_config

    async def _connect(self) -> None:
        """Establish TCP connection to the runner via mapped port."""
        await asyncio.to_thread(self._container.reload)
        port_bindings = self._container.ports.get(f"{_RUNNER_PORT}/tcp")
        if not port_bindings:
            raise RuntimeError("Runner port not mapped")
        host_port = int(port_bindings[0]["HostPort"])

        # Retry connection — Docker's port proxy may accept connections
        # before the runner is listening, immediately closing them.
        # We retry the full connect + ready handshake.
        last_error: Exception | None = None
        for _ in range(30):
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", host_port)
            except (ConnectionRefusedError, OSError) as exc:
                last_error = exc
                await asyncio.sleep(0.2)
                continue

            try:
                line = await asyncio.wait_for(reader.readline(), timeout=2.0)
                if not line:
                    writer.close()
                    await writer.wait_closed()
                    await asyncio.sleep(0.2)
                    continue
                msg = json.loads(line.decode())
                if msg.get("type") == "ready":
                    self._reader = reader
                    self._writer = writer
                    return
                writer.close()
                await writer.wait_closed()
            except (TimeoutError, ConnectionError, OSError):
                try:
                    writer.close()
                    await writer.wait_closed()
                except OSError:
                    pass
                await asyncio.sleep(0.2)
                continue

        raise ConnectionError(f"Could not connect to runner at 127.0.0.1:{host_port}") from last_error

    async def _send(self, message: dict[str, Any]) -> None:
        """Send a newline-delimited JSON message to the runner."""
        assert self._writer is not None
        data = json.dumps(message) + "\n"
        self._writer.write(data.encode())
        await self._writer.drain()

    async def _recv(self) -> dict[str, Any]:
        """Read a newline-delimited JSON message from the runner."""
        assert self._reader is not None
        try:
            line = await asyncio.wait_for(
                self._reader.readline(),
                timeout=self._config.timeout + 5,
            )
        except TimeoutError:
            raise TimeoutError("Timed out waiting for response from sandbox") from None
        if not line:
            raise ConnectionError("Connection to sandbox runner closed")
        return dict(json.loads(line.decode()))

    async def _handle_tool_call(self, msg: dict[str, Any]) -> None:
        """Dispatch a tool call from the sandbox to the host-side dispatcher."""
        name = msg["name"]
        args = msg.get("args", {})

        if self._tool_dispatcher is None:
            await self._send(
                {
                    "type": "tool_result",
                    "result": "",
                    "error": f"No tool dispatcher configured — cannot call '{name}'",
                }
            )
            return

        try:
            result = await self._tool_dispatcher(name, args)
            await self._send(
                {
                    "type": "tool_result",
                    "result": result,
                    "error": None,
                }
            )
        except Exception as exc:
            await self._send(
                {
                    "type": "tool_result",
                    "result": "",
                    "error": str(exc),
                }
            )


def _get_runner_source() -> str:
    """Read the runner script source from the package."""
    return importlib.resources.files("nanitics.safety.sandbox").joinpath("_runner.py").read_text(encoding="utf-8")
