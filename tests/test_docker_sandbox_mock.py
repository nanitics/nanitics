"""Mock-based unit tests for DockerSandbox edge cases.

These tests exercise error paths and branches in docker.py without
requiring a running Docker daemon.  Internal state is set directly
on the sandbox instance and I/O is mocked.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanitics.safety.sandbox.docker import _CONTAINER_LABEL, DockerSandbox

# ── execute() edge cases ────────────────────────────────────────────


class TestExecuteEdgeCases:
    async def test_execute_not_started(self) -> None:
        sandbox = DockerSandbox()
        with pytest.raises(RuntimeError, match="Sandbox not started"):
            await sandbox.execute("print(1)")

    async def test_execute_unexpected_message_type(self) -> None:
        sandbox = DockerSandbox()
        sandbox._started = True

        with (
            patch.object(sandbox, "_send", new_callable=AsyncMock),
            patch.object(sandbox, "_recv", new_callable=AsyncMock, return_value={"type": "unknown_thing"}),
            pytest.raises(RuntimeError, match="Unexpected message type"),
        ):
            await sandbox.execute("print(1)")


# ── reset() edge cases ──────────────────────────────────────────────


class TestResetEdgeCases:
    async def test_reset_not_started(self) -> None:
        sandbox = DockerSandbox()
        # Should return immediately without error
        await sandbox.reset()

    async def test_reset_sends_and_validates(self) -> None:
        sandbox = DockerSandbox()
        sandbox._started = True

        with (
            patch.object(sandbox, "_send", new_callable=AsyncMock) as mock_send,
            patch.object(sandbox, "_recv", new_callable=AsyncMock, return_value={"type": "ready"}),
        ):
            await sandbox.reset()
            mock_send.assert_called_once_with({"type": "reset"})

    async def test_reset_unexpected_response(self) -> None:
        sandbox = DockerSandbox()
        sandbox._started = True

        with (
            patch.object(sandbox, "_send", new_callable=AsyncMock),
            patch.object(sandbox, "_recv", new_callable=AsyncMock, return_value={"type": "error"}),
            pytest.raises(RuntimeError, match="Expected ready after reset"),
        ):
            await sandbox.reset()


# ── cleanup_orphans() ───────────────────────────────────────────────


class TestCleanupOrphans:
    async def test_cleanup_orphans_removes_containers(self) -> None:
        container1 = MagicMock()
        container2 = MagicMock()
        mock_client = MagicMock()
        mock_client.containers.list.return_value = [container1, container2]

        with patch("nanitics.safety.sandbox.docker.asyncio.to_thread") as mock_to_thread:
            # to_thread is called for: from_env, containers.list, remove×2, close
            mock_to_thread.side_effect = [
                mock_client,  # docker.from_env
                [container1, container2],  # client.containers.list
                None,  # container1.remove
                None,  # container2.remove
                None,  # client.close
            ]
            await DockerSandbox.cleanup_orphans()

        # Verify the calls were made with correct args
        calls = mock_to_thread.call_args_list
        # from_env
        assert calls[0].args[0].__name__ == "from_env"
        # containers.list with correct filters
        assert calls[1].args[0] == mock_client.containers.list
        assert calls[1].kwargs == {"all": True, "filters": {"label": _CONTAINER_LABEL}}
        # container removes
        assert calls[2].args == (container1.remove,)
        assert calls[2].kwargs == {"force": True}
        assert calls[3].args == (container2.remove,)
        # close
        assert calls[4].args == (mock_client.close,)

    async def test_cleanup_orphans_no_containers(self) -> None:
        mock_client = MagicMock()

        with patch("nanitics.safety.sandbox.docker.asyncio.to_thread") as mock_to_thread:
            mock_to_thread.side_effect = [
                mock_client,  # docker.from_env
                [],  # empty list
                None,  # client.close
            ]
            await DockerSandbox.cleanup_orphans()

    async def test_cleanup_orphans_remove_error_suppressed(self) -> None:
        """If one container.remove raises, others are still removed."""
        container1 = MagicMock()
        container2 = MagicMock()
        mock_client = MagicMock()

        with patch("nanitics.safety.sandbox.docker.asyncio.to_thread") as mock_to_thread:
            mock_to_thread.side_effect = [
                mock_client,
                [container1, container2],
                Exception("docker API error"),  # container1 remove fails
                None,  # container2 remove succeeds
                None,  # close
            ]
            # Should not raise
            await DockerSandbox.cleanup_orphans()


# ── _connect() edge cases ───────────────────────────────────────────


class TestConnectEdgeCases:
    async def test_connect_no_port_bindings(self) -> None:
        sandbox = DockerSandbox()
        sandbox._container = MagicMock()
        sandbox._container.ports = {}

        with (
            patch("nanitics.safety.sandbox.docker.asyncio.to_thread"),
            pytest.raises(RuntimeError, match="Runner port not mapped"),
        ):
            await sandbox._connect()

    async def test_connect_connection_refused_then_success(self) -> None:
        sandbox = DockerSandbox()
        sandbox._container = MagicMock()
        sandbox._container.ports = {"9999/tcp": [{"HostPort": "12345"}]}

        mock_reader = AsyncMock()
        ready_line = json.dumps({"type": "ready"}).encode() + b"\n"
        mock_reader.readline.return_value = ready_line

        mock_writer = MagicMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        call_count = 0

        async def open_connection_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionRefusedError("refused")
            return mock_reader, mock_writer

        with (
            patch("nanitics.safety.sandbox.docker.asyncio.to_thread"),
            patch("nanitics.safety.sandbox.docker.asyncio.open_connection", side_effect=open_connection_side_effect),
            patch("nanitics.safety.sandbox.docker.asyncio.sleep", new_callable=AsyncMock),
            patch("nanitics.safety.sandbox.docker.asyncio.wait_for", return_value=ready_line),
        ):
            await sandbox._connect()
            assert sandbox._reader == mock_reader
            assert sandbox._writer == mock_writer

    async def test_connect_handshake_timeout(self) -> None:
        sandbox = DockerSandbox()
        sandbox._container = MagicMock()
        sandbox._container.ports = {"9999/tcp": [{"HostPort": "12345"}]}

        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with (
            patch("nanitics.safety.sandbox.docker.asyncio.to_thread"),
            patch(
                "nanitics.safety.sandbox.docker.asyncio.open_connection",
                new_callable=AsyncMock,
                return_value=(mock_reader, mock_writer),
            ),
            patch("nanitics.safety.sandbox.docker.asyncio.sleep", new_callable=AsyncMock),
            patch("nanitics.safety.sandbox.docker.asyncio.wait_for", side_effect=TimeoutError),
            pytest.raises(ConnectionError, match="Could not connect"),
        ):
            await sandbox._connect()

    async def test_connect_handshake_empty_line(self) -> None:
        """Runner connects but sends empty line (closed before ready)."""
        sandbox = DockerSandbox()
        sandbox._container = MagicMock()
        sandbox._container.ports = {"9999/tcp": [{"HostPort": "12345"}]}

        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with (
            patch("nanitics.safety.sandbox.docker.asyncio.to_thread"),
            patch(
                "nanitics.safety.sandbox.docker.asyncio.open_connection",
                new_callable=AsyncMock,
                return_value=(mock_reader, mock_writer),
            ),
            patch("nanitics.safety.sandbox.docker.asyncio.sleep", new_callable=AsyncMock),
            patch("nanitics.safety.sandbox.docker.asyncio.wait_for", return_value=b""),
            pytest.raises(ConnectionError, match="Could not connect"),
        ):
            await sandbox._connect()

    async def test_connect_non_ready_json_message(self) -> None:
        """Runner sends valid JSON but not a ready message — writer is closed."""
        sandbox = DockerSandbox()
        sandbox._container = MagicMock()
        sandbox._container.ports = {"9999/tcp": [{"HostPort": "12345"}]}

        non_ready = json.dumps({"type": "error", "msg": "init failed"}).encode() + b"\n"

        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with (
            patch("nanitics.safety.sandbox.docker.asyncio.to_thread"),
            patch(
                "nanitics.safety.sandbox.docker.asyncio.open_connection",
                new_callable=AsyncMock,
                return_value=(mock_reader, mock_writer),
            ),
            patch("nanitics.safety.sandbox.docker.asyncio.sleep", new_callable=AsyncMock),
            patch("nanitics.safety.sandbox.docker.asyncio.wait_for", return_value=non_ready),
            pytest.raises(ConnectionError, match="Could not connect"),
        ):
            await sandbox._connect()

    async def test_connect_connection_error_during_handshake(self) -> None:
        """ConnectionError during readline triggers writer cleanup."""
        sandbox = DockerSandbox()
        sandbox._container = MagicMock()
        sandbox._container.ports = {"9999/tcp": [{"HostPort": "12345"}]}

        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        with (
            patch("nanitics.safety.sandbox.docker.asyncio.to_thread"),
            patch(
                "nanitics.safety.sandbox.docker.asyncio.open_connection",
                new_callable=AsyncMock,
                return_value=(mock_reader, mock_writer),
            ),
            patch("nanitics.safety.sandbox.docker.asyncio.sleep", new_callable=AsyncMock),
            patch("nanitics.safety.sandbox.docker.asyncio.wait_for", side_effect=ConnectionError("reset")),
            pytest.raises(ConnectionError, match="Could not connect"),
        ):
            await sandbox._connect()

    async def test_connect_writer_cleanup_oserror_suppressed(self) -> None:
        """OSError during writer cleanup in handshake is suppressed."""
        sandbox = DockerSandbox()
        sandbox._container = MagicMock()
        sandbox._container.ports = {"9999/tcp": [{"HostPort": "12345"}]}

        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock(side_effect=OSError("broken pipe"))

        with (
            patch("nanitics.safety.sandbox.docker.asyncio.to_thread"),
            patch(
                "nanitics.safety.sandbox.docker.asyncio.open_connection",
                new_callable=AsyncMock,
                return_value=(mock_reader, mock_writer),
            ),
            patch("nanitics.safety.sandbox.docker.asyncio.sleep", new_callable=AsyncMock),
            patch("nanitics.safety.sandbox.docker.asyncio.wait_for", side_effect=TimeoutError),
            pytest.raises(ConnectionError, match="Could not connect"),
        ):
            await sandbox._connect()


# ── _send() / _recv() ────────────────────────────────────────────────


class TestSendRecv:
    async def test_send_writes_json_line(self) -> None:
        sandbox = DockerSandbox()
        mock_writer = MagicMock()
        mock_writer.write = MagicMock()
        mock_writer.drain = AsyncMock()
        sandbox._writer = mock_writer

        await sandbox._send({"type": "execute", "code": "1+1"})

        written = mock_writer.write.call_args[0][0]
        parsed = json.loads(written.decode())
        assert parsed == {"type": "execute", "code": "1+1"}
        assert written.endswith(b"\n")
        mock_writer.drain.assert_awaited_once()

    async def test_recv_returns_parsed_json(self) -> None:
        sandbox = DockerSandbox()
        sandbox._reader = AsyncMock()
        msg = {"type": "execution_result", "stdout": "hi", "success": True}
        line = json.dumps(msg).encode() + b"\n"

        with patch("nanitics.safety.sandbox.docker.asyncio.wait_for", return_value=line):
            result = await sandbox._recv()

        assert result == msg


# ── _recv() edge cases ──────────────────────────────────────────────


class TestRecvEdgeCases:
    async def test_recv_timeout(self) -> None:
        sandbox = DockerSandbox()
        sandbox._reader = AsyncMock()

        with (
            patch("nanitics.safety.sandbox.docker.asyncio.wait_for", side_effect=TimeoutError),
            pytest.raises(TimeoutError, match="Timed out waiting"),
        ):
            await sandbox._recv()

    async def test_recv_empty_line(self) -> None:
        sandbox = DockerSandbox()
        sandbox._reader = AsyncMock()

        with (
            patch("nanitics.safety.sandbox.docker.asyncio.wait_for", return_value=b""),
            pytest.raises(ConnectionError, match="Connection to sandbox runner closed"),
        ):
            await sandbox._recv()
