#!/usr/bin/env python3
"""Container-side runner for sandboxed code execution.

Self-contained — uses only Python stdlib, no nanitics imports.
Runs inside Docker containers and communicates with the host via
newline-delimited JSON messages over TCP.

Message protocol (all are newline-delimited JSON):

    Host → Container:
        {"type": "execute", "code": "..."}
        {"type": "tool_result", "result": "...", "error": null}
        {"type": "reset"}

    Container → Host:
        {"type": "ready"}
        {"type": "tool_call", "name": "...", "args": {...}}
        {"type": "execution_result", "stdout": "...", ...}
"""

import ast
import contextlib
import io
import json
import signal
import socket
import sys
import threading
import time
import traceback
from collections.abc import Callable
from typing import Any

RUNNER_PORT = 9999
_BUFFER_SIZE = 65536


def send_message(conn: socket.socket, message: dict[str, Any]) -> None:
    """Send a newline-delimited JSON message."""
    conn.sendall((json.dumps(message) + "\n").encode())


class MessageReader:
    """Reads newline-delimited JSON messages from a socket with buffering."""

    def __init__(self) -> None:
        self._buffer = ""

    def recv(self, conn: socket.socket) -> dict[str, Any]:
        """Read the next complete JSON message, blocking until available."""
        while "\n" not in self._buffer:
            chunk = conn.recv(_BUFFER_SIZE)
            if not chunk:
                raise ConnectionError("Connection closed")
            self._buffer += chunk.decode()
        line, self._buffer = self._buffer.split("\n", 1)
        return dict(json.loads(line))


def prepare_code(source: str) -> tuple[Any, Any | None]:
    """Compile source, splitting off the last expression for value capture.

    If the last statement is an expression (``ast.Expr``), it is compiled
    separately so its value can be captured via ``eval()``.

    Returns ``(exec_code, eval_code)`` where *eval_code* is ``None`` when
    the last statement is not an expression.  Raises ``SyntaxError`` for
    unparseable source.
    """
    tree = ast.parse(source)

    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return compile(tree, "<code>", "exec"), None

    last_expr = tree.body.pop()
    exec_code = compile(
        ast.Module(body=tree.body, type_ignores=tree.type_ignores),
        "<code>",
        "exec",
    )
    eval_code = compile(
        ast.Expression(body=last_expr.value),  # type: ignore[attr-defined]
        "<code>",
        "eval",
    )
    return exec_code, eval_code


def execute_code(
    source: str,
    namespace: dict[str, Any],
    timeout: int = 0,
) -> dict[str, Any]:
    """Execute Python code with stdout/stderr capture and return value extraction.

    Args:
        source: Python source code to execute.
        namespace: Persistent execution namespace (shared across calls).
        timeout: Seconds before ``SIGALRM`` aborts execution.  0 disables.

    Returns:
        Dict with ``execution_result`` message fields.
    """
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    return_value: str | None = None
    success = True
    error: str | None = None
    start = time.monotonic()

    def _alarm_handler(signum: int, frame: Any) -> None:
        raise TimeoutError("Execution timed out")

    prev_handler = None
    can_use_alarm = timeout > 0 and threading.current_thread() is threading.main_thread()
    if can_use_alarm:
        prev_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(timeout)

    try:
        exec_code, eval_code = prepare_code(source)
        with (
            contextlib.redirect_stdout(stdout_buf),
            contextlib.redirect_stderr(stderr_buf),
        ):
            exec(exec_code, namespace)
            if eval_code is not None:
                result = eval(eval_code, namespace)
                if result is not None:
                    return_value = repr(result)
    except TimeoutError:
        success = False
        error = "Execution timed out"
    except Exception:
        success = False
        error = traceback.format_exc()
    finally:
        if can_use_alarm:
            signal.alarm(0)
            if prev_handler is not None:
                signal.signal(signal.SIGALRM, prev_handler)

    return {
        "type": "execution_result",
        "stdout": stdout_buf.getvalue(),
        "stderr": stderr_buf.getvalue(),
        "return_value": return_value,
        "success": success,
        "error": error,
        "duration_ms": (time.monotonic() - start) * 1000,
    }


def make_tool_caller(
    conn: socket.socket,
    reader: MessageReader,
) -> Callable[[str, dict[str, Any] | None], str]:
    """Create a ``__call_tool__`` function that bridges to host-side tools."""

    def __call_tool__(name: str, args: dict[str, Any] | None = None) -> str:
        """Call a host-side tool by name and return its string result."""
        send_message(conn, {"type": "tool_call", "name": name, "args": args or {}})
        response = reader.recv(conn)
        if response.get("type") != "tool_result":
            raise RuntimeError(f"Expected tool_result, got {response.get('type')}")
        if response.get("error"):
            raise RuntimeError(f"Tool '{name}' failed: {response['error']}")
        return str(response.get("result", ""))

    return __call_tool__


def serve(port: int = RUNNER_PORT, timeout: int = 30) -> None:
    """Run the TCP command server until the connection closes.

    Accepts a single connection, sends ``ready``, then loops processing
    ``execute`` and ``reset`` commands.
    """
    namespace: dict[str, Any] = {}

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("0.0.0.0", port))
    server_sock.listen(1)

    conn, _ = server_sock.accept()
    reader = MessageReader()

    namespace["__call_tool__"] = make_tool_caller(conn, reader)
    send_message(conn, {"type": "ready"})

    try:
        while True:
            msg = reader.recv(conn)
            msg_type = msg.get("type")

            if msg_type == "execute":
                result = execute_code(msg["code"], namespace, timeout=timeout)
                send_message(conn, result)

            elif msg_type == "reset":
                tool_caller = namespace.get("__call_tool__")
                namespace.clear()
                if tool_caller is not None:
                    namespace["__call_tool__"] = tool_caller
                send_message(conn, {"type": "ready"})

    except ConnectionError:
        pass
    finally:
        conn.close()
        server_sock.close()


if __name__ == "__main__":  # pragma: no cover — script entrypoint, executed only inside the sandbox container
    _timeout = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    serve(timeout=_timeout)
