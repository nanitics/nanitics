import json
import signal
import socket
import threading

import pytest

from nanitics.safety.sandbox._runner import (
    MessageReader,
    execute_code,
    make_tool_caller,
    prepare_code,
    send_message,
    serve,
)


class TestPrepareCode:
    def test_expression_only(self) -> None:
        exec_code, eval_code = prepare_code("42")
        ns: dict = {}
        exec(exec_code, ns)
        assert eval_code is not None
        assert eval(eval_code, ns) == 42

    def test_statement_then_expression(self) -> None:
        exec_code, eval_code = prepare_code("x = 10\nx + 1")
        ns: dict = {}
        exec(exec_code, ns)
        assert eval_code is not None
        assert eval(eval_code, ns) == 11

    def test_statements_only(self) -> None:
        exec_code, eval_code = prepare_code("x = 1\ny = 2")
        ns: dict = {}
        exec(exec_code, ns)
        assert eval_code is None
        assert ns["x"] == 1
        assert ns["y"] == 2

    def test_function_definition(self) -> None:
        exec_code, eval_code = prepare_code("def foo():\n    return 1")
        assert eval_code is None
        ns: dict = {}
        exec(exec_code, ns)
        assert ns["foo"]() == 1

    def test_empty_code(self) -> None:
        exec_code, eval_code = prepare_code("")
        assert eval_code is None
        ns: dict = {}
        exec(exec_code, ns)  # no-op, no error

    def test_syntax_error(self) -> None:
        with pytest.raises(SyntaxError):
            prepare_code("def foo(")

    def test_multiple_expressions(self) -> None:
        exec_code, eval_code = prepare_code("1\n2\n3")
        ns: dict = {}
        exec(exec_code, ns)
        assert eval_code is not None
        assert eval(eval_code, ns) == 3

    def test_print_as_last_expression(self) -> None:
        _exec_code, eval_code = prepare_code("print('hello')")
        assert eval_code is not None
        # print() returns None, which is a valid expression to eval

    def test_assignment_is_not_expression(self) -> None:
        _, eval_code = prepare_code("x = 42")
        assert eval_code is None

    def test_augmented_assignment(self) -> None:
        _, eval_code = prepare_code("x = 1\nx += 1")
        assert eval_code is None


class TestExecuteCode:
    def test_stdout_capture(self) -> None:
        result = execute_code("print('hello')", {})
        assert result["success"] is True
        assert result["stdout"] == "hello\n"
        assert result["stderr"] == ""

    def test_stderr_capture(self) -> None:
        result = execute_code("import sys; print('err', file=sys.stderr)", {})
        assert result["success"] is True
        assert result["stderr"] == "err\n"

    def test_return_value(self) -> None:
        result = execute_code("1 + 2", {})
        assert result["success"] is True
        assert result["return_value"] == "3"

    def test_return_value_with_statements(self) -> None:
        result = execute_code("x = 10\nx * 2", {})
        assert result["success"] is True
        assert result["return_value"] == "20"

    def test_no_return_value_for_statements(self) -> None:
        result = execute_code("x = 42", {})
        assert result["success"] is True
        assert result["return_value"] is None

    def test_none_return_value_not_captured(self) -> None:
        result = execute_code("print('hi')", {})
        assert result["success"] is True
        assert result["return_value"] is None
        assert result["stdout"] == "hi\n"

    def test_runtime_error(self) -> None:
        result = execute_code("x", {})
        assert result["success"] is False
        assert result["error"] is not None
        assert "NameError" in result["error"]

    def test_syntax_error(self) -> None:
        result = execute_code("def foo(", {})
        assert result["success"] is False
        assert result["error"] is not None
        assert "SyntaxError" in result["error"]

    def test_persistent_namespace(self) -> None:
        ns: dict = {}
        execute_code("x = 42", ns)
        result = execute_code("x + 1", ns)
        assert result["success"] is True
        assert result["return_value"] == "43"

    def test_duration_ms(self) -> None:
        result = execute_code("1 + 1", {})
        assert result["duration_ms"] >= 0

    def test_empty_code(self) -> None:
        result = execute_code("", {})
        assert result["success"] is True
        assert result["stdout"] == ""
        assert result["return_value"] is None

    def test_multiline_code(self) -> None:
        code = "for i in range(3):\n    print(i)"
        result = execute_code(code, {})
        assert result["success"] is True
        assert result["stdout"] == "0\n1\n2\n"

    def test_result_type_field(self) -> None:
        result = execute_code("1", {})
        assert result["type"] == "execution_result"

    def test_exception_preserves_partial_stdout(self) -> None:
        code = "print('before')\nraise ValueError('boom')"
        result = execute_code(code, {})
        assert result["success"] is False
        assert result["stdout"] == "before\n"
        assert "ValueError" in (result["error"] or "")

    @pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="SIGALRM not available")
    def test_timeout(self) -> None:
        code = "import time; time.sleep(10)"
        result = execute_code(code, {}, timeout=1)
        assert result["success"] is False
        assert result["error"] == "Execution timed out"

    def test_namespace_with_tool_caller(self) -> None:
        calls: list = []

        def fake_tool(name: str, args: dict | None = None) -> str:
            calls.append((name, args))
            return "result"

        ns: dict = {"__call_tool__": fake_tool}
        result = execute_code("r = __call_tool__('search', {'q': 'test'})", ns)
        assert result["success"] is True
        assert calls == [("search", {"q": "test"})]
        assert ns["r"] == "result"


class TestMessageProtocol:
    def test_send_and_receive(self) -> None:
        a, b = socket.socketpair()
        try:
            reader = MessageReader()
            send_message(a, {"type": "ready"})
            msg = reader.recv(b)
            assert msg == {"type": "ready"}
        finally:
            a.close()
            b.close()

    def test_multiple_messages(self) -> None:
        a, b = socket.socketpair()
        try:
            reader = MessageReader()
            send_message(a, {"type": "msg1", "data": 1})
            send_message(a, {"type": "msg2", "data": 2})
            assert reader.recv(b)["type"] == "msg1"
            assert reader.recv(b)["type"] == "msg2"
        finally:
            a.close()
            b.close()

    def test_connection_closed_raises(self) -> None:
        a, b = socket.socketpair()
        try:
            reader = MessageReader()
            a.close()
            with pytest.raises(ConnectionError):
                reader.recv(b)
        finally:
            b.close()

    def test_buffered_partial_reads(self) -> None:
        a, b = socket.socketpair()
        try:
            reader = MessageReader()
            # Send two messages as a single chunk
            data = json.dumps({"type": "first"}) + "\n" + json.dumps({"type": "second"}) + "\n"
            a.sendall(data.encode())
            assert reader.recv(b)["type"] == "first"
            assert reader.recv(b)["type"] == "second"
        finally:
            a.close()
            b.close()


class TestToolCaller:
    def test_successful_call(self) -> None:
        container_sock, host_sock = socket.socketpair()
        try:
            reader = MessageReader()
            host_reader = MessageReader()
            tool_caller = make_tool_caller(container_sock, reader)

            def respond():
                msg = host_reader.recv(host_sock)
                assert msg["type"] == "tool_call"
                assert msg["name"] == "search"
                assert msg["args"] == {"query": "test"}
                send_message(host_sock, {"type": "tool_result", "result": "found it", "error": None})

            t = threading.Thread(target=respond)
            t.start()
            result = tool_caller("search", {"query": "test"})
            t.join()
            assert result == "found it"
        finally:
            container_sock.close()
            host_sock.close()

    def test_tool_error(self) -> None:
        container_sock, host_sock = socket.socketpair()
        try:
            reader = MessageReader()
            host_reader = MessageReader()
            tool_caller = make_tool_caller(container_sock, reader)

            def respond():
                host_reader.recv(host_sock)
                send_message(host_sock, {"type": "tool_result", "result": "", "error": "not found"})

            t = threading.Thread(target=respond)
            t.start()
            with pytest.raises(RuntimeError, match="Tool 'search' failed"):
                tool_caller("search", None)
            t.join()
        finally:
            container_sock.close()
            host_sock.close()

    def test_unexpected_response_type(self) -> None:
        container_sock, host_sock = socket.socketpair()
        try:
            reader = MessageReader()
            host_reader = MessageReader()
            tool_caller = make_tool_caller(container_sock, reader)

            def respond():
                host_reader.recv(host_sock)
                send_message(host_sock, {"type": "something_else"})

            t = threading.Thread(target=respond)
            t.start()
            with pytest.raises(RuntimeError, match="Expected tool_result"):
                tool_caller("search", None)
            t.join()
        finally:
            container_sock.close()
            host_sock.close()

    def test_default_args(self) -> None:
        container_sock, host_sock = socket.socketpair()
        try:
            reader = MessageReader()
            host_reader = MessageReader()
            tool_caller = make_tool_caller(container_sock, reader)

            def respond():
                msg = host_reader.recv(host_sock)
                assert msg["args"] == {}
                send_message(host_sock, {"type": "tool_result", "result": "ok", "error": None})

            t = threading.Thread(target=respond)
            t.start()
            tool_caller("ping", None)
            t.join()
        finally:
            container_sock.close()
            host_sock.close()


class TestServe:
    def _start_server(self, port: int, timeout: int = 30) -> threading.Thread:
        t = threading.Thread(target=serve, args=(port, timeout), daemon=True)
        t.start()
        return t

    def _connect(self, port: int, max_retries: int = 20) -> socket.socket:
        import time

        for _ in range(max_retries):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect(("127.0.0.1", port))
                return s
            except ConnectionRefusedError:
                s.close()
                time.sleep(0.05)
        raise RuntimeError("Could not connect to runner")

    def test_ready_on_connect(self) -> None:
        port = 19901
        self._start_server(port)
        conn = self._connect(port)
        try:
            reader = MessageReader()
            msg = reader.recv(conn)
            assert msg == {"type": "ready"}
        finally:
            conn.close()

    def test_execute_and_result(self) -> None:
        port = 19902
        self._start_server(port)
        conn = self._connect(port)
        try:
            reader = MessageReader()
            reader.recv(conn)  # ready

            send_message(conn, {"type": "execute", "code": "1 + 1"})
            result = reader.recv(conn)
            assert result["type"] == "execution_result"
            assert result["success"] is True
            assert result["return_value"] == "2"
        finally:
            conn.close()

    def test_reset_clears_namespace(self) -> None:
        port = 19903
        self._start_server(port)
        conn = self._connect(port)
        try:
            reader = MessageReader()
            reader.recv(conn)  # ready

            send_message(conn, {"type": "execute", "code": "x = 42"})
            reader.recv(conn)  # execution_result

            send_message(conn, {"type": "reset"})
            msg = reader.recv(conn)
            assert msg == {"type": "ready"}

            send_message(conn, {"type": "execute", "code": "x"})
            result = reader.recv(conn)
            assert result["success"] is False
            assert "NameError" in result["error"]
        finally:
            conn.close()

    def test_reset_preserves_tool_caller(self) -> None:
        port = 19904
        self._start_server(port)
        conn = self._connect(port)
        try:
            reader = MessageReader()
            reader.recv(conn)  # ready

            send_message(conn, {"type": "reset"})
            reader.recv(conn)  # ready

            # __call_tool__ should still be available after reset
            send_message(conn, {"type": "execute", "code": "'__call_tool__' in dir()"})
            result = reader.recv(conn)
            assert result["success"] is True
            assert result["return_value"] == "True"
        finally:
            conn.close()

    def test_persistent_namespace_across_executions(self) -> None:
        port = 19905
        self._start_server(port)
        conn = self._connect(port)
        try:
            reader = MessageReader()
            reader.recv(conn)  # ready

            send_message(conn, {"type": "execute", "code": "x = 100"})
            reader.recv(conn)

            send_message(conn, {"type": "execute", "code": "x + 1"})
            result = reader.recv(conn)
            assert result["return_value"] == "101"
        finally:
            conn.close()

    def test_tool_call_bridge(self) -> None:
        port = 19906
        self._start_server(port)
        conn = self._connect(port)
        try:
            reader = MessageReader()
            reader.recv(conn)  # ready

            # Execute code that calls a tool
            send_message(
                conn,
                {"type": "execute", "code": "result = __call_tool__('echo', {'msg': 'hi'})\nresult"},
            )

            # Runner should send a tool_call
            tool_call = reader.recv(conn)
            assert tool_call["type"] == "tool_call"
            assert tool_call["name"] == "echo"
            assert tool_call["args"] == {"msg": "hi"}

            # Respond with tool_result
            send_message(conn, {"type": "tool_result", "result": "hi back", "error": None})

            # Get execution result
            result = reader.recv(conn)
            assert result["type"] == "execution_result"
            assert result["success"] is True
            assert result["return_value"] == "'hi back'"
        finally:
            conn.close()
