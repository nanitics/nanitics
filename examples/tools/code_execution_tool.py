"""Code-execution reference tool: ``create_code_execution_tool`` wraps any Sandbox.

Demonstrates ``create_code_execution_tool`` — the fourth of the four curated
reference tools.  The factory is a thin adapter over the ``Sandbox`` protocol:
it takes any object conforming to it (``MockSandbox``, ``DockerSandbox``, or a
user-supplied implementation) and exposes ``execute(code=...)`` as a tool that
satisfies the ordinary ``Tool`` protocol.  The tool does NOT own the sandbox
lifecycle — the caller enters the sandbox's async context manager around the
agent run.

Section 1 runs the happy path against ``MockSandbox``: one successful execution
and one failing execution, both routed through a ``MockLLMClient``-backed
``ReActAgent``.  The failing case shows that sandbox-level failures surface
through ``ToolResult.metadata.success=False`` and an ``error:`` prefix in the
content — the LLM is expected to read stderr and try again, not crash.
Section 2 attempts the same against ``DockerSandbox`` and skips gracefully when
the Docker daemon is unavailable, mirroring the pattern in
``examples/tools/sandbox.py``.

Related guide: docs/guides/tools.md (see the "Reference Tools" section).
"""

from __future__ import annotations

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics.infrastructure import MockLLMClient, ToolInvokeEvent, ToolResultEvent
from nanitics.safety import (
    ExecutionResult,
    MockSandbox,
)
from nanitics.strategies import ReActAgent
from nanitics.tools import create_code_execution_tool
from nanitics.tracing import ToolCall


async def main() -> None:
    # --- Section 1: MockSandbox via ReActAgent (always runs) ---
    print("--- Section 1: MockSandbox via ReActAgent (hermetic) ---")

    # Two scripted outcomes: one successful run, one run that fails with a
    # NameError — the agent observes both and can react to each.
    responses = [
        ExecutionResult(
            stdout="42\n",
            stderr="",
            return_value="42",
            success=True,
            error=None,
            duration_ms=3.2,
        ),
        ExecutionResult(
            stdout="",
            stderr="Traceback (most recent call last):\n  NameError: name 'x' is not defined\n",
            return_value=None,
            success=False,
            error="NameError: name 'x' is not defined",
            duration_ms=1.1,
        ),
    ]

    sandbox = MockSandbox(responses=responses)
    code_tool = create_code_execution_tool(sandbox=sandbox)
    assert code_tool.schema.name == "code_execution"

    llm = MockLLMClient(
        responses=[
            make_response(
                "Running your code.",
                tool_calls=[
                    ToolCall(
                        id="tc-run-1",
                        name="code_execution",
                        arguments={"code": "print(40 + 2)"},
                    )
                ],
                stop_reason="tool_use",
            ),
            make_response(
                "Let me try a different snippet.",
                tool_calls=[
                    ToolCall(
                        id="tc-run-2",
                        name="code_execution",
                        arguments={"code": "print(x)  # deliberately fails"},
                    )
                ],
                stop_reason="tool_use",
            ),
            make_response("First produced 42; the second raised NameError because `x` was not defined."),
        ]
    )
    emitter = make_emitter("code-exec-section-1")

    # The sandbox lifecycle is owned by the caller — enter its context manager
    # around the agent run.  ``MockSandbox.start``/``cleanup`` are no-ops but
    # the same pattern applies to real ``DockerSandbox`` containers.
    async with sandbox:
        agent = ReActAgent(
            name="code-agent",
            llm_client=llm,
            emitter=emitter,
            system_prompt="Use code_execution to run Python on behalf of the user.",
            tools=[code_tool],
        )
        result = await agent.run("Compute 40 + 2, then try printing an undefined variable.")

    assert result.termination_reason == "complete"

    invokes = [e for e in emitter.events if isinstance(e, ToolInvokeEvent)]
    results = [e for e in emitter.events if isinstance(e, ToolResultEvent)]
    assert [e.tool_name for e in invokes] == ["code_execution", "code_execution"]
    # Both events report success=True because the tool surfaced the sandbox
    # failure via ToolResult metadata rather than by raising.
    assert all(e.success for e in results)
    assert results[0].result is not None and "42" in results[0].result
    assert results[1].result is not None and "NameError" in results[1].result
    assert results[1].result.startswith("error:")

    print(f"  Output: {result.output}")
    print(f"  Events: {len(invokes)} invoke, {len(results)} result (both success=True at the tool layer)")
    print(f"  Run 1 content (first line): {results[0].result.splitlines()[0] if results[0].result else ''!r}")
    print(f"  Run 2 content (first line): {results[1].result.splitlines()[0] if results[1].result else ''!r}")
    print("✓ Sandbox-level failures surface through metadata, not exceptions")

    # --- Section 2: DockerSandbox (conditional — skips when Docker unavailable) ---
    print("\n--- Section 2: DockerSandbox (conditional) ---")
    try:
        import docker as _docker_client_lib  # noqa: F401
        from nanitics.safety import (
            DockerSandbox,
            SandboxConfig,
        )

        docker_available = True
    except ImportError:
        docker_available = False
        skip_reason = "docker SDK not installed (pip install nanitics[code_execution])"

    if docker_available:
        try:
            # Touch the Docker daemon to distinguish "SDK installed but daemon
            # down" from "SDK installed and daemon healthy".  Mirrors the
            # graceful-skip pattern used in examples/tools/sandbox.py.
            docker_sandbox = DockerSandbox(config=SandboxConfig())
            async with docker_sandbox:
                exec_result = await docker_sandbox.execute("print('hello from docker')")
                assert "hello from docker" in exec_result.stdout
                # The same create_code_execution_tool wrapper works unchanged.
                docker_tool = create_code_execution_tool(sandbox=docker_sandbox)
                direct = await docker_tool.execute(code="print(2 ** 10)")
                assert "1024" in direct.content
            print(f"  Docker stdout: {exec_result.stdout.strip()!r}")
            print(f"  Wrapped tool output contains: {'1024' in direct.content}")
            print("✓ DockerSandbox works identically under create_code_execution_tool")
        except Exception as exc:
            # Environment-specific: Docker may be installed but daemon stopped.
            print(f"  Docker daemon unavailable — skipping: {type(exc).__name__}: {exc}")
    else:
        print(f"  {skip_reason} — skipping")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
