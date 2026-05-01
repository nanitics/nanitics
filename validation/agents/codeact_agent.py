"""CodeActAgent validation: code generation, sandboxed execution, self-correction, tool bridge, and state persistence.

Acceptance criteria:
  - Trace contains a ``CodeExecutionEvent`` (proves code was generated).
  - Trace contains a ``CodeExecutionResultEvent`` with ``success=True``.
  - The expected answer appears in a ``CodeExecutionResultEvent`` payload
    (``stdout`` or ``return_value``) — ties the final output to the sandbox
    round-trip rather than LLM recall. The task (SHA-256 prefix) is not a
    fact the model can recover without executing code.
  - Fuzzy judge check: final output reports the expected hex digest prefix.
  - ``result.total_steps`` stays within the ``max_iterations`` budget.

Self-correction section (``test_codeact_self_correction``):
  - At least one ``CodeExecutionResultEvent`` with ``success=False`` fires
    (deliberate bug in the provided snippet).
  - A ``success=True`` event follows the failure in emission order — proves
    the agent read the traceback and produced a working fix.
  - The corrected answer propagates through both the execution event and
    the final output.

Tool bridge section (``test_codeact_tool_bridge``):
  - Trace contains a ``CodeExecutionEvent`` and a ``CodeExecutionResultEvent``
    with ``success=True``.
  - Trace contains a ``ToolInvokeEvent`` with
    ``tool_name == "lookup_magic_number"`` — fires only when the host-side
    ``ToolRegistry.dispatch`` actually ran, proving the sandbox Python call
    crossed back through ``__call_tool__``.
  - The pinned magic-number string (``4242``) returned by the registered
    ``lookup_magic_number`` tool appears in a successful
    ``CodeExecutionResultEvent`` payload — proves the tool's return value
    made it back into the sandbox as a Python return value.
  - The pinned signature string (``sig-85``) returned by
    ``compose_signature(a=4, b=17)`` appears in a successful
    ``CodeExecutionResultEvent`` payload.
  - Both pinned values appear in the final ``result.output``.
  - ``result.total_steps`` stays within the ``max_iterations`` budget.
  - No fuzzy-judge check here: the bridge mechanism is verified via the
    ``ToolInvokeEvent`` and canned-value payload assertions above.
    A judge call cannot verify *process* claims ("constructed by calling
    host-registered helpers") from output text alone — that would be a
    category error.

State persistence section (``test_codeact_sandbox_state_persistence``):
  - At least three ``CodeExecutionEvent`` entries fire, one per numbered step.
  - At least three ``CodeExecutionResultEvent`` entries with ``success=True``
    fire; no ``success=False`` precedes the first success (guards against a
    kernel-reset regression where the step-2 ``counter += 5`` would raise
    ``NameError``).
  - The pinned token (``cz-state-9f2a4c``) set in step 1 appears in the
    ``stdout`` / ``return_value`` of the final successful
    ``CodeExecutionResultEvent``.
  - The pinned counter value (``11``) — reachable only by mutating the same
    ``counter`` variable across three turns — appears in the ``stdout`` /
    ``return_value`` of the final successful ``CodeExecutionResultEvent``.
  - Both pinned values appear in the final ``result.output``.
  - ``result.total_steps`` stays within the ``max_iterations`` budget.
  - Fuzzy judge check: output reports ``final: cz-state-9f2a4c|11`` as a
    value reused from prior sandboxed steps, not recomputed in a single step.

Requires the Docker daemon. ``@requires_docker`` skips cleanly when not present.
"""

from __future__ import annotations

from nanitics import CodeActAgent, DockerSandbox, InMemoryEmitter, tool
from nanitics.infrastructure import CodeExecutionEvent, CodeExecutionResultEvent, ToolInvokeEvent
from validation.helpers import (
    assert_result_satisfies,
    assert_trace_contains,
    make_llm_client,
    requires_docker,
    run_with_retry,
)

# SHA-256(b"nanitics-validation").hexdigest()[:8] — computed once; pinned so a
# broken sandbox that returns empty/garbage output cannot satisfy the assertion.
_EXPECTED_DIGEST_PREFIX = "08682abb"

# Tool-bridge test constants. The magic number "4242" exists *only* in the
# host-side body of ``lookup_magic_number`` — it is not in any prompt — so a
# regression that breaks the bridge (stubs not injected, dispatcher not wired,
# __call_tool__ routing broken) cannot satisfy the magic-number payload check.
_TOOL_BRIDGE_MAGIC = "4242"
# compose_signature(a=4, b=17) -> f"sig-{4 * 17 + 17}" -> "sig-85".
_TOOL_BRIDGE_SIGNATURE = "sig-85"

# State-persistence test constants. The token is deliberately non-guessable and
# only present in step 1's prompted code literal; the counter value "11" is
# only reachable by mutating ``counter`` (3 -> 8 -> 11) across three separate
# ``execute_code`` turns. A kernel-reset regression would raise ``NameError``
# on step 2 before any counter assertion can pass.
_STATE_TOKEN = "cz-state-9f2a4c"
_STATE_COUNTER_FINAL = 11


def _result_events(emitter: InMemoryEmitter) -> list[CodeExecutionResultEvent]:
    return [e for e in emitter.events if isinstance(e, CodeExecutionResultEvent)]


def _payload_contains(event: CodeExecutionResultEvent, needle: str) -> bool:
    return needle in (event.stdout or "") or needle in (event.return_value or "")


@requires_docker
async def test_codeact_sha256_prefix(traced_emitter: InMemoryEmitter) -> None:
    client = make_llm_client("anthropic")

    async with DockerSandbox() as sandbox:
        agent = CodeActAgent(
            name="codeact-agent",
            llm_client=client,
            emitter=traced_emitter,
            system_prompt=(
                "You are a Python assistant. Use the execute_code tool to run "
                "Python and report results. Do not answer from memory — always "
                "compute hashes and other non-trivial values by running code."
            ),
            sandbox=sandbox,
            max_iterations=5,
        )

        result = await run_with_retry(
            lambda: agent.run(
                "Compute the SHA-256 hex digest of the bytes b'nanitics-validation' "
                "and report the first 8 hex characters of the digest. Use hashlib."
            ),
            max_attempts=2,
        )

    # --- Trace-shape invariants ---
    assert_trace_contains(traced_emitter, CodeExecutionEvent)
    assert_trace_contains(
        traced_emitter,
        CodeExecutionResultEvent,
        predicate=lambda e: e.success is True,
    )

    # --- Distinguishing: the answer came from the sandbox, not from LLM recall ---
    successful = [e for e in _result_events(traced_emitter) if e.success]
    assert any(_payload_contains(e, _EXPECTED_DIGEST_PREFIX) for e in successful), (
        f"Expected {_EXPECTED_DIGEST_PREFIX!r} in a successful CodeExecutionResultEvent "
        f"payload, got: {[(e.stdout, e.return_value) for e in successful]!r}"
    )

    # --- Final output reflects the executed result ---
    assert _EXPECTED_DIGEST_PREFIX in (result.output or ""), (
        f"Expected {_EXPECTED_DIGEST_PREFIX!r} in output, got: {result.output!r}"
    )

    # --- Iteration budget ---
    assert result.total_steps <= 5, f"Expected <=5 steps, got: {result.total_steps}"

    # --- Fuzzy judge (defensive against formatting variations) ---
    await assert_result_satisfies(
        result.output or "",
        f"The output reports that the first 8 hex characters of the SHA-256 digest are {_EXPECTED_DIGEST_PREFIX}.",
    )


@requires_docker
async def test_codeact_self_correction(traced_emitter: InMemoryEmitter) -> None:
    """Buggy snippet → failed execution → traceback observation → fix → answer."""

    client = make_llm_client("anthropic")

    # Missing colon after ``for n in nums`` — guaranteed SyntaxError on first run.
    buggy_snippet = "nums = [1, 2, 3, 4, 5]\ntotal = 0\nfor n in nums\n    total += n * n\nprint(total)\n"

    async with DockerSandbox() as sandbox:
        agent = CodeActAgent(
            name="codeact-fixer",
            llm_client=client,
            emitter=traced_emitter,
            system_prompt=(
                "You are a Python debugging assistant. When given a snippet, "
                "first execute it exactly as provided — do not silently repair "
                "it before running. Observe the traceback, then fix the bug, "
                "re-run, and report the final numeric result."
            ),
            sandbox=sandbox,
            max_iterations=6,
        )

        result = await run_with_retry(
            lambda: agent.run(
                "Here is a Python snippet that is supposed to print the sum of "
                "squares of the list [1, 2, 3, 4, 5]. Execute it as-is first, "
                "then fix the bug based on the traceback, re-run, and report "
                "the final number.\n\n"
                f"```python\n{buggy_snippet}```"
            ),
            max_attempts=2,
        )

    # --- Trace-shape invariants ---
    results = _result_events(traced_emitter)
    assert len(results) >= 2, f"Expected >=2 code executions (bug + fix), got: {len(results)}"

    # --- Distinguishing: failure observed, then success, in that order ---
    first_failure = next((i for i, e in enumerate(results) if not e.success), None)
    assert first_failure is not None, (
        "Expected at least one failed execution from the buggy snippet, got none — "
        "agent likely bypassed the observe-then-fix loop. "
        f"Results: {[(e.success, e.error) for e in results]!r}"
    )
    later_success = next(
        (i for i, e in enumerate(results) if i > first_failure and e.success),
        None,
    )
    assert later_success is not None, (
        "Expected a successful execution after the initial failure, got none. "
        f"Results: {[(e.success, e.error) for e in results]!r}"
    )

    # --- The fix actually produced the correct answer (1+4+9+16+25 = 55) ---
    assert _payload_contains(results[later_success], "55"), (
        "Expected '55' in the post-fix execution payload, got: "
        f"stdout={results[later_success].stdout!r} "
        f"return_value={results[later_success].return_value!r}"
    )
    assert "55" in (result.output or ""), f"Expected '55' in final output, got: {result.output!r}"

    # --- Iteration budget ---
    assert result.total_steps <= 6, f"Expected <=6 steps, got: {result.total_steps}"

    # --- Fuzzy judge ---
    await assert_result_satisfies(
        result.output or "",
        "The output acknowledges that the original snippet had a bug (such as "
        "a syntax error), reports that after fixing it the sum of squares is 55.",
    )


# --- Tool bridge fixtures ---------------------------------------------------


_MAGIC_TABLE = {
    "alpha": "The magic number for alpha is 4242.",
    "beta": "The magic number for beta is 1337.",
    "gamma": "The magic number for gamma is 2718.",
}


@tool("lookup_magic_number", "Look up the magic number associated with a key.")
async def lookup_magic_number(key: str) -> str:
    return _MAGIC_TABLE.get(key, "Unknown key.")


@tool("compose_signature", "Compose a signature string from two integers.")
async def compose_signature(a: int, b: int) -> str:
    return f"sig-{a * b + 17}"


@requires_docker
async def test_codeact_tool_bridge(traced_emitter: InMemoryEmitter) -> None:
    """SDK tools exposed as callable Python functions in the sandbox.

    Proves the full bridge path: stub injected into the sandbox namespace ->
    LLM calls the stub by name -> ``__call_tool__`` fires inside the container
    -> ``ToolRegistry.dispatch`` runs on the host and emits ``ToolInvokeEvent``
    -> tool return value appears as the Python return value of the stub call
    -> value observable in ``CodeExecutionResultEvent.stdout``.
    """

    client = make_llm_client("anthropic")

    async with DockerSandbox() as sandbox:
        agent = CodeActAgent(
            name="codeact-bridge",
            llm_client=client,
            emitter=traced_emitter,
            system_prompt=(
                "You are a Python assistant with access to helper functions "
                "registered by the host. When you need information from the "
                "host, call the registered functions by name directly in "
                "Python — do NOT redefine them, do NOT hardcode their return "
                "values, and do NOT guess. The registered functions are the "
                "only source of truth for these lookups. After each execution, "
                "read the observation and decide the next step."
            ),
            sandbox=sandbox,
            tools=[lookup_magic_number, compose_signature],
            max_iterations=6,
        )

        result = await run_with_retry(
            lambda: agent.run(
                "Using the registered helper functions, do the following, "
                "step by step, in Python code: (1) call "
                "`lookup_magic_number('alpha')` and extract the integer from "
                "its return string; (2) call `compose_signature(a=4, b=17)` "
                "to obtain a signature string; (3) print a single line of the "
                "form `result: <magic>|<signature>`, where `<magic>` is the "
                "integer from step 1 and `<signature>` is the string from "
                "step 2."
            ),
            max_attempts=2,
        )

    # --- Trace-shape invariants ---
    assert_trace_contains(traced_emitter, CodeExecutionEvent)
    assert_trace_contains(
        traced_emitter,
        CodeExecutionResultEvent,
        predicate=lambda e: e.success is True,
    )

    # --- Distinguishing: host-side dispatch actually ran ---
    # ToolInvokeEvent fires from ToolRegistry.dispatch, which is only reached
    # when the sandbox stub routes __call_tool__ back to the host. A regression
    # that no-ops the dispatcher, skips stub injection, or leads the LLM to
    # reimplement the tool inline will leave this event absent.
    assert_trace_contains(
        traced_emitter,
        ToolInvokeEvent,
        predicate=lambda e: e.tool_name == "lookup_magic_number",
    )

    # --- Distinguishing: canned tool return values came back into the sandbox ---
    successful = [e for e in _result_events(traced_emitter) if e.success]
    assert any(_payload_contains(e, _TOOL_BRIDGE_MAGIC) for e in successful), (
        f"Expected {_TOOL_BRIDGE_MAGIC!r} (host-side tool body) in a "
        f"successful CodeExecutionResultEvent payload, got: "
        f"{[(e.stdout, e.return_value) for e in successful]!r}"
    )
    assert any(_payload_contains(e, _TOOL_BRIDGE_SIGNATURE) for e in successful), (
        f"Expected {_TOOL_BRIDGE_SIGNATURE!r} in a successful "
        f"CodeExecutionResultEvent payload, got: "
        f"{[(e.stdout, e.return_value) for e in successful]!r}"
    )

    # --- Final output reflects the bridged values ---
    assert _TOOL_BRIDGE_MAGIC in (result.output or ""), (
        f"Expected {_TOOL_BRIDGE_MAGIC!r} in output, got: {result.output!r}"
    )
    assert _TOOL_BRIDGE_SIGNATURE in (result.output or ""), (
        f"Expected {_TOOL_BRIDGE_SIGNATURE!r} in output, got: {result.output!r}"
    )

    # --- Iteration budget ---
    assert result.total_steps <= 6, f"Expected <=6 steps, got: {result.total_steps}"

    # No fuzzy-judge call here. Process verification ("constructed by calling
    # host-registered helpers") is structurally unverifiable from output text
    # alone — the bridge mechanism is proven by the ToolInvokeEvent assertion
    # above and by the canned-value (4242, sig-85) checks against the
    # CodeExecutionResultEvent payload, which together constitute the
    # distinguishing acceptance set.


@requires_docker
async def test_codeact_sandbox_state_persistence(traced_emitter: InMemoryEmitter) -> None:
    """Variables set in one ``execute_code`` turn survive into later turns.

    Three-turn task: store a non-guessable token plus a counter, mutate the
    counter twice across separate ``execute_code`` calls, then read both back.
    A kernel-reset regression (new Python process per call) would raise
    ``NameError`` on the second turn before any counter assertion can pass.
    """

    client = make_llm_client("anthropic")

    async with DockerSandbox() as sandbox:
        agent = CodeActAgent(
            name="codeact-persist",
            llm_client=client,
            emitter=traced_emitter,
            system_prompt=(
                "You are a Python assistant working in a persistent sandbox. "
                "Variables, imports, and function definitions you create in "
                "one `execute_code` call remain available in later calls. Use "
                "that persistence — do not redefine values you have already "
                "stored, and do not inline-recompute values you stored "
                "earlier. Follow the user's instructions one step at a time; "
                "make one `execute_code` call per numbered step unless "
                "explicitly told otherwise."
            ),
            sandbox=sandbox,
            max_iterations=6,
        )

        result = await run_with_retry(
            lambda: agent.run(
                "Do the following in separate `execute_code` calls, one per "
                "step:\n\n"
                f"(1) Store the string '{_STATE_TOKEN}' in a variable named "
                "`stored_token`, and store the integer 3 in a variable named "
                "`counter`. Do not print anything yet.\n\n"
                "(2) Without reassigning `stored_token`, increase `counter` "
                "by 5. Print `counter` and nothing else.\n\n"
                "(3) Without reassigning `stored_token`, increase `counter` "
                "by 3 again. Print a single line of the form "
                "`final: <stored_token>|<counter>`.\n\n"
                "(4) When you are done executing code, report the final "
                "printed line as your answer."
            ),
            max_attempts=2,
        )

    # --- Trace-shape invariants ---
    exec_events = [e for e in traced_emitter.events if isinstance(e, CodeExecutionEvent)]
    assert len(exec_events) >= 3, f"Expected >=3 code executions (one per numbered step), got: {len(exec_events)}"

    results = _result_events(traced_emitter)
    successes = [e for e in results if e.success]
    assert len(successes) >= 3, (
        f"Expected >=3 successful executions, got: {len(successes)}. "
        f"Full results: {[(e.success, e.error) for e in results]!r}"
    )

    # --- Distinguishing: no failure precedes the first success ---
    # A kernel reset between turns would make step 2's `counter += 5` raise
    # NameError *before* any success could land — inverting the order.
    first_success_idx = next(i for i, e in enumerate(results) if e.success)
    first_failure_idx = next(
        (i for i, e in enumerate(results) if not e.success),
        None,
    )
    if first_failure_idx is not None:
        assert first_success_idx <= first_failure_idx, (
            "Expected the first successful execution to land before any "
            "failure (a kernel-reset regression would invert this order, with "
            "step 2 raising NameError on `counter`). Observed: "
            f"{[(e.success, e.error) for e in results]!r}"
        )

    # --- Distinguishing: persisted token + counter visible in the final success ---
    last_success = successes[-1]
    assert _payload_contains(last_success, _STATE_TOKEN), (
        f"Expected {_STATE_TOKEN!r} in the final successful execution "
        f"payload — proves step 3 read a token stored in step 1. Got: "
        f"stdout={last_success.stdout!r} return_value={last_success.return_value!r}"
    )
    assert _payload_contains(last_success, str(_STATE_COUNTER_FINAL)), (
        f"Expected {str(_STATE_COUNTER_FINAL)!r} in the final successful "
        f"execution payload — reachable only by mutating `counter` across "
        f"three turns. Got: stdout={last_success.stdout!r} "
        f"return_value={last_success.return_value!r}"
    )

    # --- Final output reflects the persisted values ---
    assert _STATE_TOKEN in (result.output or ""), f"Expected {_STATE_TOKEN!r} in output, got: {result.output!r}"
    assert str(_STATE_COUNTER_FINAL) in (result.output or ""), (
        f"Expected {str(_STATE_COUNTER_FINAL)!r} in output, got: {result.output!r}"
    )

    # --- Iteration budget ---
    assert result.total_steps <= 6, f"Expected <=6 steps, got: {result.total_steps}"

    # --- Fuzzy judge ---
    await assert_result_satisfies(
        result.output or "",
        f"The output reports a final line of the form 'final: {_STATE_TOKEN}|{_STATE_COUNTER_FINAL}'.",
    )
