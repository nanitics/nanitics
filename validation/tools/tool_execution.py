"""Tool schema population under a real LLM.

Validates that a real LLM populates a non-trivial tool schema correctly:
enum fields are filled with the enum's exact values and numeric fields are
parsed from the natural-language task. Schema population — not mock-sequenced
tool dispatch — is the subject of this script.

Acceptance criteria:
  - Serialised schema pins shape: ``value`` is ``number`` and ``enum`` is
    populated with the full ``{celsius, fahrenheit, kelvin}`` set for both
    unit fields; all three parameters are required.
  - Agent invokes ``convert_temperature`` for the first leg with
    ``from_unit=celsius``, ``to_unit=fahrenheit``, ``value≈42``.
  - Agent invokes ``convert_temperature`` for the second leg with
    ``from_unit=fahrenheit``, ``to_unit=kelvin`` (exercising the third enum
    member and loop+tool-state round-trip).
  - Both ``ToolResultEvent``s report ``success=True``, non-empty ``result``
    content, and positive ``duration_ms``.
  - Final answer reports the kelvin value (~315.9 K) within the fuzzy-judge
    tolerance.
"""

from __future__ import annotations

from typing import Literal

import pytest

from nanitics import InMemoryEmitter, ReActAgent, tool
from nanitics.infrastructure import ToolInvokeEvent, ToolResultEvent
from validation.helpers import (
    assert_result_satisfies,
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)

Unit = Literal["celsius", "fahrenheit", "kelvin"]


@tool(
    "convert_temperature",
    "Convert a temperature value between Celsius, Fahrenheit, and Kelvin.",
)
async def convert_temperature(value: float, from_unit: Unit, to_unit: Unit) -> str:
    to_c = {"celsius": value, "fahrenheit": (value - 32.0) * 5.0 / 9.0, "kelvin": value - 273.15}
    celsius = to_c[from_unit]
    from_c = {"celsius": celsius, "fahrenheit": celsius * 9.0 / 5.0 + 32.0, "kelvin": celsius + 273.15}
    return f"{value} {from_unit} = {from_c[to_unit]:.2f} {to_unit}"


@pytest.mark.quick
async def test_tool_execution(traced_emitter: InMemoryEmitter) -> None:
    # --- Direct schema-shape assertion (the subject of this script) ---
    parameters = convert_temperature.schema.parameters
    required = set(parameters.get("required", []))
    assert required == {"value", "from_unit", "to_unit"}, (
        f"Expected required={{value, from_unit, to_unit}}, got {required}."
    )
    props = parameters.get("properties", {})
    value_schema = props.get("value", {})
    assert value_schema.get("type") == "number", f"Expected value.type='number', got {value_schema.get('type')!r}."
    expected_enum = ["celsius", "fahrenheit", "kelvin"]
    for unit_field in ("from_unit", "to_unit"):
        # Pydantic emits enum-typed fields as $ref'd definitions; resolve if needed.
        field_schema = props.get(unit_field, {})
        if "enum" in field_schema:
            enum_values = field_schema["enum"]
        else:
            ref = field_schema.get("$ref", "")
            def_name = ref.rsplit("/", 1)[-1] if ref else ""
            defs = parameters.get("$defs", {}) or parameters.get("definitions", {})
            enum_values = defs.get(def_name, {}).get("enum", [])
        assert sorted(enum_values) == sorted(expected_enum), (
            f"Expected {unit_field} enum={expected_enum}, got {enum_values}."
        )

    # --- End-to-end run with chained conversion (exercises loop + all three enum members) ---
    client = make_llm_client("anthropic")
    agent = ReActAgent(
        name="tool-execution-agent",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You are a helpful assistant. Use the provided tool to perform "
            "temperature conversions, then report the result to the user. "
            "When a task requires converting through an intermediate unit, "
            "call the tool once per leg."
        ),
        tools=[convert_temperature],
        max_iterations=5,
    )

    result = await run_with_retry(
        lambda: agent.run(
            "Convert 42 degrees Celsius to Fahrenheit using the convert_temperature tool, "
            "then convert that Fahrenheit result to Kelvin with the same tool, "
            "and report the final Kelvin value."
        ),
        max_attempts=2,
    )

    # --- First leg: C -> F ---
    first = assert_trace_contains(
        traced_emitter,
        ToolInvokeEvent,
        predicate=lambda e: (
            e.tool_name == "convert_temperature"
            and e.parameters.get("from_unit") == "celsius"
            and e.parameters.get("to_unit") == "fahrenheit"
        ),
    )
    assert abs(float(first.parameters["value"]) - 42.0) < 0.01, f"Expected first-leg value≈42, got: {first.parameters}"
    first_result = assert_trace_contains(
        traced_emitter,
        ToolResultEvent,
        predicate=lambda e: e.tool_name == "convert_temperature" and e.tool_call_id == first.tool_call_id,
    )
    assert first_result.success is True, f"First-leg result not success: {first_result.error}"
    assert first_result.result, f"First-leg result content empty: {first_result!r}"
    assert first_result.duration_ms > 0, (
        f"Expected positive duration_ms on first leg, got {first_result.duration_ms!r}."
    )

    # --- Second leg: F -> K (kelvin enum member) ---
    second = assert_trace_contains(
        traced_emitter,
        ToolInvokeEvent,
        predicate=lambda e: (
            e.tool_name == "convert_temperature"
            and e.parameters.get("from_unit") == "fahrenheit"
            and e.parameters.get("to_unit") == "kelvin"
        ),
    )
    second_result = assert_trace_contains(
        traced_emitter,
        ToolResultEvent,
        predicate=lambda e: e.tool_name == "convert_temperature" and e.tool_call_id == second.tool_call_id,
    )
    assert second_result.success is True, f"Second-leg result not success: {second_result.error}"
    assert second_result.result, f"Second-leg result content empty: {second_result!r}"
    assert second_result.duration_ms > 0, (
        f"Expected positive duration_ms on second leg, got {second_result.duration_ms!r}."
    )

    # --- Fuzzy output: 42°C == 107.6°F == ~315.15 K ---
    await assert_result_satisfies(
        result.output or "",
        "The output reports the final temperature in Kelvin as approximately 315 K (acceptable range 314-317 K).",
    )
