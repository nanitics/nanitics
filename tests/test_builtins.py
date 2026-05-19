"""Tests for built-in tool implementations (calculator, text_length, current_datetime).

These test the @tool decorator and FunctionTool behavior.
"""

import ast
import operator
import re
from collections.abc import Callable
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pytest

from nanitics.strategies import tool

# --- Inline tool definitions (identical to former builtins.py) ---

_SAFE_OPS: dict[type, Callable[..., float | int]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return op_fn(left, right)
    if isinstance(node, ast.UnaryOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_fn(_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@tool(
    "calculator",
    "Evaluate a mathematical expression safely. Supports +, -, *, /, //, %, ** and parentheses.",
)
async def calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression: {e}") from e
    result = _safe_eval(tree)
    is_finite_int = (
        isinstance(result, float) and result == int(result) and not (result == float("inf") or result == float("-inf"))
    )
    if is_finite_int:
        return str(int(result))
    return str(result)


@tool("text_length", "Count characters, words, and sentences in a given text.")
async def text_length(text: str) -> str:
    char_count = len(text)
    word_count = len(text.split()) if text.strip() else 0
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sentence_count = len(sentences)
    return f"Characters: {char_count}, Words: {word_count}, Sentences: {sentence_count}"


@tool("current_datetime", "Return the current date and time in ISO format.")
async def current_datetime(timezone_name: str = "UTC") -> str:
    try:
        tz = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, KeyError) as e:
        raise ValueError(f"Invalid timezone: {timezone_name}") from e
    return datetime.now(tz).isoformat()


class TestCalculator:
    async def test_basic_arithmetic(self) -> None:
        result = await calculator.execute(expression="2 + 3")
        assert result.content == "5"

    async def test_multiplication(self) -> None:
        result = await calculator.execute(expression="6 * 7")
        assert result.content == "42"

    async def test_division(self) -> None:
        result = await calculator.execute(expression="10 / 3")
        assert "3.333" in result.content

    async def test_parentheses(self) -> None:
        result = await calculator.execute(expression="(2 + 3) * 4")
        assert result.content == "20"

    async def test_exponentiation(self) -> None:
        result = await calculator.execute(expression="2 ** 10")
        assert result.content == "1024"

    async def test_negative_numbers(self) -> None:
        result = await calculator.execute(expression="-5 + 3")
        assert result.content == "-2"

    async def test_floating_point(self) -> None:
        result = await calculator.execute(expression="3.14 * 2")
        assert result.content == "6.28"

    async def test_floor_division(self) -> None:
        result = await calculator.execute(expression="7 // 2")
        assert result.content == "3"

    async def test_modulo(self) -> None:
        result = await calculator.execute(expression="10 % 3")
        assert result.content == "1"

    async def test_invalid_expression(self) -> None:
        with pytest.raises(ValueError, match="Invalid expression"):
            await calculator.execute(expression="2 +")

    async def test_rejects_function_calls(self) -> None:
        with pytest.raises(ValueError, match="Unsupported expression"):
            await calculator.execute(expression="__import__('os').system('ls')")

    async def test_rejects_variables(self) -> None:
        with pytest.raises(ValueError, match="Unsupported expression"):
            await calculator.execute(expression="x + 1")

    async def test_division_by_zero(self) -> None:
        with pytest.raises(ZeroDivisionError):
            await calculator.execute(expression="1 / 0")

    async def test_has_schema(self) -> None:
        assert calculator.schema.name == "calculator"
        assert "expression" in calculator.schema.parameters.get("properties", {})


class TestTextLength:
    async def test_basic_text(self) -> None:
        result = await text_length.execute(text="Hello world")
        assert "Characters: 11" in result.content
        assert "Words: 2" in result.content

    async def test_empty_string(self) -> None:
        result = await text_length.execute(text="")
        assert "Characters: 0" in result.content
        assert "Words: 0" in result.content
        assert "Sentences: 0" in result.content

    async def test_multiple_sentences(self) -> None:
        result = await text_length.execute(text="Hello. How are you? Fine!")
        assert "Sentences: 3" in result.content

    async def test_single_word(self) -> None:
        result = await text_length.execute(text="Hello")
        assert "Words: 1" in result.content

    async def test_has_schema(self) -> None:
        assert text_length.schema.name == "text_length"
        assert "text" in text_length.schema.parameters.get("properties", {})


class TestCurrentDatetime:
    async def test_utc_default(self) -> None:
        result = await current_datetime.execute()
        assert "T" in result.content  # ISO format
        assert "+" in result.content or "Z" in result.content

    async def test_specific_timezone(self) -> None:
        result = await current_datetime.execute(timezone_name="US/Eastern")
        assert "T" in result.content

    async def test_invalid_timezone(self) -> None:
        with pytest.raises(ValueError, match="Invalid timezone"):
            await current_datetime.execute(timezone_name="NotATimezone")

    async def test_has_schema(self) -> None:
        assert current_datetime.schema.name == "current_datetime"
