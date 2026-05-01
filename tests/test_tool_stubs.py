"""Tests for tool stub generation in codeact.py."""

from __future__ import annotations

from nanitics.core.agents.codeact import (
    generate_tool_documentation,
    generate_tool_stubs,
)
from nanitics.infrastructure.llm.protocol import ToolSchema


def _make_schema(
    name: str,
    description: str,
    properties: dict,
    required: list[str] | None = None,
) -> ToolSchema:
    return ToolSchema(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": properties,
            "required": required or [],
        },
    )


class TestGenerateToolStubs:
    def test_single_tool_basic(self) -> None:
        schema = _make_schema(
            "search",
            "Search the web.",
            {"query": {"type": "string", "description": "Search query"}},
            ["query"],
        )
        code = generate_tool_stubs([schema])
        assert "def search(query: str) -> str:" in code
        assert '__call_tool__("search"' in code
        # Must be valid Python
        compile(code, "<test>", "exec")

    def test_multiple_tools(self) -> None:
        schemas = [
            _make_schema(
                "read_file",
                "Read a file.",
                {"path": {"type": "string", "description": "File path"}},
                ["path"],
            ),
            _make_schema(
                "write_file",
                "Write a file.",
                {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "File content"},
                },
                ["path", "content"],
            ),
        ]
        code = generate_tool_stubs(schemas)
        assert "def read_file(" in code
        assert "def write_file(" in code
        compile(code, "<test>", "exec")

    def test_optional_parameters(self) -> None:
        schema = _make_schema(
            "search",
            "Search.",
            {
                "query": {"type": "string", "description": "Query"},
                "limit": {"type": "integer", "description": "Max results", "default": 10},
            },
            ["query"],
        )
        code = generate_tool_stubs([schema])
        assert "query: str" in code
        assert "limit: int = 10" in code
        compile(code, "<test>", "exec")

    def test_various_types(self) -> None:
        schema = _make_schema(
            "process",
            "Process data.",
            {
                "name": {"type": "string"},
                "count": {"type": "integer"},
                "ratio": {"type": "number"},
                "enabled": {"type": "boolean"},
                "items": {"type": "array"},
                "config": {"type": "object"},
            },
            ["name", "count", "ratio", "enabled", "items", "config"],
        )
        code = generate_tool_stubs([schema])
        assert "name: str" in code
        assert "count: int" in code
        assert "ratio: float" in code
        assert "enabled: bool" in code
        assert "items: list" in code
        assert "config: dict" in code
        compile(code, "<test>", "exec")

    def test_no_parameters(self) -> None:
        schema = _make_schema("get_time", "Get the current time.", {})
        code = generate_tool_stubs([schema])
        assert "def get_time() -> str:" in code
        compile(code, "<test>", "exec")

    def test_empty_schemas_list(self) -> None:
        code = generate_tool_stubs([])
        assert code == ""

    def test_stub_calls_call_tool(self) -> None:
        schema = _make_schema(
            "greet",
            "Greet someone.",
            {"name": {"type": "string"}},
            ["name"],
        )
        code = generate_tool_stubs([schema])
        # Stub should call __call_tool__ with tool name and args dict
        assert '__call_tool__("greet", {"name": name})' in code

    def test_stub_executes_with_mock_call_tool(self) -> None:
        """Generated stubs are executable when __call_tool__ is defined."""
        schema = _make_schema(
            "add",
            "Add two numbers.",
            {
                "a": {"type": "integer", "description": "First number"},
                "b": {"type": "integer", "description": "Second number"},
            },
            ["a", "b"],
        )
        code = generate_tool_stubs([schema])

        namespace: dict = {}
        namespace["__call_tool__"] = lambda name, args: f"result:{name}:{args}"
        exec(code, namespace)

        result = namespace["add"](a=1, b=2)
        assert result == "result:add:{'a': 1, 'b': 2}"


class TestGenerateToolDocumentation:
    def test_basic_documentation(self) -> None:
        schema = _make_schema(
            "search",
            "Search for information.",
            {"query": {"type": "string", "description": "The search query"}},
            ["query"],
        )
        doc = generate_tool_documentation([schema])
        assert "## Available Functions" in doc
        assert "def search(query: str) -> str:" in doc
        assert "Search for information." in doc
        assert "query: The search query" in doc

    def test_empty_schemas(self) -> None:
        doc = generate_tool_documentation([])
        assert doc == ""

    def test_multiple_tools_documented(self) -> None:
        schemas = [
            _make_schema(
                "read",
                "Read a file.",
                {"path": {"type": "string", "description": "File path"}},
                ["path"],
            ),
            _make_schema(
                "write",
                "Write a file.",
                {"path": {"type": "string"}, "content": {"type": "string"}},
                ["path", "content"],
            ),
        ]
        doc = generate_tool_documentation(schemas)
        assert "def read(" in doc
        assert "def write(" in doc


class TestNullableTypes:
    def test_nullable_type_generates_optional(self) -> None:
        """JSON type ["string", "null"] maps to str | None."""
        schema = _make_schema(
            "search",
            "Search.",
            {"query": {"type": ["string", "null"], "description": "Query"}},
            ["query"],
        )
        code = generate_tool_stubs([schema])
        assert "query: str | None" in code
        compile(code, "<test>", "exec")

    def test_union_type_generates_pipe(self) -> None:
        """JSON type ["string", "integer"] maps to str | int."""
        schema = _make_schema(
            "process",
            "Process.",
            {"value": {"type": ["string", "integer"], "description": "Value"}},
            ["value"],
        )
        code = generate_tool_stubs([schema])
        assert "value: str | int" in code
        compile(code, "<test>", "exec")

    def test_single_non_null_in_list(self) -> None:
        """JSON type ["boolean"] maps to just bool."""
        schema = _make_schema(
            "toggle",
            "Toggle.",
            {"flag": {"type": ["boolean"], "description": "Flag"}},
            ["flag"],
        )
        code = generate_tool_stubs([schema])
        assert "flag: bool" in code
        compile(code, "<test>", "exec")

    def test_nullable_documentation(self) -> None:
        """Optional parameters appear in documentation with defaults."""
        schema = _make_schema(
            "search",
            "Search.",
            {
                "query": {"type": "string", "description": "Query"},
                "limit": {"type": "integer", "description": "Max results", "default": 10},
            },
            ["query"],
        )
        doc = generate_tool_documentation([schema])
        assert "limit: int = 10" in doc
        assert "Max results" in doc
