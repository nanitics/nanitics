"""Tests for parse_code_blocks in parsing.py."""

from nanitics.strategies.agents.parsing import parse_code_blocks


class TestParseCodeBlocks:
    def test_single_python_block(self) -> None:
        content = "Let me calculate this.\n```python\nprint(1 + 1)\n```\nDone."
        thought, blocks = parse_code_blocks(content)
        assert blocks == ["print(1 + 1)"]
        assert "Let me calculate this." in thought
        assert "Done." in thought

    def test_single_unfenced_block(self) -> None:
        content = "Here is the code:\n```\nx = 42\nprint(x)\n```"
        thought, blocks = parse_code_blocks(content)
        assert blocks == ["x = 42\nprint(x)"]
        assert "Here is the code:" in thought

    def test_multiple_blocks(self) -> None:
        content = "First step:\n```python\nx = 1\n```\nSecond step:\n```python\ny = 2\n```\nAll done."
        thought, blocks = parse_code_blocks(content)
        assert blocks == ["x = 1", "y = 2"]
        assert "First step:" in thought
        assert "Second step:" in thought
        assert "All done." in thought

    def test_no_code_blocks(self) -> None:
        content = "The answer is 42."
        thought, blocks = parse_code_blocks(content)
        assert blocks == []
        assert thought == "The answer is 42."

    def test_empty_code_block_skipped(self) -> None:
        content = "Empty block:\n```python\n\n```\nAfter."
        _thought, blocks = parse_code_blocks(content)
        assert blocks == []

    def test_multiline_code(self) -> None:
        code = "def greet(name):\n    return f'Hello {name}'\n\nresult = greet('world')\nprint(result)"
        content = f"Let me write a function:\n```python\n{code}\n```"
        _thought, blocks = parse_code_blocks(content)
        assert blocks == [code]

    def test_thought_only_when_no_blocks(self) -> None:
        content = "I have analyzed the data and the answer is 42.\nNo code needed."
        thought, blocks = parse_code_blocks(content)
        assert thought == content
        assert blocks == []

    def test_mixed_fenced_and_unfenced(self) -> None:
        content = "Step 1:\n```python\na = 1\n```\nStep 2:\n```\nb = 2\n```"
        _thought, blocks = parse_code_blocks(content)
        assert blocks == ["a = 1", "b = 2"]

    def test_code_with_triple_backticks_in_string(self) -> None:
        """Code containing string literals with backtick-like content."""
        content = '```python\nx = "some value"\nprint(x)\n```'
        _thought, blocks = parse_code_blocks(content)
        assert blocks == ['x = "some value"\nprint(x)']

    def test_empty_input(self) -> None:
        thought, blocks = parse_code_blocks("")
        assert thought == ""
        assert blocks == []

    def test_only_code_no_thought(self) -> None:
        content = "```python\nprint('hello')\n```"
        thought, blocks = parse_code_blocks(content)
        assert blocks == ["print('hello')"]
        assert thought == ""

    def test_whitespace_around_code(self) -> None:
        content = "```python\n  \n  x = 1\n  \n```"
        _thought, blocks = parse_code_blocks(content)
        assert len(blocks) == 1
        assert "x = 1" in blocks[0]
