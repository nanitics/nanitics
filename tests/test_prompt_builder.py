"""Tests for SystemPromptBuilder."""

from nanitics.infrastructure.llm.protocol import SystemPromptSection
from nanitics.strategies.prompts import SystemPromptBuilder

# --- SystemPromptBuilder ---


class TestSystemPromptBuilder:
    def test_add_and_build(self) -> None:
        builder = SystemPromptBuilder()
        builder.add_section("role", "You are a helpful assistant.")
        builder.add_section("constraints", "Be concise.")
        result = builder.build()
        assert "You are a helpful assistant." in result
        assert "Be concise." in result

    def test_insertion_order(self) -> None:
        builder = SystemPromptBuilder()
        builder.add_section("first", "First section")
        builder.add_section("second", "Second section")
        result = builder.build()
        assert result.index("First section") < result.index("Second section")

    def test_replace_preserves_position(self) -> None:
        builder = SystemPromptBuilder()
        builder.add_section("a", "A")
        builder.add_section("b", "B")
        builder.add_section("c", "C")
        builder.add_section("a", "A-replaced")
        result = builder.build()
        # "a" was first, should still be before "b"
        assert result.index("A-replaced") < result.index("B")

    def test_remove_section(self) -> None:
        builder = SystemPromptBuilder()
        builder.add_section("keep", "Keep this")
        builder.add_section("remove", "Remove this")
        builder.remove_section("remove")
        result = builder.build()
        assert "Keep this" in result
        assert "Remove this" not in result

    def test_remove_nonexistent_is_noop(self) -> None:
        builder = SystemPromptBuilder()
        builder.remove_section("nope")  # no error

    def test_empty_sections_skipped(self) -> None:
        builder = SystemPromptBuilder()
        builder.add_section("a", "Content")
        builder.add_section("empty", "   ")
        builder.add_section("b", "More content")
        result = builder.build()
        assert result == "Content\n\nMore content"

    def test_chaining(self) -> None:
        result = SystemPromptBuilder().add_section("a", "A").add_section("b", "B").remove_section("a").build()
        assert result == "B"

    def test_has_section(self) -> None:
        builder = SystemPromptBuilder()
        builder.add_section("exists", "content")
        assert builder.has_section("exists") is True
        assert builder.has_section("missing") is False

    def test_strips_whitespace(self) -> None:
        builder = SystemPromptBuilder()
        builder.add_section("a", "  padded  ")
        assert builder.build() == "padded"

    def test_double_newline_separator(self) -> None:
        builder = SystemPromptBuilder()
        builder.add_section("a", "A")
        builder.add_section("b", "B")
        assert builder.build() == "A\n\nB"

    def test_build_sections_returns_structured_sections(self) -> None:
        builder = SystemPromptBuilder()
        builder.add_section("base", "Base prompt", cacheable=True)
        builder.add_section("state", "Dynamic state", cacheable=False)
        sections = builder.build_sections()
        assert len(sections) == 2
        assert sections[0] == SystemPromptSection(content="Base prompt", cacheable=True)
        assert sections[1] == SystemPromptSection(content="Dynamic state", cacheable=False)

    def test_build_sections_default_cacheable(self) -> None:
        builder = SystemPromptBuilder()
        builder.add_section("a", "Content")
        sections = builder.build_sections()
        assert len(sections) == 1
        assert sections[0].cacheable is True

    def test_build_sections_skips_empty(self) -> None:
        builder = SystemPromptBuilder()
        builder.add_section("a", "Content", cacheable=True)
        builder.add_section("empty", "   ", cacheable=False)
        builder.add_section("b", "More", cacheable=False)
        sections = builder.build_sections()
        assert len(sections) == 2
        assert sections[0].content == "Content"
        assert sections[1].content == "More"

    def test_build_and_build_sections_consistent(self) -> None:
        builder = SystemPromptBuilder()
        builder.add_section("base", "Base prompt", cacheable=True)
        builder.add_section("state", "Dynamic state", cacheable=False)
        flat = builder.build()
        sections = builder.build_sections()
        # Flat output should be the concatenation of section contents
        assert flat == "\n\n".join(s.content for s in sections)
