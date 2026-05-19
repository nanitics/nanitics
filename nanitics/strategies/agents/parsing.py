from __future__ import annotations

import re

_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:python)?\s*\n(.*?)```",
    re.DOTALL,
)


def parse_code_blocks(content: str) -> tuple[str, list[str]]:
    """Extract fenced Python code blocks from LLM response text.

    Matches ```python\\n...\\n``` and ```\\n...\\n``` blocks.

    Returns:
        Tuple of (thought_text, code_blocks) where thought_text is everything
        outside code fences and code_blocks is the list of extracted code strings.
        If no code blocks are found, thought_text is the entire content.
    """
    code_blocks: list[str] = []
    thought_parts: list[str] = []
    last_end = 0

    for match in _CODE_BLOCK_PATTERN.finditer(content):
        thought_parts.append(content[last_end : match.start()])
        block = match.group(1).strip()
        if block:
            code_blocks.append(block)
        last_end = match.end()

    thought_parts.append(content[last_end:])
    thought_text = "\n".join(part.strip() for part in thought_parts if part.strip())

    return thought_text, code_blocks


def parse_working_memory_update(response_content: str) -> str | None:
    """Extract content between <working_memory> and </working_memory> tags.

    Returns the extracted content, or None if tags are absent.
    """
    match = re.search(
        r"<working_memory>\s*(.*?)\s*</working_memory>",
        response_content,
        re.DOTALL,
    )
    if match is None:
        return None
    content = match.group(1).strip()
    return content if content else None


def strip_working_memory_block(response_content: str) -> str:
    """Remove the <working_memory>...</working_memory> block from response content."""
    return re.sub(
        r"\s*<working_memory>.*?</working_memory>\s*",
        "",
        response_content,
        flags=re.DOTALL,
    ).strip()
