"""Runnable-snippet verifier for all guides with runnable Python.

This is a correctness gate wired into ``just check``.  It extracts every
fenced ``python`` code block from every guide under ``docs/guides/`` listed
in ``TARGET_GUIDES`` and compile-checks each one with the built-in
``compile()`` (AST parse; no execution).  Snippets that cannot compile-check
cleanly in isolation (live-service fixtures, illustrative stubs, ``...``
placeholders, intentional ``raise`` examples) must be annotated with

    <!-- verify: skip — <reason> -->

immediately before the fence in the Markdown source.  The reason is
required — a bare ``<!-- verify: skip -->`` or an empty reason fails the
gate.

Usage:

    uv run python docs/_verify_guide_snippets.py

Exit code 0 on success; non-zero if any un-skipped snippet fails to
compile, or if any skip annotation is missing a reason.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GUIDES_DIR = REPO_ROOT / "docs" / "guides"

# Every guide under docs/guides/ that contains at least one fenced `python`
# block. Guides that are prose-only or contain only non-Python fences
# (bash, json, ascii diagrams, etc.) are intentionally omitted — there is
# nothing for this validator to check in them.
TARGET_GUIDES = [
    "architecture-guide.md",
    "building-applications.md",
    "built-in-tools.md",
    "core-concepts.md",
    "getting-started.md",
    "human-in-the-loop.md",
    "memory.md",
    "migrating-from-working-memory-workaround.md",
    "multi-agent-coordination.md",
    "observability.md",
    "observatory-integration.md",
    "security.md",
    "streaming.md",
    "testing.md",
    "tools.md",
]

SKIP_MARKER = re.compile(r"<!--\s*verify:\s*skip(?P<tail>[^>]*)-->")
# Fences must start at the beginning of a line — otherwise blockquoted code
# samples (``> ```python``) would be captured and fail compile-check with
# `> ` prefixes embedded in every line.
FENCE = re.compile(
    r"(?P<preamble>(?:^<!--[^>]*-->\s*\n)*)^```(?P<lang>\w+)\n(?P<body>.*?)^```",
    re.DOTALL | re.MULTILINE,
)


@dataclass
class Snippet:
    guide: str
    index: int
    language: str
    body: str
    # None => no skip annotation (snippet must compile).
    # "" => skip annotation present but reason missing/empty (invalid; fails gate).
    # Non-empty => skip annotation with a real reason (snippet is skipped).
    skip_reason: str | None


def _extract_skip_reason(tail: str) -> str:
    """Return the reason text from the tail of a ``verify: skip`` marker.

    ``tail`` is whatever follows ``skip`` and precedes ``-->``. A real reason
    starts with an em-dash (—) or ASCII dash (-) followed by text containing
    at least one word character. Any other tail (empty, whitespace-only,
    dash with no text, punctuation-only) returns the empty string to signal
    an invalid annotation.
    """
    stripped = tail.strip()
    if not stripped:
        return ""
    # Must begin with an em-dash or ASCII dash separator.
    if stripped[0] not in {"—", "-"}:
        return ""
    reason = stripped[1:].strip()
    # A reason must contain at least one word character — bare punctuation
    # does not count as a real reason.
    if not re.search(r"\w", reason):
        return ""
    return reason


def extract_snippets(guide_path: Path) -> list[Snippet]:
    text = guide_path.read_text()
    snippets: list[Snippet] = []
    for idx, match in enumerate(FENCE.finditer(text)):
        preamble = match.group("preamble") or ""
        language = match.group("lang")
        body = match.group("body")
        skip_match = SKIP_MARKER.search(preamble)
        if skip_match is None:
            skip_reason: str | None = None
        else:
            skip_reason = _extract_skip_reason(skip_match.group("tail"))
        snippets.append(
            Snippet(
                guide=guide_path.name,
                index=idx,
                language=language,
                body=body,
                skip_reason=skip_reason,
            )
        )
    return snippets


def run_snippet(snippet: Snippet) -> tuple[bool, str]:
    try:
        compile(snippet.body, f"{snippet.guide}#{snippet.index}", "exec")
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc.msg} (line {exc.lineno})"
    return True, ""


def main() -> int:
    total = 0
    skipped = 0
    executed = 0
    passed = 0
    failures: list[tuple[Snippet, str]] = []
    bad_skips: list[Snippet] = []

    for guide_name in TARGET_GUIDES:
        guide_path = GUIDES_DIR / guide_name
        if not guide_path.exists():
            print(f"MISSING: {guide_path}")
            return 1
        for snippet in extract_snippets(guide_path):
            if snippet.language != "python":
                continue
            total += 1
            if snippet.skip_reason is not None:
                if snippet.skip_reason == "":
                    bad_skips.append(snippet)
                    print(
                        f"BAD-SKIP {snippet.guide}#{snippet.index}: "
                        f"skip annotation present but reason is missing or empty"
                    )
                    continue
                skipped += 1
                print(f"SKIP  {snippet.guide}#{snippet.index}: {snippet.skip_reason}")
                continue
            executed += 1
            ok, err = run_snippet(snippet)
            if ok:
                passed += 1
                print(f"PASS  {snippet.guide}#{snippet.index}")
            else:
                failures.append((snippet, err))
                print(f"FAIL  {snippet.guide}#{snippet.index}\n{err}")

    print(
        f"\nTotal python snippets: {total}  skipped: {skipped}  "
        f"executed: {executed}  passed: {passed}  failed: {len(failures)}  "
        f"bad-skips: {len(bad_skips)}"
    )
    return 0 if not failures and not bad_skips else 1


if __name__ == "__main__":
    sys.exit(main())
