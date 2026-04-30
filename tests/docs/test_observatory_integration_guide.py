"""Mechanical-drift tests for docs/guides/observatory-integration.md.

- Relative links and anchors resolve to existing files and headings.
- The docker-compose file in docker/observatory-dev/ is syntactically
  valid (skipped if ``docker`` is not on ``PATH``).
- Python code fences import cleanly against the installed SDK —
  snippets annotated with ``<!-- verify: skip — <reason> -->`` are
  skipped.

Test-only file; no impact on ``nanitics/`` line coverage.
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GUIDE_PATH = REPO_ROOT / "docs" / "guides" / "observatory-integration.md"
COMPOSE_PATH = REPO_ROOT / "docker" / "observatory-dev" / "docker-compose.yml"

SKIP_MARKER = re.compile(r"<!--\s*verify:\s*skip(?:\s*[—-]\s*(?P<reason>[^>]*?))?\s*-->")
# Greedy-enough fence parser; tolerates preceding HTML comments so skip
# markers attach to the following fence.
FENCE = re.compile(
    r"(?P<preamble>(?:<!--[^>]*-->\s*\n)*)```(?P<lang>[\w-]+)\n(?P<body>.*?)```",
    re.DOTALL,
)
# ``[text](target)`` — captures inline Markdown links.  Excludes images
# (``![alt](src)``) via a negative look-behind.
LINK = re.compile(r"(?<!\!)\[(?P<text>[^\]]+)\]\((?P<target>[^)]+)\)")


def _slugify(heading: str) -> str:
    """Produce a GitHub-style anchor slug for a heading."""
    slug = heading.strip().lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    return re.sub(r"\s+", "-", slug)


def _headings(md_text: str) -> set[str]:
    """Return the set of anchor slugs for every heading in ``md_text``."""
    slugs: set[str] = set()
    for match in re.finditer(r"^(#{1,6})\s+(.+?)\s*$", md_text, re.MULTILINE):
        slugs.add(_slugify(match.group(2)))
    return slugs


@pytest.fixture(scope="module")
def guide_text() -> str:
    return GUIDE_PATH.read_text()


def test_relative_links_resolve(guide_text: str) -> None:
    """Every relative link's file target exists; anchors resolve to headings."""
    errors: list[str] = []
    for match in LINK.finditer(guide_text):
        target = match.group("target").strip()
        # Skip absolute URLs and purely in-page anchors.
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        if target.startswith("#"):
            slug = target[1:]
            if slug and slug not in _headings(guide_text):
                errors.append(f"in-page anchor missing: #{slug}")
            continue

        file_part, _, anchor = target.partition("#")
        file_path = (GUIDE_PATH.parent / file_part).resolve()
        if not file_path.exists():
            errors.append(f"broken file link: {target} -> {file_path}")
            continue
        if anchor and file_path.suffix == ".md":
            headings = _headings(file_path.read_text())
            if anchor not in headings:
                errors.append(f"broken anchor: {target} (file exists, slug {anchor!r} not found)")
    assert not errors, "Broken links in guide:\n" + "\n".join(errors)


@pytest.mark.skipif(
    shutil.which("docker") is None,
    reason="docker not on PATH; runtime verification deferred to [human-verify: docker]",
)
def test_compose_config_is_valid() -> None:
    """``docker compose config`` succeeds on the new compose file."""
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(COMPOSE_PATH),
            "config",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"docker compose config failed: stderr={result.stderr}"


def _extract_python_snippets(md_text: str) -> list[tuple[int, str, str | None]]:
    """Return ``(index, body, skip_reason)`` for every python fence."""
    snippets: list[tuple[int, str, str | None]] = []
    idx = 0
    for match in FENCE.finditer(md_text):
        if match.group("lang") != "python":
            continue
        preamble = match.group("preamble") or ""
        skip_match = SKIP_MARKER.search(preamble)
        skip_reason = (
            skip_match.group("reason").strip()
            if skip_match and skip_match.group("reason")
            else ("<unspecified>" if skip_match else None)
        )
        snippets.append((idx, match.group("body"), skip_reason))
        idx += 1
    return snippets


def _extract_imports(source: str) -> list[ast.stmt]:
    """Return every ``import`` / ``from ... import ...`` statement at module level."""
    tree = ast.parse(source)
    return [node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom)]


def test_python_snippets_imports_resolve(guide_text: str) -> None:
    """Every non-skipped python snippet's imports resolve against the SDK."""
    snippets = _extract_python_snippets(guide_text)
    assert snippets, "expected at least one python snippet in the guide"

    errors: list[str] = []
    for idx, body, skip_reason in snippets:
        if skip_reason is not None:
            continue
        try:
            import_nodes = _extract_imports(body)
        except SyntaxError as exc:
            errors.append(f"snippet #{idx} is not valid Python: {exc}")
            continue
        # Build a minimal import-only script so we catch symbol-level drift
        # (e.g. removed re-exports) without executing the rest of the snippet.
        import_source = ast.unparse(ast.Module(body=import_nodes, type_ignores=[]))
        try:
            exec(import_source, {"__name__": "__snippet__"})
        except Exception as exc:  # pragma: no cover - failure path is the test assertion
            errors.append(f"snippet #{idx} imports failed: {type(exc).__name__}: {exc}\nimports:\n{import_source}")
    assert not errors, "Import drift in guide snippets:\n" + "\n".join(errors)
