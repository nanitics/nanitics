"""Generate `llms.txt` for the hosted API reference.

Walks the `__all__` of every public Nanitics subpackage plus
`docs/guides/*.md` and emits an `llms.txt` file conforming to the
llms.txt spec (https://llmstxt.org/). Invoked by `just docs` after pdoc
runs and by the `docs.yml` GitHub Actions workflow.

One source of truth: each package's own `__all__` and the existing guide
tree. No hand-maintained listing — when an `__all__` changes or a guide
is added, the next `just docs` run regenerates `llms.txt` to match.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

_DEFAULT_OUTPUT = Path("build/docs/llms.txt")
_DEFAULT_GUIDES_DIR = Path("docs/guides")
_DEFAULT_PYPROJECT = Path("pyproject.toml")

_HOSTED_BASE_URL = "https://docs.nanitics.dev"
_GUIDES_GITHUB_BASE = "https://github.com/nanitics/nanitics/blob/main/docs/guides"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate llms.txt for the hosted Nanitics API reference.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Output path. Default: {_DEFAULT_OUTPUT}.",
    )
    return parser.parse_args(argv)


def _first_docstring_line(obj: Any) -> str | None:
    """Return the first non-blank line of `obj`'s docstring, or None if absent.

    `inspect.getdoc` returns a whitespace-stripped string (empty when the
    docstring is blank), so a single `splitlines()[0]` check is sufficient.
    """
    doc = inspect.getdoc(obj)
    if not doc:
        return None
    return doc.splitlines()[0].strip()


def _walk_attribute_docstrings(package: ModuleType) -> dict[str, str]:
    """Return ``{name: first-line-of-attribute-docstring}`` for every module-level
    assignment in every ``.py`` under *package*.

    Attribute docstrings (a string literal immediately following an assignment)
    are not stored on the runtime object, so ``inspect.getdoc`` can't retrieve
    them. They are the idiomatic way to document module-level type aliases
    (``TypeAlias``, ``Literal``, ``Callable``-based aliases), which is why this
    walker exists. First occurrence wins; later re-definitions do not override.
    """
    mapping: dict[str, str] = {}
    pkg_file = getattr(package, "__file__", None)
    if pkg_file is None:
        return mapping
    root = Path(pkg_file).parent
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        body = tree.body
        for i, node in enumerate(body):
            name: str | None = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                name = node.target.id
            if name is None or i + 1 >= len(body):
                continue
            nxt = body[i + 1]
            if not isinstance(nxt, ast.Expr) or not isinstance(nxt.value, ast.Constant):
                continue
            raw = nxt.value.value
            if not isinstance(raw, str):
                continue
            stripped = raw.strip()
            if not stripped:
                continue
            mapping.setdefault(name, stripped.splitlines()[0].strip())
    return mapping


def _render_api_section(
    package: ModuleType,
    warnings: list[str],
    *,
    html_path: str = "nanitics.html",
    section_header: str | None = None,
) -> list[str]:
    """Render an API block. Missing docstrings land in `warnings`.

    Args:
        package: Module whose ``__all__`` to walk.
        warnings: Accumulator for symbols with no docstring.
        html_path: Path on the hosted docs site where the symbols are rendered
            (e.g. ``"nanitics.html"`` for core, ``"nanitics/patterns.html"``
            for ``nanitics.patterns``). Used to construct each symbol's anchor URL.
        section_header: Optional subsection name. When ``None`` (the default),
            the block opens with ``## API``. When provided, the block opens
            with ``### {section_header}`` — used to mark each
            ``nanitics.<subpackage>`` subsection under the single
            top-level ``## API`` heading.
    """
    heading = "## API" if section_header is None else f"### {section_header}"
    lines = [heading, ""]
    attribute_docstrings = _walk_attribute_docstrings(package)
    symbols = sorted(getattr(package, "__all__", []))
    for name in symbols:
        symbol = getattr(package, name, None)
        description = _first_docstring_line(symbol) if symbol is not None else None
        if description is None:
            description = attribute_docstrings.get(name)
        if description is None:
            warnings.append(name)
            description = "No description available."
        anchor = f"{_HOSTED_BASE_URL}/{html_path}#{name}"
        lines.append(f"- [{name}]({anchor}): {description}")
    lines.append("")
    return lines


def _guide_heading_and_intro(path: Path) -> tuple[str | None, str]:
    """Extract the H1 heading and the first prose paragraph after it.

    Lines starting with `>` (blockquote pointer) are skipped. Returns
    `(None, "")` when the file has no top-level heading.
    """
    title: str | None = None
    intro = ""
    past_heading = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not past_heading:
            if line.startswith("# "):
                title = line[2:].strip()
                past_heading = True
            continue
        if not line or line.startswith(">"):
            continue
        if line.startswith("#"):
            # Reached a subheading before any prose — no intro line.
            break
        intro = line.strip()
        break
    return title, intro


def _render_guides_section(guides_dir: Path, warnings: list[str]) -> list[str]:
    lines = ["## Guides", ""]
    for path in sorted(guides_dir.glob("*.md")):
        if path.name == "README.md":
            continue
        title, intro = _guide_heading_and_intro(path)
        if title is None:
            warnings.append(f"guide without heading: {path.name}")
            continue
        url = f"{_GUIDES_GITHUB_BASE}/{path.name}"
        if intro:
            lines.append(f"- [{title}]({url}): {intro}")
        else:
            lines.append(f"- [{title}]({url})")
    lines.append("")
    return lines


def _load_description_from_pyproject(pyproject: Path) -> str:
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return str(data.get("project", {}).get("description", ""))


def main(
    argv: Sequence[str] | None = None,
    *,
    package: ModuleType | None = None,
    extra_packages: Sequence[tuple[ModuleType, str, str]] | None = None,
    guides_dir: Path | None = None,
    description: str | None = None,
) -> int:
    """Generate `llms.txt`. Returns an exit code.

    Parameters are injectable so the unit tests can exercise the walker
    against a deterministic fixture tree without importing the real
    `nanitics` package or touching the real guides.

    Args:
        argv: CLI arguments.
        package: Top-level package whose ``__all__`` is walked into the
            ``## API`` block. When ``None``, the CLI auto-loads ``nanitics``
            and seeds ``extra_packages`` with every public Nanitics
            subpackage (when not already provided).
        extra_packages: Additional packages to render as ``### {header}``
            subsections under the same ``## API`` heading. Each entry is
            ``(package, html_path, section_header)``. Tests that inject a
            ``package`` argument can leave this ``None`` to keep the
            single-section output shape.
        guides_dir: Path to ``docs/guides``.
        description: One-line project tagline (typically loaded from
            ``pyproject.toml``).
    """
    args = _parse_args(argv)

    if package is None:
        try:
            import importlib

            import nanitics as _nanitics

            package = _nanitics
            if extra_packages is None:
                # Mirrors the public-subpackage list in
                # ``docs/deprecation-policy.md`` and the ``just docs`` recipe.
                # Order matches the docstring of ``nanitics/__init__.py`` —
                # primitives first, then the curated-composition namespaces.
                _subpackage_names = (
                    "strategies",
                    "memory",
                    "composition",
                    "tracing",
                    "errors",
                    "hitl",
                    "evaluation",
                    "planning",
                    "context",
                    "safety",
                    "tools",
                    "infrastructure",
                    "patterns",
                    "specialized",
                )
                extra_packages = [
                    (
                        importlib.import_module(f"nanitics.{name}"),
                        f"nanitics/{name}.html",
                        f"nanitics.{name}",
                    )
                    for name in _subpackage_names
                ]
        except ImportError as exc:
            print(f"error: cannot import `nanitics`: {exc}", file=sys.stderr)
            return 2

    if guides_dir is None:
        guides_dir = _DEFAULT_GUIDES_DIR
    if not guides_dir.is_dir():
        print(f"error: guides directory not found: {guides_dir}", file=sys.stderr)
        return 2

    if description is None:
        try:
            description = _load_description_from_pyproject(_DEFAULT_PYPROJECT)
        except OSError as exc:
            print(f"error: cannot read pyproject.toml: {exc}", file=sys.stderr)
            return 2
    if not description:
        print("error: project description is empty in pyproject.toml", file=sys.stderr)
        return 2

    warnings: list[str] = []
    lines: list[str] = ["# Nanitics", "", f"> {description}", ""]
    lines.extend(_render_api_section(package, warnings))
    for pkg, html_path, section_header in extra_packages or []:
        lines.extend(
            _render_api_section(
                pkg,
                warnings,
                html_path=html_path,
                section_header=section_header,
            )
        )
    lines.extend(_render_guides_section(guides_dir, warnings))

    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    output: Path = args.output
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines), encoding="utf-8")
    except OSError as exc:
        print(f"error: cannot write llms.txt to {output}: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
