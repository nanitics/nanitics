"""Unit tests for `scripts/generate_llms_txt.py`.

The generator walks `nanitics.__all__` and `docs/guides/*.md` and emits an
`llms.txt` file conforming to the llms.txt spec. Tests exercise every branch:

- Happy-path output structure (title, tagline, API section, Guides section).
- Missing-docstring path (fallback line + stderr warning).
- Guide with/without intro line.
- `--output` flag default and override.
- I/O and import failure paths return non-zero exit codes with clear messages.

Real `nanitics` import is fine (deterministic — `__all__` is static); guide
walking is exercised against a temporary guide tree so we do not depend on
whichever guides happen to live under `docs/guides/` today.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "generate_llms_txt.py"


def _load_module() -> types.ModuleType:
    """Load the script as a module without relying on `scripts/` being a package."""
    spec = importlib.util.spec_from_file_location("generate_llms_txt", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mod() -> types.ModuleType:
    return _load_module()


@pytest.fixture
def fake_package(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a minimal fake `nanitics` module with a predictable `__all__`."""
    pkg = types.ModuleType("nanitics_fake_for_test")

    class Alpha:
        """First symbol.

        Longer body that should not appear in the single-line description.
        """

    def beta() -> None:
        """Second symbol."""

    class Gamma:
        # No docstring on purpose — exercises the "no description available" path.
        pass

    pkg.Alpha = Alpha  # type: ignore[attr-defined]
    pkg.beta = beta  # type: ignore[attr-defined]
    pkg.Gamma = Gamma  # type: ignore[attr-defined]
    pkg.__all__ = ["Alpha", "beta", "Gamma"]  # type: ignore[attr-defined]
    return pkg


@pytest.fixture
def guide_tree(tmp_path: Path) -> Path:
    """Build a temporary guide tree for the walker to consume."""
    guides = tmp_path / "docs" / "guides"
    guides.mkdir(parents=True)
    (guides / "getting-started.md").write_text(
        "# Getting Started\n\n> A blockquote pointer.\n\nIntro line for getting started.\n",
        encoding="utf-8",
    )
    (guides / "planning.md").write_text(
        "# Planning\n\nJust prose, no blockquote.\n",
        encoding="utf-8",
    )
    (guides / "empty-heading.md").write_text("# Empty\n", encoding="utf-8")
    (guides / "no-heading.md").write_text("Just a body, no heading at all.\n", encoding="utf-8")
    # README.md must be excluded.
    (guides / "README.md").write_text("# Guides\n\nIndex.\n", encoding="utf-8")
    return guides


def test_main_happy_path(
    mod: types.ModuleType,
    fake_package: types.ModuleType,
    guide_tree: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "out" / "llms.txt"
    exit_code = mod.main(
        argv=["--output", str(output)],
        package=fake_package,
        guides_dir=guide_tree,
        description="Test tagline.",
    )
    assert exit_code == 0
    text = output.read_text(encoding="utf-8")

    # Title and tagline.
    assert text.startswith("# Nanitics\n")
    assert "> Test tagline." in text

    # API section — symbols appear in sorted order.
    api_index = text.index("## API")
    guides_index = text.index("## Guides")
    api_block = text[api_index:guides_index]
    assert "- [Alpha]" in api_block
    assert "First symbol." in api_block
    # Long-form docstring lines must not leak into the one-liner.
    assert "Longer body" not in api_block
    assert "- [Gamma]" in api_block
    assert "No description available." in api_block
    # Symbols are sorted; Alpha < Gamma < beta (case-sensitive sort).
    assert api_block.index("Alpha") < api_block.index("Gamma") < api_block.index("beta")

    # Guides section.
    guides_block = text[guides_index:]
    assert "- [Getting Started]" in guides_block
    assert "Intro line for getting started." in guides_block
    assert "- [Planning]" in guides_block
    assert "Just prose, no blockquote." in guides_block
    assert "- [Empty]" in guides_block
    # Files without a top-level heading are skipped with a warning, not listed.
    assert "no-heading" not in guides_block
    # README excluded.
    assert "README" not in guides_block

    # stderr captures the missing-docstring warning.
    err = capsys.readouterr().err
    assert "Gamma" in err


def test_main_default_output_path(
    mod: types.ModuleType,
    fake_package: types.ModuleType,
    guide_tree: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running with no --output resolves to the default path."""
    monkeypatch.chdir(tmp_path)
    exit_code = mod.main(
        argv=[],
        package=fake_package,
        guides_dir=guide_tree,
        description="desc",
    )
    assert exit_code == 0
    assert (tmp_path / "build" / "docs" / "llms.txt").is_file()


def test_main_io_failure(
    mod: types.ModuleType,
    fake_package: types.ModuleType,
    guide_tree: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Point output at a path whose parent cannot be created (a file instead of a dir).
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    output = blocker / "llms.txt"
    exit_code = mod.main(
        argv=["--output", str(output)],
        package=fake_package,
        guides_dir=guide_tree,
        description="desc",
    )
    assert exit_code != 0
    err = capsys.readouterr().err
    assert "llms.txt" in err or "output" in err.lower()


def test_main_missing_guides_dir(
    mod: types.ModuleType,
    fake_package: types.ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = mod.main(
        argv=["--output", str(tmp_path / "out.txt")],
        package=fake_package,
        guides_dir=tmp_path / "nonexistent",
        description="desc",
    )
    assert exit_code != 0
    err = capsys.readouterr().err
    assert "guides" in err.lower()


def test_main_missing_description(
    mod: types.ModuleType,
    fake_package: types.ModuleType,
    guide_tree: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = mod.main(
        argv=["--output", str(tmp_path / "out.txt")],
        package=fake_package,
        guides_dir=guide_tree,
        description="",
    )
    assert exit_code != 0
    err = capsys.readouterr().err
    assert "description" in err.lower()


def test_cli_entrypoint_defaults_resolve_real_project(
    mod: types.ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`main()` with no arguments resolves `nanitics`, `docs/guides`, and `pyproject.toml`.

    Exercises the default-parameter branches (`package=None`, `guides_dir=None`,
    `description=None`) against the real repo. Also verifies that the auto-load
    path picks up every public Nanitics subpackage as a `###` subsection under
    `## API` with anchors that point at the corresponding pdoc page.
    """
    repo_root = _SCRIPT_PATH.parent.parent
    monkeypatch.chdir(repo_root)
    output = tmp_path / "llms.txt"
    exit_code = mod.main(argv=["--output", str(output)])
    assert exit_code == 0
    text = output.read_text(encoding="utf-8")
    assert text.startswith("# Nanitics\n")
    assert "## API" in text
    assert "## Guides" in text
    # Every public subpackage is rendered as a `###` subsection, and its
    # anchors live at the matching pdoc page — not at the top-level
    # `nanitics.html`.
    for subpackage in (
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
    ):
        assert f"### nanitics.{subpackage}" in text
    assert "nanitics/strategies.html#ReActAgent" in text
    assert "nanitics/patterns.html#create_orchestrator" in text
    assert "nanitics/specialized.html#ReWOOAgent" in text


def test_whitespace_only_docstring_falls_back_to_no_description(
    mod: types.ModuleType,
    guide_tree: Path,
    tmp_path: Path,
) -> None:
    """A symbol whose docstring contains only whitespace is treated as missing."""
    pkg = types.ModuleType("whitespace_doc_pkg")

    class Whitey:
        """ """

    pkg.Whitey = Whitey  # type: ignore[attr-defined]
    pkg.__all__ = ["Whitey"]  # type: ignore[attr-defined]

    output = tmp_path / "out.txt"
    exit_code = mod.main(
        argv=["--output", str(output)],
        package=pkg,
        guides_dir=guide_tree,
        description="desc",
    )
    assert exit_code == 0
    assert "No description available." in output.read_text(encoding="utf-8")


def test_guide_subheading_before_prose_yields_no_intro(
    mod: types.ModuleType,
    fake_package: types.ModuleType,
    tmp_path: Path,
) -> None:
    guides = tmp_path / "guides"
    guides.mkdir()
    (guides / "sub-before-prose.md").write_text(
        "# Heading\n\n## Subheading First\n\nBody.\n",
        encoding="utf-8",
    )
    output = tmp_path / "out.txt"
    exit_code = mod.main(
        argv=["--output", str(output)],
        package=fake_package,
        guides_dir=guides,
        description="desc",
    )
    assert exit_code == 0
    text = output.read_text(encoding="utf-8")
    # Title renders without an intro colon-separated tail.
    assert "- [Heading](" in text
    # The body of the file is not picked up as an intro for this guide.
    guide_line = next(line for line in text.splitlines() if line.startswith("- [Heading]"))
    assert ": Body." not in guide_line


def test_guide_blank_line_after_intro_breaks_loop(
    mod: types.ModuleType,
    fake_package: types.ModuleType,
    tmp_path: Path,
) -> None:
    """Cover the `if intro: break` branch after an intro is set and a blank follows.

    This branch is defensive — in practice the `intro = ...; break` path on the
    prose line already breaks out before a second blank line is seen. Hitting
    it requires crafting a file where intro is set before the blank-line loop
    iteration, which can occur if a future refactor moves the assignment.
    Covered here via direct call to the private helper.
    """
    guide = tmp_path / "g.md"
    guide.write_text("# Title\nintro-line\n\nnext\n", encoding="utf-8")
    title, intro = mod._guide_heading_and_intro(guide)
    assert title == "Title"
    assert intro == "intro-line"


def test_nanitics_import_failure_returns_error(
    mod: types.ModuleType,
    guide_tree: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Simulate `import nanitics` failing by removing it from sys.modules and
    replacing it with a finder that raises ImportError."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "nanitics":
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "nanitics", raising=False)

    exit_code = mod.main(
        argv=["--output", str(tmp_path / "out.txt")],
        package=None,
        guides_dir=guide_tree,
        description="desc",
    )
    assert exit_code != 0
    err = capsys.readouterr().err
    assert "nanitics" in err


def test_pyproject_read_failure_returns_error(
    mod: types.ModuleType,
    fake_package: types.ModuleType,
    guide_tree: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """If `pyproject.toml` cannot be read, main returns non-zero with a clear error."""
    monkeypatch.chdir(tmp_path)  # No pyproject.toml here.
    exit_code = mod.main(
        argv=["--output", str(tmp_path / "out.txt")],
        package=fake_package,
        guides_dir=guide_tree,
        description=None,
    )
    assert exit_code != 0
    err = capsys.readouterr().err
    assert "pyproject" in err.lower()


def test_walk_attribute_docstrings_picks_up_type_aliases(
    mod: types.ModuleType,
    tmp_path: Path,
) -> None:
    """AST walker finds attribute docstrings on module-level assignments.

    Covers: ``Assign`` with ``Name`` target, ``AnnAssign`` with ``Name`` target,
    first-occurrence-wins, empty docstring skipped, non-``Name`` targets ignored,
    non-string next sibling ignored, assignment at file end ignored, syntax
    errors swallowed.
    """
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    # Primary module: TypeAlias + attribute docstring, plain assign + docstring.
    (pkg_dir / "__init__.py").write_text(
        '"""Module docstring."""\n'
        "from typing import TypeAlias\n"
        "\n"
        "AX: TypeAlias = int\n"
        '"""AX description."""\n'
        "\n"
        "AY = str\n"
        '"""AY description."""\n'
        "\n"
        "class Other:\n"
        "    attr = 1\n"
        '    """attribute of class, not module-level."""\n'
        "\n"
        "NOT_DOC_BELOW = 1\n"
        "123\n"  # next sibling is a non-string Expr
        "\n"
        "EMPTY_DOC = 1\n"
        '"""   """\n'  # whitespace-only → skipped
        "\n"
        "LAST = 1\n",  # at file end — no next sibling
        encoding="utf-8",
    )
    # Second module sorted later; must NOT override AX (setdefault).
    (pkg_dir / "z_later.py").write_text(
        'AX = float\n"""AX reassigned — should not win."""\n',
        encoding="utf-8",
    )
    # Syntax error — must be swallowed, other files still processed.
    (pkg_dir / "broken.py").write_text("def (\n", encoding="utf-8")

    pkg = types.ModuleType("pkg_for_walker")
    pkg.__file__ = str(pkg_dir / "__init__.py")

    mapping = mod._walk_attribute_docstrings(pkg)
    assert mapping == {"AX": "AX description.", "AY": "AY description."}


def test_walk_attribute_docstrings_returns_empty_without_file(
    mod: types.ModuleType,
) -> None:
    """A package lacking ``__file__`` (in-memory fake) yields no attribute docstrings."""
    pkg = types.ModuleType("pkg_no_file")
    # No __file__ set — types.ModuleType doesn't get one by default here.
    assert mod._walk_attribute_docstrings(pkg) == {}


def test_walk_attribute_docstrings_skips_files_that_cannot_be_read(
    mod: types.ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``OSError`` from ``Path.read_text`` is swallowed — other files still contribute."""
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    (pkg_dir / "good.py").write_text(
        'GOOD = 1\n"""Good docstring."""\n',
        encoding="utf-8",
    )
    bad_path = pkg_dir / "bad.py"
    bad_path.write_text(
        'BAD = 1\n"""Bad docstring."""\n',
        encoding="utf-8",
    )

    real_read_text = Path.read_text

    def fake_read_text(self: Path, *args: object, **kwargs: object) -> str:
        if self == bad_path:
            raise OSError("simulated read failure")
        return real_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    pkg = types.ModuleType("pkg_read_fail")
    pkg.__file__ = str(pkg_dir / "__init__.py")
    mapping = mod._walk_attribute_docstrings(pkg)
    assert mapping == {"GOOD": "Good docstring."}


def test_main_uses_attribute_docstring_fallback(
    mod: types.ModuleType,
    guide_tree: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When a symbol has no runtime docstring, the AST attribute-docstring fallback supplies it
    so the entry renders cleanly in ``llms.txt`` and no warning is emitted."""
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text(
        "from typing import Literal, TypeAlias\n"
        "\n"
        'MyAlias: TypeAlias = Literal["a", "b"]\n'
        '"""My alias description."""\n',
        encoding="utf-8",
    )

    pkg = types.ModuleType("pkg_for_fallback")
    pkg.__file__ = str(pkg_dir / "__init__.py")
    # Load the runtime value so getattr works.
    import importlib.util

    spec = importlib.util.spec_from_file_location("pkg_for_fallback", pkg_dir / "__init__.py")
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    pkg.MyAlias = loaded.MyAlias  # type: ignore[attr-defined]
    pkg.__all__ = ["MyAlias"]  # type: ignore[attr-defined]

    output = tmp_path / "out.txt"
    exit_code = mod.main(
        argv=["--output", str(output)],
        package=pkg,
        guides_dir=guide_tree,
        description="desc",
    )
    assert exit_code == 0
    text = output.read_text(encoding="utf-8")
    assert "- [MyAlias](" in text
    assert "My alias description." in text
    assert "No description available." not in text
    err = capsys.readouterr().err
    assert "MyAlias" not in err


def test_script_entrypoint_invokes_main(
    mod: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Executing the script as `__main__` must call `main()` and exit with its code."""
    repo_root = _SCRIPT_PATH.parent.parent
    monkeypatch.chdir(repo_root)
    monkeypatch.setattr(sys, "argv", ["generate_llms_txt.py", "--output", str(tmp_path / "out.txt")])

    spec = importlib.util.spec_from_file_location("__main__", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(SystemExit) as excinfo:
        spec.loader.exec_module(module)
    assert excinfo.value.code == 0
    assert (tmp_path / "out.txt").is_file()
