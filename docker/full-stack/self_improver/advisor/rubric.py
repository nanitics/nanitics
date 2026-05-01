"""Rubric model, loader, and frontmatter parser.

The loader enumerates the builtin corpus at ``self_improver/advisor/rubrics/``
and merges in adopter-supplied paths. Each loaded rubric carries a
:class:`self_improver.advisor.RubricSource` label so downstream consumers
(and the output layer) can attribute proposals correctly.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from self_improver.advisor.proposal import ProposalCategory, ProposalSeverity, RubricSource

_BUILTIN_RUBRICS_DIR = Path(__file__).parent / "rubrics"
_REQUIRED_FRONTMATTER_KEYS = ("id", "severity", "category", "target_dimension")
_FRONTMATTER_DELIMITER = "---"
_VALID_TARGET_DIMENSIONS = frozenset(
    {
        "prompts",
        "tool_descriptions",
        "coordination_patterns",
        "agent_strategy",
        "iteration_budgets",
    }
)


class MalformedRubricError(ValueError):
    """A rubric file is missing frontmatter, missing required keys, or unparseable."""

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Malformed rubric at {path}: {reason}")


class RubricFileNameMismatchError(ValueError):
    """A rubric file's filename does not match its frontmatter ``id``."""

    def __init__(self, path: Path, rubric_id: str) -> None:
        self.path = path
        self.rubric_id = rubric_id
        super().__init__(f"Rubric filename '{path.name}' does not match frontmatter id '{rubric_id}' at {path}")


class DuplicateRubricError(ValueError):
    """Two rubrics share the same ``id``."""

    def __init__(self, paths: tuple[Path, Path], rubric_id: str) -> None:
        self.paths = paths
        self.rubric_id = rubric_id
        first, second = paths
        super().__init__(f"Duplicate rubric id '{rubric_id}' found at {first} and {second}")


class Rubric(BaseModel):
    """A single classification / proposal-authoring criterion.

    Attributes:
        id: Globally unique kebab-case identifier.
        severity: Severity bucket shared with :class:`ProposalSeverity`.
        category: Taxonomy category shared with :class:`ProposalCategory`.
        target_dimension: Which specialist owns this rubric (e.g., ``prompts``,
            ``tool_descriptions``, ``coordination_patterns``).
        body: Markdown body passed verbatim to specialist agents as context.
        source: ``builtin`` or ``custom`` — the loader sets this based on
            which path the file came from.
        path: On-disk path, retained for error reporting and deduplication.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    severity: ProposalSeverity
    category: ProposalCategory
    target_dimension: str
    body: str
    source: RubricSource
    path: Path


def _parse_rubric_file(path: Path, source: RubricSource) -> Rubric:
    """Parse a single rubric file into a :class:`Rubric`."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MalformedRubricError(path, f"cannot read file: {exc}") from exc

    frontmatter, body = _split_frontmatter(path, text)
    try:
        data = yaml.safe_load(frontmatter)
    except yaml.YAMLError as exc:
        raise MalformedRubricError(path, f"unparseable YAML frontmatter: {exc}") from exc

    if not isinstance(data, dict):
        raise MalformedRubricError(path, f"frontmatter must be a YAML mapping (got {type(data).__name__!r})")

    for key in _REQUIRED_FRONTMATTER_KEYS:
        if key not in data:
            raise MalformedRubricError(path, f"missing required frontmatter key '{key}'")

    rubric_id = data["id"]
    if not isinstance(rubric_id, str) or not rubric_id:
        raise MalformedRubricError(path, "frontmatter 'id' must be a non-empty string")
    if path.stem != rubric_id:
        raise RubricFileNameMismatchError(path, rubric_id)

    target_dimension = data["target_dimension"]
    if target_dimension not in _VALID_TARGET_DIMENSIONS:
        raise MalformedRubricError(
            path,
            f"frontmatter 'target_dimension' must be one of "
            f"{sorted(_VALID_TARGET_DIMENSIONS)}; got '{target_dimension}'",
        )

    try:
        return Rubric(
            id=rubric_id,
            severity=data["severity"],
            category=data["category"],
            target_dimension=target_dimension,
            body=body,
            source=source,
            path=path,
        )
    except ValidationError as exc:
        raise MalformedRubricError(path, f"invalid frontmatter values: {exc}") from exc


def _split_frontmatter(path: Path, text: str) -> tuple[str, str]:
    """Split ``text`` into (frontmatter_yaml, body).

    Expected layout:

        ---
        key: value
        ---

        Body markdown...

    Leading whitespace in the body is stripped.
    """
    stripped = text.lstrip()
    if not stripped.startswith(_FRONTMATTER_DELIMITER):
        raise MalformedRubricError(path, "missing YAML frontmatter")

    # Remove the leading delimiter line.
    after_open = stripped[len(_FRONTMATTER_DELIMITER) :].lstrip("\r\n")
    closing_idx = after_open.find(f"\n{_FRONTMATTER_DELIMITER}")
    if closing_idx == -1:
        raise MalformedRubricError(path, "unterminated YAML frontmatter (no closing '---')")

    frontmatter = after_open[:closing_idx]
    body = after_open[closing_idx + len(_FRONTMATTER_DELIMITER) + 1 :].lstrip()
    return frontmatter, body


def _enumerate_markdown_files(target: Path) -> list[Path]:
    """Return the sorted list of ``*.md`` files contributed by ``target``.

    - Files are returned as-is.
    - Directories are scanned non-recursively and ``README.md`` is excluded.

    Raises :class:`FileNotFoundError` when ``target`` does not exist — a
    surfacing failure rather than a silent empty result, per the project's
    boundary-validation rule.
    """
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(p for p in target.glob("*.md") if p.name != "README.md")
    raise FileNotFoundError(f"Rubric path does not exist or is not a file/directory: {target}")


def load_rubrics(
    paths: Iterable[Path] | None = None,
    *,
    include_builtins: bool = True,
) -> list[Rubric]:
    """Load builtin and/or adopter-custom rubric files.

    Args:
        paths: Optional iterable of custom rubric files or directories. Each
            directory entry is scanned non-recursively for ``*.md`` files
            (``README.md`` excluded); each file entry is loaded directly.
        include_builtins: When ``True`` (default), the shipped rubric corpus
            at ``self_improver/advisor/rubrics/`` is loaded and every returned
            rubric carries :attr:`RubricSource.BUILTIN`. When ``False``, the
            builtins are skipped.

    Returns:
        List of :class:`Rubric`, sorted ASCII-ascending by ``id``.

    Raises:
        MalformedRubricError: A rubric file is missing frontmatter, has
            invalid frontmatter values, or cannot be read.
        RubricFileNameMismatchError: A rubric's filename does not match its
            frontmatter ``id``.
        DuplicateRubricError: Two rubrics share an ``id``.
    """
    collected: dict[str, Rubric] = {}

    if include_builtins:
        for file_path in _enumerate_markdown_files(_BUILTIN_RUBRICS_DIR):
            rubric = _parse_rubric_file(file_path, RubricSource.BUILTIN)
            _register(collected, rubric)

    if paths is not None:
        for path in paths:
            for file_path in _enumerate_markdown_files(path):
                rubric = _parse_rubric_file(file_path, RubricSource.CUSTOM)
                _register(collected, rubric)

    return sorted(collected.values(), key=lambda r: r.id)


def _register(collected: dict[str, Rubric], rubric: Rubric) -> None:
    """Add ``rubric`` to ``collected``, raising on id collision."""
    existing = collected.get(rubric.id)
    if existing is not None:
        raise DuplicateRubricError(paths=(existing.path, rubric.path), rubric_id=rubric.id)
    collected[rubric.id] = rubric


__all__ = [
    "DuplicateRubricError",
    "MalformedRubricError",
    "Rubric",
    "RubricFileNameMismatchError",
    "load_rubrics",
]
