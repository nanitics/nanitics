"""Unit tests for :mod:`self_improver.advisor.rubric`.

Covers builtin discovery, custom-path merging, error paths
(malformed frontmatter, filename/id mismatch, duplicates), deterministic
ordering, and source-label correctness.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from self_improver.advisor import (
    DuplicateRubricError,
    MalformedRubricError,
    ProposalCategory,
    ProposalSeverity,
    Rubric,
    RubricFileNameMismatchError,
    RubricSource,
    load_rubrics,
)


def _write_rubric(
    directory: Path,
    rubric_id: str,
    *,
    severity: str = "observation",
    category: str = "prompts",
    target_dimension: str = "prompts",
    body: str = "Body text.\n",
    filename: str | None = None,
    extra_frontmatter: str = "",
) -> Path:
    filename = filename or f"{rubric_id}.md"
    path = directory / filename
    content = (
        "---\n"
        f"id: {rubric_id}\n"
        f"severity: {severity}\n"
        f"category: {category}\n"
        f"target_dimension: {target_dimension}\n"
        f"{extra_frontmatter}"
        "---\n"
        "\n"
        f"{body}"
    )
    path.write_text(content, encoding="utf-8")
    return path


class TestBuiltinLoad:
    def test_load_returns_builtin_corpus(self) -> None:
        rubrics = load_rubrics()
        assert rubrics, "builtin corpus must not be empty"
        assert all(isinstance(r, Rubric) for r in rubrics)
        assert all(r.source is RubricSource.BUILTIN for r in rubrics)

    def test_ids_are_unique(self) -> None:
        rubrics = load_rubrics()
        ids = [r.id for r in rubrics]
        assert len(ids) == len(set(ids))

    def test_sorted_by_id(self) -> None:
        rubrics = load_rubrics()
        ids = [r.id for r in rubrics]
        assert ids == sorted(ids)

    def test_covers_three_launch_dimensions(self) -> None:
        rubrics = load_rubrics()
        dimensions = {r.target_dimension for r in rubrics}
        assert {"prompts", "tool_descriptions", "coordination_patterns"}.issubset(dimensions)

    def test_covers_three_severities(self) -> None:
        rubrics = load_rubrics()
        severities = {r.severity for r in rubrics}
        assert severities == {
            ProposalSeverity.CRITICAL,
            ProposalSeverity.WARNING,
            ProposalSeverity.OBSERVATION,
        }

    def test_builtin_bodies_non_empty(self) -> None:
        for rubric in load_rubrics():
            assert rubric.body.strip(), f"rubric {rubric.id} has empty body"


class TestCustomLoad:
    def test_custom_path_merges_with_builtins(self, tmp_path: Path) -> None:
        _write_rubric(tmp_path, "adopter-latency-budget", category="configuration")
        rubrics = load_rubrics(paths=[tmp_path])
        ids = [r.id for r in rubrics]
        assert "adopter-latency-budget" in ids
        adopter = next(r for r in rubrics if r.id == "adopter-latency-budget")
        assert adopter.source is RubricSource.CUSTOM
        # Builtins are still present.
        assert any(r.source is RubricSource.BUILTIN for r in rubrics)

    def test_custom_file_path_accepted(self, tmp_path: Path) -> None:
        file_path = _write_rubric(tmp_path, "adopter-single-file", category="evaluation")
        rubrics = load_rubrics(paths=[file_path], include_builtins=False)
        assert len(rubrics) == 1
        assert rubrics[0].id == "adopter-single-file"
        assert rubrics[0].source is RubricSource.CUSTOM

    def test_include_builtins_false_returns_adopter_only(self, tmp_path: Path) -> None:
        _write_rubric(tmp_path, "adopter-one")
        _write_rubric(tmp_path, "adopter-two")
        rubrics = load_rubrics(paths=[tmp_path], include_builtins=False)
        ids = {r.id for r in rubrics}
        assert ids == {"adopter-one", "adopter-two"}
        assert all(r.source is RubricSource.CUSTOM for r in rubrics)

    def test_include_builtins_false_without_paths_returns_empty(self) -> None:
        assert load_rubrics(include_builtins=False) == []

    def test_directory_skips_readme(self, tmp_path: Path) -> None:
        _write_rubric(tmp_path, "adopter-with-readme")
        (tmp_path / "README.md").write_text("# Adopter rubrics\n", encoding="utf-8")
        rubrics = load_rubrics(paths=[tmp_path], include_builtins=False)
        assert [r.id for r in rubrics] == ["adopter-with-readme"]

    def test_ordering_is_deterministic_across_sources(self, tmp_path: Path) -> None:
        _write_rubric(tmp_path, "aaaa-adopter")
        _write_rubric(tmp_path, "zzzz-adopter")
        rubrics = load_rubrics(paths=[tmp_path])
        ids = [r.id for r in rubrics]
        assert ids == sorted(ids)


class TestErrors:
    def test_missing_frontmatter_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.md"
        path.write_text("No frontmatter here.\n", encoding="utf-8")
        with pytest.raises(MalformedRubricError) as excinfo:
            load_rubrics(paths=[tmp_path], include_builtins=False)
        assert excinfo.value.path == path
        assert "frontmatter" in excinfo.value.reason.lower()

    def test_unterminated_frontmatter_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "unterminated.md"
        path.write_text(
            "---\nid: unterminated\nseverity: warning\ncategory: prompts\ntarget_dimension: prompts\n",
            encoding="utf-8",
        )
        with pytest.raises(MalformedRubricError) as excinfo:
            load_rubrics(paths=[tmp_path], include_builtins=False)
        assert excinfo.value.path == path
        assert "unterminated" in excinfo.value.reason.lower()

    def test_unparseable_yaml_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad-yaml.md"
        path.write_text(
            "---\nid: bad-yaml\nseverity: : :\ncategory: prompts\ntarget_dimension: prompts\n---\nBody\n",
            encoding="utf-8",
        )
        with pytest.raises(MalformedRubricError) as excinfo:
            load_rubrics(paths=[tmp_path], include_builtins=False)
        assert excinfo.value.path == path

    def test_non_mapping_frontmatter_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "list-frontmatter.md"
        path.write_text("---\n- one\n- two\n---\nBody\n", encoding="utf-8")
        with pytest.raises(MalformedRubricError) as excinfo:
            load_rubrics(paths=[tmp_path], include_builtins=False)
        assert "mapping" in excinfo.value.reason.lower()

    def test_missing_required_key_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "missing-category.md"
        path.write_text(
            "---\nid: missing-category\nseverity: warning\ntarget_dimension: prompts\n---\nBody\n",
            encoding="utf-8",
        )
        with pytest.raises(MalformedRubricError) as excinfo:
            load_rubrics(paths=[tmp_path], include_builtins=False)
        assert "category" in excinfo.value.reason

    def test_invalid_severity_raises(self, tmp_path: Path) -> None:
        _write_rubric(tmp_path, "bad-severity", severity="catastrophic")
        with pytest.raises(MalformedRubricError):
            load_rubrics(paths=[tmp_path], include_builtins=False)

    def test_invalid_category_raises(self, tmp_path: Path) -> None:
        _write_rubric(tmp_path, "bad-category", category="mystery")
        with pytest.raises(MalformedRubricError):
            load_rubrics(paths=[tmp_path], include_builtins=False)

    def test_invalid_target_dimension_raises(self, tmp_path: Path) -> None:
        _write_rubric(tmp_path, "bad-dimension", target_dimension="unknown")
        with pytest.raises(MalformedRubricError) as excinfo:
            load_rubrics(paths=[tmp_path], include_builtins=False)
        assert "target_dimension" in excinfo.value.reason

    def test_filename_id_mismatch_raises(self, tmp_path: Path) -> None:
        path = _write_rubric(tmp_path, "real-id", filename="different.md")
        with pytest.raises(RubricFileNameMismatchError) as excinfo:
            load_rubrics(paths=[tmp_path], include_builtins=False)
        assert excinfo.value.path == path
        assert excinfo.value.rubric_id == "real-id"

    def test_empty_id_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "empty-id.md"
        path.write_text(
            "---\nid: ''\nseverity: warning\ncategory: prompts\ntarget_dimension: prompts\n---\nBody\n",
            encoding="utf-8",
        )
        with pytest.raises(MalformedRubricError) as excinfo:
            load_rubrics(paths=[tmp_path], include_builtins=False)
        assert "id" in excinfo.value.reason

    def test_non_string_id_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "numeric-id.md"
        path.write_text(
            "---\nid: 42\nseverity: warning\ncategory: prompts\ntarget_dimension: prompts\n---\nBody\n",
            encoding="utf-8",
        )
        with pytest.raises(MalformedRubricError) as excinfo:
            load_rubrics(paths=[tmp_path], include_builtins=False)
        assert "id" in excinfo.value.reason

    def test_duplicate_id_within_custom_raises(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        path_a = _write_rubric(dir_a, "shared-id")
        path_b = _write_rubric(dir_b, "shared-id")
        with pytest.raises(DuplicateRubricError) as excinfo:
            load_rubrics(paths=[dir_a, dir_b], include_builtins=False)
        assert excinfo.value.rubric_id == "shared-id"
        assert set(excinfo.value.paths) == {path_a, path_b}

    def test_duplicate_id_across_builtin_and_custom_raises(self, tmp_path: Path) -> None:
        # Pick a known-shipped builtin id to collide with.
        builtins = load_rubrics()
        colliding_id = builtins[0].id
        path = _write_rubric(tmp_path, colliding_id)
        with pytest.raises(DuplicateRubricError) as excinfo:
            load_rubrics(paths=[tmp_path])
        assert excinfo.value.rubric_id == colliding_id
        # The custom path appears in the error.
        assert path in excinfo.value.paths

    def test_missing_path_raises_file_not_found(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist"
        with pytest.raises(FileNotFoundError):
            load_rubrics(paths=[missing], include_builtins=False)

    def test_unreadable_file_raises_malformed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Simulate an unreadable file by monkey-patching ``Path.read_text``
        # for a single path — the loader must surface the failure rather
        # than silently skip it.
        path = _write_rubric(tmp_path, "unreadable")

        original_read_text = Path.read_text

        def failing_read_text(self: Path, *args: object, **kwargs: object) -> str:
            if self == path:
                raise OSError("simulated permission denied")
            return original_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(Path, "read_text", failing_read_text)

        with pytest.raises(MalformedRubricError) as excinfo:
            load_rubrics(paths=[tmp_path], include_builtins=False)
        assert excinfo.value.path == path
        assert "cannot read" in excinfo.value.reason.lower()


class TestRubricModel:
    def test_rubric_frozen(self, tmp_path: Path) -> None:
        _write_rubric(tmp_path, "frozen-check")
        (rubric,) = load_rubrics(paths=[tmp_path], include_builtins=False)
        with pytest.raises(ValidationError):
            rubric.body = "mutated"  # type: ignore[misc]

    def test_forward_compat_extra_frontmatter_keys_ignored(self, tmp_path: Path) -> None:
        _write_rubric(
            tmp_path,
            "extra-keys",
            extra_frontmatter="future_field: some_value\nanother_future_field: 42\n",
        )
        (rubric,) = load_rubrics(paths=[tmp_path], include_builtins=False)
        assert rubric.id == "extra-keys"

    def test_body_leading_whitespace_stripped(self, tmp_path: Path) -> None:
        path = tmp_path / "ws-body.md"
        path.write_text(
            "---\n"
            "id: ws-body\n"
            "severity: warning\n"
            "category: prompts\n"
            "target_dimension: prompts\n"
            "---\n"
            "\n\n"
            "Actual body text.\n",
            encoding="utf-8",
        )
        (rubric,) = load_rubrics(paths=[tmp_path], include_builtins=False)
        assert rubric.body.startswith("Actual body")

    def test_category_is_proposal_category_enum(self, tmp_path: Path) -> None:
        _write_rubric(tmp_path, "enum-check", category="tool-descriptions")
        (rubric,) = load_rubrics(paths=[tmp_path], include_builtins=False)
        assert rubric.category is ProposalCategory.TOOL_DESCRIPTIONS
