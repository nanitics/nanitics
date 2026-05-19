"""Tests for planning strategy contributors."""

import pytest

from nanitics.capabilities.planning.contributors import (
    AdaptivePlanningContributor,
    DecompositionContributor,
    GoalTrackingContributor,
    UpfrontPlanContributor,
)
from nanitics.strategies.prompts.builder import SystemPromptContributor

ALL_CONTRIBUTORS: list[type[SystemPromptContributor]] = [
    AdaptivePlanningContributor,
    DecompositionContributor,
    UpfrontPlanContributor,
    GoalTrackingContributor,
]


@pytest.mark.parametrize(
    "contributor_cls",
    ALL_CONTRIBUTORS,
    ids=lambda c: c.__name__,
)
class TestContributorProtocol:
    """Every contributor satisfies the SystemPromptContributor protocol."""

    def test_implements_protocol(self, contributor_cls: type) -> None:
        instance = contributor_cls()
        assert isinstance(instance, SystemPromptContributor)

    def test_returns_tuple(self, contributor_cls: type) -> None:
        instance = contributor_cls()
        assert isinstance(instance, SystemPromptContributor)
        result = instance.system_prompt_section()
        assert result is not None
        section_name, content = result
        assert isinstance(section_name, str)
        assert isinstance(content, str)

    def test_section_name_non_empty(self, contributor_cls: type) -> None:
        instance = contributor_cls()
        assert isinstance(instance, SystemPromptContributor)
        result = instance.system_prompt_section()
        assert result is not None
        section_name, _ = result
        assert len(section_name) > 0

    def test_content_non_empty(self, contributor_cls: type) -> None:
        instance = contributor_cls()
        assert isinstance(instance, SystemPromptContributor)
        result = instance.system_prompt_section()
        assert result is not None
        _, content = result
        assert len(content) > 0


def test_section_names_are_unique() -> None:
    """Each contributor produces a distinct section name."""
    names = [result[0] for cls in ALL_CONTRIBUTORS if (result := cls().system_prompt_section()) is not None]
    assert len(names) == len(set(names))


class TestAdaptivePlanningContributor:
    def test_section_name(self) -> None:
        name, _ = AdaptivePlanningContributor().system_prompt_section()
        assert name == "adaptive_planning"

    def test_mentions_create_plan(self) -> None:
        _, content = AdaptivePlanningContributor().system_prompt_section()
        assert "create_plan" in content

    def test_mentions_revision(self) -> None:
        _, content = AdaptivePlanningContributor().system_prompt_section()
        assert "revise" in content.lower()

    def test_mentions_revision_discipline(self) -> None:
        _, content = AdaptivePlanningContributor().system_prompt_section()
        assert "discipline" in content.lower() or "minor deviation" in content.lower()


class TestDecompositionContributor:
    def test_section_name(self) -> None:
        name, _ = DecompositionContributor().system_prompt_section()
        assert name == "decomposition_planning"

    def test_mentions_dependencies(self) -> None:
        _, content = DecompositionContributor().system_prompt_section()
        assert "dependenc" in content.lower()

    def test_mentions_assembly(self) -> None:
        _, content = DecompositionContributor().system_prompt_section()
        assert "assembl" in content.lower() or "synthesize" in content.lower()


class TestUpfrontPlanContributor:
    def test_section_name(self) -> None:
        name, _ = UpfrontPlanContributor().system_prompt_section()
        assert name == "upfront_planning"

    def test_mentions_no_revision(self) -> None:
        _, content = UpfrontPlanContributor().system_prompt_section()
        assert "NOT revise" in content or "no revision" in content.lower()

    def test_mentions_mechanical_execution(self) -> None:
        _, content = UpfrontPlanContributor().system_prompt_section()
        assert "mechanic" in content.lower()

    def test_mentions_continue_on_failure(self) -> None:
        _, content = UpfrontPlanContributor().system_prompt_section()
        assert "fail" in content.lower() and "continue" in content.lower()


class TestGoalTrackingContributor:
    def test_section_name(self) -> None:
        name, _ = GoalTrackingContributor().system_prompt_section()
        assert name == "goal_tracking"

    def test_mentions_priority(self) -> None:
        _, content = GoalTrackingContributor().system_prompt_section()
        assert "priorit" in content.lower()

    def test_mentions_update_goal(self) -> None:
        _, content = GoalTrackingContributor().system_prompt_section()
        assert "update_goal" in content

    def test_mentions_conflict_resolution(self) -> None:
        _, content = GoalTrackingContributor().system_prompt_section()
        assert "conflict" in content.lower()
