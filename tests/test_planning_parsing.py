"""Tests for parsing plan/goal state from working memory markdown."""

from nanitics.capabilities.planning.models import GoalStatus, StepStatus
from nanitics.capabilities.planning.parsing import (
    parse_goals_from_working_memory,
    parse_plan_from_working_memory,
)


class TestParsePlanFromWorkingMemory:
    def test_all_step_markers(self):
        content = (
            "- [ ] Not started step\n- [→] In progress step\n- [✓] Completed step\n"
            "- [✗] Failed step\n- [~] Skipped step\n"
        )
        result = parse_plan_from_working_memory(content)
        assert result == [
            ("Not started step", StepStatus.not_started),
            ("In progress step", StepStatus.in_progress),
            ("Completed step", StepStatus.completed),
            ("Failed step", StepStatus.failed),
            ("Skipped step", StepStatus.skipped),
        ]

    def test_empty_input(self):
        assert parse_plan_from_working_memory("") == []

    def test_no_checkboxes(self):
        content = "Some random text\nAnother line\n## Heading"
        assert parse_plan_from_working_memory(content) == []

    def test_mixed_content(self):
        content = (
            "## Current Plan\n\n- [✓] Research the topic\n- [→] Write the draft\n"
            "Some notes here\n- [ ] Review and edit\n"
        )
        result = parse_plan_from_working_memory(content)
        assert result == [
            ("Research the topic", StepStatus.completed),
            ("Write the draft", StepStatus.in_progress),
            ("Review and edit", StepStatus.not_started),
        ]

    def test_extra_whitespace(self):
        content = "  - [ ] Indented step\n    - [✓] Double indented\n- [ ]   Extra spaces in desc  \n"
        result = parse_plan_from_working_memory(content)
        assert len(result) == 3
        assert result[0] == ("Indented step", StepStatus.not_started)
        assert result[1] == ("Double indented", StepStatus.completed)
        assert result[2] == ("Extra spaces in desc", StepStatus.not_started)

    def test_empty_marker_defaults_to_not_started(self):
        content = "- [] Some step\n"
        result = parse_plan_from_working_memory(content)
        assert result == [("Some step", StepStatus.not_started)]

    def test_unrecognized_marker_defaults_to_not_started(self):
        content = "- [?] Unknown marker step\n"
        result = parse_plan_from_working_memory(content)
        assert result == [("Unknown marker step", StepStatus.not_started)]

    def test_malformed_input_ignored(self):
        content = (
            "- [✓ Missing bracket\n"
            "[✓] No dash prefix\n"
            "- [✓]\n"  # marker but no description — regex requires .+
        )
        assert parse_plan_from_working_memory(content) == []


class TestParseGoalsFromWorkingMemory:
    def test_all_goal_markers(self):
        content = (
            "- [ ] Active goal\n- [→] Also active goal\n- [✓] Achieved goal\n- [✗] Abandoned goal\n- [~] Blocked goal\n"
        )
        result = parse_goals_from_working_memory(content)
        assert result == [
            ("Active goal", GoalStatus.active),
            ("Also active goal", GoalStatus.active),
            ("Achieved goal", GoalStatus.achieved),
            ("Abandoned goal", GoalStatus.abandoned),
            ("Blocked goal", GoalStatus.blocked),
        ]

    def test_empty_input(self):
        assert parse_goals_from_working_memory("") == []

    def test_mixed_content(self):
        content = "## Goals\n\n- [✓] Complete phase 1\nOther text here\n- [ ] Start phase 2\n"
        result = parse_goals_from_working_memory(content)
        assert result == [
            ("Complete phase 1", GoalStatus.achieved),
            ("Start phase 2", GoalStatus.active),
        ]

    def test_unrecognized_marker_defaults_to_active(self):
        content = "- [!] Priority goal\n"
        result = parse_goals_from_working_memory(content)
        assert result == [("Priority goal", GoalStatus.active)]

    def test_whitespace_tolerance(self):
        content = "  - [ ]   Indented goal with spaces  \n"
        result = parse_goals_from_working_memory(content)
        assert result == [("Indented goal with spaces", GoalStatus.active)]

    def test_malformed_input_ignored(self):
        content = "- [✓ No closing bracket\n[✓] Missing dash\n"
        assert parse_goals_from_working_memory(content) == []
