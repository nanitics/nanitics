import pytest
from pydantic import ValidationError

from nanitics.composition.multi_agent.context_transfer import (
    ContextTransferStrategy,
)
from nanitics.composition.multi_agent.handoff_protocol import (
    HandoffPayload,
    HandoffTransfer,
    handoff_receiver_instructions,
    handoff_sender_instructions,
)
from nanitics.infrastructure.llm.protocol import Message
from nanitics.infrastructure.observability.events import Usage
from nanitics.strategies.agents.base import AgentResult


def _make_result(
    output: str | None = "final answer",
    messages: list[Message] | None = None,
) -> AgentResult:
    return AgentResult(
        output=output,
        total_steps=1,
        termination_reason="completed",
        messages=messages or [],
        usage=Usage(input_tokens=10, output_tokens=5),
    )


class TestHandoffPayload:
    def test_construction_required_only(self):
        payload = HandoffPayload(task_state="Researching topic X")
        assert payload.task_state == "Researching topic X"
        assert payload.findings == []
        assert payload.decisions == []
        assert payload.open_questions == []
        assert payload.artifacts == {}
        assert payload.metadata == {}

    def test_construction_all_fields(self):
        payload = HandoffPayload(
            task_state="Analysis complete",
            findings=["Found pattern A", "Found pattern B"],
            decisions=["Chose approach 1"],
            open_questions=["What about edge case?"],
            artifacts={"report": "Full report text"},
            metadata={"confidence": 0.9},
        )
        assert len(payload.findings) == 2
        assert payload.artifacts["report"] == "Full report text"

    def test_frozen(self):
        payload = HandoffPayload(task_state="test")

        with pytest.raises(ValidationError):
            payload.task_state = "modified"

    def test_render_all_sections(self):
        payload = HandoffPayload(
            task_state="Completed research",
            findings=["Fact 1", "Fact 2"],
            decisions=["Decision A"],
            open_questions=["Question?"],
            artifacts={"doc": "Document content"},
        )
        text = payload.render()
        assert "## Handoff Context" in text
        assert "### Task State" in text
        assert "Completed research" in text
        assert "### Findings" in text
        assert "- Fact 1" in text
        assert "- Fact 2" in text
        assert "### Decisions" in text
        assert "- Decision A" in text
        assert "### Open Questions" in text
        assert "- Question?" in text
        assert "### Artifacts" in text
        assert "#### doc" in text
        assert "Document content" in text

    def test_render_omits_empty_sections(self):
        payload = HandoffPayload(task_state="Only state")
        text = payload.render()
        assert "## Handoff Context" in text
        assert "### Task State" in text
        assert "Only state" in text
        assert "### Findings" not in text
        assert "### Decisions" not in text
        assert "### Open Questions" not in text
        assert "### Artifacts" not in text

    def test_render_partial_sections(self):
        payload = HandoffPayload(
            task_state="In progress",
            findings=["One finding"],
        )
        text = payload.render()
        assert "### Findings" in text
        assert "- One finding" in text
        assert "### Decisions" not in text

    def test_serialization_roundtrip(self):
        payload = HandoffPayload(
            task_state="test",
            findings=["f1"],
            artifacts={"a": "b"},
        )
        data = payload.model_dump()
        restored = HandoffPayload.model_validate(data)
        assert restored == payload


class TestHandoffTransfer:
    def test_satisfies_protocol(self):
        def builder(result: AgentResult) -> HandoffPayload:
            return HandoffPayload(task_state=result.output or "")

        assert isinstance(HandoffTransfer(builder), ContextTransferStrategy)

    async def test_extract_calls_builder_and_renders(self):
        def builder(result: AgentResult) -> HandoffPayload:
            return HandoffPayload(
                task_state=result.output or "",
                findings=["extracted from result"],
            )

        transfer = HandoffTransfer(builder)
        result = _make_result(output="Done with task")
        text = await transfer.extract(result)
        assert "Done with task" in text
        assert "- extracted from result" in text

    async def test_extract_with_message_based_builder(self):
        messages = [
            Message(role="user", content="Do task"),
            Message(role="assistant", content="Completed task"),
        ]

        def builder(result: AgentResult) -> HandoffPayload:
            return HandoffPayload(
                task_state="Task completed",
                findings=[m.content for m in result.messages if isinstance(m.content, str) and m.role == "assistant"],
            )

        transfer = HandoffTransfer(builder)
        result = _make_result(output="done", messages=messages)
        text = await transfer.extract(result)
        assert "Task completed" in text
        assert "- Completed task" in text


class TestHandoffSenderInstructions:
    def test_default_fields(self):
        text = handoff_sender_instructions()
        assert "task_state" in text
        assert "findings" in text
        assert "decisions" in text
        assert "open_questions" in text
        assert "artifacts" in text

    def test_custom_fields(self):
        text = handoff_sender_instructions(payload_fields=["task_state", "findings"])
        assert "task_state" in text
        assert "findings" in text
        assert "decisions" not in text
        assert "open_questions" not in text


class TestHandoffReceiverInstructions:
    def test_returns_instructions(self):
        text = handoff_receiver_instructions()
        assert "handoff" in text.lower()
        assert "Task State" in text
        assert "Findings" in text
        assert "Open Questions" in text
