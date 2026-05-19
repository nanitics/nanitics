"""Regression tests for the ``ToolResult.metadata`` round-trip contract.

Phase 3 of the Clearpath-feedback-fixes Epic decided that
``ToolResult.metadata`` is propagated onto the constructed ``tool_result``
``Message.metadata`` by every agent that consumes the registry's dispatch
result. These tests pin that contract on the surfaces it touches:

1. ``ReActAgent`` success path — a tool returning
   ``ToolResult(metadata={...})`` produces a ``tool_result`` ``Message``
   carrying the same metadata in ``result.messages``.
2. ``ReActAgent`` suspend/resume — metadata survives the durability
   checkpoint and is restored onto the rebuilt message after resume.
3. ``LATSAgent`` trajectory — ``ActionNode.metadata`` round-trips through
   ``_build_trajectory_messages`` onto the ``tool_result`` ``Message``.
4. ``CodeActAgent`` (pin) — codeact's ``tool_result`` messages have
   ``metadata is None`` because there is no ``ToolResult`` on the codeact
   path. This pins today's behavior so a future Phase that decides to
   project ``ExecutionResult`` fields into metadata makes a deliberate
   change.
5. Anthropic adapter strip — ``Message.metadata`` is not surfaced into
   the Anthropic wire format (negative coverage for the LLM-strip
   guarantee).
"""

from __future__ import annotations

from typing import Any

from nanitics import (
    ExecutionResult,
    Message,
    MockLLMClient,
    MockSandbox,
    ReActAgent,
    ToolCall,
    ToolResult,
    tool,
)
from nanitics.collaboration.approval_wrapped import ApprovalWrappedTool
from nanitics.collaboration.durable_provider import DurableHumanInputProvider
from nanitics.collaboration.hitl_store import InMemoryHitlRequestStore
from nanitics.collaboration.protocol import (
    HumanDecision,
    HumanInputResponse,
)
from nanitics.composition.durability.resume import (
    DurableRun,
    ResumeContext,
    ResumeResult,
    ResumeService,
    SuspendedRun,
)
from nanitics.composition.durability.store import InMemoryCheckpointStore
from nanitics.composition.orchestration.adapters import AgentStep
from nanitics.composition.orchestration.sequential import Sequential
from nanitics.infrastructure.llm.anthropic import _to_anthropic_messages
from nanitics.strategies.agents.codeact import CodeActAgent
from nanitics.strategies.agents.evaluation import (
    EvaluationContext,
    EvaluationResult,
    EvaluationVerdict,
)
from nanitics.strategies.agents.lats import ActionNode, LATSAgent
from tests.testing_helpers import make_emitter, make_response

# ── Fixture tools ────────────────────────────────────────────


@tool(name="lookup", description="Return canned data with metadata")
async def lookup_tool(key: str) -> ToolResult:
    return ToolResult(
        content=f"value for {key}",
        metadata={"key": "value", "protected": True},
    )


@tool(name="empty_meta", description="Return data with empty metadata")
async def empty_meta_tool(x: int) -> ToolResult:
    return ToolResult(content=str(x))  # default metadata == {}


# ── 1. ReActAgent success path ──────────────────────────────


class TestReActAgentMetadataRoundTrip:
    async def test_tool_result_metadata_propagates_to_message(self) -> None:
        """A ``ToolResult(metadata={...})`` round-trips onto the ``tool_result`` ``Message``."""
        responses = [
            make_response(
                content="Looking up",
                tool_calls=[ToolCall(id="tc1", name="lookup", arguments={"key": "k"})],
            ),
            make_response(content="Done"),
        ]
        client = MockLLMClient(responses)
        agent = ReActAgent(
            name="react",
            llm_client=client,
            emitter=make_emitter(),
            system_prompt="Be helpful.",
            tools=[lookup_tool],
        )

        result = await agent.run("look up k")

        tool_result_msgs = [m for m in result.messages if m.role == "tool_result"]
        assert len(tool_result_msgs) == 1
        msg = tool_result_msgs[0]
        assert msg.content == "value for k"
        assert msg.tool_call_id == "tc1"
        assert msg.metadata == {"key": "value", "protected": True}

    async def test_empty_metadata_normalises_to_none_on_message(self) -> None:
        """``ToolResult.metadata == {}`` becomes ``Message.metadata is None``."""
        responses = [
            make_response(
                content="Computing",
                tool_calls=[ToolCall(id="tc1", name="empty_meta", arguments={"x": 7})],
            ),
            make_response(content="Done"),
        ]
        client = MockLLMClient(responses)
        agent = ReActAgent(
            name="react",
            llm_client=client,
            emitter=make_emitter(),
            system_prompt="Be helpful.",
            tools=[empty_meta_tool],
        )

        result = await agent.run("compute")

        tool_result_msgs = [m for m in result.messages if m.role == "tool_result"]
        assert len(tool_result_msgs) == 1
        # The empty-dict default normalises to ``None`` on the message so
        # ``Message.metadata is None`` continues to mean "no metadata."
        assert tool_result_msgs[0].metadata is None


# ── 2. ReActAgent suspend/resume ─────────────────────────────

_TEST_RUN_ID = "test-run-metadata"


@tool(name="meta_lookup", description="Lookup with metadata for resume tests")
async def meta_lookup_tool(key: str) -> ToolResult:
    return ToolResult(
        content=f"meta-result for {key}",
        metadata={"key": "value"},
    )


@tool(name="add_meta", description="Add and trigger approval")
async def add_meta_tool(a: int, b: int) -> str:
    return str(a + b)


class TestReActAgentResumeMetadataRoundTrip:
    async def test_metadata_survives_suspend_resume(self) -> None:
        """Metadata on a completed tool result is preserved across the
        durability checkpoint and restored onto the rebuilt message after
        resume.
        """
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)
        wrapped_tool = ApprovalWrappedTool(tool=add_meta_tool, provider=provider)

        # First run: meta_lookup completes (with metadata), then add suspends.
        client1 = MockLLMClient(
            [
                make_response(
                    content="Both",
                    tool_calls=[
                        ToolCall(id="tc1", name="meta_lookup", arguments={"key": "k"}),
                        ToolCall(id="tc2", name="add_meta", arguments={"a": 1, "b": 2}),
                    ],
                ),
            ]
        )
        agent1 = ReActAgent(
            name="test-agent",
            llm_client=client1,
            emitter=make_emitter(),
            system_prompt="Be helpful.",
            tools=[meta_lookup_tool, wrapped_tool],
            run_id=_TEST_RUN_ID,
        )
        workflow1 = Sequential(
            name="workflow",
            steps=[AgentStep(agent1)],
            emitter=make_emitter(),
            checkpoint_store=checkpoint_store,
            run_id=_TEST_RUN_ID,
        )
        durable = DurableRun(
            workflow1,
            hitl_store=hitl_store,
            checkpoint_store=checkpoint_store,
        )
        suspended = await durable.start("look up and add")
        assert isinstance(suspended, SuspendedRun)

        # Inspect the checkpoint state for the metadata key.
        stored = await checkpoint_store.load(_TEST_RUN_ID)
        assert stored is not None
        agent_checkpoint = stored.state["agent_checkpoint"]
        completed = agent_checkpoint["completed_tool_results"]
        assert "0" in completed
        assert completed["0"]["metadata"] == {"key": "value"}

        # Resume: the rebuilt messages list must carry the same metadata
        # on the first tool-result message.
        captured_messages: list[list[Message]] = []

        def factory(ctx: ResumeContext) -> DurableRun:
            provider2 = DurableHumanInputProvider(request_store=ctx.hitl_store)
            wrapped_tool2 = ApprovalWrappedTool(tool=add_meta_tool, provider=provider2)
            client2 = MockLLMClient([make_response(content="The sum is 3")])
            agent2 = ReActAgent(
                name="test-agent",
                llm_client=client2,
                emitter=make_emitter(),
                system_prompt="Be helpful.",
                tools=[meta_lookup_tool, wrapped_tool2],
                run_id=ctx.run_id,
            )

            # Wrap the agent's _execute_resume so we can capture the
            # rebuilt messages list — that's the only way to observe
            # the post-restore message shape from the outside.
            original_resume = agent2._execute_resume

            async def capturing_resume(*args: Any, **kwargs: Any) -> Any:
                result = await original_resume(*args, **kwargs)
                captured_messages.append(result.messages)
                return result

            agent2._execute_resume = capturing_resume  # type: ignore[method-assign]

            workflow2 = Sequential(
                name="workflow",
                steps=[AgentStep(agent2)],
                emitter=make_emitter(),
                checkpoint_store=ctx.checkpoint_store,
                run_id=ctx.run_id,
            )
            return DurableRun(
                workflow2,
                hitl_store=ctx.hitl_store,
                checkpoint_store=ctx.checkpoint_store,
            )

        service = ResumeService(
            hitl_store=hitl_store,
            checkpoint_store=checkpoint_store,
            factory=factory,
        )
        response = HumanInputResponse(
            request_id=suspended.pending_request.request_id,
            decision=HumanDecision.APPROVE,
        )
        result = await service.resume(suspended.run_id, response)
        assert isinstance(result, ResumeResult)

        assert len(captured_messages) == 1
        rebuilt = captured_messages[0]
        tool_result_msgs = [m for m in rebuilt if m.role == "tool_result"]
        # Two tool results: the restored meta_lookup (with metadata) and
        # the just-completed add (no metadata, synthesised by the wrapper).
        assert len(tool_result_msgs) >= 1
        first = next(m for m in tool_result_msgs if m.tool_call_id == "tc1")
        assert first.metadata == {"key": "value"}

    async def test_pre_phase_3_checkpoint_loads_with_none_metadata(self) -> None:
        """A checkpoint dict that lacks the ``"metadata"`` key still resumes —
        the restored message has ``metadata=None``.
        """
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)
        wrapped_tool = ApprovalWrappedTool(tool=add_meta_tool, provider=provider)

        # Drive a normal suspend so we have a real checkpoint…
        client1 = MockLLMClient(
            [
                make_response(
                    content="Both",
                    tool_calls=[
                        ToolCall(id="tc1", name="meta_lookup", arguments={"key": "k"}),
                        ToolCall(id="tc2", name="add_meta", arguments={"a": 1, "b": 2}),
                    ],
                ),
            ]
        )
        agent1 = ReActAgent(
            name="test-agent",
            llm_client=client1,
            emitter=make_emitter(),
            system_prompt="Be helpful.",
            tools=[meta_lookup_tool, wrapped_tool],
            run_id=_TEST_RUN_ID + "-old",
        )
        workflow1 = Sequential(
            name="workflow",
            steps=[AgentStep(agent1)],
            emitter=make_emitter(),
            checkpoint_store=checkpoint_store,
            run_id=_TEST_RUN_ID + "-old",
        )
        durable = DurableRun(
            workflow1,
            hitl_store=hitl_store,
            checkpoint_store=checkpoint_store,
        )
        suspended = await durable.start("look up and add")
        assert isinstance(suspended, SuspendedRun)

        # …then mutate the on-disk checkpoint to look like a pre-Phase-3
        # entry — strip the ``"metadata"`` key from each completed result
        # and from any serialized tool_result message. A genuine
        # pre-Phase-3 checkpoint never carried metadata anywhere because
        # the propagation didn't exist yet.
        # ``RunCheckpoint`` is frozen but ``state`` is a regular dict, so
        # in-place mutation of the nested dict is fine.
        stored = await checkpoint_store.load(_TEST_RUN_ID + "-old")
        assert stored is not None
        agent_checkpoint = stored.state["agent_checkpoint"]
        for entry in agent_checkpoint["completed_tool_results"].values():
            entry.pop("metadata", None)
        for msg_dict in agent_checkpoint["messages"]:
            if msg_dict.get("role") == "tool_result":
                msg_dict["metadata"] = None

        captured_messages: list[list[Message]] = []

        def factory(ctx: ResumeContext) -> DurableRun:
            provider2 = DurableHumanInputProvider(request_store=ctx.hitl_store)
            wrapped_tool2 = ApprovalWrappedTool(tool=add_meta_tool, provider=provider2)
            client2 = MockLLMClient([make_response(content="The sum is 3")])
            agent2 = ReActAgent(
                name="test-agent",
                llm_client=client2,
                emitter=make_emitter(),
                system_prompt="Be helpful.",
                tools=[meta_lookup_tool, wrapped_tool2],
                run_id=ctx.run_id,
            )
            original_resume = agent2._execute_resume

            async def capturing_resume(*args: Any, **kwargs: Any) -> Any:
                result = await original_resume(*args, **kwargs)
                captured_messages.append(result.messages)
                return result

            agent2._execute_resume = capturing_resume  # type: ignore[method-assign]

            workflow2 = Sequential(
                name="workflow",
                steps=[AgentStep(agent2)],
                emitter=make_emitter(),
                checkpoint_store=ctx.checkpoint_store,
                run_id=ctx.run_id,
            )
            return DurableRun(
                workflow2,
                hitl_store=ctx.hitl_store,
                checkpoint_store=ctx.checkpoint_store,
            )

        service = ResumeService(
            hitl_store=hitl_store,
            checkpoint_store=checkpoint_store,
            factory=factory,
        )
        response = HumanInputResponse(
            request_id=suspended.pending_request.request_id,
            decision=HumanDecision.APPROVE,
        )
        result = await service.resume(suspended.run_id, response)
        assert isinstance(result, ResumeResult)

        assert len(captured_messages) == 1
        rebuilt = captured_messages[0]
        first = next(m for m in rebuilt if m.role == "tool_result" and m.tool_call_id == "tc1")
        # No KeyError, and the absent metadata restores as ``None``.
        assert first.metadata is None


# ── 3. LATSAgent trajectory ──────────────────────────────────


class _AcceptEvaluator:
    """Always-accept node evaluator (no-op for these structural tests)."""

    @property
    def max_revisions(self) -> int:
        return 0

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        return EvaluationResult(
            verdict=EvaluationVerdict.ACCEPT,
            score=1.0,
            evaluator_name="accept",
        )


class TestLATSMetadataRoundTrip:
    def _make_agent(self) -> LATSAgent:
        return LATSAgent(
            name="lats",
            llm_client=MockLLMClient([]),
            emitter=make_emitter(),
            system_prompt="Be helpful.",
            tools=[],
            node_evaluator=_AcceptEvaluator(),
        )

    def test_action_node_metadata_round_trips_through_trajectory(self) -> None:
        """``ActionNode.metadata`` is copied onto the ``tool_result`` ``Message``
        produced by ``_build_trajectory_messages``.
        """
        agent = self._make_agent()
        root = ActionNode(id="root", thought="task", depth=0, children_ids=["child"])
        child = ActionNode(
            id="child",
            parent_id="root",
            depth=1,
            thought="search",
            action="lookup",
            action_input={"key": "k"},
            observation="value for k",
            metadata={"key": "value"},
        )
        agent._nodes = {"root": root, "child": child}
        agent._root_id = "root"

        messages = agent._build_trajectory_messages("child", "look up k")

        tool_result_msgs = [m for m in messages if m.role == "tool_result"]
        assert len(tool_result_msgs) == 1
        msg = tool_result_msgs[0]
        assert msg.content == "value for k"
        assert msg.tool_call_id == "child"
        assert msg.metadata == {"key": "value"}

    def test_action_node_without_metadata_yields_none_on_message(self) -> None:
        """A node with ``metadata=None`` (the default for thought-only or
        legacy nodes) yields ``Message.metadata is None``.
        """
        agent = self._make_agent()
        root = ActionNode(id="root", thought="task", depth=0, children_ids=["child"])
        child = ActionNode(
            id="child",
            parent_id="root",
            depth=1,
            thought="search",
            action="lookup",
            action_input={"key": "k"},
            observation="value for k",
            # metadata not provided — defaults to None
        )
        agent._nodes = {"root": root, "child": child}
        agent._root_id = "root"

        messages = agent._build_trajectory_messages("child", "look up k")
        tool_result_msgs = [m for m in messages if m.role == "tool_result"]
        assert tool_result_msgs[0].metadata is None


# ── 4. CodeAct: pin Message.metadata is None ────────────────


class TestCodeActMetadataPin:
    async def test_codeact_tool_result_message_metadata_is_none(self) -> None:
        """CodeAct's ``tool_result`` ``Message.metadata`` is ``None`` because
        the codeact path constructs observations from ``ExecutionResult``,
        not from a ``ToolResult``. Pinning today's behavior so a future
        Phase that decides to project ``ExecutionResult`` fields into
        metadata makes a deliberate change.
        """
        from nanitics import LLMResponse, Usage

        # Single iteration: code → exec result → final answer.
        code_response = LLMResponse(
            content="Let me compute",
            tool_calls=[ToolCall(id="tc1", name="execute_code", arguments={"code": "print(42)"})],
            usage=Usage(input_tokens=10, output_tokens=5),
            model="test-model",
            stop_reason="tool_use",
        )
        final_response = LLMResponse(
            content="The answer is 42",
            tool_calls=[],
            usage=Usage(input_tokens=10, output_tokens=5),
            model="test-model",
            stop_reason="end_turn",
        )
        client = MockLLMClient([code_response, final_response])
        sandbox = MockSandbox(
            [
                ExecutionResult(
                    stdout="42\n",
                    stderr="",
                    return_value="42",
                    success=True,
                    error=None,
                    duration_ms=1.0,
                ),
            ]
        )
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=make_emitter(),
            system_prompt="Be helpful.",
            sandbox=sandbox,
        )

        result = await agent.run("compute 42")

        tool_result_msgs = [m for m in result.messages if m.role == "tool_result"]
        assert len(tool_result_msgs) == 1
        # Pinned: codeact does not project ``ExecutionResult`` fields into
        # ``Message.metadata`` — the gap is intentional and out of Phase
        # 3 scope.
        assert tool_result_msgs[0].metadata is None


# ── 5. Anthropic adapter strip ──────────────────────────────


class TestAnthropicAdapterStripsMessageMetadata:
    def test_message_metadata_is_not_surfaced_in_anthropic_format(self) -> None:
        """``_to_anthropic_messages`` does not read or forward ``Message.metadata``.

        Pins the LLM-strip guarantee: propagating ``ToolResult.metadata`` onto
        ``Message.metadata`` cannot leak structured data into the LLM context
        because the Anthropic adapter constructs the wire ``tool_result`` block
        from ``content`` + ``tool_use_id`` only.
        """
        msg = Message(
            role="tool_result",
            content="ok",
            tool_call_id="t1",
            metadata={"key": "value", "protected": True},
        )

        result = _to_anthropic_messages([msg])

        assert result == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "ok",
                    }
                ],
            }
        ]

        # Defense-in-depth: walk the produced structure and assert no
        # key or value derived from the message metadata appears.
        def _walk(value: object) -> None:
            if isinstance(value, dict):
                for k, v in value.items():
                    assert k != "metadata"
                    assert v != {"key": "value", "protected": True}
                    _walk(k)
                    _walk(v)
            elif isinstance(value, list):
                for item in value:
                    _walk(item)
            elif isinstance(value, str):
                assert "protected" not in value

        _walk(result)
