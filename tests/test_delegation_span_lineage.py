"""Delegation + capability span-lineage regression tests.

Pins the invariant that agent-owned emitter-caching capabilities
(memory providers, output evaluators) emit trace events under the
*current* per-task bound emitter — not the emitter cached at
construction time — so memory / evaluation events in a delegated
inner-agent run carry the outer caller's ``trace_id`` and the
delegate's ``span_id`` as their ``parent_span_id``.

Uses ``MockLLMClient`` for determinism; no real provider calls.
"""

from __future__ import annotations

from nanitics import InMemoryEmitter, ReActAgent
from nanitics.capabilities.evaluation.llm_evaluator import LLMEvaluator
from nanitics.capabilities.memory.episodic import (
    Episode,
    EpisodicMemoryProvider,
    InMemoryEpisodeStore,
    OutcomeType,
)
from nanitics.capabilities.memory.shared import (
    InMemorySharedMemory,
    SharedMemoryProvider,
)
from nanitics.capabilities.memory.working_memory import (
    InMemoryWorkingMemory,
    WorkingMemoryProvider,
)
from nanitics.composition.multi_agent.agent_tool import AgentTool
from nanitics.infrastructure.embeddings.mock import MockEmbeddingClient
from nanitics.infrastructure.llm.mock import MockLLMClient
from nanitics.infrastructure.llm.protocol import LLMResponse, Message, ToolCall
from nanitics.infrastructure.observability.events import (
    AgentStartEvent,
    EpisodeRecallEvent,
    LLMRequestEvent,
    SharedMemoryReadEvent,
    WorkingMemoryReadEvent,
)
from nanitics.strategies.agents.evaluation import EvaluationContext
from tests.testing_helpers import make_response, make_usage


class TestDelegationMemorySpanLineage:
    async def test_working_memory_event_attributes_to_delegate_span(self) -> None:
        """Memory events emitted inside an AgentTool-delegated run carry the
        outer trace_id and the delegate agent's span_id as parent_span_id.

        The inner agent is constructed with its own ``InMemoryEmitter``, and
        the ``WorkingMemoryProvider`` is constructed with no emitter.
        Agent.__init__ auto-wires ``emitter_provider=lambda: self._emitter``
        on the attached provider. Under delegation, the inner agent's
        ``self._emitter`` resolves (via ``ContextVar``) to the per-task
        bound child emitter, so the provider emits into the caller's trace.
        """
        memory = InMemoryWorkingMemory()
        memory.write("## scratch\nfoo")
        provider = WorkingMemoryProvider(memory=memory)
        # Provider has no static emitter and no pre-wired provider.
        assert provider._static_emitter is None

        inner_emitter = InMemoryEmitter(trace_id="inner-default-trace")
        inner_agent = ReActAgent(
            name="inner",
            llm_client=MockLLMClient([make_response(content="inner-answer")]),
            emitter=inner_emitter,
            system_prompt="inner",
            tools=[],
            context_providers=[provider],
        )

        # Auto-wiring installed the callback.
        assert provider._emitter_provider is not None

        # Build an outer coordinator that uses AgentTool to delegate to
        # the inner agent. The outer emitter is what tracks trace_id.
        outer_emitter = InMemoryEmitter(trace_id="outer-trace")
        delegate_tool = AgentTool(
            agent=inner_agent,
            emitter=outer_emitter,
            description="Delegate to the inner agent",
            caller_name="coordinator",
        )

        outer_responses = [
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="tc1", name="inner", arguments={"task": "do work"})],
                usage=make_usage(),
                model="test",
                stop_reason="tool_use",
            ),
            make_response(content="outer-done"),
        ]
        outer_agent = ReActAgent(
            name="coordinator",
            llm_client=MockLLMClient(outer_responses),
            emitter=outer_emitter,
            system_prompt="outer",
            tools=[delegate_tool],
        )

        await outer_agent.run("kick off")

        memory_events = [e for e in outer_emitter.events if isinstance(e, WorkingMemoryReadEvent)]
        assert memory_events, "Expected WorkingMemoryReadEvent on the outer emitter"
        # The memory event was emitted inside the delegated inner run,
        # so it should carry the outer trace_id (delegation routes through
        # the outer's child emitter) — NOT the inner agent's default trace.
        assert all(e.trace_id == "outer-trace" for e in memory_events)
        # And its parent_span_id should be one of the inner agent's span_ids
        # (the span opened when the inner agent started running).
        inner_start_events = [
            e for e in outer_emitter.events if isinstance(e, AgentStartEvent) and e.agent_name == "inner"
        ]
        assert inner_start_events, "Expected inner AgentStartEvent on the outer emitter"
        inner_span_ids = {e.span_id for e in inner_start_events}
        # The memory event was emitted within the inner agent's span, so
        # its parent_span_id is the inner agent's span_id.
        assert {e.parent_span_id for e in memory_events} <= inner_span_ids


class TestEmitterProviderResolution:
    """Directly exercise the ``emitter_provider`` callback path on the
    memory providers and ``LLMEvaluator`` — the provider-branch of each
    capability's ``_emitter`` property.
    """

    async def test_episodic_provider_resolves_via_emitter_provider(self) -> None:
        store: InMemoryEpisodeStore = InMemoryEpisodeStore(embedding_client=MockEmbeddingClient())
        await store.record(
            Episode(
                situation="ask-x",
                action="do-y",
                outcome=OutcomeType.SUCCESS,
                outcome_detail="ok",
            )
        )
        live_emitter = InMemoryEmitter(trace_id="live")
        provider = EpisodicMemoryProvider(store=store, emitter_provider=lambda: live_emitter)
        assert provider._emitter is live_emitter
        await provider.provide([Message(role="user", content="ask-x")])
        recall_events = [e for e in live_emitter.events if isinstance(e, EpisodeRecallEvent)]
        assert len(recall_events) == 1
        assert recall_events[0].trace_id == "live"

    async def test_shared_provider_resolves_via_emitter_provider(self) -> None:
        mem = InMemorySharedMemory()
        await mem.write(content="seed", author="a")
        live_emitter = InMemoryEmitter(trace_id="live")
        provider = SharedMemoryProvider(memory=mem, emitter_provider=lambda: live_emitter)
        assert provider._emitter is live_emitter
        await provider.provide([])
        events = [e for e in live_emitter.events if isinstance(e, SharedMemoryReadEvent)]
        assert len(events) == 1
        assert events[0].trace_id == "live"

    async def test_llm_evaluator_resolves_via_emitter_provider(self) -> None:
        # Mock LLM returns a well-formed JSON evaluation response.
        evaluator_response = LLMResponse(
            content='{"score": 0.9, "reasoning": "looks good", "issues": []}',
            tool_calls=[],
            parsed={"score": 0.9, "reasoning": "looks good", "issues": []},
            usage=make_usage(),
            model="eval-model",
            stop_reason="end_turn",
        )
        client = MockLLMClient([evaluator_response])
        live_emitter = InMemoryEmitter(trace_id="live-eval")
        evaluator = LLMEvaluator(
            llm_client=client,
            criteria="must be accurate",
            emitter_provider=lambda: live_emitter,
        )
        assert evaluator._emitter is live_emitter
        await evaluator.evaluate(
            "some output",
            EvaluationContext(messages=[], task_input="task", attempt_number=1),
        )
        llm_events = [e for e in live_emitter.events if isinstance(e, LLMRequestEvent)]
        assert len(llm_events) >= 1
        assert all(e.trace_id == "live-eval" for e in llm_events)
