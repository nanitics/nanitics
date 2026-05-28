"""Tests for PeerNetwork: construction, budget, events, consultation flows."""

import pytest

from nanitics.composition.multi_agent.peer_network import (
    InvocationBudget,
    PeerBudgetExceededError,
    PeerNetwork,
    PeerSpec,
    PeerTool,
)
from nanitics.infrastructure import (
    LLMResponse,
    MockLLMClient,
)
from nanitics.infrastructure.llm.protocol import ToolCall
from nanitics.infrastructure.observability.events import (
    PeerConsultationEvent,
    PeerNetworkCompleteEvent,
    PeerNetworkStartEvent,
)
from nanitics.safety.cancellation import CancellationToken
from nanitics.strategies import ReActAgent
from tests.testing_helpers import make_emitter, make_response, make_usage


def make_tool_call_response(tool_name: str, arguments: dict) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[
            ToolCall(id="tc1", name=tool_name, arguments=arguments),
        ],
        usage=make_usage(),
        model="test-model",
        stop_reason="tool_use",
    )


def make_spec(
    name: str,
    responses: list[LLMResponse] | None = None,
    description: str = "",
) -> PeerSpec:
    return PeerSpec(
        name=name,
        description=description or f"{name} specialist",
        llm_client=MockLLMClient(responses or [make_response(f"{name} output")]),
        system_prompt=f"You are {name}.",
        tools=[],
    )


class TestPeerNetworkConstruction:
    def test_agents_have_correct_tools(self):
        emitter = make_emitter()
        network = PeerNetwork(
            peers=[
                make_spec("alice"),
                make_spec("bob"),
                make_spec("carol"),
            ],
            emitter=emitter,
        )

        alice = network._registry["alice"]
        bob = network._registry["bob"]
        carol = network._registry["carol"]

        alice_tool_names = [t.name for t in alice._tool_registry.list_schemas()]
        bob_tool_names = [t.name for t in bob._tool_registry.list_schemas()]
        carol_tool_names = [t.name for t in carol._tool_registry.list_schemas()]

        # Alice can consult bob and carol, but not herself
        assert "consult_bob" in alice_tool_names
        assert "consult_carol" in alice_tool_names
        assert "consult_alice" not in alice_tool_names

        # Bob can consult alice and carol
        assert "consult_alice" in bob_tool_names
        assert "consult_carol" in bob_tool_names
        assert "consult_bob" not in bob_tool_names

        # Carol can consult alice and bob
        assert "consult_alice" in carol_tool_names
        assert "consult_bob" in carol_tool_names
        assert "consult_carol" not in carol_tool_names

    def test_duplicate_names_raises(self):
        emitter = make_emitter()
        with pytest.raises(ValueError, match="Duplicate"):
            PeerNetwork(
                peers=[make_spec("alice"), make_spec("alice")],
                emitter=emitter,
            )

    def test_system_prompt_augmented(self):
        emitter = make_emitter()
        network = PeerNetwork(
            peers=[
                make_spec("alice", description="Alice the expert"),
                make_spec("bob", description="Bob the analyst"),
            ],
            emitter=emitter,
        )

        alice = network._registry["alice"]
        # Alice's prompt should mention bob but not herself
        assert "bob" in alice._system_prompt.lower()
        assert "Bob the analyst" in alice._system_prompt
        assert "Available Peers" in alice._system_prompt


class TestPeerToolLazyResolution:
    async def test_tool_resolves_agent_at_execute_time(self):
        emitter = make_emitter()
        registry: dict[str, ReActAgent] = {}
        budget = InvocationBudget(10)
        consulted: set[str] = set()

        # Create tool before agent exists in registry
        tool = PeerTool(
            peer_name="bob",
            peer_description="Bob specialist",
            caller_name="alice",
            registry=registry,
            budget=budget,
            emitter=emitter,
            consulted=consulted,
        )

        # Register agent after tool creation
        registry["bob"] = ReActAgent(
            name="bob",
            llm_client=MockLLMClient([make_response("bob's answer")]),
            emitter=emitter,
            system_prompt="You are bob.",
            tools=[],
        )

        result = await tool.execute(message="What do you think?")
        assert result.content == "bob's answer"

    def test_schema_name(self):
        emitter = make_emitter()
        tool = PeerTool(
            peer_name="financial_analyst",
            peer_description="Expert in finance",
            caller_name="lead",
            registry={},
            budget=InvocationBudget(10),
            emitter=emitter,
            consulted=set(),
        )
        assert tool.schema.name == "consult_financial_analyst"
        assert tool.schema.description == "Expert in finance"
        assert "message" in tool.schema.parameters["properties"]


class TestInvocationBudget:
    def test_consume_decrements(self):
        budget = InvocationBudget(3)
        assert budget.remaining == 3
        assert budget.used == 0

        n = budget.consume()
        assert n == 1
        assert budget.remaining == 2
        assert budget.used == 1

    def test_exhaustion_raises(self):
        budget = InvocationBudget(1)
        budget.consume()

        with pytest.raises(PeerBudgetExceededError):
            budget.consume()

    def test_error_message_is_descriptive(self):
        budget = InvocationBudget(2)
        budget.consume()
        budget.consume()

        with pytest.raises(PeerBudgetExceededError) as exc_info:
            budget.consume()
        assert "2/2" in str(exc_info.value)
        assert exc_info.value.invocations_used == 2
        assert exc_info.value.max_invocations == 2


class TestEventEmission:
    async def test_start_and_complete_events(self):
        emitter = make_emitter()
        network = PeerNetwork(
            peers=[
                make_spec("alice"),
                make_spec("bob"),
            ],
            emitter=emitter,
            max_invocations=10,
        )

        await network.run("alice", "Analyze this")

        start_events = [e for e in emitter.events if isinstance(e, PeerNetworkStartEvent)]
        assert len(start_events) == 1
        assert start_events[0].task == "Analyze this"
        assert start_events[0].entry_agent == "alice"
        assert set(start_events[0].peer_names) == {"alice", "bob"}
        assert start_events[0].max_invocations == 10

        complete_events = [e for e in emitter.events if isinstance(e, PeerNetworkCompleteEvent)]
        assert len(complete_events) == 1
        assert complete_events[0].entry_agent == "alice"
        assert complete_events[0].termination_reason == "complete"

    async def test_consultation_event_emitted(self):
        emitter = make_emitter()

        # Alice calls consult_bob, then produces final answer
        # Bob just answers directly
        alice_llm = MockLLMClient(
            [
                make_tool_call_response("consult_bob", {"message": "Help me"}),
                make_response("Final answer from alice"),
            ]
        )
        bob_llm = MockLLMClient([make_response("Bob's input")])

        network = PeerNetwork(
            peers=[
                PeerSpec(
                    name="alice",
                    description="Leader",
                    llm_client=alice_llm,
                    system_prompt="You are alice.",
                    tools=[],
                ),
                PeerSpec(
                    name="bob",
                    description="Analyst",
                    llm_client=bob_llm,
                    system_prompt="You are bob.",
                    tools=[],
                ),
            ],
            emitter=emitter,
        )

        result = await network.run("alice", "Do analysis")

        assert result.output == "Final answer from alice"

        consultation_events = [e for e in emitter.events if isinstance(e, PeerConsultationEvent)]
        assert len(consultation_events) == 1
        assert consultation_events[0].from_agent == "alice"
        assert consultation_events[0].to_agent == "bob"
        assert consultation_events[0].message == "Help me"
        assert consultation_events[0].consultation_number == 1

        complete_events = [e for e in emitter.events if isinstance(e, PeerNetworkCompleteEvent)]
        assert complete_events[0].total_consultations == 1
        assert complete_events[0].agents_consulted == ["bob"]


class TestCancellation:
    async def test_shared_cancellation_token(self):
        emitter = make_emitter()
        token = CancellationToken()

        network = PeerNetwork(
            peers=[make_spec("alice"), make_spec("bob")],
            emitter=emitter,
            cancellation_token=token,
        )

        # Both agents should share the same token
        assert network._registry["alice"]._cancellation_token is token
        assert network._registry["bob"]._cancellation_token is token


class TestConsultationFlow:
    async def test_agent_a_consults_agent_b(self):
        emitter = make_emitter()

        alice_llm = MockLLMClient(
            [
                make_tool_call_response("consult_bob", {"message": "What's the risk?"}),
                make_response("Risk is low based on bob's analysis"),
            ]
        )
        bob_llm = MockLLMClient([make_response("Risk assessment: minimal exposure")])

        network = PeerNetwork(
            peers=[
                PeerSpec(
                    name="alice",
                    description="Lead",
                    llm_client=alice_llm,
                    system_prompt="You are alice.",
                    tools=[],
                ),
                PeerSpec(
                    name="bob",
                    description="Risk analyst",
                    llm_client=bob_llm,
                    system_prompt="You are bob.",
                    tools=[],
                ),
            ],
            emitter=emitter,
        )

        result = await network.run("alice", "Evaluate deal")

        assert result.output == "Risk is low based on bob's analysis"
        assert result.total_steps == 2

    async def test_transitive_consultation(self):
        """A consults B, B consults C — chain works within budget."""
        emitter = make_emitter()

        alice_llm = MockLLMClient(
            [
                make_tool_call_response("consult_bob", {"message": "Check financials"}),
                make_response("Final synthesis"),
            ]
        )
        bob_llm = MockLLMClient(
            [
                make_tool_call_response("consult_carol", {"message": "Verify numbers"}),
                make_response("Financials verified by carol"),
            ]
        )
        carol_llm = MockLLMClient([make_response("Numbers check out")])

        network = PeerNetwork(
            peers=[
                PeerSpec(name="alice", description="Lead", llm_client=alice_llm, system_prompt="Alice.", tools=[]),
                PeerSpec(name="bob", description="Financial", llm_client=bob_llm, system_prompt="Bob.", tools=[]),
                PeerSpec(name="carol", description="Auditor", llm_client=carol_llm, system_prompt="Carol.", tools=[]),
            ],
            emitter=emitter,
            max_invocations=10,
        )

        result = await network.run("alice", "Full analysis")
        assert result.output == "Final synthesis"

        consultation_events = [e for e in emitter.events if isinstance(e, PeerConsultationEvent)]
        assert len(consultation_events) == 2
        assert consultation_events[0].from_agent == "alice"
        assert consultation_events[0].to_agent == "bob"
        assert consultation_events[1].from_agent == "bob"
        assert consultation_events[1].to_agent == "carol"

        complete = [e for e in emitter.events if isinstance(e, PeerNetworkCompleteEvent)]
        assert complete[0].total_consultations == 2
        assert sorted(complete[0].agents_consulted) == ["bob", "carol"]

    async def test_invalid_agent_name_raises(self):
        emitter = make_emitter()
        network = PeerNetwork(
            peers=[make_spec("alice")],
            emitter=emitter,
        )

        with pytest.raises(ValueError, match="nonexistent"):
            await network.run("nonexistent", "task")


@pytest.mark.parametrize("value", [0, -1])
def test_invocation_budget_rejects_non_positive(value: int) -> None:
    with pytest.raises(ValueError, match="max_invocations must be positive"):
        InvocationBudget(max_invocations=value)


def _consult_tool_names(agent: ReActAgent) -> list[str]:
    return [schema.name for schema in agent._tool_registry.list_schemas() if schema.name.startswith("consult_")]


class TestAllowedPeers:
    def test_default_allowed_peers_excludes_self(self) -> None:
        emitter = make_emitter()
        network = PeerNetwork(
            peers=[
                make_spec("alice"),
                make_spec("bob"),
                make_spec("carol"),
            ],
            emitter=emitter,
        )

        for name in ("alice", "bob", "carol"):
            consult_tools = _consult_tool_names(network._registry[name])
            assert f"consult_{name}" not in consult_tools
            expected_others = {f"consult_{other}" for other in ("alice", "bob", "carol") if other != name}
            assert set(consult_tools) == expected_others

    def test_explicit_allowed_peers_restricts_graph(self) -> None:
        emitter = make_emitter()
        network = PeerNetwork(
            peers=[
                PeerSpec(
                    name="alice",
                    description="Alice",
                    llm_client=MockLLMClient([make_response("alice output")]),
                    system_prompt="You are alice.",
                    tools=[],
                    allowed_peers=["bob"],
                ),
                make_spec("bob"),
                make_spec("carol"),
            ],
            emitter=emitter,
        )

        alice_tools = _consult_tool_names(network._registry["alice"])
        assert "consult_bob" in alice_tools
        assert "consult_carol" not in alice_tools
        assert "consult_alice" not in alice_tools

    def test_allowed_peers_self_reference_raises(self) -> None:
        emitter = make_emitter()
        with pytest.raises(ValueError, match=r"'alice'.*self-consultation"):
            PeerNetwork(
                peers=[
                    PeerSpec(
                        name="alice",
                        description="Alice",
                        llm_client=MockLLMClient([make_response("alice output")]),
                        system_prompt="You are alice.",
                        tools=[],
                        allowed_peers=["alice"],
                    ),
                    make_spec("bob"),
                ],
                emitter=emitter,
            )

    def test_allowed_peers_unknown_name_raises(self) -> None:
        emitter = make_emitter()
        with pytest.raises(ValueError) as exc_info:
            PeerNetwork(
                peers=[
                    PeerSpec(
                        name="alice",
                        description="Alice",
                        llm_client=MockLLMClient([make_response("alice output")]),
                        system_prompt="You are alice.",
                        tools=[],
                        allowed_peers=["nonexistent"],
                    ),
                    make_spec("bob"),
                ],
                emitter=emitter,
            )
        msg = str(exc_info.value)
        assert "nonexistent" in msg
        assert "alice" in msg
        assert "bob" in msg

    def test_allowed_peers_empty_list(self) -> None:
        emitter = make_emitter()
        network = PeerNetwork(
            peers=[
                PeerSpec(
                    name="alice",
                    description="Alice",
                    llm_client=MockLLMClient([make_response("alice output")]),
                    system_prompt="You are alice.",
                    tools=[],
                    allowed_peers=[],
                ),
                make_spec("bob"),
            ],
            emitter=emitter,
        )

        alice = network._registry["alice"]
        assert _consult_tool_names(alice) == []
        # Prompt stays clean — no "Available Peers" block for a leaf consultant.
        assert "Available Peers" not in alice._system_prompt

    def test_prompt_roster_reflects_allowed_peers(self) -> None:
        emitter = make_emitter()
        network = PeerNetwork(
            peers=[
                PeerSpec(
                    name="alice",
                    description="Alice",
                    llm_client=MockLLMClient([make_response("alice output")]),
                    system_prompt="You are alice.",
                    tools=[],
                    allowed_peers=["bob"],
                ),
                make_spec("bob", description="Bob the analyst"),
                make_spec("carol", description="Carol the auditor"),
            ],
            emitter=emitter,
        )

        alice = network._registry["alice"]
        assert "Bob the analyst" in alice._system_prompt
        assert "Carol the auditor" not in alice._system_prompt
        assert "carol" not in alice._system_prompt.lower().split("## available peers")[1]

    def test_no_consult_self_across_all_tests_in_file(self) -> None:
        """Hostile-case sweep: under every construction path exercised in
        this module, no peer ever has a ``consult_<self>`` tool in its
        registry. Re-exercise the principal constructions and verify.
        """
        emitter = make_emitter()

        constructions: list[PeerNetwork] = [
            PeerNetwork(
                peers=[make_spec("alice"), make_spec("bob"), make_spec("carol")],
                emitter=emitter,
            ),
            PeerNetwork(
                peers=[
                    PeerSpec(
                        name="alice",
                        description="Alice",
                        llm_client=MockLLMClient([make_response("alice output")]),
                        system_prompt="You are alice.",
                        tools=[],
                        allowed_peers=["bob"],
                    ),
                    make_spec("bob"),
                    make_spec("carol"),
                ],
                emitter=emitter,
            ),
            PeerNetwork(
                peers=[
                    PeerSpec(
                        name="alice",
                        description="Alice",
                        llm_client=MockLLMClient([make_response("alice output")]),
                        system_prompt="You are alice.",
                        tools=[],
                        allowed_peers=[],
                    ),
                    make_spec("bob"),
                ],
                emitter=emitter,
            ),
        ]

        for network in constructions:
            for name, agent in network._registry.items():
                assert f"consult_{name}" not in _consult_tool_names(agent), (
                    f"{name} must not have consult_{name} in its tool registry"
                )


# ── thread_key propagation ────────────────────────────────


class TestPeerNetworkThreadKey:
    """Per-peer-identity threading: each peer carries its own thread,
    keyed by ``PeerSpec.thread_key``. Every consultation of a peer —
    whether entry-point or via ``consult_<peer>`` — uses the peer's key,
    so the peer accumulates its prior turns across consultations.

    Per-pair / per-network scoping is deferred until a real consumer
    asks; the default position is per-peer-identity.
    """

    def test_thread_key_defaults_to_none_on_peerspec(self) -> None:
        spec = make_spec("alice")
        assert spec.thread_key is None

    def test_thread_key_field_accepted(self) -> None:
        spec = PeerSpec(
            name="alice",
            description="alice",
            llm_client=MockLLMClient([make_response("hi")]),
            system_prompt="you are alice",
            tools=[],
            thread_key="alice-thread",
        )
        assert spec.thread_key == "alice-thread"

    def test_peer_tool_carries_peer_thread_key(self) -> None:
        emitter = make_emitter()
        bob_spec = PeerSpec(
            name="bob",
            description="bob",
            llm_client=MockLLMClient([make_response("bob output")]),
            system_prompt="you are bob",
            tools=[],
            thread_key="bob-thread",
        )
        network = PeerNetwork(
            peers=[make_spec("alice"), bob_spec],
            emitter=emitter,
        )
        # Alice has a consult_bob tool; that PeerTool should carry
        # bob's thread_key, not alice's.
        alice = network._registry["alice"]
        consult_bob = alice._tool_registry.get("consult_bob")
        assert isinstance(consult_bob, PeerTool)
        assert consult_bob._thread_key == "bob-thread"

    async def test_entry_run_uses_peerspec_thread_key_by_default(self) -> None:
        from nanitics.composition import InMemoryThreadStore

        emitter = make_emitter()
        store = InMemoryThreadStore()
        alice_spec = PeerSpec(
            name="alice",
            description="alice",
            llm_client=MockLLMClient([make_response("alice answer")]),
            system_prompt="you are alice",
            tools=[],
            thread_key="alice-thread",
            allowed_peers=[],
        )
        network = PeerNetwork(
            peers=[alice_spec, make_spec("bob")],
            emitter=emitter,
            thread_store=store,
        )
        await network.run("alice", "say hi")
        loaded = await store.load("alice-thread")
        assert any(m.role == "assistant" for m in loaded)

    async def test_run_thread_key_overrides_peerspec_key(self) -> None:
        from nanitics.composition import InMemoryThreadStore

        emitter = make_emitter()
        store = InMemoryThreadStore()
        alice_spec = PeerSpec(
            name="alice",
            description="alice",
            llm_client=MockLLMClient([make_response("alice answer")]),
            system_prompt="you are alice",
            tools=[],
            thread_key="alice-thread",
            allowed_peers=[],
        )
        network = PeerNetwork(
            peers=[alice_spec, make_spec("bob")],
            emitter=emitter,
            thread_store=store,
        )
        await network.run("alice", "say hi", thread_key="session-42")
        # The override key got the writes, not the default per-peer key.
        assert len(await store.load("session-42")) > 0
        assert len(await store.load("alice-thread")) == 0

    async def test_per_peer_accumulation_across_two_network_runs(self) -> None:
        """Two separate `network.run` calls hit the entry peer twice; with
        per-peer-identity, the second call sees the first's turns."""
        from nanitics.composition import InMemoryThreadStore

        emitter = make_emitter()
        store = InMemoryThreadStore()
        alice_spec = PeerSpec(
            name="alice",
            description="alice",
            llm_client=MockLLMClient([make_response("first answer"), make_response("second answer")]),
            system_prompt="you are alice",
            tools=[],
            thread_key="alice-thread",
            allowed_peers=[],
        )
        network = PeerNetwork(
            peers=[alice_spec, make_spec("bob")],
            emitter=emitter,
            thread_store=store,
        )
        await network.run("alice", "q1")
        await network.run("alice", "q2")
        loaded = await store.load("alice-thread")
        # Two user inputs + two assistant outputs at minimum.
        assert sum(1 for m in loaded if m.role == "user") >= 2
        assert sum(1 for m in loaded if m.role == "assistant") >= 2
