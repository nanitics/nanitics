from __future__ import annotations

from enum import StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from nanitics.infrastructure.llm.protocol import LLMClient, Message
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import (
    TreeSearchCompleteEvent,
    TreeSearchNodeCreatedEvent,
    TreeSearchNodeEvaluatedEvent,
    TreeSearchNodePrunedEvent,
    Usage,
)
from nanitics.safety.cancellation import CancellationToken
from nanitics.strategies.agents.base import Agent, AgentInput, AgentResult, _input_to_text
from nanitics.strategies.agents.context import ContextManagement, ContextProvider
from nanitics.strategies.agents.evaluation import EvaluationVerdict, OutputEvaluator
from nanitics.strategies.prompts.builder import SystemPromptContributor


class SearchStrategy(StrEnum):
    """Strategy for selecting which nodes to expand in tree search."""

    BFS = "bfs"
    """Breadth-first: expand all nodes at the shallowest depth first."""
    DFS = "dfs"
    """Depth-first: expand the deepest node first."""
    BEST_FIRST = "best_first"
    """Greedy: expand the highest-scored node first."""


class ThoughtNode(BaseModel):
    """A node in the Tree-of-Thought search tree.

    Attributes:
        id: Unique node identifier.
        content: The reasoning content at this node.
        parent_id: Parent node ID, or ``None`` for the root.
        children_ids: IDs of child nodes.
        depth: Distance from the root node.
        score: Evaluation score assigned by the node evaluator.
        is_terminal: Whether this node represents a complete solution.
        is_pruned: Whether the evaluator rejected this node.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    parent_id: str | None = None
    children_ids: list[str] = Field(default_factory=list)
    depth: int = 0
    score: float | None = None
    is_terminal: bool = False
    is_pruned: bool = False


class _Candidate(BaseModel):
    """One LLM-generated continuation in a tree-of-thought expansion."""

    model_config = ConfigDict(frozen=True)

    reasoning: str
    is_complete: bool


class _GenerationResponse(BaseModel):
    """Structured schema the LLM returns when expanding a thought node."""

    model_config = ConfigDict(frozen=True)

    candidates: list[_Candidate]


class TreeOfThoughtAgent(Agent):
    """Agent that explores multiple reasoning paths via tree search.

    On each step, the agent generates several candidate continuations
    (controlled by ``branching_factor``), evaluates each with a node
    evaluator, and selects which branches to expand next based on the
    search strategy. Nodes that the evaluator rejects are pruned.

    Does not support tool calls. For tree search with tools, use
    ``LATSAgent``.

    Args:
        name: Identifies the agent in events and traces.
        llm_client: Language model to use.
        emitter: Event emitter for observability.
        system_prompt: Base system prompt text.
        node_evaluator: Evaluates each candidate node. Score determines
            expansion priority (for ``BEST_FIRST``). ``REJECT`` verdict
            prunes the node.
        search_strategy: How to select nodes for expansion.
        branching_factor: Number of candidate continuations per
            expansion.
        max_depth: Maximum tree depth.
        max_nodes: Total node budget.
        min_depth: Minimum depth before terminal nodes are allowed.
            Below this depth, candidates are forced non-terminal.
            Defaults to 0 (no minimum).
        cancellation_token: External cancellation signal.
        context_manager: Context window management.
        context_providers: Inject context before each LLM call.
        prompt_contributors: Additional system prompt sections.
    """

    def __init__(
        self,
        *,
        name: str,
        llm_client: LLMClient,
        emitter: EventEmitter,
        system_prompt: str,
        node_evaluator: OutputEvaluator,
        search_strategy: SearchStrategy = SearchStrategy.BFS,
        branching_factor: int = 3,
        max_depth: int = 5,
        max_nodes: int = 50,
        min_depth: int = 0,
        cancellation_token: CancellationToken | None = None,
        context_manager: ContextManagement | None = None,
        context_providers: list[ContextProvider] | None = None,
        prompt_contributors: list[SystemPromptContributor] | None = None,
    ) -> None:
        if min_depth > max_depth:
            raise ValueError(f"min_depth ({min_depth}) must be <= max_depth ({max_depth})")
        super().__init__(
            name=name,
            llm_client=llm_client,
            emitter=emitter,
            system_prompt=system_prompt,
            cancellation_token=cancellation_token,
            context_manager=context_manager,
            context_providers=context_providers,
            output_evaluator=node_evaluator,
            prompt_contributors=prompt_contributors,
        )
        self._search_strategy = search_strategy
        self._branching_factor = branching_factor
        self._max_depth = max_depth
        self._min_depth = min_depth
        self._max_nodes = max_nodes
        self._nodes: dict[str, ThoughtNode] = {}

    def _agent_type(self) -> str:
        return "tree_of_thought"

    def _get_path_to_root(self, node_id: str) -> list[ThoughtNode]:
        path: list[ThoughtNode] = []
        current_id: str | None = node_id
        while current_id is not None:
            node = self._nodes[current_id]
            path.append(node)
            current_id = node.parent_id
        path.reverse()
        return path

    def _expandable_nodes(self) -> list[ThoughtNode]:
        return [
            n
            for n in self._nodes.values()
            if not n.is_pruned and not n.is_terminal and n.depth < self._max_depth and len(n.children_ids) == 0
        ]

    def _select_nodes(self) -> list[ThoughtNode]:
        expandable = self._expandable_nodes()
        if not expandable:
            return []

        if self._search_strategy == SearchStrategy.BFS:
            min_depth = min(n.depth for n in expandable)
            return [n for n in expandable if n.depth == min_depth]

        if self._search_strategy == SearchStrategy.DFS:
            return [max(expandable, key=lambda n: n.depth)]

        # BEST_FIRST
        return [max(expandable, key=lambda n: n.score if n.score is not None else float("-inf"))]

    def _add_node(self, node: ThoughtNode) -> None:
        self._nodes[node.id] = node

    def _update_node(self, node_id: str, **kwargs: object) -> ThoughtNode:
        old = self._nodes[node_id]
        updated = old.model_copy(update=kwargs)
        self._nodes[node_id] = updated
        return updated

    def _build_completion_guidance(self, child_depth: int) -> str:
        if child_depth < self._min_depth:
            return (
                f"You are at step {child_depth} of up to {self._max_depth} reasoning steps. "
                "Focus on developing distinct intermediate reasoning steps. Break down the problem, "
                "explore assumptions, or develop sub-arguments. Do not mark any continuation as complete yet."
            )
        return (
            f"You are at step {child_depth} of up to {self._max_depth} reasoning steps. "
            "Mark a continuation as complete if it represents a fully developed final answer. "
            "If the reasoning still has gaps or unexplored angles, continue developing it."
        )

    async def _generate_candidates(self, node: ThoughtNode, task_input: str) -> list[ThoughtNode]:
        path = self._get_path_to_root(node.id)

        messages: list[Message] = []
        for i, step_node in enumerate(path):
            role: Literal["user", "assistant"] = "user" if i % 2 == 0 else "assistant"
            messages.append(Message(role=role, content=step_node.content))

        child_depth = node.depth + 1
        completion_guidance = self._build_completion_guidance(child_depth)

        # Ensure last message is from user (for LLM call)
        if messages and messages[-1].role == "assistant":
            messages.append(
                Message(
                    role="user",
                    content=(
                        f"Generate exactly {self._branching_factor} distinct continuations of this reasoning. "
                        "Each continuation should explore a different approach or direction. "
                        f"{completion_guidance}"
                    ),
                )
            )
        else:
            # Path is a single root node (user message)
            messages.append(
                Message(
                    role="assistant",
                    content="I'll explore multiple reasoning paths.",
                )
            )
            messages.append(
                Message(
                    role="user",
                    content=(
                        f"Generate exactly {self._branching_factor} distinct continuations of this reasoning. "
                        "Each continuation should explore a different approach or direction. "
                        f"{completion_guidance}"
                    ),
                )
            )

        response = await self._call_llm(messages, output_schema=_GenerationResponse)

        gen_response = _GenerationResponse.model_validate_json(response.content or "{}")

        children: list[ThoughtNode] = []
        for candidate in gen_response.candidates:
            terminal_suppressed = False
            is_terminal = candidate.is_complete
            if is_terminal and child_depth < self._min_depth:
                is_terminal = False
                terminal_suppressed = True

            child = ThoughtNode(
                content=candidate.reasoning,
                parent_id=node.id,
                depth=child_depth,
                is_terminal=is_terminal,
            )
            self._add_node(child)
            children.append(child)

            # Update parent's children_ids
            self._update_node(
                node.id,
                children_ids=[*self._nodes[node.id].children_ids, child.id],
            )

            self._emitter.emit(
                TreeSearchNodeCreatedEvent(
                    trace_id=self._emitter.trace_id,
                    span_id=self._emitter.span_id,
                    parent_span_id=self._emitter.parent_span_id,
                    node_id=child.id,
                    parent_id=node.id,
                    depth=child.depth,
                    content=child.content,
                    node_type="thought",
                    is_terminal=child.is_terminal,
                    terminal_suppressed=terminal_suppressed if terminal_suppressed else None,
                )
            )

        return children

    async def _evaluate_node(self, node: ThoughtNode, task_input: str) -> ThoughtNode:
        eval_result = await self._evaluate_output(
            node.content,
            task_input,
            [],  # No message history needed for tree node evaluation
            0,
            depth=node.depth,
            max_depth=self._max_depth,
            total_nodes_explored=len(self._nodes),
        )

        score = eval_result.score if eval_result.score is not None else 0.0
        updated = self._update_node(node.id, score=score)

        self._emitter.emit(
            TreeSearchNodeEvaluatedEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                node_id=node.id,
                score=score,
                is_terminal=node.is_terminal,
            )
        )

        if eval_result.verdict == EvaluationVerdict.REJECT:
            updated = self._update_node(node.id, is_pruned=True)
            self._emitter.emit(
                TreeSearchNodePrunedEvent(
                    trace_id=self._emitter.trace_id,
                    span_id=self._emitter.span_id,
                    parent_span_id=self._emitter.parent_span_id,
                    node_id=node.id,
                    reason="evaluation_rejected",
                )
            )

        return updated

    async def _execute(self, input: AgentInput) -> AgentResult:
        self._nodes.clear()
        usages: list[Usage] = []
        step_number = 0
        accepted_terminal_ids: list[str] = []
        task_text = _input_to_text(input)

        # Create root node
        root = ThoughtNode(content=task_text, depth=0)
        self._add_node(root)

        self._emitter.emit(
            TreeSearchNodeCreatedEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                node_id=root.id,
                parent_id=None,
                depth=0,
                content=task_text,
                node_type="thought",
            )
        )

        termination_reason = "complete"

        while True:
            if self._is_cancelled:
                self._emit_safety_cancellation(step_number)
                termination_reason = "cancelled"
                break

            if len(self._nodes) >= self._max_nodes:
                termination_reason = "max_nodes"
                break

            selected = self._select_nodes()
            if not selected:
                termination_reason = "no_expandable_nodes"
                break

            step_number += 1

            with self._emitter.span(f"step-{step_number}"):
                for node in selected:
                    if len(self._nodes) >= self._max_nodes:
                        break

                    children = await self._generate_candidates(node, task_text)

                    for child in children:
                        updated_child = await self._evaluate_node(child, task_text)

                        if updated_child.is_terminal and not updated_child.is_pruned:
                            accepted_terminal_ids.append(updated_child.id)

                self._emit_step(step_number)

        # Select best from accepted terminals, or fall back to best node
        if accepted_terminal_ids:
            best_node = self._select_best_node(candidate_ids=set(accepted_terminal_ids))
        else:
            best_node = self._select_best_node()
        max_depth = max(n.depth for n in self._nodes.values()) if self._nodes else 0

        self._emitter.emit(
            TreeSearchCompleteEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                total_nodes=len(self._nodes),
                max_depth_reached=max_depth,
                selected_node_id=best_node.id if best_node else "",
                termination_reason=termination_reason,
                search_strategy=self._search_strategy.value,
                accepted_count=len(accepted_terminal_ids),
            )
        )

        total_usage = self._aggregate_usage(usages)
        return AgentResult(
            output=best_node.content if best_node else None,
            total_steps=step_number,
            termination_reason=termination_reason,
            messages=[],
            usage=total_usage,
        )

    def _select_best_node(self, *, candidate_ids: set[str] | None = None) -> ThoughtNode | None:
        if not self._nodes:
            return None

        pool = (
            [self._nodes[nid] for nid in candidate_ids if nid in self._nodes]
            if candidate_ids is not None
            else list(self._nodes.values())
        )

        if candidate_ids is not None and pool:
            return max(pool, key=lambda n: n.score if n.score is not None else float("-inf"))

        # Prefer terminal nodes
        terminals = [n for n in pool if n.is_terminal and not n.is_pruned]
        if terminals:
            return max(terminals, key=lambda n: n.score if n.score is not None else float("-inf"))

        # Fall back to highest-scoring non-root, non-pruned node
        candidates = [n for n in pool if not n.is_pruned and n.depth > 0]
        if candidates:
            return max(candidates, key=lambda n: n.score if n.score is not None else float("-inf"))

        # Last resort: root
        return next(iter(self._nodes.values()))

    @staticmethod
    def _aggregate_usage(usages: list[Usage]) -> Usage:
        return Usage(
            input_tokens=sum(u.input_tokens for u in usages),
            output_tokens=sum(u.output_tokens for u in usages),
        )
