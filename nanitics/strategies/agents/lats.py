from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from nanitics.infrastructure.llm.instrumented import InstrumentedLLMClient
from nanitics.infrastructure.llm.protocol import LLMClient, Message, ToolCall
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import (
    MCTSBackpropagationEvent,
    MCTSIterationEvent,
    ToolInfo,
    TreeSearchCompleteEvent,
    TreeSearchNodeCreatedEvent,
    TreeSearchNodeEvaluatedEvent,
    TreeSearchNodePrunedEvent,
    Usage,
)
from nanitics.safety.cancellable_dispatch import RunCancelled, run_cancellable
from nanitics.safety.cancellation import CancellationToken
from nanitics.strategies.agents.base import Agent, AgentInput, AgentResult, _input_to_text
from nanitics.strategies.agents.context import ContextManagement, ContextProvider
from nanitics.strategies.agents.evaluation import EvaluationVerdict, OutputEvaluator
from nanitics.strategies.prompts.builder import SystemPromptContributor
from nanitics.strategies.tools import Tool, ToolRegistry

if TYPE_CHECKING:
    from nanitics.capabilities.memory.episodic import EpisodeStore


class ActionNode(BaseModel):
    """A node in the LATS Monte Carlo Tree Search tree.

    Each node represents a state in the search: a thought, an optional
    tool action with its observation, or a terminal answer.

    Attributes:
        id: Unique node identifier.
        parent_id: Parent node ID, or ``None`` for the root.
        children_ids: IDs of child nodes.
        depth: Distance from the root node.
        thought: The LLM's reasoning at this node.
        action: Tool name called, if any.
        action_input: Tool arguments, if any.
        observation: Tool result, if any.
        is_terminal: Whether this is a final answer node.
        terminal_output: Final answer text, if terminal.
        value: Accumulated reward from backpropagation.
        visit_count: Number of times this node was visited during
            MCTS selection.
        is_failed: Whether this node is disqualified from further
            exploration. Set when tool execution raises, when the node
            evaluator returns ``REJECT``, or when every child of the
            node is itself failed (hard-prune cascade). Selection skips
            these nodes; ``_select_best_node`` excludes them from the
            final answer.
        metadata: Structured metadata copied from ``ToolResult.metadata``
            on the dispatch that produced this node. Carried onto the
            ``tool_result`` ``Message.metadata`` when the node's
            trajectory is rebuilt by ``_build_trajectory_messages``.
            ``None`` when no tool dispatched (root, thought-only nodes,
            terminal nodes) or when the tool returned no metadata.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    parent_id: str | None = None
    children_ids: list[str] = Field(default_factory=list)
    depth: int = 0
    thought: str = ""
    action: str | None = None
    action_input: dict[str, Any] | None = None
    observation: str | None = None
    is_terminal: bool = False
    terminal_output: str | None = None
    value: float = 0.0
    visit_count: int = 0
    is_failed: bool = False
    error_message: str | None = None
    metadata: dict[str, Any] | None = None


_REFLECTION_SYSTEM_PROMPT = (
    "Analyze a failed search over multiple action trajectories for a task. "
    "Produce a reflection that will help on future similar tasks.\n\n"
    "Focus on:\n"
    "- What approaches were tried across different branches\n"
    "- Why they failed\n"
    "- What concrete alternative strategies to try\n"
    "- What assumptions were wrong\n\n"
    "Be specific and actionable. Avoid generic advice."
)


class LATSAgent(Agent):
    """Language Agent Tree Search — MCTS-based agent with tool use.

    The most powerful (and most expensive) agent type. Combines Monte
    Carlo Tree Search with tool use, evaluation-guided expansion,
    backpropagation, and optional episodic memory for cross-run learning.

    Each iteration: select a leaf via UCB1, expand it into multiple
    branches (each an LLM response that may call a tool or produce a
    final answer), evaluate each branch, and backpropagate the scores
    up the tree.

    Args:
        name: Identifies the agent in events and traces.
        llm_client: Language model to use.
        emitter: Event emitter for observability.
        system_prompt: Base system prompt text.
        tools: Tools available to the agent.
        node_evaluator: Scores each node to guide the search.
            ``REJECT`` verdict marks nodes as failed.
        max_iterations: Maximum MCTS iterations.
        max_depth: Maximum tree depth.
        branching_factor: Children generated per expansion.
        min_depth: Minimum depth before terminal nodes are allowed.
            Below this depth, non-tool-call responses are treated as
            intermediate thoughts rather than final answers.
            Defaults to 0 (no minimum).
        terminal_depth: Depth at which tools are withheld to force
            terminal text-only responses. Must satisfy
            ``min_depth <= terminal_depth <= max_depth``.
            Default None disables forced termination.
        exploration_constant: UCB1 exploration weight. Higher values
            favor exploration over exploitation.
        episode_store: Optional episodic memory for cross-run learning.
            Past episodes are retrieved as context; results are recorded
            as new episodes.
        cancellation_token: External cancellation signal.
        context_manager: Context window management.
        context_providers: Inject context before each LLM call.
        prompt_contributors: Additional system prompt sections.
        tool_state: Per-run state dict injected into tools via
            ``ToolContext``.
    """

    def __init__(
        self,
        *,
        name: str,
        llm_client: LLMClient,
        emitter: EventEmitter,
        system_prompt: str,
        tools: Sequence[Tool],
        node_evaluator: OutputEvaluator,
        max_iterations: int = 20,
        max_depth: int = 10,
        branching_factor: int = 3,
        exploration_constant: float = 1.414,
        min_depth: int = 0,
        terminal_depth: int | None = None,
        episode_store: EpisodeStore | None = None,
        cancellation_token: CancellationToken | None = None,
        context_manager: ContextManagement | None = None,
        context_providers: list[ContextProvider] | None = None,
        prompt_contributors: list[SystemPromptContributor] | None = None,
        tool_state: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> None:
        if run_id is not None:
            tool_state = dict(tool_state) if tool_state else {}
            tool_state.setdefault("run_id", run_id)
        if min_depth > max_depth:
            raise ValueError(f"min_depth ({min_depth}) must be <= max_depth ({max_depth})")
        if terminal_depth is not None:
            if terminal_depth < min_depth:
                raise ValueError(f"terminal_depth ({terminal_depth}) must be >= min_depth ({min_depth})")
            if terminal_depth > max_depth:
                raise ValueError(f"terminal_depth ({terminal_depth}) must be <= max_depth ({max_depth})")
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
        self._tool_registry = ToolRegistry(
            tool_state=tool_state,
            emitter_provider=lambda: self._emitter,
        )
        self._tool_registry.register_all(tools)
        self._max_iterations = max_iterations
        self._max_depth = max_depth
        self._min_depth = min_depth
        self._terminal_depth = terminal_depth
        self._branching_factor = branching_factor
        self._exploration_constant = exploration_constant
        self._episode_store = episode_store
        self._nodes: dict[str, ActionNode] = {}
        self._root_id: str = ""

    def _agent_type(self) -> str:
        return "lats"

    def _active_capabilities(self) -> list[str]:
        caps = super()._active_capabilities()
        caps.append("tool_use")
        if self._episode_store is not None:
            caps.append("episodic_memory")
        return caps

    def update_tool_state(self, key: str, value: Any) -> None:
        self._tool_registry.update_state(key, value)

    def _get_tools_available(self) -> list[str]:
        return [s.name for s in self._tool_registry.list_schemas()]

    def _get_tool_schemas(self) -> list[ToolInfo]:
        return [
            ToolInfo(
                name=s.name,
                description=s.description,
                requires_approval=s.requires_approval,
            )
            for s in self._tool_registry.list_schemas()
        ]

    # ── Tree Management ──────────────────────────────────────

    def _add_node(self, node: ActionNode) -> None:
        self._nodes[node.id] = node

    def _update_node(self, node_id: str, **kwargs: object) -> ActionNode:
        old = self._nodes[node_id]
        updated = old.model_copy(update=kwargs)
        self._nodes[node_id] = updated
        return updated

    def _get_path_to_root(self, node_id: str) -> list[ActionNode]:
        path: list[ActionNode] = []
        current_id: str | None = node_id
        while current_id is not None:
            node = self._nodes[current_id]
            path.append(node)
            current_id = node.parent_id
        path.reverse()
        return path

    def _build_trajectory_messages(
        self, node_id: str, task_input: str, episode_context: str | None = None
    ) -> list[Message]:
        path = self._get_path_to_root(node_id)
        messages: list[Message] = []

        if episode_context:
            messages.append(Message(role="user", content=episode_context))
            messages.append(Message(role="assistant", content="I'll consider these past experiences in my approach."))

        messages.append(Message(role="user", content=task_input))

        for node in path[1:]:  # Skip root
            if node.action:
                tc = ToolCall(id=node.id, name=node.action, arguments=node.action_input or {})
                messages.append(Message(role="assistant", content=node.thought, tool_calls=[tc]))
                messages.append(
                    Message(
                        role="tool_result",
                        content=node.observation or "",
                        tool_call_id=node.id,
                        metadata=node.metadata,
                    )
                )
            elif node.is_terminal:
                messages.append(Message(role="assistant", content=node.terminal_output or node.thought))
            else:
                messages.append(Message(role="assistant", content=node.thought))

        return messages

    # ── UCB1 Selection ────────────────────────────────────────

    def _ucb1(self, node: ActionNode) -> float:
        if node.visit_count == 0:
            return float("inf")
        parent = self._nodes[node.parent_id] if node.parent_id else None
        parent_visits = parent.visit_count if parent else 1
        exploitation = node.value / node.visit_count
        exploration = self._exploration_constant * math.sqrt(math.log(max(parent_visits, 1)) / node.visit_count)
        return exploitation + exploration

    def _select_leaf(self) -> ActionNode:
        node = self._nodes[self._root_id]
        while True:
            if node.is_terminal or node.is_failed:
                return node
            # Skip pruned children during descent. UCB1 returns +inf for any
            # unvisited node, so without this filter a freshly-pruned child
            # would win argmax and the descent would step through a dead
            # subtree. A node with no live children is itself a dead leaf.
            live_children = [self._nodes[cid] for cid in node.children_ids if not self._nodes[cid].is_failed]
            if not live_children:
                return node
            node = max(live_children, key=self._ucb1)

    # ── Expansion ─────────────────────────────────────────────

    @staticmethod
    def _truncate(value: str, limit: int) -> str:
        return value if len(value) <= limit else value[:limit] + "..."

    def _describe_siblings(self, node_id: str) -> list[str]:
        """Build human-readable descriptions of existing children for the diversity prompt."""
        descriptions: list[str] = []
        for cid in self._nodes[node_id].children_ids:
            child = self._nodes[cid]
            if child.action and child.action_input:
                args = ", ".join(f'{k}="{self._truncate(str(v), 100)}"' for k, v in child.action_input.items())
                descriptions.append(f"{child.action}({args})")
            elif child.action:
                descriptions.append(child.action)
            elif child.is_terminal:
                text = self._truncate(child.terminal_output or child.thought, 150)
                descriptions.append(f'[terminal: "{text}"]')
            elif child.thought:
                descriptions.append(f'[thought: "{self._truncate(child.thought, 150)}"]')
        return descriptions

    async def _expand(
        self,
        node: ActionNode,
        task_input: str,
        tool_schemas: list[Any],
        episode_context: str | None,
    ) -> list[ActionNode]:
        children: list[ActionNode] = []
        child_depth = node.depth + 1
        force_terminal = self._terminal_depth is not None and child_depth >= self._terminal_depth

        for _ in range(self._branching_factor):
            messages = self._build_trajectory_messages(node.id, task_input, episode_context)

            # Diversity prompt: include sibling actions with arguments to encourage different approaches
            sibling_descriptions = self._describe_siblings(node.id)
            if sibling_descriptions:
                lines = "\n".join(f"- {d}" for d in sibling_descriptions)
                messages.append(
                    Message(
                        role="user",
                        content=(
                            f"Previous sibling branches explored these approaches:\n{lines}\n\n"
                            "Choose a meaningfully different approach — a different query, "
                            "a different tool, or a different angle on the problem."
                        ),
                    )
                )

            expand_tools = None if force_terminal else tool_schemas
            response = await self._call_llm(messages, tools=expand_tools)

            terminal_suppressed = False

            if response.tool_calls:
                tc = response.tool_calls[0]
                thought = response.content or ""
                try:
                    result = await run_cancellable(
                        self._tool_registry.dispatch(tc),
                        self._cancellation_token,
                        tool_name=tc.name,
                    )
                    child = ActionNode(
                        parent_id=node.id,
                        depth=node.depth + 1,
                        thought=thought,
                        action=tc.name,
                        action_input=tc.arguments,
                        observation=result.content,
                        metadata=result.metadata or None,
                    )
                except RunCancelled:
                    raise
                except Exception as exc:
                    child = ActionNode(
                        parent_id=node.id,
                        depth=node.depth + 1,
                        thought=thought,
                        action=tc.name,
                        action_input=tc.arguments,
                        is_failed=True,
                        error_message=str(exc),
                    )
            else:
                child_depth = node.depth + 1
                if child_depth < self._min_depth:
                    # Below min_depth: treat as intermediate thought, not terminal
                    child = ActionNode(
                        parent_id=node.id,
                        depth=child_depth,
                        thought=response.content or "",
                        is_terminal=False,
                    )
                    terminal_suppressed = True
                else:
                    child = ActionNode(
                        parent_id=node.id,
                        depth=child_depth,
                        thought=response.content or "",
                        is_terminal=True,
                        terminal_output=response.content,
                    )

            self._add_node(child)
            self._update_node(
                node.id,
                children_ids=[*self._nodes[node.id].children_ids, child.id],
            )
            children.append(child)

            self._emitter.emit(
                TreeSearchNodeCreatedEvent(
                    trace_id=self._emitter.trace_id,
                    span_id=self._emitter.span_id,
                    parent_span_id=self._emitter.parent_span_id,
                    node_id=child.id,
                    parent_id=node.id,
                    depth=child.depth,
                    content=child.thought,
                    node_type="action",
                    action=child.action,
                    observation=child.observation,
                    is_terminal=child.is_terminal,
                    is_failed=child.is_failed,
                    error_message=child.error_message,
                    terminal_suppressed=terminal_suppressed if terminal_suppressed else None,
                )
            )

        return children

    # ── Evaluation ────────────────────────────────────────────

    async def _evaluate_node(self, node: ActionNode, task_input: str) -> tuple[float, EvaluationVerdict]:
        if node.is_terminal:
            content = node.terminal_output or node.thought
        else:
            content = self._format_trajectory(node.id)

        eval_result = await self._evaluate_output(
            content,
            task_input,
            [],
            0,
            depth=node.depth,
            max_depth=self._max_depth,
            trajectory_length=len(self._get_path_to_root(node.id)),
            total_nodes_explored=len(self._nodes),
        )
        score = eval_result.score if eval_result.score is not None else 0.0

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
            self._update_node(node.id, is_failed=True)
            self._emitter.emit(
                TreeSearchNodePrunedEvent(
                    trace_id=self._emitter.trace_id,
                    span_id=self._emitter.span_id,
                    parent_span_id=self._emitter.parent_span_id,
                    node_id=node.id,
                    reason="evaluation_rejected",
                )
            )

        return score, eval_result.verdict

    def _format_trajectory(self, node_id: str) -> str:
        path = self._get_path_to_root(node_id)
        parts: list[str] = []
        for node in path:
            parts.append(f"Thought: {node.thought}")
            if node.action:
                parts.append(f"Action: {node.action}")
                if node.observation:
                    parts.append(f"Observation: {node.observation}")
        return "\n".join(parts)

    # ── Prune Propagation ─────────────────────────────────────

    def _propagate_pruning(self, node_id: str) -> None:
        """Cascade ``is_failed`` upward when every child is pruned.

        Called after expanding and evaluating a node's children. Walks
        from ``node_id`` toward the root; stops at the first ancestor
        that still has a live child, or at the root (the root is never
        marked failed — leaving the door open for the search to expand
        a fresh set of siblings if iteration budget remains).
        """
        current_id: str | None = node_id
        while current_id is not None:
            node = self._nodes[current_id]
            if node.is_failed or node.parent_id is None or not node.children_ids:
                break
            if not all(self._nodes[cid].is_failed for cid in node.children_ids):
                break
            self._update_node(current_id, is_failed=True)
            self._emitter.emit(
                TreeSearchNodePrunedEvent(
                    trace_id=self._emitter.trace_id,
                    span_id=self._emitter.span_id,
                    parent_span_id=self._emitter.parent_span_id,
                    node_id=current_id,
                    reason="all_children_pruned",
                )
            )
            current_id = node.parent_id

    # ── Backpropagation ───────────────────────────────────────

    def _backpropagate(self, node_id: str, value: float) -> None:
        updated_ids: list[str] = []
        current_id: str | None = node_id
        while current_id is not None:
            node = self._nodes[current_id]
            self._update_node(
                current_id,
                value=node.value + value,
                visit_count=node.visit_count + 1,
            )
            updated_ids.append(current_id)
            current_id = node.parent_id

        self._emitter.emit(
            MCTSBackpropagationEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                propagated_value=value,
                path_length=len(updated_ids),
                updated_node_ids=updated_ids,
            )
        )

    # ── Episodic Memory ───────────────────────────────────────

    async def _recall_episodes(self, task_input: str) -> str | None:
        if self._episode_store is None:
            return None
        results = await self._episode_store.recall(task_input, limit=3)
        if not results:
            return None
        lines = ["[Past Experiences]", ""]
        for i, r in enumerate(results, 1):
            ep = r.episode
            lines.append(f"## Experience {i} ({ep.outcome.value})")
            lines.append(f"Situation: {ep.situation}")
            lines.append(f"Action: {ep.action}")
            if ep.reflection:
                lines.append(f"Reflection: {ep.reflection}")
            lines.append("")
        return "\n".join(lines).rstrip()

    async def _generate_reflection(self, task_input: str, best_output: str | None) -> str:
        user_content = (
            f"Task: {task_input}\n\n"
            f"Best result found: {best_output or 'No solution found'}\n\n"
            f"Total branches explored: {len(self._nodes)}\n\n"
            "What went wrong across the search tree and what should be tried differently?"
        )
        # InstrumentedLLMClient is request-scoped — rebuilt with the current
        # emitter each call rather than held across bind() so concurrent
        # binds do not share an emitter.
        instrumented = InstrumentedLLMClient(self._llm_client, emitter=self._emitter, label="reflection")
        response = await instrumented.generate(
            system_prompt=_REFLECTION_SYSTEM_PROMPT,
            messages=[Message(role="user", content=user_content)],
        )
        return response.content or ""

    async def _record_episode(self, task_input: str, result: AgentResult, *, failed: bool = False) -> None:
        if self._episode_store is None:
            return
        reflection = None
        if failed:
            reflection = await self._generate_reflection(task_input, result.output)
        from nanitics.capabilities.memory.episodic import OutcomeType, extract_episode

        episode = extract_episode(
            task_input=task_input,
            result=result,
            outcome=OutcomeType.FAILURE if failed else None,
            reflection=reflection,
            metadata={"agent": self._name, "nodes_explored": len(self._nodes)},
        )
        await self._episode_store.record(episode)

    # ── Main MCTS Loop ────────────────────────────────────────

    async def _execute(self, input: AgentInput) -> AgentResult:
        self._nodes.clear()
        usages: list[Usage] = []
        tool_schemas = self._tool_registry.list_schemas()
        task_text = _input_to_text(input)

        # Create root node
        root = ActionNode(thought=task_text, depth=0)
        self._add_node(root)
        self._root_id = root.id

        self._emitter.emit(
            TreeSearchNodeCreatedEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                node_id=root.id,
                parent_id=None,
                depth=0,
                content=task_text,
                node_type="action",
            )
        )

        # Retrieve episodic context
        episode_context = await self._recall_episodes(task_text)

        termination_reason = "max_iterations"
        step_number = 0
        accepted_terminal_ids: list[str] = []

        for iteration in range(1, self._max_iterations + 1):
            if self._is_cancelled:
                self._emit_safety_cancellation(step_number)
                termination_reason = "cancelled"
                break

            step_number = iteration

            try:
                with self._emitter.span(f"step-{iteration}"):
                    leaf = self._select_leaf()
                    selection_path = [n.id for n in self._get_path_to_root(leaf.id)]

                    if leaf.is_terminal or leaf.is_failed:
                        # Re-selected a terminal/failed node — backpropagate existing value
                        avg_value = leaf.value / max(leaf.visit_count, 1)
                        self._backpropagate(leaf.id, avg_value)

                        self._emit_mcts_iteration(iteration, leaf.id, selection_path, 0)
                        self._emit_step(step_number)
                        continue

                    if leaf.depth >= self._max_depth:
                        # At max depth — evaluate but don't expand
                        score, _verdict = await self._evaluate_node(leaf, task_text)
                        self._backpropagate(leaf.id, score)

                        self._emit_mcts_iteration(iteration, leaf.id, selection_path, 0)
                        self._emit_step(step_number)
                        continue

                    # Expand
                    children = await self._expand(leaf, task_text, tool_schemas, episode_context)

                    # Evaluate and backpropagate each child
                    for child in children:
                        current_child = self._nodes[child.id]
                        if current_child.is_failed:
                            self._backpropagate(child.id, 0.0)
                            continue
                        score, verdict = await self._evaluate_node(current_child, task_text)
                        self._backpropagate(child.id, score)
                        if current_child.is_terminal and verdict == EvaluationVerdict.ACCEPT:
                            accepted_terminal_ids.append(current_child.id)

                    # If every child of the expanded leaf ended up failed, mark
                    # the leaf (and, transitively, its ancestors) as dead too —
                    # otherwise the next iteration's _select_leaf would return
                    # the same leaf and _expand would pile on more zombie children.
                    self._propagate_pruning(leaf.id)

                    self._emit_mcts_iteration(iteration, leaf.id, selection_path, len(children))
                    self._emit_step(step_number)
            except RunCancelled as exc:
                self._emit_safety_cancellation(exc.step_number or step_number)
                termination_reason = "cancelled"
                break

        # Select best from accepted terminals, or fall back to best node
        if accepted_terminal_ids:
            best = self._select_best_node(candidate_ids=set(accepted_terminal_ids))
        else:
            best = self._select_best_node()
        max_depth = max(n.depth for n in self._nodes.values()) if self._nodes else 0

        self._emitter.emit(
            TreeSearchCompleteEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                total_nodes=len(self._nodes),
                max_depth_reached=max_depth,
                selected_node_id=best.id if best else "",
                termination_reason=termination_reason,
                search_strategy="mcts",
                accepted_count=len(accepted_terminal_ids),
            )
        )

        output = None
        if best:
            if best.is_terminal:
                output = best.terminal_output or best.thought
            else:
                output = self._format_trajectory(best.id)

        result = AgentResult(
            output=output,
            total_steps=step_number,
            termination_reason=termination_reason,
            messages=[],
            usage=self._aggregate_usage(usages),
        )

        # Record episode
        is_failure = len(accepted_terminal_ids) == 0 and termination_reason not in ("complete",)
        await self._record_episode(task_text, result, failed=is_failure)

        return result

    def _emit_mcts_iteration(
        self,
        iteration: int,
        selected_id: str,
        selection_path: list[str],
        expanded_count: int,
    ) -> None:
        best_value = max(
            (n.value / max(n.visit_count, 1) for n in self._nodes.values() if n.visit_count > 0),
            default=0.0,
        )
        node_values = {n.id: n.value / n.visit_count for n in self._nodes.values() if n.visit_count > 0}
        self._emitter.emit(
            MCTSIterationEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                iteration_number=iteration,
                selected_node_id=selected_id,
                selection_path=selection_path,
                expanded_count=expanded_count,
                best_value_so_far=best_value,
                node_values=node_values,
            )
        )

    def _select_best_node(self, *, candidate_ids: set[str] | None = None) -> ActionNode | None:
        if not self._nodes:
            return None

        pool = (
            [self._nodes[nid] for nid in candidate_ids if nid in self._nodes]
            if candidate_ids is not None
            else list(self._nodes.values())
        )

        if candidate_ids is not None and pool:
            return max(
                pool,
                key=lambda n: n.value / max(n.visit_count, 1),
            )

        # Prefer terminal non-failed nodes
        terminals = [n for n in pool if n.is_terminal and not n.is_failed]
        if terminals:
            return max(
                terminals,
                key=lambda n: n.value / max(n.visit_count, 1),
            )

        # Fall back to highest-value non-root, non-failed node
        candidates = [n for n in pool if not n.is_failed and n.depth > 0 and n.visit_count > 0]
        if candidates:
            return max(
                candidates,
                key=lambda n: n.value / max(n.visit_count, 1),
            )

        # Last resort: root
        return next(iter(self._nodes.values()))

    @staticmethod
    def _aggregate_usage(usages: list[Usage]) -> Usage:
        return Usage(
            input_tokens=sum(u.input_tokens for u in usages),
            output_tokens=sum(u.output_tokens for u in usages),
        )
