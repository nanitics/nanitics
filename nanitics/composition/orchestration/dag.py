from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from typing import Any

from nanitics.composition.durability.models import RunCheckpoint
from nanitics.composition.durability.store import CheckpointStore
from nanitics.composition.durability.suspension import SuspendExecution
from nanitics.composition.orchestration.protocol import FailurePolicy, Step, StepResult
from nanitics.composition.orchestration.workflow import Workflow
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import (
    ExecutionSuspendedEvent,
    WorkflowErrorEvent,
    WorkflowStartEvent,
    WorkflowStepCompleteEvent,
    WorkflowStepDefinition,
)
from nanitics.infrastructure.observability.storage import PersistentTraceStore
from nanitics.safety.cancellation import CancellationToken


@dataclass
class DAGNode:
    """A node in a DAG workflow.

    Args:
        step: The step to execute for this node.
        depends_on: List of node IDs (keys in the DAG's ``nodes`` dict) that
            must complete before this node can execute.
    """

    step: Step
    depends_on: list[str] = field(default_factory=list)


class DAG(Workflow):
    """Executes steps as a directed acyclic graph, respecting dependencies.

    Nodes whose dependencies are satisfied run concurrently. Node input depends on
    the number of dependencies:

    - No dependencies: receives the original workflow input.
    - One dependency: receives that dependency's output directly.
    - Multiple dependencies: receives a dict mapping dependency names to outputs.

    Output comes from terminal nodes (nodes that no other node depends on).
    A single terminal node's output is the result; multiple terminal nodes
    produce a dict mapping names to outputs.

    Validates at construction time that all dependency references exist and
    there are no cycles.

    Args:
        name: Workflow identifier.
        nodes: Mapping of node IDs to DAGNode objects. Must contain at least one node.
        emitter: Event emitter for observability.
        failure_policy: How to handle node failures. Default is ALL_OR_NOTHING.
            With BEST_EFFORT, failed nodes and their transitive dependents are skipped.
        max_concurrency: Maximum number of nodes executing simultaneously. None for unlimited.
        cancellation_token: Optional cooperative cancellation signal.
        checkpoint_store: Optional store for suspension checkpoints.
        run_id: Run identifier for checkpoint records.

    Raises:
        ValueError: If nodes dict is empty, a dependency references a non-existent
            node, or the graph contains a cycle.
    """

    def __init__(
        self,
        *,
        name: str,
        nodes: dict[str, DAGNode],
        emitter: EventEmitter,
        failure_policy: FailurePolicy = FailurePolicy.ALL_OR_NOTHING,
        max_concurrency: int | None = None,
        cancellation_token: CancellationToken | None = None,
        checkpoint_store: CheckpointStore | None = None,
        run_id: str | None = None,
        trace_store: PersistentTraceStore | None = None,
    ) -> None:
        if not nodes:
            raise ValueError("DAG requires at least one node")
        if max_concurrency is not None and max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        self._validate_references(nodes)
        self._validate_no_cycles(nodes)
        super().__init__(
            name=name,
            emitter=emitter,
            cancellation_token=cancellation_token,
            checkpoint_store=checkpoint_store,
            run_id=run_id,
            trace_store=trace_store,
        )
        self._nodes = nodes
        self._failure_policy = failure_policy
        self._max_concurrency = max_concurrency

    @staticmethod
    def _validate_references(nodes: dict[str, DAGNode]) -> None:
        for node_name, node in nodes.items():
            for dep in node.depends_on:
                if dep not in nodes:
                    raise ValueError(f"Node '{node_name}' depends on '{dep}' which does not exist")

    @staticmethod
    def _validate_no_cycles(nodes: dict[str, DAGNode]) -> None:
        # Kahn's algorithm for topological sort / cycle detection
        in_degree: dict[str, int] = dict.fromkeys(nodes, 0)
        # Build adjacency: edges go from dependency to dependent
        adjacency: dict[str, list[str]] = {name: [] for name in nodes}
        for node_name, node in nodes.items():
            in_degree[node_name] = len(node.depends_on)
            for dep in node.depends_on:
                adjacency[dep].append(node_name)

        queue = [name for name, deg in in_degree.items() if deg == 0]
        if not queue:
            raise ValueError("DAG has no source nodes — cycle detected among all nodes")

        visited = 0
        processing = list(queue)
        while processing:
            current = processing.pop(0)
            visited += 1
            for dependent in adjacency[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    processing.append(dependent)

        if visited != len(nodes):
            cycle_nodes = [name for name, deg in in_degree.items() if deg > 0]
            raise ValueError(f"DAG contains a cycle involving nodes: {cycle_nodes}")

    def _workflow_type(self) -> str:
        return "dag"

    def _step_count(self) -> int:
        return len(self._nodes)

    def _get_step_definitions(self) -> list[WorkflowStepDefinition]:
        defs = []
        for i, (name, node) in enumerate(self._nodes.items()):
            step_type, metadata = self._classify_step(node.step)
            defs.append(
                WorkflowStepDefinition(
                    name=name,
                    step_type=step_type,
                    index=i,
                    depends_on=list(node.depends_on),
                    metadata=metadata,
                )
            )
        return defs

    def _emit_start(self) -> None:
        edges = [[dep, node_name] for node_name, node in self._nodes.items() for dep in node.depends_on]
        self._emitter.emit(
            WorkflowStartEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                workflow_name=self._name,
                workflow_type=self._workflow_type(),
                step_count=self._step_count(),
                metadata={
                    "nodes": list(self._nodes.keys()),
                    "edges": edges,
                },
            )
        )

    async def _run(self, input: Any, *, resume_from: RunCheckpoint | None = None) -> StepResult:
        completed: dict[str, StepResult] = {}
        failed_nodes: dict[str, dict[str, str]] = {}
        skipped_nodes: list[str] = []
        node_index: dict[str, int] = {name: i for i, name in enumerate(self._nodes.keys())}

        semaphore = asyncio.Semaphore(self._max_concurrency) if self._max_concurrency is not None else None

        # Track remaining in-degrees
        in_degree: dict[str, int] = {name: len(node.depends_on) for name, node in self._nodes.items()}

        # Build adjacency for dependents
        dependents: dict[str, list[str]] = {name: [] for name in self._nodes}
        for node_name, node in self._nodes.items():
            for dep in node.depends_on:
                dependents[dep].append(node_name)

        # Resume path: restore completed nodes
        suspended_node_name: str | None = None
        if resume_from is not None:
            state = resume_from.state
            suspended_node_name = state["suspended_node"]
            for name, output in state["completed_nodes"].items():
                completed[name] = StepResult(output=output)
                # Reduce in-degrees for dependents of completed nodes
                for dep_name in dependents[name]:
                    in_degree[dep_name] -= 1
            self._emit_resumed(resume_from, suspended_node_name)

        # Ready nodes: those with in_degree == 0 and not already completed
        ready: set[str] = {name for name, deg in in_degree.items() if deg == 0 and name not in completed}
        in_flight: dict[str, asyncio.Task[StepResult]] = {}

        while ready or in_flight:
            # Launch ready nodes
            for node_name in list(ready):
                ready.discard(node_name)

                async def _execute_node(
                    n_name: str = node_name,
                ) -> StepResult:
                    if semaphore:
                        async with semaphore:
                            return await self._run_node(n_name, input, completed, node_index)
                    return await self._run_node(n_name, input, completed, node_index)

                task = asyncio.create_task(_execute_node())
                in_flight[node_name] = task

            if not in_flight:  # pragma: no cover – safety guard; every ready item creates a task
                break

            # Wait for at least one task to complete
            done, _ = await asyncio.wait(in_flight.values(), return_when=asyncio.FIRST_COMPLETED)

            # Process completed tasks
            for task in done:
                # Find which node this task belongs to
                finished_name = None
                for name, t in in_flight.items():
                    if t is task:
                        finished_name = name
                        break
                assert finished_name is not None
                del in_flight[finished_name]

                try:
                    result = task.result()
                except SuspendExecution as exc:
                    # Drain: wait for all remaining in-flight tasks
                    drained_results = await self._drain_in_flight(in_flight)
                    completed.update(drained_results)
                    in_flight.clear()

                    if self._checkpoint_store:
                        checkpoint_state: dict[str, Any] = {
                            "orchestrator_type": "dag",
                            "completed_nodes": {k: v.output for k, v in completed.items()},
                            "suspended_node": finished_name,
                            "original_input": input,
                        }
                        if exc.checkpoint_data:
                            checkpoint_state["agent_checkpoint"] = exc.checkpoint_data
                        checkpoint = await self._save_checkpoint(exc, checkpoint_state)
                        self._emitter.emit(
                            ExecutionSuspendedEvent(
                                trace_id=self._emitter.trace_id,
                                span_id=self._emitter.span_id,
                                parent_span_id=self._emitter.parent_span_id,
                                suspension_id=exc.suspension_info.suspension_id,
                                suspension_type="hitl",
                                checkpoint_id=checkpoint.checkpoint_id,
                                step_name=finished_name,
                                agent_name=exc.suspension_info.agent_name,
                            )
                        )
                    raise
                except Exception as exc:
                    if self._failure_policy == FailurePolicy.ALL_OR_NOTHING:
                        # Cancel all in-flight tasks
                        for t in in_flight.values():
                            if not t.done():
                                t.cancel()
                        if in_flight:
                            await asyncio.gather(*in_flight.values(), return_exceptions=True)
                        raise

                    # BEST_EFFORT: mark as failed, skip transitive dependents
                    failed_nodes[finished_name] = {
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                    self._emitter.emit(
                        WorkflowErrorEvent(
                            trace_id=self._emitter.trace_id,
                            span_id=self._emitter.span_id,
                            parent_span_id=self._emitter.parent_span_id,
                            workflow_name=self._name,
                            workflow_type="dag",
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                            failed_step=finished_name,
                        )
                    )
                    self._skip_dependents(finished_name, dependents, skipped_nodes, in_flight)
                    continue

                # If this node was cancelled, drain remaining and stop
                if result.metadata.get("terminated") == "cancelled":
                    for t in in_flight.values():
                        if not t.done():
                            t.cancel()
                    if in_flight:
                        await asyncio.gather(*in_flight.values(), return_exceptions=True)
                    in_flight.clear()
                    break

                completed[finished_name] = result

                # Unblock dependents
                for dep_name in dependents[finished_name]:
                    if dep_name in skipped_nodes:
                        continue
                    in_degree[dep_name] -= 1
                    if in_degree[dep_name] == 0:
                        ready.add(dep_name)

        # Determine output
        terminal_nodes = [
            name
            for name in self._nodes
            if not dependents[name] and name not in failed_nodes and name not in skipped_nodes and name in completed
        ]

        if len(terminal_nodes) == 1:
            output = completed[terminal_nodes[0]].output
        elif terminal_nodes:
            output = {name: completed[name].output for name in terminal_nodes}
        else:
            output = None

        metadata: dict[str, Any] = {
            "node_results": {name: r.output for name, r in completed.items()},
            "total_steps_executed": len(completed),
        }
        if self._failure_policy == FailurePolicy.BEST_EFFORT:
            if failed_nodes:
                metadata["failed_nodes"] = failed_nodes
            if skipped_nodes:
                metadata["skipped_nodes"] = skipped_nodes
        if self._cancellation_token and self._cancellation_token.is_cancelled:
            metadata["terminated"] = "cancelled"

        return StepResult(output=output, metadata=metadata)

    async def _drain_in_flight(
        self,
        in_flight: dict[str, asyncio.Task[StepResult]],
    ) -> dict[str, StepResult]:
        """Wait for remaining in-flight tasks and collect completed results."""
        drained: dict[str, StepResult] = {}
        if not in_flight:
            return drained
        done, _ = await asyncio.wait(in_flight.values())
        for task in done:
            name = None
            for n, t in in_flight.items():
                if t is task:
                    name = n
                    break
            assert name is not None
            with contextlib.suppress(Exception):
                drained[name] = task.result()
        return drained

    async def _run_node(
        self,
        node_name: str,
        dag_input: Any,
        completed: dict[str, StepResult],
        node_index: dict[str, int],
    ) -> StepResult:
        node = self._nodes[node_name]

        # Determine input for this node
        if not node.depends_on:
            node_input = dag_input
        elif len(node.depends_on) == 1:
            node_input = completed[node.depends_on[0]].output
        else:
            node_input = {dep: completed[dep].output for dep in node.depends_on}

        with self._emitter.span(node_name):
            bound_step = self._bind_step(node.step)
            if self._cancellation_token and self._cancellation_token.is_cancelled:
                return StepResult(output=None, metadata={"terminated": "cancelled"})
            step_start = time.monotonic()
            result = await bound_step.execute(node_input)
            step_duration_ms = int((time.monotonic() - step_start) * 1000)

        self._emitter.emit(
            WorkflowStepCompleteEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                workflow_name=self._name,
                step_name=node_name,
                step_index=node_index[node_name],
                step_duration_ms=step_duration_ms,
                step_output=str(result.output) if result.output is not None else None,
            )
        )
        return result

    def _skip_dependents(
        self,
        failed_name: str,
        dependents: dict[str, list[str]],
        skipped_nodes: list[str],
        in_flight: dict[str, asyncio.Task[StepResult]],
    ) -> None:
        """Transitively skip all nodes that depend on a failed node."""
        queue = list(dependents[failed_name])
        while queue:
            dep = queue.pop(0)
            if dep in skipped_nodes:
                continue
            skipped_nodes.append(dep)
            # Cancel if in-flight
            if dep in in_flight:  # pragma: no cover – dependents of failed nodes can't be in-flight
                task = in_flight.pop(dep)
                if not task.done():
                    task.cancel()
            queue.extend(dependents[dep])
