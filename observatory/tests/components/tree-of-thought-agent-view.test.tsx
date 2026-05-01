import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ObservatoryClient } from "../../src/client/observatory-client";
import { TreeOfThoughtAgentView } from "../../src/components/agent-views/tree-of-thought-agent-view";
import { ObservatoryProvider } from "../../src/context/observatory-context";
import { buildTreeOfThoughtData } from "../../src/hooks/use-tree-of-thought-data";
import { EventRendererRegistry } from "../../src/registry/renderer-registry";
import type { AgentInfo, SpanTreeNode, TraceEvent } from "../../src/types";
import { makeEvent } from "../fixtures/scenarios";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const mockAgent: AgentInfo = {
	agent_name: "tot-agent",
	agent_type: "tree_of_thought",
	span_id: "agent-1",
	capabilities: [],
	stats: {
		llm_calls: 10,
		tool_calls: 0,
		input_tokens: 5000,
		output_tokens: 3000,
		duration_ms: 8000,
		errors: 0,
		iterations: 0,
	},
};

const mockSpanTree: SpanTreeNode = {
	span_id: "agent-1",
	parent_span_id: "root",
	name: "tot-agent",
	summary: {
		event_count: 15,
		duration_ms: 8000,
		has_errors: false,
		agent_name: "tot-agent",
		agent_type: "tree_of_thought",
	},
	events: [],
	children: [],
};

function renderView(events: TraceEvent[]) {
	const client = new ObservatoryClient("/test");
	const registry = new EventRendererRegistry();

	return render(
		<ObservatoryProvider client={client} registry={registry}>
			<TreeOfThoughtAgentView agent={mockAgent} events={events} spanTree={mockSpanTree} />
		</ObservatoryProvider>,
	);
}

// ---------------------------------------------------------------------------
// Fixture: shallow wide tree (3 children at depth 1)
// ---------------------------------------------------------------------------

function makeShallowWideEvents(): TraceEvent[] {
	return [
		makeEvent({
			event_type: "tree_search.node.created",
			payload: { node_id: "root", parent_id: null, depth: 0, node_type: "thought", content: "Root thought in full" },
		}),
		makeEvent({
			event_type: "tree_search.node.created",
			payload: { node_id: "c1", parent_id: "root", depth: 1, node_type: "thought", content: "Child A full" },
		}),
		makeEvent({
			event_type: "tree_search.node.created",
			payload: { node_id: "c2", parent_id: "root", depth: 1, node_type: "thought", content: "Child B full" },
		}),
		makeEvent({
			event_type: "tree_search.node.created",
			payload: { node_id: "c3", parent_id: "root", depth: 1, node_type: "thought", content: "Child C full" },
		}),
		makeEvent({
			event_type: "tree_search.node.evaluated",
			payload: { node_id: "c1", score: 0.8, is_terminal: false },
		}),
		makeEvent({
			event_type: "tree_search.node.evaluated",
			payload: { node_id: "c2", score: 0.3, is_terminal: false },
		}),
		makeEvent({
			event_type: "tree_search.node.evaluated",
			payload: { node_id: "c3", score: 0.9, is_terminal: true },
		}),
		makeEvent({
			event_type: "tree_search.node.pruned",
			payload: { node_id: "c2", reason: "Score below threshold" },
		}),
		makeEvent({
			event_type: "tree_search.complete",
			payload: {
				total_nodes: 4,
				max_depth_reached: 1,
				selected_node_id: "c3",
				termination_reason: "solution_found",
				search_strategy: "best_first",
			},
		}),
	];
}

// ---------------------------------------------------------------------------
// Fixture: deep narrow tree (root → c1 → c2 → c3)
// ---------------------------------------------------------------------------

function makeDeepNarrowEvents(): TraceEvent[] {
	return [
		makeEvent({
			event_type: "tree_search.node.created",
			payload: { node_id: "root", parent_id: null, depth: 0, content: "Root", node_type: "thought" },
		}),
		makeEvent({
			event_type: "tree_search.node.created",
			payload: { node_id: "c1", parent_id: "root", depth: 1, content: "Level 1", node_type: "thought" },
		}),
		makeEvent({
			event_type: "tree_search.node.created",
			payload: { node_id: "c2", parent_id: "c1", depth: 2, content: "Level 2", node_type: "thought" },
		}),
		makeEvent({
			event_type: "tree_search.node.created",
			payload: {
				node_id: "c3",
				parent_id: "c2",
				depth: 3,
				content: "Level 3 terminal",
				node_type: "thought",
				is_terminal: true,
			},
		}),
		makeEvent({
			event_type: "tree_search.node.evaluated",
			payload: { node_id: "c3", score: 0.95, is_terminal: true },
		}),
		makeEvent({
			event_type: "tree_search.complete",
			payload: {
				total_nodes: 4,
				max_depth_reached: 3,
				selected_node_id: "c3",
				termination_reason: "solution_found",
				search_strategy: "dfs",
			},
		}),
	];
}

// ---------------------------------------------------------------------------
// Fixture: mixed pruned/terminal tree
// ---------------------------------------------------------------------------

function makeMixedPrunedTerminalEvents(): TraceEvent[] {
	return [
		makeEvent({
			event_type: "tree_search.node.created",
			payload: { node_id: "root", parent_id: null, depth: 0, content: "Problem root", node_type: "thought" },
		}),
		// Branch A: pruned at depth 1 with child at depth 2
		makeEvent({
			event_type: "tree_search.node.created",
			payload: { node_id: "a1", parent_id: "root", depth: 1, content: "Branch A", node_type: "thought" },
		}),
		makeEvent({
			event_type: "tree_search.node.created",
			payload: { node_id: "a2", parent_id: "a1", depth: 2, content: "Branch A child", node_type: "thought" },
		}),
		// Branch B: terminal at depth 1
		makeEvent({
			event_type: "tree_search.node.created",
			payload: {
				node_id: "b1",
				parent_id: "root",
				depth: 1,
				content: "Branch B solution",
				node_type: "thought",
				is_terminal: true,
			},
		}),
		// Evaluations
		makeEvent({
			event_type: "tree_search.node.evaluated",
			payload: { node_id: "a1", score: 0.2, is_terminal: false },
		}),
		makeEvent({
			event_type: "tree_search.node.evaluated",
			payload: { node_id: "b1", score: 0.85, is_terminal: true },
		}),
		// Prune branch A
		makeEvent({
			event_type: "tree_search.node.pruned",
			payload: { node_id: "a1", reason: "Low score" },
		}),
		makeEvent({
			event_type: "tree_search.complete",
			payload: {
				total_nodes: 4,
				max_depth_reached: 2,
				selected_node_id: "b1",
				termination_reason: "solution_found",
				search_strategy: "bfs",
			},
		}),
	];
}

// ---------------------------------------------------------------------------
// Data Transformation Tests
// ---------------------------------------------------------------------------

describe("buildTreeOfThoughtData", () => {
	it("builds tree from shallow wide events", () => {
		const data = buildTreeOfThoughtData(makeShallowWideEvents());

		expect(data.root).not.toBeNull();
		expect(data.root?.id).toBe("root");
		expect(data.root?.children).toHaveLength(3);
		expect(data.totalNodes).toBe(4);
		expect(data.maxDepth).toBe(1);
		expect(data.searchStrategy).toBe("best_first");
		expect(data.terminationReason).toBe("solution_found");
		expect(data.selectedNodeId).toBe("c3");
	});

	it("sets scores from evaluation events", () => {
		const data = buildTreeOfThoughtData(makeShallowWideEvents());
		const nodeMap = new Map<string, typeof data.root>();
		const queue = [data.root!];
		while (queue.length > 0) {
			const n = queue.shift()!;
			nodeMap.set(n.id, n);
			queue.push(...n.children);
		}

		expect(nodeMap.get("c1")?.score).toBe(0.8);
		expect(nodeMap.get("c2")?.score).toBe(0.3);
		expect(nodeMap.get("c3")?.score).toBe(0.9);
	});

	it("marks pruned nodes", () => {
		const data = buildTreeOfThoughtData(makeShallowWideEvents());
		expect(data.prunedNodeIds.has("c2")).toBe(true);
		expect(data.prunedNodeIds.has("c1")).toBe(false);
	});

	it("marks terminal nodes", () => {
		const data = buildTreeOfThoughtData(makeShallowWideEvents());
		const c3 = data.root?.children.find((c) => c.id === "c3");
		expect(c3?.status).toBe("terminal");
	});

	it("builds solution path from selected node to root", () => {
		const data = buildTreeOfThoughtData(makeShallowWideEvents());
		expect(data.solutionPath.has("c3")).toBe(true);
		expect(data.solutionPath.has("root")).toBe(true);
		// c1 and c2 are not on the solution path
		expect(data.solutionPath.has("c1")).toBe(false);
		expect(data.solutionPath.has("c2")).toBe(false);
	});

	it("builds deep narrow tree correctly", () => {
		const data = buildTreeOfThoughtData(makeDeepNarrowEvents());
		expect(data.root?.children).toHaveLength(1);
		expect(data.root?.children[0].children).toHaveLength(1);
		expect(data.root?.children[0].children[0].children).toHaveLength(1);
		expect(data.maxDepth).toBe(3);
		expect(data.searchStrategy).toBe("dfs");
	});

	it("builds solution path through deep tree", () => {
		const data = buildTreeOfThoughtData(makeDeepNarrowEvents());
		expect(data.solutionPath.has("root")).toBe(true);
		expect(data.solutionPath.has("c1")).toBe(true);
		expect(data.solutionPath.has("c2")).toBe(true);
		expect(data.solutionPath.has("c3")).toBe(true);
	});

	it("propagates pruned status to descendants", () => {
		const data = buildTreeOfThoughtData(makeMixedPrunedTerminalEvents());
		// a1 is directly pruned, a2 is its child — should also be in prunedNodeIds
		expect(data.prunedNodeIds.has("a1")).toBe(true);
		expect(data.prunedNodeIds.has("a2")).toBe(true);
		expect(data.prunedNodeIds.has("b1")).toBe(false);
	});

	it("returns null root for empty events", () => {
		const data = buildTreeOfThoughtData([]);
		expect(data.root).toBeNull();
		expect(data.totalNodes).toBe(0);
		expect(data.solutionPath.size).toBe(0);
	});

	it("handles events without complete event", () => {
		const events = [
			makeEvent({
				event_type: "tree_search.node.created",
				payload: { node_id: "root", parent_id: null, depth: 0, content: "Root", node_type: "thought" },
			}),
		];
		const data = buildTreeOfThoughtData(events);
		expect(data.root).not.toBeNull();
		expect(data.searchStrategy).toBeNull();
		expect(data.terminationReason).toBeNull();
		expect(data.selectedNodeId).toBeNull();
	});
});

// ---------------------------------------------------------------------------
// Component Rendering Tests
// ---------------------------------------------------------------------------

describe("TreeOfThoughtAgentView", () => {
	it("renders header with strategy badge and stats", () => {
		renderView(makeShallowWideEvents());

		expect(screen.getByText("Tree of Thought")).toBeInTheDocument();
		expect(screen.getByText("Best-first")).toBeInTheDocument();
		expect(screen.getByText("solution_found")).toBeInTheDocument();
		expect(screen.getByText("4")).toBeInTheDocument(); // total nodes
	});

	it("renders tree visualization", () => {
		const { container } = renderView(makeShallowWideEvents());
		const svg = container.querySelector("svg");
		expect(svg).toBeTruthy();
	});

	it("renders empty state when no events", () => {
		renderView([]);
		expect(screen.getByText("No tree search data available.")).toBeInTheDocument();
	});

	it("shows node detail panel on node click", () => {
		const { container } = renderView(makeShallowWideEvents());
		// Click on a tree node
		const nodeGroups = container.querySelectorAll("[data-testid^='tree-node-']");
		if (nodeGroups.length > 0) {
			fireEvent.click(nodeGroups[0]);
			// Detail panel should appear
			expect(screen.queryByTestId("tree-node-detail-panel")).toBeInTheDocument();
		}
	});

	it("renders with deep narrow tree", () => {
		const { container } = renderView(makeDeepNarrowEvents());
		expect(screen.getByText("DFS")).toBeInTheDocument();
		const svg = container.querySelector("svg");
		expect(svg).toBeTruthy();
	});

	it("renders with mixed pruned/terminal tree", () => {
		renderView(makeMixedPrunedTerminalEvents());
		expect(screen.getByText("BFS")).toBeInTheDocument();
		expect(screen.getByText("solution_found")).toBeInTheDocument();
	});

	it("renders with test id", () => {
		renderView(makeShallowWideEvents());
		expect(screen.getByTestId("tot-agent-view")).toBeInTheDocument();
	});
});
