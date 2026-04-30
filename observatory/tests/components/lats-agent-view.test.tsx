import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ObservatoryClient } from "../../src/client/observatory-client";
import { LATSAgentView } from "../../src/components/agent-views/lats-agent-view";
import { ObservatoryProvider } from "../../src/context/observatory-context";
import { buildLATSData } from "../../src/hooks/use-lats-data";
import { EventRendererRegistry } from "../../src/registry/renderer-registry";
import type { AgentInfo, SpanTreeNode, TraceEvent } from "../../src/types";
import { makeEvent } from "../fixtures/scenarios";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const mockAgent: AgentInfo = {
	agent_name: "lats-agent",
	agent_type: "lats",
	span_id: "agent-1",
	capabilities: ["memory"],
	stats: {
		llm_calls: 15,
		tool_calls: 8,
		input_tokens: 8000,
		output_tokens: 4000,
		duration_ms: 25000,
		errors: 0,
		iterations: 5,
	},
};

const mockSpanTree: SpanTreeNode = {
	span_id: "agent-1",
	parent_span_id: "root",
	name: "lats-agent",
	summary: {
		event_count: 30,
		duration_ms: 25000,
		has_errors: false,
		agent_name: "lats-agent",
		agent_type: "lats",
	},
	events: [],
	children: [],
};

function renderView(events: TraceEvent[]) {
	const client = new ObservatoryClient("/test");
	const registry = new EventRendererRegistry();

	return render(
		<ObservatoryProvider client={client} registry={registry}>
			<LATSAgentView agent={mockAgent} events={events} spanTree={mockSpanTree} />
		</ObservatoryProvider>,
	);
}

// ---------------------------------------------------------------------------
// Fixture: basic LATS scenario with 3 iterations
// ---------------------------------------------------------------------------

function makeLATSEvents(): TraceEvent[] {
	return [
		makeEvent({
			event_type: "agent.start",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:00Z",
			payload: {
				agent_name: "lats-agent",
				agent_type: "lats",
				tools_available: ["web_search", "calculator"],
				exploration_constant: 1.4,
			},
		}),
		// Root node
		makeEvent({
			event_type: "tree_search.node.created",
			timestamp: "2026-03-05T10:00:01Z",
			payload: {
				node_id: "root",
				parent_id: null,
				depth: 0,
				content: "Initial state of the problem",
			},
		}),
		// Iteration 1: expand root
		makeEvent({
			event_type: "tree_search.node.created",
			timestamp: "2026-03-05T10:00:02Z",
			payload: {
				node_id: "n1",
				parent_id: "root",
				depth: 1,
				content: "Search for data",
				node_type: "action",
				action: "web_search",
				observation: "Found 5 results about...",
			},
		}),
		makeEvent({
			event_type: "tree_search.node.evaluated",
			timestamp: "2026-03-05T10:00:03Z",
			payload: { node_id: "n1", score: 0.6, is_terminal: false },
		}),
		makeEvent({
			event_type: "mcts.iteration",
			timestamp: "2026-03-05T10:00:04Z",
			payload: {
				iteration_number: 1,
				selected_node_id: "root",
				selection_path: ["root"],
				expanded_count: 1,
				best_value_so_far: 0.6,
				node_values: { root: 0.6 },
			},
		}),
		makeEvent({
			event_type: "mcts.backpropagation",
			timestamp: "2026-03-05T10:00:04.5Z",
			payload: { propagated_value: 0.6, path_length: 1, updated_node_ids: ["root"] },
		}),
		// Iteration 2: expand n1
		makeEvent({
			event_type: "tree_search.node.created",
			timestamp: "2026-03-05T10:00:05Z",
			payload: {
				node_id: "n2",
				parent_id: "n1",
				depth: 2,
				content: "Calculate result",
				node_type: "action",
				action: "calculator",
				observation: "Result: 42",
			},
		}),
		makeEvent({
			event_type: "tree_search.node.evaluated",
			timestamp: "2026-03-05T10:00:06Z",
			payload: { node_id: "n2", score: 0.8, is_terminal: false },
		}),
		makeEvent({
			event_type: "mcts.iteration",
			timestamp: "2026-03-05T10:00:07Z",
			payload: {
				iteration_number: 2,
				selected_node_id: "n1",
				selection_path: ["root", "n1"],
				expanded_count: 1,
				best_value_so_far: 0.8,
				node_values: { root: 0.7, n1: 0.8 },
			},
		}),
		makeEvent({
			event_type: "mcts.backpropagation",
			timestamp: "2026-03-05T10:00:07.5Z",
			payload: { propagated_value: 0.8, path_length: 2, updated_node_ids: ["n1", "root"] },
		}),
		// Iteration 3: expand n2 to terminal
		makeEvent({
			event_type: "tree_search.node.created",
			timestamp: "2026-03-05T10:00:08Z",
			payload: {
				node_id: "n3",
				parent_id: "n2",
				depth: 3,
				content: "Final answer: 42",
				node_type: "thought",
				is_terminal: true,
			},
		}),
		makeEvent({
			event_type: "tree_search.node.evaluated",
			timestamp: "2026-03-05T10:00:09Z",
			payload: { node_id: "n3", score: 0.95, is_terminal: true },
		}),
		makeEvent({
			event_type: "mcts.iteration",
			timestamp: "2026-03-05T10:00:10Z",
			payload: {
				iteration_number: 3,
				selected_node_id: "n2",
				selection_path: ["root", "n1", "n2"],
				expanded_count: 1,
				best_value_so_far: 0.95,
				node_values: { root: 0.78, n1: 0.85, n2: 0.95 },
			},
		}),
		makeEvent({
			event_type: "mcts.backpropagation",
			timestamp: "2026-03-05T10:00:10.5Z",
			payload: { propagated_value: 0.95, path_length: 3, updated_node_ids: ["n2", "n1", "root"] },
		}),
		makeEvent({
			event_type: "tree_search.complete",
			timestamp: "2026-03-05T10:00:11Z",
			payload: {
				total_nodes: 4,
				max_depth_reached: 3,
				selected_node_id: "n3",
				termination_reason: "solution_found",
				search_strategy: "mcts",
			},
		}),
	];
}

// ---------------------------------------------------------------------------
// Fixture: LATS with episodic memory
// ---------------------------------------------------------------------------

function makeLATSWithMemoryEvents(): TraceEvent[] {
	return [
		...makeLATSEvents(),
		makeEvent({
			event_type: "memory.episode.recall",
			timestamp: "2026-03-05T10:00:00.5Z",
			payload: {
				query: "similar math problems",
				results_count: 2,
				top_score: 0.88,
			},
		}),
	];
}

// ---------------------------------------------------------------------------
// Fixture: LATS with failed node
// ---------------------------------------------------------------------------

function makeLATSWithFailedNodeEvents(): TraceEvent[] {
	return [
		makeEvent({
			event_type: "tree_search.node.created",
			payload: {
				node_id: "root",
				parent_id: null,
				depth: 0,
				content: "Root",
				node_type: "thought",
			},
		}),
		makeEvent({
			event_type: "tree_search.node.created",
			payload: {
				node_id: "n1",
				parent_id: "root",
				depth: 1,
				content: "Failed tool call",
				node_type: "action",
				action: "web_search",
				is_failed: true,
			},
		}),
		makeEvent({
			event_type: "tree_search.node.created",
			payload: {
				node_id: "n2",
				parent_id: "root",
				depth: 1,
				content: "Successful path",
				node_type: "action",
				action: "calculator",
				is_terminal: true,
			},
		}),
		makeEvent({
			event_type: "tree_search.complete",
			payload: {
				total_nodes: 3,
				max_depth_reached: 1,
				selected_node_id: "n2",
				termination_reason: "solution_found",
				search_strategy: "mcts",
			},
		}),
	];
}

// ---------------------------------------------------------------------------
// Data Transformation Tests
// ---------------------------------------------------------------------------

describe("buildLATSData", () => {
	it("builds tree with correct structure", () => {
		const data = buildLATSData(makeLATSEvents());

		expect(data.root).not.toBeNull();
		expect(data.root?.id).toBe("root");
		expect(data.totalNodes).toBe(4);
		expect(data.terminationReason).toBe("solution_found");
		expect(data.bestNodeId).toBe("n3");
	});

	it("extracts iterations in order", () => {
		const data = buildLATSData(makeLATSEvents());

		expect(data.iterations).toHaveLength(3);
		expect(data.iterations[0].iterationNumber).toBe(1);
		expect(data.iterations[1].iterationNumber).toBe(2);
		expect(data.iterations[2].iterationNumber).toBe(3);
	});

	it("extracts node values from iteration events", () => {
		const data = buildLATSData(makeLATSEvents());

		expect(data.iterations[2].nodeValues).toEqual({
			root: 0.78,
			n1: 0.85,
			n2: 0.95,
		});
	});

	it("associates backpropagation events with iterations", () => {
		const data = buildLATSData(makeLATSEvents());

		expect(data.backpropagations).toHaveLength(3);
		expect(data.backpropagations[0].iterationNumber).toBe(1);
		expect(data.backpropagations[1].iterationNumber).toBe(2);
		expect(data.backpropagations[2].iterationNumber).toBe(3);
		expect(data.backpropagations[2].updatedNodeIds).toEqual(["n2", "n1", "root"]);
	});

	it("builds solution path", () => {
		const data = buildLATSData(makeLATSEvents());

		expect(data.solutionPath.has("n3")).toBe(true);
		expect(data.solutionPath.has("n2")).toBe(true);
		expect(data.solutionPath.has("n1")).toBe(true);
		expect(data.solutionPath.has("root")).toBe(true);
	});

	it("stores action metadata on nodes", () => {
		const data = buildLATSData(makeLATSEvents());
		const n1 = data.root?.children[0];
		expect(n1!.metadata.action).toBe("web_search");
		expect(n1!.metadata.observation).toBe("Found 5 results about...");
	});

	it("marks failed nodes correctly", () => {
		const data = buildLATSData(makeLATSWithFailedNodeEvents());
		const n1 = data.root?.children.find((c) => c.id === "n1");
		expect(n1?.status).toBe("failed");
	});

	it("marks terminal nodes correctly", () => {
		const data = buildLATSData(makeLATSWithFailedNodeEvents());
		const n2 = data.root?.children.find((c) => c.id === "n2");
		expect(n2?.status).toBe("terminal");
	});

	it("extracts episodic recall events", () => {
		const data = buildLATSData(makeLATSWithMemoryEvents());
		expect(data.episodicRecalls).toHaveLength(1);
		expect(data.episodicRecalls[0].payload.query).toBe("similar math problems");
	});

	it("extracts exploration constant from agent.start", () => {
		const data = buildLATSData(makeLATSEvents());
		expect(data.explorationConstant).toBe(1.4);
	});

	it("returns null root for empty events", () => {
		const data = buildLATSData([]);
		expect(data.root).toBeNull();
		expect(data.totalNodes).toBe(0);
		expect(data.iterations).toHaveLength(0);
	});
});

// ---------------------------------------------------------------------------
// Component Rendering Tests
// ---------------------------------------------------------------------------

describe("LATSAgentView", () => {
	it("renders header with MCTS badge and stats", () => {
		renderView(makeLATSEvents());

		expect(screen.getByText("LATS")).toBeInTheDocument();
		expect(screen.getByText("MCTS")).toBeInTheDocument();
		expect(screen.getByText("solution_found")).toBeInTheDocument();
		expect(screen.getAllByText("3").length).toBeGreaterThanOrEqual(1); // iterations
		expect(screen.getAllByText("4").length).toBeGreaterThanOrEqual(1); // nodes
	});

	it("renders iteration bar", () => {
		renderView(makeLATSEvents());
		expect(screen.getByTestId("iteration-bar")).toBeInTheDocument();
	});

	it("renders tree visualization", () => {
		const { container } = renderView(makeLATSEvents());
		const svg = container.querySelector("svg");
		expect(svg).toBeTruthy();
	});

	it("renders empty state when no events", () => {
		renderView([]);
		expect(screen.getByText("No LATS data available.")).toBeInTheDocument();
	});

	it("shows node detail panel on node click", () => {
		const { container } = renderView(makeLATSEvents());
		const nodeGroups = container.querySelectorAll("[data-testid^='tree-node-']");
		if (nodeGroups.length > 0) {
			fireEvent.click(nodeGroups[0]);
			expect(screen.queryByTestId("tree-node-detail-panel")).toBeInTheDocument();
		}
	});

	it("renders episodic memory section when recall events exist", () => {
		renderView(makeLATSWithMemoryEvents());
		expect(screen.getByTestId("episodic-memory-section")).toBeInTheDocument();
		expect(screen.getByText(/Episodic Memory/)).toBeInTheDocument();
	});

	it("does not render episodic memory section when no recall events", () => {
		renderView(makeLATSEvents());
		expect(screen.queryByTestId("episodic-memory-section")).not.toBeInTheDocument();
	});

	it("renders with failed nodes", () => {
		renderView(makeLATSWithFailedNodeEvents());
		expect(screen.getByTestId("lats-agent-view")).toBeInTheDocument();
	});

	it("toggles iteration selection", () => {
		renderView(makeLATSEvents());

		// Click iteration 2
		const iterButtons = screen.getAllByRole("button", { name: /Iteration \d+/ });
		if (iterButtons.length > 1) {
			fireEvent.click(iterButtons[1]); // Iteration 2
			// Clear button should appear
			expect(screen.getByLabelText("Clear iteration selection")).toBeInTheDocument();

			// Click clear
			fireEvent.click(screen.getByLabelText("Clear iteration selection"));
			expect(screen.queryByLabelText("Clear iteration selection")).not.toBeInTheDocument();
		}
	});

	it("renders exploration constant", () => {
		renderView(makeLATSEvents());
		expect(screen.getByText("1.4")).toBeInTheDocument();
	});

	it("renders with test id", () => {
		renderView(makeLATSEvents());
		expect(screen.getByTestId("lats-agent-view")).toBeInTheDocument();
	});
});
