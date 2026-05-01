import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ObservatoryClient } from "../../src/client/observatory-client";
import { EventDetailPanel } from "../../src/components/event-detail/event-detail-panel";
import { ObservatoryProvider } from "../../src/context/observatory-context";
import { createDefaultRegistrations, createDefaultRegistry } from "../../src/registry/default-renderers";
import type { TraceEvent } from "../../src/types";
import { makeEvent } from "../fixtures/scenarios";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function renderEvent(event: TraceEvent) {
	const client = new ObservatoryClient("/test");
	const registry = createDefaultRegistry();

	return render(
		<ObservatoryProvider client={client} registry={registry}>
			<EventDetailPanel event={event} />
		</ObservatoryProvider>,
	);
}

function getSummary(event: TraceEvent): string {
	const registrations = createDefaultRegistrations();
	for (const reg of registrations) {
		if (reg.matches(event.event_type) && reg.summary) {
			return reg.summary(event);
		}
	}
	return event.event_type;
}

// ---------------------------------------------------------------------------
// Tree Search Event Renderers
// ---------------------------------------------------------------------------

describe("Tree search event renderers", () => {
	describe("tree_search.node.created", () => {
		it("renders node type, depth, and content", () => {
			renderEvent(
				makeEvent({
					event_type: "tree_search.node.created",
					payload: {
						node_id: "node-1",
						parent_id: null,
						depth: 0,
						node_type: "thought",
						content: "Initial thought about the problem in full detail",
					},
				}),
			);
			expect(screen.getByText("thought")).toBeInTheDocument();
			expect(screen.getByText("depth 0")).toBeInTheDocument();
			expect(screen.getByText("node-1")).toBeInTheDocument();
			expect(screen.getByText("Initial thought about the problem in full detail")).toBeInTheDocument();
		});

		it("renders terminal and failed badges", () => {
			renderEvent(
				makeEvent({
					event_type: "tree_search.node.created",
					payload: {
						node_id: "node-2",
						parent_id: "node-1",
						depth: 1,
						content: "Terminal node",
						node_type: "action",
						is_terminal: true,
						is_failed: true,
					},
				}),
			);
			expect(screen.getByText("terminal")).toBeInTheDocument();
			expect(screen.getByText("failed")).toBeInTheDocument();
			expect(screen.getByText("action")).toBeInTheDocument();
		});

		it("renders action and observation preview for LATS nodes", () => {
			renderEvent(
				makeEvent({
					event_type: "tree_search.node.created",
					payload: {
						node_id: "node-3",
						parent_id: "node-1",
						depth: 1,
						content: "Calling search tool",
						node_type: "action",
						action: "web_search",
						observation: "Found 3 relevant results...",
					},
				}),
			);
			expect(screen.getByText("web_search")).toBeInTheDocument();
			expect(screen.getByText("Found 3 relevant results...")).toBeInTheDocument();
		});

		it("renders parent id when present", () => {
			renderEvent(
				makeEvent({
					event_type: "tree_search.node.created",
					payload: {
						node_id: "node-4",
						parent_id: "node-1",
						depth: 1,
						content: "Child thought",
						node_type: "thought",
					},
				}),
			);
			expect(screen.getByText("node-4")).toBeInTheDocument();
			expect(screen.getByText("node-1")).toBeInTheDocument();
		});

		it("falls back to content when content is not available", () => {
			renderEvent(
				makeEvent({
					event_type: "tree_search.node.created",
					payload: {
						node_id: "node-5",
						depth: 0,
						content: "Preview only text",
						node_type: "thought",
					},
				}),
			);
			expect(screen.getByText("Preview only text")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "tree_search.node.created",
					payload: {
						content: "Analyze the dataset",
						depth: 2,
					},
				}),
			);
			expect(summary).toBe("Node created: Analyze the dataset (depth 2)");
		});

		it("truncates long content previews in summary", () => {
			const longPreview = "A".repeat(80);
			const summary = getSummary(
				makeEvent({
					event_type: "tree_search.node.created",
					payload: { content: longPreview, depth: 0 },
				}),
			);
			expect(summary).toContain("…");
			expect(summary.length).toBeLessThan(90);
		});

		it("handles missing optional fields", () => {
			renderEvent(
				makeEvent({
					event_type: "tree_search.node.created",
					payload: {},
				}),
			);
			expect(screen.getByText("tree_search.node.created")).toBeInTheDocument();
		});
	});

	describe("tree_search.node.evaluated", () => {
		it("renders score with color coding", () => {
			renderEvent(
				makeEvent({
					event_type: "tree_search.node.evaluated",
					payload: { node_id: "node-1", score: 0.85, is_terminal: false },
				}),
			);
			expect(screen.getAllByText("0.85").length).toBeGreaterThanOrEqual(1);
			expect(screen.getByText("node-1")).toBeInTheDocument();
		});

		it("renders terminal badge", () => {
			renderEvent(
				makeEvent({
					event_type: "tree_search.node.evaluated",
					payload: { node_id: "node-2", score: 0.92, is_terminal: true },
				}),
			);
			expect(screen.getByText("terminal")).toBeInTheDocument();
		});

		it("renders score bar visual", () => {
			const { container } = renderEvent(
				makeEvent({
					event_type: "tree_search.node.evaluated",
					payload: { node_id: "node-3", score: 0.7, is_terminal: false },
				}),
			);
			// Score bar should have a width style
			const scoreBar = container.querySelector("[style]");
			expect(scoreBar).toBeTruthy();
		});

		it("produces correct summary with terminal indicator", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "tree_search.node.evaluated",
					payload: { score: 0.85, is_terminal: true },
				}),
			);
			expect(summary).toBe("Evaluated: 0.85, terminal");
		});

		it("produces correct summary without terminal", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "tree_search.node.evaluated",
					payload: { score: 0.45, is_terminal: false },
				}),
			);
			expect(summary).toBe("Evaluated: 0.45");
		});
	});

	describe("tree_search.node.pruned", () => {
		it("renders node id and reason", () => {
			renderEvent(
				makeEvent({
					event_type: "tree_search.node.pruned",
					payload: { node_id: "node-5", reason: "Score below threshold" },
				}),
			);
			expect(screen.getByText("node-5")).toBeInTheDocument();
			expect(screen.getByText("Score below threshold")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "tree_search.node.pruned",
					payload: { reason: "Low score" },
				}),
			);
			expect(summary).toBe("Pruned: Low score");
		});

		it("handles missing reason", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "tree_search.node.pruned",
					payload: {},
				}),
			);
			expect(summary).toBe("Pruned");
		});
	});

	describe("tree_search.complete", () => {
		it("renders search strategy, stats, and selected node", () => {
			renderEvent(
				makeEvent({
					event_type: "tree_search.complete",
					payload: {
						total_nodes: 15,
						max_depth_reached: 3,
						selected_node_id: "node-12",
						termination_reason: "solution_found",
						search_strategy: "best_first",
					},
				}),
			);
			expect(screen.getByText("best_first")).toBeInTheDocument();
			expect(screen.getAllByText("15").length).toBeGreaterThanOrEqual(1);
			expect(screen.getAllByText("3").length).toBeGreaterThanOrEqual(1);
			expect(screen.getByText("node-12")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "tree_search.complete",
					payload: {
						termination_reason: "max_depth",
						total_nodes: 20,
					},
				}),
			);
			expect(summary).toBe("Search complete: max_depth (20 nodes)");
		});

		it("handles missing fields in summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "tree_search.complete",
					payload: {},
				}),
			);
			expect(summary).toBe("Search complete");
		});
	});
});

// ---------------------------------------------------------------------------
// MCTS Event Renderers
// ---------------------------------------------------------------------------

describe("MCTS event renderers", () => {
	describe("mcts.iteration", () => {
		it("renders iteration number, expanded count, and best value", () => {
			renderEvent(
				makeEvent({
					event_type: "mcts.iteration",
					payload: {
						iteration_number: 3,
						selected_node_id: "node-7",
						selection_path: ["node-0", "node-3", "node-7"],
						expanded_count: 2,
						best_value_so_far: 0.78,
						node_values: { "node-0": 0.65, "node-3": 0.78 },
					},
				}),
			);
			expect(screen.getByText("#3")).toBeInTheDocument();
			expect(screen.getByText("best=0.78")).toBeInTheDocument();
			expect(screen.getAllByText("2").length).toBeGreaterThanOrEqual(1);
			expect(screen.getAllByText("node-7").length).toBeGreaterThanOrEqual(1);
		});

		it("renders selection path", () => {
			renderEvent(
				makeEvent({
					event_type: "mcts.iteration",
					payload: {
						iteration_number: 1,
						selected_node_id: "node-2",
						selection_path: ["node-0", "node-1", "node-2"],
						expanded_count: 1,
						best_value_so_far: 0.5,
					},
				}),
			);
			expect(screen.getByText("Selection path")).toBeInTheDocument();
			expect(screen.getAllByText("node-0").length).toBeGreaterThanOrEqual(1);
			expect(screen.getAllByText("node-1").length).toBeGreaterThanOrEqual(1);
			expect(screen.getAllByText("node-2").length).toBeGreaterThanOrEqual(1);
		});

		it("renders node values snapshot", () => {
			renderEvent(
				makeEvent({
					event_type: "mcts.iteration",
					payload: {
						iteration_number: 5,
						selected_node_id: "node-1",
						selection_path: ["node-0", "node-1"],
						expanded_count: 1,
						best_value_so_far: 0.9,
						node_values: { "node-0": 0.72, "node-1": 0.9 },
					},
				}),
			);
			expect(screen.getByText("Node values (2)")).toBeInTheDocument();
			expect(screen.getByText("0.72")).toBeInTheDocument();
			expect(screen.getByText("0.90")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "mcts.iteration",
					payload: {
						iteration_number: 4,
						expanded_count: 3,
						best_value_so_far: 0.82,
					},
				}),
			);
			expect(summary).toBe("MCTS #4: expanded 3, best=0.82");
		});

		it("handles minimal fields in summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "mcts.iteration",
					payload: { iteration_number: 1 },
				}),
			);
			expect(summary).toBe("MCTS #1");
		});
	});

	describe("mcts.backpropagation", () => {
		it("renders propagated value and path", () => {
			renderEvent(
				makeEvent({
					event_type: "mcts.backpropagation",
					payload: {
						propagated_value: 0.85,
						path_length: 3,
						updated_node_ids: ["node-5", "node-3", "node-0"],
					},
				}),
			);
			expect(screen.getAllByText("0.85").length).toBeGreaterThanOrEqual(1);
			expect(screen.getAllByText("3").length).toBeGreaterThanOrEqual(1);
			expect(screen.getByText("Updated nodes")).toBeInTheDocument();
			expect(screen.getAllByText("node-5").length).toBeGreaterThanOrEqual(1);
			expect(screen.getAllByText("node-3").length).toBeGreaterThanOrEqual(1);
			expect(screen.getAllByText("node-0").length).toBeGreaterThanOrEqual(1);
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "mcts.backpropagation",
					payload: { propagated_value: 0.75, path_length: 4 },
				}),
			);
			expect(summary).toBe("Backprop: 0.75 through 4 nodes");
		});

		it("handles missing fields in summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "mcts.backpropagation",
					payload: {},
				}),
			);
			expect(summary).toBe("Backpropagation");
		});
	});
});

// ---------------------------------------------------------------------------
// Renderer registration verification
// ---------------------------------------------------------------------------

describe("Default registrations include tree-search and MCTS renderers", () => {
	const registrations = createDefaultRegistrations();

	const expectedEventTypes = [
		"tree_search.node.created",
		"tree_search.node.evaluated",
		"tree_search.node.pruned",
		"tree_search.complete",
		"mcts.iteration",
		"mcts.backpropagation",
	];

	for (const eventType of expectedEventTypes) {
		it(`has a renderer for ${eventType}`, () => {
			const match = registrations.find((r) => r.matches(eventType) && r.priority >= 0);
			expect(match).toBeDefined();
			expect(match?.summary).toBeDefined();
		});
	}
});
