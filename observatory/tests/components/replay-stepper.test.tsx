import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { IterationMode } from "../../src/components/lats/iteration-bar";
import { IterationBar } from "../../src/components/lats/iteration-bar";
import { buildTreeAtIteration } from "../../src/hooks/build-tree-at-iteration";
import type { IterationData } from "../../src/hooks/use-lats-data";
import type { TraceEvent } from "../../src/types";
import { makeEvent } from "../fixtures/scenarios";

// ---------------------------------------------------------------------------
// Fixture: LATS events with 3 iterations
// ---------------------------------------------------------------------------

function makeLATSEvents(): TraceEvent[] {
	return [
		// Root node (before iteration 1)
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
		// Iteration 1: create n1
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
				observation: "Found 5 results...",
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
				node_values: { root: 0.6, n1: 0.6 },
			},
		}),
		makeEvent({
			event_type: "mcts.backpropagation",
			timestamp: "2026-03-05T10:00:04.500Z",
			payload: { propagated_value: 0.6, path_length: 1, updated_node_ids: ["root"] },
		}),
		// Iteration 2: create n2
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
				node_values: { root: 0.7, n1: 0.8, n2: 0.8 },
			},
		}),
		makeEvent({
			event_type: "mcts.backpropagation",
			timestamp: "2026-03-05T10:00:07.500Z",
			payload: { propagated_value: 0.8, path_length: 2, updated_node_ids: ["n1", "root"] },
		}),
		// Iteration 3: create n3 (terminal)
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
				node_values: { root: 0.78, n1: 0.85, n2: 0.95, n3: 0.95 },
			},
		}),
		makeEvent({
			event_type: "mcts.backpropagation",
			timestamp: "2026-03-05T10:00:10.500Z",
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
			},
		}),
	];
}

// ---------------------------------------------------------------------------
// buildTreeAtIteration Tests
// ---------------------------------------------------------------------------

describe("buildTreeAtIteration", () => {
	const events = makeLATSEvents();

	it("reconstructs tree at iteration 1 with correct node count", () => {
		const result = buildTreeAtIteration(events, 1);

		expect(result.root).not.toBeNull();
		expect(result.totalNodes).toBe(2); // root + n1
		expect(result.root?.id).toBe("root");
		expect(result.root?.children).toHaveLength(1);
		expect(result.root?.children[0].id).toBe("n1");
	});

	it("reconstructs tree at iteration 2 with correct node count", () => {
		const result = buildTreeAtIteration(events, 2);

		expect(result.totalNodes).toBe(3); // root + n1 + n2
		expect(result.root?.children[0].children).toHaveLength(1);
		expect(result.root?.children[0].children[0].id).toBe("n2");
	});

	it("reconstructs tree at iteration 3 with all nodes", () => {
		const result = buildTreeAtIteration(events, 3);

		expect(result.totalNodes).toBe(4); // root + n1 + n2 + n3
		const n2 = result.root?.children[0].children[0];
		expect(n2!.children).toHaveLength(1);
		expect(n2!.children[0].id).toBe("n3");
		expect(n2!.children[0].status).toBe("terminal");
	});

	it("applies node values from iteration snapshot", () => {
		const result = buildTreeAtIteration(events, 2);

		expect(result.root?.metadata.average_value).toBe(0.7);
		const n1 = result.root?.children[0];
		expect(n1!.metadata.average_value).toBe(0.8);
	});

	it("identifies new nodes for each iteration", () => {
		const result1 = buildTreeAtIteration(events, 1);
		// root and n1 are created before/during iteration 1
		expect(result1.newNodeIds.has("root")).toBe(true);
		expect(result1.newNodeIds.has("n1")).toBe(true);

		const result2 = buildTreeAtIteration(events, 2);
		// Only n2 is new in iteration 2
		expect(result2.newNodeIds.has("n2")).toBe(true);
		expect(result2.newNodeIds.has("root")).toBe(false);
		expect(result2.newNodeIds.has("n1")).toBe(false);

		const result3 = buildTreeAtIteration(events, 3);
		// Only n3 is new in iteration 3
		expect(result3.newNodeIds.has("n3")).toBe(true);
		expect(result3.newNodeIds.has("n2")).toBe(false);
	});

	it("records selection path from iteration event", () => {
		const result1 = buildTreeAtIteration(events, 1);
		expect(result1.selectionPath).toEqual(["root"]);

		const result2 = buildTreeAtIteration(events, 2);
		expect(result2.selectionPath).toEqual(["root", "n1"]);

		const result3 = buildTreeAtIteration(events, 3);
		expect(result3.selectionPath).toEqual(["root", "n1", "n2"]);
	});

	it("computes backprop deltas between iterations", () => {
		const result2 = buildTreeAtIteration(events, 2);
		// root went from 0.6 → 0.7 = +0.1
		expect(result2.backpropDeltas.get("root")).toBeCloseTo(0.1, 2);
		// n1 went from 0.6 → 0.8 = +0.2
		expect(result2.backpropDeltas.get("n1")).toBeCloseTo(0.2, 2);
	});

	it("records best value so far", () => {
		expect(buildTreeAtIteration(events, 1).bestValueSoFar).toBe(0.6);
		expect(buildTreeAtIteration(events, 2).bestValueSoFar).toBe(0.8);
		expect(buildTreeAtIteration(events, 3).bestValueSoFar).toBe(0.95);
	});

	it("builds solution path from best-valued node", () => {
		const result2 = buildTreeAtIteration(events, 2);
		// At iteration 2, n1 and n2 both have value 0.8, solution path includes best and ancestors
		expect(result2.solutionPath.has("n1")).toBe(true);
		expect(result2.solutionPath.has("root")).toBe(true);
		// The path includes at least root → n1 (or root → n1 → n2 depending on tie-breaking)
		expect(result2.solutionPath.size).toBeGreaterThanOrEqual(2);
	});

	it("returns empty result for non-existent iteration", () => {
		const result = buildTreeAtIteration(events, 99);
		expect(result.root).toBeNull();
		expect(result.totalNodes).toBe(0);
		expect(result.selectionPath).toEqual([]);
	});

	it("applies evaluation scores up to the iteration boundary", () => {
		const result1 = buildTreeAtIteration(events, 1);
		const n1 = result1.root?.children[0];
		expect(n1!.score).toBe(0.6); // evaluated before iteration 1
	});

	it("handles empty events", () => {
		const result = buildTreeAtIteration([], 1);
		expect(result.root).toBeNull();
		expect(result.totalNodes).toBe(0);
	});
});

// ---------------------------------------------------------------------------
// IterationBar Replay Mode Tests
// ---------------------------------------------------------------------------

describe("IterationBar", () => {
	const iterations: IterationData[] = [
		{
			iterationNumber: 1,
			selectedNodeId: "root",
			selectionPath: ["root"],
			expandedCount: 1,
			bestValueSoFar: 0.6,
			nodeValues: { root: 0.6 },
		},
		{
			iterationNumber: 2,
			selectedNodeId: "n1",
			selectionPath: ["root", "n1"],
			expandedCount: 1,
			bestValueSoFar: 0.8,
			nodeValues: { root: 0.7, n1: 0.8 },
		},
		{
			iterationNumber: 3,
			selectedNodeId: "n2",
			selectionPath: ["root", "n1", "n2"],
			expandedCount: 1,
			bestValueSoFar: 0.95,
			nodeValues: { root: 0.78, n1: 0.85, n2: 0.95 },
		},
	];

	afterEach(() => {
		vi.useRealTimers();
	});

	function renderBar(
		overrides: Partial<{
			selectedIteration: number | null;
			mode: IterationMode;
			onSelectIteration: (n: number | null) => void;
			onModeChange: (mode: IterationMode) => void;
		}> = {},
	) {
		const props = {
			iterations,
			selectedIteration: overrides.selectedIteration ?? null,
			onSelectIteration: overrides.onSelectIteration ?? vi.fn(),
			mode: overrides.mode ?? ("highlight" as IterationMode),
			onModeChange: overrides.onModeChange ?? vi.fn(),
		};
		return render(<IterationBar {...props} />);
	}

	it("renders mode toggle with highlight and replay buttons", () => {
		renderBar();
		expect(screen.getByTestId("mode-toggle")).toBeInTheDocument();
		expect(screen.getByLabelText("Highlight mode")).toBeInTheDocument();
		expect(screen.getByLabelText("Replay mode")).toBeInTheDocument();
	});

	it("calls onModeChange when replay button clicked", () => {
		const onModeChange = vi.fn();
		renderBar({ onModeChange });

		fireEvent.click(screen.getByLabelText("Replay mode"));
		expect(onModeChange).toHaveBeenCalledWith("replay");
	});

	it("calls onModeChange when highlight button clicked", () => {
		const onModeChange = vi.fn();
		renderBar({ mode: "replay", onModeChange });

		fireEvent.click(screen.getByLabelText("Highlight mode"));
		expect(onModeChange).toHaveBeenCalledWith("highlight");
	});

	it("does not show VCR controls in highlight mode", () => {
		renderBar({ mode: "highlight" });
		expect(screen.queryByTestId("vcr-controls")).not.toBeInTheDocument();
	});

	it("shows VCR controls in replay mode", () => {
		renderBar({ mode: "replay" });
		expect(screen.getByTestId("vcr-controls")).toBeInTheDocument();
	});

	it("renders all VCR buttons in replay mode", () => {
		renderBar({ mode: "replay" });
		expect(screen.getByLabelText("Jump to start")).toBeInTheDocument();
		expect(screen.getByLabelText("Step back")).toBeInTheDocument();
		expect(screen.getByLabelText("Play")).toBeInTheDocument();
		expect(screen.getByLabelText("Step forward")).toBeInTheDocument();
		expect(screen.getByLabelText("Jump to end")).toBeInTheDocument();
	});

	it("renders iteration slider in replay mode", () => {
		renderBar({ mode: "replay" });
		expect(screen.getByTestId("iteration-slider")).toBeInTheDocument();
	});

	it("renders speed selector in replay mode", () => {
		renderBar({ mode: "replay" });
		expect(screen.getByTestId("speed-selector")).toBeInTheDocument();
		expect(screen.getByLabelText("Speed 0.5×")).toBeInTheDocument();
		expect(screen.getByLabelText("Speed 1×")).toBeInTheDocument();
		expect(screen.getByLabelText("Speed 2×")).toBeInTheDocument();
	});

	it("step forward advances to next iteration", () => {
		const onSelect = vi.fn();
		renderBar({ mode: "replay", selectedIteration: 1, onSelectIteration: onSelect });

		fireEvent.click(screen.getByLabelText("Step forward"));
		expect(onSelect).toHaveBeenCalledWith(2);
	});

	it("step back goes to previous iteration", () => {
		const onSelect = vi.fn();
		renderBar({ mode: "replay", selectedIteration: 2, onSelectIteration: onSelect });

		fireEvent.click(screen.getByLabelText("Step back"));
		expect(onSelect).toHaveBeenCalledWith(1);
	});

	it("step forward with no selection goes to first iteration", () => {
		const onSelect = vi.fn();
		renderBar({ mode: "replay", selectedIteration: null, onSelectIteration: onSelect });

		fireEvent.click(screen.getByLabelText("Step forward"));
		expect(onSelect).toHaveBeenCalledWith(1);
	});

	it("jump to start selects first iteration", () => {
		const onSelect = vi.fn();
		renderBar({ mode: "replay", selectedIteration: 3, onSelectIteration: onSelect });

		fireEvent.click(screen.getByLabelText("Jump to start"));
		expect(onSelect).toHaveBeenCalledWith(1);
	});

	it("jump to end selects last iteration", () => {
		const onSelect = vi.fn();
		renderBar({ mode: "replay", selectedIteration: 1, onSelectIteration: onSelect });

		fireEvent.click(screen.getByLabelText("Jump to end"));
		expect(onSelect).toHaveBeenCalledWith(3);
	});

	it("slider scrubbing selects iteration", () => {
		const onSelect = vi.fn();
		renderBar({ mode: "replay", selectedIteration: 1, onSelectIteration: onSelect });

		const slider = screen.getByTestId("iteration-slider");
		fireEvent.change(slider, { target: { value: "2" } });
		expect(onSelect).toHaveBeenCalledWith(2);
	});

	it("play button toggles to pause", () => {
		const onSelect = vi.fn();
		renderBar({ mode: "replay", selectedIteration: 1, onSelectIteration: onSelect });

		// Click play
		fireEvent.click(screen.getByTestId("play-pause-button"));
		expect(screen.getByLabelText("Pause")).toBeInTheDocument();
	});

	it("pause button toggles to play", () => {
		const onSelect = vi.fn();
		renderBar({ mode: "replay", selectedIteration: 1, onSelectIteration: onSelect });

		// Click play then pause
		fireEvent.click(screen.getByTestId("play-pause-button"));
		fireEvent.click(screen.getByTestId("play-pause-button"));
		expect(screen.getByLabelText("Play")).toBeInTheDocument();
	});

	it("returns null for empty iterations", () => {
		const { container } = render(
			<IterationBar
				iterations={[]}
				selectedIteration={null}
				onSelectIteration={vi.fn()}
				mode="highlight"
				onModeChange={vi.fn()}
			/>,
		);
		expect(container.innerHTML).toBe("");
	});

	it("shows clear button when iteration is selected", () => {
		renderBar({ selectedIteration: 2 });
		expect(screen.getByLabelText("Clear iteration selection")).toBeInTheDocument();
	});

	it("does not show clear button when no iteration selected", () => {
		renderBar({ selectedIteration: null });
		expect(screen.queryByLabelText("Clear iteration selection")).not.toBeInTheDocument();
	});
});
