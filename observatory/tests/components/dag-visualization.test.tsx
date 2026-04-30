import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DAGVisualization } from "../../src/components/dag/dag-visualization";
import type { DAGEdge, DAGLayout, DAGNode } from "../../src/types/dag-types";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

function makeNode(overrides: Partial<DAGNode> & { id: string }): DAGNode {
	return {
		label: overrides.id,
		status: "pending",
		stepType: "function",
		durationMs: null,
		agentSpanId: null,
		parallelGroup: null,
		metadata: {},
		x: 0,
		y: 0,
		width: 220,
		height: 80,
		...overrides,
	};
}

function makeLayout(nodes: DAGNode[], edges: DAGEdge[] = []): DAGLayout {
	return {
		nodes,
		edges,
		width: 800,
		height: 600,
	};
}

function defaultRender(node: DAGNode, isSelected: boolean) {
	return (
		<div data-testid={`content-${node.id}`} className={isSelected ? "selected" : ""}>
			{node.label}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("DAGVisualization", () => {
	beforeEach(() => {
		vi.restoreAllMocks();
	});

	it("shows empty state when layout has no nodes", () => {
		const layout = makeLayout([]);
		render(<DAGVisualization layout={layout} renderNodeContent={defaultRender} nodeWidth={220} nodeHeight={80} />);
		expect(screen.getByText("No workflow data to display.")).toBeInTheDocument();
	});

	it("renders all nodes", () => {
		const nodes = [
			makeNode({ id: "a", x: 100, y: 50 }),
			makeNode({ id: "b", x: 100, y: 200 }),
			makeNode({ id: "c", x: 250, y: 200 }),
		];
		const layout = makeLayout(nodes);

		render(<DAGVisualization layout={layout} renderNodeContent={defaultRender} nodeWidth={220} nodeHeight={80} />);

		expect(screen.getByTestId("node-a")).toBeInTheDocument();
		expect(screen.getByTestId("node-b")).toBeInTheDocument();
		expect(screen.getByTestId("node-c")).toBeInTheDocument();
	});

	it("renders node content via renderNodeContent callback", () => {
		const nodes = [makeNode({ id: "step-1", label: "Research", x: 100, y: 50 })];
		const layout = makeLayout(nodes);

		render(<DAGVisualization layout={layout} renderNodeContent={defaultRender} nodeWidth={220} nodeHeight={80} />);

		expect(screen.getByTestId("content-step-1")).toBeInTheDocument();
		expect(screen.getByText("Research")).toBeInTheDocument();
	});

	it("renders edges between connected nodes", () => {
		const nodes = [makeNode({ id: "a", x: 100, y: 50 }), makeNode({ id: "b", x: 100, y: 200 })];
		const edges: DAGEdge[] = [{ source: "a", target: "b" }];
		const layout = makeLayout(nodes, edges);

		render(<DAGVisualization layout={layout} renderNodeContent={defaultRender} nodeWidth={220} nodeHeight={80} />);

		expect(screen.getByTestId("edge-a-b")).toBeInTheDocument();
	});

	it("calls onNodeSelect when a node is clicked", () => {
		const onSelect = vi.fn();
		const nodes = [makeNode({ id: "step-1", x: 100, y: 50 })];
		const layout = makeLayout(nodes);

		render(
			<DAGVisualization
				layout={layout}
				renderNodeContent={defaultRender}
				nodeWidth={220}
				nodeHeight={80}
				onNodeSelect={onSelect}
			/>,
		);

		fireEvent.click(screen.getByTestId("node-step-1"));
		expect(onSelect).toHaveBeenCalledWith("step-1");
	});

	it("passes isSelected=true for the selected node", () => {
		const nodes = [makeNode({ id: "a", x: 100, y: 50 }), makeNode({ id: "b", x: 100, y: 200 })];
		const layout = makeLayout(nodes);

		render(
			<DAGVisualization
				layout={layout}
				renderNodeContent={defaultRender}
				nodeWidth={220}
				nodeHeight={80}
				selectedNodeId="a"
			/>,
		);

		expect(screen.getByTestId("content-a")).toHaveClass("selected");
		expect(screen.getByTestId("content-b")).not.toHaveClass("selected");
	});

	it("dims blocked nodes downstream of error nodes", () => {
		const nodes = [
			makeNode({ id: "a", status: "completed", x: 100, y: 50 }),
			makeNode({ id: "b", status: "error", x: 100, y: 200 }),
			makeNode({ id: "c", status: "pending", x: 100, y: 350 }),
		];
		const edges: DAGEdge[] = [
			{ source: "a", target: "b" },
			{ source: "b", target: "c" },
		];
		const layout = makeLayout(nodes, edges);

		render(<DAGVisualization layout={layout} renderNodeContent={defaultRender} nodeWidth={220} nodeHeight={80} />);

		// The blocked node should have opacity-40 class
		const blockedNode = screen.getByTestId("node-c");
		expect(blockedNode).toHaveClass("opacity-40");

		// Error node itself should not be dimmed
		const errorNode = screen.getByTestId("node-b");
		expect(errorNode).not.toHaveClass("opacity-40");
	});

	it("uses dashed edges for blocked connections", () => {
		const nodes = [
			makeNode({ id: "a", status: "error", x: 100, y: 50 }),
			makeNode({ id: "b", status: "pending", x: 100, y: 200 }),
		];
		const edges: DAGEdge[] = [{ source: "a", target: "b" }];
		const layout = makeLayout(nodes, edges);

		render(<DAGVisualization layout={layout} renderNodeContent={defaultRender} nodeWidth={220} nodeHeight={80} />);

		const edge = screen.getByTestId("edge-a-b");
		expect(edge).toHaveAttribute("stroke-dasharray", "6 4");
	});

	it("renders zoom controls", () => {
		const nodes = [makeNode({ id: "a", x: 100, y: 50 })];
		const layout = makeLayout(nodes);

		render(<DAGVisualization layout={layout} renderNodeContent={defaultRender} nodeWidth={220} nodeHeight={80} />);

		expect(screen.getByLabelText("Zoom in")).toBeInTheDocument();
		expect(screen.getByLabelText("Zoom out")).toBeInTheDocument();
		expect(screen.getByLabelText("Fit to content")).toBeInTheDocument();
	});

	it("highlights critical path edges when showCriticalPath is true", () => {
		const nodes = [
			makeNode({ id: "a", status: "completed", x: 100, y: 50, durationMs: 500 }),
			makeNode({ id: "b", status: "completed", x: 100, y: 200, durationMs: 300 }),
		];
		const edges: DAGEdge[] = [{ source: "a", target: "b", isCriticalPath: true }];
		const layout = makeLayout(nodes, edges);

		const { rerender } = render(
			<DAGVisualization
				layout={layout}
				renderNodeContent={defaultRender}
				nodeWidth={220}
				nodeHeight={80}
				showCriticalPath={false}
			/>,
		);

		// Without showCriticalPath, edge should have default class
		let edge = screen.getByTestId("edge-a-b");
		expect(edge).toHaveClass("stroke-muted-foreground/40");

		// With showCriticalPath, edge should have critical path class
		rerender(
			<DAGVisualization
				layout={layout}
				renderNodeContent={defaultRender}
				nodeWidth={220}
				nodeHeight={80}
				showCriticalPath={true}
			/>,
		);

		edge = screen.getByTestId("edge-a-b");
		expect(edge).toHaveClass("stroke-primary");
	});
});
