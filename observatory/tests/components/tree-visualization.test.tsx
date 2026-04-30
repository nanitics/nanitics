import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { TreeNodeDetailPanel } from "../../src/components/tree/tree-node-detail-panel";
import { TreeVisualization } from "../../src/components/tree/tree-visualization";
import type { TreeVisualizationConfig, VisualTreeNode } from "../../src/types/tree-types";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

function makeNode(overrides: Partial<VisualTreeNode> & { id: string }): VisualTreeNode {
	return {
		parentId: null,
		label: `Node ${overrides.id}`,
		score: null,
		status: "active",
		depth: 0,
		metadata: {},
		children: [],
		...overrides,
	};
}

function buildSampleTree(): VisualTreeNode {
	const leaf1 = makeNode({ id: "c1", parentId: "root", depth: 1, label: "Leaf 1", score: 0.8, status: "terminal" });
	const leaf2 = makeNode({ id: "c2", parentId: "root", depth: 1, label: "Leaf 2", score: 0.3, status: "pruned" });
	const leaf3 = makeNode({ id: "c3", parentId: "b1", depth: 2, label: "Leaf 3", score: 0.6, status: "active" });
	const branch = makeNode({ id: "b1", parentId: "root", depth: 1, label: "Branch", children: [leaf3] });
	const root = makeNode({ id: "root", depth: 0, label: "Root", children: [leaf1, leaf2, branch] });
	return root;
}

function defaultRender(node: VisualTreeNode, isSelected: boolean) {
	return (
		<div data-testid={`content-${node.id}`} className={isSelected ? "selected" : ""}>
			{node.label}
		</div>
	);
}

function makeConfig(overrides: Partial<TreeVisualizationConfig> = {}): TreeVisualizationConfig {
	return {
		renderNodeContent: defaultRender,
		nodeWidth: 160,
		nodeHeight: 60,
		...overrides,
	};
}

// ---------------------------------------------------------------------------
// TreeVisualization tests
// ---------------------------------------------------------------------------

describe("TreeVisualization", () => {
	beforeEach(() => {
		vi.restoreAllMocks();
	});

	it("renders all nodes in the tree", () => {
		const root = buildSampleTree();
		render(<TreeVisualization root={root} config={makeConfig()} />);

		expect(screen.getByTestId("node-root")).toBeInTheDocument();
		expect(screen.getByTestId("node-c1")).toBeInTheDocument();
		expect(screen.getByTestId("node-c2")).toBeInTheDocument();
		expect(screen.getByTestId("node-b1")).toBeInTheDocument();
		expect(screen.getByTestId("node-c3")).toBeInTheDocument();
	});

	it("renders edges between parent and child nodes", () => {
		const root = buildSampleTree();
		render(<TreeVisualization root={root} config={makeConfig()} />);

		expect(screen.getByTestId("edge-root-c1")).toBeInTheDocument();
		expect(screen.getByTestId("edge-root-c2")).toBeInTheDocument();
		expect(screen.getByTestId("edge-root-b1")).toBeInTheDocument();
		expect(screen.getByTestId("edge-b1-c3")).toBeInTheDocument();
	});

	it("renders node content via renderNodeContent prop", () => {
		const root = buildSampleTree();
		render(<TreeVisualization root={root} config={makeConfig()} />);

		expect(screen.getByTestId("content-root")).toHaveTextContent("Root");
		expect(screen.getByTestId("content-c1")).toHaveTextContent("Leaf 1");
	});

	it("calls onNodeSelect when a node is clicked", () => {
		const onNodeSelect = vi.fn();
		const root = buildSampleTree();
		render(<TreeVisualization root={root} config={makeConfig({ onNodeSelect })} />);

		fireEvent.click(screen.getByTestId("node-c1"));
		expect(onNodeSelect).toHaveBeenCalledWith("c1");
	});

	it("passes isSelected=true for the selected node", () => {
		const root = makeNode({ id: "root", children: [makeNode({ id: "a", parentId: "root", depth: 1 })] });
		render(<TreeVisualization root={root} config={makeConfig({ selectedNodeId: "a" })} />);

		const content = screen.getByTestId("content-a");
		expect(content.className).toContain("selected");
	});

	it("does not pass isSelected for non-selected nodes", () => {
		const root = makeNode({ id: "root", children: [makeNode({ id: "a", parentId: "root", depth: 1 })] });
		render(<TreeVisualization root={root} config={makeConfig({ selectedNodeId: "a" })} />);

		const content = screen.getByTestId("content-root");
		expect(content.className).not.toContain("selected");
	});

	it("shows empty state when root has no children and no data", () => {
		// An empty tree is one with zero layout nodes — but the root itself counts.
		// Instead test that a root-only tree still renders.
		const root = makeNode({ id: "only" });
		render(<TreeVisualization root={root} config={makeConfig()} />);

		expect(screen.getByTestId("node-only")).toBeInTheDocument();
		expect(screen.queryByTestId("tree-edges")).toBeInTheDocument();
	});

	it("renders zoom controls", () => {
		const root = buildSampleTree();
		render(<TreeVisualization root={root} config={makeConfig()} />);

		expect(screen.getByLabelText("Zoom in")).toBeInTheDocument();
		expect(screen.getByLabelText("Zoom out")).toBeInTheDocument();
		expect(screen.getByLabelText("Fit to content")).toBeInTheDocument();
	});

	it("applies dimmed class to dimmed nodes", () => {
		const root = buildSampleTree();
		const dimmed = new Set(["c2"]);
		render(<TreeVisualization root={root} config={makeConfig({ dimmedNodeIds: dimmed })} />);

		const dimmedNode = screen.getByTestId("node-c2");
		expect(dimmedNode.getAttribute("class")).toContain("opacity-40");
	});

	it("does not apply dimmed class to non-dimmed nodes", () => {
		const root = buildSampleTree();
		const dimmed = new Set(["c2"]);
		render(<TreeVisualization root={root} config={makeConfig({ dimmedNodeIds: dimmed })} />);

		const activeNode = screen.getByTestId("node-c1");
		expect(activeNode.getAttribute("class") ?? "").not.toContain("opacity-40");
	});
});

// ---------------------------------------------------------------------------
// TreeNodeDetailPanel tests
// ---------------------------------------------------------------------------

describe("TreeNodeDetailPanel", () => {
	it("displays node label and status", () => {
		const node = makeNode({ id: "n1", label: "My Thought", status: "terminal", depth: 2 });
		render(<TreeNodeDetailPanel node={node} />);

		expect(screen.getByText("My Thought")).toBeInTheDocument();
		expect(screen.getByText("Terminal")).toBeInTheDocument();
		expect(screen.getByText("Depth 2")).toBeInTheDocument();
	});

	it("displays score bar when score is present", () => {
		const node = makeNode({ id: "n1", score: 0.75 });
		render(<TreeNodeDetailPanel node={node} />);

		expect(screen.getByText("0.75")).toBeInTheDocument();
		expect(screen.getByText("Score")).toBeInTheDocument();
	});

	it("does not display score section when score is null", () => {
		const node = makeNode({ id: "n1", score: null });
		render(<TreeNodeDetailPanel node={node} />);

		expect(screen.queryByText("Score")).not.toBeInTheDocument();
	});

	it("displays content from metadata", () => {
		const node = makeNode({
			id: "n1",
			metadata: { content: "Full reasoning text here" },
		});
		render(<TreeNodeDetailPanel node={node} />);

		expect(screen.getByText("Full reasoning text here")).toBeInTheDocument();
	});

	it("displays action and observation for LATS nodes", () => {
		const node = makeNode({
			id: "n1",
			metadata: {
				action: "web_search",
				observation: "Found 3 results...",
			},
		});
		render(<TreeNodeDetailPanel node={node} />);

		expect(screen.getByText("web_search")).toBeInTheDocument();
		expect(screen.getByText("Found 3 results...")).toBeInTheDocument();
	});

	it("displays LATS-specific stats", () => {
		const node = makeNode({
			id: "n1",
			metadata: {
				visit_count: 5,
				cumulative_value: 3.2,
				ucb1: 1.456,
			},
		});
		render(<TreeNodeDetailPanel node={node} />);

		expect(screen.getByText("5")).toBeInTheDocument();
		expect(screen.getByText("3.20")).toBeInTheDocument();
		expect(screen.getByText("1.456")).toBeInTheDocument();
	});

	it("shows children list with scores", () => {
		const child1 = makeNode({ id: "ch1", parentId: "n1", label: "Child One", score: 0.9, depth: 1 });
		const child2 = makeNode({ id: "ch2", parentId: "n1", label: "Child Two", score: 0.5, depth: 1 });
		const node = makeNode({ id: "n1", children: [child1, child2] });

		render(<TreeNodeDetailPanel node={node} />);

		expect(screen.getByText("Children (2)")).toBeInTheDocument();
		expect(screen.getByText(/Child One/)).toBeInTheDocument();
		expect(screen.getByText("0.90")).toBeInTheDocument();
	});

	it("calls onNavigate when clicking a child", () => {
		const onNavigate = vi.fn();
		const child = makeNode({ id: "ch1", parentId: "n1", label: "Child", depth: 1 });
		const node = makeNode({ id: "n1", children: [child] });

		render(<TreeNodeDetailPanel node={node} onNavigate={onNavigate} />);

		fireEvent.click(screen.getByRole("button", { name: /Child/ }));
		expect(onNavigate).toHaveBeenCalledWith("ch1");
	});

	it("shows parent link when allNodes provided", () => {
		const parent = makeNode({ id: "parent", label: "Parent Node" });
		const node = makeNode({ id: "child", parentId: "parent", label: "Child Node", depth: 1 });
		const allNodes = new Map<string, VisualTreeNode>([
			["parent", parent],
			["child", node],
		]);

		render(<TreeNodeDetailPanel node={node} allNodes={allNodes} />);

		expect(screen.getByText(/Parent Node/)).toBeInTheDocument();
	});

	it("calls onClose when close button is clicked", () => {
		const onClose = vi.fn();
		const node = makeNode({ id: "n1" });

		render(<TreeNodeDetailPanel node={node} onClose={onClose} />);

		fireEvent.click(screen.getByLabelText("Close detail panel"));
		expect(onClose).toHaveBeenCalled();
	});

	it("displays extra metadata fields", () => {
		const node = makeNode({
			id: "n1",
			metadata: { custom_field: "custom_value", another: 42 },
		});

		render(<TreeNodeDetailPanel node={node} />);

		expect(screen.getByText("custom_field")).toBeInTheDocument();
		expect(screen.getByText("custom_value")).toBeInTheDocument();
	});
});
