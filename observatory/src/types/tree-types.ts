/** Generic tree node for the visualization engine. */
export interface VisualTreeNode {
	id: string;
	parentId: string | null;
	label: string;
	/** Evaluation score (0–1 range). */
	score: number | null;
	status: "active" | "pruned" | "terminal" | "failed" | "expanding";
	depth: number;
	/** Agent-specific data (action, visit count, UCB1, etc.). */
	metadata: Record<string, unknown>;
	children: VisualTreeNode[];

	/** Populated by d3-hierarchy layout. */
	x?: number;
	y?: number;
}

/** Configuration for the tree visualization component. */
export interface TreeVisualizationConfig {
	/** Render prop for node content — agent views customize this. */
	renderNodeContent: (node: VisualTreeNode, isSelected: boolean) => React.ReactNode;
	/** Node width for layout spacing (px). */
	nodeWidth: number;
	/** Node height for layout spacing (px). */
	nodeHeight: number;
	/** Node IDs to highlight (solution path, iteration selection, etc.). */
	highlightedNodeIds?: Set<string>;
	/** Visual style for highlighted nodes. */
	highlightStyle?: "path" | "glow" | "outline";
	/** Node IDs to dim (pruned branches, etc.). */
	dimmedNodeIds?: Set<string>;
	/** Currently selected node ID (click). */
	selectedNodeId?: string;
	/** Callback when a node is clicked. */
	onNodeSelect?: (nodeId: string) => void;
}
