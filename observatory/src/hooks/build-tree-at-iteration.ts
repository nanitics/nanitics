import type { TraceEvent } from "../types";
import type { VisualTreeNode } from "../types/tree-types";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface TreeAtIterationResult {
	/** Root of the partial tree at this iteration (null if no nodes). */
	root: VisualTreeNode | null;
	/** Solution path node IDs (best node → root) based on best value so far. */
	solutionPath: Set<string>;
	/** Pruned node IDs (including descendants). */
	prunedNodeIds: Set<string>;
	/** Node IDs created in this specific iteration (new this step). */
	newNodeIds: Set<string>;
	/** Backpropagation node IDs for this iteration. */
	backpropNodeIds: Set<string>;
	/** Value deltas from backpropagation at this iteration (nodeId → delta). */
	backpropDeltas: Map<string, number>;
	/** Selection path for this iteration. */
	selectionPath: string[];
	/** Total number of nodes at this iteration. */
	totalNodes: number;
	/** Best value so far at this iteration. */
	bestValueSoFar: number;
}

// ---------------------------------------------------------------------------
// Iteration boundary detection
// ---------------------------------------------------------------------------

interface IterationBoundary {
	iterationNumber: number;
	/** Index in the full event array where this mcts.iteration event sits. */
	eventIndex: number;
	/** Timestamp of the mcts.iteration event. */
	timestamp: string;
	/** Node values snapshot from this iteration. */
	nodeValues: Record<string, number>;
	/** Selection path from the iteration event. */
	selectionPath: string[];
	/** Best value so far. */
	bestValueSoFar: number;
	/** Selected node ID. */
	selectedNodeId: string;
}

function findIterationBoundaries(events: TraceEvent[]): IterationBoundary[] {
	const boundaries: IterationBoundary[] = [];

	for (let i = 0; i < events.length; i++) {
		const event = events[i];
		if (event.event_type !== "mcts.iteration") continue;

		boundaries.push({
			iterationNumber: event.payload.iteration_number as number,
			eventIndex: i,
			timestamp: event.timestamp,
			nodeValues: (event.payload.node_values as Record<string, number>) ?? {},
			selectionPath: (event.payload.selection_path as string[]) ?? [],
			bestValueSoFar: (event.payload.best_value_so_far as number) ?? 0,
			selectedNodeId: event.payload.selected_node_id as string,
		});
	}

	boundaries.sort((a, b) => a.iterationNumber - b.iterationNumber);
	return boundaries;
}

// ---------------------------------------------------------------------------
// buildTreeAtIteration
// ---------------------------------------------------------------------------

/**
 * Reconstructs the tree state at a specific MCTS iteration.
 *
 * Given the full event stream and an iteration number N:
 * 1. Finds iteration boundary events (mcts.iteration)
 * 2. Includes all node creation events up to iteration N's boundary
 * 3. Applies evaluation/pruning events that occurred before the boundary
 * 4. Reads node values from MCTSIterationEvent.node_values snapshot
 * 5. Identifies nodes created specifically in iteration N
 * 6. Identifies backpropagation paths for iteration N
 *
 * This reconstruction is deterministic and stateless.
 */
export function buildTreeAtIteration(events: TraceEvent[], iterationNumber: number): TreeAtIterationResult {
	const boundaries = findIterationBoundaries(events);

	// Find the target iteration boundary
	const targetBoundary = boundaries.find((b) => b.iterationNumber === iterationNumber);

	if (!targetBoundary) {
		return {
			root: null,
			solutionPath: new Set(),
			prunedNodeIds: new Set(),
			newNodeIds: new Set(),
			backpropNodeIds: new Set(),
			backpropDeltas: new Map(),
			selectionPath: [],
			totalNodes: 0,
			bestValueSoFar: 0,
		};
	}

	// Find the previous iteration boundary (to determine which nodes are "new")
	const prevBoundary = boundaries.find((b) => b.iterationNumber === iterationNumber - 1);
	const prevBoundaryIndex = prevBoundary?.eventIndex ?? -1;

	// Collect events up to (and including) the target iteration boundary
	const cutoffIndex = targetBoundary.eventIndex;
	const relevantEvents = events.slice(0, cutoffIndex + 1);

	// 1. Build node map from creation events
	const nodeMap = new Map<string, VisualTreeNode>();
	const parentMap = new Map<string, string | null>();
	const newNodeIds = new Set<string>();

	for (let i = 0; i < relevantEvents.length; i++) {
		const event = relevantEvents[i];
		if (event.event_type !== "tree_search.node.created") continue;

		const { node_id, parent_id, depth, content, node_type, action, observation, is_terminal, is_failed } =
			event.payload as {
				node_id: string;
				parent_id: string | null;
				depth: number;
				content: string;
				node_type: string;
				action?: string;
				observation?: string;
				is_terminal?: boolean;
				is_failed?: boolean;
			};

		const status = is_failed ? "failed" : is_terminal ? "terminal" : "active";

		nodeMap.set(node_id, {
			id: node_id,
			parentId: parent_id ?? null,
			label: content,
			score: null,
			status,
			depth,
			metadata: {
				nodeType: node_type,
				content,
				action: action ?? null,
				observation: observation ?? null,
			},
			children: [],
		});
		parentMap.set(node_id, parent_id ?? null);

		// Node is "new" if created after the previous iteration boundary
		if (i > prevBoundaryIndex) {
			newNodeIds.add(node_id);
		}
	}

	// 2. Apply evaluation events
	for (const event of relevantEvents) {
		if (event.event_type !== "tree_search.node.evaluated") continue;
		const { node_id, score, is_terminal } = event.payload as {
			node_id: string;
			score: number;
			is_terminal: boolean;
		};
		const node = nodeMap.get(node_id);
		if (node) {
			node.score = score;
			if (is_terminal && node.status === "active") {
				node.status = "terminal";
			}
		}
	}

	// 3. Apply pruning events
	const prunedNodeIds = new Set<string>();
	for (const event of relevantEvents) {
		if (event.event_type !== "tree_search.node.pruned") continue;
		const { node_id, reason } = event.payload as {
			node_id: string;
			reason: string;
		};
		const node = nodeMap.get(node_id);
		if (node) {
			node.status = "pruned";
			node.metadata.pruneReason = reason;
			prunedNodeIds.add(node_id);
		}
	}

	// Propagate pruned status to descendants
	for (const nodeId of [...nodeMap.keys()]) {
		if (isAncestorPruned(nodeId, parentMap, prunedNodeIds)) {
			prunedNodeIds.add(nodeId);
		}
	}

	// 4. Apply node values from iteration snapshot
	for (const [nodeId, avgValue] of Object.entries(targetBoundary.nodeValues)) {
		const node = nodeMap.get(nodeId);
		if (node) {
			node.metadata.average_value = avgValue;
		}
	}

	// Also compute backprop deltas using node_values snapshots
	const backpropDeltas = new Map<string, number>();

	// 5. Collect backpropagation node IDs for this iteration
	// Backprop events for iteration N come AFTER the mcts.iteration event
	const nextBoundary = boundaries.find((b) => b.iterationNumber === iterationNumber + 1);
	const backpropEndIndex = nextBoundary?.eventIndex ?? events.length;
	const backpropNodeIds = new Set<string>();

	const backpropSlice = events.slice(cutoffIndex + 1, backpropEndIndex);
	for (const event of backpropSlice) {
		if (event.event_type !== "mcts.backpropagation") continue;
		const updatedIds = event.payload.updated_node_ids as string[];
		for (const id of updatedIds) {
			backpropNodeIds.add(id);
		}
	}

	// Compute value deltas for backpropagated nodes
	if (prevBoundary) {
		for (const nodeId of backpropNodeIds) {
			const prevValue = prevBoundary.nodeValues[nodeId] ?? 0;
			const currValue = targetBoundary.nodeValues[nodeId] ?? 0;
			const delta = currValue - prevValue;
			if (Math.abs(delta) > 0.001) {
				backpropDeltas.set(nodeId, delta);
			}
		}
	}

	// 6. Build tree from parent-child relationships
	let root: VisualTreeNode | null = null;
	for (const node of nodeMap.values()) {
		if (node.parentId == null) {
			root = node;
		} else {
			const parent = nodeMap.get(node.parentId);
			if (parent) {
				parent.children.push(node);
			}
		}
	}

	// Sort children
	if (root) {
		sortChildren(root);
	}

	// 7. Build solution path from best node
	const solutionPath = new Set<string>();
	// Find the node with the highest score at this iteration
	let bestNodeId: string | null = null;
	let bestScore = -1;
	for (const [nodeId, value] of Object.entries(targetBoundary.nodeValues)) {
		if (value > bestScore && nodeMap.has(nodeId)) {
			bestScore = value;
			bestNodeId = nodeId;
		}
	}
	if (bestNodeId) {
		let current: string | null = bestNodeId;
		while (current != null) {
			solutionPath.add(current);
			current = parentMap.get(current) ?? null;
		}
	}

	return {
		root,
		solutionPath,
		prunedNodeIds,
		newNodeIds,
		backpropNodeIds,
		backpropDeltas,
		selectionPath: targetBoundary.selectionPath,
		totalNodes: nodeMap.size,
		bestValueSoFar: targetBoundary.bestValueSoFar,
	};
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function isAncestorPruned(nodeId: string, parentMap: Map<string, string | null>, prunedIds: Set<string>): boolean {
	let current = parentMap.get(nodeId) ?? null;
	while (current != null) {
		if (prunedIds.has(current)) return true;
		current = parentMap.get(current) ?? null;
	}
	return false;
}

function sortChildren(node: VisualTreeNode): void {
	node.children.sort((a, b) => a.depth - b.depth || a.id.localeCompare(b.id));
	for (const child of node.children) {
		sortChildren(child);
	}
}
