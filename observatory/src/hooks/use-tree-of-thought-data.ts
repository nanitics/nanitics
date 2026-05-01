import { useMemo } from "react";
import type { TraceEvent } from "../types";
import type { VisualTreeNode } from "../types/tree-types";

export interface TreeOfThoughtData {
	/** Root of the visual tree (null if no nodes). */
	root: VisualTreeNode | null;
	/** Node IDs on the solution path (selected node → root). */
	solutionPath: Set<string>;
	/** Node IDs that are pruned (including their descendants). */
	prunedNodeIds: Set<string>;
	/** Search metadata. */
	searchStrategy: string | null;
	terminationReason: string | null;
	totalNodes: number;
	maxDepth: number;
	selectedNodeId: string | null;
}

/**
 * Transforms raw trace events into a VisualTreeNode tree for TreeOfThought agent views.
 *
 * Steps:
 * 1. Collect `tree_search.node.created` events → build node map
 * 2. Apply `tree_search.node.evaluated` events → set scores
 * 3. Apply `tree_search.node.pruned` events → set status to "pruned"
 * 4. Read `tree_search.complete` event → identify selected node, metadata
 * 5. Build tree from parent-child relationships
 * 6. Mark solution path (selected node → root)
 */
export function useTreeOfThoughtData(events: TraceEvent[]): TreeOfThoughtData {
	return useMemo(() => buildTreeOfThoughtData(events), [events]);
}

/** Pure function for testability — no React dependency. */
export function buildTreeOfThoughtData(events: TraceEvent[]): TreeOfThoughtData {
	// 1. Collect node.created events
	const nodeMap = new Map<string, VisualTreeNode>();
	const parentMap = new Map<string, string | null>();

	for (const event of events) {
		if (event.event_type !== "tree_search.node.created") continue;
		const { node_id, parent_id, depth, content, node_type, is_terminal, is_failed } = event.payload as {
			node_id: string;
			parent_id: string | null;
			depth: number;
			content: string;
			node_type: string;
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
			},
			children: [],
		});
		parentMap.set(node_id, parent_id ?? null);
	}

	// 2. Apply evaluation scores
	for (const event of events) {
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

	// 3. Apply pruning
	const prunedNodeIds = new Set<string>();
	for (const event of events) {
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
	const allNodeIds = [...nodeMap.keys()];
	for (const nodeId of allNodeIds) {
		if (isAncestorPruned(nodeId, parentMap, prunedNodeIds)) {
			prunedNodeIds.add(nodeId);
		}
	}

	// 4. Read complete event
	const completeEvent = events.find((e) => e.event_type === "tree_search.complete");
	const searchStrategy = (completeEvent?.payload.search_strategy as string) ?? null;
	const terminationReason = (completeEvent?.payload.termination_reason as string) ?? null;
	const selectedNodeId = (completeEvent?.payload.selected_node_id as string) ?? null;

	// 5. Build tree from parent-child relationships
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

	// Sort children by depth then id for consistent ordering
	if (root) {
		sortChildren(root);
	}

	// 6. Build solution path
	const solutionPath = new Set<string>();
	if (selectedNodeId) {
		let current: string | null = selectedNodeId;
		while (current != null) {
			solutionPath.add(current);
			current = parentMap.get(current) ?? null;
		}
	}

	return {
		root,
		solutionPath,
		prunedNodeIds,
		searchStrategy,
		terminationReason,
		totalNodes: nodeMap.size,
		maxDepth: completeEvent ? ((completeEvent.payload.max_depth_reached as number) ?? 0) : 0,
		selectedNodeId,
	};
}

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
