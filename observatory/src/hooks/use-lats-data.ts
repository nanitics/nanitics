import { useMemo } from "react";
import type { TraceEvent } from "../types";
import type { VisualTreeNode } from "../types/tree-types";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface IterationData {
	iterationNumber: number;
	selectedNodeId: string;
	selectionPath: string[];
	expandedCount: number;
	bestValueSoFar: number;
	/** Per-node average values at this iteration. */
	nodeValues: Record<string, number>;
}

export interface BackpropagationData {
	propagatedValue: number;
	pathLength: number;
	updatedNodeIds: string[];
	/** Iteration number this backpropagation belongs to (inferred from ordering). */
	iterationNumber: number;
}

export interface LATSData {
	/** Root of the visual tree (null if no nodes). */
	root: VisualTreeNode | null;
	/** Solution path node IDs (best node → root). */
	solutionPath: Set<string>;
	/** Pruned node IDs (including descendants). */
	prunedNodeIds: Set<string>;
	/** All iteration events in order. */
	iterations: IterationData[];
	/** All backpropagation events associated with iterations. */
	backpropagations: BackpropagationData[];
	/** Episodic memory recall events (if present). */
	episodicRecalls: TraceEvent[];
	/** Search metadata. */
	terminationReason: string | null;
	bestNodeId: string | null;
	totalNodes: number;
	maxIterations: number;
	explorationConstant: number | null;
}

/**
 * Transforms raw trace events into a VisualTreeNode tree plus iteration
 * timeline data for the LATS agent view.
 */
export function useLATSData(events: TraceEvent[]): LATSData {
	return useMemo(() => buildLATSData(events), [events]);
}

/** Pure function for testability. */
export function buildLATSData(events: TraceEvent[]): LATSData {
	// 1. Collect node.created events → build node map
	const nodeMap = new Map<string, VisualTreeNode>();
	const parentMap = new Map<string, string | null>();

	for (const event of events) {
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
	for (const nodeId of [...nodeMap.keys()]) {
		if (isAncestorPruned(nodeId, parentMap, prunedNodeIds)) {
			prunedNodeIds.add(nodeId);
		}
	}

	// 4. Collect iteration events
	const iterations: IterationData[] = [];
	for (const event of events) {
		if (event.event_type !== "mcts.iteration") continue;
		const { iteration_number, selected_node_id, selection_path, expanded_count, best_value_so_far, node_values } =
			event.payload as {
				iteration_number: number;
				selected_node_id: string;
				selection_path: string[];
				expanded_count: number;
				best_value_so_far: number;
				node_values?: Record<string, number>;
			};

		iterations.push({
			iterationNumber: iteration_number,
			selectedNodeId: selected_node_id,
			selectionPath: selection_path,
			expandedCount: expanded_count,
			bestValueSoFar: best_value_so_far,
			nodeValues: node_values ?? {},
		});
	}

	// Sort iterations by number
	iterations.sort((a, b) => a.iterationNumber - b.iterationNumber);

	// 5. Collect backpropagation events and associate with iterations
	const backpropagations: BackpropagationData[] = [];
	const backpropEvents = events.filter((e) => e.event_type === "mcts.backpropagation");

	// Associate backpropagation events with iterations by ordering:
	// Each backprop follows its corresponding iteration event
	let backpropIdx = 0;
	const iterationEvents = events.filter((e) => e.event_type === "mcts.iteration");

	for (const iterEvent of iterationEvents) {
		const iterNum = iterEvent.payload.iteration_number as number;
		const iterTimestamp = new Date(iterEvent.timestamp).getTime();

		// Find next iteration timestamp (or Infinity for last)
		const iterIndex = iterationEvents.indexOf(iterEvent);
		const nextIterTimestamp =
			iterIndex < iterationEvents.length - 1 ? new Date(iterationEvents[iterIndex + 1].timestamp).getTime() : Infinity;

		// Backprop events between this iteration and the next belong to this iteration
		while (backpropIdx < backpropEvents.length) {
			const bp = backpropEvents[backpropIdx];
			const bpTime = new Date(bp.timestamp).getTime();
			if (bpTime >= iterTimestamp && bpTime < nextIterTimestamp) {
				backpropagations.push({
					propagatedValue: bp.payload.propagated_value as number,
					pathLength: bp.payload.path_length as number,
					updatedNodeIds: bp.payload.updated_node_ids as string[],
					iterationNumber: iterNum,
				});
				backpropIdx++;
			} else {
				break;
			}
		}
	}

	// 6. Read complete event
	const completeEvent = events.find((e) => e.event_type === "tree_search.complete");
	const terminationReason = (completeEvent?.payload.termination_reason as string) ?? null;
	const bestNodeId = (completeEvent?.payload.selected_node_id as string) ?? null;

	// 7. Check for episodic memory recall events
	const episodicRecalls = events.filter((e) => e.event_type === "memory.episode.recall");

	// Build tree from parent-child relationships
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

	// Build solution path
	const solutionPath = new Set<string>();
	if (bestNodeId) {
		let current: string | null = bestNodeId;
		while (current != null) {
			solutionPath.add(current);
			current = parentMap.get(current) ?? null;
		}
	}

	// Extract exploration constant from agent.start event
	const agentStart = events.find((e) => e.event_type === "agent.start");
	const explorationConstant = (agentStart?.payload.exploration_constant as number) ?? null;

	return {
		root,
		solutionPath,
		prunedNodeIds,
		iterations,
		backpropagations,
		episodicRecalls,
		terminationReason,
		bestNodeId,
		totalNodes: nodeMap.size,
		maxIterations: iterations.length,
		explorationConstant,
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
