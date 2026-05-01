import { useCallback, useMemo, useState } from "react";
import { buildTreeAtIteration } from "../../hooks/build-tree-at-iteration";
import { useLATSData } from "../../hooks/use-lats-data";
import type { AgentViewProps } from "../../registry/agent-view-registry";
import type { VisualTreeNode } from "../../types/tree-types";
import type { IterationMode } from "../lats/iteration-bar";
import { IterationBar } from "../lats/iteration-bar";
import { TreeNodeDetailPanel } from "../tree/tree-node-detail-panel";
import { TreeVisualization } from "../tree/tree-visualization";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const NODE_WIDTH = 180;
const NODE_HEIGHT = 80;

// ---------------------------------------------------------------------------
// Node content render prop
// ---------------------------------------------------------------------------

function LATSNodeContent(node: VisualTreeNode, isSelected: boolean) {
	const action = node.metadata.action as string | null;
	const visitCount = node.metadata.visit_count as number | null;
	const avgValue = node.metadata.average_value as number | null;

	const actionLabel = action ? action : node.status === "terminal" ? "Terminal" : "Thought";

	const actionColor = action
		? "bg-info-muted text-info"
		: node.status === "terminal"
			? "bg-success-muted text-success"
			: "bg-muted text-muted-foreground";

	const valueColor =
		avgValue != null ? (avgValue >= 0.7 ? "text-success" : avgValue >= 0.4 ? "text-warning" : "text-destructive") : "";

	return (
		<div
			className={`rounded-lg border p-2 text-left transition-colors ${
				isSelected
					? "border-primary bg-primary/5"
					: node.status === "failed"
						? "border-destructive/50 bg-background hover:border-destructive"
						: "border-border bg-background hover:border-primary/50"
			}`}
			style={{ width: NODE_WIDTH, minHeight: NODE_HEIGHT }}
		>
			{/* Action badge row */}
			<div className="flex items-center justify-between mb-1">
				<span className={`text-[10px] px-1.5 py-0.5 rounded ${actionColor}`}>{actionLabel}</span>
				{node.status === "failed" && <span className="text-[10px] text-destructive">✗</span>}
			</div>

			{/* Value + visit count */}
			<div className="flex items-center gap-2 mb-1">
				{avgValue != null && (
					<span className={`text-xs font-mono tabular-nums ${valueColor}`}>{avgValue.toFixed(2)}</span>
				)}
				{visitCount != null && <span className="text-[10px] text-muted-foreground">{visitCount}×</span>}
			</div>

			{/* Content preview */}
			<p className="text-[10px] leading-snug line-clamp-2 text-muted-foreground">{node.label}</p>
		</div>
	);
}

// ---------------------------------------------------------------------------
// LATSAgentView
// ---------------------------------------------------------------------------

export function LATSAgentView({ events }: AgentViewProps) {
	const data = useLATSData(events);
	const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
	const [selectedIteration, setSelectedIteration] = useState<number | null>(null);
	const [mode, setMode] = useState<IterationMode>("highlight");

	// Replay mode: reconstruct tree at selected iteration
	const replayState = useMemo(() => {
		if (mode !== "replay" || selectedIteration == null) return null;
		return buildTreeAtIteration(events, selectedIteration);
	}, [mode, selectedIteration, events]);

	// The root to display: replay tree or full tree
	const displayRoot = replayState?.root ?? data.root;

	// Build allNodes map for detail panel navigation
	const allNodes = useMemo(() => {
		const map = new Map<string, VisualTreeNode>();
		if (!displayRoot) return map;
		const queue = [displayRoot];
		while (queue.length > 0) {
			const n = queue.shift()!;
			map.set(n.id, n);
			queue.push(...n.children);
		}
		return map;
	}, [displayRoot]);

	const selectedNode = selectedNodeId ? (allNodes.get(selectedNodeId) ?? null) : null;

	// Compute iteration-specific highlight/outline sets
	const { highlightedNodeIds } = useMemo(() => {
		if (mode === "replay" && replayState) {
			// In replay mode: selection path highlighted, new nodes also highlighted
			const highlighted = new Set<string>(replayState.selectionPath);
			for (const id of replayState.backpropNodeIds) {
				highlighted.add(id);
			}
			return {
				highlightedNodeIds: highlighted.size > 0 ? highlighted : undefined,
			};
		}

		if (selectedIteration == null) {
			return {
				highlightedNodeIds: data.solutionPath.size > 0 ? data.solutionPath : undefined,
			};
		}

		const iterData = data.iterations.find((it) => it.iterationNumber === selectedIteration);
		if (!iterData) {
			return { highlightedNodeIds: undefined };
		}

		// Selection path is highlighted
		const highlighted = new Set<string>(iterData.selectionPath);

		// Backpropagation path for this iteration
		const iterBackprops = data.backpropagations.filter((bp) => bp.iterationNumber === selectedIteration);
		for (const bp of iterBackprops) {
			for (const nodeId of bp.updatedNodeIds) {
				highlighted.add(nodeId);
			}
		}

		return {
			highlightedNodeIds: highlighted.size > 0 ? highlighted : undefined,
		};
	}, [mode, replayState, selectedIteration, data.solutionPath, data.iterations, data.backpropagations]);

	// Enrich nodes with iteration-specific values when an iteration is selected
	const enrichedRoot = useMemo(() => {
		if (!displayRoot) return null;

		// In replay mode, the tree is already built with correct values
		if (mode === "replay") return displayRoot;

		if (selectedIteration == null) return displayRoot;

		const iterData = data.iterations.find((it) => it.iterationNumber === selectedIteration);
		if (!iterData || Object.keys(iterData.nodeValues).length === 0) return displayRoot;

		return enrichWithIterationValues(displayRoot, iterData.nodeValues);
	}, [displayRoot, mode, selectedIteration, data.iterations]);

	const handleNodeSelect = useCallback((nodeId: string) => {
		setSelectedNodeId((prev) => (prev === nodeId ? null : nodeId));
	}, []);

	const handleNavigate = useCallback((nodeId: string) => {
		setSelectedNodeId(nodeId);
	}, []);

	const handleClosePanel = useCallback(() => {
		setSelectedNodeId(null);
	}, []);

	// Add UCB1 value from selected iteration to the detail panel node
	const detailNode = useMemo(() => {
		if (!selectedNode) return null;
		if (selectedIteration == null) return selectedNode;

		const iterData = data.iterations.find((it) => it.iterationNumber === selectedIteration);
		if (!iterData) return selectedNode;

		const ucb1 = iterData.nodeValues[selectedNode.id];
		if (ucb1 == null) return selectedNode;

		return {
			...selectedNode,
			metadata: {
				...selectedNode.metadata,
				ucb1,
			},
		};
	}, [selectedNode, selectedIteration, data.iterations]);

	// Build a render function that includes backprop delta annotations in replay mode
	const renderNodeContent = useCallback(
		(node: VisualTreeNode, isSelected: boolean) => {
			const delta = replayState?.backpropDeltas.get(node.id);
			const isNew = replayState?.newNodeIds.has(node.id) ?? false;
			return (
				<div className="relative">
					{LATSNodeContent(node, isSelected)}
					{/* Backpropagation delta annotation */}
					{mode === "replay" && delta != null && (
						<div
							className={`absolute -top-2 -right-2 text-[9px] font-mono tabular-nums px-1 py-0.5 rounded-full border ${
								delta > 0
									? "bg-success-muted text-success border-success/30"
									: "bg-destructive-muted text-destructive border-destructive/30"
							}`}
							data-testid="backprop-delta"
						>
							{delta > 0 ? "+" : ""}
							{delta.toFixed(2)}
						</div>
					)}
					{/* New node indicator in replay */}
					{mode === "replay" && isNew && (
						<div
							className="absolute -top-1 -left-1 w-2 h-2 rounded-full bg-info border border-info/50"
							data-testid="new-node-indicator"
							title="New in this iteration"
						/>
					)}
				</div>
			);
		},
		[mode, replayState],
	);

	const config = useMemo(
		() => ({
			renderNodeContent,
			nodeWidth: NODE_WIDTH,
			nodeHeight: NODE_HEIGHT,
			highlightedNodeIds,
			highlightStyle: (selectedIteration != null ? "outline" : "path") as "path" | "outline",
			dimmedNodeIds: mode !== "replay" && data.prunedNodeIds.size > 0 ? data.prunedNodeIds : undefined,
			selectedNodeId: selectedNodeId ?? undefined,
			onNodeSelect: handleNodeSelect,
		}),
		[
			renderNodeContent,
			highlightedNodeIds,
			selectedIteration,
			mode,
			data.prunedNodeIds,
			selectedNodeId,
			handleNodeSelect,
		],
	);

	if (!data.root || !enrichedRoot) {
		return <div className="p-4 text-sm text-muted-foreground">No LATS data available.</div>;
	}

	return (
		<div className="flex flex-col h-full" data-testid="lats-agent-view">
			{/* Header */}
			<div className="flex items-center gap-3 px-4 py-3 border-b border-border shrink-0">
				<h2 className="text-sm font-semibold">LATS</h2>
				<span className="text-xs px-1.5 py-0.5 rounded bg-info-muted text-info">MCTS</span>
				{data.terminationReason && (
					<span
						className={`text-xs px-1.5 py-0.5 rounded ${
							data.terminationReason === "solution_found"
								? "bg-success-muted text-success"
								: "bg-muted text-muted-foreground"
						}`}
					>
						{data.terminationReason}
					</span>
				)}
				<div className="flex items-center gap-3 ml-auto text-xs text-muted-foreground">
					<span>
						<span className="font-mono tabular-nums">{data.maxIterations}</span>{" "}
						{data.maxIterations === 1 ? "iteration" : "iterations"}
					</span>
					<span>
						<span className="font-mono tabular-nums">{data.totalNodes}</span> nodes
					</span>
					{data.explorationConstant != null && (
						<span>
							C=<span className="font-mono tabular-nums">{data.explorationConstant.toFixed(1)}</span>
						</span>
					)}
				</div>
			</div>

			{/* Iteration bar */}
			<IterationBar
				iterations={data.iterations}
				selectedIteration={selectedIteration}
				onSelectIteration={setSelectedIteration}
				mode={mode}
				onModeChange={setMode}
			/>

			{/* Body: tree + optional detail panel */}
			<div className="flex flex-1 min-h-0 overflow-hidden">
				{/* Tree visualization */}
				<div className="flex-1 min-w-0">
					<TreeVisualization root={enrichedRoot} config={config} />
				</div>

				{/* Detail panel */}
				{detailNode && (
					<div className="w-80 shrink-0 border-l border-border overflow-y-auto">
						<TreeNodeDetailPanel
							node={detailNode}
							onNavigate={handleNavigate}
							onClose={handleClosePanel}
							allNodes={allNodes}
						/>
					</div>
				)}
			</div>

			{/* Episodic memory section */}
			{data.episodicRecalls.length > 0 && <EpisodicMemorySection recalls={data.episodicRecalls} />}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Episodic Memory Section
// ---------------------------------------------------------------------------

function EpisodicMemorySection({ recalls }: { recalls: import("../../types").TraceEvent[] }) {
	const [isExpanded, setIsExpanded] = useState(false);

	return (
		<div className="border-t border-border shrink-0" data-testid="episodic-memory-section">
			<button
				type="button"
				onClick={() => setIsExpanded((v) => !v)}
				className="w-full flex items-center gap-2 px-4 py-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
			>
				<span>{isExpanded ? "▾" : "▸"}</span>
				<span>
					Episodic Memory ({recalls.length} recall{recalls.length !== 1 ? "s" : ""})
				</span>
			</button>
			{isExpanded && (
				<div className="px-4 pb-3 space-y-2">
					{recalls.map((recall, i) => {
						const query = recall.payload.query as string | undefined;
						const count = recall.payload.results_count as number | undefined;
						const topScore = recall.payload.top_score as number | undefined;
						return (
							// biome-ignore lint/suspicious/noArrayIndexKey: recalls have no unique ID
							<div key={i} className="text-xs bg-muted/50 rounded-md p-2 space-y-1">
								{query && <p className="text-muted-foreground">Query: {query}</p>}
								<div className="flex gap-3">
									{count != null && <span>{count} results</span>}
									{topScore != null && <span className="font-mono tabular-nums">top: {topScore.toFixed(3)}</span>}
								</div>
							</div>
						);
					})}
				</div>
			)}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function enrichWithIterationValues(node: VisualTreeNode, nodeValues: Record<string, number>): VisualTreeNode {
	const value = nodeValues[node.id];
	const enrichedChildren = node.children.map((child) => enrichWithIterationValues(child, nodeValues));

	if (value == null && enrichedChildren === node.children) {
		return node;
	}

	return {
		...node,
		metadata: {
			...node.metadata,
			...(value != null ? { average_value: value } : {}),
		},
		children: enrichedChildren,
	};
}
