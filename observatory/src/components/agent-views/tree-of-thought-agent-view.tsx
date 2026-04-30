import { useCallback, useMemo, useState } from "react";
import { useTreeOfThoughtData } from "../../hooks/use-tree-of-thought-data";
import type { AgentViewProps } from "../../registry/agent-view-registry";
import type { VisualTreeNode } from "../../types/tree-types";
import { TreeNodeDetailPanel } from "../tree/tree-node-detail-panel";
import { TreeVisualization } from "../tree/tree-visualization";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const NODE_WIDTH = 180;
const NODE_HEIGHT = 110;

// ---------------------------------------------------------------------------
// Node content render prop
// ---------------------------------------------------------------------------

function ToTNodeContent(node: VisualTreeNode, isSelected: boolean) {
	const truncatedLabel = node.label.length > 80 ? `${node.label.slice(0, 80)}…` : node.label;

	const scoreColor =
		node.score != null
			? node.score >= 0.7
				? "bg-success-muted text-success"
				: node.score >= 0.4
					? "bg-warning-muted text-warning"
					: "bg-destructive-muted text-destructive"
			: "";

	const statusIcon =
		node.status === "terminal" ? "✓" : node.status === "pruned" ? "✕" : node.status === "failed" ? "✗" : "●";

	const statusColor =
		node.status === "terminal"
			? "text-success"
			: node.status === "pruned"
				? "text-muted-foreground"
				: node.status === "failed"
					? "text-destructive"
					: "text-info";

	return (
		<div
			className={`rounded-lg border p-2 text-left transition-colors ${
				isSelected ? "border-primary bg-primary/5" : "border-border bg-background hover:border-primary/50"
			}`}
			style={{ width: NODE_WIDTH, minHeight: NODE_HEIGHT }}
		>
			{/* Status + depth row */}
			<div className="flex items-center justify-between mb-1">
				<span className={`text-xs ${statusColor}`}>{statusIcon}</span>
				<span className="text-[10px] text-muted-foreground">d{node.depth}</span>
			</div>

			{/* Content preview */}
			<p className="text-xs leading-snug line-clamp-2 mb-1.5">{truncatedLabel}</p>

			{/* Score badge */}
			{node.score != null && (
				<span className={`inline-block text-[10px] px-1.5 py-0.5 rounded font-mono tabular-nums ${scoreColor}`}>
					{node.score.toFixed(2)}
				</span>
			)}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Strategy badge
// ---------------------------------------------------------------------------

const strategyLabels: Record<string, string> = {
	bfs: "BFS",
	dfs: "DFS",
	best_first: "Best-first",
};

// ---------------------------------------------------------------------------
// TreeOfThoughtAgentView
// ---------------------------------------------------------------------------

export function TreeOfThoughtAgentView({ events }: AgentViewProps) {
	const data = useTreeOfThoughtData(events);
	const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

	// Build allNodes map for detail panel navigation
	const allNodes = useMemo(() => {
		const map = new Map<string, VisualTreeNode>();
		if (!data.root) return map;
		const queue = [data.root];
		while (queue.length > 0) {
			const n = queue.shift()!;
			map.set(n.id, n);
			queue.push(...n.children);
		}
		return map;
	}, [data.root]);

	const selectedNode = selectedNodeId ? (allNodes.get(selectedNodeId) ?? null) : null;

	const handleNodeSelect = useCallback((nodeId: string) => {
		setSelectedNodeId((prev) => (prev === nodeId ? null : nodeId));
	}, []);

	const handleNavigate = useCallback((nodeId: string) => {
		setSelectedNodeId(nodeId);
	}, []);

	const handleClosePanel = useCallback(() => {
		setSelectedNodeId(null);
	}, []);

	const config = useMemo(
		() => ({
			renderNodeContent: ToTNodeContent,
			nodeWidth: NODE_WIDTH,
			nodeHeight: NODE_HEIGHT,
			highlightedNodeIds: data.solutionPath.size > 0 ? data.solutionPath : undefined,
			highlightStyle: "path" as const,
			dimmedNodeIds: data.prunedNodeIds.size > 0 ? data.prunedNodeIds : undefined,
			selectedNodeId: selectedNodeId ?? undefined,
			onNodeSelect: handleNodeSelect,
		}),
		[data.solutionPath, data.prunedNodeIds, selectedNodeId, handleNodeSelect],
	);

	if (!data.root) {
		return <div className="p-4 text-sm text-muted-foreground">No tree search data available.</div>;
	}

	const strategyLabel = data.searchStrategy ? (strategyLabels[data.searchStrategy] ?? data.searchStrategy) : null;

	return (
		<div className="flex flex-col h-full" data-testid="tot-agent-view">
			{/* Header */}
			<div className="flex items-center gap-3 px-4 py-3 border-b border-border shrink-0">
				<h2 className="text-sm font-semibold">Tree of Thought</h2>
				{strategyLabel && (
					<span className="text-xs px-1.5 py-0.5 rounded bg-info-muted text-info">{strategyLabel}</span>
				)}
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
						<span className="font-mono tabular-nums">{data.totalNodes}</span> nodes
					</span>
					<span>
						max depth <span className="font-mono tabular-nums">{data.maxDepth}</span>
					</span>
				</div>
			</div>

			{/* Body: tree + optional detail panel */}
			<div className="flex flex-1 min-h-0 overflow-hidden">
				{/* Tree visualization */}
				<div className="flex-1 min-w-0">
					<TreeVisualization root={data.root} config={config} />
				</div>

				{/* Detail panel */}
				{selectedNode && (
					<div className="w-80 shrink-0 border-l border-border overflow-y-auto">
						<TreeNodeDetailPanel
							node={selectedNode}
							onNavigate={handleNavigate}
							onClose={handleClosePanel}
							allNodes={allNodes}
						/>
					</div>
				)}
			</div>
		</div>
	);
}
