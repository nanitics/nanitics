import type { VisualTreeNode } from "../../types/tree-types";

// ---------------------------------------------------------------------------
// Score bar
// ---------------------------------------------------------------------------

function ScoreBar({ score }: { score: number }) {
	const pct = Math.round(score * 100);
	const color = score >= 0.7 ? "bg-success" : score >= 0.4 ? "bg-warning" : "bg-destructive";

	return (
		<div className="flex items-center gap-2">
			<div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
				<div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
			</div>
			<span className="text-xs font-mono tabular-nums w-10 text-right">{score.toFixed(2)}</span>
		</div>
	);
}

// ---------------------------------------------------------------------------
// Status badge
// ---------------------------------------------------------------------------

const statusStyles: Record<string, { label: string; className: string }> = {
	active: { label: "Active", className: "bg-info-muted text-info-muted-foreground" },
	terminal: { label: "Terminal", className: "bg-success-muted text-success-muted-foreground" },
	pruned: { label: "Pruned", className: "bg-muted text-muted-foreground" },
	failed: { label: "Failed", className: "bg-destructive-muted text-destructive-muted-foreground" },
	expanding: { label: "Expanding", className: "bg-warning-muted text-warning-muted-foreground" },
};

function NodeStatusBadge({ status }: { status: VisualTreeNode["status"] }) {
	const style = statusStyles[status] ?? statusStyles.active;
	return (
		<span className={`inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium ${style.className}`}>
			{style.label}
		</span>
	);
}

// ---------------------------------------------------------------------------
// TreeNodeDetailPanel
// ---------------------------------------------------------------------------

interface TreeNodeDetailPanelProps {
	node: VisualTreeNode;
	/** Navigate to a different node. */
	onNavigate?: (nodeId: string) => void;
	/** Close the panel. */
	onClose?: () => void;
	/** All nodes in the tree, for parent lookup. */
	allNodes?: Map<string, VisualTreeNode>;
}

export function TreeNodeDetailPanel({ node, onNavigate, onClose, allNodes }: TreeNodeDetailPanelProps) {
	const parentNode = node.parentId && allNodes ? allNodes.get(node.parentId) : null;

	// Pull known metadata fields
	const content = (node.metadata.content as string) ?? null;
	const action = (node.metadata.action as string) ?? null;
	const observation = (node.metadata.observation as string) ?? null;
	const visitCount = (node.metadata.visit_count as number) ?? null;
	const cumulativeValue = (node.metadata.cumulative_value as number) ?? null;
	const ucb1 = (node.metadata.ucb1 as number) ?? null;

	// Filter out displayed metadata keys so we don't double-show them
	const displayedKeys = new Set(["content", "action", "observation", "visit_count", "cumulative_value", "ucb1"]);
	const extraMetadata = Object.entries(node.metadata).filter(([k]) => !displayedKeys.has(k));

	return (
		<div
			className="flex flex-col gap-4 p-4 border-l border-border bg-background overflow-y-auto max-h-full"
			data-testid="tree-node-detail-panel"
		>
			{/* Header */}
			<div className="flex items-start justify-between gap-2">
				<div className="flex flex-col gap-1 min-w-0">
					<h3 className="text-sm font-semibold truncate">{node.label}</h3>
					<div className="flex items-center gap-2">
						<NodeStatusBadge status={node.status} />
						<span className="text-xs text-muted-foreground">Depth {node.depth}</span>
					</div>
				</div>
				{onClose && (
					<button
						type="button"
						onClick={onClose}
						className="text-muted-foreground hover:text-foreground text-xs p-1"
						aria-label="Close detail panel"
					>
						✕
					</button>
				)}
			</div>

			{/* Score */}
			{node.score != null && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Score</span>
					<ScoreBar score={node.score} />
				</div>
			)}

			{/* Content */}
			{content && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Content</span>
					<div className="text-sm bg-muted/50 rounded-md p-2 whitespace-pre-wrap max-h-60 overflow-auto">{content}</div>
				</div>
			)}

			{/* Action + Observation (LATS) */}
			{action && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Action</span>
					<span className="text-sm font-mono bg-muted/50 rounded-md px-2 py-1 block">{action}</span>
				</div>
			)}
			{observation && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Observation</span>
					<div className="text-sm bg-muted/50 rounded-md p-2 whitespace-pre-wrap max-h-40 overflow-auto">
						{observation}
					</div>
				</div>
			)}

			{/* LATS-specific stats */}
			{visitCount != null && (
				<div className="flex items-center gap-4 text-xs">
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Visits</span>
						<span className="font-mono tabular-nums">{visitCount}</span>
					</div>
					{cumulativeValue != null && (
						<div className="flex items-center gap-1.5">
							<span className="text-muted-foreground">Cumulative value</span>
							<span className="font-mono tabular-nums">{cumulativeValue.toFixed(2)}</span>
						</div>
					)}
					{ucb1 != null && (
						<div className="flex items-center gap-1.5">
							<span className="text-muted-foreground">UCB1</span>
							<span className="font-mono tabular-nums">{ucb1.toFixed(3)}</span>
						</div>
					)}
				</div>
			)}

			{/* Parent navigation */}
			{parentNode && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Parent</span>
					<button
						type="button"
						onClick={() => onNavigate?.(parentNode.id)}
						className="text-xs text-primary hover:text-primary/80 transition-colors block truncate"
					>
						↑ {parentNode.label}
					</button>
				</div>
			)}

			{/* Children navigation */}
			{node.children.length > 0 && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Children ({node.children.length})</span>
					<ul className="space-y-1">
						{node.children.map((child) => (
							<li key={child.id} className="flex items-center gap-2">
								<button
									type="button"
									onClick={() => onNavigate?.(child.id)}
									className="text-xs text-primary hover:text-primary/80 transition-colors truncate"
								>
									↓ {child.label}
								</button>
								{child.score != null && (
									<span className="text-[10px] font-mono tabular-nums text-muted-foreground">
										{child.score.toFixed(2)}
									</span>
								)}
							</li>
						))}
					</ul>
				</div>
			)}

			{/* Extra metadata */}
			{extraMetadata.length > 0 && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Metadata</span>
					<dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
						{extraMetadata.map(([key, value]) => (
							<div key={key} className="contents">
								<dt className="text-muted-foreground font-mono">{key}</dt>
								<dd className="font-mono truncate">{typeof value === "string" ? value : JSON.stringify(value)}</dd>
							</div>
						))}
					</dl>
				</div>
			)}
		</div>
	);
}
