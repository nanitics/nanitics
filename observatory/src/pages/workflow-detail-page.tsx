import { useCallback, useMemo, useState } from "react";
import { DAGNodeContent } from "../components/dag/dag-node";
import { DAGVisualization } from "../components/dag/dag-visualization";
import { ErrorState } from "../components/feedback/error-state";
import { RunDetailSkeleton } from "../components/feedback/loading-skeleton";
import { StatusBadge } from "../components/primitives/status-badge";
import { useStreaming } from "../hooks/use-streaming";
import { useWorkflowDAG } from "../hooks/use-workflow-dag";
import type { TraceEvent } from "../types";
import type { DAGLayout, DAGNode } from "../types/dag-types";

interface WorkflowDetailPageProps {
	runId: string;
	onBack: () => void;
	onBackToRuns: () => void;
	onNavigateToAgent: (spanId: string) => void;
	runLabel?: string;
}

const NODE_WIDTH = 220;
const NODE_HEIGHT = 80;

function formatDuration(ms: number): string {
	return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`;
}

/** Derive an overall workflow status from the node statuses. */
function deriveWorkflowStatus(layout: DAGLayout | null): "pending" | "running" | "completed" | "error" {
	if (!layout || layout.nodes.length === 0) return "pending";
	if (layout.nodes.some((n) => n.status === "error")) return "error";
	if (layout.nodes.some((n) => n.status === "running")) return "running";
	if (layout.nodes.every((n) => n.status === "completed")) return "completed";
	return "pending";
}

export function WorkflowDetailPage({
	runId,
	onBack,
	onBackToRuns,
	onNavigateToAgent,
	runLabel,
}: WorkflowDetailPageProps) {
	const { workflow, layout, isLoading, error, refetch } = useWorkflowDAG(runId);

	const [selectedNodeId, setSelectedNodeId] = useState<string | undefined>();
	const [showCriticalPath, setShowCriticalPath] = useState(false);

	// Real-time updates for running workflows
	const workflowStatus = deriveWorkflowStatus(layout);
	const isRunning = workflowStatus === "running" || workflowStatus === "pending";

	const { connectionState } = useStreaming(runId, {
		enabled: isRunning,
		onEvent: useCallback(
			(event: TraceEvent) => {
				const eventType = event.event_type;
				if (
					eventType === "workflow.step.complete" ||
					eventType === "agent.start" ||
					eventType === "workflow.error" ||
					eventType === "workflow.complete"
				) {
					refetch();
				}
			},
			[refetch],
		),
	});

	// Max duration for DurationBar normalization
	const maxDurationMs = useMemo(() => {
		if (!layout) return 0;
		return Math.max(...layout.nodes.map((n) => n.durationMs ?? 0), 1);
	}, [layout]);

	// Selected node details
	const selectedNode = useMemo(
		() => (selectedNodeId ? (layout?.nodes.find((n) => n.id === selectedNodeId) ?? null) : null),
		[selectedNodeId, layout],
	);

	const renderNodeContent = useCallback(
		(node: DAGNode, isSelected: boolean) => (
			<DAGNodeContent
				node={node}
				isSelected={isSelected}
				maxDurationMs={maxDurationMs}
				onNavigateToAgent={onNavigateToAgent}
			/>
		),
		[maxDurationMs, onNavigateToAgent],
	);

	// Reuse `<RunDetailSkeleton>` — the workflow page's outer shell is similar
	// enough that a bespoke skeleton is not warranted at v0.1.0.
	if (isLoading) {
		return <RunDetailSkeleton />;
	}

	if (error) {
		return <ErrorState error={error} onRetry={refetch} />;
	}

	if (!workflow || !layout) {
		return <div className="text-muted-foreground py-8 text-center">No workflow data found</div>;
	}

	return (
		<main className="flex flex-col h-full" data-testid="workflow-detail-page">
			{/* Breadcrumb */}
			<div className="border-b px-4 py-2 flex items-center gap-1.5 text-sm text-muted-foreground">
				<button type="button" onClick={onBackToRuns} className="hover:text-foreground transition-colors">
					Runs
				</button>
				<span>›</span>
				<button
					type="button"
					onClick={onBack}
					className="hover:text-foreground transition-colors truncate max-w-[200px]"
				>
					{runLabel || runId}
				</button>
				<span>›</span>
				<span className="text-foreground truncate">{workflow.workflow_name}</span>
			</div>

			{/* Header */}
			<div className="border-b px-4 py-3">
				<div className="flex items-center gap-2 flex-wrap">
					<h2 className="text-base font-semibold">{workflow.workflow_name}</h2>
					<span className="inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium bg-accent-status-muted text-accent-status-muted-foreground capitalize">
						{workflow.workflow_type}
					</span>
					<StatusBadge status={workflowStatus} />
					{/* Live indicator */}
					{isRunning && connectionState === "connected" && (
						<span className="flex items-center gap-1.5 text-xs text-info">
							<span className="relative flex h-2 w-2">
								<span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-info opacity-75" />
								<span className="relative inline-flex rounded-full h-2 w-2 bg-info" />
							</span>
							Live
						</span>
					)}
				</div>

				{/* Controls row */}
				<div className="flex items-center gap-4 mt-2 text-xs">
					<span className="text-muted-foreground">{layout.nodes.length} steps</span>
					<label className="flex items-center gap-1.5 text-muted-foreground cursor-pointer">
						<input
							type="checkbox"
							checked={showCriticalPath}
							onChange={(e) => setShowCriticalPath(e.target.checked)}
							className="rounded border-border"
						/>
						Critical path
					</label>
				</div>
			</div>

			{/* DAG visualization */}
			<div className="flex-1 min-h-0">
				<DAGVisualization
					layout={layout}
					renderNodeContent={renderNodeContent}
					nodeWidth={NODE_WIDTH}
					nodeHeight={NODE_HEIGHT}
					selectedNodeId={selectedNodeId}
					onNodeSelect={setSelectedNodeId}
					showCriticalPath={showCriticalPath}
				/>
			</div>

			{/* Bottom detail panel for selected node — implicit role="complementary" via <aside>. */}
			{selectedNode && (
				<aside
					aria-label="Selected workflow step"
					className="border-t px-4 py-3 bg-muted/30"
					data-testid="selected-node-detail"
				>
					<div className="flex items-center gap-3 flex-wrap">
						<span className="font-medium text-sm">{selectedNode.label}</span>
						<StatusBadge status={selectedNode.status} />
						{selectedNode.agentType && (
							<span className="inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium bg-accent-status-muted text-accent-status-muted-foreground capitalize">
								{selectedNode.agentType}
							</span>
						)}
						<span className="text-xs text-muted-foreground capitalize">{selectedNode.stepType}</span>
						{selectedNode.durationMs != null && (
							<span className="text-xs text-muted-foreground">{formatDuration(selectedNode.durationMs)}</span>
						)}
						{selectedNode.agentSpanId && (
							<button
								type="button"
								onClick={() => onNavigateToAgent(selectedNode.agentSpanId!)}
								className="text-xs text-primary hover:underline"
							>
								View agent →
							</button>
						)}
					</div>
					{selectedNode.outputPreview && (
						<p className="text-xs text-muted-foreground mt-1.5 line-clamp-2">{selectedNode.outputPreview}</p>
					)}
					{selectedNode.parallelGroup && (
						<span className="text-[10px] text-muted-foreground mt-1 block">
							Parallel group: {selectedNode.parallelGroup}
						</span>
					)}
				</aside>
			)}
		</main>
	);
}
