import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { EventDetailPanel } from "../components/event-detail/event-detail-panel";
import { ErrorState } from "../components/feedback/error-state";
import { RunDetailSkeleton, TreeSkeleton } from "../components/feedback/loading-skeleton";
import { EventTypeFilter } from "../components/filters/event-type-filter";
import { LevelSelector } from "../components/filters/level-selector";
import { PatternSummary } from "../components/patterns/pattern-summary";
import { StatusBadge } from "../components/primitives/status-badge";
import { TraceTree } from "../components/trace-tree/trace-tree";
import { useAgents } from "../hooks/use-agents";
import { matchesEventTypeFilter, useFilters } from "../hooks/use-filters";
import { useRunDetail } from "../hooks/use-run-detail";
import { useSpanTree } from "../hooks/use-span-tree";
import { useStreaming } from "../hooks/use-streaming";
import type { RunStatus, SpanTreeNode, TraceEvent } from "../types";
import { detectPatterns } from "../utils/pattern-detector";

/** Recursively check if any event in the span tree has the given type. */
function hasEventType(node: SpanTreeNode, eventType: string): boolean {
	if (node.events.some((e) => e.event_type === eventType)) return true;
	return node.children.some((child) => hasEventType(child, eventType));
}

interface RunDetailPageProps {
	runId: string;
	onBack: () => void;
	onNavigateToAgent?: (spanId: string) => void;
	onNavigateToWorkflow?: () => void;
}

export function RunDetailPage({ runId, onBack, onNavigateToAgent, onNavigateToWorkflow }: RunDetailPageProps) {
	const { level, setLevel, eventTypes, toggleEventType, clearEventTypes } = useFilters();
	const { data, isLoading: runLoading, error: runError, refetch } = useRunDetail(runId);
	const {
		tree,
		expandedNodes,
		toggleNode,
		expandAll,
		collapseAll,
		selectedEvent,
		selectEvent,
		isLoading: treeLoading,
		error: treeError,
		addStreamedEvent,
	} = useSpanTree(runId, { minLevel: level });

	const { agents } = useAgents(runId);
	const patterns = useMemo(() => (tree ? detectPatterns(tree.root) : []), [tree]);

	// Track live run status — starts from API data, updated by SSE
	const [liveStatus, setLiveStatus] = useState<RunStatus | null>(null);
	const runStatus = liveStatus ?? data?.run.status ?? null;
	const isRunning = runStatus === "running";

	// Update live status when API data loads
	useEffect(() => {
		if (data?.run.status) {
			setLiveStatus(data.run.status);
		}
	}, [data?.run.status]);

	// Clear selectedEvent when the level filter changes — the tree is refetched
	// and the prior selection may no longer exist. `level` is listed as a
	// dependency intentionally: it is the trigger, not a value read inside.
	// biome-ignore lint/correctness/useExhaustiveDependencies: `level` is the trigger.
	useEffect(() => {
		selectEvent(null);
	}, [level, selectEvent]);

	// Clear selectedEvent when the event-type filter excludes it.
	useEffect(() => {
		if (selectedEvent && eventTypes.size > 0 && !matchesEventTypeFilter(selectedEvent.event_type, eventTypes)) {
			selectEvent(null);
		}
	}, [selectedEvent, eventTypes, selectEvent]);

	// Auto-scroll: track whether user has scrolled up
	const treeContainerRef = useRef<HTMLDivElement>(null);
	const [autoScroll, setAutoScroll] = useState(true);
	const handleTreeScroll = useCallback(() => {
		const el = treeContainerRef.current;
		if (!el) return;
		// If user is near the bottom (within 50px), enable auto-scroll
		const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 50;
		setAutoScroll(atBottom);
	}, []);

	// Streaming
	const { connectionState, isComplete } = useStreaming(runId, {
		enabled: isRunning,
		minLevel: level,
		onEvent: useCallback(
			(event: TraceEvent) => {
				addStreamedEvent(event);
				// Auto-expand new span's parent so it's visible
				// (the tree node is created inside addStreamedEvent)
			},
			[addStreamedEvent],
		),
		onRunComplete: useCallback(
			(status: string) => {
				setLiveStatus(status as RunStatus);
				refetch();
			},
			[refetch],
		),
	});

	// Auto-scroll to bottom when new events arrive during streaming
	useEffect(() => {
		if (!autoScroll || !isRunning) return;
		const el = treeContainerRef.current;
		if (el) {
			el.scrollTop = el.scrollHeight;
		}
	});

	const eventFilter = useCallback(
		(event: TraceEvent) => matchesEventTypeFilter(event.event_type, eventTypes),
		[eventTypes],
	);

	// Only pass filter when categories are selected
	const activeEventFilter = useMemo(
		() => (eventTypes.size > 0 ? eventFilter : undefined),
		[eventTypes.size, eventFilter],
	);

	// Run header hasn't loaded yet — show the full page skeleton.
	if (runLoading) {
		return <RunDetailSkeleton />;
	}

	// Precedence: `runError` before `treeError`. Retry only wires to
	// `runError` because that is the hook with a `refetch` seam;
	// tree-error retry is not yet supported.
	if (runError) {
		return <ErrorState error={runError} onRetry={refetch} />;
	}
	if (treeError) {
		return <ErrorState error={treeError} />;
	}

	return (
		<main className="flex flex-col h-full">
			{/* Header */}
			{data && (
				<div className="border-b px-4 py-3 flex items-center gap-4">
					<button
						type="button"
						onClick={onBack}
						className="text-sm text-muted-foreground hover:text-foreground transition-colors"
					>
						← Runs
					</button>
					{onNavigateToWorkflow && tree && hasEventType(tree.root, "workflow.structure") && (
						<button
							type="button"
							onClick={onNavigateToWorkflow}
							className="text-xs px-2 py-1 rounded border border-border text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
						>
							View Workflow
						</button>
					)}
					<div className="flex-1 min-w-0">
						<div className="flex items-center gap-2">
							<span className="font-medium text-sm truncate">
								{(data.run.metadata?.description as string) || data.run.id}
							</span>
							<StatusBadge status={liveStatus ?? data.run.status} />
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
							{isRunning && connectionState === "reconnecting" && (
								<span className="text-xs text-warning">Reconnecting…</span>
							)}
							{isComplete && <span className="text-xs text-muted-foreground">Stream ended</span>}
						</div>
						<div className="flex items-center gap-4 text-xs text-muted-foreground mt-0.5">
							<span className="font-mono">{data.run.id}</span>
							<span>
								{data.summary.llm_calls} LLM · {data.summary.tool_calls} tool ·{" "}
								{(data.summary.total_input_tokens + data.summary.total_output_tokens).toLocaleString()} tokens
								{data.summary.errors > 0 && (
									<span className="text-destructive ml-1">· {data.summary.errors} errors</span>
								)}
							</span>
						</div>
					</div>
				</div>
			)}

			{/* Filter toolbar */}
			<div className="border-b px-4 py-2 flex items-center gap-4">
				<LevelSelector value={level} onChange={setLevel} />
				<div className="h-4 w-px bg-border" />
				<EventTypeFilter enabled={eventTypes} onToggle={toggleEventType} onClear={clearEventTypes} />
			</div>

			{/* Pattern summary */}
			{patterns.length > 0 && onNavigateToAgent && (
				<PatternSummary patterns={patterns} agents={agents} onNavigateToAgent={onNavigateToAgent} />
			)}

			{/* Tree + Detail split */}
			<div className="flex-1 flex min-h-0">
				{/* Tree panel. Header-shell-first: once the run header has loaded
				    but the tree is still fetching, we render the tree-row skeleton
				    here so the user sees the run title/status immediately while the
				    tree fills in underneath. */}
				<div className="flex-1 min-w-0 border-r overflow-hidden">
					{treeLoading && !tree && <TreeSkeleton />}
					{tree && (
						<TraceTree
							tree={tree}
							expandedNodes={expandedNodes}
							selectedEvent={selectedEvent}
							onToggleNode={toggleNode}
							onExpandAll={expandAll}
							onCollapseAll={collapseAll}
							onSelectEvent={selectEvent}
							eventFilter={activeEventFilter}
							treeContainerRef={treeContainerRef}
							onTreeScroll={handleTreeScroll}
							onNavigateToAgent={onNavigateToAgent}
						/>
					)}
				</div>

				{/* Detail panel — implicit role="complementary" via <aside>. */}
				<aside aria-label="Selected event details" className="w-[400px] flex-shrink-0 overflow-y-auto">
					{selectedEvent ? (
						<EventDetailPanel event={selectedEvent} onNavigateToAgent={onNavigateToAgent} />
					) : (
						<div className="flex items-center justify-center h-full text-sm text-muted-foreground">
							Select an event to view details
						</div>
					)}
				</aside>
			</div>
		</main>
	);
}
