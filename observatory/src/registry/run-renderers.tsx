import { PayloadViewer } from "../components/event-detail/payload-viewer";
import { StatusBadge } from "../components/primitives/status-badge";
import type { TraceEvent } from "../types";
import type { EventDetailProps, EventRendererRegistration } from "./renderer-registry";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatDuration(ms: number | null | undefined): string {
	if (ms == null) return "—";
	if (ms < 1000) return `${ms}ms`;
	return `${(ms / 1000).toFixed(1)}s`;
}

// ---------------------------------------------------------------------------
// Run Lifecycle Renderers
// ---------------------------------------------------------------------------

function RunStartRenderer({ event }: EventDetailProps) {
	const { run_id, workflow_name, metadata } = event.payload as {
		run_id?: string;
		workflow_name?: string;
		metadata?: Record<string, unknown>;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{run_id && <span className="text-sm font-mono">{run_id}</span>}
				<StatusBadge status="running" />
			</div>
			{workflow_name && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Workflow</span>
					<span className="font-medium">{workflow_name}</span>
				</div>
			)}
			{metadata && Object.keys(metadata).length > 0 && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Metadata</span>
					<PayloadViewer payload={metadata} />
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function runStartSummary(event: TraceEvent): string {
	const { workflow_name, run_id } = event.payload as {
		workflow_name?: string;
		run_id?: string;
	};
	if (workflow_name) return `Run started: '${workflow_name}'`;
	if (run_id) return `Run started: ${run_id}`;
	return "Run started";
}

function RunCompleteRenderer({ event }: EventDetailProps) {
	const { run_id, duration_ms } = event.payload as {
		run_id?: string;
		duration_ms?: number;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{run_id && <span className="text-sm font-mono">{run_id}</span>}
				<StatusBadge status="completed" />
			</div>
			{duration_ms != null && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Duration</span>
					<span className="font-mono tabular-nums">{formatDuration(duration_ms)}</span>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function runCompleteSummary(event: TraceEvent): string {
	const { duration_ms } = event.payload as { duration_ms?: number };
	return `Run completed (${formatDuration(duration_ms)})`;
}

function RunFailedRenderer({ event }: EventDetailProps) {
	const { run_id, error_type, error_message } = event.payload as {
		run_id?: string;
		error_type?: string;
		error_message?: string;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{run_id && <span className="text-sm font-mono">{run_id}</span>}
				<StatusBadge status="failed" />
			</div>
			{error_type && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Error type</span>
					<span className="font-mono">{error_type}</span>
				</div>
			)}
			{error_message && (
				<div className="text-sm text-destructive-muted-foreground bg-destructive-muted rounded-md p-2">
					{error_message}
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function runFailedSummary(event: TraceEvent): string {
	const { error_type } = event.payload as { error_type?: string };
	return `Run failed: ${error_type ?? "unknown error"}`;
}

function RunSuspendedRenderer({ event }: EventDetailProps) {
	const { run_id, suspension_id } = event.payload as {
		run_id?: string;
		suspension_id?: string;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{run_id && <span className="text-sm font-mono">{run_id}</span>}
				<StatusBadge status="suspended" />
			</div>
			{suspension_id && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Suspension</span>
					<span className="font-mono">{suspension_id}</span>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function runSuspendedSummary(event: TraceEvent): string {
	const { suspension_id } = event.payload as { suspension_id?: string };
	if (suspension_id) return `Run suspended (suspension: ${suspension_id})`;
	return "Run suspended";
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/** Creates all run lifecycle event renderer registrations. */
export function createRunRegistrations(): EventRendererRegistration[] {
	return [
		{ matches: (t) => t === "run.start", priority: 0, component: RunStartRenderer, summary: runStartSummary },
		{ matches: (t) => t === "run.complete", priority: 0, component: RunCompleteRenderer, summary: runCompleteSummary },
		{ matches: (t) => t === "run.failed", priority: 0, component: RunFailedRenderer, summary: runFailedSummary },
		{
			matches: (t) => t === "run.suspended",
			priority: 0,
			component: RunSuspendedRenderer,
			summary: runSuspendedSummary,
		},
	];
}
