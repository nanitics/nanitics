import { useState } from "react";
import type { CapabilityPanelProps } from "../../registry/capability-panel-registry";
import type { TraceEvent } from "../../types";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface CheckpointLifecycle {
	groupKey: string;
	checkpointId: string;
	checkpointType: string | null;
	runId: string | null;
	checkpointEvent: TraceEvent | null;
	suspensions: SuspensionInfo[];
	resumption: TraceEvent | null;
}

interface SuspensionInfo {
	event: TraceEvent;
	suspensionId: string;
	suspensionType: string;
	stepName: string | null;
	agentName: string | null;
}

// ---------------------------------------------------------------------------
// Correlation logic
// ---------------------------------------------------------------------------

function buildLifecycles(events: TraceEvent[]): CheckpointLifecycle[] {
	const checkpoints = events.filter((e) => e.event_type === "checkpoint.saved");
	const suspensions = events.filter((e) => e.event_type === "execution.suspended");
	const resumptions = events.filter((e) => e.event_type === "execution.resumed");

	// Index checkpoint events and resumptions by checkpoint_id
	const checkpointByIdMap = new Map<string, TraceEvent>();
	for (const cp of checkpoints) {
		const p = cp.payload as Record<string, unknown>;
		const id = String(p.checkpoint_id ?? "");
		if (id) checkpointByIdMap.set(id, cp);
	}

	const resumptionByCheckpointId = new Map<string, TraceEvent>();
	const resumptionBySuspensionId = new Map<string, TraceEvent>();
	for (const r of resumptions) {
		const p = r.payload as Record<string, unknown>;
		const cpId = String(p.checkpoint_id ?? "");
		const susId = String(p.suspension_id ?? "");
		if (cpId) resumptionByCheckpointId.set(cpId, r);
		if (susId) resumptionBySuspensionId.set(susId, r);
	}

	// Group suspensions by checkpoint_id
	const suspensionsByCheckpointId = new Map<string, SuspensionInfo[]>();
	const standaloneSuspensions: SuspensionInfo[] = [];

	for (const sus of suspensions) {
		const p = sus.payload as Record<string, unknown>;
		const cpId = String(p.checkpoint_id ?? "");
		const info: SuspensionInfo = {
			event: sus,
			suspensionId: String(p.suspension_id ?? ""),
			suspensionType: String(p.suspension_type ?? "unknown"),
			stepName: p.step_name != null ? String(p.step_name) : null,
			agentName: p.agent_name != null ? String(p.agent_name) : null,
		};

		if (cpId) {
			const list = suspensionsByCheckpointId.get(cpId) ?? [];
			list.push(info);
			suspensionsByCheckpointId.set(cpId, list);
		} else {
			standaloneSuspensions.push(info);
		}
	}

	// Collect all checkpoint IDs from all sources
	const allCheckpointIds = new Set<string>();
	for (const id of checkpointByIdMap.keys()) allCheckpointIds.add(id);
	for (const id of suspensionsByCheckpointId.keys()) allCheckpointIds.add(id);
	for (const id of resumptionByCheckpointId.keys()) allCheckpointIds.add(id);

	const lifecycles: CheckpointLifecycle[] = [];

	// Build lifecycle per checkpoint_id
	for (const cpId of allCheckpointIds) {
		const cpEvent = checkpointByIdMap.get(cpId) ?? null;
		const cpPayload = cpEvent ? (cpEvent.payload as Record<string, unknown>) : null;

		lifecycles.push({
			groupKey: cpId,
			checkpointId: cpId,
			checkpointType: cpPayload?.checkpoint_type ? String(cpPayload.checkpoint_type) : null,
			runId: cpPayload?.run_id ? String(cpPayload.run_id) : null,
			checkpointEvent: cpEvent,
			suspensions: suspensionsByCheckpointId.get(cpId) ?? [],
			resumption: resumptionByCheckpointId.get(cpId) ?? null,
		});
	}

	// Add standalone suspensions (empty checkpoint_id)
	for (const sus of standaloneSuspensions) {
		lifecycles.push({
			groupKey: sus.suspensionId || sus.event.id.toString(),
			checkpointId: "",
			checkpointType: null,
			runId: null,
			checkpointEvent: null,
			suspensions: [sus],
			resumption: resumptionBySuspensionId.get(sus.suspensionId) ?? null,
		});
	}

	// Sort by earliest timestamp
	lifecycles.sort((a, b) => {
		const ta = earliestTimestamp(a);
		const tb = earliestTimestamp(b);
		return ta.localeCompare(tb);
	});

	return lifecycles;
}

function earliestTimestamp(lc: CheckpointLifecycle): string {
	const candidates: string[] = [];
	if (lc.checkpointEvent) candidates.push(lc.checkpointEvent.timestamp);
	for (const s of lc.suspensions) candidates.push(s.event.timestamp);
	if (lc.resumption) candidates.push(lc.resumption.timestamp);
	candidates.sort();
	return candidates[0] ?? "";
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------

/** Durability panel — checkpoint lifecycle tracking and suspension/resumption correlation. */
export function DurabilityPanel({ events }: CapabilityPanelProps) {
	const durabilityEvents = events.filter(
		(e) =>
			e.event_type === "checkpoint.saved" ||
			e.event_type === "execution.suspended" ||
			e.event_type === "execution.resumed",
	);

	if (durabilityEvents.length === 0) {
		return <div className="p-4 text-sm text-muted-foreground">No durability events recorded for this agent.</div>;
	}

	const lifecycles = buildLifecycles(durabilityEvents);

	// Summary statistics
	const totalCheckpoints = lifecycles.filter((lc) => lc.checkpointEvent).length;
	const totalSuspensions = lifecycles.reduce((sum, lc) => sum + lc.suspensions.length, 0);
	const pending = lifecycles.filter((lc) => lc.suspensions.length > 0 && !lc.resumption).length;
	const completed = lifecycles.filter((lc) => lc.suspensions.length > 0 && lc.resumption).length;

	return (
		<div className="p-4 space-y-4">
			{/* Summary */}
			<div className="flex flex-wrap gap-4 text-xs">
				<SummaryPill label="Checkpoints" value={String(totalCheckpoints)} />
				<SummaryPill label="Suspensions" value={String(totalSuspensions)} />
				<SummaryPill label="Pending" value={String(pending)} />
				<SummaryPill label="Completed" value={String(completed)} />
			</div>

			{/* Lifecycle cards */}
			<div className="space-y-2">
				{lifecycles.map((lc) => (
					<LifecycleCard key={lc.groupKey} lifecycle={lc} />
				))}
			</div>
		</div>
	);
}

// ---------------------------------------------------------------------------
// Lifecycle Card
// ---------------------------------------------------------------------------

function LifecycleCard({ lifecycle }: { lifecycle: CheckpointLifecycle }) {
	const [isExpanded, setIsExpanded] = useState(false);
	const isStandalone = !lifecycle.checkpointId;
	const isPending = lifecycle.suspensions.length > 0 && !lifecycle.resumption;

	// Compute duration if both suspension and resumption exist
	let durationMs: number | null = null;
	if (lifecycle.suspensions.length > 0 && lifecycle.resumption) {
		const suspTime = new Date(lifecycle.suspensions[0].event.timestamp).getTime();
		const resTime = new Date(lifecycle.resumption.timestamp).getTime();
		const diff = resTime - suspTime;
		if (diff >= 0) durationMs = diff;
	}

	return (
		<div
			className={`border rounded-lg overflow-hidden ${isPending ? "border-warning-border bg-warning-muted/30" : ""}`}
		>
			<button
				type="button"
				className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-accent/50 transition-colors text-left"
				onClick={() => setIsExpanded(!isExpanded)}
			>
				<span className="text-xs text-muted-foreground w-4">{isExpanded ? "▾" : "▸"}</span>

				{/* Type badge */}
				{isStandalone ? (
					<span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-warning-muted text-warning-muted-foreground">
						unchecked
					</span>
				) : lifecycle.checkpointType ? (
					<span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-info-muted text-info-muted-foreground">
						{lifecycle.checkpointType}
					</span>
				) : null}

				{/* Status */}
				{isPending ? (
					<span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-warning-muted text-warning">Pending</span>
				) : lifecycle.resumption ? (
					<span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-success-muted text-success">Resumed</span>
				) : null}

				{/* Step/agent context from first suspension */}
				{lifecycle.suspensions[0]?.stepName && (
					<span className="text-xs truncate text-muted-foreground">at {lifecycle.suspensions[0].stepName}</span>
				)}

				{/* Duration */}
				{durationMs != null && (
					<span className="text-[10px] text-muted-foreground tabular-nums ml-auto">{formatDuration(durationMs)}</span>
				)}
			</button>

			{isExpanded && (
				<div className="border-t px-3 py-2 space-y-2 text-xs">
					{/* Checkpoint event */}
					{lifecycle.checkpointEvent && (
						<div className="space-y-1">
							<div className="text-xs font-medium text-muted-foreground">Checkpoint</div>
							<DetailRow label="ID" value={lifecycle.checkpointId} mono />
							{lifecycle.runId && <DetailRow label="Run" value={lifecycle.runId} mono />}
							<DetailRow label="Time" value={formatTimestamp(lifecycle.checkpointEvent.timestamp)} />
						</div>
					)}

					{/* Suspensions */}
					{lifecycle.suspensions.map((sus, i) => (
						<div key={sus.suspensionId || i} className="space-y-1">
							<div className="text-xs font-medium text-muted-foreground">
								Suspension{lifecycle.suspensions.length > 1 ? ` ${i + 1}` : ""}
							</div>
							<DetailRow label="Type" value={sus.suspensionType} />
							{sus.stepName && <DetailRow label="Step" value={sus.stepName} />}
							{sus.agentName && <DetailRow label="Agent" value={sus.agentName} />}
							{sus.suspensionId && <DetailRow label="ID" value={sus.suspensionId} mono />}
							<DetailRow label="Time" value={formatTimestamp(sus.event.timestamp)} />
						</div>
					))}

					{/* Resumption */}
					{lifecycle.resumption && (
						<div className="space-y-1">
							<div className="text-xs font-medium text-muted-foreground">Resumption</div>
							{!!(lifecycle.resumption.payload as Record<string, unknown>).resumed_from_step && (
								<DetailRow
									label="From step"
									value={String((lifecycle.resumption.payload as Record<string, unknown>).resumed_from_step)}
								/>
							)}
							<DetailRow label="Time" value={formatTimestamp(lifecycle.resumption.timestamp)} />
							{durationMs != null && <DetailRow label="Duration" value={formatDuration(durationMs)} />}
						</div>
					)}

					{!lifecycle.resumption && lifecycle.suspensions.length > 0 && (
						<div className="text-xs text-warning-muted-foreground italic">Awaiting resumption…</div>
					)}
				</div>
			)}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Shared primitives
// ---------------------------------------------------------------------------

function SummaryPill({ label, value }: { label: string; value: string }) {
	return (
		<div className="flex items-center gap-1.5 px-2 py-1 rounded bg-muted/50">
			<span className="text-muted-foreground">{label}</span>
			<span className="font-medium">{value}</span>
		</div>
	);
}

function DetailRow({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
	return (
		<div className="flex items-start gap-2">
			<span className="text-muted-foreground shrink-0">{label}:</span>
			<span className={mono ? "font-mono text-[10px] break-all" : ""}>{value}</span>
		</div>
	);
}

function formatDuration(ms: number): string {
	if (ms < 1000) return `${Math.round(ms)}ms`;
	if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
	const minutes = Math.floor(ms / 60000);
	const seconds = Math.round((ms % 60000) / 1000);
	return `${minutes}m ${seconds}s`;
}

function formatTimestamp(ts: string): string {
	try {
		return new Date(ts).toLocaleTimeString();
	} catch {
		return ts;
	}
}
