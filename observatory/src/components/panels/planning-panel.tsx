import { useState } from "react";
import type { CapabilityPanelProps } from "../../registry/capability-panel-registry";
import type { TraceEvent } from "../../types";

// ---------------------------------------------------------------------------
// Data extraction
// ---------------------------------------------------------------------------

interface PlanInfo {
	planId: string | null;
	planName: string;
	stepCount: number;
	goalCount: number;
	steps: Array<{ step_id: string; description: string; metadata?: Record<string, unknown> }>;
}

interface StepUpdate {
	event: TraceEvent;
	stepId: string;
	description: string;
	previousStatus: string;
	newStatus: string;
	hasResult: boolean;
}

interface PlanRevision {
	event: TraceEvent;
	reason: string;
	stepsBefore: number;
	stepsAfter: number;
	stepsPreserved: number;
}

interface GoalUpdate {
	event: TraceEvent;
	goalId: string;
	description: string;
	previousStatus: string;
	newStatus: string;
}

function extractPlan(events: TraceEvent[]): PlanInfo | null {
	const created = events.find((e) => e.event_type === "planning.plan.created");
	if (!created) return null;
	const p = created.payload as Record<string, unknown>;
	const steps = Array.isArray(p.steps)
		? (p.steps as Array<Record<string, unknown>>).map((s) => ({
				step_id: String(s.step_id ?? ""),
				description: String(s.description ?? ""),
				metadata: s.metadata as Record<string, unknown> | undefined,
			}))
		: [];
	return {
		planId: (p.plan_id as string) ?? null,
		planName: (p.plan_name as string) ?? "Unnamed plan",
		stepCount: (p.step_count as number) ?? steps.length,
		goalCount: (p.goal_count as number) ?? 0,
		steps,
	};
}

function extractStepUpdates(events: TraceEvent[]): StepUpdate[] {
	return events
		.filter((e) => e.event_type === "planning.step.updated")
		.map((e) => {
			const p = e.payload as Record<string, unknown>;
			return {
				event: e,
				stepId: String(p.step_id ?? ""),
				description: String(p.step_description ?? p.description ?? ""),
				previousStatus: String(p.previous_status ?? "unknown"),
				newStatus: String(p.new_status ?? "unknown"),
				hasResult: Boolean(p.has_result),
			};
		});
}

function extractRevisions(events: TraceEvent[]): PlanRevision[] {
	return events
		.filter((e) => e.event_type === "planning.plan.revised")
		.map((e) => {
			const p = e.payload as Record<string, unknown>;
			return {
				event: e,
				reason: String(p.reason ?? "No reason provided"),
				stepsBefore: (p.steps_before as number) ?? 0,
				stepsAfter: (p.steps_after as number) ?? 0,
				stepsPreserved: (p.steps_preserved as number) ?? 0,
			};
		});
}

function extractGoalUpdates(events: TraceEvent[]): GoalUpdate[] {
	return events
		.filter((e) => e.event_type === "planning.goal.status_changed")
		.map((e) => {
			const p = e.payload as Record<string, unknown>;
			return {
				event: e,
				goalId: String(p.goal_id ?? ""),
				description: String(p.goal_description ?? p.description ?? ""),
				previousStatus: String(p.previous_status ?? "unknown"),
				newStatus: String(p.new_status ?? "unknown"),
			};
		});
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------

/** Planning panel — plan lifecycle, step status, revisions, and goal tracking. */
export function PlanningPanel({ events }: CapabilityPanelProps) {
	const planningEvents = events.filter((e) => e.event_type.startsWith("planning."));

	if (planningEvents.length === 0) {
		return <div className="p-4 text-sm text-muted-foreground">No planning events recorded for this agent.</div>;
	}

	const plan = extractPlan(planningEvents);
	const stepUpdates = extractStepUpdates(planningEvents);
	const revisions = extractRevisions(planningEvents);
	const goalUpdates = extractGoalUpdates(planningEvents);

	return (
		<div className="p-4 space-y-4">
			<div className="text-xs text-muted-foreground">
				{planningEvents.length} planning event{planningEvents.length !== 1 ? "s" : ""}
			</div>

			{plan && <PlanOverview plan={plan} />}
			{stepUpdates.length > 0 && <StepStatusTimeline updates={stepUpdates} />}
			{revisions.length > 0 && <PlanRevisions revisions={revisions} />}
			{goalUpdates.length > 0 && <GoalTracking updates={goalUpdates} />}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Plan Overview
// ---------------------------------------------------------------------------

function PlanOverview({ plan }: { plan: PlanInfo }) {
	const [isExpanded, setIsExpanded] = useState(true);

	return (
		<div className="border rounded-lg overflow-hidden">
			<button
				type="button"
				className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-accent/50 transition-colors text-left"
				onClick={() => setIsExpanded(!isExpanded)}
			>
				<span className="text-xs text-muted-foreground w-4">{isExpanded ? "▾" : "▸"}</span>
				<span className="font-medium text-xs">Plan: {plan.planName}</span>
				<span className="text-[10px] text-muted-foreground bg-muted rounded-full px-1.5 py-0.5">
					{plan.stepCount} step{plan.stepCount !== 1 ? "s" : ""}
				</span>
				{plan.goalCount > 0 && (
					<span className="text-[10px] text-muted-foreground bg-muted rounded-full px-1.5 py-0.5">
						{plan.goalCount} goal{plan.goalCount !== 1 ? "s" : ""}
					</span>
				)}
			</button>
			{isExpanded && plan.steps.length > 0 && (
				<div className="border-t px-3 py-2 space-y-1">
					{plan.steps.map((step, i) => (
						<div key={step.step_id || i} className="flex items-start gap-2 text-xs">
							<span className="text-muted-foreground tabular-nums w-5 shrink-0">{i + 1}.</span>
							<span>{step.description}</span>
						</div>
					))}
				</div>
			)}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Step Status Timeline
// ---------------------------------------------------------------------------

function StepStatusTimeline({ updates }: { updates: StepUpdate[] }) {
	// Group by step ID to show status progression per step
	const byStep = new Map<string, StepUpdate[]>();
	for (const u of updates) {
		const existing = byStep.get(u.stepId) ?? [];
		existing.push(u);
		byStep.set(u.stepId, existing);
	}

	return (
		<div className="space-y-2">
			<div className="text-xs font-medium text-muted-foreground">Step Status Timeline</div>
			{Array.from(byStep.entries()).map(([stepId, stepUpdates]) => (
				<StepProgressRow key={stepId} stepId={stepId} description={stepUpdates[0].description} updates={stepUpdates} />
			))}
		</div>
	);
}

function StepProgressRow({
	stepId,
	description,
	updates,
}: {
	stepId: string;
	description: string;
	updates: StepUpdate[];
}) {
	const finalStatus = updates[updates.length - 1].newStatus;

	return (
		<div className="border rounded-lg px-3 py-2">
			<div className="flex items-center gap-2 text-xs">
				<span className="text-muted-foreground font-mono">{stepId}</span>
				<span className="truncate">{description}</span>
				<StatusBadge status={finalStatus} />
			</div>
			<div className="flex items-center gap-1 mt-1.5">
				{updates.map((u, i) => (
					// biome-ignore lint/suspicious/noArrayIndexKey: status updates have no unique ID
					<div key={i} className="flex items-center gap-1">
						{i > 0 && <span className="text-[10px] text-muted-foreground">→</span>}
						<span className={`text-[10px] px-1 py-0.5 rounded ${statusColor(u.newStatus)}`}>{u.newStatus}</span>
					</div>
				))}
			</div>
		</div>
	);
}

// ---------------------------------------------------------------------------
// Plan Revisions
// ---------------------------------------------------------------------------

function PlanRevisions({ revisions }: { revisions: PlanRevision[] }) {
	return (
		<div className="space-y-2">
			<div className="text-xs font-medium text-muted-foreground">Plan Revisions</div>
			{revisions.map((rev, i) => (
				<div key={rev.event.id} className="border rounded-lg px-3 py-2 border-warning-border bg-warning-muted/50">
					<div className="flex items-center gap-2 text-xs">
						<span className="font-medium">Revision {i + 1}</span>
						<span className="text-[10px] text-muted-foreground bg-muted rounded-full px-1.5 py-0.5">
							{rev.stepsBefore} → {rev.stepsAfter} steps
						</span>
						{rev.stepsPreserved > 0 && (
							<span className="text-[10px] text-muted-foreground">({rev.stepsPreserved} preserved)</span>
						)}
						<span className="ml-auto text-[10px] text-muted-foreground">{formatTime(rev.event.timestamp)}</span>
					</div>
					<div className="text-xs mt-1 text-muted-foreground">{rev.reason}</div>
				</div>
			))}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Goal Tracking
// ---------------------------------------------------------------------------

function GoalTracking({ updates }: { updates: GoalUpdate[] }) {
	// Group by goal ID
	const byGoal = new Map<string, GoalUpdate[]>();
	for (const u of updates) {
		const existing = byGoal.get(u.goalId) ?? [];
		existing.push(u);
		byGoal.set(u.goalId, existing);
	}

	return (
		<div className="space-y-2">
			<div className="text-xs font-medium text-muted-foreground">Goal Tracking</div>
			{Array.from(byGoal.entries()).map(([goalId, goalUpdates]) => {
				const finalStatus = goalUpdates[goalUpdates.length - 1].newStatus;
				const description = goalUpdates[0].description;

				return (
					<div key={goalId} className="border rounded-lg px-3 py-2">
						<div className="flex items-center gap-2 text-xs">
							<span className="truncate">{description}</span>
							<StatusBadge status={finalStatus} />
						</div>
						<div className="flex items-center gap-1 mt-1">
							{goalUpdates.map((u, i) => (
								// biome-ignore lint/suspicious/noArrayIndexKey: goal status updates have no unique ID
								<div key={i} className="flex items-center gap-1">
									{i > 0 && <span className="text-[10px] text-muted-foreground">→</span>}
									<span className={`text-[10px] px-1 py-0.5 rounded ${statusColor(u.newStatus)}`}>{u.newStatus}</span>
								</div>
							))}
						</div>
					</div>
				);
			})}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Shared UI primitives
// ---------------------------------------------------------------------------

const STATUS_COLORS: Record<string, string> = {
	completed: "bg-success-muted text-success-muted-foreground",
	achieved: "bg-success-muted text-success-muted-foreground",
	in_progress: "bg-info-muted text-info-muted-foreground",
	not_started: "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400",
	failed: "bg-destructive-muted text-destructive-muted-foreground",
	blocked: "bg-warning-muted text-warning-muted-foreground",
};

function statusColor(status: string): string {
	return STATUS_COLORS[status] ?? "bg-muted text-muted-foreground";
}

function StatusBadge({ status }: { status: string }) {
	return <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded ${statusColor(status)}`}>{status}</span>;
}

function formatTime(timestamp: string): string {
	try {
		const d = new Date(timestamp);
		return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
	} catch {
		return timestamp;
	}
}
