import { useState } from "react";
import type { CapabilityPanelProps } from "../../registry/capability-panel-registry";
import type { TraceEvent } from "../../types";

// ---------------------------------------------------------------------------
// Revision workflow extraction
// ---------------------------------------------------------------------------

interface RevisionAttempt {
	attemptNumber: number;
	feedback: string;
}

interface RevisionWorkflow {
	stepName: string;
	workerCount: number | null;
	maxRevisions: number | null;
	attempts: RevisionAttempt[];
	finalDecision: string | null;
	totalAttempts: number | null;
}

function buildRevisionWorkflows(events: TraceEvent[]): RevisionWorkflow[] {
	const starts = events.filter((e) => e.event_type === "revision.start");
	const attempts = events.filter((e) => e.event_type === "revision.attempt");
	const completes = events.filter((e) => e.event_type === "revision.complete");

	// Group by step_name
	const workflowMap = new Map<string, RevisionWorkflow>();

	for (const start of starts) {
		const p = start.payload as Record<string, unknown>;
		const stepName = String(p.step_name ?? "unknown");
		workflowMap.set(stepName, {
			stepName,
			workerCount: p.worker_count != null ? Number(p.worker_count) : null,
			maxRevisions: p.max_revisions != null ? Number(p.max_revisions) : null,
			attempts: [],
			finalDecision: null,
			totalAttempts: null,
		});
	}

	for (const attempt of attempts) {
		const p = attempt.payload as Record<string, unknown>;
		const stepName = String(p.step_name ?? "unknown");
		const workflow = workflowMap.get(stepName);
		if (workflow) {
			workflow.attempts.push({
				attemptNumber: Number(p.attempt_number ?? 0),
				feedback: String(p.feedback ?? ""),
			});
		}
	}

	for (const complete of completes) {
		const p = complete.payload as Record<string, unknown>;
		const stepName = String(p.step_name ?? "unknown");
		const workflow = workflowMap.get(stepName);
		if (workflow) {
			workflow.finalDecision = p.final_decision != null ? String(p.final_decision) : null;
			workflow.totalAttempts = p.total_attempts != null ? Number(p.total_attempts) : null;
		}
	}

	return Array.from(workflowMap.values());
}

// ---------------------------------------------------------------------------
// Data extraction
// ---------------------------------------------------------------------------

interface HITLInteraction {
	request: TraceEvent;
	response: TraceEvent | null;
	requestId: string;
	requestType: string;
	prompt: string;
	context: string | null;
	agentName: string | null;
	toolName: string | null;
	decision: string | null;
	responseContent: string | null;
	waitDurationMs: number | null;
}

function buildInteractions(events: TraceEvent[]): HITLInteraction[] {
	const requests = events.filter((e) => e.event_type === "hitl.request");
	const responses = events.filter((e) => e.event_type === "hitl.response");

	// Index responses by request_id for fast lookup
	const responseByRequestId = new Map<string, TraceEvent>();
	for (const resp of responses) {
		const p = resp.payload as Record<string, unknown>;
		const reqId = String(p.request_id ?? "");
		if (reqId) responseByRequestId.set(reqId, resp);
	}

	return requests.map((req) => {
		const rp = req.payload as Record<string, unknown>;
		const requestId = String(rp.request_id ?? req.id);
		const resp = responseByRequestId.get(requestId) ?? null;
		const sp = resp ? (resp.payload as Record<string, unknown>) : null;

		let waitDurationMs: number | null = null;
		if (sp?.wait_duration_ms != null) {
			waitDurationMs = sp.wait_duration_ms as number;
		} else if (resp) {
			const start = new Date(req.timestamp).getTime();
			const end = new Date(resp.timestamp).getTime();
			const diff = end - start;
			if (diff >= 0) waitDurationMs = diff;
		}

		return {
			request: req,
			response: resp,
			requestId,
			requestType: String(rp.request_type ?? "unknown"),
			prompt: String(rp.prompt ?? ""),
			context: rp.context != null ? String(rp.context) : null,
			agentName: rp.agent_name != null ? String(rp.agent_name) : null,
			toolName: rp.tool_name != null ? String(rp.tool_name) : null,
			decision: sp?.decision != null ? String(sp.decision) : null,
			responseContent: sp?.has_content ? String(sp.content ?? "") : null,
			waitDurationMs,
		};
	});
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------

/** HITL panel — human-in-the-loop request/response pairs and statistics. */
export function HITLPanel({ events }: CapabilityPanelProps) {
	const hitlEvents = events.filter((e) => e.event_type === "hitl.request" || e.event_type === "hitl.response");
	const revisionEvents = events.filter((e) => e.event_type.startsWith("revision."));
	const revisionWorkflows = buildRevisionWorkflows(revisionEvents);

	if (hitlEvents.length === 0 && revisionEvents.length === 0) {
		return (
			<div className="p-4 text-sm text-muted-foreground">No human-in-the-loop events recorded for this agent.</div>
		);
	}

	const interactions = buildInteractions(hitlEvents);
	const pending = interactions.filter((i) => !i.response);
	const completed = interactions.filter((i) => i.response);

	// Summary stats
	const totalCount = interactions.length;
	const byType = new Map<string, number>();
	for (const i of interactions) {
		byType.set(i.requestType, (byType.get(i.requestType) ?? 0) + 1);
	}
	const approvalInteractions = interactions.filter((i) => i.requestType === "approval" && i.decision);
	const approvedCount = approvalInteractions.filter(
		(i) => i.decision === "approve" || i.decision === "approved",
	).length;
	const approvalRate =
		approvalInteractions.length > 0 ? Math.round((approvedCount / approvalInteractions.length) * 100) : null;
	const waitTimes = interactions.map((i) => i.waitDurationMs).filter((w): w is number => w != null);
	const avgWait = waitTimes.length > 0 ? Math.round(waitTimes.reduce((a, b) => a + b, 0) / waitTimes.length) : null;

	return (
		<div className="p-4 space-y-4">
			{/* Summary */}
			<div className="flex flex-wrap gap-4 text-xs">
				<SummaryPill label="Total" value={String(totalCount)} />
				{Array.from(byType.entries()).map(([type, count]) => (
					<SummaryPill key={type} label={type} value={String(count)} />
				))}
				{avgWait != null && <SummaryPill label="Avg wait" value={formatDuration(avgWait)} />}
				{approvalRate != null && <SummaryPill label="Approval rate" value={`${approvalRate}%`} />}
				{revisionWorkflows.length > 0 && <SummaryPill label="Revisions" value={String(revisionWorkflows.length)} />}
			</div>

			{/* Revision workflows */}
			{revisionWorkflows.length > 0 && (
				<div className="space-y-2">
					<div className="text-xs font-medium text-muted-foreground">Revision Workflows</div>
					{revisionWorkflows.map((workflow) => (
						<RevisionWorkflowCard key={workflow.stepName} workflow={workflow} />
					))}
				</div>
			)}

			{/* Pending requests */}
			{pending.length > 0 && (
				<div className="space-y-2">
					<div className="text-xs font-medium text-warning-muted-foreground">
						{pending.length} pending request{pending.length !== 1 ? "s" : ""}
					</div>
					{pending.map((interaction) => (
						<InteractionCard key={interaction.request.id} interaction={interaction} isPending />
					))}
				</div>
			)}

			{/* Completed interactions */}
			{completed.length > 0 && (
				<div className="space-y-2">
					{pending.length > 0 && <div className="text-xs font-medium text-muted-foreground">Completed</div>}
					{completed.map((interaction) => (
						<InteractionCard key={interaction.request.id} interaction={interaction} isPending={false} />
					))}
				</div>
			)}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Interaction Card
// ---------------------------------------------------------------------------

function InteractionCard({ interaction, isPending }: { interaction: HITLInteraction; isPending: boolean }) {
	const [isExpanded, setIsExpanded] = useState(false);

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
				<RequestTypeBadge type={interaction.requestType} />
				<span className="text-xs truncate flex-1">{truncate(interaction.prompt, 80)}</span>
				{interaction.decision && <DecisionBadge decision={interaction.decision} />}
				{isPending && (
					<span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-warning-muted text-warning-muted-foreground">
						pending
					</span>
				)}
				{interaction.waitDurationMs != null && (
					<span className="text-[10px] text-muted-foreground tabular-nums">
						{formatDuration(interaction.waitDurationMs)}
					</span>
				)}
			</button>

			{isExpanded && (
				<div className="border-t px-3 py-2 space-y-2 text-xs">
					{/* Request details */}
					<div className="space-y-1">
						<SectionLabel label="Request" />
						<DetailRow label="Type" value={interaction.requestType} />
						{interaction.agentName && <DetailRow label="Agent" value={interaction.agentName} />}
						{interaction.toolName && <DetailRow label="Tool" value={interaction.toolName} />}
						<div>
							<span className="text-muted-foreground">Prompt:</span>
							<div className="mt-1 bg-muted/50 rounded p-2 whitespace-pre-wrap max-h-[150px] overflow-y-auto">
								{interaction.prompt}
							</div>
						</div>
						{interaction.context && (
							<div>
								<span className="text-muted-foreground">Context:</span>
								<div className="mt-1 bg-muted/50 rounded p-2 whitespace-pre-wrap max-h-[100px] overflow-y-auto">
									{interaction.context}
								</div>
							</div>
						)}
					</div>

					{/* Response details */}
					{interaction.response && (
						<div className="space-y-1">
							<SectionLabel label="Response" />
							{interaction.decision && (
								<div className="flex items-center gap-2">
									<span className="text-muted-foreground">Decision:</span>
									<DecisionBadge decision={interaction.decision} />
								</div>
							)}
							{interaction.responseContent && (
								<div>
									<span className="text-muted-foreground">Content:</span>
									<div className="mt-1 bg-muted/50 rounded p-2 whitespace-pre-wrap max-h-[150px] overflow-y-auto">
										{interaction.responseContent}
									</div>
								</div>
							)}
							{interaction.waitDurationMs != null && (
								<DetailRow label="Wait time" value={formatDuration(interaction.waitDurationMs)} />
							)}
						</div>
					)}

					{!interaction.response && (
						<div className="text-xs text-warning-muted-foreground italic">Awaiting response…</div>
					)}
				</div>
			)}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Revision Workflow Card
// ---------------------------------------------------------------------------

const REVISION_DECISION_COLORS: Record<string, string> = {
	approve: "bg-success-muted text-success-muted-foreground",
	reject: "bg-destructive-muted text-destructive-muted-foreground",
	max_revisions_exceeded: "bg-warning-muted text-warning-muted-foreground",
};

function RevisionWorkflowCard({ workflow }: { workflow: RevisionWorkflow }) {
	const [isExpanded, setIsExpanded] = useState(false);

	const decisionColors = workflow.finalDecision
		? (REVISION_DECISION_COLORS[workflow.finalDecision] ?? "bg-muted text-muted-foreground")
		: null;

	return (
		<div className="border rounded-lg overflow-hidden">
			<button
				type="button"
				className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-accent/50 transition-colors text-left"
				onClick={() => setIsExpanded(!isExpanded)}
			>
				<span className="text-xs text-muted-foreground w-4">{isExpanded ? "▾" : "▸"}</span>
				<span className="text-xs font-medium truncate flex-1">{workflow.stepName}</span>
				{workflow.finalDecision && decisionColors && (
					<span className={`text-[10px] font-medium px-1.5 py-0.5 rounded ${decisionColors}`}>
						{workflow.finalDecision}
					</span>
				)}
				{workflow.totalAttempts != null && (
					<span className="text-[10px] text-muted-foreground tabular-nums">
						{workflow.totalAttempts} attempt{workflow.totalAttempts !== 1 ? "s" : ""}
					</span>
				)}
				{!workflow.finalDecision && (
					<span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-warning-muted text-warning-muted-foreground">
						in progress
					</span>
				)}
			</button>

			{isExpanded && (
				<div className="border-t px-3 py-2 space-y-2 text-xs">
					<div className="flex items-center gap-4">
						{workflow.workerCount != null && <DetailRow label="Workers" value={String(workflow.workerCount)} />}
						{workflow.maxRevisions != null && <DetailRow label="Max revisions" value={String(workflow.maxRevisions)} />}
					</div>

					{workflow.attempts.length > 0 && (
						<div className="space-y-1">
							<SectionLabel label="Attempts" />
							{workflow.attempts.map((attempt) => (
								<div key={attempt.attemptNumber} className="space-y-1">
									<div className="text-muted-foreground">Attempt {attempt.attemptNumber}</div>
									{attempt.feedback && <div className="bg-warning-muted rounded-md p-2">{attempt.feedback}</div>}
								</div>
							))}
						</div>
					)}

					{workflow.finalDecision && (
						<div className="space-y-1">
							<SectionLabel label="Outcome" />
							<div className="flex items-center gap-2">
								<span className="text-muted-foreground">Decision:</span>
								{decisionColors && (
									<span className={`text-[10px] font-medium px-1.5 py-0.5 rounded ${decisionColors}`}>
										{workflow.finalDecision}
									</span>
								)}
							</div>
						</div>
					)}
				</div>
			)}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Shared primitives
// ---------------------------------------------------------------------------

const REQUEST_TYPE_COLORS: Record<string, string> = {
	approval: "bg-info-muted text-info-muted-foreground",
	question: "bg-accent-status-muted text-accent-status-muted-foreground",
};

function RequestTypeBadge({ type }: { type: string }) {
	const colors = REQUEST_TYPE_COLORS[type] ?? "bg-muted text-muted-foreground";
	return <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded ${colors}`}>{type}</span>;
}

const DECISION_COLORS: Record<string, string> = {
	approve: "bg-success-muted text-success-muted-foreground",
	approved: "bg-success-muted text-success-muted-foreground",
	reject: "bg-destructive-muted text-destructive-muted-foreground",
	rejected: "bg-destructive-muted text-destructive-muted-foreground",
	modify: "bg-warning-muted text-warning-muted-foreground",
	answer: "bg-info-muted text-info-muted-foreground",
	escalate: "bg-destructive-muted text-destructive-muted-foreground",
	revise: "bg-warning-muted text-warning-muted-foreground",
};

function DecisionBadge({ decision }: { decision: string }) {
	const colors = DECISION_COLORS[decision] ?? "bg-muted text-muted-foreground";
	return <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded ${colors}`}>{decision}</span>;
}

function SummaryPill({ label, value }: { label: string; value: string }) {
	return (
		<div className="flex items-center gap-1.5 px-2 py-1 rounded bg-muted/50">
			<span className="text-muted-foreground">{label}</span>
			<span className="font-medium">{value}</span>
		</div>
	);
}

function SectionLabel({ label }: { label: string }) {
	return <div className="text-xs font-medium text-muted-foreground">{label}</div>;
}

function DetailRow({ label, value }: { label: string; value: string }) {
	return (
		<div className="flex items-start gap-2">
			<span className="text-muted-foreground shrink-0">{label}:</span>
			<span className="font-mono break-all">{value}</span>
		</div>
	);
}

function truncate(text: string, max: number): string {
	return text.length > max ? `${text.slice(0, max)}…` : text;
}

function formatDuration(ms: number): string {
	if (ms < 1000) return `${Math.round(ms)}ms`;
	if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
	const minutes = Math.floor(ms / 60000);
	const seconds = Math.round((ms % 60000) / 1000);
	return `${minutes}m ${seconds}s`;
}
