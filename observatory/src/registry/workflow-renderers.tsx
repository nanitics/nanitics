import { PayloadViewer } from "../components/event-detail/payload-viewer";
import type { TraceEvent } from "../types";
import type { EventDetailProps, EventRendererRegistration } from "./renderer-registry";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function truncate(text: string | undefined | null, max = 80): string {
	if (!text) return "";
	return text.length > max ? `${text.slice(0, max)}…` : text;
}

function formatDuration(ms: number | null | undefined): string {
	if (ms == null) return "—";
	if (ms < 1000) return `${ms}ms`;
	return `${(ms / 1000).toFixed(1)}s`;
}

// ---------------------------------------------------------------------------
// Workflow Renderers
// ---------------------------------------------------------------------------

function WorkflowStartRenderer({ event }: EventDetailProps) {
	const { workflow_name, workflow_type, step_count, metadata } = event.payload as {
		workflow_name?: string;
		workflow_type?: string;
		step_count?: number;
		metadata?: Record<string, unknown>;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{workflow_name && <span className="text-sm font-medium">{workflow_name}</span>}
				{workflow_type && (
					<span className="text-xs px-1.5 py-0.5 rounded bg-info-muted text-info">{workflow_type}</span>
				)}
			</div>
			{step_count != null && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Steps</span>
					<span className="font-mono tabular-nums">{step_count}</span>
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

function workflowStartSummary(event: TraceEvent): string {
	const { workflow_name, workflow_type, step_count } = event.payload as {
		workflow_name?: string;
		workflow_type?: string;
		step_count?: number;
	};
	const parts: string[] = [];
	if (workflow_name) parts.push(`'${workflow_name}'`);
	const details: string[] = [];
	if (workflow_type) details.push(workflow_type);
	if (step_count != null) details.push(`${step_count} steps`);
	if (details.length) parts.push(`(${details.join(", ")})`);
	return `Started workflow ${parts.join(" ")}`;
}

function WorkflowStructureRenderer({ event }: EventDetailProps) {
	const { workflow_name, steps } = event.payload as {
		workflow_name?: string;
		steps?: Array<{
			name: string;
			step_type: string;
			depends_on?: string[];
			parallel_group?: string | null;
		}>;
	};

	return (
		<div className="space-y-3">
			{workflow_name && <div className="text-sm font-medium">{workflow_name}</div>}
			{steps && steps.length > 0 && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Steps ({steps.length})</span>
					<div className="border border-border rounded-md overflow-hidden">
						<table className="w-full text-xs">
							<thead>
								<tr className="bg-muted/50">
									<th className="text-left px-2 py-1 font-medium text-muted-foreground">Name</th>
									<th className="text-left px-2 py-1 font-medium text-muted-foreground">Type</th>
									<th className="text-left px-2 py-1 font-medium text-muted-foreground">Depends On</th>
									<th className="text-left px-2 py-1 font-medium text-muted-foreground">Group</th>
								</tr>
							</thead>
							<tbody>
								{steps.map((step) => (
									<tr key={step.name} className="border-t border-border">
										<td className="px-2 py-1 font-mono">{step.name}</td>
										<td className="px-2 py-1">{step.step_type}</td>
										<td className="px-2 py-1">{step.depends_on?.join(", ") || "—"}</td>
										<td className="px-2 py-1">{step.parallel_group || "—"}</td>
									</tr>
								))}
							</tbody>
						</table>
					</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function workflowStructureSummary(event: TraceEvent): string {
	const { steps } = event.payload as { steps?: unknown[] };
	const count = steps?.length ?? 0;
	return `Workflow structure: ${count} steps`;
}

function WorkflowStepCompleteRenderer({ event }: EventDetailProps) {
	const { step_name, step_index, step_duration_ms, step_output } = event.payload as {
		step_name?: string;
		step_index?: number;
		step_duration_ms?: number | null;
		step_output?: string | null;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{step_name && <span className="text-sm font-mono font-medium">{step_name}</span>}
				{step_index != null && <span className="text-xs text-muted-foreground">#{step_index}</span>}
				<span className="text-xs px-1.5 py-0.5 rounded bg-success-muted text-success">completed</span>
			</div>
			{step_duration_ms != null && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Duration</span>
					<span className="font-mono tabular-nums">{formatDuration(step_duration_ms)}</span>
				</div>
			)}
			{step_output && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Output</span>
					<div className="text-sm text-foreground bg-muted/50 rounded-md p-2">{step_output}</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function workflowStepCompleteSummary(event: TraceEvent): string {
	const { step_name, step_duration_ms } = event.payload as {
		step_name?: string;
		step_duration_ms?: number | null;
	};
	const parts = [`Step '${step_name ?? "unknown"}' completed`];
	if (step_duration_ms != null) parts.push(`(${formatDuration(step_duration_ms)})`);
	return parts.join(" ");
}

function WorkflowCompleteRenderer({ event }: EventDetailProps) {
	const { workflow_name, workflow_type, total_steps_executed } = event.payload as {
		workflow_name?: string;
		workflow_type?: string;
		total_steps_executed?: number;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{workflow_name && <span className="text-sm font-medium">{workflow_name}</span>}
				{workflow_type && (
					<span className="text-xs px-1.5 py-0.5 rounded bg-info-muted text-info">{workflow_type}</span>
				)}
				<span className="text-xs px-1.5 py-0.5 rounded bg-success-muted text-success">completed</span>
			</div>
			{total_steps_executed != null && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Steps executed</span>
					<span className="font-mono tabular-nums">{total_steps_executed}</span>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function workflowCompleteSummary(event: TraceEvent): string {
	const { workflow_name, total_steps_executed } = event.payload as {
		workflow_name?: string;
		total_steps_executed?: number;
	};
	const parts = [`Workflow '${workflow_name ?? "unknown"}' completed`];
	if (total_steps_executed != null) parts.push(`(${total_steps_executed} steps)`);
	return parts.join(" ");
}

function WorkflowErrorRenderer({ event }: EventDetailProps) {
	const { workflow_name, workflow_type, error_type, error_message, failed_step } = event.payload as {
		workflow_name?: string;
		workflow_type?: string;
		error_type?: string;
		error_message?: string;
		failed_step?: string | null;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{workflow_name && <span className="text-sm font-medium">{workflow_name}</span>}
				{workflow_type && (
					<span className="text-xs px-1.5 py-0.5 rounded bg-info-muted text-info">{workflow_type}</span>
				)}
				<span className="text-xs px-1.5 py-0.5 rounded bg-destructive-muted text-destructive">error</span>
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
			{failed_step && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Failed step</span>
					<span className="font-mono">{failed_step}</span>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function workflowErrorSummary(event: TraceEvent): string {
	const { workflow_name, error_type } = event.payload as {
		workflow_name?: string;
		error_type?: string;
	};
	return `Workflow '${workflow_name ?? "unknown"}' failed: ${error_type ?? "unknown error"}`;
}

// ---------------------------------------------------------------------------
// Delegation / Handoff / Supervision Renderers
// ---------------------------------------------------------------------------

function DelegationRenderer({ event }: EventDetailProps) {
	const { caller_agent, delegate_agent, task, transfer_strategy } = event.payload as {
		caller_agent?: string;
		delegate_agent?: string;
		task?: string;
		transfer_strategy?: string;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2 text-sm">
				<span className="font-mono">{caller_agent ?? "?"}</span>
				<span className="text-muted-foreground">→</span>
				<span className="font-mono">{delegate_agent ?? "?"}</span>
			</div>
			{transfer_strategy && (
				<span className="text-xs px-1.5 py-0.5 rounded bg-info-muted text-info">{transfer_strategy}</span>
			)}
			{task && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Task</span>
					<div className="text-sm text-foreground">{task}</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function delegationSummary(event: TraceEvent): string {
	const { delegate_agent, transfer_strategy } = event.payload as {
		delegate_agent?: string;
		transfer_strategy?: string;
	};
	const parts = [`Delegated to ${delegate_agent ?? "unknown"}`];
	if (transfer_strategy) parts.push(`(${transfer_strategy})`);
	return parts.join(" ");
}

function HandoffRenderer({ event }: EventDetailProps) {
	const { from_agent, to_agent, payload_fields, payload_size } = event.payload as {
		from_agent?: string;
		to_agent?: string;
		payload_fields?: string[];
		payload_size?: number;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2 text-sm">
				<span className="font-mono">{from_agent ?? "?"}</span>
				<span className="text-muted-foreground">→</span>
				<span className="font-mono">{to_agent ?? "?"}</span>
			</div>
			{payload_size != null && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Payload size</span>
					<span className="font-mono tabular-nums">{payload_size} bytes</span>
				</div>
			)}
			{payload_fields && payload_fields.length > 0 && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Fields</span>
					<div className="flex flex-wrap gap-1">
						{payload_fields.map((f) => (
							<span key={f} className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-mono">
								{f}
							</span>
						))}
					</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function handoffSummary(event: TraceEvent): string {
	const { from_agent, to_agent, payload_size } = event.payload as {
		from_agent?: string;
		to_agent?: string;
		payload_size?: number;
	};
	const parts = [`Handoff: ${from_agent ?? "?"} → ${to_agent ?? "?"}`];
	if (payload_size != null) parts.push(`(${payload_size} bytes)`);
	return parts.join(" ");
}

function SupervisionRenderer({ event }: EventDetailProps) {
	const { supervised_agent, action, trigger_name, feedback, reassigned_to, attempt } = event.payload as {
		supervised_agent?: string;
		action?: string;
		trigger_name?: string;
		feedback?: string | null;
		reassigned_to?: string | null;
		attempt?: number;
	};

	const actionColors: Record<string, string> = {
		approve: "bg-success-muted text-success",
		retry: "bg-warning-muted text-warning",
		reassign: "bg-info-muted text-info",
		abort: "bg-destructive-muted text-destructive",
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{supervised_agent && <span className="text-sm font-mono">{supervised_agent}</span>}
				{action && (
					<span className={`text-xs px-1.5 py-0.5 rounded ${actionColors[action] ?? "bg-muted text-muted-foreground"}`}>
						{action}
					</span>
				)}
				{attempt != null && <span className="text-xs text-muted-foreground">attempt #{attempt}</span>}
			</div>
			{trigger_name && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Trigger</span>
					<span className="font-mono">{trigger_name}</span>
				</div>
			)}
			{feedback && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Feedback</span>
					<div className="text-sm text-foreground">{feedback}</div>
				</div>
			)}
			{reassigned_to && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Reassigned to</span>
					<span className="font-mono">{reassigned_to}</span>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function supervisionSummary(event: TraceEvent): string {
	const { action, supervised_agent } = event.payload as {
		action?: string;
		supervised_agent?: string;
	};
	return `Supervision: ${action ?? "unknown"} on ${supervised_agent ?? "unknown"}`;
}

// ---------------------------------------------------------------------------
// Bidding Renderers
// ---------------------------------------------------------------------------

function BiddingStartRenderer({ event }: EventDetailProps) {
	const { task, participant_names } = event.payload as {
		task?: string;
		participant_names?: string[];
	};

	return (
		<div className="space-y-3">
			{participant_names && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Participants</span>
					<span className="font-mono tabular-nums">{participant_names.length}</span>
				</div>
			)}
			{participant_names && participant_names.length > 0 && (
				<div className="flex flex-wrap gap-1">
					{participant_names.map((n) => (
						<span key={n} className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
							{n}
						</span>
					))}
				</div>
			)}
			{task && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Task</span>
					<div className="text-sm text-foreground">{task}</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function biddingStartSummary(event: TraceEvent): string {
	const { participant_names } = event.payload as { participant_names?: string[] };
	return `Bidding started (${participant_names?.length ?? 0} participants)`;
}

function BidReceivedRenderer({ event }: EventDetailProps) {
	const { agent_name, confidence, reasoning, estimated_cost } = event.payload as {
		agent_name?: string;
		confidence?: number;
		reasoning?: string;
		estimated_cost?: number | null;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{agent_name && <span className="text-sm font-mono">{agent_name}</span>}
				{confidence != null && <span className="text-xs font-mono tabular-nums">{Math.round(confidence * 100)}%</span>}
			</div>
			{confidence != null && (
				<div className="w-full bg-muted rounded-full h-1.5">
					<div className="bg-primary rounded-full h-1.5" style={{ width: `${Math.round(confidence * 100)}%` }} />
				</div>
			)}
			{reasoning && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Reasoning</span>
					<div className="text-sm text-foreground">{reasoning}</div>
				</div>
			)}
			{estimated_cost != null && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Estimated cost</span>
					<span className="font-mono tabular-nums">{estimated_cost}</span>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function bidReceivedSummary(event: TraceEvent): string {
	const { agent_name, confidence } = event.payload as {
		agent_name?: string;
		confidence?: number;
	};
	const pct = confidence != null ? `${Math.round(confidence * 100)}%` : "?";
	return `Bid from ${agent_name ?? "unknown"}: ${pct}`;
}

function BidAllocatedRenderer({ event }: EventDetailProps) {
	const { winner, confidence, total_bids, rejection_reason } = event.payload as {
		winner?: string | null;
		confidence?: number | null;
		total_bids?: number;
		rejection_reason?: string | null;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{winner ? (
					<>
						<span className="text-sm font-mono font-medium">{winner}</span>
						<span className="text-xs px-1.5 py-0.5 rounded bg-success-muted text-success">winner</span>
					</>
				) : (
					<span className="text-xs px-1.5 py-0.5 rounded bg-warning-muted text-warning">no winner</span>
				)}
			</div>
			{confidence != null && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Confidence</span>
					<span className="font-mono tabular-nums">{Math.round(confidence * 100)}%</span>
				</div>
			)}
			{total_bids != null && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Total bids</span>
					<span className="font-mono tabular-nums">{total_bids}</span>
				</div>
			)}
			{rejection_reason && (
				<div className="text-sm text-warning bg-warning-muted rounded-md p-2">{rejection_reason}</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function bidAllocatedSummary(event: TraceEvent): string {
	const { winner } = event.payload as { winner?: string | null };
	return winner ? `Allocated to ${winner}` : "All bids rejected";
}

function BiddingCompleteRenderer({ event }: EventDetailProps) {
	const { winner, total_participants, allocated } = event.payload as {
		winner?: string | null;
		total_participants?: number;
		allocated?: boolean;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{winner ? (
					<span className="text-sm font-mono font-medium">{winner}</span>
				) : (
					<span className="text-sm text-muted-foreground">No winner</span>
				)}
				{allocated != null && (
					<span
						className={`text-xs px-1.5 py-0.5 rounded ${allocated ? "bg-success-muted text-success" : "bg-warning-muted text-warning"}`}
					>
						{allocated ? "allocated" : "not allocated"}
					</span>
				)}
			</div>
			{total_participants != null && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Participants</span>
					<span className="font-mono tabular-nums">{total_participants}</span>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function biddingCompleteSummary(event: TraceEvent): string {
	const { winner } = event.payload as { winner?: string | null };
	return `Bidding complete: ${winner ?? "no winner"}`;
}

// ---------------------------------------------------------------------------
// Debate Renderers
// ---------------------------------------------------------------------------

function DebateStartRenderer({ event }: EventDetailProps) {
	const { task, debater_names, positions, max_rounds, resolution_strategy } = event.payload as {
		task?: string;
		debater_names?: string[];
		positions?: Record<string, string>;
		max_rounds?: number;
		resolution_strategy?: string;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2 text-xs">
				{debater_names && <span>{debater_names.length} debaters</span>}
				{max_rounds != null && <span className="text-muted-foreground">max {max_rounds} rounds</span>}
				{resolution_strategy && (
					<span className="px-1.5 py-0.5 rounded bg-info-muted text-info">{resolution_strategy}</span>
				)}
			</div>
			{positions && Object.keys(positions).length > 0 && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Initial positions</span>
					{Object.entries(positions).map(([agent, pos]) => (
						<div key={agent} className="flex items-center gap-2 text-xs">
							<span className="font-mono">{agent}</span>
							<span className="text-muted-foreground">—</span>
							<span>{pos}</span>
						</div>
					))}
				</div>
			)}
			{task && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Task</span>
					<div className="text-sm text-foreground">{task}</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function debateStartSummary(event: TraceEvent): string {
	const { debater_names, max_rounds } = event.payload as {
		debater_names?: string[];
		max_rounds?: number;
	};
	const parts = [`Debate started (${debater_names?.length ?? 0} debaters`];
	if (max_rounds != null) parts.push(`, max ${max_rounds} rounds)`);
	else parts.push(")");
	return parts.join("");
}

function DebateArgumentRenderer({ event }: EventDetailProps) {
	const { round, agent_name, position, argument } = event.payload as {
		round?: number;
		agent_name?: string;
		position?: string;
		argument?: string;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{round != null && <span className="text-xs text-muted-foreground">Round {round}</span>}
				{agent_name && <span className="text-sm font-mono">{agent_name}</span>}
				{position && <span className="text-xs px-1.5 py-0.5 rounded bg-info-muted text-info">{position}</span>}
			</div>
			{argument && <div className="text-sm text-foreground bg-muted/50 rounded-md p-2">{argument}</div>}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function debateArgumentSummary(event: TraceEvent): string {
	const { round, agent_name, position } = event.payload as {
		round?: number;
		agent_name?: string;
		position?: string;
	};
	const parts: string[] = [];
	if (round != null) parts.push(`Round ${round}:`);
	parts.push(agent_name ?? "unknown");
	if (position) parts.push(`argues (${position})`);
	return parts.join(" ");
}

function DebateResolutionRenderer({ event }: EventDetailProps) {
	const { winner, reasoning, rounds_completed } = event.payload as {
		winner?: string | null;
		reasoning?: string;
		rounds_completed?: number;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{winner ? (
					<span className="text-sm font-mono font-medium">{winner}</span>
				) : (
					<span className="text-sm text-muted-foreground">No winner</span>
				)}
				{rounds_completed != null && <span className="text-xs text-muted-foreground">{rounds_completed} rounds</span>}
			</div>
			{reasoning && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Reasoning</span>
					<div className="text-sm text-foreground bg-muted/50 rounded-md p-2">{reasoning}</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function debateResolutionSummary(event: TraceEvent): string {
	const { winner, rounds_completed } = event.payload as {
		winner?: string | null;
		rounds_completed?: number;
	};
	const parts = [`Resolved: ${winner ?? "no winner"}`];
	if (rounds_completed != null) parts.push(`(${rounds_completed} rounds)`);
	return parts.join(" ");
}

function DebateCompleteRenderer({ event }: EventDetailProps) {
	const { winner, rounds_completed, total_arguments, termination_reason } = event.payload as {
		winner?: string | null;
		rounds_completed?: number;
		total_arguments?: number;
		termination_reason?: string;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{winner ? (
					<span className="text-sm font-mono font-medium">{winner}</span>
				) : (
					<span className="text-sm text-muted-foreground">No winner</span>
				)}
				<span className="text-xs px-1.5 py-0.5 rounded bg-success-muted text-success">complete</span>
			</div>
			<div className="flex items-center gap-4 text-xs">
				{rounds_completed != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Rounds</span>
						<span className="font-mono tabular-nums">{rounds_completed}</span>
					</div>
				)}
				{total_arguments != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Arguments</span>
						<span className="font-mono tabular-nums">{total_arguments}</span>
					</div>
				)}
			</div>
			{termination_reason && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Termination</span>
					<span>{termination_reason}</span>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function debateCompleteSummary(event: TraceEvent): string {
	const { winner, total_arguments } = event.payload as {
		winner?: string | null;
		total_arguments?: number;
	};
	const parts = [`Debate complete: ${winner ?? "no winner"}`];
	if (total_arguments != null) parts.push(`(${total_arguments} arguments)`);
	return parts.join(" ");
}

// ---------------------------------------------------------------------------
// Consensus Renderers
// ---------------------------------------------------------------------------

function ConsensusStartRenderer({ event }: EventDetailProps) {
	const { task, agent_names, strategy, deliberation_enabled } = event.payload as {
		task?: string;
		agent_names?: string[];
		strategy?: string;
		deliberation_enabled?: boolean;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2 text-xs">
				{agent_names && <span>{agent_names.length} agents</span>}
				{strategy && <span className="px-1.5 py-0.5 rounded bg-info-muted text-info">{strategy}</span>}
				{deliberation_enabled && (
					<span className="px-1.5 py-0.5 rounded bg-muted text-muted-foreground">deliberation</span>
				)}
			</div>
			{agent_names && agent_names.length > 0 && (
				<div className="flex flex-wrap gap-1">
					{agent_names.map((n) => (
						<span key={n} className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
							{n}
						</span>
					))}
				</div>
			)}
			{task && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Task</span>
					<div className="text-sm text-foreground">{task}</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function consensusStartSummary(event: TraceEvent): string {
	const { agent_names, strategy } = event.payload as {
		agent_names?: string[];
		strategy?: string;
	};
	return `Consensus started (${agent_names?.length ?? 0} agents, ${strategy ?? "unknown"})`;
}

function ConsensusVoteRenderer({ event }: EventDetailProps) {
	const { agent_name, output, round, error } = event.payload as {
		agent_name?: string;
		output?: string;
		round?: number;
		error?: string | null;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{agent_name && <span className="text-sm font-mono">{agent_name}</span>}
				{round != null && <span className="text-xs text-muted-foreground">round {round}</span>}
				{error && <span className="text-xs px-1.5 py-0.5 rounded bg-destructive-muted text-destructive">error</span>}
			</div>
			{output && (
				<div className="text-sm text-foreground bg-muted/50 rounded-md p-2 whitespace-pre-wrap max-h-60 overflow-auto">
					{output}
				</div>
			)}
			{error && (
				<div className="text-sm text-destructive-muted-foreground bg-destructive-muted rounded-md p-2">{error}</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function consensusVoteSummary(event: TraceEvent): string {
	const { agent_name, round } = event.payload as {
		agent_name?: string;
		round?: number;
	};
	const parts = [`Vote from ${agent_name ?? "unknown"}`];
	if (round != null) parts.push(`(round ${round})`);
	return parts.join(" ");
}

function ConsensusAgreementRenderer({ event }: EventDetailProps) {
	const { round, agreement_level, converged } = event.payload as {
		round?: number;
		agreement_level?: number;
		converged?: boolean;
	};

	const pct = agreement_level != null ? Math.round(agreement_level * 100) : null;

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{pct != null && <span className="text-sm font-mono tabular-nums">{pct}%</span>}
				{round != null && <span className="text-xs text-muted-foreground">round {round}</span>}
				{converged && <span className="text-xs px-1.5 py-0.5 rounded bg-success-muted text-success">converged</span>}
			</div>
			{pct != null && (
				<div className="w-full bg-muted rounded-full h-1.5">
					<div
						className={`rounded-full h-1.5 ${converged ? "bg-success" : "bg-primary"}`}
						style={{ width: `${pct}%` }}
					/>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function consensusAgreementSummary(event: TraceEvent): string {
	const { agreement_level, round } = event.payload as {
		agreement_level?: number;
		round?: number;
	};
	const pct = agreement_level != null ? `${Math.round(agreement_level * 100)}%` : "?";
	const parts = [`Agreement: ${pct}`];
	if (round != null) parts.push(`(round ${round})`);
	return parts.join(" ");
}

function ConsensusCompleteRenderer({ event }: EventDetailProps) {
	const { strategy, rounds_completed, final_agreement, agents_participated, termination_reason } = event.payload as {
		strategy?: string;
		rounds_completed?: number;
		final_agreement?: number;
		agents_participated?: number;
		termination_reason?: string;
	};

	const pct = final_agreement != null ? Math.round(final_agreement * 100) : null;

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{strategy && <span className="text-xs px-1.5 py-0.5 rounded bg-info-muted text-info">{strategy}</span>}
				<span className="text-xs px-1.5 py-0.5 rounded bg-success-muted text-success">complete</span>
			</div>
			{pct != null && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Final agreement</span>
					<div className="flex items-center gap-2">
						<div className="w-full bg-muted rounded-full h-1.5">
							<div className="bg-primary rounded-full h-1.5" style={{ width: `${pct}%` }} />
						</div>
						<span className="text-xs font-mono tabular-nums">{pct}%</span>
					</div>
				</div>
			)}
			<div className="flex items-center gap-4 text-xs">
				{rounds_completed != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Rounds</span>
						<span className="font-mono tabular-nums">{rounds_completed}</span>
					</div>
				)}
				{agents_participated != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Agents</span>
						<span className="font-mono tabular-nums">{agents_participated}</span>
					</div>
				)}
			</div>
			{termination_reason && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Termination</span>
					<span>{termination_reason}</span>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function consensusCompleteSummary(event: TraceEvent): string {
	const { final_agreement, strategy } = event.payload as {
		final_agreement?: number;
		strategy?: string;
	};
	const pct = final_agreement != null ? `${Math.round(final_agreement * 100)}%` : "?";
	return `Consensus: ${pct} (${strategy ?? "unknown"})`;
}

// ---------------------------------------------------------------------------
// Broadcast Renderers
// ---------------------------------------------------------------------------

function BroadcastStartRenderer({ event }: EventDetailProps) {
	const { task, agent_names, response_strategy } = event.payload as {
		task?: string;
		agent_names?: string[];
		response_strategy?: string;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2 text-xs">
				{agent_names && <span>{agent_names.length} agents</span>}
				{response_strategy && (
					<span className="px-1.5 py-0.5 rounded bg-info-muted text-info">{response_strategy}</span>
				)}
			</div>
			{agent_names && agent_names.length > 0 && (
				<div className="flex flex-wrap gap-1">
					{agent_names.map((n) => (
						<span key={n} className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
							{n}
						</span>
					))}
				</div>
			)}
			{task && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Task</span>
					<div className="text-sm text-foreground">{task}</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function broadcastStartSummary(event: TraceEvent): string {
	const { agent_names, response_strategy } = event.payload as {
		agent_names?: string[];
		response_strategy?: string;
	};
	const parts = [`Broadcast to ${agent_names?.length ?? 0} agents`];
	if (response_strategy) parts.push(`(${response_strategy})`);
	return parts.join(" ");
}

function BroadcastResponseRenderer({ event }: EventDetailProps) {
	const { agent_name, output, steps, error } = event.payload as {
		agent_name?: string;
		output?: string;
		steps?: number;
		error?: string | null;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{agent_name && <span className="text-sm font-mono">{agent_name}</span>}
				{steps != null && <span className="text-xs text-muted-foreground">{steps} steps</span>}
				{error && <span className="text-xs px-1.5 py-0.5 rounded bg-destructive-muted text-destructive">error</span>}
			</div>
			{output && (
				<div className="text-sm text-foreground bg-muted/50 rounded-md p-2 whitespace-pre-wrap max-h-60 overflow-auto">
					{output}
				</div>
			)}
			{error && (
				<div className="text-sm text-destructive-muted-foreground bg-destructive-muted rounded-md p-2">{error}</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function broadcastResponseSummary(event: TraceEvent): string {
	const { agent_name, steps } = event.payload as {
		agent_name?: string;
		steps?: number;
	};
	const parts = [`Response from ${agent_name ?? "unknown"}`];
	if (steps != null) parts.push(`(${steps} steps)`);
	return parts.join(" ");
}

function BroadcastCompleteRenderer({ event }: EventDetailProps) {
	const { total_agents, responses_collected, response_strategy, aggregated_output } = event.payload as {
		total_agents?: number;
		responses_collected?: number;
		response_strategy?: string;
		aggregated_output?: string;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				<span className="text-xs px-1.5 py-0.5 rounded bg-success-muted text-success">complete</span>
				{response_strategy && (
					<span className="text-xs px-1.5 py-0.5 rounded bg-info-muted text-info">{response_strategy}</span>
				)}
			</div>
			<div className="flex items-center gap-4 text-xs">
				{responses_collected != null && total_agents != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Responses</span>
						<span className="font-mono tabular-nums">
							{responses_collected}/{total_agents}
						</span>
					</div>
				)}
			</div>
			{aggregated_output && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Aggregated output</span>
					<div className="text-sm text-foreground bg-muted/50 rounded-md p-2 whitespace-pre-wrap max-h-60 overflow-auto">
						{aggregated_output}
					</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function broadcastCompleteSummary(event: TraceEvent): string {
	const { responses_collected, total_agents } = event.payload as {
		responses_collected?: number;
		total_agents?: number;
	};
	return `Broadcast complete (${responses_collected ?? 0}/${total_agents ?? 0} responses)`;
}

// ---------------------------------------------------------------------------
// Blackboard Renderers
// ---------------------------------------------------------------------------

function BlackboardStartRenderer({ event }: EventDetailProps) {
	const { task, agent_names, control_strategy, max_rounds } = event.payload as {
		task?: string;
		agent_names?: string[];
		control_strategy?: string;
		max_rounds?: number;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2 text-xs">
				{agent_names && <span>{agent_names.length} agents</span>}
				{control_strategy && <span className="px-1.5 py-0.5 rounded bg-info-muted text-info">{control_strategy}</span>}
				{max_rounds != null && <span className="text-muted-foreground">max {max_rounds} rounds</span>}
			</div>
			{agent_names && agent_names.length > 0 && (
				<div className="flex flex-wrap gap-1">
					{agent_names.map((n) => (
						<span key={n} className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
							{n}
						</span>
					))}
				</div>
			)}
			{task && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Task</span>
					<div className="text-sm text-foreground">{task}</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function blackboardStartSummary(event: TraceEvent): string {
	const { agent_names, max_rounds } = event.payload as {
		agent_names?: string[];
		max_rounds?: number;
	};
	return `Blackboard started (${agent_names?.length ?? 0} agents, max ${max_rounds ?? "?"} rounds)`;
}

function BlackboardRoundRenderer({ event }: EventDetailProps) {
	const { round_number, agents_activated, contributions, total_contributions, round_entries } = event.payload as {
		round_number?: number;
		agents_activated?: string[];
		contributions?: number;
		total_contributions?: number;
		round_entries?: Array<{
			operation: string;
			author: string;
			content: string;
			scope?: string | null;
			entry_id: string;
			original_entry_id?: string | null;
			retract_reason?: string | null;
		}>;
	};

	const entries = round_entries ?? [];

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{round_number != null && <span className="text-sm font-medium">Round {round_number}</span>}
				{contributions != null && <span className="text-xs text-muted-foreground">{contributions} contributions</span>}
			</div>
			{agents_activated && agents_activated.length > 0 && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Agents activated</span>
					<div className="flex flex-wrap gap-1">
						{agents_activated.map((n) => (
							<span key={n} className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
								{n}
							</span>
						))}
					</div>
				</div>
			)}
			{entries.length > 0 && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Entries</span>
					{entries.map((entry) => (
						<div key={entry.entry_id} className="text-xs border-l-2 border-border pl-2 py-0.5">
							<div className="flex items-center gap-1.5 flex-wrap">
								<span className="text-[10px] px-1 py-0 rounded bg-muted text-muted-foreground">{entry.operation}</span>
								<span className="font-medium">{entry.author}</span>
								{entry.scope && (
									<span className="text-[10px] px-1 py-0 rounded bg-muted text-muted-foreground">{entry.scope}</span>
								)}
								{entry.operation === "supersede" && entry.original_entry_id && (
									<span className="text-muted-foreground">
										←{" "}
										{entry.original_entry_id.length > 8
											? `${entry.original_entry_id.slice(0, 8)}…`
											: entry.original_entry_id}
									</span>
								)}
								{entry.operation === "retract" && entry.retract_reason && (
									<span className="text-muted-foreground italic">{entry.retract_reason}</span>
								)}
							</div>
							{entry.content && (
								<div className="mt-0.5 text-muted-foreground whitespace-pre-wrap break-words">{entry.content}</div>
							)}
						</div>
					))}
				</div>
			)}
			{total_contributions != null && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Total contributions</span>
					<span className="font-mono tabular-nums">{total_contributions}</span>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function blackboardRoundSummary(event: TraceEvent): string {
	const { round_number, contributions } = event.payload as {
		round_number?: number;
		contributions?: number;
	};
	return `Round ${round_number ?? "?"}: ${contributions ?? 0} contributions`;
}

function BlackboardCompleteRenderer({ event }: EventDetailProps) {
	const { rounds_completed, termination_reason, total_contributions, agent_contributions } = event.payload as {
		rounds_completed?: number;
		termination_reason?: string;
		total_contributions?: number;
		agent_contributions?: Record<string, number>;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				<span className="text-xs px-1.5 py-0.5 rounded bg-success-muted text-success">complete</span>
			</div>
			<div className="flex items-center gap-4 text-xs">
				{rounds_completed != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Rounds</span>
						<span className="font-mono tabular-nums">{rounds_completed}</span>
					</div>
				)}
				{total_contributions != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Contributions</span>
						<span className="font-mono tabular-nums">{total_contributions}</span>
					</div>
				)}
			</div>
			{termination_reason && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Termination</span>
					<span>{termination_reason}</span>
				</div>
			)}
			{agent_contributions && Object.keys(agent_contributions).length > 0 && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Per-agent contributions</span>
					{Object.entries(agent_contributions).map(([agent, count]) => (
						<div key={agent} className="flex items-center gap-2 text-xs">
							<span className="font-mono">{agent}</span>
							<span className="font-mono tabular-nums">{count}</span>
						</div>
					))}
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function blackboardCompleteSummary(event: TraceEvent): string {
	const { rounds_completed, total_contributions } = event.payload as {
		rounds_completed?: number;
		total_contributions?: number;
	};
	return `Blackboard complete (${rounds_completed ?? 0} rounds, ${total_contributions ?? 0} contributions)`;
}

// ---------------------------------------------------------------------------
// Peer Network Renderers
// ---------------------------------------------------------------------------

function PeerNetworkStartRenderer({ event }: EventDetailProps) {
	const { task, entry_agent, peer_names, peer_descriptions, max_invocations } = event.payload as {
		task?: string;
		entry_agent?: string;
		peer_names?: string[];
		peer_descriptions?: Record<string, string>;
		max_invocations?: number;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2 text-xs">
				{entry_agent && (
					<>
						<span className="text-muted-foreground">Entry:</span>
						<span className="font-mono">{entry_agent}</span>
					</>
				)}
				{max_invocations != null && <span className="text-muted-foreground">budget: {max_invocations}</span>}
			</div>
			{peer_names && peer_names.length > 0 && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Peers</span>
					{peer_names.map((n) => (
						<div key={n} className="flex items-center gap-2 text-xs">
							<span className="font-mono">{n}</span>
							{peer_descriptions?.[n] && (
								<span className="text-muted-foreground">{truncate(peer_descriptions[n], 60)}</span>
							)}
						</div>
					))}
				</div>
			)}
			{task && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Task</span>
					<div className="text-sm text-foreground">{task}</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function peerNetworkStartSummary(event: TraceEvent): string {
	const { entry_agent } = event.payload as { entry_agent?: string };
	return `Peer network started (entry: ${entry_agent ?? "unknown"})`;
}

function PeerConsultationRenderer({ event }: EventDetailProps) {
	const { from_agent, to_agent, message, consultation_number, remaining_budget } = event.payload as {
		from_agent?: string;
		to_agent?: string;
		message?: string;
		consultation_number?: number;
		remaining_budget?: number;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2 text-sm">
				<span className="font-mono">{from_agent ?? "?"}</span>
				<span className="text-muted-foreground">→</span>
				<span className="font-mono">{to_agent ?? "?"}</span>
				{consultation_number != null && <span className="text-xs text-muted-foreground">#{consultation_number}</span>}
			</div>
			{remaining_budget != null && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Remaining budget</span>
					<span className="font-mono tabular-nums">{remaining_budget}</span>
				</div>
			)}
			{message && (
				<div className="text-sm text-foreground bg-muted/50 rounded-md p-2 whitespace-pre-wrap max-h-60 overflow-auto">
					{message}
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function peerConsultationSummary(event: TraceEvent): string {
	const { from_agent, to_agent } = event.payload as {
		from_agent?: string;
		to_agent?: string;
	};
	return `Consultation: ${from_agent ?? "?"} → ${to_agent ?? "?"}`;
}

function PeerNetworkCompleteRenderer({ event }: EventDetailProps) {
	const { entry_agent, total_consultations, invocations_used, agents_consulted, termination_reason } =
		event.payload as {
			entry_agent?: string;
			total_consultations?: number;
			invocations_used?: number;
			agents_consulted?: string[];
			termination_reason?: string;
		};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				<span className="text-xs px-1.5 py-0.5 rounded bg-success-muted text-success">complete</span>
				{entry_agent && <span className="text-xs font-mono">{entry_agent}</span>}
			</div>
			<div className="flex items-center gap-4 text-xs">
				{total_consultations != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Consultations</span>
						<span className="font-mono tabular-nums">{total_consultations}</span>
					</div>
				)}
				{invocations_used != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Budget used</span>
						<span className="font-mono tabular-nums">{invocations_used}</span>
					</div>
				)}
			</div>
			{agents_consulted && agents_consulted.length > 0 && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Agents consulted</span>
					<div className="flex flex-wrap gap-1">
						{agents_consulted.map((n) => (
							<span key={n} className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
								{n}
							</span>
						))}
					</div>
				</div>
			)}
			{termination_reason && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Termination</span>
					<span>{termination_reason}</span>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function peerNetworkCompleteSummary(event: TraceEvent): string {
	const { total_consultations } = event.payload as { total_consultations?: number };
	return `Peer network complete (${total_consultations ?? 0} consultations)`;
}

// ---------------------------------------------------------------------------
// Message Bus Renderers
// ---------------------------------------------------------------------------

function MessageBusStartRenderer({ event }: EventDetailProps) {
	const { seed_topics, subscriber_count, subscriptions, max_messages, max_depth } = event.payload as {
		seed_topics?: string[];
		subscriber_count?: number;
		subscriptions?: Record<string, string[]>;
		max_messages?: number;
		max_depth?: number;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2 text-xs">
				{seed_topics && <span>{seed_topics.length} topics</span>}
				{subscriber_count != null && <span>{subscriber_count} subscribers</span>}
			</div>
			{seed_topics && seed_topics.length > 0 && (
				<div className="flex flex-wrap gap-1">
					{seed_topics.map((t) => (
						<span key={t} className="text-[10px] px-1.5 py-0.5 rounded bg-info-muted text-info font-mono">
							{t}
						</span>
					))}
				</div>
			)}
			<div className="flex items-center gap-4 text-xs">
				{max_messages != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Max messages</span>
						<span className="font-mono tabular-nums">{max_messages}</span>
					</div>
				)}
				{max_depth != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Max depth</span>
						<span className="font-mono tabular-nums">{max_depth}</span>
					</div>
				)}
			</div>
			{subscriptions && Object.keys(subscriptions).length > 0 && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Subscriptions</span>
					<PayloadViewer payload={subscriptions} />
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function messageBusStartSummary(event: TraceEvent): string {
	const { seed_topics, subscriber_count } = event.payload as {
		seed_topics?: string[];
		subscriber_count?: number;
	};
	return `Message bus started (${seed_topics?.length ?? 0} topics, ${subscriber_count ?? 0} subscribers)`;
}

function MessagePublishedRenderer({ event }: EventDetailProps) {
	const { message_id, topic, author, content, depth, parent_message_id } = event.payload as {
		message_id?: string;
		topic?: string;
		author?: string;
		content?: string;
		depth?: number;
		parent_message_id?: string | null;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{topic && <span className="text-xs px-1.5 py-0.5 rounded bg-info-muted text-info font-mono">{topic}</span>}
				{author && <span className="text-xs font-mono">{author}</span>}
				{depth != null && <span className="text-xs text-muted-foreground">depth {depth}</span>}
			</div>
			{message_id && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Message ID</span>
					<span className="font-mono">{message_id}</span>
				</div>
			)}
			{parent_message_id && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Parent</span>
					<span className="font-mono">{parent_message_id}</span>
				</div>
			)}
			{content && (
				<div className="text-sm text-foreground bg-muted/50 rounded-md p-2 whitespace-pre-wrap max-h-60 overflow-auto">
					{content}
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function messagePublishedSummary(event: TraceEvent): string {
	const { topic, author } = event.payload as {
		topic?: string;
		author?: string;
	};
	return `Published to ${topic ?? "unknown"} by ${author ?? "unknown"}`;
}

function MessageDeliveredRenderer({ event }: EventDetailProps) {
	const { message_id, topic, agent_name, output, steps, messages_published, error } = event.payload as {
		message_id?: string;
		topic?: string;
		agent_name?: string;
		output?: string;
		steps?: number;
		messages_published?: number;
		error?: string | null;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{topic && <span className="text-xs px-1.5 py-0.5 rounded bg-info-muted text-info font-mono">{topic}</span>}
				{agent_name && <span className="text-xs font-mono">{agent_name}</span>}
				{error && <span className="text-xs px-1.5 py-0.5 rounded bg-destructive-muted text-destructive">error</span>}
			</div>
			<div className="flex items-center gap-4 text-xs">
				{steps != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Steps</span>
						<span className="font-mono tabular-nums">{steps}</span>
					</div>
				)}
				{messages_published != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Published</span>
						<span className="font-mono tabular-nums">{messages_published}</span>
					</div>
				)}
			</div>
			{message_id && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Message ID</span>
					<span className="font-mono">{message_id}</span>
				</div>
			)}
			{output && (
				<div className="text-sm text-foreground bg-muted/50 rounded-md p-2 whitespace-pre-wrap max-h-60 overflow-auto">
					{output}
				</div>
			)}
			{error && (
				<div className="text-sm text-destructive-muted-foreground bg-destructive-muted rounded-md p-2">{error}</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function messageDeliveredSummary(event: TraceEvent): string {
	const { topic, agent_name } = event.payload as {
		topic?: string;
		agent_name?: string;
	};
	return `Delivered ${topic ?? "unknown"} to ${agent_name ?? "unknown"}`;
}

function MessageBusCompleteRenderer({ event }: EventDetailProps) {
	const { total_messages, total_executions, max_depth_reached, termination_reason, agent_execution_counts } =
		event.payload as {
			total_messages?: number;
			total_executions?: number;
			max_depth_reached?: number;
			termination_reason?: string;
			agent_execution_counts?: Record<string, number>;
		};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				<span className="text-xs px-1.5 py-0.5 rounded bg-success-muted text-success">complete</span>
			</div>
			<div className="flex items-center gap-4 text-xs">
				{total_messages != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Messages</span>
						<span className="font-mono tabular-nums">{total_messages}</span>
					</div>
				)}
				{total_executions != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Executions</span>
						<span className="font-mono tabular-nums">{total_executions}</span>
					</div>
				)}
				{max_depth_reached != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Max depth</span>
						<span className="font-mono tabular-nums">{max_depth_reached}</span>
					</div>
				)}
			</div>
			{termination_reason && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Termination</span>
					<span>{termination_reason}</span>
				</div>
			)}
			{agent_execution_counts && Object.keys(agent_execution_counts).length > 0 && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Per-agent executions</span>
					{Object.entries(agent_execution_counts).map(([agent, count]) => (
						<div key={agent} className="flex items-center gap-2 text-xs">
							<span className="font-mono">{agent}</span>
							<span className="font-mono tabular-nums">{count}</span>
						</div>
					))}
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function messageBusCompleteSummary(event: TraceEvent): string {
	const { total_messages, max_depth_reached } = event.payload as {
		total_messages?: number;
		max_depth_reached?: number;
	};
	return `Bus complete (${total_messages ?? 0} messages, depth ${max_depth_reached ?? 0})`;
}

// ---------------------------------------------------------------------------
// Durability Renderers
// ---------------------------------------------------------------------------

function CheckpointSavedRenderer({ event }: EventDetailProps) {
	const { checkpoint_id, checkpoint_type, run_id } = event.payload as {
		checkpoint_id?: string;
		checkpoint_type?: string;
		run_id?: string;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{checkpoint_type && (
					<span className="text-xs px-1.5 py-0.5 rounded bg-info-muted text-info">{checkpoint_type}</span>
				)}
			</div>
			{checkpoint_id && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Checkpoint</span>
					<span className="font-mono text-[10px]">{checkpoint_id}</span>
				</div>
			)}
			{run_id && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Run</span>
					<span className="font-mono text-[10px]">{run_id}</span>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function checkpointSavedSummary(event: TraceEvent): string {
	const { checkpoint_type } = event.payload as { checkpoint_type?: string };
	return `Checkpoint saved (${checkpoint_type ?? "unknown"})`;
}

function ExecutionSuspendedRenderer({ event }: EventDetailProps) {
	const { suspension_id, suspension_type, checkpoint_id, step_name, agent_name } = event.payload as {
		suspension_id?: string;
		suspension_type?: string;
		checkpoint_id?: string;
		step_name?: string | null;
		agent_name?: string | null;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{suspension_type && (
					<span className="text-xs px-1.5 py-0.5 rounded bg-warning-muted text-warning">{suspension_type}</span>
				)}
			</div>
			{step_name && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Step</span>
					<span className="font-mono">{step_name}</span>
				</div>
			)}
			{agent_name && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Agent</span>
					<span className="font-mono">{agent_name}</span>
				</div>
			)}
			{suspension_id && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Suspension</span>
					<span className="font-mono text-[10px]">{suspension_id}</span>
				</div>
			)}
			{checkpoint_id && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Checkpoint</span>
					<span className="font-mono text-[10px]">{checkpoint_id}</span>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function executionSuspendedSummary(event: TraceEvent): string {
	const { step_name, suspension_type } = event.payload as {
		step_name?: string | null;
		suspension_type?: string;
	};
	if (step_name) return `Suspended at '${step_name}' (${suspension_type ?? "unknown"})`;
	return `Execution suspended (${suspension_type ?? "unknown"})`;
}

function ExecutionResumedRenderer({ event }: EventDetailProps) {
	const { checkpoint_id, suspension_id, resumed_from_step } = event.payload as {
		checkpoint_id?: string;
		suspension_id?: string;
		resumed_from_step?: string | null;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				<span className="text-xs px-1.5 py-0.5 rounded bg-success-muted text-success">Resumed</span>
			</div>
			{resumed_from_step && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">From step</span>
					<span className="font-mono">{resumed_from_step}</span>
				</div>
			)}
			{checkpoint_id && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Checkpoint</span>
					<span className="font-mono text-[10px]">{checkpoint_id}</span>
				</div>
			)}
			{suspension_id && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Suspension</span>
					<span className="font-mono text-[10px]">{suspension_id}</span>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function executionResumedSummary(event: TraceEvent): string {
	const { resumed_from_step } = event.payload as { resumed_from_step?: string | null };
	if (resumed_from_step) return `Resumed from '${resumed_from_step}'`;
	return "Execution resumed";
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/** Creates all workflow and multi-agent event renderer registrations. */
export function createWorkflowRegistrations(): EventRendererRegistration[] {
	return [
		// Workflow
		{
			matches: (t) => t === "workflow.start",
			priority: 0,
			component: WorkflowStartRenderer,
			summary: workflowStartSummary,
		},
		{
			matches: (t) => t === "workflow.structure",
			priority: 0,
			component: WorkflowStructureRenderer,
			summary: workflowStructureSummary,
		},
		{
			matches: (t) => t === "workflow.step.complete",
			priority: 0,
			component: WorkflowStepCompleteRenderer,
			summary: workflowStepCompleteSummary,
		},
		{
			matches: (t) => t === "workflow.complete",
			priority: 0,
			component: WorkflowCompleteRenderer,
			summary: workflowCompleteSummary,
		},
		{
			matches: (t) => t === "workflow.error",
			priority: 0,
			component: WorkflowErrorRenderer,
			summary: workflowErrorSummary,
		},
		// Durability
		{
			matches: (t) => t === "checkpoint.saved",
			priority: 0,
			component: CheckpointSavedRenderer,
			summary: checkpointSavedSummary,
		},
		{
			matches: (t) => t === "execution.suspended",
			priority: 0,
			component: ExecutionSuspendedRenderer,
			summary: executionSuspendedSummary,
		},
		{
			matches: (t) => t === "execution.resumed",
			priority: 0,
			component: ExecutionResumedRenderer,
			summary: executionResumedSummary,
		},
		// Delegation / Handoff / Supervision
		{
			matches: (t) => t === "multi_agent.delegation",
			priority: 0,
			component: DelegationRenderer,
			summary: delegationSummary,
		},
		{ matches: (t) => t === "multi_agent.handoff", priority: 0, component: HandoffRenderer, summary: handoffSummary },
		{
			matches: (t) => t === "multi_agent.supervision",
			priority: 0,
			component: SupervisionRenderer,
			summary: supervisionSummary,
		},
		// Bidding
		{
			matches: (t) => t === "multi_agent.bidding.start",
			priority: 0,
			component: BiddingStartRenderer,
			summary: biddingStartSummary,
		},
		{
			matches: (t) => t === "multi_agent.bidding.bid",
			priority: 0,
			component: BidReceivedRenderer,
			summary: bidReceivedSummary,
		},
		{
			matches: (t) => t === "multi_agent.bidding.allocated",
			priority: 0,
			component: BidAllocatedRenderer,
			summary: bidAllocatedSummary,
		},
		{
			matches: (t) => t === "multi_agent.bidding.complete",
			priority: 0,
			component: BiddingCompleteRenderer,
			summary: biddingCompleteSummary,
		},
		// Debate
		{
			matches: (t) => t === "multi_agent.debate.start",
			priority: 0,
			component: DebateStartRenderer,
			summary: debateStartSummary,
		},
		{
			matches: (t) => t === "multi_agent.debate.argument",
			priority: 0,
			component: DebateArgumentRenderer,
			summary: debateArgumentSummary,
		},
		{
			matches: (t) => t === "multi_agent.debate.resolution",
			priority: 0,
			component: DebateResolutionRenderer,
			summary: debateResolutionSummary,
		},
		{
			matches: (t) => t === "multi_agent.debate.complete",
			priority: 0,
			component: DebateCompleteRenderer,
			summary: debateCompleteSummary,
		},
		// Consensus
		{
			matches: (t) => t === "multi_agent.consensus.start",
			priority: 0,
			component: ConsensusStartRenderer,
			summary: consensusStartSummary,
		},
		{
			matches: (t) => t === "multi_agent.consensus.vote",
			priority: 0,
			component: ConsensusVoteRenderer,
			summary: consensusVoteSummary,
		},
		{
			matches: (t) => t === "multi_agent.consensus.agreement",
			priority: 0,
			component: ConsensusAgreementRenderer,
			summary: consensusAgreementSummary,
		},
		{
			matches: (t) => t === "multi_agent.consensus.complete",
			priority: 0,
			component: ConsensusCompleteRenderer,
			summary: consensusCompleteSummary,
		},
		// Broadcast
		{
			matches: (t) => t === "multi_agent.broadcast.start",
			priority: 0,
			component: BroadcastStartRenderer,
			summary: broadcastStartSummary,
		},
		{
			matches: (t) => t === "multi_agent.broadcast.response",
			priority: 0,
			component: BroadcastResponseRenderer,
			summary: broadcastResponseSummary,
		},
		{
			matches: (t) => t === "multi_agent.broadcast.complete",
			priority: 0,
			component: BroadcastCompleteRenderer,
			summary: broadcastCompleteSummary,
		},
		// Blackboard
		{
			matches: (t) => t === "blackboard.start",
			priority: 0,
			component: BlackboardStartRenderer,
			summary: blackboardStartSummary,
		},
		{
			matches: (t) => t === "blackboard.round",
			priority: 0,
			component: BlackboardRoundRenderer,
			summary: blackboardRoundSummary,
		},
		{
			matches: (t) => t === "blackboard.complete",
			priority: 0,
			component: BlackboardCompleteRenderer,
			summary: blackboardCompleteSummary,
		},
		// Peer Network
		{
			matches: (t) => t === "multi_agent.peer.start",
			priority: 0,
			component: PeerNetworkStartRenderer,
			summary: peerNetworkStartSummary,
		},
		{
			matches: (t) => t === "multi_agent.peer.consultation",
			priority: 0,
			component: PeerConsultationRenderer,
			summary: peerConsultationSummary,
		},
		{
			matches: (t) => t === "multi_agent.peer.complete",
			priority: 0,
			component: PeerNetworkCompleteRenderer,
			summary: peerNetworkCompleteSummary,
		},
		// Message Bus
		{
			matches: (t) => t === "multi_agent.bus.start",
			priority: 0,
			component: MessageBusStartRenderer,
			summary: messageBusStartSummary,
		},
		{
			matches: (t) => t === "multi_agent.bus.published",
			priority: 0,
			component: MessagePublishedRenderer,
			summary: messagePublishedSummary,
		},
		{
			matches: (t) => t === "multi_agent.bus.delivered",
			priority: 0,
			component: MessageDeliveredRenderer,
			summary: messageDeliveredSummary,
		},
		{
			matches: (t) => t === "multi_agent.bus.complete",
			priority: 0,
			component: MessageBusCompleteRenderer,
			summary: messageBusCompleteSummary,
		},
	];
}
