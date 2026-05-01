import type { AgentInfo, TraceEvent } from "../../types";
import { AgentLink } from "./delegation-panel";

const ACTION_COLORS: Record<string, string> = {
	retry: "bg-warning-muted text-warning",
	reassign: "bg-info-muted text-info",
	escalate: "bg-destructive-muted text-destructive",
	override: "bg-destructive-muted text-destructive",
};

interface SupervisionPanelProps {
	events: TraceEvent[];
	agents: AgentInfo[];
	onNavigateToAgent: (spanId: string) => void;
}

export function SupervisionPanel({ events, agents, onNavigateToAgent }: SupervisionPanelProps) {
	const agentMap = new Map(agents.map((a) => [a.agent_name, a.span_id]));

	// Get the supervised agent from the first event
	const firstPayload = events[0]?.payload as Record<string, unknown> | undefined;
	const supervisedAgent = firstPayload?.supervised_agent as string | undefined;

	// Sort events by attempt number
	const sorted = [...events].sort((a, b) => {
		const aAttempt = (a.payload as Record<string, unknown>).attempt as number | undefined;
		const bAttempt = (b.payload as Record<string, unknown>).attempt as number | undefined;
		return (aAttempt ?? 0) - (bAttempt ?? 0);
	});

	// Determine final outcome
	const lastEvent = sorted[sorted.length - 1];
	const lastPayload = lastEvent?.payload as Record<string, unknown> | undefined;
	const lastAction = lastPayload?.action as string | undefined;
	const lastReassigned = lastPayload?.reassigned_to as string | undefined;

	return (
		<div className="space-y-3">
			{/* Supervised agent name */}
			{supervisedAgent && (
				<div className="flex items-center gap-2 text-sm">
					<span className="text-xs text-muted-foreground">Supervised:</span>
					<AgentLink name={supervisedAgent} agentMap={agentMap} onNavigateToAgent={onNavigateToAgent} />
				</div>
			)}

			{/* Intervention timeline */}
			<div className="space-y-2">
				{sorted.map((event) => {
					const p = event.payload as Record<string, unknown>;
					const attempt = p.attempt as number | undefined;
					const triggerName = p.trigger_name as string | undefined;
					const action = p.action as string | undefined;
					const feedback = p.feedback as string | null | undefined;
					const reassignedTo = p.reassigned_to as string | null | undefined;

					return (
						<div key={event.id} className="border border-border rounded-md p-2 space-y-1.5">
							<div className="flex items-center gap-2 text-xs">
								{attempt != null && <span className="text-muted-foreground font-mono">Attempt {attempt}</span>}
								{triggerName && (
									<span className="px-1.5 py-0.5 rounded bg-muted text-muted-foreground">{triggerName}</span>
								)}
								{action && (
									<span
										className={`px-1.5 py-0.5 rounded ${ACTION_COLORS[action] ?? "bg-muted text-muted-foreground"}`}
									>
										{action}
									</span>
								)}
							</div>
							{feedback && <div className="text-xs text-foreground">{feedback}</div>}
							{reassignedTo && (
								<div className="flex items-center gap-1.5 text-xs">
									<span className="text-muted-foreground">→</span>
									<AgentLink name={reassignedTo} agentMap={agentMap} onNavigateToAgent={onNavigateToAgent} />
								</div>
							)}
						</div>
					);
				})}
			</div>

			{/* Summary */}
			<div className="text-xs text-muted-foreground">
				{events.length} attempt{events.length !== 1 ? "s" : ""}
				{lastAction === "reassign" && lastReassigned
					? `, reassigned to ${lastReassigned}`
					: lastAction
						? `, final action: ${lastAction}`
						: ""}
			</div>
		</div>
	);
}
