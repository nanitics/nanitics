import type { AgentInfo, TraceEvent } from "../../types";
import { AgentLink } from "./delegation-panel";

interface BroadcastPanelProps {
	events: TraceEvent[];
	agents: AgentInfo[];
	onNavigateToAgent: (spanId: string) => void;
}

export function BroadcastPanel({ events, agents, onNavigateToAgent }: BroadcastPanelProps) {
	const agentMap = new Map(agents.map((a) => [a.agent_name, a.span_id]));

	const startEvent = events.find((e) => e.event_type === "multi_agent.broadcast.start");
	const responseEvents = events.filter((e) => e.event_type === "multi_agent.broadcast.response");
	const completeEvent = events.find((e) => e.event_type === "multi_agent.broadcast.complete");

	const startPayload = startEvent?.payload as Record<string, unknown> | undefined;
	const task = startPayload?.task as string | undefined;
	const responseStrategy = startPayload?.response_strategy as string | undefined;

	const completePayload = completeEvent?.payload as Record<string, unknown> | undefined;
	const totalAgents = completePayload?.total_agents as number | undefined;
	const responsesCollected = completePayload?.responses_collected as number | undefined;
	const aggregatedOutput = completePayload?.aggregated_output as string | undefined;

	return (
		<div className="space-y-3">
			{/* Header */}
			<div className="flex items-center gap-2">
				{responseStrategy && (
					<span className="text-[10px] px-1.5 py-0.5 rounded bg-info-muted text-info">{responseStrategy}</span>
				)}
				{responsesCollected != null && totalAgents != null && (
					<span className="text-xs text-muted-foreground">
						{responsesCollected}/{totalAgents} responses
					</span>
				)}
			</div>

			{/* Task */}
			{task && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Task</span>
					<div className="text-xs text-foreground">{task}</div>
				</div>
			)}

			{/* Response table */}
			{responseEvents.length > 0 && (
				<div className="border border-border rounded-md overflow-hidden">
					<table className="w-full text-xs">
						<thead>
							<tr className="bg-muted/50">
								<th className="text-left px-2 py-1 font-medium text-muted-foreground">Agent</th>
								<th className="text-left px-2 py-1 font-medium text-muted-foreground">Output</th>
								<th className="text-right px-2 py-1 font-medium text-muted-foreground">Steps</th>
								<th className="text-center px-2 py-1 font-medium text-muted-foreground">Error</th>
							</tr>
						</thead>
						<tbody>
							{responseEvents.map((event) => {
								const p = event.payload as Record<string, unknown>;
								const agentName = p.agent_name as string | undefined;
								const outputPreview = p.output as string | undefined;
								const steps = p.steps as number | undefined;
								const error = p.error as string | null | undefined;

								return (
									<tr key={event.id} className="border-t border-border">
										<td className="px-2 py-1">
											<AgentLink name={agentName} agentMap={agentMap} onNavigateToAgent={onNavigateToAgent} />
										</td>
										<td className="px-2 py-1 max-w-[200px] truncate">{outputPreview ?? "—"}</td>
										<td className="px-2 py-1 text-right font-mono tabular-nums">{steps ?? "—"}</td>
										<td className="px-2 py-1 text-center">
											{error ? (
												<span className="text-destructive" title={error}>
													✗
												</span>
											) : (
												""
											)}
										</td>
									</tr>
								);
							})}
						</tbody>
					</table>
				</div>
			)}

			{/* Aggregated output */}
			{aggregatedOutput && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Aggregated output</span>
					<div className="text-xs text-foreground bg-muted/50 rounded-md p-2">{aggregatedOutput}</div>
				</div>
			)}
		</div>
	);
}
