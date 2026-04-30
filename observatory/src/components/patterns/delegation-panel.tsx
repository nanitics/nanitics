import { useState } from "react";
import type { AgentInfo, TraceEvent } from "../../types";

interface DelegationPanelProps {
	events: TraceEvent[];
	agents: AgentInfo[];
	onNavigateToAgent: (spanId: string) => void;
}

export function DelegationPanel({ events, agents, onNavigateToAgent }: DelegationPanelProps) {
	const agentMap = new Map(agents.map((a) => [a.agent_name, a.span_id]));

	return (
		<div className="space-y-2">
			{events.map((event) => (
				<DelegationCard key={event.id} event={event} agentMap={agentMap} onNavigateToAgent={onNavigateToAgent} />
			))}
		</div>
	);
}

function DelegationCard({
	event,
	agentMap,
	onNavigateToAgent,
}: {
	event: TraceEvent;
	agentMap: Map<string, string>;
	onNavigateToAgent: (spanId: string) => void;
}) {
	const [taskExpanded, setTaskExpanded] = useState(false);
	const p = event.payload as Record<string, unknown>;
	const caller = p.caller_agent as string | undefined;
	const delegate = p.delegate_agent as string | undefined;
	const task = p.task as string | undefined;
	const strategy = p.transfer_strategy as string | undefined;

	const TRUNCATE_LENGTH = 120;
	const isLongTask = task != null && task.length > TRUNCATE_LENGTH;

	return (
		<div className="space-y-2">
			<div className="flex items-center gap-2 text-sm">
				<AgentLink name={caller} agentMap={agentMap} onNavigateToAgent={onNavigateToAgent} />
				<span className="text-muted-foreground">→</span>
				<AgentLink name={delegate} agentMap={agentMap} onNavigateToAgent={onNavigateToAgent} />
			</div>

			{strategy && <span className="text-[10px] px-1.5 py-0.5 rounded bg-info-muted text-info">{strategy}</span>}

			{task && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Task</span>
					<div className="text-xs text-foreground">
						{taskExpanded || !isLongTask ? task : `${task.slice(0, TRUNCATE_LENGTH)}…`}
						{isLongTask && (
							<button
								type="button"
								onClick={() => setTaskExpanded(!taskExpanded)}
								className="ml-1 text-primary hover:underline"
							>
								{taskExpanded ? "less" : "more"}
							</button>
						)}
					</div>
				</div>
			)}
		</div>
	);
}

export function AgentLink({
	name,
	agentMap,
	onNavigateToAgent,
}: {
	name: string | undefined;
	agentMap: Map<string, string>;
	onNavigateToAgent: (spanId: string) => void;
}) {
	if (!name) return <span className="font-mono text-muted-foreground">?</span>;

	const spanId = agentMap.get(name);
	if (!spanId) {
		return <span className="font-mono">{name}</span>;
	}

	return (
		<button
			type="button"
			onClick={() => onNavigateToAgent(spanId)}
			className="font-mono text-primary hover:underline cursor-pointer"
		>
			{name}
		</button>
	);
}
