import type { AgentInfo, TraceEvent } from "../../types";
import { AgentLink } from "./delegation-panel";

interface HandoffPanelProps {
	events: TraceEvent[];
	agents: AgentInfo[];
	onNavigateToAgent: (spanId: string) => void;
}

export function HandoffPanel({ events, agents, onNavigateToAgent }: HandoffPanelProps) {
	const agentMap = new Map(agents.map((a) => [a.agent_name, a.span_id]));

	// Build chain names from events (already in order from detection)
	const chainNames: string[] = [];
	for (const event of events) {
		const p = event.payload as Record<string, unknown>;
		const from = p.from_agent as string;
		const to = p.to_agent as string;
		if (chainNames.length === 0) chainNames.push(from);
		chainNames.push(to);
	}

	return (
		<div className="space-y-3">
			{/* Chain diagram */}
			<div className="flex items-center gap-1.5 flex-wrap">
				{chainNames.map((name, i) => (
					// biome-ignore lint/suspicious/noArrayIndexKey: agent names can repeat in handoff chains
					<span key={`${name}-${i}`} className="flex items-center gap-1.5">
						{i > 0 && <span className="text-muted-foreground text-xs">→</span>}
						<AgentLink name={name} agentMap={agentMap} onNavigateToAgent={onNavigateToAgent} />
					</span>
				))}
			</div>

			{/* Per-transfer details */}
			<div className="space-y-1.5">
				<span className="text-xs text-muted-foreground">Transfers</span>
				{events.map((event, i) => {
					const p = event.payload as Record<string, unknown>;
					const from = p.from_agent as string;
					const to = p.to_agent as string;
					const fields = p.payload_fields as string[] | undefined;
					const size = p.payload_size as number | undefined;

					return (
						<div key={event.id} className="flex items-center gap-2 text-xs">
							<span className="text-muted-foreground font-mono">{i + 1}.</span>
							<span className="font-mono">{from}</span>
							<span className="text-muted-foreground">→</span>
							<span className="font-mono">{to}</span>
							{fields && fields.length > 0 && (
								<div className="flex gap-1">
									{fields.map((f) => (
										<span key={f} className="text-[10px] px-1 py-0.5 rounded bg-muted text-muted-foreground font-mono">
											{f}
										</span>
									))}
								</div>
							)}
							{size != null && (
								<span className="text-muted-foreground">
									{size >= 1024 ? `${(size / 1024).toFixed(1)} KB` : `${size} B`}
								</span>
							)}
						</div>
					);
				})}
			</div>
		</div>
	);
}
