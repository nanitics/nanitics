import { useState } from "react";
import type { AgentInfo, TraceEvent } from "../../types";
import { AgentLink } from "./delegation-panel";

interface BiddingPanelProps {
	events: TraceEvent[];
	agents: AgentInfo[];
	onNavigateToAgent: (spanId: string) => void;
}

interface BidRow {
	agentName: string;
	confidence: number;
	reasoning: string;
	estimatedCost: number | null;
}

export function BiddingPanel({ events, agents, onNavigateToAgent }: BiddingPanelProps) {
	const agentMap = new Map(agents.map((a) => [a.agent_name, a.span_id]));

	const startEvent = events.find((e) => e.event_type === "multi_agent.bidding.start");
	const bidEvents = events.filter((e) => e.event_type === "multi_agent.bidding.bid");
	const allocatedEvent = events.find((e) => e.event_type === "multi_agent.bidding.allocated");
	const task = startEvent ? ((startEvent.payload as Record<string, unknown>).task as string | undefined) : undefined;

	const allocPayload = allocatedEvent?.payload as Record<string, unknown> | undefined;
	const winner = allocPayload?.winner as string | null | undefined;
	const rejectionReason = allocPayload?.rejection_reason as string | null | undefined;

	const bids: BidRow[] = bidEvents.map((e) => {
		const p = e.payload as Record<string, unknown>;
		return {
			agentName: (p.agent_name as string) ?? "unknown",
			confidence: (p.confidence as number) ?? 0,
			reasoning: (p.reasoning as string) ?? "",
			estimatedCost: (p.estimated_cost as number | null) ?? null,
		};
	});

	// Sort by confidence descending
	bids.sort((a, b) => b.confidence - a.confidence);

	return (
		<div className="space-y-3">
			{task && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Task</span>
					<div className="text-xs text-foreground">{task}</div>
				</div>
			)}

			{bids.length > 0 && (
				<div className="border border-border rounded-md overflow-hidden">
					<table className="w-full text-xs">
						<thead>
							<tr className="bg-muted/50">
								<th className="text-left px-2 py-1 font-medium text-muted-foreground">Agent</th>
								<th className="text-left px-2 py-1 font-medium text-muted-foreground">Confidence</th>
								<th className="text-left px-2 py-1 font-medium text-muted-foreground">Reasoning</th>
								<th className="text-right px-2 py-1 font-medium text-muted-foreground">Cost</th>
							</tr>
						</thead>
						<tbody>
							{bids.map((bid) => (
								<BidTableRow
									key={bid.agentName}
									bid={bid}
									isWinner={bid.agentName === winner}
									agentMap={agentMap}
									onNavigateToAgent={onNavigateToAgent}
								/>
							))}
						</tbody>
					</table>
				</div>
			)}

			{/* Winner / rejection summary */}
			<div className="text-xs">
				{winner ? (
					<span>
						Winner: <AgentLink name={winner} agentMap={agentMap} onNavigateToAgent={onNavigateToAgent} />
					</span>
				) : rejectionReason ? (
					<span className="text-warning">{rejectionReason}</span>
				) : allocatedEvent ? (
					<span className="text-muted-foreground">No winner</span>
				) : null}
			</div>
		</div>
	);
}

function BidTableRow({
	bid,
	isWinner,
	agentMap,
	onNavigateToAgent,
}: {
	bid: BidRow;
	isWinner: boolean;
	agentMap: Map<string, string>;
	onNavigateToAgent: (spanId: string) => void;
}) {
	const [expanded, setExpanded] = useState(false);
	const pct = Math.round(bid.confidence * 100);
	const TRUNCATE = 60;
	const isLong = bid.reasoning.length > TRUNCATE;

	return (
		<tr
			className={`border-t border-border ${isWinner ? "bg-success-muted/30" : ""}`}
			data-testid={isWinner ? "winner-row" : undefined}
		>
			<td className="px-2 py-1">
				<AgentLink name={bid.agentName} agentMap={agentMap} onNavigateToAgent={onNavigateToAgent} />
			</td>
			<td className="px-2 py-1">
				<div className="flex items-center gap-1.5">
					<div className="w-16 bg-muted rounded-full h-1.5">
						<div className="bg-primary rounded-full h-1.5" style={{ width: `${pct}%` }} data-testid="confidence-bar" />
					</div>
					<span className="font-mono tabular-nums">{pct}%</span>
				</div>
			</td>
			<td className="px-2 py-1">
				{expanded || !isLong ? bid.reasoning : `${bid.reasoning.slice(0, TRUNCATE)}…`}
				{isLong && (
					<button type="button" onClick={() => setExpanded(!expanded)} className="ml-1 text-primary hover:underline">
						{expanded ? "less" : "more"}
					</button>
				)}
			</td>
			<td className="px-2 py-1 text-right font-mono tabular-nums">
				{bid.estimatedCost != null ? `$${bid.estimatedCost}` : "—"}
			</td>
		</tr>
	);
}
