import { useState } from "react";
import type { AgentInfo } from "../../types";
import type { DetectedPattern } from "../../utils/pattern-detector";
import { BiddingPanel } from "./bidding-panel";
import { BlackboardPanel } from "./blackboard-panel";
import { BroadcastPanel } from "./broadcast-panel";
import { ConsensusPanel } from "./consensus-panel";
import { DebatePanel } from "./debate-panel";
import { DelegationPanel } from "./delegation-panel";
import { HandoffPanel } from "./handoff-panel";
import { MessageBusPanel } from "./message-bus-panel";
import { PeerNetworkPanel } from "./peer-network-panel";
import { SupervisionPanel } from "./supervision-panel";

const PATTERN_LABELS: Record<string, string> = {
	delegation: "Delegation",
	broadcast: "Broadcast",
	bidding: "Bidding",
	supervision: "Supervision",
	handoff: "Handoff",
	debate: "Debate",
	consensus: "Consensus",
	blackboard: "Blackboard",
	peer_network: "Peer Network",
	message_bus: "Message Bus",
};

const PATTERN_COLORS: Record<string, string> = {
	delegation: "bg-info-muted text-info",
	broadcast: "bg-primary/10 text-primary",
	bidding: "bg-warning-muted text-warning",
	supervision: "bg-destructive-muted text-destructive",
	handoff: "bg-success-muted text-success",
	debate: "bg-orange-100 text-orange-700 dark:bg-orange-950 dark:text-orange-300",
	consensus: "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300",
	blackboard: "bg-violet-100 text-violet-700 dark:bg-violet-950 dark:text-violet-300",
	peer_network: "bg-cyan-100 text-cyan-700 dark:bg-cyan-950 dark:text-cyan-300",
	message_bus: "bg-rose-100 text-rose-700 dark:bg-rose-950 dark:text-rose-300",
};

interface PatternSummaryProps {
	patterns: DetectedPattern[];
	agents: AgentInfo[];
	onNavigateToAgent: (spanId: string) => void;
}

export function PatternSummary({ patterns, agents, onNavigateToAgent }: PatternSummaryProps) {
	const [expanded, setExpanded] = useState(false);

	if (patterns.length === 0) return null;

	const uniqueTypes = [...new Set(patterns.map((p) => p.type))];

	return (
		<div className="border-b">
			{/* Collapsed header */}
			<button
				type="button"
				onClick={() => setExpanded(!expanded)}
				className="w-full px-4 py-2 flex items-center gap-3 text-left hover:bg-muted/50 transition-colors"
			>
				<span className="text-xs text-muted-foreground">{expanded ? "▾" : "▸"}</span>
				<span className="text-xs font-medium">
					{patterns.length} pattern{patterns.length !== 1 ? "s" : ""} detected
				</span>
				<div className="flex items-center gap-1.5">
					{uniqueTypes.map((type) => (
						<span
							key={type}
							className={`text-[10px] px-1.5 py-0.5 rounded ${PATTERN_COLORS[type] ?? "bg-muted text-muted-foreground"}`}
						>
							{PATTERN_LABELS[type] ?? type}
						</span>
					))}
				</div>
			</button>

			{/* Expanded content */}
			{expanded && (
				<div className="px-4 pb-3 space-y-3">
					{patterns.map((pattern) => (
						<PatternCard
							key={`${pattern.type}-${pattern.spanId}`}
							pattern={pattern}
							agents={agents}
							onNavigateToAgent={onNavigateToAgent}
						/>
					))}
				</div>
			)}
		</div>
	);
}

function PatternCard({
	pattern,
	agents,
	onNavigateToAgent,
}: {
	pattern: DetectedPattern;
	agents: AgentInfo[];
	onNavigateToAgent: (spanId: string) => void;
}) {
	return (
		<div className="border border-border rounded-md p-3">
			<div className="flex items-center gap-2 mb-2">
				<span
					className={`text-[10px] px-1.5 py-0.5 rounded ${PATTERN_COLORS[pattern.type] ?? "bg-muted text-muted-foreground"}`}
				>
					{PATTERN_LABELS[pattern.type] ?? pattern.type}
				</span>
				<span className="text-xs text-muted-foreground">{pattern.label}</span>
			</div>
			<PatternPanelContent pattern={pattern} agents={agents} onNavigateToAgent={onNavigateToAgent} />
		</div>
	);
}

function PatternPanelContent({
	pattern,
	agents,
	onNavigateToAgent,
}: {
	pattern: DetectedPattern;
	agents: AgentInfo[];
	onNavigateToAgent: (spanId: string) => void;
}) {
	switch (pattern.type) {
		case "delegation":
			return <DelegationPanel events={pattern.events} agents={agents} onNavigateToAgent={onNavigateToAgent} />;
		case "handoff":
			return <HandoffPanel events={pattern.events} agents={agents} onNavigateToAgent={onNavigateToAgent} />;
		case "bidding":
			return <BiddingPanel events={pattern.events} agents={agents} onNavigateToAgent={onNavigateToAgent} />;
		case "broadcast":
			return <BroadcastPanel events={pattern.events} agents={agents} onNavigateToAgent={onNavigateToAgent} />;
		case "supervision":
			return <SupervisionPanel events={pattern.events} agents={agents} onNavigateToAgent={onNavigateToAgent} />;
		case "debate":
			return <DebatePanel events={pattern.events} agents={agents} onNavigateToAgent={onNavigateToAgent} />;
		case "consensus":
			return <ConsensusPanel events={pattern.events} agents={agents} onNavigateToAgent={onNavigateToAgent} />;
		case "blackboard":
			return <BlackboardPanel events={pattern.events} agents={agents} onNavigateToAgent={onNavigateToAgent} />;
		case "peer_network":
			return <PeerNetworkPanel events={pattern.events} agents={agents} onNavigateToAgent={onNavigateToAgent} />;
		case "message_bus":
			return <MessageBusPanel events={pattern.events} agents={agents} onNavigateToAgent={onNavigateToAgent} />;
		default:
			return (
				<div className="text-xs text-muted-foreground">
					{pattern.events.length} event{pattern.events.length !== 1 ? "s" : ""}
				</div>
			);
	}
}
