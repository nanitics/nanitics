import { useState } from "react";
import type { AgentInfo, TraceEvent } from "../../types";
import { AgentLink } from "./delegation-panel";

interface BlackboardPanelProps {
	events: TraceEvent[];
	agents: AgentInfo[];
	onNavigateToAgent: (spanId: string) => void;
}

interface RoundEntry {
	operation: "write" | "supersede" | "retract";
	author: string;
	content: string;
	scope: string | null;
	entry_id: string;
	original_entry_id: string | null;
	retract_reason: string | null;
}

interface RoundRow {
	roundNumber: number;
	agentsActivated: string[];
	contributions: number;
	totalContributions: number;
	entries: RoundEntry[];
}

export function BlackboardPanel({ events, agents, onNavigateToAgent }: BlackboardPanelProps) {
	const agentMap = new Map(agents.map((a) => [a.agent_name, a.span_id]));

	const startEvent = events.find((e) => e.event_type === "blackboard.start");
	const roundEvents = events.filter((e) => e.event_type === "blackboard.round");
	const completeEvent = events.find((e) => e.event_type === "blackboard.complete");

	const sp = startEvent?.payload as Record<string, unknown> | undefined;
	const controlStrategy = sp?.control_strategy as string | undefined;

	const cp = completeEvent?.payload as Record<string, unknown> | undefined;
	const roundsCompleted = cp?.rounds_completed as number | undefined;
	const terminationReason = cp?.termination_reason as string | undefined;
	const totalContributions = cp?.total_contributions as number | undefined;
	const agentContributions = (cp?.agent_contributions as Record<string, number>) ?? {};

	// Parse round events
	const rounds: RoundRow[] = roundEvents
		.map((e) => {
			const p = e.payload as Record<string, unknown>;
			return {
				roundNumber: (p.round_number as number) ?? 0,
				agentsActivated: (p.agents_activated as string[]) ?? [],
				contributions: (p.contributions as number) ?? 0,
				totalContributions: (p.total_contributions as number) ?? 0,
				entries: parseRoundEntries(p.round_entries),
			};
		})
		.sort((a, b) => a.roundNumber - b.roundNumber);

	// Build leaderboard from complete event or from round events
	const leaderboard = Object.entries(agentContributions)
		.map(([name, count]) => ({ name, count }))
		.sort((a, b) => b.count - a.count);
	const maxContributions = leaderboard.length > 0 ? leaderboard[0].count : 1;

	return (
		<div className="space-y-3">
			{/* Header */}
			<div className="flex items-center gap-2 flex-wrap">
				{controlStrategy && <ControlStrategyBadge strategy={controlStrategy} />}
				{totalContributions != null && (
					<span className="text-xs text-muted-foreground">
						{totalContributions} contribution
						{totalContributions !== 1 ? "s" : ""}
					</span>
				)}
				{roundsCompleted != null && (
					<span className="text-xs text-muted-foreground">
						· {roundsCompleted} round{roundsCompleted !== 1 ? "s" : ""}
					</span>
				)}
			</div>

			{/* Round cards */}
			{rounds.map((round) => (
				<RoundCard key={round.roundNumber} round={round} agentMap={agentMap} onNavigateToAgent={onNavigateToAgent} />
			))}

			{/* Agent contribution leaderboard */}
			{leaderboard.length > 0 && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Agent contributions</span>
					<div className="border border-border rounded-md overflow-hidden">
						<table className="w-full text-xs" data-testid="contribution-leaderboard">
							<thead>
								<tr className="bg-muted/50">
									<th className="text-left px-2 py-1 font-medium text-muted-foreground">Agent</th>
									<th className="text-left px-2 py-1 font-medium text-muted-foreground">Contributions</th>
								</tr>
							</thead>
							<tbody>
								{leaderboard.map((entry) => (
									<tr key={entry.name} className="border-t border-border">
										<td className="px-2 py-1">
											<AgentLink name={entry.name} agentMap={agentMap} onNavigateToAgent={onNavigateToAgent} />
										</td>
										<td className="px-2 py-1">
											<div className="flex items-center gap-1.5">
												<div className="w-16 bg-muted rounded-full h-1.5">
													<div
														className="bg-primary rounded-full h-1.5"
														style={{
															width: `${(entry.count / maxContributions) * 100}%`,
														}}
													/>
												</div>
												<span className="font-mono tabular-nums">{entry.count}</span>
											</div>
										</td>
									</tr>
								))}
							</tbody>
						</table>
					</div>
				</div>
			)}

			{/* Footer */}
			{terminationReason && <div className="text-xs text-muted-foreground">{terminationReason}</div>}
		</div>
	);
}

function ControlStrategyBadge({ strategy }: { strategy: string }) {
	let colorClass: string;
	switch (strategy) {
		case "PrioritizedControl":
			colorClass = "bg-info-muted text-info";
			break;
		case "OpportunisticControl":
			colorClass = "bg-primary/10 text-primary";
			break;
		default: // ScheduledControl and others
			colorClass = "bg-muted text-muted-foreground";
			break;
	}

	return (
		<span className={`text-[10px] px-1.5 py-0.5 rounded ${colorClass}`} data-testid="control-strategy-badge">
			{strategy}
		</span>
	);
}

function parseRoundEntries(raw: unknown): RoundEntry[] {
	if (!Array.isArray(raw)) return [];
	return raw.map((e: Record<string, unknown>) => ({
		operation: (e.operation as RoundEntry["operation"]) ?? "write",
		author: (e.author as string) ?? "",
		content: (e.content as string) ?? "",
		scope: (e.scope as string) ?? null,
		entry_id: (e.entry_id as string) ?? "",
		original_entry_id: (e.original_entry_id as string) ?? null,
		retract_reason: (e.retract_reason as string) ?? null,
	}));
}

function RoundCard({
	round,
	agentMap,
	onNavigateToAgent,
}: {
	round: RoundRow;
	agentMap: Map<string, string>;
	onNavigateToAgent: (spanId: string) => void;
}) {
	return (
		<div
			className="border border-border rounded-md p-2 space-y-1"
			data-testid={`blackboard-round-${round.roundNumber}`}
		>
			<div className="flex items-center justify-between">
				<span className="text-xs font-medium text-muted-foreground">Round {round.roundNumber}</span>
				<span className="text-xs text-muted-foreground">
					{round.contributions} contribution
					{round.contributions !== 1 ? "s" : ""} · {round.totalContributions} total
				</span>
			</div>
			<div className="flex items-center gap-1 flex-wrap text-xs">
				<span className="text-muted-foreground">Agents:</span>
				{round.agentsActivated.map((name, i) => (
					<span key={name}>
						<AgentLink name={name} agentMap={agentMap} onNavigateToAgent={onNavigateToAgent} />
						{i < round.agentsActivated.length - 1 && <span className="text-muted-foreground">,</span>}
					</span>
				))}
			</div>
			{round.entries.length > 0 && (
				<div className="mt-1 space-y-0.5" data-testid={`round-${round.roundNumber}-entries`}>
					{round.entries.map((entry) => (
						<RoundEntryRow key={entry.entry_id} entry={entry} />
					))}
				</div>
			)}
		</div>
	);
}

function RoundEntryRow({ entry }: { entry: RoundEntry }) {
	const [expanded, setExpanded] = useState(false);
	const hasContent = !!entry.content;

	return (
		<div className={`text-xs border-l-2 ${operationBorderClass(entry.operation)}`} data-testid="round-entry">
			<button
				type="button"
				className="w-full flex items-center gap-1.5 pl-2 py-1 text-left hover:bg-accent/50 transition-colors"
				onClick={() => hasContent && setExpanded(!expanded)}
				data-testid="toggle-preview"
			>
				{hasContent && <span className="text-muted-foreground w-3 shrink-0">{expanded ? "▾" : "▸"}</span>}
				<OperationBadge operation={entry.operation} />
				<span className="font-medium">{entry.author}</span>
				{entry.scope && (
					<span className="text-[10px] px-1 py-0 rounded bg-muted text-muted-foreground">{entry.scope}</span>
				)}
				{entry.operation === "supersede" && entry.original_entry_id && (
					<span className="text-muted-foreground">← {truncateId(entry.original_entry_id)}</span>
				)}
				{entry.operation === "retract" && entry.retract_reason && (
					<span className="text-muted-foreground italic">{entry.retract_reason}</span>
				)}
			</button>
			{expanded && hasContent && (
				<div className="pl-2 pr-2 pb-1" data-testid="content-preview">
					<pre className="text-xs text-muted-foreground whitespace-pre-wrap break-words bg-muted/50 rounded p-2 max-h-[300px] overflow-y-auto">
						{entry.content}
					</pre>
				</div>
			)}
		</div>
	);
}

function OperationBadge({ operation }: { operation: RoundEntry["operation"] }) {
	const config = {
		write: { label: "write", className: "bg-muted text-muted-foreground" },
		supersede: { label: "supersede", className: "bg-info-muted text-info" },
		retract: { label: "retract", className: "bg-warning-muted text-warning" },
	}[operation];

	return (
		<span className={`text-[10px] px-1 py-0 rounded ${config.className}`} data-testid="operation-badge">
			{config.label}
		</span>
	);
}

function operationBorderClass(operation: RoundEntry["operation"]): string {
	switch (operation) {
		case "supersede":
			return "border-info/50";
		case "retract":
			return "border-warning/50";
		default:
			return "border-border";
	}
}

function truncateId(id: string): string {
	return id.length > 8 ? `${id.slice(0, 8)}…` : id;
}
