import { useState } from "react";
import type { AgentInfo, TraceEvent } from "../../types";
import { AgentLink } from "./delegation-panel";

interface DebatePanelProps {
	events: TraceEvent[];
	agents: AgentInfo[];
	onNavigateToAgent: (spanId: string) => void;
}

interface ArgumentRow {
	round: number;
	agentName: string;
	position: string;
	argumentPreview: string;
}

// Stable color palette for position badges
const POSITION_COLORS = [
	"bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300",
	"bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300",
	"bg-purple-100 text-purple-700 dark:bg-purple-950 dark:text-purple-300",
	"bg-teal-100 text-teal-700 dark:bg-teal-950 dark:text-teal-300",
	"bg-pink-100 text-pink-700 dark:bg-pink-950 dark:text-pink-300",
];

export function DebatePanel({ events, agents, onNavigateToAgent }: DebatePanelProps) {
	const agentMap = new Map(agents.map((a) => [a.agent_name, a.span_id]));

	const startEvent = events.find((e) => e.event_type === "multi_agent.debate.start");
	const argumentEvents = events.filter((e) => e.event_type === "multi_agent.debate.argument");
	const resolutionEvent = events.find((e) => e.event_type === "multi_agent.debate.resolution");
	const completeEvent = events.find((e) => e.event_type === "multi_agent.debate.complete");

	const sp = startEvent?.payload as Record<string, unknown> | undefined;
	const task = sp?.task as string | undefined;
	const resolutionStrategy = sp?.resolution_strategy as string | undefined;
	const positions = (sp?.positions as Record<string, string>) ?? {};

	const cp = completeEvent?.payload as Record<string, unknown> | undefined;
	const totalArguments = cp?.total_arguments as number | undefined;
	const terminationReason = cp?.termination_reason as string | undefined;
	const roundsCompleted = cp?.rounds_completed as number | undefined;

	const rp = resolutionEvent?.payload as Record<string, unknown> | undefined;
	const winner = rp?.winner as string | null | undefined;
	const reasoningPreview = rp?.reasoning as string | undefined;

	// Parse argument events
	const args: ArgumentRow[] = argumentEvents.map((e) => {
		const p = e.payload as Record<string, unknown>;
		return {
			round: (p.round as number) ?? 0,
			agentName: (p.agent_name as string) ?? "unknown",
			position: (p.position as string) ?? "",
			argumentPreview: (p.argument as string) ?? "",
		};
	});

	// Group arguments by round
	const roundMap = new Map<number, ArgumentRow[]>();
	for (const arg of args) {
		if (!roundMap.has(arg.round)) roundMap.set(arg.round, []);
		roundMap.get(arg.round)?.push(arg);
	}
	const rounds = [...roundMap.entries()].sort(([a], [b]) => a - b);

	// Build position → color mapping (stable across rounds)
	const uniquePositions = [...new Set(args.map((a) => a.position))];
	const positionColorMap = new Map<string, string>();
	uniquePositions.forEach((pos, i) => {
		positionColorMap.set(pos, POSITION_COLORS[i % POSITION_COLORS.length]);
	});

	// Determine debater count for layout
	const debaterNames = Object.keys(positions);
	const isTwoDebaters = debaterNames.length === 2;

	return (
		<div className="space-y-3">
			{/* Header */}
			<div className="flex items-center gap-2 flex-wrap">
				{resolutionStrategy && (
					<span className="text-[10px] px-1.5 py-0.5 rounded bg-info-muted text-info">{resolutionStrategy}</span>
				)}
				{roundsCompleted != null && (
					<span className="text-xs text-muted-foreground">
						{roundsCompleted} round{roundsCompleted !== 1 ? "s" : ""}
					</span>
				)}
				{totalArguments != null && (
					<span className="text-xs text-muted-foreground">
						· {totalArguments} argument{totalArguments !== 1 ? "s" : ""}
					</span>
				)}
			</div>

			{/* Task */}
			{task && <TruncatedText label="Task" text={task} />}

			{/* Initial positions */}
			{Object.keys(positions).length > 0 && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Initial positions</span>
					<div className="flex flex-wrap gap-2">
						{Object.entries(positions).map(([agent, position]) => (
							<div key={agent} className="flex items-center gap-1.5 text-xs">
								<AgentLink name={agent} agentMap={agentMap} onNavigateToAgent={onNavigateToAgent} />
								<PositionBadge position={position} colorMap={positionColorMap} />
							</div>
						))}
					</div>
				</div>
			)}

			{/* Rounds */}
			{rounds.map(([roundNum, roundArgs]) => (
				<div
					key={roundNum}
					className="border border-border rounded-md p-2 space-y-2"
					data-testid={`debate-round-${roundNum}`}
				>
					<span className="text-xs font-medium text-muted-foreground">Round {roundNum}</span>
					{isTwoDebaters ? (
						<div className="grid grid-cols-2 gap-2">
							{roundArgs.map((arg, i) => (
								<ArgumentCard
									// biome-ignore lint/suspicious/noArrayIndexKey: debate arguments have no unique ID
									key={i}
									arg={arg}
									positionColorMap={positionColorMap}
									agentMap={agentMap}
									onNavigateToAgent={onNavigateToAgent}
								/>
							))}
						</div>
					) : (
						<div className="space-y-2">
							{roundArgs.map((arg, i) => (
								<ArgumentCard
									// biome-ignore lint/suspicious/noArrayIndexKey: debate arguments have no unique ID
									key={i}
									arg={arg}
									positionColorMap={positionColorMap}
									agentMap={agentMap}
									onNavigateToAgent={onNavigateToAgent}
								/>
							))}
						</div>
					)}
				</div>
			))}

			{/* Resolution */}
			{resolutionEvent && (
				<div
					className="border border-border rounded-md p-2 space-y-1 bg-success-muted/20"
					data-testid="debate-resolution"
				>
					<span className="text-xs font-medium">Resolution</span>
					{winner && (
						<div className="text-xs">
							Winner: <AgentLink name={winner} agentMap={agentMap} onNavigateToAgent={onNavigateToAgent} />
						</div>
					)}
					{reasoningPreview && <TruncatedText label="Reasoning" text={reasoningPreview} />}
					{terminationReason && <div className="text-xs text-muted-foreground">{terminationReason}</div>}
				</div>
			)}
		</div>
	);
}

function ArgumentCard({
	arg,
	positionColorMap,
	agentMap,
	onNavigateToAgent,
}: {
	arg: ArgumentRow;
	positionColorMap: Map<string, string>;
	agentMap: Map<string, string>;
	onNavigateToAgent: (spanId: string) => void;
}) {
	const [expanded, setExpanded] = useState(false);
	const TRUNCATE = 200;
	const isLong = arg.argumentPreview.length > TRUNCATE;

	return (
		<div className="space-y-1">
			<div className="flex items-center gap-1.5">
				<AgentLink name={arg.agentName} agentMap={agentMap} onNavigateToAgent={onNavigateToAgent} />
				<PositionBadge position={arg.position} colorMap={positionColorMap} />
			</div>
			<div className="text-xs text-foreground">
				{expanded || !isLong ? arg.argumentPreview : `${arg.argumentPreview.slice(0, TRUNCATE)}…`}
				{isLong && (
					<button type="button" onClick={() => setExpanded(!expanded)} className="ml-1 text-primary hover:underline">
						{expanded ? "less" : "more"}
					</button>
				)}
			</div>
		</div>
	);
}

function PositionBadge({ position, colorMap }: { position: string; colorMap: Map<string, string> }) {
	const color = colorMap.get(position) ?? "bg-muted text-muted-foreground";
	return (
		<span className={`text-[10px] px-1.5 py-0.5 rounded ${color}`} data-testid="position-badge">
			{position}
		</span>
	);
}

function TruncatedText({ label, text }: { label: string; text: string }) {
	const [expanded, setExpanded] = useState(false);
	const TRUNCATE = 200;
	const isLong = text.length > TRUNCATE;

	return (
		<div className="space-y-1">
			<span className="text-xs text-muted-foreground">{label}</span>
			<div className="text-xs text-foreground">
				{expanded || !isLong ? text : `${text.slice(0, TRUNCATE)}…`}
				{isLong && (
					<button type="button" onClick={() => setExpanded(!expanded)} className="ml-1 text-primary hover:underline">
						{expanded ? "less" : "more"}
					</button>
				)}
			</div>
		</div>
	);
}
