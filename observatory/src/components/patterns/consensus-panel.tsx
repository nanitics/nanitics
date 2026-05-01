import type { AgentInfo, TraceEvent } from "../../types";
import { AgentLink } from "./delegation-panel";

interface ConsensusPanelProps {
	events: TraceEvent[];
	agents: AgentInfo[];
	onNavigateToAgent: (spanId: string) => void;
}

interface VoteRow {
	agentName: string;
	outputPreview: string;
	round: number;
	error: string | null;
}

interface AgreementRow {
	round: number;
	agreementLevel: number;
	converged: boolean;
}

export function ConsensusPanel({ events, agents, onNavigateToAgent }: ConsensusPanelProps) {
	const agentMap = new Map(agents.map((a) => [a.agent_name, a.span_id]));

	const startEvent = events.find((e) => e.event_type === "multi_agent.consensus.start");
	const voteEvents = events.filter((e) => e.event_type === "multi_agent.consensus.vote");
	const agreementEvents = events.filter((e) => e.event_type === "multi_agent.consensus.agreement");
	const completeEvent = events.find((e) => e.event_type === "multi_agent.consensus.complete");

	const sp = startEvent?.payload as Record<string, unknown> | undefined;
	const strategy = sp?.strategy as string | undefined;
	const agentNames = (sp?.agent_names as string[]) ?? [];
	const deliberationEnabled = sp?.deliberation_enabled as boolean | undefined;

	const cp = completeEvent?.payload as Record<string, unknown> | undefined;
	const finalAgreement = cp?.final_agreement as number | undefined;
	const roundsCompleted = cp?.rounds_completed as number | undefined;
	const terminationReason = cp?.termination_reason as string | undefined;

	// Parse votes
	const votes: VoteRow[] = voteEvents.map((e) => {
		const p = e.payload as Record<string, unknown>;
		return {
			agentName: (p.agent_name as string) ?? "unknown",
			outputPreview: (p.output as string) ?? "",
			round: (p.round as number) ?? 1,
			error: (p.error as string | null) ?? null,
		};
	});

	// Parse agreements
	const agreements: AgreementRow[] = agreementEvents.map((e) => {
		const p = e.payload as Record<string, unknown>;
		return {
			round: (p.round as number) ?? 1,
			agreementLevel: (p.agreement_level as number) ?? 0,
			converged: (p.converged as boolean) ?? false,
		};
	});

	const maxRound = Math.max(...votes.map((v) => v.round), ...agreements.map((a) => a.round), 1);
	const isMultiRound = deliberationEnabled && maxRound > 1;
	const didConverge = agreements.some((a) => a.converged);

	return (
		<div className="space-y-3">
			{/* Header */}
			<div className="flex items-center gap-2 flex-wrap">
				{strategy && <span className="text-[10px] px-1.5 py-0.5 rounded bg-info-muted text-info">{strategy}</span>}
				<span className="text-xs text-muted-foreground">
					{agentNames.length} agent{agentNames.length !== 1 ? "s" : ""}
				</span>
				{isMultiRound && roundsCompleted != null && (
					<span className="text-xs text-muted-foreground">
						· {roundsCompleted} round{roundsCompleted !== 1 ? "s" : ""}
					</span>
				)}
				{isMultiRound && (
					<span
						className={`text-[10px] px-1.5 py-0.5 rounded ${
							didConverge
								? "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300"
								: "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300"
						}`}
						data-testid="convergence-badge"
					>
						{didConverge ? "Converged" : "Did not converge"}
					</span>
				)}
			</div>

			{isMultiRound ? (
				<MultiRoundView
					votes={votes}
					agreements={agreements}
					agentNames={agentNames}
					maxRound={maxRound}
					agentMap={agentMap}
					onNavigateToAgent={onNavigateToAgent}
				/>
			) : (
				<SingleRoundView
					votes={votes.filter((v) => v.round === 1)}
					agentMap={agentMap}
					onNavigateToAgent={onNavigateToAgent}
				/>
			)}

			{/* Agreement bar */}
			{finalAgreement != null && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Final agreement</span>
					<AgreementBar level={finalAgreement} converged={didConverge} />
				</div>
			)}

			{/* Footer */}
			{terminationReason && <div className="text-xs text-muted-foreground">{terminationReason}</div>}
		</div>
	);
}

function SingleRoundView({
	votes,
	agentMap,
	onNavigateToAgent,
}: {
	votes: VoteRow[];
	agentMap: Map<string, string>;
	onNavigateToAgent: (spanId: string) => void;
}) {
	return (
		<div className="border border-border rounded-md overflow-hidden">
			<table className="w-full text-xs">
				<thead>
					<tr className="bg-muted/50">
						<th className="text-left px-2 py-1 font-medium text-muted-foreground">Agent</th>
						<th className="text-left px-2 py-1 font-medium text-muted-foreground">Output</th>
						<th className="text-center px-2 py-1 font-medium text-muted-foreground">Error</th>
					</tr>
				</thead>
				<tbody>
					{votes.map((vote, i) => (
						// biome-ignore lint/suspicious/noArrayIndexKey: votes lack guaranteed-unique key
						<tr key={i} className="border-t border-border">
							<td className="px-2 py-1">
								<AgentLink name={vote.agentName} agentMap={agentMap} onNavigateToAgent={onNavigateToAgent} />
							</td>
							<td className="px-2 py-1 max-w-[200px] truncate">{vote.outputPreview || "—"}</td>
							<td className="px-2 py-1 text-center">
								{vote.error ? (
									<span className="text-destructive" title={vote.error}>
										✗
									</span>
								) : (
									""
								)}
							</td>
						</tr>
					))}
				</tbody>
			</table>
		</div>
	);
}

function MultiRoundView({
	votes,
	agreements,
	agentNames,
	maxRound,
	agentMap,
	onNavigateToAgent,
}: {
	votes: VoteRow[];
	agreements: AgreementRow[];
	agentNames: string[];
	maxRound: number;
	agentMap: Map<string, string>;
	onNavigateToAgent: (spanId: string) => void;
}) {
	const rounds = Array.from({ length: maxRound }, (_, i) => i + 1);

	// Build vote lookup: agent → round → vote
	const voteMap = new Map<string, Map<number, VoteRow>>();
	for (const v of votes) {
		if (!voteMap.has(v.agentName)) voteMap.set(v.agentName, new Map());
		voteMap.get(v.agentName)?.set(v.round, v);
	}

	// Build agreement lookup: round → agreement
	const agreementMap = new Map<number, AgreementRow>();
	for (const a of agreements) {
		agreementMap.set(a.round, a);
	}

	return (
		<div className="space-y-3">
			{/* Vote matrix */}
			<div className="border border-border rounded-md overflow-hidden overflow-x-auto">
				<table className="w-full text-xs" data-testid="vote-matrix">
					<thead>
						<tr className="bg-muted/50">
							<th className="text-left px-2 py-1 font-medium text-muted-foreground">Agent</th>
							{rounds.map((r) => (
								<th key={r} className="text-left px-2 py-1 font-medium text-muted-foreground">
									R{r}
								</th>
							))}
						</tr>
					</thead>
					<tbody>
						{agentNames.map((agent) => (
							<tr key={agent} className="border-t border-border">
								<td className="px-2 py-1">
									<AgentLink name={agent} agentMap={agentMap} onNavigateToAgent={onNavigateToAgent} />
								</td>
								{rounds.map((r) => {
									const vote = voteMap.get(agent)?.get(r);
									return (
										<td key={r} className="px-2 py-1 max-w-[120px] truncate">
											{vote?.error ? (
												<span className="text-destructive" title={vote.error}>
													✗
												</span>
											) : vote?.outputPreview ? (
												<span title={vote.outputPreview}>
													{vote.outputPreview.length > 40 ? `${vote.outputPreview.slice(0, 40)}…` : vote.outputPreview}
												</span>
											) : (
												<span className="text-muted-foreground">—</span>
											)}
										</td>
									);
								})}
							</tr>
						))}
					</tbody>
				</table>
			</div>

			{/* Agreement progression */}
			<div className="space-y-1">
				<span className="text-xs text-muted-foreground">Agreement progression</span>
				<div className="space-y-1">
					{rounds.map((r) => {
						const agreement = agreementMap.get(r);
						if (!agreement) return null;
						return (
							<div key={r} className="flex items-center gap-2">
								<span className="text-xs text-muted-foreground w-6">R{r}</span>
								<AgreementBar level={agreement.agreementLevel} converged={agreement.converged} />
							</div>
						);
					})}
				</div>
			</div>
		</div>
	);
}

function AgreementBar({ level, converged }: { level: number; converged: boolean }) {
	const pct = Math.round(level * 100);
	return (
		<div className="flex items-center gap-1.5 flex-1" data-testid="agreement-bar">
			<div className="flex-1 bg-muted rounded-full h-1.5">
				<div
					className={`rounded-full h-1.5 ${converged ? "bg-emerald-500" : "bg-amber-500"}`}
					style={{ width: `${pct}%` }}
				/>
			</div>
			<span className="text-xs font-mono tabular-nums w-8 text-right">{pct}%</span>
		</div>
	);
}
