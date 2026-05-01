import { useEffect, useMemo, useState } from "react";
import { ErrorState } from "../components/feedback/error-state";
import { AgentDetailSkeleton } from "../components/feedback/loading-skeleton";
import { TokenUsage } from "../components/primitives/token-usage";
import { useObservatory } from "../context/observatory-context";
import { useAgentDetail } from "../hooks/use-agent-detail";
import type { AgentInfo, TraceEvent } from "../types";
import type { RelatedAgent } from "../utils/related-agents";
import { findRelatedAgents } from "../utils/related-agents";

interface AgentDetailPageProps {
	runId: string;
	spanId: string;
	onBack: () => void;
	onBackToRuns: () => void;
	onNavigateToAgent?: (spanId: string) => void;
	runLabel?: string;
}

/** Multi-agent event types used for relationship detection. */
const MULTI_AGENT_EVENT_TYPES = [
	"multi_agent.delegation",
	"multi_agent.handoff",
	"multi_agent.supervision",
	"multi_agent.bidding.start",
	"multi_agent.debate.start",
	"multi_agent.consensus.start",
	"multi_agent.broadcast.start",
	"blackboard.start",
	"multi_agent.peer.start",
	"multi_agent.peer.consultation",
	"multi_agent.bus.start",
];

export function AgentDetailPage({
	runId,
	spanId,
	onBack,
	onBackToRuns,
	onNavigateToAgent,
	runLabel,
}: AgentDetailPageProps) {
	const { client, agentViewRegistry, panelRegistry } = useObservatory();
	const { agent, events, spanTree, isLoading, error } = useAgentDetail(runId, spanId);

	// Fetch all agents + multi-agent events for relationship detection
	const [allAgents, setAllAgents] = useState<AgentInfo[]>([]);
	const [multiAgentEvents, setMultiAgentEvents] = useState<TraceEvent[]>([]);
	const [relatedAgentsError, setRelatedAgentsError] = useState<string | null>(null);

	useEffect(() => {
		let cancelled = false;
		// Clear any prior error so navigating between agents does not show stale state.
		setRelatedAgentsError(null);

		Promise.all([
			client.listAgents(runId),
			client.queryEvents(runId, { eventTypes: MULTI_AGENT_EVENT_TYPES, limit: 500 }),
		])
			.then(([agentsResp, eventsResp]) => {
				if (!cancelled) {
					setAllAgents(agentsResp.agents);
					setMultiAgentEvents(eventsResp.events);
				}
			})
			.catch((err) => {
				if (!cancelled) {
					setRelatedAgentsError(String(err));
				}
			});

		return () => {
			cancelled = true;
		};
	}, [client, runId]);

	const relatedAgents = useMemo<RelatedAgent[]>(() => {
		if (!agent || allAgents.length === 0) return [];
		return findRelatedAgents(agent.agent_name, multiAgentEvents, allAgents);
	}, [agent, multiAgentEvents, allAgents]);

	// "timeline" is the built-in first tab; capability panel IDs are dynamic
	const [activeTab, setActiveTab] = useState<string>("timeline");

	// Reset to timeline if the active panel is no longer visible
	const visiblePanelIds = agent ? new Set(panelRegistry.getPanels(agent, events).map((p) => p.id)) : null;
	const resolvedTab = activeTab === "timeline" || visiblePanelIds?.has(activeTab) ? activeTab : "timeline";

	if (isLoading) {
		return <AgentDetailSkeleton />;
	}

	if (error) {
		return <ErrorState error={error} />;
	}

	if (!agent || !spanTree) {
		return <div className="text-muted-foreground py-8 text-center">Agent not found</div>;
	}

	const visiblePanels = panelRegistry.getPanels(agent, events);
	const TimelineView = agentViewRegistry.getView(agent.agent_type);

	return (
		<main className="flex flex-col h-full">
			{/* Breadcrumb */}
			<div className="border-b px-4 py-2 flex items-center gap-1.5 text-sm text-muted-foreground">
				<button type="button" onClick={onBackToRuns} className="hover:text-foreground transition-colors">
					Runs
				</button>
				<span>›</span>
				<button
					type="button"
					onClick={onBack}
					className="hover:text-foreground transition-colors truncate max-w-[200px]"
				>
					{runLabel || runId}
				</button>
				<span>›</span>
				<span className="text-foreground truncate">
					{agent.agent_name}
					{agent.agent_type && <span className="text-muted-foreground ml-1">({agent.agent_type})</span>}
				</span>
			</div>

			{/* Related Agents */}
			{relatedAgents.length > 0 && onNavigateToAgent && (
				<div className="border-b px-4 py-2 flex items-center gap-2 flex-wrap">
					<span className="text-xs text-muted-foreground">Related:</span>
					{relatedAgents.map((ra) => (
						<button
							type="button"
							key={ra.spanId}
							onClick={() => onNavigateToAgent(ra.spanId)}
							className="inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium bg-accent text-accent-foreground hover:bg-accent/80 transition-colors cursor-pointer"
						>
							<span className="text-muted-foreground">{ra.relationship}:</span>
							<span>{ra.agentName}</span>
						</button>
					))}
				</div>
			)}
			{relatedAgents.length === 0 && relatedAgentsError && (
				<div className="border-b">
					<ErrorState variant="inline" title="Couldn't load related agents" error={relatedAgentsError} />
				</div>
			)}

			{/* Agent Header */}
			<div className="border-b px-4 py-3">
				<div className="flex items-center gap-2 flex-wrap">
					<h2 className="text-base font-semibold">{agent.agent_name}</h2>
					{agent.agent_type && (
						<span className="inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium bg-accent-status-muted text-accent-status-muted-foreground capitalize">
							{agent.agent_type}
						</span>
					)}
					{agent.capabilities.map((cap) => (
						<span
							key={cap}
							className="inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-medium bg-muted text-muted-foreground"
						>
							{cap}
						</span>
					))}
				</div>

				{/* Stats row */}
				<div className="flex items-center gap-4 mt-2 text-xs flex-wrap">
					<StatPill label="LLM calls" value={agent.stats.llm_calls} />
					<StatPill label="Tool calls" value={agent.stats.tool_calls} />
					<StatPill label="Iterations" value={agent.stats.iterations} />
					<StatPill label="Errors" value={agent.stats.errors} variant={agent.stats.errors > 0 ? "error" : "default"} />
					<TokenUsage inputTokens={agent.stats.input_tokens} outputTokens={agent.stats.output_tokens} />
					{agent.stats.duration_ms != null && (
						<StatPill label="Duration" value={formatDuration(agent.stats.duration_ms)} />
					)}
				</div>
			</div>

			{/* Tab Bar — toolbar of toggle buttons. The tab content section
			    references the active tab's button id via aria-labelledby. */}
			<div role="toolbar" aria-label="Agent detail views" className="border-b px-4 flex gap-0">
				<TabButton
					id="tab-timeline"
					label="Timeline"
					isActive={resolvedTab === "timeline"}
					onClick={() => setActiveTab("timeline")}
				/>
				{visiblePanels.map((panel) => (
					<TabButton
						id={`tab-${panel.id}`}
						key={panel.id}
						label={panel.label}
						isActive={resolvedTab === panel.id}
						onClick={() => setActiveTab(panel.id)}
					/>
				))}
			</div>

			{/* Tab Content — <section> with aria-labelledby provides the implicit
			    region role labelled by the active tab button. */}
			<section aria-labelledby={`tab-${resolvedTab}`} className="flex-1 overflow-y-auto">
				{resolvedTab === "timeline" ? (
					<TimelineView agent={agent} events={events} spanTree={spanTree} />
				) : (
					(() => {
						const panel = visiblePanels.find((p) => p.id === resolvedTab);
						if (!panel) return null;
						const PanelComponent = panel.component;
						return <PanelComponent agent={agent} events={events} spanTree={spanTree} />;
					})()
				)}
			</section>
		</main>
	);
}

// ---------------------------------------------------------------------------
// Internal components
// ---------------------------------------------------------------------------

function formatDuration(ms: number): string {
	return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`;
}

function StatPill({
	label,
	value,
	variant = "default",
}: {
	label: string;
	value: string | number;
	variant?: "default" | "error";
}) {
	const colorClass = variant === "error" ? "text-destructive font-medium" : "text-foreground";
	return (
		<div className="flex items-center gap-1.5">
			<span className="text-muted-foreground">{label}</span>
			<span className={`font-mono tabular-nums ${colorClass}`}>{value}</span>
		</div>
	);
}

function TabButton({
	id,
	label,
	isActive,
	onClick,
}: {
	id: string;
	label: string;
	isActive: boolean;
	onClick: () => void;
}) {
	return (
		<button
			type="button"
			id={id}
			aria-pressed={isActive}
			onClick={onClick}
			className={`px-3 py-2 text-sm font-medium border-b-2 transition-colors ${
				isActive
					? "border-primary text-foreground"
					: "border-transparent text-muted-foreground hover:text-foreground hover:border-muted-foreground/30"
			}`}
		>
			{label}
		</button>
	);
}
