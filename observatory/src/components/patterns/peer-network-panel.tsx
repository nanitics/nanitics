import { useMemo, useState } from "react";
import type { ContentBounds } from "../../hooks/use-svg-viewport";
import { useSVGViewport } from "../../hooks/use-svg-viewport";
import type { AgentInfo, TraceEvent } from "../../types";
import { AgentLink } from "./delegation-panel";

interface PeerNetworkPanelProps {
	events: TraceEvent[];
	agents: AgentInfo[];
	onNavigateToAgent: (spanId: string) => void;
}

interface AgentPosition {
	agentName: string;
	x: number;
	y: number;
	isEntry: boolean;
}

interface ConsultationEdge {
	from: string;
	to: string;
	count: number;
	messagePreviews: string[];
}

interface ConsultationRow {
	from: string;
	to: string;
	messagePreview: string;
	consultationNumber: number;
}

const NODE_RADIUS = 24;

export function PeerNetworkPanel({ events, agents, onNavigateToAgent }: PeerNetworkPanelProps) {
	const agentMap = new Map(agents.map((a) => [a.agent_name, a.span_id]));
	const [logExpanded, setLogExpanded] = useState(false);

	const startEvent = events.find((e) => e.event_type === "multi_agent.peer.start");
	const consultationEvents = events.filter((e) => e.event_type === "multi_agent.peer.consultation");
	const completeEvent = events.find((e) => e.event_type === "multi_agent.peer.complete");

	const sp = startEvent?.payload as Record<string, unknown> | undefined;
	const entryAgent = (sp?.entry_agent as string) ?? "unknown";
	const peerNames = (sp?.peer_names as string[]) ?? [];
	const maxInvocations = (sp?.max_invocations as number) ?? 0;

	const cp = completeEvent?.payload as Record<string, unknown> | undefined;
	const totalConsultations = cp?.total_consultations as number | undefined;
	const invocationsUsed = cp?.invocations_used as number | undefined;
	const agentsConsulted = cp?.agents_consulted as string[] | undefined;
	const terminationReason = cp?.termination_reason as string | undefined;

	// Parse consultation events
	const consultations: ConsultationRow[] = useMemo(
		() =>
			consultationEvents.map((e) => {
				const p = e.payload as Record<string, unknown>;
				return {
					from: (p.from_agent as string) ?? "unknown",
					to: (p.to_agent as string) ?? "unknown",
					messagePreview: (p.message as string) ?? "",
					consultationNumber: (p.consultation_number as number) ?? 0,
				};
			}),
		[consultationEvents],
	);

	// Aggregate edges
	const edges = useMemo(() => {
		const edgeMap = new Map<string, ConsultationEdge>();
		for (const c of consultations) {
			const key = `${c.from}→${c.to}`;
			const existing = edgeMap.get(key);
			if (existing) {
				existing.count++;
				existing.messagePreviews.push(c.messagePreview);
			} else {
				edgeMap.set(key, {
					from: c.from,
					to: c.to,
					count: 1,
					messagePreviews: [c.messagePreview],
				});
			}
		}
		return [...edgeMap.values()];
	}, [consultations]);

	// Compute circular layout
	const allAgents = useMemo(() => {
		const nameSet = new Set([entryAgent, ...peerNames]);
		return [...nameSet];
	}, [entryAgent, peerNames]);

	const positions = useMemo(() => computeCircularLayout(allAgents, entryAgent), [allAgents, entryAgent]);

	// Compute content bounds for viewport
	const contentBounds = useMemo<ContentBounds | null>(() => {
		if (positions.length === 0) return null;
		const pad = NODE_RADIUS + 30;
		let minX = Infinity,
			maxX = -Infinity,
			minY = Infinity,
			maxY = -Infinity;
		for (const pos of positions) {
			minX = Math.min(minX, pos.x - pad);
			maxX = Math.max(maxX, pos.x + pad);
			minY = Math.min(minY, pos.y - pad);
			maxY = Math.max(maxY, pos.y + pad);
		}
		return { minX, minY, maxX, maxY };
	}, [positions]);

	const { svgRef, viewBox, isDragging, zoomIn, zoomOut, fitToContent, svgHandlers } = useSVGViewport(contentBounds);

	const positionMap = useMemo(() => new Map(positions.map((p) => [p.agentName, p])), [positions]);

	// Build a set of reverse edge keys for bidirectional detection
	const edgeKeySet = useMemo(() => new Set(edges.map((e) => `${e.from}→${e.to}`)), [edges]);

	// Budget progress
	const budgetUsed = invocationsUsed ?? consultations.length;
	const budgetFraction = maxInvocations > 0 ? budgetUsed / maxInvocations : 0;

	return (
		<div className="space-y-3" data-testid="peer-network-panel">
			{/* Budget indicator */}
			{maxInvocations > 0 && (
				<div className="space-y-1" data-testid="budget-indicator">
					<div className="flex items-center justify-between text-xs text-muted-foreground">
						<span>Budget</span>
						<span>
							{budgetUsed} / {maxInvocations} invocations
						</span>
					</div>
					<div className="w-full bg-muted rounded-full h-1.5">
						<div
							className={`rounded-full h-1.5 transition-all ${
								budgetFraction >= 1 ? "bg-destructive" : budgetFraction >= 0.8 ? "bg-warning" : "bg-primary"
							}`}
							style={{ width: `${Math.min(budgetFraction * 100, 100)}%` }}
						/>
					</div>
				</div>
			)}

			{/* SVG Graph */}
			<div
				className="relative w-full overflow-hidden select-none border border-border rounded-md"
				style={{ minHeight: 300 }}
			>
				{/* Zoom controls */}
				<div className="absolute top-2 right-2 z-10 flex gap-1">
					<button
						type="button"
						onClick={zoomIn}
						className="p-1 rounded bg-background border border-border text-muted-foreground hover:text-foreground text-xs"
						aria-label="Zoom in"
					>
						+
					</button>
					<button
						type="button"
						onClick={zoomOut}
						className="p-1 rounded bg-background border border-border text-muted-foreground hover:text-foreground text-xs"
						aria-label="Zoom out"
					>
						−
					</button>
					<button
						type="button"
						onClick={fitToContent}
						className="p-1 rounded bg-background border border-border text-muted-foreground hover:text-foreground text-xs"
						aria-label="Fit to content"
					>
						⊞
					</button>
				</div>

				<svg
					ref={svgRef}
					viewBox={viewBox}
					className={`w-full h-full ${isDragging ? "cursor-grabbing" : "cursor-grab"}`}
					style={{ minHeight: 300 }}
					{...svgHandlers}
				>
					<defs>
						<marker id="peer-arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
							<polygon points="0 0, 8 3, 0 6" className="fill-muted-foreground/60" />
						</marker>
					</defs>

					{/* Edges */}
					<g data-testid="peer-edges">
						{edges.map((edge) => {
							const fromPos = positionMap.get(edge.from);
							const toPos = positionMap.get(edge.to);
							if (!fromPos || !toPos) return null;

							const reverseKey = `${edge.to}→${edge.from}`;
							const hasBidirectional = edgeKeySet.has(reverseKey);

							const path = computeEdgePath(fromPos, toPos, hasBidirectional, edge.from < edge.to);

							const thickness = Math.min(1.5 + edge.count * 0.5, 4);

							// Label position at midpoint of the curve
							const mid = computeEdgeMidpoint(fromPos, toPos, hasBidirectional, edge.from < edge.to);

							return (
								<g key={`${edge.from}→${edge.to}`} data-testid={`edge-${edge.from}-${edge.to}`}>
									<path
										d={path}
										fill="none"
										className="stroke-muted-foreground/50"
										strokeWidth={thickness}
										markerEnd="url(#peer-arrowhead)"
									/>
									{edge.count > 1 && (
										<text
											x={mid.x}
											y={mid.y}
											textAnchor="middle"
											dominantBaseline="central"
											className="fill-muted-foreground text-[9px]"
										>
											×{edge.count}
										</text>
									)}
								</g>
							);
						})}
					</g>

					{/* Nodes */}
					<g data-testid="peer-nodes">
						{positions.map((pos) => (
							<g
								key={pos.agentName}
								data-dag-node
								data-testid={`peer-node-${pos.agentName}`}
								className="cursor-pointer"
								onClick={(e) => {
									e.stopPropagation();
									const spanId = agentMap.get(pos.agentName);
									if (spanId) onNavigateToAgent(spanId);
								}}
							>
								<circle
									cx={pos.x}
									cy={pos.y}
									r={NODE_RADIUS}
									className={`fill-background ${
										pos.isEntry ? "stroke-primary stroke-[2.5]" : "stroke-border stroke-[1.5]"
									}`}
								/>
								<text
									x={pos.x}
									y={pos.y}
									textAnchor="middle"
									dominantBaseline="central"
									className={`text-[10px] ${pos.isEntry ? "fill-primary font-medium" : "fill-foreground"}`}
								>
									{truncateName(pos.agentName, 10)}
								</text>
							</g>
						))}
					</g>
				</svg>
			</div>

			{/* Consultation log */}
			{consultations.length > 0 && (
				<div className="space-y-1">
					<button
						type="button"
						onClick={() => setLogExpanded(!logExpanded)}
						className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1"
						data-testid="consultation-log-toggle"
					>
						<span>{logExpanded ? "▾" : "▸"}</span>
						<span>Consultation log ({consultations.length})</span>
					</button>
					{logExpanded && (
						<div className="space-y-1 pl-2 border-l-2 border-border">
							{consultations.map((c) => (
								<div key={c.consultationNumber} className="text-xs flex items-start gap-1">
									<span className="text-muted-foreground tabular-nums shrink-0">#{c.consultationNumber}</span>
									<AgentLink name={c.from} agentMap={agentMap} onNavigateToAgent={onNavigateToAgent} />
									<span className="text-muted-foreground">→</span>
									<AgentLink name={c.to} agentMap={agentMap} onNavigateToAgent={onNavigateToAgent} />
									{c.messagePreview && <span className="text-muted-foreground truncate">— {c.messagePreview}</span>}
								</div>
							))}
						</div>
					)}
				</div>
			)}

			{/* Footer */}
			<div className="flex items-center gap-2 flex-wrap text-xs text-muted-foreground">
				{totalConsultations != null && (
					<span>
						{totalConsultations} consultation
						{totalConsultations !== 1 ? "s" : ""}
					</span>
				)}
				{agentsConsulted && <span>· {agentsConsulted.length} agents consulted</span>}
				{terminationReason && <span>· {terminationReason}</span>}
			</div>
		</div>
	);
}

// ---------------------------------------------------------------------------
// Layout helpers
// ---------------------------------------------------------------------------

function computeCircularLayout(agentNames: string[], entryAgent: string): AgentPosition[] {
	if (agentNames.length === 0) return [];

	const R = Math.max(80, agentNames.length * 25);

	// Entry agent at 12 o'clock (angle = -π/2), others sorted alphabetically
	const others = agentNames.filter((n) => n !== entryAgent).sort((a, b) => a.localeCompare(b));

	const orderedNames = [entryAgent, ...others];
	const angleStep = (2 * Math.PI) / orderedNames.length;

	return orderedNames.map((name, i) => {
		const angle = -Math.PI / 2 + i * angleStep;
		return {
			agentName: name,
			x: Math.round(R * Math.cos(angle)),
			y: Math.round(R * Math.sin(angle)),
			isEntry: name === entryAgent,
		};
	});
}

function computeEdgePath(
	from: AgentPosition,
	to: AgentPosition,
	hasBidirectional: boolean,
	isFirstOfPair: boolean,
): string {
	const dx = to.x - from.x;
	const dy = to.y - from.y;
	const dist = Math.sqrt(dx * dx + dy * dy);
	if (dist === 0) return "";

	// Normalize direction
	const nx = dx / dist;
	const ny = dy / dist;

	// Shorten by node radius at both ends
	const startX = from.x + nx * NODE_RADIUS;
	const startY = from.y + ny * NODE_RADIUS;
	const endX = to.x - nx * (NODE_RADIUS + 8); // +8 for arrowhead
	const endY = to.y - ny * (NODE_RADIUS + 8);

	if (!hasBidirectional) {
		// Straight line (via quadratic with midpoint control)
		const midX = (startX + endX) / 2;
		const midY = (startY + endY) / 2;
		return `M ${startX} ${startY} Q ${midX} ${midY} ${endX} ${endY}`;
	}

	// Offset control point perpendicular to the line for bidirectional pairs
	const perpX = -ny;
	const perpY = nx;
	const offset = isFirstOfPair ? 20 : -20;
	const cpX = (from.x + to.x) / 2 + perpX * offset;
	const cpY = (from.y + to.y) / 2 + perpY * offset;

	return `M ${startX} ${startY} Q ${cpX} ${cpY} ${endX} ${endY}`;
}

function computeEdgeMidpoint(
	from: AgentPosition,
	to: AgentPosition,
	hasBidirectional: boolean,
	isFirstOfPair: boolean,
): { x: number; y: number } {
	const dx = to.x - from.x;
	const dy = to.y - from.y;
	const dist = Math.sqrt(dx * dx + dy * dy);

	if (!hasBidirectional || dist === 0) {
		return { x: (from.x + to.x) / 2, y: (from.y + to.y) / 2 };
	}

	const nx = dx / dist;
	const ny = dy / dist;
	const perpX = -ny;
	const perpY = nx;
	const offset = isFirstOfPair ? 20 : -20;

	// Quadratic Bézier midpoint at t=0.5: P = 0.25*P0 + 0.5*Cp + 0.25*P1
	const cpX = (from.x + to.x) / 2 + perpX * offset;
	const cpY = (from.y + to.y) / 2 + perpY * offset;

	return {
		x: 0.25 * from.x + 0.5 * cpX + 0.25 * to.x,
		y: 0.25 * from.y + 0.5 * cpY + 0.25 * to.y,
	};
}

function truncateName(name: string, maxLen: number): string {
	if (name.length <= maxLen) return name;
	return `${name.slice(0, maxLen - 1)}…`;
}
