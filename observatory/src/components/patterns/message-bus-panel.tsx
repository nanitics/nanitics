import dagre from "@dagrejs/dagre";
import { useCallback, useMemo, useState } from "react";
import type { ContentBounds } from "../../hooks/use-svg-viewport";
import { useSVGViewport } from "../../hooks/use-svg-viewport";
import type { AgentInfo, TraceEvent } from "../../types";
import { AgentLink } from "./delegation-panel";

interface MessageBusPanelProps {
	events: TraceEvent[];
	agents: AgentInfo[];
	onNavigateToAgent: (spanId: string) => void;
}

interface PublishedMessage {
	messageId: string;
	topic: string;
	author: string;
	contentPreview: string;
	depth: number;
	parentMessageId: string | null;
}

interface DeliveryInfo {
	messageId: string;
	agentName: string;
	outputPreview: string;
	messagesPublished: number;
	error: string | null;
}

interface MessageNode {
	id: string;
	topic: string;
	author: string;
	contentPreview: string;
	depth: number;
	x?: number;
	y?: number;
}

const NODE_WIDTH = 180;
const NODE_HEIGHT = 60;
const DAG_THRESHOLD = 40;

// Palette for topic colors
const TOPIC_COLORS = [
	{ bg: "bg-blue-100 dark:bg-blue-950", text: "text-blue-700 dark:text-blue-300", fill: "#3b82f6" },
	{ bg: "bg-amber-100 dark:bg-amber-950", text: "text-amber-700 dark:text-amber-300", fill: "#f59e0b" },
	{ bg: "bg-purple-100 dark:bg-purple-950", text: "text-purple-700 dark:text-purple-300", fill: "#8b5cf6" },
	{ bg: "bg-teal-100 dark:bg-teal-950", text: "text-teal-700 dark:text-teal-300", fill: "#14b8a6" },
	{ bg: "bg-pink-100 dark:bg-pink-950", text: "text-pink-700 dark:text-pink-300", fill: "#ec4899" },
	{ bg: "bg-indigo-100 dark:bg-indigo-950", text: "text-indigo-700 dark:text-indigo-300", fill: "#6366f1" },
	{ bg: "bg-orange-100 dark:bg-orange-950", text: "text-orange-700 dark:text-orange-300", fill: "#f97316" },
	{ bg: "bg-emerald-100 dark:bg-emerald-950", text: "text-emerald-700 dark:text-emerald-300", fill: "#10b981" },
];

export function MessageBusPanel({ events, agents, onNavigateToAgent }: MessageBusPanelProps) {
	const agentMap = new Map(agents.map((a) => [a.agent_name, a.span_id]));

	const publishedEvents = events.filter((e) => e.event_type === "multi_agent.bus.published");
	const deliveredEvents = events.filter((e) => e.event_type === "multi_agent.bus.delivered");
	const completeEvent = events.find((e) => e.event_type === "multi_agent.bus.complete");

	const cp = completeEvent?.payload as Record<string, unknown> | undefined;
	const totalMessages = cp?.total_messages as number | undefined;
	const maxDepthReached = cp?.max_depth_reached as number | undefined;
	const terminationReason = cp?.termination_reason as string | undefined;
	const agentExecutionCounts = (cp?.agent_execution_counts as Record<string, number>) ?? {};

	// Parse published events
	const messages: PublishedMessage[] = useMemo(
		() =>
			publishedEvents.map((e) => {
				const p = e.payload as Record<string, unknown>;
				return {
					messageId: (p.message_id as string) ?? "",
					topic: (p.topic as string) ?? "",
					author: (p.author as string) ?? "",
					contentPreview: (p.content as string) ?? "",
					depth: (p.depth as number) ?? 0,
					parentMessageId: (p.parent_message_id as string | null) ?? null,
				};
			}),
		[publishedEvents],
	);

	// Parse delivered events
	const deliveries: DeliveryInfo[] = useMemo(
		() =>
			deliveredEvents.map((e) => {
				const p = e.payload as Record<string, unknown>;
				return {
					messageId: (p.message_id as string) ?? "",
					agentName: (p.agent_name as string) ?? "",
					outputPreview: (p.output as string) ?? "",
					messagesPublished: (p.messages_published as number) ?? 0,
					error: (p.error as string | null) ?? null,
				};
			}),
		[deliveredEvents],
	);

	// Build topic → color mapping
	const topicColorMap = useMemo(() => {
		const topics = [...new Set(messages.map((m) => m.topic))];
		const map = new Map<string, (typeof TOPIC_COLORS)[0]>();
		topics.forEach((t, i) => {
			map.set(t, TOPIC_COLORS[i % TOPIC_COLORS.length]);
		});
		return map;
	}, [messages]);

	const uniqueTopics = [...new Set(messages.map((m) => m.topic))];
	const displayTotalMessages = totalMessages ?? messages.length;
	const displayMaxDepth = maxDepthReached ?? Math.max(0, ...messages.map((m) => m.depth));

	const useDAG = publishedEvents.length <= DAG_THRESHOLD;

	return (
		<div className="space-y-3" data-testid="message-bus-panel">
			{/* Header */}
			<div className="flex items-center gap-2 flex-wrap text-xs text-muted-foreground">
				<span>
					{uniqueTopics.length} topic{uniqueTopics.length !== 1 ? "s" : ""}
				</span>
				<span>
					· {displayTotalMessages} message{displayTotalMessages !== 1 ? "s" : ""}
				</span>
				<span>· max depth {displayMaxDepth}</span>
				{terminationReason && <span>· {terminationReason}</span>}
			</div>

			{/* Main content: DAG or summary */}
			{useDAG ? (
				<MessageDAGView
					messages={messages}
					deliveries={deliveries}
					topicColorMap={topicColorMap}
					agentMap={agentMap}
					onNavigateToAgent={onNavigateToAgent}
				/>
			) : (
				<MessageSummaryView messages={messages} topicColorMap={topicColorMap} />
			)}

			{/* Agent activity table */}
			<AgentActivityTable
				agentExecutionCounts={agentExecutionCounts}
				messages={messages}
				agentMap={agentMap}
				onNavigateToAgent={onNavigateToAgent}
			/>
		</div>
	);
}

// ---------------------------------------------------------------------------
// DAG View — for ≤40 published messages
// ---------------------------------------------------------------------------

function MessageDAGView({
	messages,
	deliveries,
	topicColorMap,
	agentMap,
	onNavigateToAgent,
}: {
	messages: PublishedMessage[];
	deliveries: DeliveryInfo[];
	topicColorMap: Map<string, (typeof TOPIC_COLORS)[0]>;
	agentMap: Map<string, string>;
	onNavigateToAgent: (spanId: string) => void;
}) {
	const [expandedNode, setExpandedNode] = useState<string | null>(null);

	// Compute dagre layout
	const { nodes, edges } = useMemo(() => computeMessageDAGLayout(messages), [messages]);

	// Content bounds
	const contentBounds = useMemo<ContentBounds | null>(() => {
		if (nodes.length === 0) return null;
		let minX = Infinity,
			maxX = -Infinity,
			minY = Infinity,
			maxY = -Infinity;
		for (const n of nodes) {
			const x = n.x ?? 0;
			const y = n.y ?? 0;
			minX = Math.min(minX, x - NODE_WIDTH / 2);
			maxX = Math.max(maxX, x + NODE_WIDTH / 2);
			minY = Math.min(minY, y - NODE_HEIGHT / 2);
			maxY = Math.max(maxY, y + NODE_HEIGHT / 2);
		}
		return { minX, minY, maxX, maxY };
	}, [nodes]);

	const { svgRef, viewBox, isDragging, zoomIn, zoomOut, fitToContent, svgHandlers } = useSVGViewport(contentBounds);

	const nodeMap = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes]);

	const deliveriesByMessage = useMemo(() => {
		const map = new Map<string, DeliveryInfo[]>();
		for (const d of deliveries) {
			const list = map.get(d.messageId) ?? [];
			list.push(d);
			map.set(d.messageId, list);
		}
		return map;
	}, [deliveries]);

	const handleNodeClick = useCallback((nodeId: string) => {
		setExpandedNode((prev) => (prev === nodeId ? null : nodeId));
	}, []);

	if (nodes.length === 0) {
		return <div className="text-xs text-muted-foreground">No messages published.</div>;
	}

	return (
		<div className="space-y-2">
			<div
				className="relative w-full overflow-hidden select-none border border-border rounded-md"
				style={{ minHeight: 300 }}
				data-testid="message-dag"
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
						<marker id="bus-arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
							<polygon points="0 0, 8 3, 0 6" className="fill-muted-foreground/40" />
						</marker>
					</defs>

					{/* Edges */}
					<g data-testid="bus-edges">
						{edges.map(({ source, target }) => {
							const sNode = nodeMap.get(source);
							const tNode = nodeMap.get(target);
							if (!sNode || !tNode) return null;

							const sx = sNode.x ?? 0;
							const sy = (sNode.y ?? 0) + NODE_HEIGHT / 2;
							const tx = tNode.x ?? 0;
							const ty = (tNode.y ?? 0) - NODE_HEIGHT / 2;
							const midY = (sy + ty) / 2;
							const path = `M ${sx} ${sy} C ${sx} ${midY}, ${tx} ${midY}, ${tx} ${ty}`;

							return (
								<path
									key={`${source}-${target}`}
									d={path}
									fill="none"
									className="stroke-muted-foreground/40 stroke-[1.5]"
									markerEnd="url(#bus-arrowhead)"
									data-testid={`bus-edge-${source}-${target}`}
								/>
							);
						})}
					</g>

					{/* Nodes */}
					<g data-testid="bus-nodes">
						{nodes.map((node) => {
							const x = (node.x ?? 0) - NODE_WIDTH / 2;
							const y = (node.y ?? 0) - NODE_HEIGHT / 2;
							const topicColor = topicColorMap.get(node.topic);
							const isExpanded = expandedNode === node.id;

							return (
								<g
									key={node.id}
									transform={`translate(${x}, ${y})`}
									data-dag-node
									data-testid={`bus-node-${node.id}`}
									className="cursor-pointer"
									onClick={(e) => {
										e.stopPropagation();
										handleNodeClick(node.id);
									}}
								>
									<foreignObject width={NODE_WIDTH} height={isExpanded ? "auto" : NODE_HEIGHT}>
										<div
											className={`w-full border border-border rounded-md p-1.5 bg-background text-xs ${
												isExpanded ? "" : "h-full"
											}`}
										>
											<div className="flex items-center gap-1 mb-0.5">
												{topicColor && (
													<span className={`text-[9px] px-1 py-0 rounded ${topicColor.bg} ${topicColor.text}`}>
														{node.topic}
													</span>
												)}
												<span className="text-muted-foreground text-[9px]">d{node.depth}</span>
											</div>
											<div className="text-[10px] text-muted-foreground truncate">{node.author}</div>
											<div className="text-[10px] truncate">{node.contentPreview}</div>
										</div>
									</foreignObject>
								</g>
							);
						})}
					</g>
				</svg>
			</div>

			{/* Expanded node details */}
			{expandedNode && (
				<ExpandedMessageDetails
					messageId={expandedNode}
					messages={messages}
					deliveries={deliveriesByMessage.get(expandedNode) ?? []}
					topicColorMap={topicColorMap}
					agentMap={agentMap}
					onNavigateToAgent={onNavigateToAgent}
					onClose={() => setExpandedNode(null)}
				/>
			)}
		</div>
	);
}

function ExpandedMessageDetails({
	messageId,
	messages,
	deliveries,
	topicColorMap,
	agentMap,
	onNavigateToAgent,
	onClose,
}: {
	messageId: string;
	messages: PublishedMessage[];
	deliveries: DeliveryInfo[];
	topicColorMap: Map<string, (typeof TOPIC_COLORS)[0]>;
	agentMap: Map<string, string>;
	onNavigateToAgent: (spanId: string) => void;
	onClose: () => void;
}) {
	const msg = messages.find((m) => m.messageId === messageId);
	if (!msg) return null;

	const topicColor = topicColorMap.get(msg.topic);

	return (
		<div className="border border-border rounded-md p-2 space-y-2 text-xs" data-testid="expanded-message-details">
			<div className="flex items-center justify-between">
				<div className="flex items-center gap-1.5">
					{topicColor && (
						<span className={`text-[9px] px-1 py-0 rounded ${topicColor.bg} ${topicColor.text}`}>{msg.topic}</span>
					)}
					<span className="text-muted-foreground">depth {msg.depth}</span>
					<span className="text-muted-foreground">·</span>
					<AgentLink name={msg.author} agentMap={agentMap} onNavigateToAgent={onNavigateToAgent} />
				</div>
				<button type="button" onClick={onClose} className="text-muted-foreground hover:text-foreground">
					✕
				</button>
			</div>
			<div>{msg.contentPreview}</div>
			{deliveries.length > 0 && (
				<div className="space-y-1">
					<span className="text-muted-foreground">
						Delivered to {deliveries.length} agent{deliveries.length !== 1 ? "s" : ""}
					</span>
					{deliveries.map((d) => (
						<div key={d.agentName} className="flex items-start gap-1 pl-2">
							<AgentLink name={d.agentName} agentMap={agentMap} onNavigateToAgent={onNavigateToAgent} />
							{d.error ? (
								<span className="text-destructive">✗ {d.error}</span>
							) : (
								<span className="text-muted-foreground truncate">{d.outputPreview}</span>
							)}
						</div>
					))}
				</div>
			)}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Summary View — for >40 published messages
// ---------------------------------------------------------------------------

function MessageSummaryView({
	messages,
	topicColorMap,
}: {
	messages: PublishedMessage[];
	topicColorMap: Map<string, (typeof TOPIC_COLORS)[0]>;
}) {
	// Topic breakdown
	const topicBreakdown = useMemo(() => {
		const map = new Map<string, { count: number; maxDepth: number }>();
		for (const m of messages) {
			const existing = map.get(m.topic);
			if (existing) {
				existing.count++;
				existing.maxDepth = Math.max(existing.maxDepth, m.depth);
			} else {
				map.set(m.topic, { count: 1, maxDepth: m.depth });
			}
		}
		return [...map.entries()].map(([topic, stats]) => ({ topic, ...stats })).sort((a, b) => b.count - a.count);
	}, [messages]);

	return (
		<div className="space-y-3" data-testid="message-summary">
			{/* Topic breakdown table */}
			<div className="space-y-1">
				<span className="text-xs text-muted-foreground">Topic breakdown</span>
				<div className="border border-border rounded-md overflow-hidden">
					<table className="w-full text-xs" data-testid="topic-breakdown">
						<thead>
							<tr className="bg-muted/50">
								<th className="text-left px-2 py-1 font-medium text-muted-foreground">Topic</th>
								<th className="text-left px-2 py-1 font-medium text-muted-foreground">Messages</th>
								<th className="text-left px-2 py-1 font-medium text-muted-foreground">Max Depth</th>
							</tr>
						</thead>
						<tbody>
							{topicBreakdown.map((row) => {
								const color = topicColorMap.get(row.topic);
								return (
									<tr key={row.topic} className="border-t border-border">
										<td className="px-2 py-1">
											{color ? (
												<span className={`text-[10px] px-1.5 py-0.5 rounded ${color.bg} ${color.text}`}>
													{row.topic}
												</span>
											) : (
												row.topic
											)}
										</td>
										<td className="px-2 py-1 tabular-nums">{row.count}</td>
										<td className="px-2 py-1 tabular-nums">{row.maxDepth}</td>
									</tr>
								);
							})}
						</tbody>
					</table>
				</div>
			</div>
		</div>
	);
}

// ---------------------------------------------------------------------------
// Agent Activity Table — shared between both views
// ---------------------------------------------------------------------------

function AgentActivityTable({
	agentExecutionCounts,
	messages,
	agentMap,
	onNavigateToAgent,
}: {
	agentExecutionCounts: Record<string, number>;
	messages: PublishedMessage[];
	agentMap: Map<string, string>;
	onNavigateToAgent: (spanId: string) => void;
}) {
	const rows = useMemo(() => {
		// Count messages published per agent
		const msgCounts = new Map<string, number>();
		for (const m of messages) {
			msgCounts.set(m.author, (msgCounts.get(m.author) ?? 0) + 1);
		}

		// Merge agent names from both sources
		const allNames = new Set([...Object.keys(agentExecutionCounts), ...msgCounts.keys()]);

		return [...allNames]
			.map((name) => ({
				name,
				executions: agentExecutionCounts[name] ?? 0,
				messagesPublished: msgCounts.get(name) ?? 0,
			}))
			.sort((a, b) => b.executions - a.executions || b.messagesPublished - a.messagesPublished);
	}, [agentExecutionCounts, messages]);

	if (rows.length === 0) return null;

	return (
		<div className="space-y-1">
			<span className="text-xs text-muted-foreground">Agent activity</span>
			<div className="border border-border rounded-md overflow-hidden">
				<table className="w-full text-xs" data-testid="agent-activity">
					<thead>
						<tr className="bg-muted/50">
							<th className="text-left px-2 py-1 font-medium text-muted-foreground">Agent</th>
							<th className="text-left px-2 py-1 font-medium text-muted-foreground">Executions</th>
							<th className="text-left px-2 py-1 font-medium text-muted-foreground">Published</th>
						</tr>
					</thead>
					<tbody>
						{rows.map((row) => (
							<tr key={row.name} className="border-t border-border">
								<td className="px-2 py-1">
									<AgentLink name={row.name} agentMap={agentMap} onNavigateToAgent={onNavigateToAgent} />
								</td>
								<td className="px-2 py-1 tabular-nums">{row.executions}</td>
								<td className="px-2 py-1 tabular-nums">{row.messagesPublished}</td>
							</tr>
						))}
					</tbody>
				</table>
			</div>
		</div>
	);
}

// ---------------------------------------------------------------------------
// Layout computation
// ---------------------------------------------------------------------------

function computeMessageDAGLayout(messages: PublishedMessage[]): {
	nodes: MessageNode[];
	edges: { source: string; target: string }[];
} {
	if (messages.length === 0) return { nodes: [], edges: [] };

	const g = new dagre.graphlib.Graph();
	g.setGraph({
		rankdir: "TB",
		ranksep: 60,
		nodesep: 30,
	});
	g.setDefaultEdgeLabel(() => ({}));

	const nodes: MessageNode[] = messages.map((m) => ({
		id: m.messageId,
		topic: m.topic,
		author: m.author,
		contentPreview: m.contentPreview,
		depth: m.depth,
	}));

	const messageIds = new Set(messages.map((m) => m.messageId));

	for (const node of nodes) {
		g.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
	}

	const edges: { source: string; target: string }[] = [];
	for (const m of messages) {
		if (m.parentMessageId && messageIds.has(m.parentMessageId)) {
			edges.push({ source: m.parentMessageId, target: m.messageId });
			g.setEdge(m.parentMessageId, m.messageId);
		}
	}

	dagre.layout(g);

	for (const node of nodes) {
		const dagreNode = g.node(node.id);
		if (dagreNode) {
			node.x = dagreNode.x;
			node.y = dagreNode.y;
		}
	}

	return { nodes, edges };
}
