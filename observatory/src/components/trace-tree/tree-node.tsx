import { useObservatory } from "../../context/observatory-context";
import type { SpanTreeNode, TraceEvent } from "../../types";
import { DurationBar } from "../primitives/duration-bar";
import { EventIcon } from "../primitives/event-icon";

/** Derive a one-line summary from a span node. */
function spanSummary(node: SpanTreeNode): string {
	if (node.summary.agent_name) {
		const type = node.summary.agent_type ? ` (${node.summary.agent_type})` : "";
		return `${node.summary.agent_name}${type}`;
	}
	return node.name || node.span_id;
}

/** Count total descendants (spans + events) in a collapsed subtree. */
function countDescendants(node: SpanTreeNode): number {
	let count = node.events.length;
	for (const child of node.children) {
		count += 1 + countDescendants(child);
	}
	return count;
}

interface TreeNodeProps {
	node: SpanTreeNode;
	depth: number;
	maxDuration: number;
	isExpanded: boolean;
	selectedEvent: TraceEvent | null;
	onToggle: (spanId: string) => void;
	onSelectEvent: (event: TraceEvent) => void;
	eventFilter?: (event: TraceEvent) => boolean;
	onNavigateToAgent?: (spanId: string) => void;
	renderChildren: (node: SpanTreeNode, depth: number) => React.ReactNode;
}

export function TreeNode({
	node,
	depth,
	maxDuration,
	isExpanded,
	selectedEvent,
	onToggle,
	onSelectEvent,
	eventFilter,
	onNavigateToAgent,
	renderChildren,
}: TreeNodeProps) {
	const { registry } = useObservatory();
	const filteredEvents = eventFilter ? node.events.filter(eventFilter) : node.events;
	const hasChildren = node.children.length > 0 || filteredEvents.length > 0;
	const descendantCount = !isExpanded && hasChildren ? countDescendants(node) : 0;

	// Determine the "primary" event type for the icon
	const primaryType =
		node.events.length > 0 ? node.events[0].event_type : node.children.length > 0 ? "span.group" : "span.leaf";

	return (
		<div>
			{/* Span row */}
			<div
				className="group flex items-center gap-1.5 py-1 px-2 rounded-md hover:bg-accent/50 cursor-pointer transition-colors"
				style={{ paddingLeft: `${depth * 20 + 8}px` }}
				onClick={() => hasChildren && onToggle(node.span_id)}
			>
				{/* Expand/collapse toggle */}
				<span className="w-4 text-center text-xs text-muted-foreground flex-shrink-0">
					{hasChildren ? (isExpanded ? "▾" : "▸") : " "}
				</span>

				{/* Icon */}
				<span className="text-sm flex-shrink-0">
					<EventIcon eventType={primaryType} />
				</span>

				{/* Summary */}
				<span className="text-sm truncate flex-1 min-w-0">{spanSummary(node)}</span>

				{/* Collapsed count */}
				{!isExpanded && descendantCount > 0 && (
					<span className="text-[10px] text-muted-foreground bg-muted rounded-full px-1.5 py-0.5 flex-shrink-0">
						{descendantCount}
					</span>
				)}

				{/* Error indicator */}
				{node.summary.has_errors && <span className="text-destructive text-xs flex-shrink-0">●</span>}

				{/* Agent detail link */}
				{node.summary.agent_name != null && onNavigateToAgent && (
					<button
						type="button"
						className="text-xs text-primary hover:text-primary/80 flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity"
						onClick={(e) => {
							e.stopPropagation();
							onNavigateToAgent(node.span_id);
						}}
						title="View agent details"
					>
						→
					</button>
				)}

				{/* Duration */}
				<div className="flex-shrink-0 w-[100px]">
					<DurationBar value={node.summary.duration_ms} max={maxDuration} />
				</div>
			</div>

			{/* Expanded children: events first, then child spans */}
			{isExpanded && (
				<>
					{filteredEvents.map((event) => (
						<div
							key={event.id}
							className={`flex items-center gap-1.5 py-1 px-2 rounded-md cursor-pointer transition-colors ${
								selectedEvent?.id === event.id ? "bg-primary/10 text-primary" : "hover:bg-accent/50"
							}`}
							style={{ paddingLeft: `${(depth + 1) * 20 + 8}px` }}
							onClick={() => onSelectEvent(event)}
						>
							<span className="w-4 flex-shrink-0" />
							<span className="text-sm flex-shrink-0">
								<EventIcon eventType={event.event_type} />
							</span>
							<span className="text-sm truncate flex-1 min-w-0 text-muted-foreground">
								{registry.getSummary(event)}
							</span>
							<span className="text-[10px] text-muted-foreground tabular-nums flex-shrink-0">
								{new Date(event.timestamp).toLocaleTimeString()}
							</span>
						</div>
					))}
					{renderChildren(node, depth)}
				</>
			)}
		</div>
	);
}
