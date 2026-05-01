import type { DAGNode } from "../../types/dag-types";
import { DurationBar } from "../primitives/duration-bar";
import { StatusBadge } from "../primitives/status-badge";

interface DAGNodeContentProps {
	node: DAGNode;
	isSelected: boolean;
	maxDurationMs?: number;
	onNavigateToAgent?: (spanId: string) => void;
}

const statusBorderColors: Record<DAGNode["status"], string> = {
	pending: "border-border",
	running: "border-info",
	completed: "border-success",
	error: "border-destructive",
	skipped: "border-muted-foreground/30",
};

export function DAGNodeContent({ node, isSelected, maxDurationMs = 0, onNavigateToAgent }: DAGNodeContentProps) {
	const borderColor = statusBorderColors[node.status] ?? "border-border";
	const selectedRing = isSelected ? "ring-2 ring-primary" : "";
	const errorBg = node.status === "error" ? "bg-destructive/5" : "bg-background";

	return (
		<div
			className={`h-full rounded-lg border ${borderColor} ${selectedRing} ${errorBg} px-3 py-2 flex flex-col justify-between text-xs overflow-hidden`}
			data-testid={`dag-node-content-${node.id}`}
		>
			{/* Top row: name + badges */}
			<div className="flex items-center gap-1.5 min-w-0">
				<span className="font-medium text-foreground truncate flex-1">{node.label}</span>
				<StatusBadge status={node.status} />
			</div>

			{/* Bottom row: agent type + duration */}
			<div className="flex items-center gap-2 mt-1">
				{node.agentType && (
					<span className="inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-medium bg-accent-status-muted text-accent-status-muted-foreground capitalize">
						{node.agentType}
					</span>
				)}
				{node.status === "completed" && node.durationMs != null && (
					<div className="flex-1 min-w-0">
						<DurationBar value={node.durationMs} max={maxDurationMs} />
					</div>
				)}
				{node.agentSpanId && onNavigateToAgent && (
					<button
						type="button"
						onClick={(e) => {
							e.stopPropagation();
							onNavigateToAgent(node.agentSpanId!);
						}}
						className="text-[10px] text-primary hover:underline whitespace-nowrap"
					>
						View agent →
					</button>
				)}
			</div>
		</div>
	);
}
