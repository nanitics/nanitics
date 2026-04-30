import { useObservatory } from "../../context/observatory-context";
import type { TraceEvent } from "../../types";
import { LevelBadge } from "../primitives/level-badge";
import { PayloadViewer } from "./payload-viewer";

export function EventDetailPanel({
	event,
	onNavigateToAgent,
}: {
	event: TraceEvent;
	onNavigateToAgent?: (spanId: string) => void;
}) {
	const { registry } = useObservatory();

	const Renderer = registry.getRenderer(event.event_type);

	return (
		<div className="p-4 space-y-4">
			{/* Metadata header */}
			<div className="space-y-1.5">
				<div className="flex items-center gap-2">
					<span className="text-sm font-medium">{event.event_type}</span>
					<LevelBadge level={event.level} />
				</div>
				<div className="flex items-center gap-2 text-xs text-muted-foreground">
					<span className="font-mono">{event.span_id}</span>
					<span>·</span>
					<span>{new Date(event.timestamp).toLocaleTimeString()}</span>
				</div>
			</div>

			<div className="border-t pt-3">
				{Renderer ? (
					<Renderer event={event} onNavigateToAgent={onNavigateToAgent} />
				) : (
					<PayloadViewer payload={event.payload} />
				)}
			</div>
		</div>
	);
}
