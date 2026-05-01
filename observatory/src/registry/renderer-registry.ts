import type { TraceEvent } from "../types";

export interface EventDetailProps {
	event: TraceEvent;
	onNavigateToAgent?: (spanId: string) => void;
}

export interface EventRendererRegistration {
	/** Returns true if this renderer handles the given event type. */
	matches: (eventType: string) => boolean;
	/** Higher priority wins when multiple registrations match (default: 0). */
	priority: number;
	/** React component to render event detail. */
	component: React.ComponentType<EventDetailProps>;
	/** Optional custom one-line summary for tree nodes. */
	summary?: (event: TraceEvent) => string;
}

export class EventRendererRegistry {
	private registrations: EventRendererRegistration[] = [];

	register(registration: EventRendererRegistration): void {
		this.registrations.push(registration);
		// Keep sorted by descending priority for fast lookup
		this.registrations.sort((a, b) => b.priority - a.priority);
	}

	getRenderer(eventType: string): React.ComponentType<EventDetailProps> | null {
		for (const reg of this.registrations) {
			if (reg.matches(eventType)) {
				return reg.component;
			}
		}
		return null;
	}

	getSummary(event: TraceEvent): string {
		for (const reg of this.registrations) {
			if (reg.matches(event.event_type) && reg.summary) {
				return reg.summary(event);
			}
		}
		return event.event_type;
	}
}
