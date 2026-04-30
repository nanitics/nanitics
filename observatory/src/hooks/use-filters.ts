import { useCallback, useMemo, useState } from "react";
import type { TraceLevel } from "../types";

/** Event type categories for filtering. */
export const EVENT_TYPE_CATEGORIES = [
	"Agent",
	"LLM",
	"Tool",
	"Memory",
	"Planning",
	"Error",
	"Workflow",
	"HITL",
	"Context",
	"Span",
] as const;

export type EventTypeCategory = (typeof EVENT_TYPE_CATEGORIES)[number];

/** Maps a category to event type prefixes. */
const CATEGORY_PREFIXES: Record<EventTypeCategory, string[]> = {
	Agent: ["agent."],
	LLM: ["llm."],
	Tool: ["tool."],
	Memory: ["memory."],
	Planning: ["planning."],
	Error: ["error.", "correction."],
	Workflow: ["workflow."],
	HITL: ["hitl."],
	Context: ["context."],
	Span: ["span."],
};

/** Check if an event type matches any of the enabled categories. */
export function matchesEventTypeFilter(eventType: string, enabledCategories: Set<EventTypeCategory>): boolean {
	if (enabledCategories.size === 0) return true;
	for (const category of enabledCategories) {
		const prefixes = CATEGORY_PREFIXES[category];
		if (prefixes.some((p) => eventType.startsWith(p))) return true;
	}
	return false;
}

interface UseFiltersResult {
	level: TraceLevel;
	setLevel: (level: TraceLevel) => void;
	eventTypes: Set<EventTypeCategory>;
	toggleEventType: (category: EventTypeCategory) => void;
	clearEventTypes: () => void;
}

export function useFilters(): UseFiltersResult {
	const [level, setLevel] = useState<TraceLevel>("info");
	const [eventTypes, setEventTypes] = useState<Set<EventTypeCategory>>(new Set());

	const toggleEventType = useCallback((category: EventTypeCategory) => {
		setEventTypes((prev) => {
			const next = new Set(prev);
			if (next.has(category)) {
				next.delete(category);
			} else {
				next.add(category);
			}
			return next;
		});
	}, []);

	const clearEventTypes = useCallback(() => {
		setEventTypes(new Set());
	}, []);

	return useMemo(
		() => ({ level, setLevel, eventTypes, toggleEventType, clearEventTypes }),
		[level, eventTypes, toggleEventType, clearEventTypes],
	);
}
