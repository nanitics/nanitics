import { EVENT_TYPE_CATEGORIES, type EventTypeCategory } from "../../hooks/use-filters";

interface EventTypeFilterProps {
	enabled: Set<EventTypeCategory>;
	onToggle: (category: EventTypeCategory) => void;
	onClear: () => void;
}

export function EventTypeFilter({ enabled, onToggle, onClear }: EventTypeFilterProps) {
	const hasFilters = enabled.size > 0;

	return (
		<div role="toolbar" aria-label="Event type filters" className="flex items-center gap-1.5 flex-wrap">
			{hasFilters && (
				<button
					type="button"
					onClick={onClear}
					className="text-[10px] px-2 py-1 rounded-full border text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
				>
					Clear
				</button>
			)}
			{EVENT_TYPE_CATEGORIES.map((category) => (
				<button
					type="button"
					key={category}
					aria-pressed={enabled.has(category)}
					onClick={() => onToggle(category)}
					className={`text-[10px] px-2 py-1 rounded-full border transition-colors ${
						enabled.has(category)
							? "bg-primary/10 text-primary border-primary/30"
							: "text-muted-foreground hover:text-foreground hover:bg-accent"
					}`}
				>
					{category}
				</button>
			))}
		</div>
	);
}
