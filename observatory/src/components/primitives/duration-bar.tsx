/**
 * Duration bar — shows relative duration as a horizontal bar.
 * `value` is the duration in ms, `max` is the reference maximum for proportion.
 */
export function DurationBar({ value, max }: { value: number | null; max: number }) {
	if (value == null || max <= 0) return null;
	const pct = Math.min((value / max) * 100, 100);
	return (
		<div className="flex items-center gap-2 min-w-[80px]">
			<div className="h-1.5 flex-1 rounded-full bg-muted overflow-hidden">
				<div className="h-full rounded-full bg-primary/40" style={{ width: `${pct}%` }} />
			</div>
			<span className="text-[10px] text-muted-foreground tabular-nums whitespace-nowrap">
				{value < 1000 ? `${Math.round(value)}ms` : `${(value / 1000).toFixed(1)}s`}
			</span>
		</div>
	);
}
