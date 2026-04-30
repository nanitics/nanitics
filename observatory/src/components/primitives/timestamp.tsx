const SECOND = 1000;
const MINUTE = 60 * SECOND;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

function formatRelative(date: Date): string {
	const diff = Date.now() - date.getTime();
	if (diff < MINUTE) return "just now";
	if (diff < HOUR) return `${Math.floor(diff / MINUTE)}m ago`;
	if (diff < DAY) return `${Math.floor(diff / HOUR)}h ago`;
	return `${Math.floor(diff / DAY)}d ago`;
}

export function Timestamp({ value }: { value: string }) {
	const date = new Date(value);
	return (
		<span className="text-xs text-muted-foreground tabular-nums" title={date.toISOString()}>
			{formatRelative(date)}
		</span>
	);
}
