import type { TraceLevel } from "../../types";

const levelStyles: Record<TraceLevel, string> = {
	info: "bg-info-muted text-info-muted-foreground",
	debug: "bg-muted text-muted-foreground",
	verbose: "bg-muted text-muted-foreground opacity-80",
};

export function LevelBadge({ level }: { level: TraceLevel }) {
	return (
		<span className={`inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-medium ${levelStyles[level]}`}>
			{level}
		</span>
	);
}
