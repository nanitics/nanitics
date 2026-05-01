import type { TraceLevel } from "../../types";

const LEVELS: { value: TraceLevel; label: string }[] = [
	{ value: "info", label: "Info" },
	{ value: "debug", label: "Debug" },
	{ value: "verbose", label: "Verbose" },
];

interface LevelSelectorProps {
	value: TraceLevel;
	onChange: (level: TraceLevel) => void;
}

export function LevelSelector({ value, onChange }: LevelSelectorProps) {
	return (
		<div role="toolbar" aria-label="Trace level" className="flex items-center rounded-md border overflow-hidden">
			{LEVELS.map((level) => (
				<button
					type="button"
					key={level.value}
					aria-pressed={value === level.value}
					onClick={() => onChange(level.value)}
					className={`text-xs px-3 py-1.5 transition-colors ${
						value === level.value
							? "bg-primary text-primary-foreground"
							: "text-muted-foreground hover:text-foreground hover:bg-accent"
					}`}
				>
					{level.label}
				</button>
			))}
		</div>
	);
}
