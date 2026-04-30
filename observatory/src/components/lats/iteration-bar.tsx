import { FastForward, Pause, Play, Rewind, SkipBack, SkipForward } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import type { IterationData } from "../../hooks/use-lats-data";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type IterationMode = "highlight" | "replay";
export type PlaybackSpeed = 0.5 | 1 | 2;

export interface IterationBarProps {
	iterations: IterationData[];
	/** Currently selected iteration number (null = default/solution view). */
	selectedIteration: number | null;
	onSelectIteration: (iterationNumber: number | null) => void;
	/** Current interaction mode. */
	mode: IterationMode;
	onModeChange: (mode: IterationMode) => void;
}

// ---------------------------------------------------------------------------
// Speed label map
// ---------------------------------------------------------------------------

const SPEED_OPTIONS: PlaybackSpeed[] = [0.5, 1, 2];
const SPEED_LABELS: Record<PlaybackSpeed, string> = {
	0.5: "0.5×",
	1: "1×",
	2: "2×",
};

// ---------------------------------------------------------------------------
// IterationBar (with replay stepper)
// ---------------------------------------------------------------------------

export function IterationBar({
	iterations,
	selectedIteration,
	onSelectIteration,
	mode,
	onModeChange,
}: IterationBarProps) {
	const [isPlaying, setIsPlaying] = useState(false);
	const [speed, setSpeed] = useState<PlaybackSpeed>(1);
	const playTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

	const iterationCount = iterations.length;

	const minIter = iterationCount > 0 ? iterations[0].iterationNumber : 0;
	const maxIter = iterationCount > 0 ? iterations[iterations.length - 1].iterationNumber : 0;

	// Compute sparkline values
	const values = iterations.map((it) => it.bestValueSoFar);
	const maxVal = Math.max(...values, 0.01);

	// ---------------------------------------------------------------------------
	// VCR controls
	// ---------------------------------------------------------------------------

	const stepForward = useCallback(() => {
		const currentIdx = iterations.findIndex((it) => it.iterationNumber === selectedIteration);
		if (currentIdx < 0) {
			onSelectIteration(minIter);
		} else if (currentIdx < iterationCount - 1) {
			onSelectIteration(iterations[currentIdx + 1].iterationNumber);
		} else {
			setIsPlaying(false);
		}
	}, [selectedIteration, iterations, iterationCount, minIter, onSelectIteration]);

	const stepBack = useCallback(() => {
		const currentIdx = iterations.findIndex((it) => it.iterationNumber === selectedIteration);
		if (currentIdx <= 0) {
			onSelectIteration(minIter);
		} else {
			onSelectIteration(iterations[currentIdx - 1].iterationNumber);
		}
	}, [selectedIteration, iterations, minIter, onSelectIteration]);

	const jumpToStart = useCallback(() => {
		setIsPlaying(false);
		onSelectIteration(minIter);
	}, [minIter, onSelectIteration]);

	const jumpToEnd = useCallback(() => {
		setIsPlaying(false);
		onSelectIteration(maxIter);
	}, [maxIter, onSelectIteration]);

	const togglePlay = useCallback(() => {
		setIsPlaying((prev) => {
			if (!prev) {
				if (selectedIteration === maxIter || selectedIteration == null) {
					onSelectIteration(minIter);
				}
			}
			return !prev;
		});
	}, [selectedIteration, maxIter, minIter, onSelectIteration]);

	// Auto-advance timer
	useEffect(() => {
		if (isPlaying && mode === "replay") {
			const intervalMs = 1500 / speed;
			playTimerRef.current = setInterval(() => {
				stepForward();
			}, intervalMs);
		}
		return () => {
			if (playTimerRef.current) {
				clearInterval(playTimerRef.current);
				playTimerRef.current = null;
			}
		};
	}, [isPlaying, mode, speed, stepForward]);

	// Stop playing when reaching end
	useEffect(() => {
		if (selectedIteration === maxIter && isPlaying) {
			setIsPlaying(false);
		}
	}, [selectedIteration, maxIter, isPlaying]);

	// Stop playing when mode changes away from replay
	useEffect(() => {
		if (mode !== "replay") {
			setIsPlaying(false);
		}
	}, [mode]);

	// Slider change handler
	const handleSliderChange = useCallback(
		(e: React.ChangeEvent<HTMLInputElement>) => {
			const value = parseInt(e.target.value, 10);
			const iter = iterations.find((it) => it.iterationNumber === value);
			if (iter) {
				onSelectIteration(iter.iterationNumber);
			}
		},
		[iterations, onSelectIteration],
	);

	if (iterationCount === 0) return null;

	return (
		<div className="px-4 py-2 border-b border-border shrink-0" data-testid="iteration-bar">
			{/* Top row: mode toggle + sparkline */}
			<div className="flex items-center gap-3 mb-1.5">
				{/* Mode toggle */}
				<div className="flex rounded-md border border-border overflow-hidden shrink-0" data-testid="mode-toggle">
					<button
						type="button"
						onClick={() => onModeChange("highlight")}
						className={`text-[10px] px-2 py-0.5 transition-colors ${
							mode === "highlight"
								? "bg-primary text-primary-foreground"
								: "bg-muted/50 text-muted-foreground hover:bg-muted"
						}`}
						aria-label="Highlight mode"
					>
						Highlight
					</button>
					<button
						type="button"
						onClick={() => onModeChange("replay")}
						className={`text-[10px] px-2 py-0.5 transition-colors ${
							mode === "replay"
								? "bg-primary text-primary-foreground"
								: "bg-muted/50 text-muted-foreground hover:bg-muted"
						}`}
						aria-label="Replay mode"
					>
						Replay
					</button>
				</div>

				{/* Sparkline */}
				<div className="flex items-end gap-px h-6 flex-1">
					{iterations.map((it) => {
						const heightPct = (it.bestValueSoFar / maxVal) * 100;
						const isSelected = selectedIteration === it.iterationNumber;
						return (
							<button
								type="button"
								key={it.iterationNumber}
								onClick={() => onSelectIteration(isSelected ? null : it.iterationNumber)}
								className={`flex-1 min-w-[4px] rounded-t-sm transition-colors ${
									isSelected ? "bg-primary" : "bg-muted hover:bg-muted-foreground/30"
								}`}
								style={{ height: `${Math.max(heightPct, 8)}%` }}
								title={`Iteration ${it.iterationNumber}: best=${it.bestValueSoFar.toFixed(2)}`}
								aria-label={`Iteration ${it.iterationNumber}`}
							/>
						);
					})}
				</div>
			</div>

			{/* Iteration number indicators */}
			<div className="flex items-center gap-1 mb-1.5">
				<div className="flex items-center gap-px flex-1">
					{iterations.map((it) => {
						const isSelected = selectedIteration === it.iterationNumber;
						return (
							<button
								type="button"
								key={it.iterationNumber}
								onClick={() => onSelectIteration(isSelected ? null : it.iterationNumber)}
								className={`flex-1 min-w-[12px] h-5 text-[9px] font-mono tabular-nums rounded transition-colors ${
									isSelected ? "bg-primary text-primary-foreground" : "bg-muted/50 text-muted-foreground hover:bg-muted"
								}`}
							>
								{it.iterationNumber}
							</button>
						);
					})}
				</div>

				{/* Clear button */}
				{selectedIteration != null && (
					<button
						type="button"
						onClick={() => onSelectIteration(null)}
						className="text-[10px] text-muted-foreground hover:text-foreground px-1.5 py-0.5 rounded hover:bg-accent transition-colors shrink-0"
						aria-label="Clear iteration selection"
					>
						Clear
					</button>
				)}
			</div>

			{/* VCR controls — only in replay mode */}
			{mode === "replay" && (
				<div className="flex items-center gap-2 pt-1 border-t border-border/50" data-testid="vcr-controls">
					{/* Jump to start */}
					<button
						type="button"
						onClick={jumpToStart}
						className="text-xs text-muted-foreground hover:text-foreground p-1 rounded hover:bg-accent transition-colors"
						aria-label="Jump to start"
						title="Jump to start"
					>
						<SkipBack aria-hidden="true" className="h-4 w-4" />
					</button>

					{/* Step back */}
					<button
						type="button"
						onClick={stepBack}
						className="text-xs text-muted-foreground hover:text-foreground p-1 rounded hover:bg-accent transition-colors"
						aria-label="Step back"
						title="Step back"
						disabled={selectedIteration === minIter}
					>
						<Rewind aria-hidden="true" className="h-4 w-4" />
					</button>

					{/* Play/Pause */}
					<button
						type="button"
						onClick={togglePlay}
						className="text-sm text-muted-foreground hover:text-foreground p-1 rounded hover:bg-accent transition-colors"
						aria-label={isPlaying ? "Pause" : "Play"}
						title={isPlaying ? "Pause" : "Play"}
						data-testid="play-pause-button"
					>
						{isPlaying ? (
							<Pause aria-hidden="true" className="h-4 w-4" />
						) : (
							<Play aria-hidden="true" className="h-4 w-4" />
						)}
					</button>

					{/* Step forward */}
					<button
						type="button"
						onClick={stepForward}
						className="text-xs text-muted-foreground hover:text-foreground p-1 rounded hover:bg-accent transition-colors"
						aria-label="Step forward"
						title="Step forward"
						disabled={selectedIteration === maxIter}
					>
						<FastForward aria-hidden="true" className="h-4 w-4" />
					</button>

					{/* Jump to end */}
					<button
						type="button"
						onClick={jumpToEnd}
						className="text-xs text-muted-foreground hover:text-foreground p-1 rounded hover:bg-accent transition-colors"
						aria-label="Jump to end"
						title="Jump to end"
					>
						<SkipForward aria-hidden="true" className="h-4 w-4" />
					</button>

					{/* Slider */}
					<input
						type="range"
						min={minIter}
						max={maxIter}
						value={selectedIteration ?? minIter}
						onChange={handleSliderChange}
						className="flex-1 h-1.5 accent-primary cursor-pointer"
						aria-label="Iteration slider"
						data-testid="iteration-slider"
					/>

					{/* Iteration counter */}
					<span className="text-[10px] font-mono tabular-nums text-muted-foreground shrink-0 w-12 text-center">
						{selectedIteration ?? "–"}/{maxIter}
					</span>

					{/* Speed selector */}
					<div className="flex rounded-md border border-border overflow-hidden shrink-0" data-testid="speed-selector">
						{SPEED_OPTIONS.map((s) => (
							<button
								type="button"
								key={s}
								onClick={() => setSpeed(s)}
								className={`text-[10px] px-1.5 py-0.5 transition-colors ${
									speed === s
										? "bg-primary text-primary-foreground"
										: "bg-muted/50 text-muted-foreground hover:bg-muted"
								}`}
								aria-label={`Speed ${SPEED_LABELS[s]}`}
							>
								{SPEED_LABELS[s]}
							</button>
						))}
					</div>
				</div>
			)}
		</div>
	);
}
