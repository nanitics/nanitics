/**
 * Compact token count display with input/output split.
 */
export function TokenUsage({ inputTokens, outputTokens }: { inputTokens: number; outputTokens: number }) {
	const total = inputTokens + outputTokens;
	return (
		<div className="flex items-center gap-3 text-xs">
			<div className="flex items-center gap-1.5">
				<span className="text-muted-foreground">Total</span>
				<span className="font-mono font-medium tabular-nums">{total.toLocaleString()}</span>
			</div>
			<div className="flex items-center gap-1.5 text-muted-foreground">
				<span>↑</span>
				<span className="font-mono tabular-nums">{inputTokens.toLocaleString()}</span>
			</div>
			<div className="flex items-center gap-1.5 text-muted-foreground">
				<span>↓</span>
				<span className="font-mono tabular-nums">{outputTokens.toLocaleString()}</span>
			</div>
		</div>
	);
}
