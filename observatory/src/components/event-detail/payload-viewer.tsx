import { useCallback, useState } from "react";

interface CollapsibleProps {
	label: string;
	children: React.ReactNode;
	defaultOpen?: boolean;
}

function Collapsible({ label, children, defaultOpen = false }: CollapsibleProps) {
	const [open, setOpen] = useState(defaultOpen);
	return (
		<div>
			<button
				type="button"
				onClick={() => setOpen(!open)}
				className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
			>
				<span>{open ? "▾" : "▸"}</span>
				<span className="font-mono">{label}</span>
			</button>
			{open && <div className="ml-4">{children}</div>}
		</div>
	);
}

function JsonValue({ value, depth }: { value: unknown; depth: number }) {
	if (value === null) {
		return <span className="text-muted-foreground italic">null</span>;
	}

	if (typeof value === "boolean") {
		return <span className="text-blue-600">{String(value)}</span>;
	}

	if (typeof value === "number") {
		return <span className="text-green-700">{value}</span>;
	}

	if (typeof value === "string") {
		// Multi-line strings render as a block
		if (value.includes("\n")) {
			return <pre className="text-amber-700 whitespace-pre-wrap break-words text-xs mt-0.5">{value}</pre>;
		}
		return <span className="text-amber-700">"{value}"</span>;
	}

	if (Array.isArray(value)) {
		if (value.length === 0) {
			return <span className="text-muted-foreground">[]</span>;
		}
		// Collapse arrays deeper than 1 level by default
		const content = (
			<div className="space-y-0.5">
				{value.map((item, i) => (
					// biome-ignore lint/suspicious/noArrayIndexKey: generic array display, index shown as label
					<div key={i} className="flex gap-1">
						<span className="text-muted-foreground text-[10px] w-4 text-right flex-shrink-0 tabular-nums">{i}</span>
						<JsonValue value={item} depth={depth + 1} />
					</div>
				))}
			</div>
		);

		if (depth > 0) {
			return <Collapsible label={`[${value.length}]`}>{content}</Collapsible>;
		}
		return content;
	}

	if (typeof value === "object") {
		const entries = Object.entries(value);
		if (entries.length === 0) {
			return <span className="text-muted-foreground">{"{}"}</span>;
		}

		const content = (
			<div className="space-y-0.5">
				{entries.map(([key, val]) => (
					<div key={key} className="flex gap-1.5">
						<span className="text-purple-600 font-mono text-xs flex-shrink-0">{key}:</span>
						<div className="min-w-0">
							<JsonValue value={val} depth={depth + 1} />
						</div>
					</div>
				))}
			</div>
		);

		if (depth > 0) {
			return <Collapsible label={`{${entries.length}}`}>{content}</Collapsible>;
		}
		return content;
	}

	return <span>{String(value)}</span>;
}

/**
 * Renders a JSON payload with collapsible nested objects and syntax highlighting.
 * Toggle between structured view and raw JSON.
 */
export function PayloadViewer({ payload }: { payload: Record<string, unknown> }) {
	const [showRaw, setShowRaw] = useState(false);

	const toggleView = useCallback(() => setShowRaw((v) => !v), []);

	if (Object.keys(payload).length === 0) {
		return <div className="text-xs text-muted-foreground italic py-2">No payload data</div>;
	}

	return (
		<div>
			<div className="flex justify-end mb-1">
				<button
					type="button"
					onClick={toggleView}
					className="text-[10px] text-muted-foreground hover:text-foreground transition-colors px-1.5 py-0.5 rounded hover:bg-accent"
				>
					{showRaw ? "Structured" : "Raw JSON"}
				</button>
			</div>
			{showRaw ? (
				<pre className="text-xs bg-muted rounded-md p-3 overflow-x-auto whitespace-pre-wrap break-words">
					{JSON.stringify(payload, null, 2)}
				</pre>
			) : (
				<div className="text-xs space-y-0.5">
					<JsonValue value={payload} depth={0} />
				</div>
			)}
		</div>
	);
}
