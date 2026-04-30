import { useState } from "react";
import type { RunResponse, TraceSummaryResponse } from "../../types";
import { StatusBadge } from "../primitives/status-badge";
import { Timestamp } from "../primitives/timestamp";

interface RunCardProps {
	run: RunResponse;
	summary?: TraceSummaryResponse;
	onClick: () => void;
	onDelete?: (runId: string) => Promise<void>;
}

export function RunCard({ run, summary, onClick, onDelete }: RunCardProps) {
	const description = run.metadata?.description as string | undefined;
	const [confirmingDelete, setConfirmingDelete] = useState(false);
	const [isDeleting, setIsDeleting] = useState(false);

	const handleDelete = async (e: React.MouseEvent) => {
		e.stopPropagation();
		if (!confirmingDelete) {
			setConfirmingDelete(true);
			return;
		}
		setIsDeleting(true);
		try {
			await onDelete?.(run.id);
		} finally {
			setIsDeleting(false);
			setConfirmingDelete(false);
		}
	};

	const handleCancelDelete = (e: React.MouseEvent) => {
		e.stopPropagation();
		setConfirmingDelete(false);
	};

	return (
		<div className="border rounded-lg px-4 py-3 hover:bg-accent/50 transition-colors cursor-pointer" onClick={onClick}>
			<div className="flex items-start justify-between gap-3">
				<div className="min-w-0 flex-1">
					<div className="flex items-center gap-2">
						<span className="font-medium text-sm truncate">{description || run.id}</span>
						<StatusBadge status={run.status} />
					</div>
					<div className="text-xs text-muted-foreground font-mono mt-0.5">{run.id}</div>
				</div>
				<div className="flex items-center gap-2 flex-shrink-0">
					<div className="text-xs text-muted-foreground">
						<Timestamp value={run.started_at} />
					</div>
					{onDelete && (
						<div className="flex items-center gap-1">
							{confirmingDelete ? (
								<>
									<button
										type="button"
										onClick={handleDelete}
										disabled={isDeleting}
										className="text-xs px-2 py-0.5 rounded bg-destructive text-destructive-foreground hover:bg-destructive/90 disabled:opacity-50"
									>
										{isDeleting ? "…" : "Confirm"}
									</button>
									<button
										type="button"
										onClick={handleCancelDelete}
										className="text-xs px-2 py-0.5 rounded text-muted-foreground hover:text-foreground hover:bg-accent"
									>
										Cancel
									</button>
								</>
							) : (
								<button
									type="button"
									onClick={handleDelete}
									className="text-xs px-1.5 py-0.5 rounded text-muted-foreground hover:text-destructive hover:bg-accent transition-colors"
									title="Delete run"
								>
									✕
								</button>
							)}
						</div>
					)}
				</div>
			</div>

			{summary && (
				<div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
					<span>{summary.llm_calls} LLM calls</span>
					<span>{summary.tool_calls} tool calls</span>
					<span>
						{(summary.total_input_tokens + summary.total_output_tokens).toLocaleString()} tokens
						{summary.cache_read_tokens > 0 && <> ({summary.cache_read_tokens.toLocaleString()} cached)</>}
					</span>
					{summary.errors > 0 && (
						<span className="text-destructive">
							{summary.errors} {summary.errors === 1 ? "error" : "errors"}
						</span>
					)}
				</div>
			)}
		</div>
	);
}
