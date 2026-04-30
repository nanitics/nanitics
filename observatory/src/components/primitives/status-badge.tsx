import type { RunStatus } from "../../types";

const statusStyles: Record<string, string> = {
	running: "bg-info-muted text-info-muted-foreground",
	completed: "bg-success-muted text-success-muted-foreground",
	failed: "bg-destructive-muted text-destructive-muted-foreground",
	suspended: "bg-warning-muted text-warning-muted-foreground",
};

export function StatusBadge({ status }: { status: RunStatus | string }) {
	const style = statusStyles[status] ?? "bg-muted text-muted-foreground";
	return (
		<span className={`inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium capitalize ${style}`}>
			{status}
		</span>
	);
}
