import { RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";
import { useRuns } from "../../hooks/use-runs";
import type { RunSortOption, RunStatus } from "../../types";
import { ErrorState } from "../feedback/error-state";
import { RunListSkeleton } from "../feedback/loading-skeleton";
import { RunCard } from "./run-card";

/** Re-compute interval (ms) for the relative-timestamp display. */
const RELATIVE_TIMESTAMP_TICK_MS = 10_000;

/**
 * Formats a refresh timestamp relative to the current moment:
 * - `null` → "Never refreshed"
 * - diff < 5s → "Updated just now"
 * - diff < 60s → "Updated Ns ago"
 * - diff < 3600s → "Updated Nm ago"
 * - otherwise → "Updated Nh ago"
 */
function formatRelative(lastRefreshedAt: Date | null, now: number): string {
	if (lastRefreshedAt === null) return "Never refreshed";
	const diffSeconds = Math.max(0, Math.floor((now - lastRefreshedAt.getTime()) / 1000));
	if (diffSeconds < 5) return "Updated just now";
	if (diffSeconds < 60) return `Updated ${diffSeconds}s ago`;
	if (diffSeconds < 3600) return `Updated ${Math.floor(diffSeconds / 60)}m ago`;
	return `Updated ${Math.floor(diffSeconds / 3600)}h ago`;
}

/**
 * Muted "last refreshed" label that re-computes itself on a fixed cadence so
 * the label stays truthful even when no parent state changes. The interval is
 * cleared on unmount.
 */
function RelativeTimestamp({ lastRefreshedAt }: { lastRefreshedAt: Date | null }) {
	const [now, setNow] = useState(() => Date.now());
	useEffect(() => {
		const tick = setInterval(() => setNow(Date.now()), RELATIVE_TIMESTAMP_TICK_MS);
		return () => clearInterval(tick);
	}, []);
	return <span className="text-xs text-muted-foreground">{formatRelative(lastRefreshedAt, now)}</span>;
}

const STATUS_OPTIONS: { value: RunStatus | "all"; label: string }[] = [
	{ value: "all", label: "All" },
	{ value: "running", label: "Running" },
	{ value: "completed", label: "Completed" },
	{ value: "failed", label: "Failed" },
	{ value: "suspended", label: "Suspended" },
];

const SORT_OPTIONS: { value: RunSortOption; label: string }[] = [
	{ value: "started_at_desc", label: "Newest first" },
	{ value: "started_at_asc", label: "Oldest first" },
	{ value: "duration_desc", label: "Longest first" },
	{ value: "duration_asc", label: "Shortest first" },
];

interface RunListProps {
	statusFilter?: RunStatus;
	sortOrder?: RunSortOption;
	search?: string;
	startedAfter?: string;
	startedBefore?: string;
	onStatusFilterChange: (status: RunStatus | undefined) => void;
	onSortChange: (sort: RunSortOption) => void;
	onSearchChange: (search: string) => void;
	onDateRangeChange: (after: string | undefined, before: string | undefined) => void;
	onSelectRun: (runId: string) => void;
}

export function RunList({
	statusFilter,
	sortOrder = "started_at_desc",
	search,
	startedAfter,
	startedBefore,
	onStatusFilterChange,
	onSortChange,
	onSearchChange,
	onDateRangeChange,
	onSelectRun,
}: RunListProps) {
	const { runs, isLoading, error, refetch, loadMore, hasMore, deleteRun, lastRefreshedAt } = useRuns({
		status: statusFilter,
		sort: sortOrder,
		search: search || undefined,
		startedAfter,
		startedBefore,
	});

	return (
		<div className="space-y-4">
			{/* Toolbar */}
			<div className="flex items-center justify-between">
				<div role="toolbar" aria-label="Run list filters and refresh" className="flex items-center gap-2">
					{STATUS_OPTIONS.map((opt) => (
						<button
							type="button"
							key={opt.value}
							aria-pressed={(statusFilter ?? "all") === opt.value}
							onClick={() => onStatusFilterChange(opt.value === "all" ? undefined : opt.value)}
							className={`text-xs px-3 py-1.5 rounded-md transition-colors ${
								(statusFilter ?? "all") === opt.value
									? "bg-primary text-primary-foreground"
									: "text-muted-foreground hover:text-foreground hover:bg-accent"
							}`}
						>
							{opt.label}
						</button>
					))}
					<button
						type="button"
						onClick={refetch}
						className="inline-flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md transition-colors text-muted-foreground hover:text-foreground hover:bg-accent"
					>
						<RefreshCw className="h-4 w-4" aria-hidden="true" />
						Refresh
					</button>
					<RelativeTimestamp lastRefreshedAt={lastRefreshedAt} />
				</div>
			</div>

			{/* Search, sort, and date range */}
			<div className="flex items-center gap-3 flex-wrap">
				<label>
					<span className="sr-only">Search runs</span>
					<input
						type="text"
						placeholder="Search runs…"
						value={search ?? ""}
						onChange={(e) => onSearchChange(e.target.value)}
						className="text-xs px-3 py-1.5 rounded-md border bg-background text-foreground placeholder:text-muted-foreground w-48"
					/>
				</label>
				<label>
					<span className="sr-only">Sort runs</span>
					<select
						value={sortOrder}
						onChange={(e) => onSortChange(e.target.value as RunSortOption)}
						className="text-xs px-3 py-1.5 rounded-md border bg-background text-foreground"
					>
						{SORT_OPTIONS.map((opt) => (
							<option key={opt.value} value={opt.value}>
								{opt.label}
							</option>
						))}
					</select>
				</label>
				<label>
					<span className="sr-only">Started after</span>
					<input
						type="date"
						value={startedAfter?.split("T")[0] ?? ""}
						onChange={(e) =>
							onDateRangeChange(e.target.value ? `${e.target.value}T00:00:00Z` : undefined, startedBefore)
						}
						className="text-xs px-2 py-1.5 rounded-md border bg-background text-foreground"
					/>
				</label>
				<span className="text-xs text-muted-foreground" aria-hidden="true">
					–
				</span>
				<label>
					<span className="sr-only">Started before</span>
					<input
						type="date"
						value={startedBefore?.split("T")[0] ?? ""}
						onChange={(e) =>
							onDateRangeChange(startedAfter, e.target.value ? `${e.target.value}T23:59:59Z` : undefined)
						}
						className="text-xs px-2 py-1.5 rounded-md border bg-background text-foreground"
					/>
				</label>
			</div>

			{/* Run list */}
			{isLoading && runs.length === 0 && <RunListSkeleton />}

			{error && <ErrorState error={error} onRetry={refetch} />}

			{!isLoading && !error && runs.length === 0 && (
				<div className="text-muted-foreground py-8 text-center">No runs found</div>
			)}

			<div className="space-y-2">
				{runs.map((item) => (
					<RunCard
						key={item.run.id}
						run={item.run}
						summary={item.summary}
						onClick={() => onSelectRun(item.run.id)}
						onDelete={deleteRun}
					/>
				))}
			</div>

			{hasMore && (
				<div className="text-center">
					<button
						type="button"
						onClick={loadMore}
						disabled={isLoading}
						className="text-xs text-muted-foreground hover:text-foreground transition-colors px-3 py-1.5 rounded hover:bg-accent disabled:opacity-50"
					>
						{isLoading ? "Loading…" : "Load more"}
					</button>
				</div>
			)}
		</div>
	);
}
