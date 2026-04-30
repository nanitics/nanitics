import { useCallback } from "react";
import { RunList } from "../components/run-list/run-list";
import { type FilterSchema, useUrlFilters } from "../hooks/use-hash-query";
import type { RunSortOption, RunStatus } from "../types";

interface RunListPageProps {
	onSelectRun: (runId: string) => void;
}

const KNOWN_STATUSES: ReadonlySet<string> = new Set(["running", "completed", "failed", "suspended"]);
const KNOWN_SORTS: ReadonlySet<string> = new Set([
	"started_at_desc",
	"started_at_asc",
	"duration_desc",
	"duration_asc",
]);
const DEFAULT_SORT: RunSortOption = "started_at_desc";

/**
 * URL-state schema for the Run List filters. Each entry's `parse` projects a
 * raw URL value (or `null` for absent keys) into the in-memory shape, and
 * `stringify` projects the in-memory value back to the URL — returning `null`
 * to elide the key when the value matches its default.
 */
const RUN_LIST_FILTER_SCHEMA = {
	status: {
		parse: (raw: string | null): RunStatus | undefined =>
			raw !== null && KNOWN_STATUSES.has(raw) ? (raw as RunStatus) : undefined,
		stringify: (value: RunStatus | undefined): string | null => (value === undefined ? null : value),
	},
	sort: {
		parse: (raw: string | null): RunSortOption =>
			raw !== null && KNOWN_SORTS.has(raw) ? (raw as RunSortOption) : DEFAULT_SORT,
		stringify: (value: RunSortOption): string | null => (value === DEFAULT_SORT ? null : value),
	},
	search: {
		parse: (raw: string | null): string => raw ?? "",
		stringify: (value: string): string | null => (value === "" ? null : value),
	},
	started_after: {
		parse: (raw: string | null): string | undefined => (raw === null || raw === "" ? undefined : raw),
		stringify: (value: string | undefined): string | null => (value === undefined ? null : value),
	},
	started_before: {
		parse: (raw: string | null): string | undefined => (raw === null || raw === "" ? undefined : raw),
		stringify: (value: string | undefined): string | null => (value === undefined ? null : value),
	},
} satisfies FilterSchema;

export function RunListPage({ onSelectRun }: RunListPageProps) {
	const { values, setters, setMany } = useUrlFilters(RUN_LIST_FILTER_SCHEMA);

	const handleDateRangeChange = useCallback(
		(after: string | undefined, before: string | undefined) => {
			setMany({ started_after: after, started_before: before });
		},
		[setMany],
	);

	return (
		<main className="flex-1 overflow-y-auto" aria-label="Runs">
			<div className="max-w-4xl mx-auto px-6 py-6 w-full">
				<RunList
					statusFilter={values.status}
					sortOrder={values.sort}
					search={values.search}
					startedAfter={values.started_after}
					startedBefore={values.started_before}
					onStatusFilterChange={setters.status}
					onSortChange={setters.sort}
					onSearchChange={setters.search}
					onDateRangeChange={handleDateRangeChange}
					onSelectRun={onSelectRun}
				/>
			</div>
		</main>
	);
}
