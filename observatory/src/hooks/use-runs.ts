import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useObservatory } from "../context/observatory-context";
import type { RunListItem, RunSortOption, RunStatus } from "../types";

/** Debounce window (ms) applied to the `search` option before it reaches the network. */
const SEARCH_DEBOUNCE_MS = 300;

interface UseRunsOptions {
	status?: RunStatus;
	sort?: RunSortOption;
	search?: string;
	startedAfter?: string;
	startedBefore?: string;
	limit?: number;
}

interface UseRunsResult {
	runs: RunListItem[];
	total: number;
	isLoading: boolean;
	error: string | null;
	refetch: () => void;
	loadMore: () => void;
	hasMore: boolean;
	deleteRun: (runId: string) => Promise<void>;
	/**
	 * Timestamp of the most recent successful fetch resolution, or `null` if no
	 * fetch has resolved yet. Updated inside `fetchRuns`'s `.then(...)` after
	 * the abort guard and after `setRuns` / `setTotal`, so a superseded race
	 * never moves the timestamp. Not updated on error paths.
	 */
	lastRefreshedAt: Date | null;
}

export function useRuns(options?: UseRunsOptions): UseRunsResult {
	const { client } = useObservatory();
	const [runs, setRuns] = useState<RunListItem[]>([]);
	const [total, setTotal] = useState(0);
	const [isLoading, setIsLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);
	const [lastRefreshedAt, setLastRefreshedAt] = useState<Date | null>(null);

	const limit = options?.limit ?? 50;
	const status = options?.status;
	const sort = options?.sort;
	const search = options?.search;
	const startedAfter = options?.startedAfter;
	const startedBefore = options?.startedBefore;

	// Debounce `search` before it flows into the fetch dependency chain so that
	// typing does not fire a network call per keystroke. The initial value is
	// the initial `search` so the first render's fetch does not race against
	// a deferred setter.
	const [debouncedSearch, setDebouncedSearch] = useState<string | undefined>(search);
	useEffect(() => {
		const timer = setTimeout(() => setDebouncedSearch(search), SEARCH_DEBOUNCE_MS);
		return () => clearTimeout(timer);
	}, [search]);

	// Holds the AbortController of the request in-flight. `refetch` and
	// `loadMore` also consult this ref so they can abort a prior request
	// before issuing their own.
	const activeControllerRef = useRef<AbortController | null>(null);

	const fetchRuns = useCallback(
		(offset: number, append: boolean) => {
			activeControllerRef.current?.abort();
			const controller = new AbortController();
			activeControllerRef.current = controller;

			setIsLoading(true);
			setError(null);

			client
				.listRuns({
					status,
					limit,
					offset,
					sort,
					search: debouncedSearch,
					started_after: startedAfter,
					started_before: startedBefore,
					signal: controller.signal,
				})
				.then((data) => {
					if (controller.signal.aborted) return;
					setRuns((prev) => (append ? [...prev, ...data.runs] : data.runs));
					setTotal(data.total);
					setLastRefreshedAt(new Date());
				})
				.catch((err) => {
					if (controller.signal.aborted) return;
					if (err instanceof DOMException && err.name === "AbortError") return;
					setError(String(err));
				})
				.finally(() => {
					if (controller.signal.aborted) return;
					setIsLoading(false);
				});
		},
		[client, status, limit, sort, debouncedSearch, startedAfter, startedBefore],
	);

	useEffect(() => {
		fetchRuns(0, false);
		return () => {
			activeControllerRef.current?.abort();
		};
	}, [fetchRuns]);

	const refetch = useCallback(() => {
		fetchRuns(0, false);
	}, [fetchRuns]);

	const loadMore = useCallback(() => {
		fetchRuns(runs.length, true);
	}, [fetchRuns, runs.length]);

	const hasMore = runs.length < total;

	const deleteRun = useCallback(
		async (runId: string) => {
			await client.deleteRun(runId);
			fetchRuns(0, false);
		},
		[client, fetchRuns],
	);

	return useMemo(
		() => ({ runs, total, isLoading, error, refetch, loadMore, hasMore, deleteRun, lastRefreshedAt }),
		[runs, total, isLoading, error, refetch, loadMore, hasMore, deleteRun, lastRefreshedAt],
	);
}
