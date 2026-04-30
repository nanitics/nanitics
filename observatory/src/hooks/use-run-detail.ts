import { useCallback, useEffect, useMemo, useState } from "react";
import { useObservatory } from "../context/observatory-context";
import type { RunDetailResponse } from "../types";

interface UseRunDetailResult {
	data: RunDetailResponse | null;
	isLoading: boolean;
	error: string | null;
	refetch: () => void;
}

export function useRunDetail(runId: string): UseRunDetailResult {
	const { client } = useObservatory();
	const [data, setData] = useState<RunDetailResponse | null>(null);
	const [isLoading, setIsLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);
	const [_fetchKey, setFetchKey] = useState(0);

	const refetch = useCallback(() => {
		setFetchKey((k) => k + 1);
	}, []);

	useEffect(() => {
		let cancelled = false;
		setIsLoading(true);
		setError(null);

		client
			.getRun(runId)
			.then((result) => {
				if (!cancelled) setData(result);
			})
			.catch((err) => {
				if (!cancelled) setError(String(err));
			})
			.finally(() => {
				if (!cancelled) setIsLoading(false);
			});

		return () => {
			cancelled = true;
		};
	}, [client, runId]);

	return useMemo(() => ({ data, isLoading, error, refetch }), [data, isLoading, error, refetch]);
}
