import { useCallback, useEffect, useMemo, useState } from "react";
import { useObservatory } from "../context/observatory-context";
import type { AgentInfo } from "../types";

export interface UseAgentsResult {
	agents: AgentInfo[];
	isLoading: boolean;
	error: string | null;
	refetch: () => void;
}

export function useAgents(runId: string): UseAgentsResult {
	const { client } = useObservatory();
	const [agents, setAgents] = useState<AgentInfo[]>([]);
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
			.listAgents(runId)
			.then((result) => {
				if (!cancelled) setAgents(result.agents);
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

	return useMemo(() => ({ agents, isLoading, error, refetch }), [agents, isLoading, error, refetch]);
}
