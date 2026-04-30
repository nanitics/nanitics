import { useCallback, useEffect, useMemo, useState } from "react";
import { useObservatory } from "../context/observatory-context";
import type { AgentInfo, SpanTreeNode, TraceEvent } from "../types";

export interface UseAgentDetailResult {
	agent: AgentInfo | null;
	events: TraceEvent[];
	spanTree: SpanTreeNode | null;
	isLoading: boolean;
	error: string | null;
	refetch: () => void;
}

export function useAgentDetail(runId: string, spanId: string): UseAgentDetailResult {
	const { client } = useObservatory();
	const [agent, setAgent] = useState<AgentInfo | null>(null);
	const [events, setEvents] = useState<TraceEvent[]>([]);
	const [spanTree, setSpanTree] = useState<SpanTreeNode | null>(null);
	const [isLoading, setIsLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);
	const [_fetchKey, setFetchKey] = useState(0);

	const refetch = useCallback(() => {
		setFetchKey((k) => k + 1);
	}, []);

	// biome-ignore lint/correctness/useExhaustiveDependencies: _fetchKey is an intentional dependency to trigger re-fetch
	useEffect(() => {
		let cancelled = false;
		setIsLoading(true);
		setError(null);

		client
			.getAgentDetail(runId, spanId)
			.then((result) => {
				if (!cancelled) {
					setAgent(result.agent);
					setEvents(result.events);
					setSpanTree(result.span_tree);
				}
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
	}, [client, runId, spanId, _fetchKey]);

	return useMemo(
		() => ({ agent, events, spanTree, isLoading, error, refetch }),
		[agent, events, spanTree, isLoading, error, refetch],
	);
}
