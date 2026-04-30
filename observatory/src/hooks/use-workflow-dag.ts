import { useCallback, useEffect, useMemo, useState } from "react";
import { computeDAGLayout } from "../components/dag/dag-layout";
import { useObservatory } from "../context/observatory-context";
import type { WorkflowDAGResponse } from "../types";
import type { DAGLayout } from "../types/dag-types";

export interface UseWorkflowDAGResult {
	workflow: WorkflowDAGResponse | null;
	layout: DAGLayout | null;
	isLoading: boolean;
	error: string | null;
	refetch: () => void;
}

export function useWorkflowDAG(runId: string): UseWorkflowDAGResult {
	const { client } = useObservatory();
	const [workflow, setWorkflow] = useState<WorkflowDAGResponse | null>(null);
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
			.getWorkflow(runId)
			.then((result) => {
				if (!cancelled) {
					setWorkflow(result);
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
	}, [client, runId, _fetchKey]);

	const layout = useMemo(() => {
		if (!workflow) return null;
		return computeDAGLayout(workflow.steps);
	}, [workflow]);

	return useMemo(
		() => ({ workflow, layout, isLoading, error, refetch }),
		[workflow, layout, isLoading, error, refetch],
	);
}
