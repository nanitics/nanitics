import type {
	AgentDetailResponse,
	AgentListResponse,
	EventListResponse,
	RunDetailResponse,
	RunListResponse,
	RunSortOption,
	RunStatus,
	SpanTreeResponse,
	TraceEvent,
	TraceLevel,
	TraceSummaryResponse,
	WorkflowDAGResponse,
} from "../types";

declare global {
	interface Window {
		__NANITICS_OBSERVATORY_BASE__?: string;
	}
}

/**
 * Resolve the API base URL the Observatory should talk to.
 *
 * The Python UI router injects `window.__NANITICS_OBSERVATORY_BASE__` so
 * the same prebuilt bundle works at any mount prefix. When the global is
 * absent (embedding the components in a custom SPA, running unit tests),
 * fall back to the explicit constructor argument.
 */
function resolveBaseUrl(explicit: string | undefined): string {
	if (explicit !== undefined) return explicit;
	const injected = typeof window !== "undefined" ? window.__NANITICS_OBSERVATORY_BASE__ : undefined;
	if (typeof injected === "string") return injected;
	throw new Error(
		"ObservatoryClient: no base URL provided and window.__NANITICS_OBSERVATORY_BASE__ is not set. " +
			"Either pass the API base URL to the constructor or mount the Python observatory router so " +
			"it injects the global at request time.",
	);
}

export class ObservatoryClient {
	private readonly baseUrl: string;

	constructor(baseUrl?: string) {
		this.baseUrl = resolveBaseUrl(baseUrl);
	}

	getBaseUrl(): string {
		return this.baseUrl;
	}

	async listRuns(options?: {
		status?: RunStatus;
		limit?: number;
		offset?: number;
		started_after?: string;
		started_before?: string;
		sort?: RunSortOption;
		search?: string;
		signal?: AbortSignal;
	}): Promise<RunListResponse> {
		const params = new URLSearchParams();
		if (options?.status) params.set("status", options.status);
		if (options?.limit != null) params.set("limit", String(options.limit));
		if (options?.offset != null) params.set("offset", String(options.offset));
		if (options?.started_after) params.set("started_after", options.started_after);
		if (options?.started_before) params.set("started_before", options.started_before);
		if (options?.sort) params.set("sort", options.sort);
		if (options?.search) params.set("search", options.search);
		const qs = params.toString();
		const url = `${this.baseUrl}/runs${qs ? `?${qs}` : ""}`;
		const res = await fetch(url, { signal: options?.signal });
		if (!res.ok) {
			throw new Error(`Failed to list runs: ${res.status} ${res.statusText}`);
		}
		return res.json();
	}

	async getRun(runId: string): Promise<RunDetailResponse> {
		const res = await fetch(`${this.baseUrl}/runs/${encodeURIComponent(runId)}`);
		if (!res.ok) {
			throw new Error(`Failed to get run ${runId}: ${res.status} ${res.statusText}`);
		}
		return res.json();
	}

	async getSpanTree(runId: string, options?: { minLevel?: TraceLevel }): Promise<SpanTreeResponse> {
		const params = new URLSearchParams();
		if (options?.minLevel) params.set("min_level", options.minLevel);
		const qs = params.toString();
		const url = `${this.baseUrl}/runs/${encodeURIComponent(runId)}/tree${qs ? `?${qs}` : ""}`;
		const res = await fetch(url);
		if (!res.ok) {
			throw new Error(`Failed to get span tree for run ${runId}: ${res.status} ${res.statusText}`);
		}
		return res.json();
	}

	async queryEvents(
		runId: string,
		options?: {
			level?: TraceLevel;
			eventTypes?: string[];
			limit?: number;
			after?: number;
		},
	): Promise<EventListResponse> {
		const params = new URLSearchParams();
		if (options?.level) params.set("level", options.level);
		if (options?.eventTypes?.length) params.set("event_types", options.eventTypes.join(","));
		if (options?.limit != null) params.set("limit", String(options.limit));
		if (options?.after != null) params.set("after", String(options.after));
		const qs = params.toString();
		const url = `${this.baseUrl}/runs/${encodeURIComponent(runId)}/events${qs ? `?${qs}` : ""}`;
		const res = await fetch(url);
		if (!res.ok) {
			throw new Error(`Failed to query events for run ${runId}: ${res.status} ${res.statusText}`);
		}
		return res.json();
	}

	async getEvent(eventId: number): Promise<TraceEvent> {
		const res = await fetch(`${this.baseUrl}/events/${eventId}`);
		if (!res.ok) {
			throw new Error(`Failed to get event ${eventId}: ${res.status} ${res.statusText}`);
		}
		return res.json();
	}

	async getSummary(runId: string): Promise<TraceSummaryResponse> {
		const res = await fetch(`${this.baseUrl}/runs/${encodeURIComponent(runId)}/summary`);
		if (!res.ok) {
			throw new Error(`Failed to get summary for run ${runId}: ${res.status} ${res.statusText}`);
		}
		return res.json();
	}

	async listAgents(runId: string): Promise<AgentListResponse> {
		const res = await fetch(`${this.baseUrl}/runs/${encodeURIComponent(runId)}/agents`);
		if (!res.ok) {
			throw new Error(`Failed to list agents for run ${runId}: ${res.status} ${res.statusText}`);
		}
		return res.json();
	}

	async getAgentDetail(runId: string, spanId: string): Promise<AgentDetailResponse> {
		const res = await fetch(`${this.baseUrl}/runs/${encodeURIComponent(runId)}/agents/${encodeURIComponent(spanId)}`);
		if (!res.ok) {
			throw new Error(`Failed to get agent detail for span ${spanId}: ${res.status} ${res.statusText}`);
		}
		return res.json();
	}

	async getWorkflow(runId: string): Promise<WorkflowDAGResponse> {
		const res = await fetch(`${this.baseUrl}/runs/${encodeURIComponent(runId)}/workflow`);
		if (!res.ok) {
			throw new Error(`Failed to get workflow for run ${runId}: ${res.status} ${res.statusText}`);
		}
		return res.json();
	}

	async deleteRun(runId: string): Promise<void> {
		const res = await fetch(`${this.baseUrl}/runs/${encodeURIComponent(runId)}`, { method: "DELETE" });
		if (!res.ok) {
			throw new Error(`Failed to delete run ${runId}: ${res.status} ${res.statusText}`);
		}
	}
}
