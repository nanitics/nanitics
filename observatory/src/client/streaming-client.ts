import type { TraceEvent, TraceLevel } from "../types";

export type ConnectionState = "connecting" | "connected" | "reconnecting" | "closed";

export interface StreamConnection {
	close(): void;
	readonly state: ConnectionState;
}

export interface StreamOptions {
	runId: string;
	minLevel?: TraceLevel;
	onEvent: (event: TraceEvent) => void;
	onRunComplete: (status: string) => void;
	onStateChange?: (state: ConnectionState) => void;
	onError?: (error: Event) => void;
}

export class StreamingClient {
	constructor(private readonly baseUrl: string) {}

	connect(options: StreamOptions): StreamConnection {
		const { runId, minLevel, onEvent, onRunComplete, onStateChange, onError } = options;

		const params = new URLSearchParams();
		if (minLevel) params.set("min_level", minLevel);
		const qs = params.toString();
		const url = `${this.baseUrl}/runs/${encodeURIComponent(runId)}/stream${qs ? `?${qs}` : ""}`;

		let state: ConnectionState = "connecting";
		let hasConnected = false;

		const setState = (next: ConnectionState) => {
			if (state === next) return;
			state = next;
			onStateChange?.(next);
		};

		const source = new EventSource(url);

		source.addEventListener("open", () => {
			hasConnected = true;
			setState("connected");
		});

		source.addEventListener("trace", (e: MessageEvent) => {
			const event: TraceEvent = JSON.parse(e.data);
			onEvent(event);
		});

		source.addEventListener("run_complete", (e: MessageEvent) => {
			const data: { status: string } = JSON.parse(e.data);
			onRunComplete(data.status);
			source.close();
			setState("closed");
		});

		source.addEventListener("error", (e: Event) => {
			if (source.readyState === EventSource.CLOSED) {
				setState("closed");
			} else if (hasConnected) {
				// EventSource will auto-reconnect
				setState("reconnecting");
			}
			onError?.(e);
		});

		const connection: StreamConnection = {
			close() {
				source.close();
				setState("closed");
			},
			get state() {
				return state;
			},
		};

		return connection;
	}
}
