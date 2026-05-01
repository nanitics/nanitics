import { useEffect, useRef, useState } from "react";
import type { ConnectionState } from "../client/streaming-client";
import { StreamingClient } from "../client/streaming-client";
import { useObservatory } from "../context/observatory-context";
import type { TraceEvent, TraceLevel } from "../types";

interface UseStreamingOptions {
	/** Whether the run is active and should stream. */
	enabled: boolean;
	/** Minimum event level to receive. */
	minLevel?: TraceLevel;
	/** Callback when a new event arrives. */
	onEvent?: (event: TraceEvent) => void;
	/** Callback when the run completes. */
	onRunComplete?: (status: string) => void;
}

interface UseStreamingResult {
	/** Accumulated streamed events. */
	events: TraceEvent[];
	/** SSE connection state. */
	connectionState: ConnectionState;
	/** Whether the run has completed via SSE. */
	isComplete: boolean;
}

export function useStreaming(runId: string, options: UseStreamingOptions): UseStreamingResult {
	const { client } = useObservatory();
	const { enabled, minLevel, onEvent, onRunComplete } = options;

	const [events, setEvents] = useState<TraceEvent[]>([]);
	const [connectionState, setConnectionState] = useState<ConnectionState>("closed");
	const [isComplete, setIsComplete] = useState(false);

	// Stable refs for callbacks to avoid reconnecting on every render
	const onEventRef = useRef(onEvent);
	onEventRef.current = onEvent;
	const onRunCompleteRef = useRef(onRunComplete);
	onRunCompleteRef.current = onRunComplete;

	useEffect(() => {
		if (!enabled) {
			setConnectionState("closed");
			return;
		}

		const streamingClient = new StreamingClient(client.getBaseUrl());

		setEvents([]);
		setIsComplete(false);

		const connection = streamingClient.connect({
			runId,
			minLevel,
			onEvent(event) {
				setEvents((prev) => [...prev, event]);
				onEventRef.current?.(event);
			},
			onRunComplete(status) {
				setIsComplete(true);
				onRunCompleteRef.current?.(status);
			},
			onStateChange: setConnectionState,
		});

		return () => {
			connection.close();
		};
	}, [client, runId, enabled, minLevel]);

	return { events, connectionState, isComplete };
}
