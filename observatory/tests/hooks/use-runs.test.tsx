import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ObservatoryClient } from "../../src/client/observatory-client";
import { ObservatoryProvider } from "../../src/context/observatory-context";
import { useRuns } from "../../src/hooks/use-runs";
import { EventRendererRegistry } from "../../src/registry/renderer-registry";
import type { RunListResponse, RunStatus } from "../../src/types";
import { makeRun, makeSummary } from "../fixtures/scenarios";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeResponse(ids: string[], total?: number): RunListResponse {
	return {
		runs: ids.map((id) => ({ run: makeRun({ id }), summary: makeSummary() })),
		total: total ?? ids.length,
	};
}

function createWrapper(client: ObservatoryClient) {
	const registry = new EventRendererRegistry();
	return function Wrapper({ children }: { children: React.ReactNode }) {
		return (
			<ObservatoryProvider client={client} registry={registry}>
				{children}
			</ObservatoryProvider>
		);
	};
}

interface Deferred<T> {
	promise: Promise<T>;
	resolve: (value: T) => void;
	reject: (reason?: unknown) => void;
}

function deferred<T>(): Deferred<T> {
	let resolve!: (value: T) => void;
	let reject!: (reason?: unknown) => void;
	const promise = new Promise<T>((res, rej) => {
		resolve = res;
		reject = rej;
	});
	return { promise, resolve, reject };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("useRuns", () => {
	let client: ObservatoryClient;

	beforeEach(() => {
		client = new ObservatoryClient("/test");
	});

	it("returns loading state initially, then populates runs and total on resolve", async () => {
		const spy = vi.spyOn(client, "listRuns").mockResolvedValue(makeResponse(["run-a", "run-b"], 2));

		const { result } = renderHook(() => useRuns(), {
			wrapper: createWrapper(client),
		});

		expect(result.current.isLoading).toBe(true);
		expect(result.current.runs).toEqual([]);
		expect(result.current.total).toBe(0);

		await waitFor(() => {
			expect(result.current.isLoading).toBe(false);
		});

		expect(result.current.runs.map((r) => r.run.id)).toEqual(["run-a", "run-b"]);
		expect(result.current.total).toBe(2);
		expect(result.current.error).toBeNull();
		expect(spy).toHaveBeenCalled();
	});

	it("filter-change aborts the in-flight initial fetch; stale response does not overwrite state", async () => {
		const first = deferred<RunListResponse>();
		const second = deferred<RunListResponse>();

		vi.spyOn(client, "listRuns").mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);

		const { result, rerender } = renderHook(({ status }: { status?: RunStatus }) => useRuns({ status }), {
			wrapper: createWrapper(client),
			initialProps: { status: "running" as RunStatus | undefined },
		});

		// Switch filter while first fetch is still pending
		rerender({ status: "completed" as RunStatus | undefined });

		// Resolve the second fetch first — it should land
		await act(async () => {
			second.resolve(makeResponse(["run-completed"], 1));
			await second.promise;
		});

		await waitFor(() => {
			expect(result.current.isLoading).toBe(false);
		});

		// Now let the aborted first fetch resolve — it must NOT overwrite state
		await act(async () => {
			first.resolve(makeResponse(["run-running"], 1));
			await first.promise;
		});

		// Give microtasks a chance to flush
		await new Promise((r) => setTimeout(r, 10));

		expect(result.current.runs.map((r) => r.run.id)).toEqual(["run-completed"]);
		expect(result.current.total).toBe(1);
	});

	it("filter-change aborts an in-flight loadMore; stale rows are not appended", async () => {
		const initial = deferred<RunListResponse>();
		const loadMoreCall = deferred<RunListResponse>();
		const refetchAfterFilter = deferred<RunListResponse>();

		vi.spyOn(client, "listRuns")
			.mockReturnValueOnce(initial.promise)
			.mockReturnValueOnce(loadMoreCall.promise)
			.mockReturnValueOnce(refetchAfterFilter.promise);

		const { result, rerender } = renderHook(({ status }: { status?: RunStatus }) => useRuns({ status }), {
			wrapper: createWrapper(client),
			initialProps: { status: "running" as RunStatus | undefined },
		});

		// Resolve initial fetch with two rows, total 10 (so hasMore)
		await act(async () => {
			initial.resolve(makeResponse(["r1", "r2"], 10));
			await initial.promise;
		});

		await waitFor(() => {
			expect(result.current.isLoading).toBe(false);
		});

		// Kick off loadMore (pending)
		act(() => {
			result.current.loadMore();
		});

		// Filter changes while loadMore is pending
		rerender({ status: "completed" as RunStatus | undefined });

		// The filter-change fetch resolves with fresh data
		await act(async () => {
			refetchAfterFilter.resolve(makeResponse(["c1"], 1));
			await refetchAfterFilter.promise;
		});

		await waitFor(() => {
			expect(result.current.isLoading).toBe(false);
		});

		// Now let the stale loadMore resolve — rows must not be appended
		await act(async () => {
			loadMoreCall.resolve(makeResponse(["stale-1", "stale-2"], 10));
			await loadMoreCall.promise;
		});

		await new Promise((r) => setTimeout(r, 10));

		expect(result.current.runs.map((r) => r.run.id)).toEqual(["c1"]);
		expect(result.current.total).toBe(1);
	});

	describe("lastRefreshedAt", () => {
		afterEach(() => {
			vi.useRealTimers();
		});

		it("is null on initial render before the first fetch resolves", () => {
			const pending = deferred<RunListResponse>();
			vi.spyOn(client, "listRuns").mockReturnValueOnce(pending.promise);

			const { result } = renderHook(() => useRuns(), {
				wrapper: createWrapper(client),
			});

			expect(result.current.lastRefreshedAt).toBeNull();

			// Clean up: resolve so no unhandled state
			pending.resolve(makeResponse([]));
		});

		it("is set to the current Date after a successful fetch resolves", async () => {
			const fixed = new Date("2025-06-01T12:00:00Z");
			// Only fake Date so vitest's own setTimeout (used by waitFor) still ticks.
			vi.useFakeTimers({ toFake: ["Date"] });
			vi.setSystemTime(fixed);

			vi.spyOn(client, "listRuns").mockResolvedValue(makeResponse(["r1"], 1));

			const { result } = renderHook(() => useRuns(), {
				wrapper: createWrapper(client),
			});

			await waitFor(() => {
				expect(result.current.isLoading).toBe(false);
			});

			expect(result.current.lastRefreshedAt).toBeInstanceOf(Date);
			expect(result.current.lastRefreshedAt?.getTime()).toBe(fixed.getTime());
		});

		it("does not update when the fetch is superseded by an abort before .then runs", async () => {
			const first = deferred<RunListResponse>();
			const second = deferred<RunListResponse>();

			vi.spyOn(client, "listRuns").mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);

			const { result, rerender } = renderHook(({ status }: { status?: RunStatus }) => useRuns({ status }), {
				wrapper: createWrapper(client),
				initialProps: { status: "running" as RunStatus | undefined },
			});

			// Before any fetch resolves, lastRefreshedAt is still null.
			expect(result.current.lastRefreshedAt).toBeNull();

			// Filter change aborts the first fetch.
			rerender({ status: "completed" as RunStatus | undefined });

			// Resolve the second (valid) fetch — this should set lastRefreshedAt.
			await act(async () => {
				second.resolve(makeResponse(["c1"], 1));
				await second.promise;
			});

			await waitFor(() => {
				expect(result.current.isLoading).toBe(false);
			});

			const afterSecond = result.current.lastRefreshedAt;
			expect(afterSecond).toBeInstanceOf(Date);

			// Now let the aborted first fetch resolve — it must NOT move the timestamp.
			await act(async () => {
				first.resolve(makeResponse(["r1"], 1));
				await first.promise;
			});

			await new Promise((r) => setTimeout(r, 10));

			expect(result.current.lastRefreshedAt).toBe(afterSecond);
		});

		it("does not update on error paths", async () => {
			vi.spyOn(client, "listRuns").mockRejectedValue(new Error("boom"));

			const { result } = renderHook(() => useRuns(), {
				wrapper: createWrapper(client),
			});

			await waitFor(() => {
				expect(result.current.isLoading).toBe(false);
			});

			expect(result.current.error).not.toBeNull();
			expect(result.current.lastRefreshedAt).toBeNull();
		});

		it("updates on the loadMore (append=true) path", async () => {
			const firstAt = new Date("2025-06-01T12:00:00Z");
			const laterAt = new Date("2025-06-01T12:05:00Z");

			// Real timers + deferred promises give us deterministic control over
			// when each fetch resolves, while `Date.now()` / `new Date()` is
			// driven by `vi.setSystemTime` — the two are independent in vitest.
			vi.useFakeTimers({ toFake: ["Date"] });
			vi.setSystemTime(firstAt);

			const initial = deferred<RunListResponse>();
			const more = deferred<RunListResponse>();
			const spy = vi.spyOn(client, "listRuns").mockReturnValueOnce(initial.promise).mockReturnValueOnce(more.promise);

			const { result } = renderHook(() => useRuns(), {
				wrapper: createWrapper(client),
			});

			await act(async () => {
				initial.resolve(makeResponse(["r1", "r2"], 10));
				await initial.promise;
			});

			await waitFor(() => {
				expect(result.current.isLoading).toBe(false);
			});

			expect(result.current.lastRefreshedAt?.getTime()).toBe(firstAt.getTime());

			// Jump the wall clock forward and trigger loadMore.
			vi.setSystemTime(laterAt);
			act(() => {
				result.current.loadMore();
			});

			await act(async () => {
				more.resolve(makeResponse(["r3", "r4"], 10));
				await more.promise;
			});

			await waitFor(() => {
				expect(result.current.runs.length).toBe(4);
			});

			expect(result.current.lastRefreshedAt?.getTime()).toBe(laterAt.getTime());
			expect(spy).toHaveBeenCalledTimes(2);
		});
	});

	it("passes an AbortSignal on every listRuns call and aborts superseded requests", async () => {
		const first = deferred<RunListResponse>();
		const second = deferred<RunListResponse>();

		const spy = vi.spyOn(client, "listRuns").mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);

		const { rerender } = renderHook(({ status }: { status?: RunStatus }) => useRuns({ status }), {
			wrapper: createWrapper(client),
			initialProps: { status: "running" as RunStatus | undefined },
		});

		// First call carries an AbortSignal
		expect(spy).toHaveBeenCalledTimes(1);
		const firstCallArgs = spy.mock.calls[0][0];
		expect(firstCallArgs?.signal).toBeInstanceOf(AbortSignal);
		const firstSignal = firstCallArgs!.signal!;
		expect(firstSignal.aborted).toBe(false);

		// Filter change supersedes the first request
		rerender({ status: "completed" as RunStatus | undefined });

		await waitFor(() => {
			expect(spy).toHaveBeenCalledTimes(2);
		});

		// The first signal is aborted; the second call gets a fresh, non-aborted signal
		expect(firstSignal.aborted).toBe(true);
		const secondCallArgs = spy.mock.calls[1][0];
		expect(secondCallArgs?.signal).toBeInstanceOf(AbortSignal);
		expect(secondCallArgs!.signal).not.toBe(firstSignal);

		// Clean up: let both pending promises settle so no unhandled rejections
		first.resolve(makeResponse([]));
		second.resolve(makeResponse([]));
		await new Promise((r) => setTimeout(r, 0));
	});
});
