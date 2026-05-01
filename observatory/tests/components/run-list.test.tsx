import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { axe } from "vitest-axe";
import { ObservatoryClient } from "../../src/client/observatory-client";
import { RunCard } from "../../src/components/run-list/run-card";
import { RunList } from "../../src/components/run-list/run-list";
import { ObservatoryProvider } from "../../src/context/observatory-context";
import { RunListPage } from "../../src/pages/run-list-page";
import { EventRendererRegistry } from "../../src/registry/renderer-registry";
import type { RunListResponse } from "../../src/types";
import { makeRun, makeSummary } from "../fixtures/scenarios";

describe("RunCard", () => {
	it("renders run description when present in metadata", () => {
		const run = makeRun({ metadata: { description: "Research Assistant" } });
		render(<RunCard run={run} onClick={vi.fn()} />);
		expect(screen.getByText("Research Assistant")).toBeInTheDocument();
	});

	it("falls back to run ID when no description", () => {
		const run = makeRun({ id: "run-abc-123", metadata: {} });
		render(<RunCard run={run} onClick={vi.fn()} />);
		// RunCard shows the ID as both the title and a subtitle — use getAllByText
		const matches = screen.getAllByText("run-abc-123");
		expect(matches.length).toBeGreaterThanOrEqual(1);
	});

	it("displays status badge", () => {
		const run = makeRun({ status: "running" });
		render(<RunCard run={run} onClick={vi.fn()} />);
		expect(screen.getByText("running")).toBeInTheDocument();
	});

	it("displays summary stats when summary is provided", () => {
		const run = makeRun();
		const summary = makeSummary({
			llm_calls: 6,
			tool_calls: 4,
			total_input_tokens: 1200,
			total_output_tokens: 800,
			errors: 0,
		});
		render(<RunCard run={run} summary={summary} onClick={vi.fn()} />);
		expect(screen.getByText("6 LLM calls")).toBeInTheDocument();
		expect(screen.getByText("4 tool calls")).toBeInTheDocument();
		expect(screen.getByText(/2,000/)).toBeInTheDocument(); // 1200 + 800
	});

	it("shows error count when errors > 0", () => {
		const run = makeRun({ status: "failed" });
		const summary = makeSummary({ errors: 3 });
		render(<RunCard run={run} summary={summary} onClick={vi.fn()} />);
		expect(screen.getByText("3 errors")).toBeInTheDocument();
	});

	it("does not show errors text when errors = 0", () => {
		const run = makeRun();
		const summary = makeSummary({ errors: 0 });
		render(<RunCard run={run} summary={summary} onClick={vi.fn()} />);
		expect(screen.queryByText(/\d+ errors?/)).not.toBeInTheDocument();
	});

	it("does not render summary stats when no summary", () => {
		const run = makeRun();
		render(<RunCard run={run} onClick={vi.fn()} />);
		expect(screen.queryByText(/LLM calls/)).not.toBeInTheDocument();
		expect(screen.queryByText(/tool calls/)).not.toBeInTheDocument();
	});

	it("calls onClick when clicked", () => {
		const onClick = vi.fn();
		const run = makeRun({ metadata: { description: "Clickable Run" } });
		render(<RunCard run={run} onClick={onClick} />);
		fireEvent.click(screen.getByText("Clickable Run"));
		expect(onClick).toHaveBeenCalled();
	});
});

// ---------------------------------------------------------------------------
// RunList — refresh affordance + RelativeTimestamp
// ---------------------------------------------------------------------------

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

function makeRunListResponse(ids: string[], total?: number): RunListResponse {
	return {
		runs: ids.map((id) => ({ run: makeRun({ id }), summary: makeSummary() })),
		total: total ?? ids.length,
	};
}

function renderRunList(client: ObservatoryClient) {
	const registry = new EventRendererRegistry();
	const props = {
		onStatusFilterChange: vi.fn(),
		onSortChange: vi.fn(),
		onSearchChange: vi.fn(),
		onDateRangeChange: vi.fn(),
		onSelectRun: vi.fn(),
	};
	return {
		...render(
			<ObservatoryProvider client={client} registry={registry}>
				<RunList {...props} />
			</ObservatoryProvider>,
		),
		props,
	};
}

describe("RunList — refresh affordance", () => {
	afterEach(() => {
		vi.useRealTimers();
	});

	it("renders a promoted Refresh button (not the ↻ text glyph)", async () => {
		const client = new ObservatoryClient("/test");
		vi.spyOn(client, "listRuns").mockResolvedValue(makeRunListResponse([]));

		renderRunList(client);

		const button = screen.getByRole("button", { name: /Refresh/i });
		expect(button).toBeInTheDocument();
		expect(button.tagName).toBe("BUTTON");
		expect(button).toHaveAttribute("type", "button");
		// The old glyph must not appear anywhere in the document.
		expect(screen.queryByText(/↻/)).not.toBeInTheDocument();
	});

	it("calls refetch when the Refresh button is clicked", async () => {
		const client = new ObservatoryClient("/test");
		const spy = vi.spyOn(client, "listRuns").mockResolvedValue(makeRunListResponse([]));

		renderRunList(client);

		await waitFor(() => expect(spy).toHaveBeenCalledTimes(1));

		fireEvent.click(screen.getByRole("button", { name: /Refresh/i }));

		await waitFor(() => expect(spy).toHaveBeenCalledTimes(2));
	});

	it("invokes refetch when Enter is pressed on the focused Refresh button", async () => {
		const client = new ObservatoryClient("/test");
		const spy = vi.spyOn(client, "listRuns").mockResolvedValue(makeRunListResponse([]));

		renderRunList(client);

		await waitFor(() => expect(spy).toHaveBeenCalledTimes(1));

		const button = screen.getByRole("button", { name: /Refresh/i });
		button.focus();
		expect(document.activeElement).toBe(button);
		// jsdom does not synthesize click from keyDown on native button, so simulate
		// both the keyDown (for keyboard-focusability signal) and the resulting click.
		fireEvent.keyDown(button, { key: "Enter" });
		fireEvent.click(button);

		await waitFor(() => expect(spy).toHaveBeenCalledTimes(2));
	});

	it("renders 'Never refreshed' initially before the first fetch resolves", () => {
		const client = new ObservatoryClient("/test");
		const pending = deferred<RunListResponse>();
		vi.spyOn(client, "listRuns").mockReturnValueOnce(pending.promise);

		renderRunList(client);

		expect(screen.getByText("Never refreshed")).toBeInTheDocument();

		// Settle the promise so there's no dangling pending.
		pending.resolve(makeRunListResponse([]));
	});

	it("rolls from 'Updated just now' to 'Updated Ns ago' as time passes", async () => {
		vi.useFakeTimers();
		vi.setSystemTime(new Date("2025-06-01T12:00:00Z"));

		const client = new ObservatoryClient("/test");
		vi.spyOn(client, "listRuns").mockResolvedValue(makeRunListResponse([]));

		renderRunList(client);

		// Let the mocked fetch promise resolve and React commit the state.
		await act(async () => {
			await vi.advanceTimersByTimeAsync(0);
		});

		expect(screen.getByText("Updated just now")).toBeInTheDocument();

		// Advance clock by 30 seconds. The RelativeTimestamp interval ticks every
		// 10s, so by 30s we should be firmly in the seconds-ago bucket.
		await act(async () => {
			await vi.advanceTimersByTimeAsync(30_000);
		});

		expect(screen.getByText(/Updated \d+s ago/)).toBeInTheDocument();
	});

	it("the full RunList container has no axe violations", async () => {
		const client = new ObservatoryClient("/test");
		vi.spyOn(client, "listRuns").mockResolvedValue(makeRunListResponse([]));

		const { container } = renderRunList(client);

		await waitFor(() => {
			expect(screen.getByText("Updated just now")).toBeInTheDocument();
		});

		// Axe scope covers the full RunList container. Search input, date
		// inputs, sort `<select>`, and filter buttons all carry labels and
		// `aria-pressed` semantics.
		const results = await axe(container);
		expect(results).toHaveNoViolations();
	});
});

// ---------------------------------------------------------------------------
// RunListPage — URL-state wiring (Step 4)
// ---------------------------------------------------------------------------

function renderRunListPage(client: ObservatoryClient) {
	const registry = new EventRendererRegistry();
	return render(
		<ObservatoryProvider client={client} registry={registry}>
			<RunListPage onSelectRun={vi.fn()} />
		</ObservatoryProvider>,
	);
}

describe("RunListPage — URL-state wiring", () => {
	beforeEach(() => {
		window.history.replaceState(null, "", "#/");
	});

	afterEach(() => {
		vi.useRealTimers();
		window.history.replaceState(null, "", "#/");
	});

	it("clicking a status filter button writes to window.location.hash after the debounce", async () => {
		vi.useFakeTimers({ shouldAdvanceTime: true });

		const client = new ObservatoryClient("/test");
		vi.spyOn(client, "listRuns").mockResolvedValue(makeRunListResponse([]));

		renderRunListPage(client);

		const completedButton = await screen.findByRole("button", { name: "Completed" });
		fireEvent.click(completedButton);

		// Drive the 100ms debounce.
		act(() => {
			vi.advanceTimersByTime(200);
		});

		expect(window.location.hash).toBe("#/?status=completed");
	});

	it("initial filter state is read from a deep-link URL", async () => {
		window.history.replaceState(null, "", "#/?status=running");

		const client = new ObservatoryClient("/test");
		const spy = vi.spyOn(client, "listRuns").mockResolvedValue(makeRunListResponse([]));

		renderRunListPage(client);

		// `useRuns` is called with the deep-linked status filter immediately —
		// no need to wait for any state propagation.
		await waitFor(() => {
			expect(spy).toHaveBeenCalled();
			const lastCall = spy.mock.calls[spy.mock.calls.length - 1];
			expect(lastCall[0]).toMatchObject({ status: "running" });
		});
	});
});
