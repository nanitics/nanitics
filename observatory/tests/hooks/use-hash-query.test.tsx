import { act, render, screen } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
	type FilterSchema,
	parseHashQuery,
	splitHashRoute,
	stringifyHashQuery,
	useUrlFilters,
} from "../../src/hooks/use-hash-query";

// ---------------------------------------------------------------------------
// Pure functions — splitHashRoute / parseHashQuery / stringifyHashQuery
// ---------------------------------------------------------------------------

describe("splitHashRoute", () => {
	it("returns the route '/' for an empty hash", () => {
		expect(splitHashRoute("")).toEqual({ route: "/", queryString: "" });
	});

	it("strips a leading '#' if present", () => {
		expect(splitHashRoute("#/runs")).toEqual({ route: "/runs", queryString: "" });
	});

	it("returns the full hash as the route when no '?' is present", () => {
		expect(splitHashRoute("/runs/abc")).toEqual({ route: "/runs/abc", queryString: "" });
	});

	it("splits route and query on the first '?'", () => {
		expect(splitHashRoute("/runs?status=running")).toEqual({
			route: "/runs",
			queryString: "status=running",
		});
	});

	it("includes subsequent '?' characters in the query string", () => {
		// `URLSearchParams` accepts the resulting string; consumers do not need
		// to special-case multiple `?` characters.
		expect(splitHashRoute("/runs?search=foo?bar")).toEqual({
			route: "/runs",
			queryString: "search=foo?bar",
		});
	});

	it("treats a route that is only '?…' as route '/'", () => {
		expect(splitHashRoute("?status=running")).toEqual({
			route: "/",
			queryString: "status=running",
		});
	});
});

describe("parseHashQuery", () => {
	it("returns route='/' and empty params for an empty hash", () => {
		const { route, params } = parseHashQuery("");
		expect(route).toBe("/");
		expect(Array.from(params.entries())).toEqual([]);
	});

	it("returns route and empty params for a pure route", () => {
		const { route, params } = parseHashQuery("#/runs/abc");
		expect(route).toBe("/runs/abc");
		expect(Array.from(params.entries())).toEqual([]);
	});

	it("parses a route with a single key", () => {
		const { route, params } = parseHashQuery("#/runs?status=running");
		expect(route).toBe("/runs");
		expect(params.get("status")).toBe("running");
	});

	it("parses a route with multiple keys", () => {
		const { params } = parseHashQuery("#/runs?status=running&sort=duration_desc");
		expect(params.get("status")).toBe("running");
		expect(params.get("sort")).toBe("duration_desc");
	});

	it("appends repeated keys per URLSearchParams semantics", () => {
		const { params } = parseHashQuery("#/runs?status=running&status=failed");
		expect(params.getAll("status")).toEqual(["running", "failed"]);
	});

	it("tolerates malformed entries (`?=foo` and `?foo=`)", () => {
		const { params } = parseHashQuery("#/runs?=foo&bar=");
		// `?=foo` produces an empty-string key with value 'foo'.
		expect(params.get("")).toBe("foo");
		expect(params.get("bar")).toBe("");
	});
});

describe("stringifyHashQuery", () => {
	it("returns just the route when params are empty", () => {
		expect(stringifyHashQuery("/runs", new URLSearchParams())).toBe("/runs");
	});

	it("appends params with a single '?'", () => {
		const params = new URLSearchParams();
		params.set("status", "running");
		expect(stringifyHashQuery("/runs", params)).toBe("/runs?status=running");
	});

	it("preserves multiple keys", () => {
		const params = new URLSearchParams();
		params.set("status", "running");
		params.set("sort", "duration_desc");
		expect(stringifyHashQuery("/runs", params)).toBe("/runs?status=running&sort=duration_desc");
	});

	it("round-trips parse → stringify on a representative payload", () => {
		const original = "/runs?status=running&sort=duration_desc&search=foo";
		const { route, params } = parseHashQuery(original);
		expect(stringifyHashQuery(route, params)).toBe(original);
	});
});

// ---------------------------------------------------------------------------
// useUrlFilters hook
// ---------------------------------------------------------------------------

interface TestFilterValues {
	status: string | undefined;
	sort: string;
	search: string;
}

const TEST_SCHEMA: FilterSchema = {
	status: {
		parse: (raw) => (raw === null || raw === "" ? undefined : raw),
		stringify: (value) => (value === undefined ? null : (value as string)),
	},
	sort: {
		parse: (raw) => (raw === null ? "started_at_desc" : raw),
		stringify: (value) => (value === "started_at_desc" ? null : (value as string)),
	},
	search: {
		parse: (raw) => (raw === null ? "" : raw),
		stringify: (value) => (value === "" ? null : (value as string)),
	},
};

interface ProbeProps {
	onRender: (api: ReturnType<typeof useUrlFilters<typeof TEST_SCHEMA>>) => void;
}

function Probe({ onRender }: ProbeProps) {
	const api = useUrlFilters(TEST_SCHEMA);
	// `useEffect` after-commit so `onRender` only fires once per committed render.
	useEffect(() => {
		onRender(api);
	});
	const v = api.values as TestFilterValues;
	return (
		<div>
			<span data-testid="status">{v.status ?? "-"}</span>
			<span data-testid="sort">{v.sort}</span>
			<span data-testid="search">{v.search}</span>
		</div>
	);
}

describe("useUrlFilters", () => {
	beforeEach(() => {
		vi.useFakeTimers();
		// Reset hash before every test.
		window.history.replaceState(null, "", "#/runs");
	});

	afterEach(() => {
		vi.useRealTimers();
		window.history.replaceState(null, "", "#/runs");
	});

	it("performs an initial read from window.location.hash", () => {
		window.history.replaceState(null, "", "#/runs?status=running&search=foo");
		render(<Probe onRender={() => {}} />);
		expect(screen.getByTestId("status").textContent).toBe("running");
		expect(screen.getByTestId("search").textContent).toBe("foo");
		// Defaulted key is the schema default.
		expect(screen.getByTestId("sort").textContent).toBe("started_at_desc");
	});

	it("setter updates state synchronously", () => {
		let api: ReturnType<typeof useUrlFilters<typeof TEST_SCHEMA>> | null = null;
		render(
			<Probe
				onRender={(a) => {
					api = a;
				}}
			/>,
		);

		expect(api).not.toBeNull();
		act(() => {
			(api as NonNullable<typeof api>).setters.status("running");
		});
		expect(screen.getByTestId("status").textContent).toBe("running");
	});

	it("setter writes window.location.hash after the debounce", () => {
		let api: ReturnType<typeof useUrlFilters<typeof TEST_SCHEMA>> | null = null;
		render(
			<Probe
				onRender={(a) => {
					api = a;
				}}
			/>,
		);

		act(() => {
			(api as NonNullable<typeof api>).setters.status("running");
		});
		// Before the debounce fires, the URL is unchanged.
		expect(window.location.hash).toBe("#/runs");

		act(() => {
			vi.advanceTimersByTime(200);
		});
		expect(window.location.hash).toBe("#/runs?status=running");
	});

	it("setMany updates multiple keys in a single URL write", () => {
		let api: ReturnType<typeof useUrlFilters<typeof TEST_SCHEMA>> | null = null;
		render(
			<Probe
				onRender={(a) => {
					api = a;
				}}
			/>,
		);

		const writeSpy = vi.spyOn(window.history, "replaceState");

		act(() => {
			(api as NonNullable<typeof api>).setMany({ status: "running", search: "foo" });
		});
		act(() => {
			vi.advanceTimersByTime(200);
		});
		// Exactly one `replaceState` call covers both updates.
		expect(writeSpy).toHaveBeenCalledTimes(1);
		expect(window.location.hash).toBe("#/runs?status=running&search=foo");
	});

	it("removes a key from the URL when the value matches the schema default", () => {
		window.history.replaceState(null, "", "#/runs?status=running&sort=duration_desc");

		let api: ReturnType<typeof useUrlFilters<typeof TEST_SCHEMA>> | null = null;
		render(
			<Probe
				onRender={(a) => {
					api = a;
				}}
			/>,
		);

		act(() => {
			(api as NonNullable<typeof api>).setters.sort("started_at_desc");
		});
		act(() => {
			vi.advanceTimersByTime(200);
		});
		// `sort` is back to default → key omitted; `status` survives.
		expect(window.location.hash).toBe("#/runs?status=running");
	});

	it("resyncs in-memory state on hashchange", () => {
		render(<Probe onRender={() => {}} />);
		expect(screen.getByTestId("status").textContent).toBe("-");

		act(() => {
			window.history.replaceState(null, "", "#/runs?status=failed");
			window.dispatchEvent(new HashChangeEvent("hashchange"));
		});
		expect(screen.getByTestId("status").textContent).toBe("failed");
	});

	it("resyncs in-memory state on popstate", () => {
		render(<Probe onRender={() => {}} />);

		act(() => {
			window.history.replaceState(null, "", "#/runs?status=completed");
			window.dispatchEvent(new PopStateEvent("popstate"));
		});
		expect(screen.getByTestId("status").textContent).toBe("completed");
	});

	it("preserves unknown keys on write (the hook only manages its own schema keys)", () => {
		window.history.replaceState(null, "", "#/runs?experimental=keep_me");

		let api: ReturnType<typeof useUrlFilters<typeof TEST_SCHEMA>> | null = null;
		render(
			<Probe
				onRender={(a) => {
					api = a;
				}}
			/>,
		);

		act(() => {
			(api as NonNullable<typeof api>).setters.status("running");
		});
		act(() => {
			vi.advanceTimersByTime(200);
		});
		// Unknown key survives the write.
		const params = new URLSearchParams(window.location.hash.split("?")[1]);
		expect(params.get("experimental")).toBe("keep_me");
		expect(params.get("status")).toBe("running");
	});

	it("uses replaceState (not pushState) for filter writes", () => {
		const replaceSpy = vi.spyOn(window.history, "replaceState");
		const pushSpy = vi.spyOn(window.history, "pushState");

		let api: ReturnType<typeof useUrlFilters<typeof TEST_SCHEMA>> | null = null;
		render(
			<Probe
				onRender={(a) => {
					api = a;
				}}
			/>,
		);

		act(() => {
			(api as NonNullable<typeof api>).setters.status("running");
		});
		act(() => {
			vi.advanceTimersByTime(200);
		});
		expect(replaceSpy).toHaveBeenCalled();
		expect(pushSpy).not.toHaveBeenCalled();
	});
});
