import { useCallback, useEffect, useMemo, useRef, useState } from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * Result of splitting a hash string into its route and query-string halves.
 * `route` is the path portion (e.g. `"/runs/abc"`); `queryString` is everything
 * after the first `?` (without the leading `?`).
 */
export interface HashQueryParts {
	/** Route portion of the hash; defaults to `"/"` when the hash is empty. */
	route: string;
	/** Query-string portion of the hash; `""` when no `?` is present. */
	queryString: string;
}

/**
 * Schema entry for a single URL-state filter. The hook calls `parse` on the
 * raw URL value (`null` when the key is absent) to compute the in-memory
 * value, and calls `stringify` to project the in-memory value back into the
 * URL. Returning `null` from `stringify` is the schema entry's signal for
 * "this matches the default; omit the key from the URL."
 */
export interface FilterSchemaEntry<T> {
	/** Convert a raw URL value (or `null` for absent) into an in-memory value. */
	parse(raw: string | null): T;
	/** Convert an in-memory value back to a URL string, or `null` to omit. */
	stringify(value: T): string | null;
}

/** A schema is a record of filter keys to their parser/stringifier pair. */
// biome-ignore lint/suspicious/noExplicitAny: schema entries are heterogeneous; concrete types are recovered through the FilterValues mapped type at the hook boundary.
export type FilterSchema = Record<string, FilterSchemaEntry<any>>;

/** Mapped type that derives the in-memory values from a schema. */
export type FilterValues<TSchema extends FilterSchema> = {
	[K in keyof TSchema]: ReturnType<TSchema[K]["parse"]>;
};

/** Mapped type that derives the per-key setters from a schema. */
export type FilterSetters<TSchema extends FilterSchema> = {
	[K in keyof TSchema]: (value: ReturnType<TSchema[K]["parse"]>) => void;
};

/** Return shape of `useUrlFilters`. */
export interface UseUrlFiltersResult<TSchema extends FilterSchema> {
	/** The current in-memory values, one per schema key. */
	values: FilterValues<TSchema>;
	/** Per-key setters; updates state synchronously and writes the URL after a debounce. */
	setters: FilterSetters<TSchema>;
	/** Update many keys in one go; coalesces the URL write into a single call. */
	setMany: (partial: Partial<FilterValues<TSchema>>) => void;
}

// ---------------------------------------------------------------------------
// Pure functions
// ---------------------------------------------------------------------------

/**
 * Split a hash string into its route and query-string halves.
 *
 * - A leading `#` is stripped if present (so callers may pass either
 *   `window.location.hash` or the already-stripped form).
 * - Splits on the first `?` only; subsequent `?` characters are part of the
 *   query string per the standard `URLSearchParams` parsing contract.
 * - Returns `route: "/"` for an empty hash and `queryString: ""` when no
 *   `?` is present.
 *
 * Pure function — does not touch `window`.
 */
export function splitHashRoute(hash: string): HashQueryParts {
	const stripped = hash.startsWith("#") ? hash.slice(1) : hash;
	const idx = stripped.indexOf("?");
	if (idx === -1) {
		return { route: stripped || "/", queryString: "" };
	}
	const route = stripped.slice(0, idx) || "/";
	const queryString = stripped.slice(idx + 1);
	return { route, queryString };
}

/**
 * Parse a hash string into its route plus a `URLSearchParams` view of the
 * query portion. Wraps `splitHashRoute` and the standard `URLSearchParams`
 * constructor.
 *
 * Pure function — does not touch `window`.
 */
export function parseHashQuery(hash: string): { route: string; params: URLSearchParams } {
	const { route, queryString } = splitHashRoute(hash);
	return { route, params: new URLSearchParams(queryString) };
}

/**
 * Stringify a route plus a `URLSearchParams` back into a hash string.
 * Returns just the route when the params are empty (no dangling `?`).
 *
 * Pure function — does not touch `window`.
 */
export function stringifyHashQuery(route: string, params: URLSearchParams): string {
	const qs = params.toString();
	if (qs === "") return route;
	return `${route}?${qs}`;
}

// ---------------------------------------------------------------------------
// React hook
// ---------------------------------------------------------------------------

/** Debounce window for URL writes. Search-box typing must not flood DevTools or the URL bar. */
const URL_WRITE_DEBOUNCE_MS = 100;

function readWindowHash(): string {
	if (typeof window === "undefined") return "";
	return window.location.hash;
}

function computeInitialValues<TSchema extends FilterSchema>(schema: TSchema): FilterValues<TSchema> {
	const { params } = parseHashQuery(readWindowHash());
	const out: Record<string, unknown> = {};
	for (const key of Object.keys(schema)) {
		out[key] = schema[key].parse(params.get(key));
	}
	return out as FilterValues<TSchema>;
}

/**
 * React hook that mirrors a set of filter values to and from the hash
 * query string of the page URL. Each schema entry owns the parse/stringify
 * pair plus the convention for "this value matches default — omit from URL."
 *
 * Behavior contract:
 *
 * - **Initial read.** On first render, parse `window.location.hash` and run
 *   each schema entry's `parse` against the corresponding param.
 * - **Setter.** Updates in-memory state synchronously, then schedules a URL
 *   write through a 100ms debounce. The URL write uses `history.replaceState`
 *   so typing in a text input does not pollute the back-stack.
 * - **`setMany`.** Multiple keys in one call; URL write coalesces into one
 *   `replaceState` call.
 * - **`popstate` / `hashchange`.** Listens for both and re-reads the URL
 *   into in-memory state when navigation occurs.
 * - **Default elision.** When `stringify(value)` returns `null`, the key is
 *   removed from the URL rather than written with an empty/default value.
 * - **Unknown keys.** Keys not in the schema are preserved on write — the
 *   hook merges its own keys into the existing `URLSearchParams` rather than
 *   replacing them.
 */
export function useUrlFilters<TSchema extends FilterSchema>(schema: TSchema): UseUrlFiltersResult<TSchema> {
	// Schema is captured once at mount; consumers pass a stable reference.
	const schemaRef = useRef(schema);

	const [values, setValues] = useState<FilterValues<TSchema>>(() => computeInitialValues(schema));

	const writeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
	const pendingValuesRef = useRef<FilterValues<TSchema>>(values);
	pendingValuesRef.current = values;

	const writeUrlNow = useCallback(() => {
		if (typeof window === "undefined") return;
		const { route, params } = parseHashQuery(window.location.hash);
		const currentValues = pendingValuesRef.current;
		for (const key of Object.keys(schemaRef.current)) {
			const stringified = schemaRef.current[key].stringify(currentValues[key]);
			if (stringified === null) {
				params.delete(key);
			} else {
				params.set(key, stringified);
			}
		}
		const next = stringifyHashQuery(route, params);
		const nextHash = `#${next}`;
		// Only write when the hash actually changes; this avoids extra
		// history entries and avoids a no-op `popstate` echo loop.
		if (window.location.hash !== nextHash) {
			window.history.replaceState(null, "", nextHash);
		}
	}, []);

	const scheduleWrite = useCallback(() => {
		if (writeTimerRef.current !== null) {
			clearTimeout(writeTimerRef.current);
		}
		writeTimerRef.current = setTimeout(() => {
			writeTimerRef.current = null;
			writeUrlNow();
		}, URL_WRITE_DEBOUNCE_MS);
	}, [writeUrlNow]);

	const setMany = useCallback(
		(partial: Partial<FilterValues<TSchema>>) => {
			setValues((prev) => ({ ...prev, ...partial }));
			scheduleWrite();
		},
		[scheduleWrite],
	);

	// Build per-key setters. The shape mirrors the schema, so we recompute
	// when the schema's key set changes (which is "never" for this codebase
	// since each page captures its schema as a module-level constant).
	const setters = useMemo(() => {
		const out: Record<string, (value: unknown) => void> = {};
		for (const key of Object.keys(schemaRef.current)) {
			out[key] = (value: unknown) => {
				setValues((prev) => ({ ...prev, [key]: value }));
				scheduleWrite();
			};
		}
		return out as FilterSetters<TSchema>;
	}, [scheduleWrite]);

	// Resync from the URL when the user navigates (back/forward, manual edit
	// of the URL bar, programmatic navigation that fires `hashchange`).
	useEffect(() => {
		if (typeof window === "undefined") return;
		const onChange = () => {
			setValues(computeInitialValues(schemaRef.current));
		};
		window.addEventListener("popstate", onChange);
		window.addEventListener("hashchange", onChange);
		return () => {
			window.removeEventListener("popstate", onChange);
			window.removeEventListener("hashchange", onChange);
		};
	}, []);

	// Cleanup any pending write on unmount so we never call `replaceState`
	// after the component has gone away.
	useEffect(() => {
		return () => {
			if (writeTimerRef.current !== null) {
				clearTimeout(writeTimerRef.current);
				writeTimerRef.current = null;
			}
		};
	}, []);

	return { values, setters, setMany };
}
