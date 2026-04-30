import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { axe } from "vitest-axe";
import { ThemeToggle } from "../../src/components/feedback/theme-toggle";

interface MediaQueryListMock {
	matches: boolean;
	media: string;
	onchange: null;
	addEventListener: ReturnType<typeof vi.fn>;
	removeEventListener: ReturnType<typeof vi.fn>;
	dispatchEvent: ReturnType<typeof vi.fn>;
	_fire: (matches: boolean) => void;
}

function createMatchMediaMock(matches: boolean): MediaQueryListMock {
	const listeners = new Set<(e: MediaQueryListEvent) => void>();
	const mql: MediaQueryListMock = {
		matches,
		media: "(prefers-color-scheme: dark)",
		onchange: null,
		addEventListener: vi.fn((_type: string, listener: (e: MediaQueryListEvent) => void) => {
			listeners.add(listener);
		}),
		removeEventListener: vi.fn((_type: string, listener: (e: MediaQueryListEvent) => void) => {
			listeners.delete(listener);
		}),
		dispatchEvent: vi.fn(),
		_fire: (newMatches: boolean) => {
			mql.matches = newMatches;
			for (const listener of listeners) {
				listener({ matches: newMatches, media: "(prefers-color-scheme: dark)" } as MediaQueryListEvent);
			}
		},
	};
	return mql;
}

function stubMatchMedia(mql: MediaQueryListMock): void {
	const factory = vi.fn(() => mql as unknown as MediaQueryList);
	vi.stubGlobal("matchMedia", factory);
	// jsdom's window.matchMedia must be stubbed on `window` too (some call sites go through the global).
	Object.defineProperty(window, "matchMedia", {
		configurable: true,
		writable: true,
		value: factory,
	});
}

function spyStorage() {
	return {
		getItem: vi.spyOn(Storage.prototype, "getItem"),
		setItem: vi.spyOn(Storage.prototype, "setItem"),
	};
}

describe("ThemeToggle — initial resolution", () => {
	beforeEach(() => {
		document.documentElement.classList.remove("dark");
	});

	afterEach(() => {
		document.documentElement.classList.remove("dark");
		vi.restoreAllMocks();
		vi.unstubAllGlobals();
	});

	it("applies the dark class and renders the sun glyph when stored preference is 'dark'", () => {
		spyStorage().getItem.mockReturnValue("dark");
		stubMatchMedia(createMatchMediaMock(false));

		const { container } = render(<ThemeToggle />);

		expect(document.documentElement.classList.contains("dark")).toBe(true);
		expect(screen.getByRole("button", { name: "Switch to light theme" })).toBeInTheDocument();
		// Lucide `Sun` renders an svg with the class "lucide-sun".
		expect(container.querySelector("svg")).not.toBeNull();
	});

	it("removes the dark class and renders the moon glyph when stored preference is 'light'", () => {
		document.documentElement.classList.add("dark"); // Pre-hydration may have added it.
		spyStorage().getItem.mockReturnValue("light");
		stubMatchMedia(createMatchMediaMock(true));

		render(<ThemeToggle />);

		expect(document.documentElement.classList.contains("dark")).toBe(false);
		expect(screen.getByRole("button", { name: "Switch to dark theme" })).toBeInTheDocument();
	});

	it("resolves to dark when no stored value and system prefers dark", () => {
		spyStorage().getItem.mockReturnValue(null);
		stubMatchMedia(createMatchMediaMock(true));

		render(<ThemeToggle />);

		expect(document.documentElement.classList.contains("dark")).toBe(true);
		expect(screen.getByRole("button", { name: "Switch to light theme" })).toBeInTheDocument();
	});

	it("resolves to light when no stored value and system prefers light", () => {
		spyStorage().getItem.mockReturnValue(null);
		stubMatchMedia(createMatchMediaMock(false));

		render(<ThemeToggle />);

		expect(document.documentElement.classList.contains("dark")).toBe(false);
		expect(screen.getByRole("button", { name: "Switch to dark theme" })).toBeInTheDocument();
	});

	it("does not call localStorage.setItem on initial mount (read-only)", () => {
		const { getItem, setItem } = spyStorage();
		getItem.mockReturnValue(null);
		stubMatchMedia(createMatchMediaMock(true));

		render(<ThemeToggle />);

		expect(setItem).not.toHaveBeenCalled();
	});
});

describe("ThemeToggle — click behavior", () => {
	beforeEach(() => {
		document.documentElement.classList.remove("dark");
	});

	afterEach(() => {
		document.documentElement.classList.remove("dark");
		vi.restoreAllMocks();
		vi.unstubAllGlobals();
	});

	it("flips state, adds the dark class, and persists when toggled from light", () => {
		const { getItem, setItem } = spyStorage();
		getItem.mockReturnValue("light");
		stubMatchMedia(createMatchMediaMock(false));

		render(<ThemeToggle />);

		const button = screen.getByRole("button", { name: "Switch to dark theme" });
		fireEvent.click(button);

		expect(document.documentElement.classList.contains("dark")).toBe(true);
		expect(setItem).toHaveBeenCalledWith("observatory-theme", "dark");
		expect(setItem).toHaveBeenCalledTimes(1);
		expect(screen.getByRole("button", { name: "Switch to light theme" })).toBeInTheDocument();
	});

	it("flips state, removes the dark class, and persists when toggled from dark", () => {
		const { getItem, setItem } = spyStorage();
		getItem.mockReturnValue("dark");
		stubMatchMedia(createMatchMediaMock(true));

		render(<ThemeToggle />);

		const button = screen.getByRole("button", { name: "Switch to light theme" });
		fireEvent.click(button);

		expect(document.documentElement.classList.contains("dark")).toBe(false);
		expect(setItem).toHaveBeenCalledWith("observatory-theme", "light");
	});

	it("toggle overrides system preference and promotes it to stored", () => {
		// System prefers dark, no stored preference.
		const { getItem, setItem } = spyStorage();
		getItem.mockReturnValue(null);
		const mql = createMatchMediaMock(true);
		stubMatchMedia(mql);

		render(<ThemeToggle />);

		// Initially dark (system).
		expect(document.documentElement.classList.contains("dark")).toBe(true);

		// Click once → light, now stored.
		fireEvent.click(screen.getByRole("button", { name: "Switch to light theme" }));
		expect(document.documentElement.classList.contains("dark")).toBe(false);
		expect(setItem).toHaveBeenCalledWith("observatory-theme", "light");

		// After toggling, system change should be IGNORED (stored preference wins).
		act(() => {
			mql._fire(true);
		});
		expect(document.documentElement.classList.contains("dark")).toBe(false);
	});
});

describe("ThemeToggle — system-change subscription", () => {
	beforeEach(() => {
		document.documentElement.classList.remove("dark");
	});

	afterEach(() => {
		document.documentElement.classList.remove("dark");
		vi.restoreAllMocks();
		vi.unstubAllGlobals();
	});

	it("follows OS changes when no stored preference", () => {
		spyStorage().getItem.mockReturnValue(null);
		const mql = createMatchMediaMock(false);
		stubMatchMedia(mql);

		render(<ThemeToggle />);

		expect(document.documentElement.classList.contains("dark")).toBe(false);

		// Simulate OS change to dark. Wrap in `act` so the state update commits.
		act(() => {
			mql._fire(true);
		});
		expect(document.documentElement.classList.contains("dark")).toBe(true);
		expect(screen.getByRole("button", { name: "Switch to light theme" })).toBeInTheDocument();

		// OS changes back to light.
		act(() => {
			mql._fire(false);
		});
		expect(document.documentElement.classList.contains("dark")).toBe(false);
		expect(screen.getByRole("button", { name: "Switch to dark theme" })).toBeInTheDocument();
	});

	it("removes the change listener on unmount", () => {
		spyStorage().getItem.mockReturnValue(null);
		const mql = createMatchMediaMock(false);
		stubMatchMedia(mql);

		const { unmount } = render(<ThemeToggle />);

		expect(mql.addEventListener).toHaveBeenCalledWith("change", expect.any(Function));
		unmount();
		expect(mql.removeEventListener).toHaveBeenCalledWith("change", expect.any(Function));
	});

	it("does not subscribe to OS changes when a stored preference exists", () => {
		spyStorage().getItem.mockReturnValue("dark");
		const mql = createMatchMediaMock(false);
		stubMatchMedia(mql);

		render(<ThemeToggle />);

		expect(mql.addEventListener).not.toHaveBeenCalled();
	});
});

describe("ThemeToggle — localStorage guardrails", () => {
	beforeEach(() => {
		document.documentElement.classList.remove("dark");
	});

	afterEach(() => {
		document.documentElement.classList.remove("dark");
		vi.restoreAllMocks();
		vi.unstubAllGlobals();
	});

	it("renders without crashing when localStorage.getItem throws", () => {
		vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
			throw new Error("Safari private mode");
		});
		stubMatchMedia(createMatchMediaMock(false));

		expect(() => render(<ThemeToggle />)).not.toThrow();
		// Without access to storage, the component falls back to system preference (light here).
		expect(screen.getByRole("button", { name: "Switch to dark theme" })).toBeInTheDocument();
	});

	it("still flips in-memory state when localStorage.setItem throws", () => {
		vi.spyOn(Storage.prototype, "getItem").mockReturnValue(null);
		vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
			throw new Error("Quota exceeded");
		});
		stubMatchMedia(createMatchMediaMock(false));

		render(<ThemeToggle />);

		const button = screen.getByRole("button", { name: "Switch to dark theme" });
		expect(() => fireEvent.click(button)).not.toThrow();
		expect(document.documentElement.classList.contains("dark")).toBe(true);
		expect(screen.getByRole("button", { name: "Switch to light theme" })).toBeInTheDocument();
	});
});

describe("ThemeToggle — persistence across remount", () => {
	beforeEach(() => {
		document.documentElement.classList.remove("dark");
	});

	afterEach(() => {
		document.documentElement.classList.remove("dark");
		vi.restoreAllMocks();
		vi.unstubAllGlobals();
	});

	it("remounts with the value that was written on a prior toggle", () => {
		// Simulate a backing store by mirroring setItem into getItem.
		let stored: string | null = null;
		vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => stored);
		vi.spyOn(Storage.prototype, "setItem").mockImplementation((_key, value) => {
			stored = value;
		});
		stubMatchMedia(createMatchMediaMock(false));

		const first = render(<ThemeToggle />);
		// Starts light (no stored, system-light). Click → stores "dark".
		fireEvent.click(screen.getByRole("button", { name: "Switch to dark theme" }));
		expect(stored).toBe("dark");
		first.unmount();

		document.documentElement.classList.remove("dark");
		render(<ThemeToggle />);

		// Second mount reads "dark" from storage and applies it.
		expect(document.documentElement.classList.contains("dark")).toBe(true);
		expect(screen.getByRole("button", { name: "Switch to light theme" })).toBeInTheDocument();
	});

	it("removing the stored value returns to system-follow behavior on remount", () => {
		let stored: string | null = "dark";
		vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => stored);
		stubMatchMedia(createMatchMediaMock(false));

		const first = render(<ThemeToggle />);
		expect(document.documentElement.classList.contains("dark")).toBe(true);
		first.unmount();

		// Clear the stored preference so the next mount falls back to system (light).
		stored = null;
		document.documentElement.classList.remove("dark");

		render(<ThemeToggle />);

		expect(document.documentElement.classList.contains("dark")).toBe(false);
		expect(screen.getByRole("button", { name: "Switch to dark theme" })).toBeInTheDocument();
	});
});

describe("ThemeToggle — axe accessibility", () => {
	beforeEach(() => {
		document.documentElement.classList.remove("dark");
	});

	afterEach(() => {
		document.documentElement.classList.remove("dark");
		vi.restoreAllMocks();
		vi.unstubAllGlobals();
	});

	it("passes axe in the light state", async () => {
		vi.spyOn(Storage.prototype, "getItem").mockReturnValue("light");
		stubMatchMedia(createMatchMediaMock(false));

		const { container } = render(<ThemeToggle />);
		expect(await axe(container)).toHaveNoViolations();
	});

	it("passes axe in the dark state", async () => {
		vi.spyOn(Storage.prototype, "getItem").mockReturnValue("dark");
		stubMatchMedia(createMatchMediaMock(true));

		const { container } = render(<ThemeToggle />);
		expect(await axe(container)).toHaveNoViolations();
	});
});
