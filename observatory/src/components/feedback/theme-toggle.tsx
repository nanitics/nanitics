import { Moon, Sun } from "lucide-react";
import { useEffect, useRef, useState } from "react";

type Theme = "light" | "dark";

const STORAGE_KEY = "observatory-theme";
const DARK_MEDIA_QUERY = "(prefers-color-scheme: dark)";

function readStoredTheme(): Theme | null {
	try {
		const raw = window.localStorage.getItem(STORAGE_KEY);
		if (raw === "dark" || raw === "light") return raw;
		return null;
	} catch {
		return null;
	}
}

function writeStoredTheme(theme: Theme): boolean {
	try {
		window.localStorage.setItem(STORAGE_KEY, theme);
		return true;
	} catch {
		return false;
	}
}

function applyThemeClass(theme: Theme): void {
	document.documentElement.classList.toggle("dark", theme === "dark");
}

/**
 * Header affordance that flips the Observatory between light and dark themes.
 *
 * On mount, resolves the initial theme from `localStorage['observatory-theme']`
 * when present, else from `prefers-color-scheme`. The resolved theme is applied
 * to `document.documentElement` via the `dark` class so the pre-hydration
 * script and the runtime toggle share the same mechanism.
 *
 * When no stored preference exists, the component subscribes to the media
 * query's `change` event and follows OS-level theme changes until the user
 * manually picks a theme — at which point the preference is persisted and the
 * subscription is released.
 *
 * Storage access is wrapped in try/catch for Safari private mode and similar
 * environments where `localStorage` throws; in that case the component still
 * works but persistence is in-memory only.
 */
export function ThemeToggle() {
	const [theme, setTheme] = useState<Theme>("light");
	const hasStoredPreferenceRef = useRef<boolean>(false);

	useEffect(() => {
		const stored = readStoredTheme();
		if (stored !== null) {
			hasStoredPreferenceRef.current = true;
			setTheme(stored);
			applyThemeClass(stored);
			return;
		}

		const mql = window.matchMedia(DARK_MEDIA_QUERY);
		const initial: Theme = mql.matches ? "dark" : "light";
		setTheme(initial);
		applyThemeClass(initial);

		const onChange = (event: MediaQueryListEvent) => {
			if (hasStoredPreferenceRef.current) return;
			const next: Theme = event.matches ? "dark" : "light";
			setTheme(next);
			applyThemeClass(next);
		};

		mql.addEventListener("change", onChange);
		return () => {
			mql.removeEventListener("change", onChange);
		};
	}, []);

	const handleClick = () => {
		const next: Theme = theme === "dark" ? "light" : "dark";
		setTheme(next);
		applyThemeClass(next);
		const persisted = writeStoredTheme(next);
		if (persisted) {
			hasStoredPreferenceRef.current = true;
		}
	};

	const isDark = theme === "dark";
	const ariaLabel = isDark ? "Switch to light theme" : "Switch to dark theme";

	return (
		<button
			type="button"
			onClick={handleClick}
			aria-label={ariaLabel}
			className="inline-flex items-center justify-center text-foreground hover:bg-accent rounded-md p-2 transition-colors"
		>
			{isDark ? <Sun className="h-5 w-5" aria-hidden="true" /> : <Moon className="h-5 w-5" aria-hidden="true" />}
		</button>
	);
}
