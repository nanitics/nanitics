import { AlertCircle, RefreshCw } from "lucide-react";

interface ErrorStateProps {
	title?: string;
	error: unknown;
	onRetry?: () => void;
	variant?: "page" | "inline";
	className?: string;
}

/**
 * Produce a developer-facing string from an arbitrary error value.
 * `Error` → message plus stack; `string` passes through; everything else is
 * JSON-serialized, with a final `String(error)` fallback for circular or
 * otherwise un-serializable values so rendering never throws.
 */
function stringifyError(error: unknown): string {
	if (error instanceof Error) return error.stack ? `${error.message}\n\n${error.stack}` : error.message;
	if (typeof error === "string") return error;
	try {
		return JSON.stringify(error, null, 2);
	} catch {
		return String(error);
	}
}

/**
 * User-facing error state used by page-level and inline failure surfaces.
 *
 * The outer container sets `role="alert"` so screen readers announce the
 * failure on insertion. The lucide glyph is decorative (`aria-hidden`); the
 * title carries the semantics. The `<details>` pane hides the raw error
 * behind a disclosure for developer inspection.
 *
 * `variant="page"` is the default full-shell presentation used when the
 * entire view has failed to load. `variant="inline"` is a compact row used
 * for non-blocking notices (e.g. a secondary panel that couldn't load);
 * inline notices are read-only and suppress the retry button even if
 * `onRetry` is provided.
 */
export function ErrorState({
	title = "Something went wrong",
	error,
	onRetry,
	variant = "page",
	className,
}: ErrorStateProps) {
	const details = stringifyError(error);

	if (variant === "inline") {
		const inlineClasses = `flex flex-row items-start gap-2 text-xs py-2 px-4${className ? ` ${className}` : ""}`;
		return (
			<div role="alert" className={inlineClasses}>
				<AlertCircle className="h-4 w-4 text-destructive flex-shrink-0 mt-0.5" aria-hidden="true" />
				<div className="flex flex-col gap-1 min-w-0">
					<span>{title}</span>
					<details className="text-[11px]">
						<summary className="cursor-pointer text-muted-foreground">Details</summary>
						<pre className="mt-1 whitespace-pre-wrap font-mono text-muted-foreground">{details}</pre>
					</details>
				</div>
			</div>
		);
	}

	const pageClasses = `flex flex-col items-center justify-center py-12 text-center${className ? ` ${className}` : ""}`;
	return (
		<div role="alert" className={pageClasses}>
			<AlertCircle className="h-8 w-8 text-destructive" aria-hidden="true" />
			<h2 className="text-base font-semibold mt-3">{title}</h2>
			<p className="text-sm text-muted-foreground mt-1">We couldn't load this view.</p>
			<details className="mt-3 max-w-xl w-full">
				<summary className="cursor-pointer text-xs text-muted-foreground">Details</summary>
				<pre className="mt-2 whitespace-pre-wrap font-mono text-xs text-muted-foreground text-left">{details}</pre>
			</details>
			{onRetry && (
				<button
					type="button"
					onClick={onRetry}
					className="mt-4 inline-flex items-center gap-1.5 rounded-md bg-primary text-primary-foreground px-3 py-1.5 text-xs font-medium transition-colors hover:bg-primary/90"
				>
					<RefreshCw className="h-4 w-4" aria-hidden="true" />
					Retry
				</button>
			)}
		</div>
	);
}
