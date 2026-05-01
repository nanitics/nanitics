/**
 * Loading-skeleton components used by the first-impression views.
 *
 * The three exported variants (`<RunListSkeleton>`, `<RunDetailSkeleton>`,
 * `<AgentDetailSkeleton>`) each render a shape that mirrors the real view so
 * that the transition from skeleton to content is visually stable. All three
 * share the same accessibility contract — the outer container carries
 * `role="status"`, `aria-busy="true"`, and a visually-hidden "Loading…" string
 * so that assistive technologies announce the loading state.
 *
 * Pulse animation is provided by Tailwind's `animate-pulse`; no additional
 * animation dependency is introduced. All placeholder blocks use the
 * `bg-muted` token so the skeleton stays theme-aware.
 */

interface RunListSkeletonProps {
	count?: number;
	className?: string;
}

interface SkeletonProps {
	className?: string;
}

/** Visually-hidden "Loading…" text rendered in every skeleton for screen readers. */
function LoadingSrText() {
	return <span className="sr-only">Loading…</span>;
}

/**
 * N `RunCard`-shaped shells matching the outer border, padding, and column
 * layout of the real card. Used as the first-paint state of the run list.
 */
export function RunListSkeleton({ count = 5, className }: RunListSkeletonProps) {
	const containerClasses = `space-y-2${className ? ` ${className}` : ""}`;
	return (
		<div role="status" aria-busy="true" className={containerClasses}>
			<LoadingSrText />
			{Array.from({ length: count }, (_, i) => (
				// biome-ignore lint/suspicious/noArrayIndexKey: fixed-length skeleton row, no identity.
				<div key={i} className="border rounded-lg px-4 py-3">
					<div className="h-4 bg-muted rounded w-48 animate-pulse" />
					<div className="h-3 bg-muted rounded w-32 mt-1 animate-pulse" />
					<div className="flex gap-4 mt-2">
						<div className="h-3 bg-muted rounded w-16 animate-pulse" />
						<div className="h-3 bg-muted rounded w-16 animate-pulse" />
						<div className="h-3 bg-muted rounded w-16 animate-pulse" />
					</div>
				</div>
			))}
		</div>
	);
}

/**
 * Indent levels (in px) for the eight tree-row shells, matching the visual
 * nesting density of a typical `TraceTree` render.
 */
const TREE_ROW_INDENTS = [0, 20, 20, 40, 40, 60, 40, 20];

/**
 * Renders the eight indented tree-row shells used by `<RunDetailSkeleton>`.
 * Exported from this module (but not re-exported from `src/index.ts`) so the
 * Run Detail page can compose a real header shell with a tree-only skeleton
 * underneath — the "header-shell-first" behavior. Consumers outside the
 * library surface should prefer the full `<RunDetailSkeleton>`.
 */
export function TreeSkeleton({ className }: SkeletonProps) {
	const containerClasses = `px-4 py-2${className ? ` ${className}` : ""}`;
	return (
		<div role="status" aria-busy="true" className={containerClasses}>
			<LoadingSrText />
			{TREE_ROW_INDENTS.map((indent, i) => (
				<div
					// biome-ignore lint/suspicious/noArrayIndexKey: fixed-length skeleton row, no identity.
					key={i}
					className="h-6 bg-muted rounded my-1 animate-pulse"
					style={{ marginLeft: `${indent}px` }}
				/>
			))}
		</div>
	);
}

/**
 * Full-page skeleton for the Run Detail view: header shell, filter toolbar
 * shell, and eight tree-row shells. The header shell is deliberately
 * high-fidelity so the hand-off to the real header is visually stable.
 */
export function RunDetailSkeleton({ className }: SkeletonProps) {
	const containerClasses = `flex flex-col h-full${className ? ` ${className}` : ""}`;
	return (
		<div role="status" aria-busy="true" className={containerClasses}>
			<LoadingSrText />
			{/* Header shell */}
			<div className="border-b px-4 py-3 flex items-center gap-4">
				<div className="h-4 w-16 bg-muted rounded animate-pulse" />
				<div className="flex-1 min-w-0 flex items-center gap-2">
					<div className="h-5 w-64 bg-muted rounded animate-pulse" />
					<div className="h-5 w-16 bg-muted rounded-full animate-pulse" />
				</div>
				<div className="flex items-center gap-4">
					<div className="h-3 w-20 bg-muted rounded animate-pulse" />
					<div className="h-3 w-20 bg-muted rounded animate-pulse" />
					<div className="h-3 w-20 bg-muted rounded animate-pulse" />
					<div className="h-3 w-20 bg-muted rounded animate-pulse" />
				</div>
			</div>
			{/* Filter toolbar shell */}
			<div className="border-b px-4 py-2 flex items-center gap-4">
				<div className="h-8 w-32 bg-muted rounded animate-pulse" />
				<div className="h-8 w-32 bg-muted rounded animate-pulse" />
			</div>
			{/* Tree-row shells */}
			<div className="px-4 py-2">
				{TREE_ROW_INDENTS.map((indent, i) => (
					<div
						// biome-ignore lint/suspicious/noArrayIndexKey: fixed-length skeleton row, no identity.
						key={i}
						className="h-6 bg-muted rounded my-1 animate-pulse"
						style={{ marginLeft: `${indent}px` }}
					/>
				))}
			</div>
		</div>
	);
}

/**
 * Full-page skeleton for the Agent Detail view: breadcrumb shell, agent
 * header (name + capability chips), stats row, and six timeline-row shells.
 */
export function AgentDetailSkeleton({ className }: SkeletonProps) {
	const containerClasses = `flex flex-col h-full${className ? ` ${className}` : ""}`;
	return (
		<div role="status" aria-busy="true" className={containerClasses}>
			<LoadingSrText />
			{/* Breadcrumb shell */}
			<div className="border-b px-4 py-2">
				<div className="h-4 w-64 bg-muted rounded animate-pulse" />
			</div>
			{/* Agent header shell */}
			<div className="border-b px-4 py-3">
				<div className="flex items-center gap-2 flex-wrap">
					<div className="h-5 w-48 bg-muted rounded animate-pulse" />
					<div className="h-4 w-16 bg-muted rounded animate-pulse" />
					<div className="h-4 w-16 bg-muted rounded animate-pulse" />
					<div className="h-4 w-16 bg-muted rounded animate-pulse" />
				</div>
				{/* Stats row */}
				<div className="flex items-center gap-4 mt-2 flex-wrap">
					<div className="h-3 w-20 bg-muted rounded animate-pulse" />
					<div className="h-3 w-20 bg-muted rounded animate-pulse" />
					<div className="h-3 w-20 bg-muted rounded animate-pulse" />
					<div className="h-3 w-20 bg-muted rounded animate-pulse" />
					<div className="h-3 w-20 bg-muted rounded animate-pulse" />
					<div className="h-3 w-20 bg-muted rounded animate-pulse" />
				</div>
			</div>
			{/* Timeline rows */}
			<div className="px-4 py-2">
				{Array.from({ length: 6 }, (_, i) => (
					// biome-ignore lint/suspicious/noArrayIndexKey: fixed-length skeleton row, no identity.
					<div key={i} className="h-6 bg-muted rounded my-1 animate-pulse" />
				))}
			</div>
		</div>
	);
}
