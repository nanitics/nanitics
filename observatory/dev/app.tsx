import { useCallback, useSyncExternalStore } from "react";
import { ObservatoryClient } from "../src/client/observatory-client";
import { ThemeToggle } from "../src/components/feedback/theme-toggle";
import { ObservatoryProvider } from "../src/context/observatory-context";
import { splitHashRoute } from "../src/hooks/use-hash-query";
import { AgentDetailPage } from "../src/pages/agent-detail-page";
import { RunDetailPage } from "../src/pages/run-detail-page";
import { RunListPage } from "../src/pages/run-list-page";
import { WorkflowDetailPage } from "../src/pages/workflow-detail-page";
import { createDefaultRegistries } from "../src/registry/default-renderers";

// In production the Python router injects window.__NANITICS_OBSERVATORY_BASE__
// so the same bundle works at any mount prefix. The dev index.html primes the
// global to the Vite proxy path so `npm run dev` keeps working.
const client = new ObservatoryClient();
const { registry, agentViewRegistry, panelRegistry } = createDefaultRegistries();

// --- Hash-based routing ---

function getHash(): string {
	return window.location.hash.slice(1) || "/";
}

function useHash(): string {
	const subscribe = useCallback((cb: () => void) => {
		window.addEventListener("hashchange", cb);
		return () => window.removeEventListener("hashchange", cb);
	}, []);
	return useSyncExternalStore(subscribe, getHash, getHash);
}

function navigate(path: string) {
	window.location.hash = path;
}

// --- App Shell ---

export function App() {
	const hash = useHash();

	// Strip the query string from the hash before route matching so URL-state
	// filters (e.g. `#/runs?status=running`) do not break the route regex.
	const { route } = splitHashRoute(hash);

	// Parse routes
	const agentMatch = route.match(/^\/runs\/([^/]+)\/agents\/([^/]+)$/);
	const workflowMatch = !agentMatch ? route.match(/^\/runs\/([^/]+)\/workflow$/) : null;
	const runMatch = !agentMatch && !workflowMatch ? route.match(/^\/runs\/([^/]+)$/) : null;

	const runId = agentMatch?.[1] ?? workflowMatch?.[1] ?? runMatch?.[1] ?? null;
	const agentSpanId = agentMatch?.[2] ?? null;
	const isWorkflow = !!workflowMatch;

	return (
		<ObservatoryProvider
			client={client}
			registry={registry}
			agentViewRegistry={agentViewRegistry}
			panelRegistry={panelRegistry}
		>
			<div className="h-screen flex flex-col bg-background text-foreground">
				<header className="border-b px-6 py-4 flex-shrink-0">
					<div className="max-w-6xl mx-auto flex items-center">
						<h1 className="text-lg font-semibold tracking-tight cursor-pointer" onClick={() => navigate("/")}>
							Nanitics Observatory
						</h1>
						<div className="ml-auto">
							<ThemeToggle />
						</div>
					</div>
				</header>
				{agentSpanId && runId ? (
					<AgentDetailPage
						runId={runId}
						spanId={agentSpanId}
						onBack={() => navigate(`/runs/${runId}`)}
						onBackToRuns={() => navigate("/")}
						onNavigateToAgent={(spanId) => navigate(`/runs/${runId}/agents/${spanId}`)}
					/>
				) : isWorkflow && runId ? (
					<WorkflowDetailPage
						runId={runId}
						onBack={() => navigate(`/runs/${runId}`)}
						onBackToRuns={() => navigate("/")}
						onNavigateToAgent={(spanId) => navigate(`/runs/${runId}/agents/${spanId}`)}
					/>
				) : runId ? (
					<RunDetailPage
						runId={runId}
						onBack={() => navigate("/")}
						onNavigateToAgent={(spanId) => navigate(`/runs/${runId}/agents/${spanId}`)}
						onNavigateToWorkflow={() => navigate(`/runs/${runId}/workflow`)}
					/>
				) : (
					// `RunListPage` owns its own `<main>` so the document has exactly
					// one `<main>` landmark across all routes.
					<RunListPage onSelectRun={(id) => navigate(`/runs/${id}`)} />
				)}
			</div>
		</ObservatoryProvider>
	);
}
