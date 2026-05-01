import { describe, expect, it } from "vitest";
import { computeDAGLayout } from "../../src/components/dag/dag-layout";
import type { WorkflowStep } from "../../src/types";

function makeStep(overrides: Partial<WorkflowStep> & { name: string }): WorkflowStep {
	return {
		step_type: "function",
		index: null,
		depends_on: [],
		parallel_group: null,
		status: "pending",
		duration_ms: null,
		agent_span_id: null,
		metadata: {},
		...overrides,
	};
}

describe("computeDAGLayout", () => {
	it("returns empty layout for empty steps", () => {
		const layout = computeDAGLayout([]);
		expect(layout.nodes).toHaveLength(0);
		expect(layout.edges).toHaveLength(0);
		expect(layout.width).toBe(0);
		expect(layout.height).toBe(0);
	});

	it("lays out a single node", () => {
		const steps = [makeStep({ name: "only" })];
		const layout = computeDAGLayout(steps);

		expect(layout.nodes).toHaveLength(1);
		expect(layout.edges).toHaveLength(0);
		expect(layout.nodes[0].id).toBe("only");
		expect(layout.nodes[0].x).toBeDefined();
		expect(layout.nodes[0].y).toBeDefined();
		expect(layout.width).toBeGreaterThan(0);
		expect(layout.height).toBeGreaterThan(0);
	});

	it("lays out a sequential chain with correct edge direction", () => {
		const steps = [
			makeStep({ name: "a", index: 0, duration_ms: 100 }),
			makeStep({ name: "b", index: 1, depends_on: ["a"], duration_ms: 200 }),
			makeStep({ name: "c", index: 2, depends_on: ["b"], duration_ms: 300 }),
		];
		const layout = computeDAGLayout(steps);

		expect(layout.nodes).toHaveLength(3);
		expect(layout.edges).toHaveLength(2);

		// Top-to-bottom layout: a.y < b.y < c.y
		const nodeMap = new Map(layout.nodes.map((n) => [n.id, n]));
		expect(nodeMap.get("a")!.y).toBeLessThan(nodeMap.get("b")!.y!);
		expect(nodeMap.get("b")!.y).toBeLessThan(nodeMap.get("c")!.y!);

		// Edges
		expect(layout.edges).toContainEqual(expect.objectContaining({ source: "a", target: "b" }));
		expect(layout.edges).toContainEqual(expect.objectContaining({ source: "b", target: "c" }));
	});

	it("lays out parallel steps at the same rank", () => {
		const steps = [
			makeStep({ name: "start", index: 0, duration_ms: 50 }),
			makeStep({ name: "p1", index: 1, depends_on: ["start"], parallel_group: "parallel", duration_ms: 100 }),
			makeStep({ name: "p2", index: 2, depends_on: ["start"], parallel_group: "parallel", duration_ms: 150 }),
			makeStep({ name: "end", index: 3, depends_on: ["p1", "p2"], duration_ms: 50 }),
		];
		const layout = computeDAGLayout(steps);

		expect(layout.nodes).toHaveLength(4);
		expect(layout.edges).toHaveLength(4);

		const nodeMap = new Map(layout.nodes.map((n) => [n.id, n]));
		// p1 and p2 should be at the same y level
		expect(nodeMap.get("p1")!.y).toBe(nodeMap.get("p2")!.y);
		// start above parallels, end below
		expect(nodeMap.get("start")!.y).toBeLessThan(nodeMap.get("p1")!.y!);
		expect(nodeMap.get("p1")!.y).toBeLessThan(nodeMap.get("end")!.y!);
	});

	it("lays out a diamond DAG correctly", () => {
		const steps = [
			makeStep({ name: "fetch", duration_ms: 100 }),
			makeStep({ name: "parse", depends_on: ["fetch"], duration_ms: 200 }),
			makeStep({ name: "validate", depends_on: ["fetch"], duration_ms: 50 }),
			makeStep({ name: "combine", depends_on: ["parse", "validate"], duration_ms: 100 }),
		];
		const layout = computeDAGLayout(steps);

		expect(layout.nodes).toHaveLength(4);
		expect(layout.edges).toHaveLength(4);

		const nodeMap = new Map(layout.nodes.map((n) => [n.id, n]));
		expect(nodeMap.get("fetch")!.y).toBeLessThan(nodeMap.get("parse")!.y!);
		expect(nodeMap.get("parse")!.y).toBeLessThan(nodeMap.get("combine")!.y!);
	});

	it("preserves step metadata in DAGNodes", () => {
		const steps = [
			makeStep({
				name: "agent-step",
				step_type: "agent",
				status: "running",
				duration_ms: 500,
				agent_span_id: "span-123",
				parallel_group: "group-a",
				metadata: { custom: "value" },
			}),
		];
		const layout = computeDAGLayout(steps);

		const node = layout.nodes[0];
		expect(node.stepType).toBe("agent");
		expect(node.status).toBe("running");
		expect(node.durationMs).toBe(500);
		expect(node.agentSpanId).toBe("span-123");
		expect(node.parallelGroup).toBe("group-a");
		expect(node.metadata).toEqual({ custom: "value" });
	});

	it("assigns dimensions to each node", () => {
		const steps = [makeStep({ name: "a" }), makeStep({ name: "b", depends_on: ["a"] })];
		const layout = computeDAGLayout(steps, { nodeWidth: 200, nodeHeight: 100 });

		for (const node of layout.nodes) {
			expect(node.width).toBe(200);
			expect(node.height).toBe(100);
		}
	});
});

describe("critical path computation", () => {
	it("marks the critical path in a sequential chain", () => {
		const steps = [
			makeStep({ name: "a", duration_ms: 100 }),
			makeStep({ name: "b", depends_on: ["a"], duration_ms: 200 }),
			makeStep({ name: "c", depends_on: ["b"], duration_ms: 300 }),
		];
		const layout = computeDAGLayout(steps);

		// All edges should be on the critical path (only one path)
		for (const edge of layout.edges) {
			expect(edge.isCriticalPath).toBe(true);
		}
	});

	it("identifies the longer branch as critical path in a diamond", () => {
		const steps = [
			makeStep({ name: "start", duration_ms: 10 }),
			makeStep({ name: "fast", depends_on: ["start"], duration_ms: 50 }),
			makeStep({ name: "slow", depends_on: ["start"], duration_ms: 500 }),
			makeStep({ name: "end", depends_on: ["fast", "slow"], duration_ms: 10 }),
		];
		const layout = computeDAGLayout(steps);

		const edgeMap = new Map(layout.edges.map((e) => [`${e.source}->${e.target}`, e]));

		// Critical path should go through the slow branch: start->slow->end
		expect(edgeMap.get("start->slow")?.isCriticalPath).toBe(true);
		expect(edgeMap.get("slow->end")?.isCriticalPath).toBe(true);

		// Fast branch should not be critical
		expect(edgeMap.get("start->fast")?.isCriticalPath).toBe(false);
		expect(edgeMap.get("fast->end")?.isCriticalPath).toBe(false);
	});

	it("handles nodes with null duration as zero weight", () => {
		const steps = [
			makeStep({ name: "a", duration_ms: null }),
			makeStep({ name: "b", depends_on: ["a"], duration_ms: 100 }),
		];
		const layout = computeDAGLayout(steps);

		// Only one path, both edges critical
		for (const edge of layout.edges) {
			expect(edge.isCriticalPath).toBe(true);
		}
	});

	it("returns no critical edges for a single node", () => {
		const steps = [makeStep({ name: "solo", duration_ms: 100 })];
		const layout = computeDAGLayout(steps);

		expect(layout.edges).toHaveLength(0);
	});
});
