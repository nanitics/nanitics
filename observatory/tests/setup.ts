import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, expect } from "vitest";
// Note: `vitest-axe/extend-expect` ships empty in the installed version, so we
// register the matcher ourselves. The package also re-exports `toHaveNoViolations`
// as a type-only symbol (`export type *`) in its public `matchers.d.ts`, while
// the actual value is present in `matchers.js`. We import the runtime value via
// the matchers subpath with a cast to bridge the broken typing.
import * as vitestAxeMatchers from "vitest-axe/matchers";

declare module "vitest" {
	interface Assertion {
		toHaveNoViolations(): void;
	}
	interface AsymmetricMatchersContaining {
		toHaveNoViolations(): void;
	}
}

type RawMatcherFn = Parameters<typeof expect.extend>[0]["toHaveNoViolations"];
const { toHaveNoViolations } = vitestAxeMatchers as unknown as { toHaveNoViolations: RawMatcherFn };

expect.extend({ toHaveNoViolations });

afterEach(() => {
	cleanup();
});
