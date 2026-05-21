# @nanitics/observatory

React components for embedding the [Nanitics](https://github.com/nanitics/nanitics) Observatory — run list, run detail, event timeline — inside downstream apps.

## Install

```bash
npm install @nanitics/observatory
```

Peer deps: `react@^19.2.6`, `react-dom@^19.2.6`.

## Use

```tsx
import { useState } from "react";
import {
  ObservatoryClient,
  ObservatoryProvider,
  RunListPage,
  RunDetailPage,
  createDefaultRegistries,
} from "@nanitics/observatory";
import "@nanitics/observatory/styles.css";

const client = new ObservatoryClient("/api/observatory");
const { registry, agentViewRegistry, panelRegistry } = createDefaultRegistries();

export function App() {
  const [runId, setRunId] = useState<string | null>(null);
  return (
    <ObservatoryProvider
      client={client}
      registry={registry}
      agentViewRegistry={agentViewRegistry}
      panelRegistry={panelRegistry}
    >
      {runId
        ? <RunDetailPage runId={runId} onBack={() => setRunId(null)} />
        : <RunListPage onSelectRun={setRunId} />}
    </ObservatoryProvider>
  );
}
```

The styles import is required. It ships precompiled Tailwind utilities and the theme tokens (light + dark variants on `<html class="dark">`).

## Server side

The Observatory components talk to the Python SDK's `create_observatory_router(store)` endpoint over HTTP. See the [integration guide](https://github.com/nanitics/nanitics/blob/main/docs/guides/observatory-integration.md) for the full setup.

## Versioning

Pre-1.0. The stable surface is the symbols documented in the integration guide (`ObservatoryProvider`, `RunListPage`, `RunDetailPage`, `createDefaultRegistries`, `AgentViewRegistration`, `CapabilityPanelRegistration`, `useUrlFilters`). Other exports may shift between minor versions.

## License

Apache-2.0
