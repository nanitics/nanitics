# Nanitics Observatory — local-dev compose

One-service Docker compose that runs a Nanitics app with the embedded
Observatory UI. Local development and demos only, not production.

Prerequisite: the embed bundle at `observatory/dist-embed/` must exist.
Run `just observatory-build` first if it is stale or missing.

Run (from the repo root):

```sh
just observatory-compose
```

UI at <http://localhost:8001/api/observatory/>.

See [`docs/guides/observatory-integration.md`](../../docs/guides/observatory-integration.md)
for mounting the backend in your own app, frontend wiring, and the
"for production" checklist.
