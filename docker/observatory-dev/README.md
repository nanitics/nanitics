# Nanitics Observatory — local-dev compose

One-service Docker compose that runs a Nanitics app with the embedded
Observatory UI. Local development and demos only, not production.

Prerequisite: the embedded SPA at `nanitics/observatory/ui_assets/`
must exist (it is `.gitignore`d). Run `just observatory-build` first.

Run (from the repo root):

```sh
just observatory-compose
```

UI at <http://localhost:8001/api/observatory/>.

See [`docs/guides/observatory-integration.md`](../../docs/guides/observatory-integration.md)
for mounting the backend in your own app, frontend wiring, and the
"for production" checklist.
