# Advisor validation fixtures

Frozen trace envelopes consumed by the advisor validation scripts. Each
fixture is a trimmed Nanitics trace captured from a real run so the
advisor validation scripts never race an emitter.

## Regenerating `smoke_react_agent.json`

1. Run the smoke validation script against a real provider:

   ```
   just validate validation/smoke/smoke.py
   ```

2. Copy the produced trace into this directory:

   ```
   cp validation/traces/smoke_react_agent.json \
      validation/advisory/fixtures/smoke_react_agent.json
   ```

3. Open the copied file and overwrite the `exported_at` field with
   `"1970-01-01T00:00:00Z"`. Normalizing the timestamp keeps file-hash
   diffs meaningful when the fixture is next refreshed — every other
   field still reflects the real run.

4. Run `uv run pytest validation/advisory/` to confirm the scripts still
   pass against the new fixture.

The fixture is checked in rather than generated on demand so advisor
validation never requires running the smoke script first — the two
suites stay independent.
