# Machine-readable schemas

- `run.schema.json` validates the complete `run.json` emitted by each benchmark run.
- `peak_summary.schema.json` validates the compact `peak_summary.json` headline report.

Both currently use `schema_version = 1`. Any future incompatible output change should increment that version instead of silently changing field semantics.
