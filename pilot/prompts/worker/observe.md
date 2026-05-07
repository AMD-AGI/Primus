# Worker: Observe

**Status**: Stub  
**Extends**: `_envelope.md`

Goal: take a running job's `run_id` and produce a `Snapshot` (§8.3).

## Skill scope

- `skills/workflow/observe.md`
- `skills/profiling/gpu.md`
- `skills/profiling/trace.md`

## Tools

- `observe.snapshot(run_id)`
- `state.checkpoint()`

## Output

`SubagentResult.artifacts[]` → `state/round_<N>/snapshot_<plan_id>.yaml`.
`summary.headline`: "tps=17800 comm_ratio=0.18 bubble=0.09 mem=158GB".
