# Worker: Diagnose

**Status**: Stub  
**Extends**: `_envelope.md`

Goal: read Snapshot, classify bottleneck, emit `DiagnosisReport` (§8.4) with `bottleneck`, `evidence`, `env_suspect[]`, `candidate_axes[]`, `suggested_strategy`.

## Skill scope

- `skills/workflow/diagnose.md`
- `skills/execution-model/SKILL.md` (+ relevant submodules `compute.md` / `communication.md` / `pipeline.md` / `memory.md`)
- `skills/env/SKILL.md` (catalog index, do not pull every flag file)
- `skills/profiling/trace.md`

## Tools

- `state.resume()` to read referenced Snapshot
- (no submit / observe — those are upstream)

## Output

`SubagentResult.artifacts[]` → `state/round_<N>/diagnosis_report.yaml`.
`summary.headline`: "PIPELINE_BOUND, bubble=0.18, env_suspect=[NCCL_BUFFSIZE], suggested=Per-Plan".
