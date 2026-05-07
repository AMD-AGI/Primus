# Worker: EnvSweep

**Status**: Stub  
**Extends**: `_envelope.md`

Goal: lock outer baseline structure, sweep weakly_local env axes, emit `EnvSweepResult` (§8.5). Per-call cap: ≤ 5 flags, ≤ 8 combinations, ≤ 50 step.

## Skill scope

- `skills/workflow/envsweep.md`
- `skills/env/SKILL.md`
- `skills/env/{rccl,hsa,alloc,threading,presets}.md` (only the catalogs related to the bottleneck)
- `skills/optimization/{<bottleneck>}/env.md` (priority pointer for this bottleneck)
- `skills/profiling/env_probe.md` (safe-probe protocol)
- `skills/constraints/env.md`

## Tools

- `constraint.check_env()` (gate dangerous combos before running)
- `env_probe.sweep()` (run the parallel short candidates)
- `state.checkpoint()`

## Output

`SubagentResult.artifacts[]` → `state/round_<N>/envsweep_<id>.yaml`.
`summary.headline`: "best NCCL_BUFFSIZE=16M+MIN_NCH=16, +4.7% over baseline, cost 0.3 GPU·h".
