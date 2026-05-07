# Worker: Re-Plan

**Status**: Stub  
**Extends**: `_envelope.md`

Goal: read DiagnosisReport + PlanGraph, generate a `CandidatePool` (§8.10) with priority scores and Strategy Select output.

The heaviest worker — has the largest Skill scope. Peak budget < 30K tokens.

## Skill scope

- `skills/workflow/replan.md`
- `skills/workflow/plan_graph.md`
- `skills/workflow/axis_taxonomy.md`
- `skills/workflow/execution_strategy.md`
- `skills/optimization/{<bottleneck>}/`  (only the bottleneck subdir reported by Diagnose)
- `skills/env/SKILL.md` + the relevant flag catalog (`rccl.md` / `alloc.md` / ...)
- `skills/knowledge/SKILL.md` (consult cases / patterns; respect retired/superseded filter)

## Tools

- `state.resume()` to read DiagnosisReport, PlanGraph, CandidatePool history
- `constraint.check()`, `constraint.estimate_mem()`, `constraint.check_env()`
- (Optional, P0) `predict.gain()` / `predict.mem()` from §S1 calibration

## Output

`SubagentResult.artifacts[]` → `state/round_<N>/candidate_pool.yaml`.
`summary.headline`: "COMPUTE_BOUND, 3 exploit + 1 explore candidates, top priority=1.55".
