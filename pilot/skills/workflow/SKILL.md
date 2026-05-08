# Workflow — Tuning Loop Overview

**Status**: Single-node v1
**Read by**: Orchestrator + every Stage Worker
**Written by**: humans
**Domain**: tuning main flow

## Purpose

Entry point for the tuning workflow. Describes the outer/inner loop split and points to the per-stage Skills:

- `state_machine.md` — state set, transitions, reentry, on_fail
- `orchestration.md` — Orchestrator ↔ Stage Worker protocol & context hygiene
- `projection.md` / `smoke.md` / `correctness.md` / ...
- `replan.md` / `settle.md` / `envsweep.md` / ...

## Single-node v1

For `cluster.yaml mode=single`, the runnable path is:

```
PREFLIGHT/PROJECTION (optional) -> SMOKE -> BASELINE ->
OPTIMIZE_LOOP.{DIAGNOSE,REPLAN,EXECUTE,CORRECTNESS_LITE,SETTLE} -> REPORT -> LEARN draft
```

The non-interactive implementation is `python -m pilot.tools.tune_single run`.
It keeps a flat `run_history`, stores the current champion in
`pilot/state/tuning_state.yaml`, and writes checkpoints at stage exits.

Round budget accounting:

- SMOKE and BASELINE do not increment `budget_used.rounds`.
- Each completed SETTLE stage increments the effective tuning round.
- The loop stops when the configured round count is reached, no candidates are
  left, or SETTLE reports stagnation.

EnvSweep remains out of the single-node v1 loop. It can be reintroduced once
`preflight.env_probe` and `preflight.env_sweep` are promoted beyond stubs.
