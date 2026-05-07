# Workflow — Tuning Loop Overview

**Status**: Stub
**Read by**: Orchestrator + every Stage Worker
**Written by**: humans
**Domain**: tuning main flow

## Purpose

Entry point for the tuning workflow. Describes the outer/inner loop split and points to the per-stage Skills:

- `state_machine.md` — state set, transitions, reentry, on_fail
- `orchestration.md` — Orchestrator ↔ Stage Worker protocol & context hygiene
- `projection.md` / `smoke.md` / `correctness.md` / ...
- `replan.md` / `settle.md` / `envsweep.md` / ...

## TODO

- [ ] Outer-vs-inner split criteria (when EnvSweep triggers, when not)
- [ ] Round budget accounting
- [ ] Cross-references to all sibling Skill files
