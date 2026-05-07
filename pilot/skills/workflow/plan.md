# Plan Structure

**Status**: Stub

Defines the Plan schema (§8.2): `parallelism`, `runtime`, `comm`, `env.diff` (scale-aware), `predicted`, `generated_by`. The Plan is the unit of execution; only env diff (not full env) is stored.

## TODO

- [ ] Field-by-field semantics
- [ ] env.diff vs env_baseline merge rules
- [ ] predicted block sourcing (from Execution Model, §6 + §S1)
