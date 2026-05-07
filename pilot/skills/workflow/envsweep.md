# EnvSweep (Inner Loop)

**Status**: Stub

Inner sweep over weakly_local env axes after structure stabilizes. Per-call cap: ≤ 5 flags, ≤ 8 combinations, ≤ 50 step. Output: EnvSweepResult (§8.5) merged into baseline `env.diff`.

## TODO

- [ ] Trigger condition (env_suspect from Diagnose + structurally stable for 1 round)
- [ ] Candidate generation (per `optimization/{bottleneck}/env.md` + safety check via `constraint.check_env()`)
- [ ] Early termination rules (best Δ found and clear winner)
