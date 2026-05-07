# Diagnose (Bottleneck Classification)

**Status**: Stub

Map Snapshot → DiagnosisReport (§8.4). Bottleneck classes: COMM / PIPELINE / MEMORY / COMPUTE. Also emits `env_suspect[]` (which triggers EnvSweep) and `candidate_axes[]` (consumed by Re-Plan).

## TODO

- [ ] Threshold table (comm_ratio / bubble_ratio / mem / gpu_util cutoffs)
- [ ] env_suspect detection rules (per env catalog hints)
- [ ] reentry triggers: ClusterProfile age > 7d → PREFLIGHT; structural mismatch → PROJECTION
- [ ] suggested_strategy mapping (Champion-Challenger / Per-Plan / Successive Halving)
