# Memory Optimization

**Status**: Stub

Trigger: predicted or observed `mem_peak_gb` close to `hbm_capacity_gb` (e.g. > 90%), or OOM failure.

Primary moves:
1. Activation recompute — see `recompute.md`
2. Offload (CPU / NVMe) — see `offload.md`
3. Fragmentation mitigation — see `fragmentation.md`
4. env candidates — see `env.md` (refers to `skills/env/alloc.md`)
