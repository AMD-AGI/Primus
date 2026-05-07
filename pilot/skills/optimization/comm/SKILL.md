# Communication Optimization

**Status**: Stub

Trigger: Diagnose returns COMM_BOUND (`comm_ratio` > threshold).

Primary moves (in order of cheapness):
1. Bucket tuning — see `bucket.md`
2. Overlap (compute/comm) — see `overlap.md`
3. Topology adjustment — see `topology.md`
4. env candidates — see `env.md` (refers to `skills/env/rccl.md`)
