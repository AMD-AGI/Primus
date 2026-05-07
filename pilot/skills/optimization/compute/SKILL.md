# Compute Optimization

**Status**: Stub

Trigger: Diagnose returns COMPUTE_BOUND (`gpu_util` low, no other bottleneck dominant).

Primary moves:
1. mbs scaling — see `mbs.md`
2. dp/tp adjustment — see `parallel.md`
3. Kernel hints — see `kernel.md`
4. env candidates — see `env.md`
