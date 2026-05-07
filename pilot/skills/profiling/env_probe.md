# Env Safe-Probe Protocol

**Status**: Stub

Three-tier safe probe before any env value is committed to baseline:
1. Connectivity (`nccl-tests` 2-node sanity, < 30s)
2. Micro-bench (RCCL ar/a2a curves)
3. Multi-node short run (1 node × 100 step)

Failure at any tier → reject the value, log to `state/blacklist.yaml` (§S3.5) when applicable.

## TODO

- [ ] Per-tier pass criteria
- [ ] Timeout budgets
- [ ] Fail-fast escalation rules
