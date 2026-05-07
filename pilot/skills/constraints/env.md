# Env Incompatibility Matrix

**Status**: Stub

Mutually exclusive / dangerous env combinations. Consumed by `constraint.check_env()`.

## TODO

- [ ] Known incompatibilities (e.g. RCCL_MSCCL_ENABLE × NCCL_NET_GDR_LEVEL=4 with certain firmware)
- [ ] Per-flag value ranges (refuse out-of-range)
- [ ] Cross-reference: skills/env/* catalog entries
