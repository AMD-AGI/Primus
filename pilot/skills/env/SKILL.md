# Env Catalog (Single Source of Truth)

**Status**: Stub
**Read by**: EnvSweep / Re-Plan Workers
**Domain**: per-flag definitions, ranges, known pitfalls

This subtree is the **only** place env flags are fully defined. `optimization/{bottleneck}/env.md` only references entries here, never re-defines.

Catalog files:
- `rccl.md` — NCCL_* / RCCL_*
- `hsa.md` — HSA_* / HIP_* / GPU_MAX_HW_QUEUES
- `alloc.md` — PYTORCH_HIP_ALLOC_CONF / MALLOC_*
- `threading.md` — OMP_* / MKL_* / numactl
- `presets.md` — per-cluster-class validated combinations

Each entry should specify: default / safe range / known pitfalls / interaction with other flags.
