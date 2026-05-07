# Execution Model

**Status**: Stub
**Read by**: Projection / Re-Plan / Diagnose Workers
**Domain**: training step-time and memory modeling

## Purpose

Closed-form / piecewise formulas to estimate `T_step` and `Mem_peak` for a candidate Plan, used by Re-Plan to score `expected_gain` and by Constraint to filter OOM-risky configs.

```
T_step = T_comp + T_comm + T_bubble - T_overlap
Mem    = M_param + M_grad + M_optim + M_act + M_buffer
```

See sub-Skills:
- `compute.md` — T_comp(layers, mbs)
- `memory.md` — Mem(layers, mbs)
- `communication.md` — T_comm / allreduce / alltoall
- `pipeline.md` — Bubble(pp, M)
- `partition.md` — layer partition / stage balance
- `examples.md` — Dense / MoE worked examples

> Calibration of free parameters (η_comp, α_overlap, β_bubble, γ_act, ...) is specified in `README.cn.supplements.md` §S1.
