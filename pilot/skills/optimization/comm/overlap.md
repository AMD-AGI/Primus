# Compute / Comm Overlap

**Status**: Stub

`T_overlap = α_overlap × min(T_comm_overlappable, T_comp_spare)`. α_overlap is calibrated (§S1).

## TODO

- [ ] Overlap modes by framework (Megatron / TorchTitan)
- [ ] alltoall overlap for MoE (cross-reference moe/dispatch.md)
- [ ] Required env flags (NCCL_*)
