# T_comm Modeling

**Status**: Stub

`T_comm = AllReduce(grad_size/dp) + AllToAll(moe_msg) + AllGather(zero_shard) + ReduceScatter(zero_shard) + Broadcast(init)`

## Bandwidth lookup (schema 2.0)

`ClusterProfile.rccl_baseline` is scoped — pick the right scope per collective by parallelism shape, then interpolate over `sizes_mb` axis:

| Parallelism / collective | Scope | Path |
|--------------------------|-------|------|
| TP AllReduce / AllGather | `intra_node` (TP usually fits in one node) | `rccl_baseline.intra_node.collectives.<coll>.roll_up.median_bw_gbs` |
| EP AllToAll (EP ≤ gpus_per_node) | `intra_node` | same as above |
| EP AllToAll (EP > gpus_per_node) | `world` | `rccl_baseline.world.collectives.alltoall.bw_gbs` |
| FSDP/ZeRO AllGather + ReduceScatter | `world` (or `intra_node` if shard ≤ 1 node) | `rccl_baseline.world.collectives.<coll>.bw_gbs` |
| DP AllReduce | `world` | `rccl_baseline.world.collectives.allreduce.bw_gbs` |
| Init Broadcast | `world` | `rccl_baseline.world.collectives.broadcast.bw_gbs` |
| PP send/recv | (not in rccl_baseline yet — derive from `inter_node` AR×0.7 as upper bound) | — |

**For `intra_node` always read `roll_up.median_bw_gbs`** for typical case; switch to `roll_up.min_bw_gbs` for worst-case bounds when projecting hung/straggler scenarios.

**For `inter_node` / `world`** read `bw_gbs` directly (single ring, no per-node aggregation).

## TODO

- [ ] Per-collective bandwidth interpolation (log-log spline over `sizes_mb`)
- [ ] η_comm[type] calibration (§S1)
- [ ] Bucket-size dependence
- [ ] Cross-node vs intra-node split (now: scope-driven; needs per-stage policy when both apply)
- [ ] Slow-node penalty: when `roll_up.slow_nodes_at_max_size` is non-empty, downgrade DP AR to `min_bw_gbs` instead of `median_bw_gbs`
