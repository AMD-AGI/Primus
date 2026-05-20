---
name: optimize-memory
description: Strategies and env candidates for memory-bound Primus training. Use when bottleneck-diagnose returned MEMORY_BOUND, when mem_peak is close to HBM capacity, when the run OOM'd, when fragmentation is suspected (mem_reserved/mem_alloc > 1.4), when the user asks about activation recompute, ZeRO, FSDP, CPU/NVMe offload, PYTORCH_HIP_ALLOC_CONF, expandable_segments, max_split_size_mb, OOM, out-of-memory, HBM capacity.
---

# Optimize for Memory Bottleneck

Free up HBM, in this priority: cheap fragmentation fixes → activation recompute → parallelism shard → offload (last resort). Memory-bound is special: it gates which configurations are runnable at all, so it's addressed before any other bottleneck in MIXED cases.

## Memory budget reminder

```
Mem = M_param + M_grad + M_optim + M_act + M_buffer
OOM gate: predicted_Mem_peak ≤ 0.92 × hbm_capacity_gb
```

See `execution-model.Memory Estimation` for term definitions.

## Strategy Decision Table

| Tier | Move | When | Cost | Typical mem saving |
|---|---|---|---|---|
| 0 | Fragmentation env (alloc) | `mem_reserved/mem_alloc > 1.4` | env-sweep | 5–15% |
| 0 | Disable opportunistic caches | rare; `--cache_*` flags off | flag | 2–5% |
| 1 | Recompute selective | currently `none` | flag | 30–50% on M_act |
| 1 | Recompute full (or block-N) | currently `selective` and still tight | flag | 60–80% on M_act |
| 2 | Raise `tp` | tp can fit intra-node | reshape | M_param/M_grad ÷ tp_ratio |
| 2 | Raise `pp` | activations dominate | reshape | M_act ÷ pp |
| 2 | ZeRO-1 / ZeRO-2 / FSDP | DP > 1, Adam optim heavy | reshape | M_optim / dp |
| 3 | ZeRO-3 / FSDP full shard | extreme cases | reshape | M_param + M_grad + M_optim ÷ dp |
| 3 | CPU offload | absolute last resort | env+flag | huge but slow (PCIe) |

## Tier 0 — Fragmentation env (always try first when not OOM)

| Flag | Try | Effect |
|---|---|---|
| `PYTORCH_HIP_ALLOC_CONF` | `expandable_segments:True` | merges adjacent free blocks, big fragmentation win |
| `PYTORCH_HIP_ALLOC_CONF` | `expandable_segments:True,max_split_size_mb:512` | also caps split — helps with mixed-size workloads |
| `PYTORCH_HIP_ALLOC_CONF` | `garbage_collection_threshold:0.8` | force gc earlier |

For NVIDIA: substitute `PYTORCH_CUDA_ALLOC_CONF`.

These flags often free 5–15% of effective HBM with zero training-side change. Always sweep them in the first MEMORY_BOUND env-sweep round.

## Tier 1 — Activation recompute

| Override | Notes |
|---|---|
| `--recompute_granularity selective` | recompute attention only — best speed/mem tradeoff |
| `--recompute_granularity full --recompute_method block --recompute_num_layers <N>` | recompute every layer (or every N layers) |
| `--recompute_granularity full --recompute_method uniform` | spread evenly across the stage |

Heuristics:

| State | Try |
|---|---|
| `recompute=none`, OOM at small mbs | `selective` first |
| `selective` insufficient | `full` with `block, num_layers = layers_per_stage / 2` |
| Mem still tight after full | `block, num_layers = layers_per_stage` (= every layer) |

Cost: each recompute level costs ~10–25% throughput. Always re-check `T_step` after enabling; the candidate's `expected_gain` should account for it.

## Tier 2 — Parallelism reshape for memory

| Move | Memory effect | Side effect |
|---|---|---|
| `tp` ↑ within a node (≤ gpus_per_node) | `M_param`, `M_grad`, `M_optim` per rank ÷ tp_ratio | TP comm ↑ (intra-node, usually OK) |
| `pp` ↑ | `M_act` per rank ÷ pp; `M_param` partly | Bubble risk (PIPELINE_BOUND), comm change |
| Switch to ZeRO-1 | `M_optim` ÷ dp | small AR overhead |
| Switch to ZeRO-2 | + grad ÷ dp | more comm |
| FSDP / ZeRO-3 | + param ÷ dp | param-gather every layer (significant comm) |
| `mbs` ↓ | `M_act` linear ↓ | M ↑ (good for bubble) but kernels smaller (compute risk) |

Rule of thumb on AMD MI300X (192 GB HBM): you usually have enough memory to run dense ≤ 13B with TP=8 + ZeRO-1 + selective recompute single-node, and dense ≤ 70B with PP=4 + TP=8 + selective + multi-node.

## Tier 3 — Offload (only when nothing else works)

| Flag | Notes |
|---|---|
| `--cpu_offload` (model-side) | optimizer state to CPU; PCIe-bound |
| `--moe_ep_offload` | expert offload to CPU/NVMe (MoE-only) |

Throughput drop is severe (often 30–60%). Use only when the alternative is "cannot run at all".

## env_diff template

```yaml
env_diff:
  PYTORCH_HIP_ALLOC_CONF: "expandable_segments:True,max_split_size_mb:512"
```

(For NVIDIA, swap to `PYTORCH_CUDA_ALLOC_CONF`.)

See `env-catalog` for the full alloc-flag dictionary.

## OOM recovery protocol (special case of MEMORY_BOUND)

When the latest run came back `status=oom`:

1. Mark this plan **dead** (`derived_axis` recorded; never re-tried as-is).
2. Pick the smallest single change that reduces mem:
   - if `mbs > 1` → try `mbs - 1` first (single change)
   - else if `recompute=none` → set `selective`
   - else if `recompute=selective` → set `full` (block N)
   - else → reshape parallelism (tp ↑ or pp ↑)
3. If multiple consecutive OOMs at minimal config, escalate — model genuinely doesn't fit at this cluster shape.

## Constraints

| Combo | Issue | Rule |
|---|---|---|
| `expandable_segments:True` + ROCm version < 6.0 | bug / crash | check preflight ROCm version |
| ZeRO-3 / FSDP + small `dp` | gather overhead > param save | `dp ≥ 4` recommended |
| Recompute=full + already tight on compute | 25% TPS loss | weigh against PIPELINE_BOUND if mem isn't the real issue |

## Worked example

```
Champion P_c: tp=2 pp=2 mbs=4 recompute=selective; mem_peak = 178 GB / 192 GB (= 0.93)
  → bottleneck-diagnose: MEMORY_BOUND (Rule 3)
  → predicted next candidate at mbs=8: M_act doubles → 178 + ~30 → 208 GB → OOM gate trips → reject

Tier-0 candidates:
  E1: PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
  E2: + max_split_size_mb:512

Run E1+E2 in parallel:
  E1 mem_peak = 165 GB, tps unchanged → win
  E2 mem_peak = 161 GB, tps unchanged → slightly better, pick E2
  Champion env_diff += {PYTORCH_HIP_ALLOC_CONF:"expandable_segments:True,max_split_size_mb:512"}

Re-diagnose: mem 161/192 = 0.84 → no longer MEMORY_BOUND. Move on.
```

## Important Notes

- **Always env-sweep alloc flags before doing structural memory moves**. The fragmentation win is huge and free.
- **OOM is a normal subagent return**, not an exception. Mark dead, derive a smaller candidate, move on.
- **Recompute is the cheapest *throughput-costing* move**. Don't reach for it when env-only fixes will do.
- **ZeRO-3 / FSDP is heavy comm**. Only use when grad+param sharding is genuinely necessary; revisit with `optimize-comm` after.
- **Don't stack mem moves with any non-mem move**. If a candidate adds vpp + recompute=full simultaneously, the throughput gain attribution is impossible.
- **CPU/NVMe offload is the nuclear option**. If you reach it and the user has flexibility, it's usually better to add a node and avoid offload entirely.
