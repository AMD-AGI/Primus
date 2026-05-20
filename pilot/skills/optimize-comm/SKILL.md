---
name: optimize-comm
description: Strategies and env candidates for communication-bound Primus training. Use when bottleneck-diagnose returned COMM_BOUND, when comm_ratio is high, when AllReduce / AllGather / ReduceScatter is dominating step time, when DDP / FSDP / ZeRO comm is unhidden, when scaling efficiency degrades across nodes, or when the user asks about NCCL / RCCL tuning, bucket size, gradient overlap, IB / xGMI, NCCL_BUFFSIZE, NCCL_MIN_NCHANNELS, NCCL_IB_HCA, NCCL_NET_GDR_LEVEL, RCCL_MSCCL_ENABLE.
---

# Optimize for Communication Bottleneck

Reduce communication pressure (or hide it). Apply moves in priority order; cheap (env-only, no recompile) before expensive (parallelism reshape).

## Strategy Decision Table

| Tier | Move | When to try | Cost | Typical gain |
|---|---|---|---|---|
| 0 | bf16 grad/optim (precision-aware optimizer) | DP/FSDP, fp32 grad still in use | flag, see `primus-defaults` | 5–15% (grad reduce volume ÷ 2) |
| 0 | Enable comm/comp overlap | `overlap_ratio < 0.6` | flag, free | 5–25% |
| 0 | Tune NCCL/RCCL env | first round when COMM_BOUND, see `env-catalog` | env-sweep, ~0.3 GPU·h | 2–10% |
| 1 | Bucket sizing | DDP grad AR present, p95 msg size != bucket | flag, free | 3–10% |
| 1 | **Drop TP (AMD-specific, high leverage)** | TP > 1 on AMD MI300X/MI355X **and** memory has slack | reshape parallelism | 10–30% |
| 1 | Topology align (TP intra-node) | TP group spans nodes (NVIDIA mostly; AMD prefer TP=1) | reshape parallelism | 10–30% |
| 2 | Reduce DP / use FSDP/ZeRO | Plain DP grad too large | reshape, model-aware | 5–15% |
| 2 | Reduce world traffic (ep ↓ for MoE) | Cross-node alltoall dominant **and DeepEP not enabled** | reshape; capacity drop risk | mixed |

## Tier 0 — bf16 grad / precision-aware optimizer (DP/FSDP)

If `use_precision_aware_optimizer` is `false` (or `main_grads_dtype` is `fp32`), the gradient AllReduce volume is 2× what it could be. Enable per `primus-defaults`:

| Flag | Set to |
|---|---|
| `use_precision_aware_optimizer` | `true` |
| `main_grads_dtype` | `bf16` |
| `exp_avg_dtype` / `exp_avg_sq_dtype` | `bf16` |

This is normally already done in `round_1` (primus-defaults). It only shows up here when the user's YAML disabled it explicitly (some research setups insist on fp32 main_grads for stability). Check with the user before flipping.

## Tier 0 — Overlap (always try first)

| Symptom | Action | Override |
|---|---|---|
| `overlap_ratio < 0.6`, plain DDP | enable grad overlap | `--overlap_grad_reduce True` |
| ZeRO-3 / FSDP, `overlap_ratio < 0.6` | enable param-gather overlap | `--overlap_param_gather True` |
| TP > 1 with sequence-parallel, no async TP | enable async TP | `--tp_comm_overlap True` (Megatron) |
| MoE alltoall serial | enable dispatch overlap | model-side flag (`--moe_token_dispatcher_type flex` or backend equivalent) |

These are flag flips, no real cost — try them as the very first round when COMM_BOUND.

## Tier 0 — Env (the env-sweep round)

When `bottleneck-diagnose` produced `env_suspect`, do a single **env-sweep** round (lock structure, vary env only, ≤ 5 flags × ≤ 8 combos × ≤ 50 steps). See `env-catalog` for the flag dictionary.

Most-effective flags for COMM_BOUND on AMD MI300X / MI355X:

| Flag | Try | Effect |
|---|---|---|
| `NCCL_BUFFSIZE` | 4 MB / 8 MB / **16 MB** / 32 MB | bigger buffer → fewer pipeline stalls on large msgs |
| `NCCL_MIN_NCHANNELS` | 8 / **16** / 32 | more rings → more concurrent traffic |
| `NCCL_IB_HCA` | match preflight env_baseline exactly | wrong HCA list → traffic on wrong NIC |
| `NCCL_NET_GDR_LEVEL` | **4** / 5 | enable GPUDirect RDMA all the way |
| `NCCL_IB_GID_INDEX` | from `show_gid` (typically 3) | wrong GID → fall back to TCP |
| `RCCL_MSCCL_ENABLE` | **1** | algorithmic AllReduce wins on certain shapes |
| `NCCL_P2P_LEVEL` | NVL (intra) / SYS | constrain P2P scope |
| `NCCL_DEBUG` | only set to `WARN` while debugging | `INFO` floods logs |

Always start the sweep from the cluster `env_baseline` (preflight). The diff merged into the new champion is `env_diff = sweep_winner − baseline`.

## Tier 1 — Bucket sizing

Megatron / TorchTitan let you tune the gradient bucket size:

| Override | Value range | Notes |
|---|---|---|
| `--ddp_bucket_size <bytes>` | 25 MB → 128 MB | match the p95 grad message size from profile |

Decision rule: if profiler shows many small AR (`p50 < 10 MB`) **and** total `comm_ratio > 0.3`, raise bucket. If you see one giant tail AR causing pipeline stalls, lower bucket.

## Tier 1 — Drop TP (AMD-specific, high leverage)

On AMD MI300X / MI355X, intra-node TP is more expensive than on NVIDIA NVLink (TP exposes per-layer AllGather/ReduceScatter on the activation, which xGMI does not hide as well as NVLink). Default heuristic for AMD:

| Setup | Recommended `tp` |
|---|---|
| Dense ≤ 13B, fits at tp=1 with selective recompute | **tp=1** |
| Dense 30–70B, single-node memory tight | tp=2 → 4 → 8 in that order; smallest that fits |
| Dense > 70B, multi-node | use pp + dp before raising tp |
| MoE | tp=1 unless attention activation memory forces it |

Decision rule: if `tp > 1` on AMD and the predicted-mem at `tp/2` is still ≤ 0.92 × HBM (per `execution-model.OOM gate`), try dropping tp. Often the single biggest comm win on AMD.

Costs:
- M_param / M_grad / M_optim per rank doubles when tp halves → re-check OOM gate.
- Per-rank kernel size grows → may *also* improve compute (often a double win).

## Tier 1 — Topology alignment (when TP > 1 is unavoidable)

| Anti-pattern | Fix |
|---|---|
| `tp` group spans 2 nodes | reshape so `tp ≤ gpus_per_node` (e.g. 16 → 8) |
| `ep` group spans many nodes for moderate MoE **and DeepEP off** | shrink `ep` to fit one node, raise `dp` |
| `ep` group spans many nodes **with DeepEP on** | leave it — DeepEP makes cross-node EP viable; see `optimize-moe` |
| `dp` group entirely intra-node, `tp` cross-node | swap: `tp` intra, `dp` cross |

Use the preflight `intra_node` vs `inter_node` BW gap as a cost signal (typical: 800 GB/s xGMI vs 25 GB/s IB roll-up — 30× ratio).

## Tier 2 — Parallelism reshape (more invasive)

Apply only when Tiers 0–1 are exhausted around the current champion.

| Move | When | Risk |
|---|---|---|
| Switch DDP → ZeRO-1/2 | DP grad AR is the dominant comm | larger comm volume but better mem budget |
| Switch ZeRO-3 → ZeRO-1 | param-gather overlap insufficient | mem usage rises |
| Reduce `ep` (MoE) | alltoall dominates and capacity slack exists | router capacity drop, eval gap |
| Increase `tp` to absorb DP work | intra-node BW available, DP cross-node | per-rank kernels shrink → COMPUTE_BOUND risk |
| Drop a node (use fewer DP ranks) | weak-scaling regime, comm cost > comp gain | obvious throughput drop in absolute terms |

Each reshape consumes a structural round; only one structural axis change per candidate.

## env_diff template (shipped with each candidate)

```yaml
env_diff:
  NCCL_BUFFSIZE: 16777216
  NCCL_MIN_NCHANNELS: 16
  RCCL_MSCCL_ENABLE: 1
```

Always express as **diff vs baseline**, never the full env. See `env-catalog` for what each flag does.

## Constraints / safety

| Combo | Result | Rule |
|---|---|---|
| `NCCL_NET_GDR_LEVEL=5` + missing `nv_peer_mem` / equivalent | hang at startup | check preflight first |
| `RCCL_MSCCL_ENABLE=1` + custom collective override flags | undefined | don't combine |
| Very large `NCCL_BUFFSIZE` (≥ 128 MB) | per-rank mem overhead 1–4 GB | check `optimize-memory` headroom |
| `NCCL_DEBUG=INFO` long-running | log file blows up | use only for debugging, not production runs |

If a sweep candidate matches a known-bad combo, reject it (do not even submit).

## Worked example

```
Round k (champion P_c): COMM_BOUND, comm_ratio=0.38, overlap_ratio=0.45,
                        env_suspect: [NCCL_BUFFSIZE]
  Tier-0 overlap not yet enabled                     → P_c+overlap (1 candidate)
  Tier-0 env-sweep:                                  → 5 candidates in parallel
    E1: NCCL_BUFFSIZE=8M
    E2: NCCL_BUFFSIZE=16M
    E3: NCCL_BUFFSIZE=16M + NCCL_MIN_NCHANNELS=16
    E4: NCCL_BUFFSIZE=32M
    E5: RCCL_MSCCL_ENABLE=1
  Pick the winner across all 5 + the overlap flip; merge env diff into champion.
  Re-diagnose: if comm_ratio dropped < 0.25 we may have left COMM_BOUND.
```

## Important Notes

- **Never change structure and env in the same candidate**. If you do, you cannot tell which one helped.
- **Run env sweeps as parallel subagents**. The env-sweep round is the prime case for `run-and-profile` parallelism.
- **Cap each env sweep at ≤ 5 flags × ≤ 8 combinations × ≤ 50 steps**. Larger sweeps waste budget; the gain is rarely > 10%.
- **The env catalog is single-sourced in `env-catalog`**. This skill points to flags by name; full descriptions / defaults / known pitfalls live there.
- **Topology / parallelism reshape is the right move when scaling N nodes ×2 gives < 1.5× tps**. If single-node comm is fine and only N>1 is bad, the answer is almost always "reshape so DP/FSDP fits the topology", not "tune more env".
