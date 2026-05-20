---
name: bottleneck-diagnose
description: Classify the bottleneck of a Primus training run from its run-and-profile snapshot. Use whenever a candidate has just finished and the agent needs to decide which optimize-* skill to load next, or to verify the subagent's coarse "bottleneck hint". Outputs exactly one of COMM_BOUND / PIPELINE_BOUND / MEMORY_BOUND / COMPUTE_BOUND / MOE_DISPATCH_BOUND / MIXED, plus optional env_suspect flags. Triggers: bottleneck analysis, why is it slow, what's limiting throughput, comm/comp ratio, bubble ratio, allreduce/alltoall is high, OOM, low gpu_util, low MFU.
---

# Bottleneck Diagnose — Decision Table

Convert a `snapshot.yaml` (or the `metrics:` block from a `run-and-profile` summary) into one bottleneck class plus optional `env_suspect` hints. The output drives which `optimize-*` skill the main agent loads next.

This skill is a **decision table**, not a model. Run the rules in order; first match wins. If multiple thresholds trip, classify as `MIXED` and address the largest contributor first.

## Inputs you need

From the latest run's snapshot:

```
metrics:
  tps, step_time_ms
  comm_ratio        # comm_kernel_time / step_time
  bubble_ratio      # pipeline_idle_time / step_time
  overlap_ratio     # overlapped_comm_time / total_comm_time
  mem_peak_gb       # vs cluster.hbm_capacity_gb
  gpu_util_avg      # mean SM utilization
status                # completed | oom | hang | ...
plan.parallelism      # tp, pp, ep, vpp, dp
```

If `comm_ratio` / `bubble_ratio` / `overlap_ratio` are `n/a` (no profiler), fall back to the heuristics in §"No-profiler fallback" below.

## Decision Rules (first match wins)

### Rule 0 — Hard failures override everything

| Condition | Class | Notes |
|---|---|---|
| `status = oom` | `MEMORY_BOUND` | also set `env_suspect: PYTORCH_HIP_ALLOC_CONF` if mem_peak < hbm × 0.95 |
| `status = hang` | not a bottleneck class — re-run `preflight` env_probe; do not enter Diagnose |
| `status = numerical` | escalate to user; not a bottleneck |
| `status = failed` | inspect log; usually config error → mark plan dead |

### Rule 1 — COMM_BOUND

```
comm_ratio > 0.25  AND  overlap_ratio < 0.6
```

Strong indicator: a large fraction of step time is unhidden communication.

`env_suspect` candidates (raise to next round if matched):

| Symptom | Suspect flag | Hint |
|---|---|---|
| inter-node `comm_ratio > 0.25`, msg p95 > 4 MB | `NCCL_BUFFSIZE` | too small buffer — bump to 16 MB |
| inter-node `comm_ratio > 0.25`, IB BW measured < 70% of preflight `world.AR.median_bw_gbs[256MB]` | `NCCL_IB_HCA`, `NCCL_NET_GDR_LEVEL` | misrouted HCA / GDR not engaged |
| MoE `alltoall` time > 15% of step | `NCCL_MIN_NCHANNELS`, `RCCL_MSCCL_ENABLE` | bump channels / enable MSCCL |
| `comm_ratio` huge but `overlap_ratio < 0.2` | n/a (model flag) | enable `--overlap_grad_reduce True` first |

→ load `optimize-comm`.

### Rule 2 — PIPELINE_BOUND

```
plan.pp > 1  AND  bubble_ratio > 0.15
```

Or, when bubble metric is unavailable, `M = gbs / (mbs × dp) < 4 × pp` (too few microbatches).

→ load `optimize-pipeline`.

### Rule 3 — MEMORY_BOUND

```
mem_peak_gb > 0.85 × cluster.hbm_capacity_gb
```

Or any prior candidate at higher mbs / lower recompute hit `status=oom`.

`env_suspect`:

| Symptom | Suspect flag | Hint |
|---|---|---|
| `mem_reserved / mem_alloc > 1.4` (fragmentation) | `PYTORCH_HIP_ALLOC_CONF` | enable `expandable_segments:True` + `max_split_size_mb:512` |

→ load `optimize-memory`.

### Rule 4 — MOE_DISPATCH_BOUND (MoE only)

```
plan.ep > 1  AND  alltoall_time / step_time > 0.15
```

Distinct from generic COMM_BOUND because the remediation is dispatch-overlap / capacity factor / load balance, not bucket / channel tuning.

→ load `optimize-moe`.

### Rule 5 — COMPUTE_BOUND

```
comm_ratio < 0.20  AND  bubble_ratio < 0.10  AND  mem_peak_gb < 0.70 × hbm_capacity_gb
AND  gpu_util_avg < 0.80
```

There is headroom in every other axis but the kernels themselves are not running fast enough.

→ load `optimize-compute`.

### Rule 6 — MIXED

If two or more rules trip with comparable severity (none is dominant by ≥ 1.5×), classify as `MIXED`.

For `MIXED`, **address the largest contributor first** in this priority order: `MEMORY_BOUND > COMM_BOUND > MOE_DISPATCH_BOUND > PIPELINE_BOUND > COMPUTE_BOUND`. Memory first because it gates which configurations are even runnable; compute last because it's the least leveraged knob.

## No-profiler fallback (heuristic from logs only)

When `comm_ratio` / `bubble_ratio` are missing, infer from coarse signals:

| Signal | Tilt classification toward |
|---|---|
| Lowering `mbs` raised TPS | MEMORY_BOUND or COMPUTE_BOUND (bigger kernels under-utilized) |
| Lowering `pp` raised TPS at constant gbs | PIPELINE_BOUND |
| Raising `tp` reduced step time | COMPUTE_BOUND or MEMORY_BOUND |
| Raising `dp` did not improve TPS scale | COMM_BOUND |
| TPS regressed when scaling N nodes ×2 | COMM_BOUND |
| OOM when raising mbs | MEMORY_BOUND |

Always re-run the candidate **with** profiler (`profile=true`) at the next opportunity to confirm.

## env_suspect handling protocol

When `env_suspect` is non-empty:

1. **Before** the next structural change, do a single **env-sweep round** that locks the structure of the current champion and varies only the suspect flags (≤ 5 flags, ≤ 8 combinations, see `optimize-comm` / `optimize-memory` / `env-catalog`).
2. Merge the winning env diff into the champion's plan.
3. Re-diagnose. Often the bottleneck class shifts after env is fixed.

If no `env_suspect` was raised, skip the env-sweep round and go straight to the matching `optimize-*` skill.

## Worked examples

```
snapshot:                                              decision:
  comm_ratio=0.38, bubble=0.12, mem=140GB, util=0.62
  pp=4, ep=8, alltoall_time/step=0.28                  → COMM_BOUND (Rule 1)
                                                         + env_suspect: NCCL_BUFFSIZE
                                                       → optimize-comm

  comm_ratio=0.18, bubble=0.18, mem=158GB, util=0.71
  pp=4, M=8                                            → PIPELINE_BOUND (Rule 2)
                                                       → optimize-pipeline

  comm_ratio=0.10, bubble=0.05, mem=178GB, util=0.74
  hbm=192GB                                            → MEMORY_BOUND (Rule 3)
                                                         (mem 178/192 = 0.93 > 0.85)
                                                       → optimize-memory

  comm_ratio=0.08, bubble=0.04, mem=120GB, util=0.55
  pp=1                                                 → COMPUTE_BOUND (Rule 5)
                                                       → optimize-compute

  status=oom                                           → MEMORY_BOUND (Rule 0)
                                                         + dead-mark this plan
                                                       → optimize-memory (try recompute=full / smaller mbs)
```

## Important Notes

- **One class per round**. Even when MIXED, do not load multiple `optimize-*` skills in one round; pick the priority winner and address it. Mixing optimization moves makes attribution impossible.
- **`env_suspect` is cheap, structure changes are expensive**. Always try the env-sweep round before changing parallelism / mbs / vpp when `env_suspect` is present.
- **Re-diagnose after every champion change**, not just every round — moving to a new champion can shift the bottleneck.
- **If COMM_BOUND but `overlap_ratio` is already > 0.7**, do not waste a round chasing more overlap; jump to bucket / topology moves directly.
- **Don't classify on a single noisy run**. If two consecutive rounds disagree on the class, give the snapshot one more measurement (re-run baseline plan) before committing.
- **Thresholds are defaults**, not absolutes. If the user has a clear goal (e.g. "this MoE training MUST hit 15K TPS"), tighten thresholds (e.g. `comm_ratio > 0.20` triggers COMM_BOUND).
