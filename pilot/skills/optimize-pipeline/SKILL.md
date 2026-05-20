---
name: optimize-pipeline
description: Strategies for pipeline-bound Primus training. Use when bottleneck-diagnose returned PIPELINE_BOUND, when bubble_ratio is high (typically > 0.15), when too few microbatches per pipeline depth, when stages are imbalanced, when the user asks about pipeline parallelism, PP, VPP, virtual pipeline, interleaved 1F1B, microbatch count, GBS, MBS, stage balance, or pipeline schedule.
---

# Optimize for Pipeline Bottleneck

Bubble = the pipeline's idle slots. The fix is one of: more microbatches per stage, more virtual stages per device, or rebalancing stage compute.

```
T_bubble = (pp - 1) / (pp - 1 + M) × T_comp_per_microbatch
M        = gbs / (mbs × dp)
```

Three knobs collapse to: increase `M`, decrease `pp`, or split into `vpp` virtual stages.

## Strategy Decision Table

| Tier | Move | When | Cost | Typical bubble drop |
|---|---|---|---|---|
| 0 | Raise `vpp` | `pp ≥ 4`, `vpp = 1`, model has enough layers | flag | bubble × (1/vpp) |
| 0 | Raise `gbs` (more microbatches) | mem headroom for larger M, no convergence concern | flag | bubble ↓ proportionally |
| 1 | Raise `mbs` (with vpp) | activation memory has slack | flag | mixed (helps comp too) |
| 1 | Reduce `pp` | the model fits without it | reshape | bubble → 0 if pp=1 |
| 2 | Rebalance stages | one stage clearly slower than others | model-side | stage time variance ↓ |
| 2 | Switch schedule (1F1B → interleaved 1F1B) | backend supports it | flag | usually +5–10% |

## Tier 0 — VPP (the cheap win)

VPP (virtual pipeline parallelism, interleaved 1F1B) splits each device's layers into `vpp` chunks; the bubble shrinks roughly linearly in `vpp`.

| Override | Notes |
|---|---|
| `--virtual_pipeline_model_parallel_size <vpp>` | Megatron flag |
| `--num_layers_per_virtual_pipeline_stage <n>` | optional explicit chunk size |

Constraints:

- `total_layers / pp` must be divisible by `vpp` — pick `vpp ∈ {2, 4}` accordingly.
- `vpp` increases activation memory by ~×`vpp / 2`; check `optimize-memory` budget before raising.
- Diminishing returns past `vpp = 4`.

## Tier 0 — Raise M (microbatch count)

`M = gbs / (mbs × dp)`. Two ways to raise M without raising mem:

1. **Raise `gbs`**. Default in many configs is conservative; up to `gbs = 2 × current` is usually safe (verify convergence with the model owner if it's a research run).
2. **Lower `mbs`** (only if mbs > 1). Smaller microbatches → more of them → smaller bubble — but also smaller, less-efficient kernels (COMPUTE risk).

Rule of thumb: aim for `M ≥ 4 × pp`. Below `M = 2 × pp` the bubble is severe.

## Tier 1 — Raise mbs (with vpp)

Once `vpp` is on, `mbs` can often be raised without increasing bubble:

| Override | |
|---|---|
| `--micro_batch_size <mbs>` | |

Watch:

- Activation memory grows linearly in mbs. Check `optimize-memory.OOM gate` (`pred_mem ≤ 0.92 × hbm`).
- Larger mbs benefits compute (kernel utilization), so this is dual-purpose.

## Tier 1 — Drop pp

If the model fits memory at `pp = pp_current / 2` with reasonable recompute, drop `pp`. Bubble disappears entirely at `pp = 1`.

| Move | Tradeoff |
|---|---|
| `pp 4 → 2` | bubble halves, but tp/dp must absorb the missing parallelism (mem +) |
| `pp 2 → 1` | bubble = 0, but each device must hold all layers (mem ++) |

Always sanity-check via `execution-model.M_param + M_act` before trying.

## Tier 2 — Rebalance stages

Pipeline imbalance shows in profiler as some stages finishing earlier than others.

| Symptom | Fix |
|---|---|
| First stage slower (embedding-heavy) | shift one transformer layer off stage 0 |
| Last stage slower (LM head + loss) | shift one layer off the last stage; or `--standalone_embedding_stage` |
| Middle stage slower (more layers in interleaving) | uneven `--num_layers_per_virtual_pipeline_stage` allocation |

These are model-side knobs (`pipeline_layout`, `num_layers_in_first_pipeline_stage`, etc.); not always exposed. If not exposed, skip — pipeline rebalance is a Tier 2 move.

## Tier 2 — Switch schedule

Backends usually offer:

- 1F1B (default after warmup)
- Interleaved 1F1B (the VPP schedule, enabled by `--virtual_pipeline_model_parallel_size > 1`)
- ZB1P / Zero-Bubble PP (newer; check backend support)

If your backend supports a zero-bubble schedule and your model layout is regular, try it as a final Tier-2 move. Memory profile differs — re-check OOM gate.

## env_diff

Pipeline tuning is largely flag-driven; env diffs here are minor. The one common one:

| Flag | When | Effect |
|---|---|---|
| `NCCL_PROTO=Simple` | `pp` send/recv stalling on small msgs | sometimes lower latency for tiny p2p |

Most pipeline gains come from flags + parallelism reshape, not env.

## Interaction with other bottlenecks

| If after this round | Likely next bottleneck |
|---|---|
| `vpp` raised, bubble dropped, mem OK | likely COMPUTE_BOUND or MOE_DISPATCH_BOUND next |
| `gbs` raised, bubble dropped | watch `comm_ratio`: bigger gbs → bigger gradient AR |
| `pp` reduced to 1 | re-check mem; likely MEMORY_BOUND or COMPUTE_BOUND next |

## Worked example

```
Champion P_c: tp=2 pp=4 ep=8 mbs=1 vpp=1 gbs=512, dp=4
  → M = 512 / (1 × 4) = 128  (decent M but bubble still 0.18 because vpp=1)
  bubble_ratio = 0.18, mem_peak = 158 GB

Tier-0 candidates:
  P1: vpp 1 → 2 (model has 32 layers, 4 stages, 4 layers/stage; vpp=2 ⇒ 2 chunks of 2 layers each)
       → expected bubble 0.18 / 2 = 0.09; mem +30 GB to ~188 GB (within 192)
  P2: gbs 512 → 1024 (M doubles)
       → expected bubble 0.18 × (M / (M+pp-1)) shift; modest
  P3: pp 4 → 2 (mem must absorb)
       → bubble halves, but predicted mem 240 GB > 192 → REJECT (OOM gate)

Run P1 and P2 in parallel:
  P1 actual: bubble 0.09, tps +11%, mem 182 GB (close to limit)  ← champion
  P2 actual: bubble 0.13, tps +4%
Promote P1.  Re-diagnose: now COMPUTE_BOUND (gpu_util 0.78).
```

## Important Notes

- **VPP is almost always the right first move** when `pp ≥ 4`. It is a flag flip with predictable, large gain.
- **Always re-check memory after raising vpp / mbs / gbs**. Bubble fixes routinely push memory close to the OOM gate.
- **Don't stack vpp + larger gbs in one candidate**. Split into two rounds; otherwise the gain attribution is muddy.
- **`pp = 1` is a valid answer**. If memory allows, it eliminates the bubble entirely. Don't keep `pp > 1` for tradition.
- **Stage rebalance often invisible without a custom layout flag**. If your backend doesn't expose it, accept the imbalance and move on — the gain is usually < 10%.
- **Interleaved schedules trade memory for bubble**. Track the OOM gate carefully when switching schedule.
