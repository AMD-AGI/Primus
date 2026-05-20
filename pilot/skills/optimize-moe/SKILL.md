---
name: optimize-moe
description: MoE-specific tuning strategies for Primus training (DeepSeek V2/V3, Mixtral, Qwen MoE, etc.). Use when bottleneck-diagnose returned MOE_DISPATCH_BOUND, when alltoall time dominates, when expert load imbalance is suspected, when capacity factor / token drop is in question, when the user asks about MoE routing, dispatch overlap, expert parallelism (EP), top-k routing, capacity factor, token drop, load balance loss, or grouped GEMM.
---

# Optimize for MoE-Specific Bottlenecks

MoE has its own failure modes on top of the dense bottlenecks: alltoall dispatch dominates step time, experts are unbalanced (some overload, others idle), or capacity factor causes token drop. This skill complements `optimize-comm` (general comm) with MoE-only moves.

## Strategy Decision Table

| Tier | Move | When | Cost | Typical gain |
|---|---|---|---|---|
| 0 | Enable Primus Turbo MoE flags (Turbo + DeepEP + grouped MLP + sync-free MoE) | not yet on | batched flag flip via `primus-defaults` | 20–60% (round_1) |
| 0 | Tune alltoall env | NCCL_MIN_NCHANNELS / RCCL_MSCCL_ENABLE | env-sweep | 3–10% |
| 1 | EP placement (DeepEP-aware: shrink `ep` when DeepEP off, tune `turbo_deepep_num_cu` when on) | comm dominant | reshape or flag | 5–25% |
| 1 | Adjust capacity factor | router drops tokens or wastes capacity | flag | 2–8% |
| 1 | Group GEMM impl (`use_turbo_grouped_mlp`, `moe_use_legacy_grouped_gemm`) | available but not enabled | flag | 5–15% |
| 2 | Increase load-balance loss weight | imbalance > 1.5× | flag | training-side |
| 2 | Topk reduction | `topk=2` → `topk=1` | model-side | training-side |
| 3 | Expert parallelism shape (EP × DP_for_experts) | extreme cases | reshape | varies |

## MoE diagnostics (what to look for)

In addition to the standard snapshot:

| Symptom | Cause |
|---|---|
| `alltoall` time / step > 0.15 | dispatch dominant — Tier 0 |
| Expert per-rank token count variance > 30% | load imbalance — Tier 2 |
| Router drops > 5% of tokens | capacity too low — adjust capacity_factor |
| All experts > 95% utilized | capacity too small / ep too small — capacity ↑ or ep ↑ |
| `gpu_util` low and `alltoall` low | not MoE-bound; reclassify with `bottleneck-diagnose` |

The router stats are usually printed by Megatron / TorchTitan in the train log; if not, enable verbose router logging.

## Tier 0 — Primus Turbo MoE batch (covered by `primus-defaults` round_1)

Most large MoE wins on Primus / AMD come from a coordinated set of Turbo flags. These are turned on as one batch in `round_1` (see `primus-defaults`); this section is the per-bottleneck reference if any of them are missing.

| Flag | Effect |
|---|---|
| `enable_primus_turbo: true` | master switch |
| `use_turbo_attention: true` | TE / Turbo attention path |
| `use_turbo_grouped_mlp: true` | grouped MLP kernel (replaces per-expert FFN loop) |
| `use_turbo_fused_act_with_probs: true` | fused activation × routing prob |
| `moe_use_fused_router_with_aux_score: true` | fused router + aux-loss |
| `turbo_sync_free_moe_stage: 2` | sync-free MoE schedule (stage 2 recommended) |
| `use_turbo_deepep: true` | DeepEP cross-node alltoall (see Tier 1 Case B) |
| `moe_use_legacy_grouped_gemm: true` | current AMD MoE path (will flip when new GG stabilizes) |

If the user's YAML disabled any of these, that's the first round 1 candidate before any structural move. Backend-specific dispatcher overrides like `--moe_token_dispatcher_type flex` apply when Turbo is **off**; with Turbo on, the dispatcher choice is bundled into `turbo_sync_free_moe_stage`.

If the model + backend genuinely don't support these flags (init fails), dispatch is unhideable; move to Tier 1.

## Tier 0 — Alltoall env-sweep

Same env-catalog as `optimize-comm`, but the high-leverage flags for alltoall on AMD:

| Flag | Try | Effect |
|---|---|---|
| `NCCL_MIN_NCHANNELS` | `16` / `32` | more rings for parallel alltoall |
| `RCCL_MSCCL_ENABLE` | `1` | algorithmic alltoall on certain shapes |
| `NCCL_BUFFSIZE` | `16M` / `32M` | bigger buffers help large dispatch packets |
| `NCCL_ALGO` | leave default unless preflight indicates otherwise | overriding rarely wins |

Lock structure, sweep ≤ 5 flags × ≤ 8 combos.

## Tier 1 — EP placement (DeepEP-aware)

The right move depends on whether **DeepEP** is enabled (`use_turbo_deepep: true` from `primus-defaults`):

### Case A — DeepEP off

Cross-node alltoall is the slow path. Dropping `ep` to fit intra-node is usually the single biggest gain.

| Move | Effect | Cost |
|---|---|---|
| `ep` cross-node → `ep` intra-node only | alltoall stays inside xGMI (≈ 30× faster than IB) | per-expert capacity ↓ → may need capacity_factor ↑ |
| `ep` ÷ 2 | per-rank expert count ×2 (mem ↑); cross-node traffic ↓ | mem check needed |

Rule of thumb: if `ep > gpus_per_node` and DeepEP is off, drop `ep ≤ gpus_per_node` first.

### Case B — DeepEP on (recommended for AMD MoE > 8 nodes)

DeepEP provides an optimized cross-node alltoall implementation, so cross-node EP is no longer punitive. **Do not rush to shrink `ep`** — instead tune DeepEP itself:

| Knob | Set to | Notes |
|---|---|---|
| `use_turbo_deepep` | `true` | turn on (covered by `primus-defaults`) |
| `turbo_deepep_num_cu` | `80` for `ep ≤ 8`, `64` fallback, `32` for `ep` 16–64 | from MI355X shipped configs; sweep these three values when DeepEP is in play |
| `turbo_deepep_use_comm_stream` | `false` (default) | flip to `true` only if profiler shows dispatch can use a separate stream |
| `moe_router_dtype` | `fp32` | required when DeepEP is on |

When DeepEP is on, the `ep` choice is a memory / load-balance question, not a comm question — defer to per-expert token count and HBM headroom.

## Tier 1 — Capacity factor

```
capacity_per_expert = capacity_factor × (tokens_per_rank × topk / num_experts_per_rank)
```

| Override | Notes |
|---|---|
| `--moe_aux_loss_coeff <c>` | load balance loss weight |
| `--moe_router_load_balancing_type aux_loss` | enable aux loss |
| `--moe_capacity_factor <f>` (when exposed) | usually 1.0 – 1.25 |
| `--moe_token_drop_policy probs` / `position` | how to drop on overflow |

Heuristics:

- token drop > 5% → raise `capacity_factor` by 0.1 – 0.2
- expert utilization < 70% across all → lower `capacity_factor` by 0.1
- variance > 30% → bump aux loss coeff (model-side, talk to model owner)

## Tier 1 — Grouped GEMM

Most modern MoE backends ship a grouped-GEMM kernel that fuses expert FFNs:

| Override | |
|---|---|
| `--moe_grouped_gemm True` | usually a flag flip; substantial gain when tokens-per-expert ≥ 64 |

If tokens-per-expert is small (very wide MoE), grouped GEMM has less benefit; benchmark.

## Tier 2 — Routing changes

Topk reduction (e.g. `topk=2` → `topk=1`) cuts dispatch traffic ~2× but is a model-quality decision; only try with the model owner's blessing.

## Constraints

| Combo | Issue |
|---|---|
| `ep × tp × pp × dp` ≠ `world_size` | invalid shape — reject |
| `ep > num_experts` | invalid — reject |
| `ep` and `tp` overlapping group ranks | most backends require disjoint; verify |
| MoE + recompute=full + grouped_gemm | sometimes incompatible; check backend |
| `capacity_factor < 1.0` | guaranteed token drop — only with model owner approval |

## Worked example

```
Champion P_c: tp=2 pp=4 ep=8 mbs=1, dispatch=serial; alltoall_time/step = 0.30
  → bottleneck-diagnose: MOE_DISPATCH_BOUND

Tier-0:
  P1: enable --moe_token_dispatcher_type flex (dispatch overlap)
  P2: + --moe_grouped_gemm True
  Run both as parallel subagents.
  P1 alltoall ratio 0.30 → 0.18, tps +14%
  P2 alltoall ratio 0.18 → 0.15, tps +6% on top of P1
  Champion: P_c + flex + grouped_gemm (call it P_c').

Re-diagnose on P_c': comm_ratio still 0.22 (just under threshold), gpu_util 0.65.
Tier-0 env-sweep on alltoall:
  E1: NCCL_MIN_NCHANNELS=16 + RCCL_MSCCL_ENABLE=1
  E2: + NCCL_BUFFSIZE=16M
  Winner: E2, tps +3%
  Champion env_diff merges.

Re-diagnose: now COMPUTE_BOUND. Hand off to optimize-compute.
```

## Important Notes

- **Dispatch overlap is the single biggest MoE win**. Always check it first.
- **`ep > gpus_per_node` is often a mistake**. Cross-node alltoall is ~30× more expensive than intra-node; shrink `ep` first.
- **Capacity factor changes affect convergence**, not just throughput. Always confirm with the model owner before tuning > ±0.2.
- **Grouped GEMM benefit scales with tokens-per-expert**. Tiny experts gain little; verify with profile.
- **MoE bottlenecks rotate fast**: after fixing dispatch, the next bottleneck is often COMPUTE (now that the GPU has work) or PIPELINE. Always re-diagnose.
- **Router stats are gold**. Always look at per-expert token counts; load imbalance is invisible from raw `comm_ratio` alone.
