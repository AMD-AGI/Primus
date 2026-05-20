---
name: primus-defaults
description: Ordered catalog of Primus Megatron / TorchTitan / Primus Turbo "should-be-on" optimization features — fusion (cross-entropy / RoPE / fused router), Primus Turbo (attention / grouped MLP / sync-free MoE / parallel linear), DeepEP, and bf16 precision-aware optimizer. Defines which flags form one "feature" (a coupled bundle that must be enabled together) and the rollout order. Used as Phase A of the tuning LOOP after BASELINE, one feature per round so each contribution is attributable. Triggers: enable_primus_turbo, use_turbo_*, use_turbo_deepep, turbo_deepep_num_cu, turbo_sync_free_moe_stage, cross_entropy_loss_fusion, gradient_accumulation_fusion, apply_rope_fusion, moe_use_fused_router_with_aux_score, use_precision_aware_optimizer, main_grads_dtype, exp_avg_dtype, bf16 grad reduce, Primus Turbo enable, what flags should I always have on.
---

# Primus Defaults — Feature Rollout Catalog

Primus ships many high-leverage flags **off by default**. This skill lists them, groups them into "features" (the unit of one tuning round), and gives the recommended rollout order. Phase A of `tuning-loop` walks the order one feature at a time so every gain is attributable to one decision.

This skill is a **feature dictionary + ordered checklist**, not a strategy file. The bottleneck-driven decision (which `optimize-*` skill to load next) lives in `bottleneck-diagnose`.

## What "one feature" means

The hard rule from `tuning-loop`: every candidate changes exactly one variable. The only allowance is a **coupled bundle**: a set of flags that must be set together to express a single logical capability. A bundle counts as one feature (= one candidate = one round). Examples:

| Bundle | Why it's one feature |
|---|---|
| `cross_entropy_fusion_impl=te` + `cross_entropy_loss_fusion=true` | the impl flag selects which kernel; the boolean enables it. Setting only one is a no-op or misconfig |
| `enable_primus_turbo=true` + `use_turbo_attention=true` | master switch + sub-flag; sub-flag does nothing without master |
| `use_turbo_grouped_mlp=true` + `use_turbo_fused_act_with_probs=true` | fused activation requires grouped MLP path |
| `use_turbo_deepep=true` + `moe_router_dtype=fp32` + `turbo_deepep_num_cu=N` | DeepEP requires fp32 router and the CU count to operate |
| `use_precision_aware_optimizer=true` + `main_grads_dtype=bf16` + `exp_avg_dtype=bf16` + `exp_avg_sq_dtype=bf16` | the master switch must coexist with all three dtype choices |
| `apply_rope_fusion=true` + `enable_experimental=true` | experimental gate is required for the fused RoPE path in current Megatron versions |

If you can run a flag standalone and it does something on its own → it is its own feature → its own round. When in doubt, split.

## When to apply

- **Phase A of every fresh tuning session** (right after BASELINE, before bottleneck-driven tuning).
- **When BASELINE used a third-party YAML** that was not tuned for Primus.
- **After a model swap** within a session — different model classes need different feature subsets.

Read the user's `exp_yaml` once first to see what's already on; skip features the YAML already enables.

## Always-on flags (every Primus run)

| Flag | Default in Primus | Set to | Why |
|---|---|---|---|
| `cross_entropy_fusion_impl` | unset | `"te"` | use TE fused CE loss path |
| `cross_entropy_loss_fusion` | `false` | `true` | fused loss kernel |
| `apply_rope_fusion` | `false` | `true` | fused RoPE |
| `enable_experimental` | `false` | `true` (only when needed by another flag, e.g. RoPE fusion) | gate for experimental kernels |
| `overlap_grad_reduce` | `false` | `true` | hide DP grad AR (model-side) |
| `overlap_param_gather` | `false` | `true` (only when ZeRO/FSDP is on) | hide param gather |

These never hurt. If the YAML doesn't have them, override at submit time.

## bf16 precision-aware optimizer (multi-node DP/FSDP — high leverage)

Reduces grad-reduce volume ~2× and optimizer state memory significantly.

| Flag | Default | Set to | Notes |
|---|---|---|---|
| `use_precision_aware_optimizer` | `false` | `true` | master switch |
| `main_grads_dtype` | `fp32` | `bf16` | grad reduce in bf16 |
| `exp_avg_dtype` | `fp32` | `bf16` | Adam first moment in bf16 |
| `exp_avg_sq_dtype` | `fp32` | `bf16` | Adam second moment in bf16 |

Risks:
- Long training stability — for research-grade convergence runs, validate on a short window vs fp32 main_grads first.
- For very small models (< 1B) the grad-reduce volume is small enough that the bandwidth win is marginal; keep fp32 and skip.

## Primus Turbo (master switch + sub-flags)

Primus Turbo is a set of patched kernels under `primus.backends.megatron.patches.turbo` (and TorchTitan equivalents). Default config: `primus/configs/modules/megatron/primus_turbo.yaml` — every sub-flag is `false`, you must opt in.

| Flag | Set to | When | Notes |
|---|---|---|---|
| `enable_primus_turbo` | `true` | always (master switch) | required before any sub-flag matters |
| `use_turbo_attention` | `true` | dense + MoE attention | substantial win on attention-bound models |
| `enable_turbo_attention_float8` | `true` | only with FP8 path | leave off for BF16 |
| `use_turbo_parallel_linear` | `true` | TP > 1 | parallel-linear kernel |
| `use_turbo_grouped_mlp` | `true` | MoE | grouped MLP kernel |
| `use_turbo_fused_act_with_probs` | `true` | MoE + grouped MLP | fused activation × routing prob |
| `use_turbo_rms_norm` | **`false` (known bug)** | n/a | leave off until upstream fix; reference: comment `# bug` in MI355X configs |
| `moe_use_fused_router_with_aux_score` | `true` | MoE | fused router + aux loss |
| `turbo_sync_free_moe_stage` | `2` | MoE | sync-free MoE; stage 2 recommended in shipped configs |

Caveat: not every model is wired up to every Turbo flag. If turning one on causes `failed` (init error), strip it back and report which flag the model doesn't support.

## DeepEP (MoE only)

DeepEP makes cross-node EP viable. Without it, EP must stay intra-node (see `optimize-moe`).

| Flag | Set to | When |
|---|---|---|
| `use_turbo_deepep` | `true` | MoE, EP > 1 |
| `turbo_deepep_num_cu` | `80` (ep ≤ 8) / `64` (ep ≤ 8 fallback) / `32` (ep 16–64) | per cluster best-practice; from MI355X shipped configs |
| `turbo_deepep_use_comm_stream` | `false` | default; flip to `true` only after profiling indicates dispatch can use a separate stream |
| `moe_router_dtype` | `fp32` | when DeepEP on; router precision |
| `moe_shared_expert_overlap` | `false` | shipped default; some shapes benefit from `true` — sweep separately under `optimize-moe` |

If `use_turbo_deepep: true` is set, the `optimize-moe` "Reduce ep" Tier-1 move has its rationale flipped: **cross-node EP is now cheaper than usual**, so don't rush to shrink ep.

## Cross-entropy / gradient accumulation fusion (model-class-conditional)

| Flag | Default | Set to | Caveat |
|---|---|---|---|
| `cross_entropy_loss_fusion` | `false` | `true` | always good |
| `gradient_accumulation_fusion` | `false` | `true` for MoE / large models; **`false`** for small dense models (some YAMLs ship it off explicitly, e.g. `qwen3_8B-BF16`) | conflicts with certain dense small-model paths |
| `moe_use_legacy_grouped_gemm` | `false` | `true` for current MoE path on AMD; will flip to `false` once new grouped GEMM stabilizes | track upstream |

## FP8-specific (only when `--fp8 hybrid` or backend FP8 mode is used)

| Flag | Set to | Notes |
|---|---|---|
| `enable_turbo_attention_float8` | `true` | FP8 attention path |
| `fp8_amax_history_len` | `1024` (default) | numerical stability knob |
| `fp8_amax_compute_algo` | `max` | safer than `most_recent` for long runs |

Always co-validate FP8 with a short loss-vs-BF16 reference run before committing.

## Feature rollout order

Run these in order. Each row is one round (one subagent call) — the candidate is `champion + this feature's flags`. Skip a row if the YAML already has it on.

| # | Feature name | Flags (the bundle) | Conditions | Expected gain |
|---|---|---|---|---|
| 1 | `cross-entropy-fusion` | `cross_entropy_fusion_impl=te` + `cross_entropy_loss_fusion=true` | always | 1–6% |
| 2 | `rope-fusion` | `apply_rope_fusion=true` + `enable_experimental=true` | always | 1–4% |
| 3 | `bf16-precision-aware-optimizer` | `use_precision_aware_optimizer=true` + `main_grads_dtype=bf16` + `exp_avg_dtype=bf16` + `exp_avg_sq_dtype=bf16` | DP/FSDP, multi-node esp. valuable | 4–10% |
| 4 | `turbo-attention` | `enable_primus_turbo=true` + `use_turbo_attention=true` | always (master switch lives here) | 3–10% |
| 5 | `turbo-parallel-linear` | `use_turbo_parallel_linear=true` | TP > 1 only | 1–4% |
| 6 | `turbo-grouped-mlp` (MoE) | `use_turbo_grouped_mlp=true` + `use_turbo_fused_act_with_probs=true` | MoE only | 3–10% |
| 7 | `fused-router` (MoE) | `moe_use_fused_router_with_aux_score=true` | MoE only | 1–3% |
| 8 | `sync-free-moe-stage-2` (MoE) | `turbo_sync_free_moe_stage=2` | MoE only; requires turbo + grouped-mlp on | 1–4% |
| 9 | `deepep` (MoE) | `use_turbo_deepep=true` + `moe_router_dtype=fp32` + `turbo_deepep_num_cu=80` (ep≤8) / `32` (ep 16–64) | MoE, EP > 1 | 5–20% (huge for cross-node EP) |
| 10 | `gradient-accumulation-fusion` (large dense / MoE) | `gradient_accumulation_fusion=true` | NOT for small dense — some YAMLs (e.g. `qwen3_8B-BF16`) ship it `false` on purpose | 1–4% |
| 11 | `legacy-grouped-gemm` (MoE, AMD current path) | `moe_use_legacy_grouped_gemm=true` | MoE on AMD until upstream new GG stabilizes | 1–3% |
| 12 | (FP8 only) `turbo-attention-fp8` | `enable_turbo_attention_float8=true` | only when the run is FP8 | 5–15% on T_comp |

**Not in the list** (intentionally skipped, do not enable as a default round):

- `use_turbo_rms_norm` — known bug per shipped MI355X configs (`# bug` comment). Wait for upstream fix.

## Per-round protocol (used by `tuning-loop` Step 4.0)

For each row in order:

1. If the YAML already has the bundle on, mark `skip` in the ledger and move to the next row.
2. Otherwise spawn one `purpose: candidate-eval` subagent with `plan = current champion + bundle.flags`, `notes: "feature=<name>"`.
3. Read the returned `result.md` summary. Decide:

   | Result | Action |
   |---|---|
   | `status=completed`, gain ≥ 1% vs parent | Promote to champion, proceed to next feature |
   | `status=completed`, gain in `[-1%, 1%]` | Keep on (small wins compound), still promote, proceed |
   | `status=completed`, gain < -1% | Roll back the feature, mark `not for this model` in ledger, proceed |
   | `status=failed` (init / runtime error) | If the bundle has > 1 flag: split the bundle by Primus convention (e.g. master+sub → enable master only first; if master alone is fine, the sub-flag is the culprit). Otherwise drop the feature and continue. |
   | `status=oom` / `numerical` / `hang` | Same as `optimize-memory` / `correctness-style` handling — escalate or roll back, never silently |

4. Append to the in-chat ledger:

   ```
   round_<n> (feature=<name>): champion + <bundle>
     result: tps=<>, gain vs parent = <%>, status=<>
     decision: promote | keep | dropped
   ```

When the order is exhausted (or the user's budget runs out), exit Phase A and hand off to `bottleneck-diagnose` (Phase B in `tuning-loop`).

### Optional parallelism within a round

If the next 2–3 features are mutually independent (e.g. rounds 6 `turbo-grouped-mlp` and 7 `fused-router` after 4 `turbo-attention` is in place), they may be spawned as parallel subagents from the same parent. Pick the winner (largest gain), promote it; the losers re-derive next round on top of the new champion.

This compresses wallclock without breaking the rule: each candidate still changes exactly one feature.

## Reference YAMLs (look here when in doubt)

| Reference | What it shows |
|---|---|
| `examples/megatron/configs/MI355X/qwen3_30B_A3B-BF16-pretrain.yaml` | dense MoE on MI355X with full Turbo + DeepEP + bf16 optim |
| `examples/megatron/configs/MI355X/qwen3_8B-BF16-pretrain.yaml` | small dense — note it explicitly disables `gradient_accumulation_fusion` |
| `examples/megatron/configs/MI355X/glm5-FP8-pretrain.yaml` | FP8 + Turbo |
| `primus/configs/modules/megatron/primus_turbo.yaml` | the master config defining every Turbo sub-flag (all default false) |

When the user gives a brand-new model, read the closest reference YAML once to see which features the Primus team has validated for that class.

## Reference YAMLs (look here when in doubt)

| Reference | What it shows |
|---|---|
| `examples/megatron/configs/MI355X/qwen3_30B_A3B-BF16-pretrain.yaml` | dense MoE on MI355X with full Turbo + DeepEP + bf16 optim |
| `examples/megatron/configs/MI355X/qwen3_8B-BF16-pretrain.yaml` | small dense — note it explicitly disables `gradient_accumulation_fusion` |
| `examples/megatron/configs/MI355X/glm5-FP8-pretrain.yaml` | FP8 + Turbo |
| `primus/configs/modules/megatron/primus_turbo.yaml` | the master config defining every Turbo sub-flag (all default false) |

When the user gives a brand-new model, read the closest reference YAML once to see which flags the Primus team has validated for that class.

## Important Notes

- **One feature per round, never batched.** Even though all features here are "should be on", enabling them one at a time is what makes per-feature gain attributable in the final report. If a multi-feature batch is attractive "to save time", spawn parallel subagents (each one feature) instead — same wallclock, no rule violation.
- **A bundle is one feature.** A coupled-flag bundle (master + sub, or 4 dtype flags that go with one master) counts as one round. The bundle table at the top of this skill is the source of truth for what may legitimately ship together.
- **This is Phase A, not part of BASELINE.** BASELINE measures the YAML as-given so the user can see how much each subsequent feature actually contributes.
- **Order matters less than coverage.** The order in the table is a sensible default (cheap & universal first, MoE-only later), but if the user's bottleneck profile suggests reordering (e.g. start with `deepep` if BASELINE was clearly cross-node alltoall-bound), it's fine — just keep one feature per round.
- **Some Turbo sub-flags are model-class-conditional** (e.g. `use_turbo_grouped_mlp` only for MoE). Skip the row when the condition doesn't hold — don't try and fail.
- **Known bug: `use_turbo_rms_norm: true` is broken.** Stays off the rollout until the comment `# bug` disappears from shipped MI355X configs.
- **bf16 precision-aware optimizer affects convergence.** For research / from-scratch training, validate on a short BF16-vs-fp32 reference window before committing for a long run.
- **Primus Turbo is patches, not magic.** If a feature breaks init, surface the actual error and decide with the user; don't silently strip the flag.
- **TP=1 for AMD is *not* in this skill** — it's a structural choice, lives in `optimize-comm` (AMD-specific note in the topology section).
