# Plan-3 — DeepSeek-V4 in Primus: Reporting fixes + Primus-Turbo enablement

> Plan-3 picks up where plan-2 left off (architecture-faithful rewrite +
> distributed re-validation, see `../plan-2/`) and is **strictly scoped**
> to two outcomes:
>
> 1. **Reporting + spec hygiene fixes** that came out of the Phase-19
>    smokes and the first full-Flash-size bring-up attempt (Phase 20 of
>    `../progress/status.md`):
>    * Megatron's per-iter TFLOPs number is computed by a generic
>      Transformer / MLA formula that does not know about V4's mHC
>      `K`-stream packing, single-latent KV, grouped low-rank O,
>      Compressor / Indexer side-paths, hash routing, or the V4 MTP
>      head — so the reported TFLOPs is misleading on V4 today.
>    * V4 attention + V4 dense-MLP projection helpers currently
>      `logger.warning(...)` and silently fall back to a vanilla
>      `nn.Linear` when `build_module(spec)` raises — a smoke at full
>      Flash dims surfaced this as `gather_output=True` /
>      `input_is_parallel=False` warnings on every projection. The
>      fallback masks real spec bugs and produces a model with
>      duplicated, non-TP-sharded weights.
> 2. **Primus-Turbo enablement** for V4 (turbo flash-attention with
>    learned sink + turbo DeepEP MoE dispatcher), mirroring the
>    `examples/megatron/configs/MI355X/deepseek_v2_lite-BF16-pretrain.yaml`
>    recipe. This unlocks the perf path that plan-2 P20 will measure
>    against and is the first step toward replacing V4's CSA dense
>    Python fallback with kernel-backed attention.
>
> Every other follow-up (full-Flash CSA kernel, FP8, convergence run,
> long-context, HF state-dict adapter) stays out of scope and is owned
> by plan-2's deferred items or a future plan.

## References

- Plan-2 wrap-up: `../progress/plan-2-summary.md`
- Plan-2 distributed validation log: `../progress/p19/`
- Phase-19 patches (1F1B + VPP `pp_tensor_shape` + `pp_token_pre_broadcast`):
  `primus/backends/megatron/patches/deepseek_v4_pp_shape_patches.py`,
  `primus/backends/megatron/patches/deepseek_v4_get_batch_patches.py`
- Primus-Turbo source tree: `../../Primus-Turbo/` (read-only reference)
- V2-Lite turbo recipe (template): `examples/megatron/configs/MI355X/deepseek_v2_lite-BF16-pretrain.yaml`
- Primus-Turbo dispatcher patch (already in tree): `primus/backends/megatron/patches/turbo/moe_dispatcher_patches.py`
- Primus-Turbo attention class (sink-attention API): `primus/backends/megatron/core/extensions/primus_turbo.py:PrimusTurboAttention`

## Documents

- [`01-roadmap.md`](./01-roadmap.md) — phase overview, dependency graph,
  exit criteria, and how plan-3 sits relative to plan-2.
- [`02-phase-details.md`](./02-phase-details.md) — phase-by-phase task
  list, design notes, edge cases, and risks.
- [`03-test-strategy.md`](./03-test-strategy.md) — gates per phase + the
  end-to-end turbo smoke that is plan-3's release gate.

## Scope

| In scope | Out of scope |
|---|---|
| V4-aware `num_floating_point_operations` patch | FLOPs accounting for non-V4 model types (untouched) |
| Strict `build_module` contract for V4 attention + dense-MLP projection | Generic Megatron spec hygiene outside V4 |
| `core_attention` submodule on `DeepseekV4AttentionSubmodules` (provider-built; turbo / TE / fused) | Replacing V4's CSA Python fallback with a fused indexer kernel (perf, separate plan) |
| `use_sink_attention` plumbing for V4 dense-SWA layers via Turbo | Kernel-side sink-attention support for HCA / CSA branches |
| V4 layer specs probe `args.use_turbo_deepep` and pick `PrimusTurboDeepEPTokenDispatcher` | New non-MoE dispatcher work |
| `run_deepseek_v4.sh` smoke with `enable_primus_turbo=true`, `use_turbo_attention=true`, `use_turbo_deepep=true` | FP8 / FP4 / Muon / convergence runs |

## Phase Map (added under Phase 19 in `../progress/status.md`)

| # | Theme | Source request |
|---|---|---|
| **P20** | V4-aware TFLOPs reporting | "分析一下megatron中每个iter打印的tflops是如何计算的，需要根据deepseek-v4模型结构重新修改一下" |
| **P21** | Strict spec build (no nn-fallback) | "代码里面不能这种情况下给warning，而是直接raise报错。修复一下这个问题。也检查一下其他的地方有没有类似的情况" |
| **P22** | `core_attention` submodule for V4 (turbo / TE) | "PrimusTurboSpecProvider里面参考PrimusTurboSpecProvider的core_attention…在deepseekv4attention里面，如果是走到core attention，那么可以直接调用" |
| **P23** | Turbo DeepEP dispatcher in V4 layer specs | "deepseek_v4_layer_specs.py里面选择dispatcher_cls时候，可以判断一下，如果开始了turbo deepep，那么使用turbo的dispatcher" |
| **P24** | Turbo attention + DeepEP smoke gate | "使用run_deepseek_v4跑通开启turbo attention和turbo deepep" |
