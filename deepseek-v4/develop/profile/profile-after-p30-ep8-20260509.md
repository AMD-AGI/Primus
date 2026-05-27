# Plan-5 P30 — V4-Flash EP=8 trace, after dense + HCA SWA K-loop pruning

> Generated 2026-05-09T03:37:15 from `output/amd/tas-mi355x-20260509/p30_profile_swa_prune_pp1_ep8_seq4096/tensorboard/primus-megatron-exp[p30_profile_swa_prune_pp1_ep8_seq4096]-rank[0].1778315746463596352.pt.trace.json`.
>
> **This is the updated post-P30 trace** after extending SWA K-loop
> pruning from dense `compress_ratio == 0` to HCA `compress_ratio == 128`.
> HCA now passes a pool-only mask `[S, P]` plus `hca_local_seqlen=S`; the
> Triton kernel runs a pruned local-SWA loop and a short pool-suffix loop
> under the same joint softmax.

## P30 outcome (TL;DR)

- **The cr=128 regression is fixed.** All five `_v4_attention_bwd_kernel`
  launches are now **30-34 ms** (`31.1, 34.1, 33.7, 30.3, 30.9 ms`).
  The two HCA `compress_ratio == 128` launches are no longer the 600 ms+
  outliers.
- **Dense + HCA `v4_attention` is no longer a bottleneck.**
  `_v4_attention_bwd_kernel` drops from post-P29 **3.18 s → 160 ms**
  (**−95.0 %**) and `_v4_attention_fwd_kernel` drops **641 ms → 30 ms**
  (**−95.3 %**). Combined dense/HCA attention kernel time is now
  **192 ms / 3.9 %** of the step.
- **Wall time improves again.** Trace steady step is **4.94 s**, down from
  post-P29 **8.63 s** (**−42.8 %**) and from the dense-only P30 trace
  **6.44 s** (**−23.3 %**). The latest 10-iter smoke reaches
  **138.4 TFLOP/s/GPU** steady average.
- **CSA is now the dominant target.** `_v4_csa_attention_bwd_kernel`
  remains **4.04 s / 81.9 %** of the step. P31 should stay focused on
  CSA BWD, especially in-kernel `topk_idxs` gather / scatter-add and
  K-tile prefetching.
- **GPU remains saturated.** GPU active is **99.0 %** with overlap factor
  **1.00×**. The optimisation shortened the critical path rather than
  shifting the run to CPU launch overhead.
- **Auto-rendered table caveat:** `_tools/render_baseline_report.py`
  still emits baseline-era prose in lower sections. The raw tables are
  correct; this P30 TL;DR is the authoritative interpretation.

## Run config provenance

| key | value |
|---|---|
| commit | `TBD-p30` |
| host | `mi355-gpu-14` |
| container | `dev_primus_wenx_693` |
| seq_length | 4096 |
| parallel | TP=1 PP=1 EP=8 |
| micro_batch_size | 1 |
| global_batch_size | 8 |
| num_layers | 8 |
| num_experts | 256 (EP=8 -> 32/rank) |
| moe_router_topk | 6 |
| moe_ffn_hidden_size | 2048 |
| index_topk | 512 |
| compress_ratios | `[0,0,4,128,4,128,4,0]` |
| **perf knobs** | |
| use_v4_triton_attention | True |
| use_v4_triton_csa_attention | True |
| use_turbo_deepep | True |
| use_turbo_grouped_mlp | True |
| use_turbo_attention | False (must be False — Turbo would override V4 Triton dense path) |

## Per-iter wall time

Sourced from the training stdout log (Megatron's `elapsed time per iteration (ms)`). The plan-4 ratchet skips the first 2 iters (`log_avg_skip_iterations: 2`), so iter 1 / 2 are absent here.

| iter | ms / iter | TFLOP/s/GPU | lm_loss |
|---:|---:|---:|---:|

## GPU vs CPU active / idle %

Steady iter window: 4.94 s of trace time.

**`GPU active`** below is the wall-clock union of kernel intervals across all HIP / ROCm compute streams (the time when at least one kernel is in flight). The `kernel-time sum` row is the per-stream `Σ dur` that the chrome-trace top-level kernel table sums up — when streams overlap, `kernel-time sum > GPU active` (the ratio is the **multi-stream overlap factor**: > 1.0 means at least two streams ran kernels in parallel for some fraction of the iter).

| metric | value | % of iter |
|---|---:|---:|
| GPU active (union over streams) | 4.89 s | 99.0 % |
| GPU idle (1 − active) | 47.44 ms | 1.0 % |
| Σ kernel dur (across streams) | 4.89 s | 99.0 % |
| multi-stream overlap factor | **1.00×** | (Σ kernel dur ÷ GPU active) |

**CPU-bound floor (1 − GPU active)** ≈ **1.0 %** of iter time. This is the headline number for plan-5 P29 (small-op fusion).

## Top-30 kernels by total time (steady iter window)

Total kernel time in steady window: 4.89 s.

| rank | kernel | count | total | self avg | % step |
|---:|---|---:|---:|---:|---:|
| 1 | `_v4_csa_attention_bwd_kernel` | 3 | 4.04 s | 1.35 s | 81.9 % |
| 2 | `_v4_attention_bwd_kernel` | 5 | 160.14 ms | 32.03 ms | 3.2 % |
| 3 | `_v4_csa_attention_fwd_kernel` | 3 | 153.19 ms | 51.06 ms | 3.1 % |
| 4 | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kern…` | 643 | 59.46 ms | 92.5 µs | 1.2 % |
| 5 | `void multi_tensor_apply_kernel<TensorListMetadata<5, false>, transformer_engin…` | 321 | 46.33 ms | 144.3 µs | 0.9 % |
| 6 | `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 9 | 31.62 ms | 3.51 ms | 0.6 % |
| 7 | `_v4_attention_fwd_kernel` | 5 | 30.13 ms | 6.03 ms | 0.6 % |
| 8 | `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_ke…` | 1393 | 23.02 ms | 16.5 µs | 0.5 % |
| 9 | `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat3…` | 1191 | 21.87 ms | 18.4 µs | 0.4 % |
| 10 | `void primus_turbo::deep_ep::intranode::cached_notify_combine<8>(void**, int*, …` | 16 | 18.26 ms | 1.14 ms | 0.4 % |
| 11 | `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_…` | 20 | 15.53 ms | 776.6 µs | 0.3 % |
| 12 | `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kern…` | 204 | 15.37 ms | 75.3 µs | 0.3 % |
| 13 | `void primus_turbo::deep_ep::intranode::cached_notify_dispatch<8>(int const*, i…` | 8 | 14.87 ms | 1.86 ms | 0.3 % |
| 14 | `void ck_tile::kentry<1, ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocal…` | 16 | 14.24 ms | 890.1 µs | 0.3 % |
| 15 | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<…` | 883 | 13.26 ms | 15.0 µs | 0.3 % |
| 16 | `void ck_tile::kentry<1, ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocal…` | 16 | 13.00 ms | 812.8 µs | 0.3 % |
| 17 | `void multi_tensor_apply_kernel<TensorListMetadata<2, false>, transformer_engin…` | 321 | 10.98 ms | 34.2 µs | 0.2 % |
| 18 | `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native:…` | 24 | 10.03 ms | 417.8 µs | 0.2 % |
| 19 | `void ck_tile::kentry<1, ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocal…` | 16 | 9.77 ms | 610.8 µs | 0.2 % |
| 20 | `void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<fl…` | 138 | 9.75 ms | 70.7 µs | 0.2 % |
| 21 | `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDS…` | 47 | 9.44 ms | 200.8 µs | 0.2 % |
| 22 | `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDS…` | 33 | 9.24 ms | 279.9 µs | 0.2 % |
| 23 | `void primus_turbo::deep_ep::intranode::dispatch<8, 1024, true>(HIP_vector_type…` | 16 | 8.54 ms | 533.7 µs | 0.2 % |
| 24 | `void primus_turbo::deep_ep::intranode::combine<hip_bfloat16, 8, 1024, true>(hi…` | 16 | 8.31 ms | 519.6 µs | 0.2 % |
| 25 | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kern…` | 122 | 8.13 ms | 66.6 µs | 0.2 % |
| 26 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<floa…` | 162 | 7.84 ms | 48.4 µs | 0.2 % |
| 27 | `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<c10::BFloat16, at:…` | 12 | 7.77 ms | 647.6 µs | 0.2 % |
| 28 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDS…` | 33 | 7.62 ms | 230.8 µs | 0.2 % |
| 29 | `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native:…` | 80 | 6.96 ms | 87.0 µs | 0.1 % |
| 30 | `void multi_tensor_apply_kernel<TensorListMetadata<1, false>, transformer_engin…` | 321 | 6.73 ms | 21.0 µs | 0.1 % |

## Kernel launch count + average launch interval (steady iter)

Total kernels launched in steady window: **10722**.
Median inter-launch interval: **13.4 µs** (p50); **93.9 µs** (p90); **726.5 µs** (p99).

| inter-launch (µs) | count | % |
|---|---:|---:|
| 0–10 | 4738 | 44.2 % |
| 10–50 | 4094 | 38.2 % |
| 50–100 | 845 | 7.9 % |
| 100–500 | 872 | 8.1 % |
| 500–5000 | 150 | 1.4 % |
| ≥5000 | 22 | 0.2 % |

## Module-level CPU op-time attribution (steady iter)

PyTorch profiler emits one `cpu_op` event for every aten / module call, **including all nested children**. So summing per-event `dur` double-counts: a top-level `DeepseekV4HybridLayer.forward` event already includes every aten op inside it. The table below shows the raw sum per pattern bucket — useful for spotting which **module subtree** dominates (top-level `Module.forward` rows like `DeepseekV4HybridLayer`, `DeepseekV4MoE`, `DeepseekV4Attention` are the meaningful numbers; the catch-all `other` row is bloated by nested aten ops and should NOT be read as 'CPU work outside V4').

| module pattern | events | Σ event dur (nests) | % iter |
|---|---:|---:|---:|
| other | 216096 | 554.52 s | 11234.9 % |
| DeepseekV4HybridLayer | 8 | 306.68 ms | 6.2 % |
| DeepseekV4MoE | 8 | 236.99 ms | 4.8 % |
| Optimizer | 2060 | 164.08 ms | 3.3 % |
| linear/matmul | 1307 | 73.01 ms | 1.5 % |
| DeepseekV4Attention | 8 | 32.67 ms | 0.7 % |
| v4_attention (Triton) | 55 | 17.02 ms | 0.3 % |
| c10d/comm | 91 | 15.75 ms | 0.3 % |
| LayerNorm/RMSNorm | 247 | 12.30 ms | 0.2 % |
| DualRoPE/RoPE | 21 | 4.87 ms | 0.1 % |
| v4_csa_attention (Triton) | 9 | 3.38 ms | 0.1 % |
| softmax | 143 | 2.74 ms | 0.1 % |

## Comm time (steady iter)

| kind | total | % iter |
|---|---:|---:|
| deepep | 0.0 µs | 0.0 % |
| nccl/c10d | 20.87 ms | 0.4 % |
| **total comm** | **20.87 ms** | **0.4 %** |

## Ranked bottleneck list + per-phase improvement budgets

Bottlenecks are ranked by **% of steady iter wall time** (not Σ kernel dur — that double-counts overlapping streams). The X / Y / Z / W per-phase budgets are the post-phase TARGETS that plan-5's `01-roadmap.md` will adopt after this report is reviewed.

| # | bottleneck | current cost | % iter | proposed budget after phase |
|---|---|---:|---:|---|
| 1 | `aten::sum` fp32 reduce kernel (top-1 template: 80 launches × ~87.0 µs) | 6.96 ms | 0.1 % | **X1** = post-P29 target — root-cause + fuse / move to bf16 master / replace with Triton fused bias-grad reduce |
| 2 | V4 Triton CSA attention kernel time (cr == 4, BWD-dominated) | 4.19 s | 85.0 % | **Z** = post-P31 target — in-kernel `topk_idxs` gather + K-tile prefetch |
| 3 | V4 Triton attention kernel time (cr ∈ {0, 128}, BWD-dominated) | 192.19 ms | 3.9 % | **Y** = post-P30 target — autotune BWD blocks, persistent-kernel sweep, HCA LSE merge |
| 4 | small-op kernel-launch tail (CPU-bound floor) | 47.44 ms | 1.0 % | **X2** = (de-scoped — see below) |
| 5 | comm time (DeepEP + c10d) | 20.87 ms | 0.4 % | **W** = (de-scoped — see below) |

### Per-phase de-scope decisions

Plan-5's de-scope rule: any bottleneck row < 10 % of step time gets its phase de-scoped. The data above is the input.

| phase | decision | rationale |
|---|---|---|
| P29 | **KEEP — RESCOPE** | CPU-bound floor is 1.0 % (≪ 10 % rule), so the original P29 mandate (small-op kernel-launch fusion via torch.compile or Triton-fused Compressor / Indexer / MoE-router chains) is **de-scoped**. P29 is **redirected** to root-cause + eliminate the dominant `aten::sum` fp32 reduce (0.1 % of step, 87 % of Σ kernel dur — the single largest line on the chrome-trace top-N table). Likely fix: identify whether it is bias-gradient sum-over-tokens in expert BWD or fp32 master-grad accumulation in `DistributedOptimizer`, and either fuse into a Triton kernel or move the reduction to bf16 / FP8. |
| P30 | **KEEP** | V4 Triton attention (dense + HCA) kernel time = 3.9 % of step (≥ 10 % rule). P30 must prioritise **BWD** (currently ~5 × FWD): BLOCK_M / BLOCK_N retune for head_dim=512, persistent-kernel sweep, HCA LSE-merge variant to cut the per-call cost. |
| P31 | **KEEP — RESCOPE** | V4 Triton CSA kernel time = 85.0 % of step (≥ 10 % rule). But HBM headroom is generous (~ 95 GiB free at peak), so **the original P31 motivation (cut the wrapper-side gather to fit Sq=4096) is no longer needed** — Sq=4096 already fits. P31 is **redirected** to BWD-speedup tasks: in-kernel `topk_idxs` gather to cut wrapper-side `torch.gather` + scatter-add overhead, K-tile prefetch in BWD, autotune BLOCK_K for K_topk=512. |
| P32 | **DE-SCOPE** | Comm time = 0.4 % of step (≪ 10 % rule). DeepEP + c10d are essentially free at single-node EP=8. Plan-5 P32 (pipeline / comm / optimizer overlap, recompute knobs) is **de-scoped** unless a P29 or P30 / P31 outcome materially raises comm cost (e.g. cross-node EP, or a structural change that re-introduces `overlap_grad_reduce` complexity). |

### Proposed plan-5 retarget (post-P28)

Plan-5's roadmap should adopt the P28 retarget on review:

- **P29** — `aten::sum` fp32 reduce: root-cause (likely MoE bias-grad sum-over-tokens or DistributedOptimizer fp32 master-grad accumulation), then fuse / replace. Budget X1: kill ≥ 50 % of the 7.6 s reduce kernel time.
- **P30** — V4 Triton dense / HCA attention BWD performance. Budget Y: ≥ 25 % BWD speed-up via BLOCK retune + persistent-kernel + HCA LSE merge.
- **P31** — V4 Triton CSA attention BWD performance (in-kernel `topk_idxs` gather + K-tile prefetch). Budget Z: ≥ 25 % CSA BWD speed-up.
- **P32** — DE-SCOPED. The comm / overlap budget is already won.

Combined target: **plan-5 final ≥ 110 TFLOP/s/GPU steady at Sq=4096 EP=8 single-node** (40 %+ over the 78 TFLOP/s/GPU baseline pinned in this report). Final perf gate (`G35`) lives in `03-test-strategy.md`.
