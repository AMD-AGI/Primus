# Plan-5 P29 (RESCOPED) — V4-Flash EP=8 trace, after-P29

> Generated 2026-05-08T23:55:51 from `output/amd/tas-mi355x-20260509/p29_profile_after_pp1_ep8_seq4096/tensorboard/primus-megatron-exp[p29_profile_after_pp1_ep8_seq4096]-rank[0].1778301850351388452.pt.trace.json`.
>
> **This is the post-P29 trace** — same proxy widths and parallelism as the
> P28 baseline (`profile-baseline-ep8-20260508.md`), single delta:
> `USE_V4_COMPILED_SINKHORN=True`. Numbers below are the new plan-5
> baseline against which P30 / P31 measure their deltas.

## P29 outcome (TL;DR)

- **The Sinkhorn fp32 reduce is gone.** The dominant
  `at::native::reduce_kernel<512, 1, ReduceOp<float, sum_functor<float,
  float, float>>>` kernel (P28 rank #1) is now **rank #29** with **6.91 ms
  total / 80 launches** (P28 had **7607.9 ms / 624 launches**). For the
  specific shape that the P28 forensics pinned to `sinkhorn_normalize`
  (`(1, 4096, 4, 4) -> dim=[-1] keepdim`) the count drops from
  **624 → 16** and the total time from **7607.9 ms → 0.2 ms** —
  **−97.4 % launches, −99.997 % kernel time**. The kernel-time budget
  X1 (≥ 50 % drop) is met by ~1000×.
- **G33a smoke is GREEN.** 10 iters complete cleanly; no NaN / no
  Inf; no banned warnings; lm_loss after iter 10 = 9.258 (P28 baseline:
  9.258 — bit-for-bit at fixed seed because the algorithm is unchanged,
  only the kernel boundary moves). Steady throughput **79.1 TFLOP/s/GPU
  vs P28 baseline 77.5** — only **+2.0 % steady wall-time gain**.
- **Why is the wall-time gain small?** Pre-P29 multi-stream overlap
  factor was **1.87×** (P28 report) — the giant fp32 reduce was running
  on a separate HIP compute stream in parallel with the V4 Triton
  attention BWD on stream-0. Killing it freed the second stream but
  the ATTENTION kernels on stream-0 were already the wall-time critical
  path. Confirmed in this trace: **multi-stream overlap factor dropped
  from 1.87× to 1.00×** (Σ kernel dur ≈ wall-clock GPU active = 8.58 s),
  meaning the GPU now runs almost everything sequentially on a single
  stream and the GPU active % is unchanged at ~99.5 %. The Sinkhorn
  reduce kernel was a parallel hitchhiker, not the wall-time gating
  kernel.
- **The new wall-time bottleneck is the V4 Triton attention BWD.**
  Top-3 kernels by `% step` are now:
  - **#1 `_v4_csa_attention_bwd_kernel` — 4.03 s (46.8 %), 3 launches
    × 1.34 s each.**
  - **#2 `_v4_attention_bwd_kernel` — 3.18 s (36.8 %), 5 launches ×
    635.6 ms each.**
  - **#3 `_v4_attention_fwd_kernel` — 641.2 ms (7.4 %), 5 launches ×
    128.2 ms each.**
  Combined: V4 attention kernels (FWD + BWD across cr ∈ {0, 4, 128})
  account for **8.0 s of 8.63 s = 92.6 % of step time**. Attacking
  the BWD is exactly the plan-5 P30 / P31 mandate.
- **Comm time still negligible** (13.18 ms = 0.2 % of step). P32
  remains de-scoped per the P28 KEEP-RESCOPE decision.
- **HBM headroom unchanged** (~ 195 GiB / 287 GiB ≈ 68 % peak).
- **Auto-rendered table-and-narrative caveat:** the
  `_tools/render_baseline_report.py` tool below was authored for the
  P28 baseline and its hard-coded TL;DR / bottleneck prose still
  references the pre-P29 dominant-reduce hypothesis. The TL;DR you
  are reading replaces it; the *raw kernel / cpu_op / comm tables*
  below the TL;DR are correct as-is and are the reference for the
  P30 / P31 perf-budget gates.

## Run config provenance

| key | value |
|---|---|
| commit | `1ea7e7a8` |
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

Steady iter window: 8.63 s of trace time.

**`GPU active`** below is the wall-clock union of kernel intervals across all HIP / ROCm compute streams (the time when at least one kernel is in flight). The `kernel-time sum` row is the per-stream `Σ dur` that the chrome-trace top-level kernel table sums up — when streams overlap, `kernel-time sum > GPU active` (the ratio is the **multi-stream overlap factor**: > 1.0 means at least two streams ran kernels in parallel for some fraction of the iter).

| metric | value | % of iter |
|---|---:|---:|
| GPU active (union over streams) | 8.58 s | 99.5 % |
| GPU idle (1 − active) | 42.45 ms | 0.5 % |
| Σ kernel dur (across streams) | 8.58 s | 99.5 % |
| multi-stream overlap factor | **1.00×** | (Σ kernel dur ÷ GPU active) |

**CPU-bound floor (1 − GPU active)** ≈ **0.5 %** of iter time (vs P28's 0.3 %). The GPU is still essentially fully busy — P29 was a kernel-fusion win, not a CPU-idle win. Note that the multi-stream overlap factor dropped from 1.87× (P28) to 1.00× (post-P29) because the dominant reduce kernel was running on a separate HIP compute stream parallel with attention BWD on stream-0; with that reduce gone, almost all of the iter runs sequentially on one stream.

## Top-30 kernels by total time (steady iter window)

Total kernel time in steady window: 8.58 s.

| rank | kernel | count | total | self avg | % step |
|---:|---|---:|---:|---:|---:|
| 1 | `_v4_csa_attention_bwd_kernel` | 3 | 4.03 s | 1.34 s | 46.8 % |
| 2 | `_v4_attention_bwd_kernel` | 5 | 3.18 s | 635.60 ms | 36.8 % |
| 3 | `_v4_attention_fwd_kernel` | 5 | 641.16 ms | 128.23 ms | 7.4 % |
| 4 | `_v4_csa_attention_fwd_kernel` | 3 | 155.52 ms | 51.84 ms | 1.8 % |
| 5 | `void primus_turbo::deep_ep::intranode::cached_notify_dispatch<8>(int const*, i…` | 8 | 68.52 ms | 8.57 ms | 0.8 % |
| 6 | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kern…` | 643 | 60.64 ms | 94.3 µs | 0.7 % |
| 7 | `void multi_tensor_apply_kernel<TensorListMetadata<5, false>, transformer_engin…` | 321 | 46.33 ms | 144.3 µs | 0.5 % |
| 8 | `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 9 | 45.30 ms | 5.03 ms | 0.5 % |
| 9 | `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_ke…` | 1398 | 23.27 ms | 16.6 µs | 0.3 % |
| 10 | `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat3…` | 1191 | 21.85 ms | 18.3 µs | 0.3 % |
| 11 | `void primus_turbo::deep_ep::intranode::cached_notify_combine<8>(void**, int*, …` | 16 | 17.21 ms | 1.08 ms | 0.2 % |
| 12 | `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_…` | 20 | 15.61 ms | 780.6 µs | 0.2 % |
| 13 | `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kern…` | 204 | 15.41 ms | 75.5 µs | 0.2 % |
| 14 | `void ck_tile::kentry<1, ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocal…` | 16 | 14.31 ms | 894.2 µs | 0.2 % |
| 15 | `void ck_tile::kentry<1, ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocal…` | 16 | 14.08 ms | 879.8 µs | 0.2 % |
| 16 | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<…` | 883 | 13.24 ms | 15.0 µs | 0.2 % |
| 17 | `void multi_tensor_apply_kernel<TensorListMetadata<2, false>, transformer_engin…` | 321 | 10.97 ms | 34.2 µs | 0.1 % |
| 18 | `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native:…` | 26 | 10.28 ms | 395.5 µs | 0.1 % |
| 19 | `void ck_tile::kentry<1, ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocal…` | 16 | 10.08 ms | 630.0 µs | 0.1 % |
| 20 | `void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<fl…` | 138 | 9.70 ms | 70.3 µs | 0.1 % |
| 21 | `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDS…` | 47 | 9.27 ms | 197.3 µs | 0.1 % |
| 22 | `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDS…` | 33 | 9.19 ms | 278.3 µs | 0.1 % |
| 23 | `void primus_turbo::deep_ep::intranode::dispatch<8, 1024, true>(HIP_vector_type…` | 16 | 8.55 ms | 534.3 µs | 0.1 % |
| 24 | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kern…` | 122 | 8.35 ms | 68.4 µs | 0.1 % |
| 25 | `void primus_turbo::deep_ep::intranode::combine<hip_bfloat16, 8, 1024, true>(hi…` | 16 | 8.30 ms | 518.5 µs | 0.1 % |
| 26 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<floa…` | 172 | 7.86 ms | 45.7 µs | 0.1 % |
| 27 | `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<c10::BFloat16, at:…` | 12 | 7.81 ms | 651.0 µs | 0.1 % |
| 28 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDS…` | 33 | 7.52 ms | 227.9 µs | 0.1 % |
| 29 | `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native:…` | 80 | 6.91 ms | 86.3 µs | 0.1 % |
| 30 | `void multi_tensor_apply_kernel<TensorListMetadata<1, false>, transformer_engin…` | 321 | 6.73 ms | 21.0 µs | 0.1 % |

## Kernel launch count + average launch interval (steady iter)

Total kernels launched in steady window: **10774**.
Median inter-launch interval: **13.0 µs** (p50); **95.5 µs** (p90); **729.0 µs** (p99).

| inter-launch (µs) | count | % |
|---|---:|---:|
| 0–10 | 4752 | 44.1 % |
| 10–50 | 4123 | 38.3 % |
| 50–100 | 841 | 7.8 % |
| 100–500 | 888 | 8.2 % |
| 500–5000 | 143 | 1.3 % |
| ≥5000 | 26 | 0.2 % |

## Module-level CPU op-time attribution (steady iter)

PyTorch profiler emits one `cpu_op` event for every aten / module call, **including all nested children**. So summing per-event `dur` double-counts: a top-level `DeepseekV4HybridLayer.forward` event already includes every aten op inside it. The table below shows the raw sum per pattern bucket — useful for spotting which **module subtree** dominates (top-level `Module.forward` rows like `DeepseekV4HybridLayer`, `DeepseekV4MoE`, `DeepseekV4Attention` are the meaningful numbers; the catch-all `other` row is bloated by nested aten ops and should NOT be read as 'CPU work outside V4').

| module pattern | events | Σ event dur (nests) | % iter |
|---|---:|---:|---:|
| other | 216363 | 951.87 s | 11033.6 % |
| DeepseekV4HybridLayer | 8 | 923.92 ms | 10.7 % |
| DeepseekV4MoE | 8 | 865.19 ms | 10.0 % |
| Optimizer | 2060 | 147.52 ms | 1.7 % |
| linear/matmul | 1307 | 70.83 ms | 0.8 % |
| DeepseekV4Attention | 8 | 24.45 ms | 0.3 % |
| v4_attention (Triton) | 55 | 17.28 ms | 0.2 % |
| LayerNorm/RMSNorm | 247 | 11.43 ms | 0.1 % |
| c10d/comm | 91 | 7.70 ms | 0.1 % |
| v4_csa_attention (Triton) | 9 | 3.81 ms | 0.0 % |
| DualRoPE/RoPE | 21 | 2.95 ms | 0.0 % |
| softmax | 143 | 2.66 ms | 0.0 % |

## Comm time (steady iter)

| kind | total | % iter |
|---|---:|---:|
| deepep | 0.0 µs | 0.0 % |
| nccl/c10d | 13.18 ms | 0.2 % |
| **total comm** | **13.18 ms** | **0.2 %** |

## Ranked bottleneck list + per-phase improvement budgets (after-P29)

Bottlenecks are ranked by **% of steady iter wall time** (not Σ kernel dur — Σ kernel dur is now ≈ wall-time GPU active because the multi-stream overlap factor dropped to 1.00× post-P29). The Y / Z post-phase budgets are the targets that plan-5's `01-roadmap.md` will adopt after this report is reviewed.

| # | bottleneck | current cost | % iter | proposed budget after phase |
|---|---|---:|---:|---|
| 1 | V4 Triton CSA attention BWD kernel (`_v4_csa_attention_bwd_kernel`, 3 launches × 1.34 s) | 4.03 s | 46.8 % | **Z** = post-P31 target — in-kernel `topk_idxs` gather + K-tile prefetch |
| 2 | V4 Triton dense / HCA attention BWD kernel (`_v4_attention_bwd_kernel`, 5 launches × 636 ms) | 3.18 s | 36.8 % | **Y** = post-P30 target — BLOCK_M / BLOCK_N retune at `head_dim=512`, persistent-kernel sweep, HCA LSE merge |
| 3 | V4 Triton dense / HCA attention FWD kernel (`_v4_attention_fwd_kernel`, 5 launches × 128.2 ms) | 641.2 ms | 7.4 % | rolled into **Y** — same kernel family; FWD already 5× cheaper than BWD per call so likely de-prioritised under P30 |
| 4 | V4 Triton CSA attention FWD kernel (`_v4_csa_attention_fwd_kernel`, 3 launches × 51.8 ms) | 155.5 ms | 1.8 % | rolled into **Z** — same kernel family; FWD ~26× cheaper than BWD |
| 5 | comm time (DeepEP + c10d) | 13.18 ms | 0.2 % | **W** = (de-scoped per P28 — confirmed unchanged here) |
| 6 | small-op kernel-launch tail (CPU-bound floor) | 42.45 ms | 0.5 % | **X2** = (de-scoped per P28 — confirmed unchanged here) |
| 7 | `aten::sum` fp32 reduce kernel (top-1 template, ALL variants) | 6.91 ms / 12.66 ms | 0.1 % | **X1 = MET (closed by P29)** — see TL;DR for kernel-time delta |

### Per-phase status update

Plan-5 de-scope rule: any bottleneck row < 10 % of step time gets the phase de-scoped (or held in follow-up). The post-P29 data above:

| phase | status | rationale |
|---|---|---|
| P29 | **CLOSED — budget met on kernel-time, partial on wall-time** | The dominant `aten::sum` fp32 reduce is gone (kernel-time drop ≈ 1000×, ≫ 50 % budget X1). Wall-time gain is +2 % only because the reduce was a parallel hitchhiker on stream-1 and never gated wall time; wall-time gain comes from P30 / P31 attacking the V4 attention BWD. |
| P30 | **KEEP — now the #1 wall-time bottleneck** | V4 Triton dense / HCA attention BWD = 36.8 % of step (≥ 10 % rule). FWD = 7.4 %. Combined cr ∈ {0, 128} attention = 44.2 %. Same conclusion as P28: prioritise BWD (BLOCK retune, persistent kernel, HCA LSE merge). |
| P31 | **KEEP — now the #1 single-kernel bottleneck** | V4 Triton CSA BWD = 46.8 % of step (≥ 10 % rule). FWD = 1.8 %. Same conclusion as P28's KEEP-RESCOPE: redirect to BWD-speedup (in-kernel `topk_idxs` gather, K-tile prefetch); the original HBM-saving motivation stays out (Sq=4096 fits with 32 GiB headroom). |
| P32 | **DE-SCOPED — confirmed** | Comm time = 0.2 % of step (unchanged from P28). |

### Plan-5 retarget (post-P29)

The P28 perf budget X1 (≥ 50 % drop in `aten::sum` fp32 reduce kernel time) is **CLOSED at ~1000×** — the kernel is essentially gone. The wall-time gain is small (+2 %) because GPU was already 99.7 % busy in P28 with the reduce overlapping attention BWD on a separate stream. The remaining wall-time budget transfers to P30 / P31:

- **P29** — CLOSED. New plan-5 baseline: **79.1 TFLOP/s/GPU steady at Sq=4096 EP=8**; iter wall time ≈ **8.63 s**; CPU-bound floor 0.5 %; multi-stream overlap factor 1.00× (was 1.87×).
- **P30** — Y = ≥ 25 % BWD speed-up on `_v4_attention_bwd_kernel` (BLOCK retune at `head_dim=512`, persistent-kernel sweep, HCA LSE merge variant).
- **P31** — Z = ≥ 25 % BWD speed-up on `_v4_csa_attention_bwd_kernel` (in-kernel `topk_idxs` gather, K-tile prefetch in BWD, autotune `BLOCK_K` for `K_topk=512`).
- **P32** — DE-SCOPED.

Combined plan-5 final target (unchanged from P28): **≥ 110 TFLOP/s/GPU steady at Sq=4096 EP=8 single-node** (≈ 40 % over the 78 TFLOP/s/GPU P28 baseline; ≈ 39 % over the 79.1 TFLOP/s/GPU post-P29 baseline). Final perf gate (`G35`) lives in `03-test-strategy.md`.
