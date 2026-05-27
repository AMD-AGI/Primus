# Plan-5 P31 — V4-Flash EP=8 trace after CSA in-kernel gather

> Generated 2026-05-09T04:26:39 from `output/amd/tas-mi355x-20260509/p31_profile_csa_in_kernel_gather_pp1_ep8_seq4096/tensorboard/primus-megatron-exp[p31_profile_csa_in_kernel_gather_pp1_ep8_seq4096]-rank[0].1778318699651415524.pt.trace.json`.

## Key findings (TL;DR)

- **P31 improves the proxy step.** Steady trace window is **4.32 s** (≈ **158.5 TFLOP/s/GPU**) vs P30b **4.94 s** (≈ **138.4 TFLOP/s/GPU**): **-12.7 % step time / +14.5 % throughput**.
- **CSA BWD remains the dominant kernel, but moved in the right direction.** P31 replaces wrapper-side `torch.gather` + gathered-gradient scatter with `_v4_csa_attention_pool_{fwd,bwd}_kernel`. CSA BWD drops from **4.04 s** (P30b) to **3.50 s** (**-13.5 %**); CSA FWD drops from ~**153 ms** to **123.5 ms** (**-19 %**).
- **GPU is still fully busy.** Wall-clock GPU active = 4.27 s of 4.32 s steady iter ≈ **99.0 %** with overlap factor **1.00×**. The residual bottleneck is kernel math / atomics, not CPU launch overhead.
- **Dense/HCA attention stayed stable.** `_v4_attention_bwd_kernel` remains **160.99 ms** across five launches and `_v4_attention_fwd_kernel` remains **30.56 ms**, matching the P30b range.
- **Residual P31 work is now clear.** In-kernel gather bought meaningful e2e improvement, but did not reach the original ≥25 % CSA-BWD budget alone. The remaining target is the per-row CSA BWD design itself: atomic density into `dpool`, sparse K-tile bandwidth, and a possible block-row / tensor-core redesign.

## Run config provenance

| key | value |
|---|---|
| commit | `pending-p31` |
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

Steady iter window: 4.32 s of trace time.

**`GPU active`** below is the wall-clock union of kernel intervals across all HIP / ROCm compute streams (the time when at least one kernel is in flight). The `kernel-time sum` row is the per-stream `Σ dur` that the chrome-trace top-level kernel table sums up — when streams overlap, `kernel-time sum > GPU active` (the ratio is the **multi-stream overlap factor**: > 1.0 means at least two streams ran kernels in parallel for some fraction of the iter).

| metric | value | % of iter |
|---|---:|---:|
| GPU active (union over streams) | 4.27 s | 99.0 % |
| GPU idle (1 − active) | 43.70 ms | 1.0 % |
| Σ kernel dur (across streams) | 4.27 s | 99.0 % |
| multi-stream overlap factor | **1.00×** | (Σ kernel dur ÷ GPU active) |

**CPU-bound floor (1 − GPU active)** ≈ **1.0 %** of iter time. This is the headline number for plan-5 P29 (small-op fusion).

## Top-30 kernels by total time (steady iter window)

Total kernel time in steady window: 4.27 s.

| rank | kernel | count | total | self avg | % step |
|---:|---|---:|---:|---:|---:|
| 1 | `_v4_csa_attention_pool_bwd_kernel` | 3 | 3.50 s | 1.17 s | 81.0 % |
| 2 | `_v4_attention_bwd_kernel` | 5 | 160.99 ms | 32.20 ms | 3.7 % |
| 3 | `_v4_csa_attention_pool_fwd_kernel` | 3 | 123.51 ms | 41.17 ms | 2.9 % |
| 4 | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kern…` | 643 | 59.53 ms | 92.6 µs | 1.4 % |
| 5 | `void multi_tensor_apply_kernel<TensorListMetadata<5, false>, transformer_engin…` | 321 | 46.31 ms | 144.3 µs | 1.1 % |
| 6 | `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 9 | 32.07 ms | 3.56 ms | 0.7 % |
| 7 | `_v4_attention_fwd_kernel` | 5 | 30.56 ms | 6.11 ms | 0.7 % |
| 8 | `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_ke…` | 1390 | 23.13 ms | 16.6 µs | 0.5 % |
| 9 | `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat3…` | 1191 | 21.86 ms | 18.4 µs | 0.5 % |
| 10 | `void primus_turbo::deep_ep::intranode::cached_notify_combine<8>(void**, int*, …` | 16 | 15.14 ms | 946.1 µs | 0.4 % |
| 11 | `void primus_turbo::deep_ep::intranode::cached_notify_dispatch<8>(int const*, i…` | 8 | 14.42 ms | 1.80 ms | 0.3 % |
| 12 | `void ck_tile::kentry<1, ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocal…` | 16 | 14.18 ms | 886.4 µs | 0.3 % |
| 13 | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<…` | 883 | 13.24 ms | 15.0 µs | 0.3 % |
| 14 | `void ck_tile::kentry<1, ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocal…` | 16 | 12.86 ms | 803.9 µs | 0.3 % |
| 15 | `void multi_tensor_apply_kernel<TensorListMetadata<2, false>, transformer_engin…` | 321 | 10.97 ms | 34.2 µs | 0.3 % |
| 16 | `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native:…` | 24 | 10.26 ms | 427.4 µs | 0.2 % |
| 17 | `void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<fl…` | 138 | 9.73 ms | 70.5 µs | 0.2 % |
| 18 | `void ck_tile::kentry<1, ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocal…` | 16 | 9.65 ms | 602.9 µs | 0.2 % |
| 19 | `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDS…` | 33 | 9.37 ms | 284.0 µs | 0.2 % |
| 20 | `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDS…` | 47 | 9.34 ms | 198.6 µs | 0.2 % |
| 21 | `void primus_turbo::deep_ep::intranode::dispatch<8, 1024, true>(HIP_vector_type…` | 16 | 8.64 ms | 539.7 µs | 0.2 % |
| 22 | `void primus_turbo::deep_ep::intranode::combine<hip_bfloat16, 8, 1024, true>(hi…` | 16 | 8.29 ms | 518.3 µs | 0.2 % |
| 23 | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kern…` | 122 | 8.25 ms | 67.6 µs | 0.2 % |
| 24 | `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<c10::BFloat16, at:…` | 12 | 7.76 ms | 646.5 µs | 0.2 % |
| 25 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDS…` | 33 | 7.37 ms | 223.4 µs | 0.2 % |
| 26 | `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kern…` | 192 | 7.23 ms | 37.7 µs | 0.2 % |
| 27 | `void multi_tensor_apply_kernel<TensorListMetadata<1, false>, transformer_engin…` | 321 | 6.71 ms | 20.9 µs | 0.2 % |
| 28 | `void at::native::vectorized_elementwise_kernel<8, at::native::AUnaryFunctor<c1…` | 12 | 5.72 ms | 476.9 µs | 0.1 % |
| 29 | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<…` | 115 | 4.93 ms | 42.9 µs | 0.1 % |
| 30 | `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x16x32_MI16x16x1_SN_LDSB1_AF…` | 16 | 4.88 ms | 304.8 µs | 0.1 % |

## Kernel launch count + average launch interval (steady iter)

Total kernels launched in steady window: **10644**.
Median inter-launch interval: **13.3 µs** (p50); **89.7 µs** (p90); **574.3 µs** (p99).

| inter-launch (µs) | count | % |
|---|---:|---:|
| 0–10 | 4695 | 44.1 % |
| 10–50 | 4123 | 38.7 % |
| 50–100 | 837 | 7.9 % |
| 100–500 | 849 | 8.0 % |
| 500–5000 | 118 | 1.1 % |
| ≥5000 | 21 | 0.2 % |

## Module-level CPU op-time attribution (steady iter)

PyTorch profiler emits one `cpu_op` event for every aten / module call, **including all nested children**. So summing per-event `dur` double-counts: a top-level `DeepseekV4HybridLayer.forward` event already includes every aten op inside it. The table below shows the raw sum per pattern bucket — useful for spotting which **module subtree** dominates (top-level `Module.forward` rows like `DeepseekV4HybridLayer`, `DeepseekV4MoE`, `DeepseekV4Attention` are the meaningful numbers; the catch-all `other` row is bloated by nested aten ops and should NOT be read as 'CPU work outside V4').

| module pattern | events | Σ event dur (nests) | % iter |
|---|---:|---:|---:|
| other | 215860 | 484.63 s | 11223.8 % |
| DeepseekV4HybridLayer | 8 | 270.68 ms | 6.3 % |
| DeepseekV4MoE | 8 | 205.35 ms | 4.8 % |
| Optimizer | 2060 | 143.11 ms | 3.3 % |
| linear/matmul | 1307 | 70.46 ms | 1.6 % |
| DeepseekV4Attention | 8 | 26.94 ms | 0.6 % |
| v4_attention (Triton) | 55 | 16.54 ms | 0.4 % |
| LayerNorm/RMSNorm | 247 | 11.65 ms | 0.3 % |
| c10d/comm | 91 | 6.71 ms | 0.2 % |
| DualRoPE/RoPE | 21 | 3.07 ms | 0.1 % |
| softmax | 143 | 2.54 ms | 0.1 % |

## Comm time (steady iter)

| kind | total | % iter |
|---|---:|---:|
| deepep | 0.0 µs | 0.0 % |
| nccl/c10d | 11.58 ms | 0.3 % |
| **total comm** | **11.58 ms** | **0.3 %** |

## Ranked bottleneck list + per-phase improvement budgets

Bottlenecks are ranked by **% of steady iter wall time** (not Σ kernel dur — that double-counts overlapping streams). This post-P31 trace keeps the P30/P31 comparison attached to the live kernel names.

| # | bottleneck | current cost | % iter | proposed budget after phase |
|---|---|---:|---:|---|
| 1 | V4 Triton CSA pool BWD (`_v4_csa_attention_pool_bwd_kernel`, 3 launches × 1.17 s) | 3.50 s | 81.0 % | Remaining P31 target: reduce per-row sparse BWD atomics / bandwidth |
| 2 | V4 Triton dense/HCA attention BWD (`_v4_attention_bwd_kernel`, 5 launches × 32.2 ms) | 160.99 ms | 3.7 % | P30b already addressed dense + HCA |
| 3 | V4 Triton CSA pool FWD (`_v4_csa_attention_pool_fwd_kernel`, 3 launches × 41.2 ms) | 123.51 ms | 2.9 % | Improved vs P30b gathered FWD; not the main limiter |
| 4 | small-op kernel-launch tail (CPU-bound floor) | 43.70 ms | 1.0 % | De-scoped while GPU is 99 % active |
| 5 | comm time (DeepEP + c10d) | 11.58 ms | 0.3 % | De-scoped for single-node EP8 |

### P31 outcome

P31's in-kernel gather/scatter is a positive partial win:

- Wrapper-side `gathered = torch.gather(pool, topk_idxs)` is no longer on the Triton CSA path; `DeepseekV4Attention._csa_forward` passes `pool + topk_idxs` directly.
- Backward now emits `dpool` through `_v4_csa_attention_pool_bwd_kernel`, avoiding the materialised `dgathered` tensor and the separate gather autograd scatter.
- The first `BLOCK_K=64` tuning experiment was slower in smoke (~155 TFLOP/s/GPU vs ~158), so the kernel stays at `BLOCK_K=32`.

The remaining bottleneck is still CSA BWD itself. The next optimisation should target atomics / bandwidth in `_v4_csa_attention_pool_bwd_kernel`, not wrapper-side gather overhead.
