# Plan-5 P31b — V4-Flash EP=8 trace after split CSA backward

> Generated 2026-05-09T06:24:50 from `output/amd/tas-mi355x-20260509/p31_profile_csa_in_kernel_gather_pp1_ep8_seq4096/tensorboard/primus-megatron-exp[p31_profile_csa_in_kernel_gather_pp1_ep8_seq4096]-rank[0].1778324637328675032.pt.trace.json`.

## Key Findings

- **P31b is the updated profile for the split CSA BWD redesign.** The previous `profile-after-p31-ep8-20260509` report still describes the first P31 in-kernel gather/scatter path; this report uses trace `1778324637328675032`.
- **End-to-end proxy speed is now in the sub-second range.** The clean post-profiler iter 10 line reports **964.8 ms / iter** and **709.3 TFLOP/s/GPU**; the running average is **1114.5 ms** and **648.0 TFLOP/s/GPU**.
- **The profiler steady window is 980.9 ms** with **940.0 ms GPU active** (95.8 %). The trace still has a small profiler / host gap, so `proxy_ep8.md` uses the clean iter 10 line as the headline.
- **The monolithic CSA BWD bottleneck is gone.** P31 had `_v4_csa_attention_pool_bwd_kernel` at **3.50 s / 3 launches**. P31b replaces it with `_v4_csa_attention_pool_sparse_bwd_kernel` at **80.8 ms / 3 launches** plus CSA-local work folded into `_v4_attention_bwd_kernel`.
- **Trace-estimated CSA BWD is about 179.6 ms**, using P30/P31 dense-HCA `_v4_attention_bwd_kernel` cost (~161.0 ms) as the baseline and attributing the extra dense-kernel launches to CSA local BWD. This is a **94.9 %** reduction from the P31 monolithic CSA BWD trace. The standalone EP8-shape benchmark remains lower at **35.43 ms** because it excludes full-training profiler overhead and unrelated attention launches.
- **Current top kernels have shifted.** `_v4_attention_bwd_kernel` is now rank 1 at **259.7 ms / 8 launches**, followed by CSA FWD at **123.1 ms** and CSA sparse BWD at **80.8 ms**. The next performance target is no longer the old per-row CSA BWD design.

## Run Config Provenance

| key | value |
| --- | --- |
| commit | `2c7cf59d` |
| host | `mi355-gpu-14` |
| container | `dev_primus_wenx_693` |
| trace | `1778324637328675032` |
| trace archive | `output/amd/tas-mi355x-20260509/p31_profile_csa_in_kernel_gather_pp1_ep8_seq4096/tensorboard/primus-megatron-exp[p31_profile_csa_in_kernel_gather_pp1_ep8_seq4096]-rank[0].1778324637328675032.pt.trace.json.tgz` |
| seq_length | 4096 |
| parallel | TP=1 PP=1 EP=8 |
| micro_batch_size / global_batch_size | 1 / 8 |
| num_layers | 8 |
| num_experts | 256 (32/rank under EP=8) |
| moe_router_topk | 6 |
| moe_ffn_hidden_size | 2048 |
| index_topk | 512 |
| compress_ratios | `[0,0,4,128,4,128,4,0]` |
| perf knobs | `use_v4_triton_attention=True`, `use_v4_triton_csa_attention=True`, `use_turbo_deepep=True`, `use_turbo_grouped_mlp=True`, `use_turbo_attention=False` |

## Per-Iter Wall Time

Training values are parsed from the rank-7 pre-trainer log for the latest run in this profile directory. The first two iterations include compile / warmup and are not used for the table headline.

| iter | ms / iter | running avg ms | TFLOP/s/GPU | running avg TFLOP/s/GPU | lm_loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 38053.3 | - | 18.0 | - | 1.189969E+01 |
| 2 | 19604.7 | - | 34.9 | - | 1.175119E+01 |
| 3 | 1143.6 | 1143.6 | 598.4 | 598.4 | 1.116454E+01 |
| 4 | 968.9 | 1056.2 | 706.3 | 652.3 | 1.094465E+01 |
| 5 | 972.5 | 1028.3 | 703.6 | 669.4 | 1.038256E+01 |
| 6 | 979.1 | 1016.0 | 698.9 | 676.8 | 1.007907E+01 |
| 7 | 1956.6 | 1204.1 | 349.7 | 611.4 | 9.816036E+00 |
| 8 | 965.0 | 1164.3 | 709.1 | 627.7 | 9.447948E+00 |
| 9 | 965.6 | 1135.9 | 708.7 | 639.2 | 9.289451E+00 |
| 10 | 964.8 | 1114.5 | 709.3 | 648.0 | 9.259426E+00 |

## GPU Active / Idle

Steady profiler window: **980.9 ms**. This is the `ProfilerStep#6` window from the trace, not the full 10-iteration training average.

| metric | value | % of profiler window |
| --- | ---: | ---: |
| GPU active (union over HIP streams) | 940.04 ms | 95.8 % |
| GPU idle / host-profiler gap | 40.87 ms | 4.2 % |
| Sum of kernel durations | 940.04 ms | 95.8 % |
| multi-stream overlap factor | 1.00x | kernel sum / GPU active |

## CSA Backward Split Attribution

| component | launches | total | avg / launch | note |
| --- | ---: | ---: |---:|---|
| `_v4_attention_bwd_kernel` | 8 | 259.74 ms | 32.47 ms | Dense/HCA BWD plus CSA-local BWD in the split path |
| Estimated CSA-local part of `_v4_attention_bwd_kernel` | 3 | 98.75 ms | 32.92 ms | Approximation: subtract P31 dense/HCA total 160.99 ms |
| `_v4_csa_attention_pool_sparse_bwd_kernel` | 3 | 80.81 ms | 26.94 ms | New sparse pool BWD kernel |
| **Estimated CSA BWD total in profile** | 3 | **179.56 ms** | **59.85 ms** | CSA-local estimate + sparse pool BWD |

Standalone operator benchmark for the same EP8 CSA shape reports **35.43 ms** BWD total (`16.48 ms` local dense + `17.83 ms` sparse), which is the kernel-only target check used during P31b development.

## Top-30 Kernels By Total Time

Total kernel time in profiler window: **940.04 ms** across **10647** launches.

| rank | kernel | count | total | self avg | % window |
|---:| --- | ---: | ---: |---:|---:|
| 1 | `_v4_attention_bwd_kernel` | 8 | 259.74 ms | 32.47 ms | 26.5 % |
| 2 | `_v4_csa_attention_pool_fwd_kernel` | 3 | 123.07 ms | 41.02 ms | 12.5 % |
| 3 | `_v4_csa_attention_pool_sparse_bwd_kernel` | 3 | 80.81 ms | 26.94 ms | 8.2 % |
| 4 | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_noca...` | 643 | 58.96 ms | 91.7 us | 6.0 % |
| 5 | `void multi_tensor_apply_kernel<TensorListMetadata<5, false>, transformer_engine::multi_ten...` | 321 | 46.60 ms | 145.2 us | 4.8 % |
| 6 | `_v4_attention_fwd_kernel` | 5 | 30.04 ms | 6.01 ms | 3.1 % |
| 7 | `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 9 | 29.96 ms | 3.33 ms | 3.1 % |
| 8 | `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at...` | 1390 | 23.31 ms | 16.8 us | 2.4 % |
| 9 | `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kerne...` | 1191 | 21.82 ms | 18.3 us | 2.2 % |
| 10 | `void primus_turbo::deep_ep::intranode::cached_notify_combine<8>(void**, int*, int, int, in...` | 16 | 14.23 ms | 889.2 us | 1.5 % |
| 11 | `void ck_tile::kentry<1, ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitio...` | 16 | 14.11 ms | 881.9 us | 1.4 % |
| 12 | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1...` | 883 | 13.23 ms | 15.0 us | 1.3 % |
| 13 | `void ck_tile::kentry<1, ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitio...` | 16 | 12.60 ms | 787.4 us | 1.3 % |
| 14 | `void multi_tensor_apply_kernel<TensorListMetadata<2, false>, transformer_engine::multi_ten...` | 321 | 11.08 ms | 34.5 us | 1.1 % |
| 15 | `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous ...` | 24 | 10.09 ms | 420.6 us | 1.0 % |
| 16 | `void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, float, ...` | 138 | 9.77 ms | 70.8 us | 1.0 % |
| 17 | `void ck_tile::kentry<1, ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitio...` | 16 | 9.55 ms | 596.8 us | 1.0 % |
| 18 | `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDSB0_AFC1_AFEM...` | 47 | 9.35 ms | 198.9 us | 1.0 % |
| 19 | `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDSB0_AFC1_AFEM...` | 33 | 9.26 ms | 280.5 us | 0.9 % |
| 20 | `void primus_turbo::deep_ep::intranode::dispatch<8, 1024, true>(HIP_vector_type<int, 4u>*, ...` | 16 | 8.49 ms | 530.3 us | 0.9 % |
| 21 | `void primus_turbo::deep_ep::intranode::combine<hip_bfloat16, 8, 1024, true>(hip_bfloat16*,...` | 16 | 8.24 ms | 515.2 us | 0.8 % |
| 22 | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_noca...` | 122 | 8.14 ms | 66.7 us | 0.8 % |
| 23 | `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<c10::BFloat16, at::native::Nor...` | 12 | 7.76 ms | 647.0 us | 0.8 % |
| 24 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDSB0_AFC1_AFEM...` | 33 | 7.37 ms | 223.3 us | 0.8 % |
| 25 | `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_noca...` | 192 | 7.22 ms | 37.6 us | 0.7 % |
| 26 | `void multi_tensor_apply_kernel<TensorListMetadata<1, false>, transformer_engine::multi_ten...` | 321 | 6.75 ms | 21.0 us | 0.7 % |
| 27 | `void at::native::vectorized_elementwise_kernel<8, at::native::AUnaryFunctor<c10::BFloat16,...` | 12 | 5.77 ms | 480.8 us | 0.6 % |
| 28 | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std:...` | 115 | 4.92 ms | 42.8 us | 0.5 % |
| 29 | `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x16x32_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFE...` | 16 | 4.79 ms | 299.2 us | 0.5 % |
| 30 | `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, ...` | 243 | 4.22 ms | 17.4 us | 0.4 % |

## Kernel Launch Cadence

Total kernels launched in the profiler window: **10647**.
Median inter-launch interval: **12.8 us** (p50), **88.2 us** (p90), **603.8 us** (p99).

## Module-Level CPU Op-Time Attribution

As in the earlier reports, PyTorch profiler CPU events are nested, so the sums below are attribution signals rather than exclusive CPU time.

| module pattern | events | sum event dur | % profiler window |
| --- | ---: | ---: |---:|
| other | 216482 | 94.52 s | 9635.5 % |
| DeepseekV4HybridLayer | 8 | 264.94 ms | 27.0 % |
| DeepseekV4MoE | 8 | 206.74 ms | 21.1 % |
| Optimizer | 2060 | 146.64 ms | 14.9 % |
| linear/matmul | 1307 | 66.55 ms | 6.8 % |
| DeepseekV4Attention | 8 | 22.61 ms | 2.3 % |
| v4_attention (Triton) | 55 | 16.78 ms | 1.7 % |
| LayerNorm/RMSNorm | 247 | 11.03 ms | 1.1 % |
| c10d/comm | 91 | 6.81 ms | 0.7 % |
| DualRoPE/RoPE | 21 | 2.99 ms | 0.3 % |
| softmax | 143 | 2.54 ms | 0.3 % |

## Comm Time

| kind | total | % profiler window |
| --- | ---: | ---: |
| deepep | 0.0 us | 0.0 % |
| nccl/c10d | 11.12 ms | 1.1 % |
| **total comm** | **11.12 ms** | **1.1 %** |

## P31 To P31b Comparison

| metric | P31 | P31b | delta |
| --- | ---: | ---: |---:|
| Proxy headline TFLOP/s/GPU | 158.5 | 709.3 | 4.48x |
| Proxy headline iter time | 4317.0 ms | 964.8 ms | -77.7 % |
| CSA BWD profile cost | 3.50 s | ~179.6 ms | -94.9 % |
| CSA sparse BWD kernel | n/a | 80.8 ms / 3 | new split kernel |
| CSA FWD kernel | 123.5 ms / 3 | 123.1 ms / 3 | stable |

## Bottleneck Readout

- The original P31 CSA BWD bottleneck is solved for the current EP8 proxy shape.
- `_v4_attention_bwd_kernel` is now the largest named kernel bucket because it includes dense, HCA, and CSA-local BWD work after the split. Further work should separate these launch classes in profiler annotations before tuning again.
- CSA FWD remains visible at ~123 ms / step. It is now larger than CSA sparse BWD in the profile, so future cr=4 work should consider FWD pool gather locality and K-tile scheduling after validating correctness coverage.
- Communication remains small in this single-node EP8 setup; DeepEP and c10d are not the next bottleneck.
