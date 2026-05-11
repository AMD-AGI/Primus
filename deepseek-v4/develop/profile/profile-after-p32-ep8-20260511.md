# Plan-5 P32 — V4-Flash EP=8 trace after split CSA FWD + atomic-free attention kernels

> Generated 2026-05-11 from `output/amd/tas-mi355x-20260511/p32_profile_split_kernels_pp1_ep8_seq4096/tensorboard/primus-megatron-exp[p32_profile_split_kernels_pp1_ep8_seq4096]-rank[0].1778476971738245137.pt.trace.json`.

## Key findings (TL;DR)

- **End-to-end EP8 proxy speed-up holds.** Clean post-profiler iter 10 reports **890.5 ms / 768.4 TFLOP/s/GPU**, an **8.3 %** speed-up vs the P31b iter 10 baseline `964.8 ms / 709.3 TFLOP/s/GPU`. Running average from iter 3 is **1037.1 ms / 703.8 TFLOP/s/GPU** vs P31b `1114.5 ms / 648.0 TFLOP/s/GPU`.
- **The CSA FWD split is the dominant proxy win.** Trace-level CSA FWD work drops from P31b **123.07 ms** (single `_v4_csa_attention_pool_fwd_kernel`) to P32 **~50 ms** (split across 3 launches of `_v4_attention_fwd_kernel` for local SWA + 3 launches of `_v4_csa_attention_pool_sparse_fwd_kernel` for sparse top-K + LSE merge). Standalone microbenchmark CSA FWD: **48.17 ms → 3.16 ms** (**15.2 ×**).
- **V4 attention BWD kernels stay at the P31b monolithic time** (`_v4_attention_bwd_kernel` 256.97 ms / 8 launches ≈ **32.1 ms / launch**, matching P31b `259.74 ms / 8`). The split dQ + dKV kernels win the microbench (dense 17.27 → 7.65 ms, HCA 20.87 → 11.91 ms) but **regress end-to-end EP8** by ~190 ms / iter because they read Q / K / V twice (one read per kernel, 2× HBM traffic) and lose to the MoE HBM contention during proxy training. They ship as opt-in via ``PRIMUS_V4_ATTN_BWD_USE_SPLIT=1``.
- **CSA sparse BWD ships the gather + atomic_add path** (`_v4_csa_attention_pool_sparse_bwd_kernel` **72.54 ms / 3 launches** = 24.18 ms / launch, vs P31b **80.81 ms / 3** = 26.94 ms). The segmented-reduction path (4 GiB partial buffer + sorted inverse index) wins the microbench (**35.43 → 16.31 ms**) but regresses EP8 by ~40 ms / iter because the partial buffer's HBM traffic competes with MoE. It ships as opt-in via ``PRIMUS_V4_CSA_BWD_SEGREDUCE=1``.
- **`_v4_attention_bwd_kernel` is still the dominant residual at 28.6 % of step.** Future work: the bench-only split kernels showed that a 2.26× speed-up is unlocked when HBM is uncontested. Re-tuning the split kernels to share a single Q / K / V read (e.g. cooperative groups, persistent kernel layout) is the natural follow-up.

## Run config provenance

| key | value |
|---|---|
| commit | `e5987f9f` |
| host | `mi355-gpu-8` |
| container | `dev_primus_wenx_693` |
| trace | `1778476971738245137` |
| trace archive | `output/amd/tas-mi355x-20260511/p32_profile_split_kernels_pp1_ep8_seq4096/tensorboard/primus-megatron-exp[p32_profile_split_kernels_pp1_ep8_seq4096]-rank[0].1778476971738245137.pt.trace.json` |
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
| P32 kernel defaults | CSA FWD split = ON, V4 attn BWD split = OFF, CSA BWD segreduce = OFF |

## Per-Iter Wall Time

Sourced from the training stdout log (Megatron's `elapsed time per iteration (ms)`). First two iterations are compile / warmup.

| iter | ms / iter | running avg ms | TFLOP/s/GPU | running avg TFLOP/s/GPU | lm_loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 38047.2 | - | 18.0 | - | 1.189968E+01 |
| 2 | 18298.6 | - | 37.4 | - | 1.175117E+01 |
| 3 | 1019.7 | 1019.7 | 671.1 | 671.1 | 1.116448E+01 |
| 4 | 889.1 | 954.4 | 769.7 | 720.4 | 1.094456E+01 |
| 5 | 895.5 | 934.8 | 764.1 | 735.0 | 1.038240E+01 |
| 6 | 900.1 | 926.1 | 760.2 | 741.3 | 1.007860E+01 |
| 7 | 1926.5 | 1126.2 | 355.2 | 664.1 | 9.815610E+00 |
| 8 | 887.0 | 1086.3 | 771.5 | 682.0 | 9.447400E+00 |
| 9 | 888.4 | 1058.0 | 770.3 | 694.6 | 9.288806E+00 |
| 10 | 890.5 | 1037.1 | 768.4 | 703.8 | 9.258560E+00 |

Iter 7 carries the in-window profiler overhead (`profile_step_end=7`). Clean post-profiler iter 8/9/10 average to **888.6 ms / 770.1 TFLOP/s/GPU**.

## GPU Active / Idle (steady iter window)

Steady profiler window: **899.99 ms**. This is the `ProfilerStep#6` window from the trace, not the full 10-iteration training average.

| metric | value | % of profiler window |
| --- | ---: | ---: |
| GPU active (union over HIP streams) | 859.54 ms | 95.5 % |
| GPU idle / host-profiler gap | 40.45 ms | 4.5 % |
| Σ kernel durations | 859.54 ms | 95.5 % |
| multi-stream overlap factor | 1.00× | (Σ kernel dur ÷ GPU active) |

## Attention Kernel Attribution (steady iter)

| component | launches | total | avg / launch | note |
| --- | ---: | ---: | ---: | --- |
| `_v4_attention_bwd_kernel` | 8 | 256.97 ms | 32.12 ms | V4 dense + HCA + CSA-local BWD (monolithic, shared default for all three) |
| `_v4_csa_attention_pool_sparse_bwd_kernel` | 3 | 72.54 ms | 24.18 ms | CSA sparse pool BWD (gather + atomic_add dpool) |
| `_v4_attention_fwd_kernel` | 8 | 46.96 ms | 5.87 ms | Dense + HCA + CSA-local FWD (5 V4-attention launches + 3 CSA-local launches) |
| `_v4_csa_attention_pool_sparse_fwd_kernel` | 3 | 33.59 ms | 11.20 ms | CSA sparse pool FWD (new split path) |
| **Total V4 attention family** | 22 | **410.06 ms** | — | **45.6 % of profiler window** |

## P31b → P32 Comparison

| metric | P31b (trace `1778324637328675032`) | P32 (trace `1778476971738245137`) | delta |
| --- | ---: | ---: | ---: |
| Proxy headline iter 10 | 964.8 ms | **890.5 ms** | **−7.7 %** |
| Proxy headline TFLOP/s/GPU | 709.3 | **768.4** | **+8.3 %** |
| Profiler steady window | 980.9 ms | **899.99 ms** | **−8.2 %** |
| Profiler GPU active | 940.04 ms | **859.54 ms** | **−8.6 %** |
| `_v4_attention_bwd_kernel` | 259.74 ms / 8 | 256.97 ms / 8 | −1.1 % (monolithic kept) |
| `_v4_csa_attention_pool_fwd_kernel` (monolithic) | 123.07 ms / 3 | 0 ms (replaced) | −100 % |
| `_v4_attention_fwd_kernel` | 30.04 ms / 5 | 46.96 ms / 8 | +56 % (now also includes 3 CSA-local FWDs) |
| `_v4_csa_attention_pool_sparse_fwd_kernel` | n/a | 33.59 ms / 3 | new (sparse FWD half of split) |
| `_v4_csa_attention_pool_sparse_bwd_kernel` | 80.81 ms / 3 | 72.54 ms / 3 | −10.2 % |
| **Net attention kernel time** | ~493 ms | **~410 ms** | **−16.8 %** |

## Top-30 Kernels By Total Time

Total kernel time in profiler window: **859.54 ms** across **10659** launches.

| rank | kernel | count | total | self avg | % window |
|---:| --- | ---: | ---: | ---: | ---: |
| 1 | `_v4_attention_bwd_kernel` | 8 | 256.97 ms | 32.12 ms | 28.6 % |
| 2 | `_v4_csa_attention_pool_sparse_bwd_kernel` | 3 | 72.54 ms | 24.18 ms | 8.1 % |
| 3 | `void at::native::elementwise_kernel_manual_unroll<128, 8, ...>` | 643 | 59.21 ms | 92.1 µs | 6.6 % |
| 4 | `_v4_attention_fwd_kernel` | 8 | 46.96 ms | 5.87 ms | 5.2 % |
| 5 | `void multi_tensor_apply_kernel<TensorListMetadata<5, false>, ...>` | 321 | 46.45 ms | 144.7 µs | 5.2 % |
| 6 | `_v4_csa_attention_pool_sparse_fwd_kernel` | 3 | 33.59 ms | 11.20 ms | 3.7 % |
| 7 | `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 9 | 30.19 ms | 3.35 ms | 3.4 % |
| 8 | `void at::native::vectorized_elementwise_kernel<4, bfloat16_copy, ...>` | 1390 | 23.09 ms | 16.6 µs | 2.6 % |
| 9 | `void at::native::vectorized_elementwise_kernel<4, bfloat16tofloat32_copy, ...>` | 1191 | 21.88 ms | 18.4 µs | 2.4 % |
| 10 | `void primus_turbo::deep_ep::intranode::cached_notify_combine<8>(...)` | 16 | 14.94 ms | 933.7 µs | 1.7 % |
| 11 | `void ck_tile::kentry<1, GroupedGemmKernel<...>>` | 16 | 14.22 ms | 888.7 µs | 1.6 % |
| 12 | `void at::native::vectorized_elementwise_kernel<8, CUDAFunctor_add<...>>` | 883 | 13.26 ms | 15.0 µs | 1.5 % |
| 13 | `void ck_tile::kentry<1, GroupedGemmKernel<...>>` | 16 | 12.83 ms | 802.1 µs | 1.4 % |
| 14 | `void multi_tensor_apply_kernel<TensorListMetadata<2, false>, ...>` | 321 | 11.14 ms | 34.7 µs | 1.2 % |
| 15 | `void at::native::CatArrayBatchedCopy_contig<...>` | 24 | 9.94 ms | 414.0 µs | 1.1 % |

## Kernel Launch Cadence

Total kernels launched in the profiler window: **10659**.
Median inter-launch interval: **12.4 µs** (p50), **89.0 µs** (p90), **593.6 µs** (p99).

| inter-launch (µs) | count | % |
| --- | ---: | ---: |
| 0–10 | 4738 | 44.5 % |
| 10–50 | 4094 | 38.4 % |
| 50–100 | 840 | 7.9 % |
| 100–500 | 840 | 7.9 % |
| 500–5000 | 120 | 1.1 % |
| ≥5000 | 26 | 0.2 % |

## Comm Time (steady iter)

| kind | total | % iter |
| --- | ---: | ---: |
| deepep | 0.0 µs | 0.0 % |
| nccl/c10d | 23.96 ms | 2.7 % |
| **total comm** | **23.96 ms** | **2.7 %** |

## Bottleneck Readout

- **CSA FWD bottleneck is gone.** P31b's monolithic CSA FWD (123.07 ms / 3 launches) drops to ~50 ms after the split, freeing ~73 ms / iter on its own.
- **`_v4_attention_bwd_kernel` is now the largest named bucket at 28.6 %.** P32 keeps it monolithic by default — the split dQ + dKV kernels are 2.26× faster as a unit but lose ~190 ms / iter in EP8 because of 2× HBM read traffic. The next attention work needs to recover the split kernels' microbench wins without doubling HBM read; cooperative-group sharing of Q / K / V reads or a persistent-kernel design are the natural directions.
- **CSA sparse BWD is now 24.18 ms / launch.** Segmented-reduction is available via env var for kernel-tuning runs, but the 4 GiB partial buffer is the wrong trade-off in EP8.
- **CPU floor (4.5 %) and comm (2.7 %)** stay small. The next bottlenecks after the V4 attention family are again the small-op-tail `elementwise` and `multi_tensor_apply` kernels, both of which are scattered across the optimiser step.

## How To Reproduce

Repro script: `progress/p32/run_baseline_trace_ep8_p32.sh` (same iter 6→7 profiler window as P31b).

Operator microbenchmarks at the EP8 CSA / V4-attention shape:

- `progress/p31/bench_csa_attention_ep8.py --warmup 20 --iters 60 --json-out progress/p32/p32_csa_shipped.json`
- `progress/p32/bench_v4_attention_ep8.py --mode dense --warmup 20 --iters 60 --json-out progress/p32/p32_attn_dense_shipped.json`
- `progress/p32/bench_v4_attention_ep8.py --mode hca   --warmup 20 --iters 60 --json-out progress/p32/p32_attn_hca_shipped.json`

To exercise the opt-in microbench-optimised kernels:

- `PRIMUS_V4_CSA_BWD_SEGREDUCE=1` — atomic-free CSA BWD via segmented reduction (microbench 35.43 → 16.31 ms; proxy regression).
- `PRIMUS_V4_ATTN_BWD_USE_SPLIT=1` — atomic-free V4 attention BWD via split dQ + dKV (microbench 17.27 → 7.65 ms dense; 20.87 → 11.91 ms HCA; proxy regression).
- `PRIMUS_V4_CSA_FWD_FORCE_MONOLITHIC=1` — fall back to the P31b single CSA FWD kernel (microbench regression; proxy parity-or-worse).
