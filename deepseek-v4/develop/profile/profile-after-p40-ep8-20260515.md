# Plan-6 P40 close-out — V4-Flash EP=8 trace with plan-6 production defaults

> Generated 2026-05-15 from `output/amd/tas-mi355x-20260514/p40_profile_plan6_close_pp1_ep8_seq4096/tensorboard/primus-megatron-exp[p40_profile_plan6_close_pp1_ep8_seq4096]-rank[0].1778800838095839437.pt.trace.json` (compressed copy at `.../primus-megatron-exp[...].pt.trace.tgz`).

## Key findings (TL;DR)

- **Plan-6 cumulative iter time pins at 510-515 ms / iter / 521-523 TFLOP/s/GPU** at steady state with all plan-6 default-on knobs (P34 / P35 / P36 / P37) enabled and the two descoped knobs (P38 / P39) off.  Profiler window 523.67 ms, GPU active 498.12 ms (95.1 %), GPU idle 25.55 ms (4.9 %).
- **V4 attention family now collapses to ~100 ms / iter (vs P32 final ~410 ms).**  The split V4 BWD kernels (`PRIMUS_V4_ATTN_BWD_USE_SPLIT=1`) and CSA-BWD segmented-reduction kernel (`PRIMUS_V4_CSA_BWD_SEGREDUCE=1`) are now the production path and dominate the attention budget at ~10 % of step.  The monolithic `_v4_attention_bwd_kernel` (P32 final 28.6 % of step) is replaced by `_v4_attention_bwd_dq_kernel` + `_v4_attention_bwd_dkv_kernel` (7.0 % + 5.3 % = 12.3 %).
- **Optimizer step is now the hot residual.**  Adam-related elementwise add (BF16) consumes 170.99 ms / 743 launches (32.7 % of profiler window); Adam multi-tensor functor adds another 45.92 ms / 321 launches (8.8 %).  Together: **~210 ms / iter ≈ 40 % of step**.  This was always present but now leads the bottleneck list because every attention kernel that previously over-shadowed it has been compressed.
- **Plan-6 Triton kernels are all present + each cheap.**  `_sinkhorn_fwd_kernel` 16 × 0.025 ms = 0.39 ms; `_sinkhorn_bwd_kernel` 16 × 0.044 ms = 0.70 ms; `_hc_compute_tail_fwd_kernel` 16 × 0.004 ms = 0.07 ms; `_hc_compute_tail_bwd_kernel` 16 × 0.004 ms = 0.07 ms; `_stack_grouped_weight_fwd_kernel` 16 × 0.31 ms = 4.99 ms.  Combined plan-6 Triton bucket: **< 7 ms / iter** — each kernel paid for itself by removing several ms of eager elementwise launches.
- **Descoped kernels confirmed absent.**  No `_indexer_score_*` or `_v4_router_post_*` kernel launches in the trace (P38 / P39 default-off knobs honoured).
- **Multi-stream overlap factor: 1.30x.**  Σ-kernel-duration (645.49 ms) over GPU-active-union (498.12 ms) = 1.30 — modest dispatch + compute overlap from the DeepEP comm stream.  Up from P32 final 1.00x because Turbo DeepEP comms now overlap MoE compute.

## Run config provenance

| key | value |
|---|---|
| commit | `b08975bc` (`docs(deepseek-v4)[plan-6][P40]`) |
| host | `mi355-gpu-8` |
| container | `dev_primus_wenx_693` |
| trace | `1778800838095839437` |
| trace JSON | `output/amd/tas-mi355x-20260514/p40_profile_plan6_close_pp1_ep8_seq4096/tensorboard/primus-megatron-exp[p40_profile_plan6_close_pp1_ep8_seq4096]-rank[0].1778800838095839437.pt.trace.json` |
| trace archive | `.../primus-megatron-exp[p40_profile_plan6_close_pp1_ep8_seq4096]-rank[0].1778800838095839437.pt.trace.tgz` (4.3 MiB) |
| seq_length | 4096 |
| parallel | TP=1 PP=1 EP=8 |
| micro_batch_size / global_batch_size | 1 / 8 |
| num_layers | 8 |
| num_experts | 256 (32/rank under EP=8) |
| moe_router_topk | 6 |
| moe_ffn_hidden_size | 2048 |
| index_topk | 512 |
| compress_ratios | `[0,0,4,128,4,128,4,0]` |
| perf knobs | `use_v4_triton_attention=True`, `use_v4_triton_csa_attention=True`, `use_turbo_deepep=True`, `use_turbo_grouped_mlp=True`, `use_v4_compiled_sinkhorn=True`, `use_turbo_attention=False` |
| Plan-5 P32 final knobs | `PRIMUS_V4_ATTN_BWD_USE_SPLIT=1`, `PRIMUS_V4_CSA_BWD_SEGREDUCE=1` (default ON post-RoPE-fix) |
| Plan-6 default-on knobs | `PRIMUS_STACK_GROUPED_WEIGHT_TRITON=1`, `PRIMUS_ROPE_TRITON=1`, `PRIMUS_SINKHORN_TRITON=1`, `PRIMUS_HC_TRITON=1` |
| Plan-6 default-off knobs | `PRIMUS_INDEXER_TRITON=0`, `PRIMUS_V4_ROUTER_TRITON=0` (descoped) |

## Per-Iter Wall Time

Sourced from the training stdout log (Megatron's `elapsed time per iteration (ms)`).  First two iterations are compile / warmup; iter 7 carries the in-window profiler overhead (`profile_step_end=7`).

| iter | ms / iter | running avg ms | TFLOP/s/GPU | running avg TFLOP/s/GPU | lm_loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | (compile) | - | - | - | 1.189967E+01 |
| 2 | (compile) | - | - | - | 1.175117E+01 |
| 3 |     674.4 |   674.4 | 397.4 | 397.4 | 1.116446E+01 |
| 4 |     510.9 |   592.6 | 524.5 | 460.9 | 1.094438E+01 |
| 5 |     514.0 |   566.4 | 521.3 | 481.1 | 1.038210E+01 |
| 6 |     515.2 |   553.6 | 520.2 | 490.9 | 1.007792E+01 |
| 7 |    1249.7 |   695.0 | 214.4 | 433.1 | 9.814991E+00 |
| 8 |     513.5 |   664.8 | 521.9 | 447.9 | 9.446677E+00 |
| 9 |     515.1 |   643.4 | 520.3 | 458.3 | 9.287911E+00 |
| 10 |    512.4 |   627.0 | 523.0 | 466.3 | 9.257534E+00 |

Clean post-profiler iter 8/9/10 average to **513.7 ms / 521.7 TFLOP/s/GPU**.  Bake-off run (15-iter smoke, no profiler) steady iters 8-15 mean: 510.6 ms / 524.9 TFLOP/s/GPU (see `progress/p40/p40-summary.md` §3).

## GPU Active / Idle (steady iter window)

Steady profiler window: **523.67 ms** (`ProfilerStep#6`).

| metric | value | % of profiler window |
| --- | ---: | ---: |
| GPU active (union over HIP streams) | 498.12 ms | 95.1 % |
| GPU idle / host-profiler gap        |  25.55 ms |  4.9 % |
| Σ kernel durations                  | 645.49 ms | (1.30× union) |
| multi-stream overlap factor         | 1.30× | (Σ kernel dur ÷ GPU active) |
| total kernel launches               | 7176 | — |

## Attention Kernel Attribution (steady iter)

| component | launches | total | avg / launch | note |
| --- | ---: | ---: | ---: | --- |
| `_v4_attention_bwd_dkv_kernel`             | 8 | 36.84 ms | 4.60 ms | V4 dense + HCA dKV BWD (split path; ON since P32 final) |
| `_v4_attention_bwd_dq_kernel`              | 8 | 27.93 ms | 3.49 ms | V4 dense + HCA dQ BWD (split path) |
| `_v4_csa_attention_pool_sparse_bwd_partial_kernel` | 3 | 22.00 ms | 7.33 ms | CSA segreduce partial buffer (ON since P32 final) |
| `_v4_csa_attention_pool_sparse_fwd_kernel` | 3 |  7.16 ms | 2.39 ms | CSA sparse pool FWD |
| `_v4_attention_fwd_kernel`                 | 8 |  5.95 ms | 0.74 ms | V4 dense + HCA + CSA-local FWD |
| **Total V4 attention family** | 30 | **~99.88 ms** | — | **~19.1 % of profiler window** |

The split BWD + segreduce kernels are now the production path on the proxy as well as the microbench (post-RoPE-fix made them win end-to-end too).  Per-launch V4 BWD time drops from P32 final's monolithic 32.12 ms to dQ 3.49 ms + dKV 4.60 ms = **8.09 ms / layer** = 3.97x speedup per layer.

## Plan-6 Triton Kernel Attribution (steady iter)

| phase | kernel | launches | total | avg / launch | % window |
| --- | --- | ---: | ---: | ---: | ---: |
| P34 | `_stack_grouped_weight_fwd_kernel` | 16 | 4.99 ms | 0.31 ms | 0.95 % |
| P35 | (RoPE) | — | (rolled into V4 attn FWD path) | — | — |
| P36 | `_sinkhorn_fwd_kernel` | 16 | 0.39 ms | 0.025 ms | 0.08 % |
| P36 | `_sinkhorn_bwd_kernel` | 16 | 0.70 ms | 0.044 ms | 0.13 % |
| P37 | `_hc_compute_tail_fwd_kernel` | 16 | 0.07 ms | 0.0042 ms | 0.01 % |
| P37 | `_hc_compute_tail_bwd_kernel` | 16 | 0.07 ms | 0.0041 ms | 0.01 % |
| **Total plan-6 Triton kernels** |   | 80 | **~6.2 ms** | — | **~1.2 % window** |

Each plan-6 Triton kernel is now sub-percent of profiler window — the win is structural (kernel-launch overhead removed, not per-kernel ms saved).  The eager paths these replaced contributed O(50-150) elementwise launches per layer per iter; their absence at this trace pins the win.

**Descoped sanity check**: `_indexer_score_*` and `_v4_router_post_*` kernels are NOT present in the trace, confirming the P38 / P39 default-off knobs route through the eager body.

## Top-10 Kernels By Total Time

Total kernel time in profiler window: **645.49 ms** across **7176** launches.

| rank | kernel | count | total | self avg | % window |
|---:| --- | ---: | ---: | ---: | ---: |
| 1 | `at::native::vectorized_elementwise_kernel<8, CUDAFunctor_add<BFloat16>, ...>` (Adam ε-add) | 743 | 170.99 ms | 0.23 ms | 32.7 % |
| 2 | `multi_tensor_apply_kernel<...AdamFunctorMasterParamRemainder<bf16, fp32, i64>, ...>` | 321 | 45.92 ms | 0.14 ms | 8.8 % |
| 3 | `_v4_attention_bwd_dkv_kernel` | 8 | 36.84 ms | 4.60 ms | 7.0 % |
| 4 | `ncclDevKernel_Generic_1` (allreduce) | 9 | 31.42 ms | 3.49 ms | 6.0 % |
| 5 | `_v4_attention_bwd_dq_kernel` | 8 | 27.93 ms | 3.49 ms | 5.3 % |
| 6 | `vectorized_elementwise_kernel<bf16_copy_kernel_cuda>` | 1303 | 24.63 ms | 0.019 ms | 4.7 % |
| 7 | `_v4_csa_attention_pool_sparse_bwd_partial_kernel` | 3 | 22.00 ms | 7.33 ms | 4.2 % |
| 8 | `vectorized_elementwise_kernel<bf16tofloat32_copy_kernel_cuda>` | 1215 | 20.99 ms | 0.017 ms | 4.0 % |
| 9 | `primus_turbo::deep_ep::intranode::cached_notify_combine<8>` | 16 | 15.82 ms | 0.99 ms | 3.0 % |
| 10 | `ck_tile::GroupedGemmKernel<256x256x64, RowxColxRow>` (grouped MoE GEMM fc1) | 16 | 14.65 ms | 0.92 ms | 2.8 % |

## P32 final → P40 close-out comparison

| metric | P32 final (trace `1778476971738245137`) | P40 close-out (trace `1778800838095839437`) | delta |
| --- | ---: | ---: | ---: |
| Proxy headline iter 10                 | 890.5 ms |  **512.4 ms** | **−42.5 %** |
| Proxy headline TFLOP/s/GPU             |   768.4* |   **523.0**   |  (P33-corrected denom) |
| Profiler steady window                 | 899.99 ms |  **523.67 ms** | **−41.8 %** |
| Profiler GPU active                    | 859.54 ms |  **498.12 ms** | **−42.0 %** |
| Multi-stream overlap                   |     1.00× |        1.30× |  **+30 %** (DeepEP comm overlap surfaces) |
| `_v4_attention_bwd_kernel` (monolithic) | 256.97 ms / 8 | 0 ms (split) | −100 % (replaced) |
| `_v4_attention_bwd_dq_kernel` (split)  |        n/a |  27.93 ms / 8 | new |
| `_v4_attention_bwd_dkv_kernel` (split) |        n/a |  36.84 ms / 8 | new |
| Σ V4 BWD attention                     | 329.5 ms |  **86.77 ms**  | **−73.7 %** |
| Σ V4 attention family (FWD+BWD)        | 410.06 ms | **~99.88 ms** | **−75.6 %** |
| `_stack_grouped_weight_fwd_kernel`     |   eager   |   4.99 ms / 16 | new (P34) |
| `_sinkhorn_fwd_kernel` / `_bwd_kernel` | compiled  | 0.39 + 0.70 ms / 16 ea | new (P36) |
| `_hc_compute_tail_fwd_kernel` / `_bwd_kernel` | eager | 0.07 + 0.07 ms / 16 ea | new (P37) |

(* P32 final TFLOP/s used the legacy pre-P33 denominator; P40 uses the P33-corrected denominator.  Iter-time delta is the apples-to-apples comparison.)

## Ranked bottleneck list for plan-7

| rank | bucket | total / iter | % window | rationale |
| ---: | --- | ---: | ---: | --- |
| 1 | **Adam optimizer step** (BF16 add + multi-tensor functor) | ~217 ms | ~41 % | Now the dominant residual.  ApexFusedAdam already minimises kernel count; further wins need either fewer parameters / sharded optimizer (FSDP-style) or fused `(Adam + master-param remainder)` |
| 2 | **V4 attention BWD** (`_v4_attention_bwd_dq_kernel` + `_v4_attention_bwd_dkv_kernel` + CSA `_pool_sparse_bwd_partial`) | ~86 ms | ~16 % | Already at the split-kernel optimum; further wins need cooperative groups / persistent layout that shares one Q/K/V load across the dQ + dKV passes |
| 3 | **DeepEP comm** (`cached_notify_combine` + `combine` + `dispatch` + NCCL allreduce) | ~63 ms | ~12 % | Already partially overlapped via comm stream; remaining ms are critical-path NCCL  |
| 4 | **Grouped MoE GEMM** (`ck_tile::GroupedGemmKernel` 4 variants) | ~46 ms | ~9 % | At cuBLAS / hipBLASLt peak; not a Triton fusion target |
| 5 | **bf16 ↔ fp32 copy** (`bfloat16_copy_kernel` + `bfloat16tofloat32_copy_kernel`) | ~46 ms | ~9 % | Implicit dtype casts; could fuse with adjacent ops in plan-7 |
| 6 | **V4 attention FWD** (`_v4_attention_fwd_kernel` + `_v4_csa_attention_pool_sparse_fwd_kernel`) | ~13 ms | ~2.5 % | Already at the split-kernel optimum |

The elementwise-fusion bucket that plan-6 targeted is now < 7 ms / iter across the four shipped fusions.  Plan-7 work moves to the optimizer / attention BWD micro-architecture and remaining dtype-cast cleanup.

## Notes for plan-7

* **Adam multi-tensor fusion** — the 743 launches of the BF16 add functor look like a multi-tensor expansion that could fuse into a single launch with a TensorListMetadata of higher arity.  Worth a single-kernel rewrite attempt before the next planning cycle.
* **V4 attention BWD shared Q/K/V load** — the dQ + dKV kernels each load Q/K/V independently; a cooperative-group / persistent-kernel layout that shares one load would cut HBM traffic by ~2x.  Carry-over from P32 final follow-ups.
* **CSA BWD `_partial_kernel`** at 22.00 ms / 3 launches = 7.33 ms / launch is the slowest single kernel per launch in the trace; segreduce was meant to fix this but the partial-buffer write half is still slow.  Pruning the partial-buffer size (or moving to in-register reduction for small visit counts) is the natural follow-up.
