# Plan-6 P41 — Trace-driven elementwise / reduce fusion candidates

> Sourced from the post-P40 chrome trace
> `output/amd/tas-mi355x-20260514/p40_profile_plan6_close_pp1_ep8_seq4096/
> tensorboard/primus-megatron-exp[p40_profile_plan6_close_pp1_ep8_seq4096]-rank[0].1778800838095839437.pt.trace.json`
> (compressed copy at the `.tgz` of the same base name).
> Trace generated 2026-05-14, analysed 2026-05-15.

## 1. Trace context

* Profiler window: **523.67 ms** (`ProfilerStep#6`)
* GPU active (union over HIP streams): 498.12 ms (95.1 %)
* GPU idle / host-profiler gap: 25.55 ms (4.9 %)
* Σ kernel durations: 645.49 ms across 7176 launches
* Multi-stream overlap factor: 1.30× (DeepEP comm stream)

## 2. Top-30 kernels (P40 production defaults)

The Indexer-tail target sits inside the eager elementwise residual
of the V4 attention family (`compress_ratio == 4` layers, 3 calls
per iter).  The residual buckets in the top-30 break down as
follows (already-fused plan-6 kernels are omitted — see
§3 for the un-fused targets):

| rank | bucket | total / iter | launches | avg / launch | category |
| ---: | --- | ---: | ---: | ---: | --- |
|  1 | `vec_elem<add_bf16>` (Adam ε-add)               | 170.99 ms |  743 |    230 µs | optimizer |
|  2 | `multi_tensor<adam_master>`                     |  45.92 ms |  321 |    143 µs | optimizer |
|  3 | `_v4_attention_bwd_dkv_kernel`                  |  36.84 ms |    8 |   4605 µs | attention (plan-5) |
|  4 | `ncclDevKernel_Generic_1` (allreduce)           |  31.42 ms |    9 |   3491 µs | comm |
|  5 | `_v4_attention_bwd_dq_kernel`                   |  27.93 ms |    8 |   3491 µs | attention (plan-5) |
|  6 | `vec_elem<bf16_copy>` (.contiguous after permute) |  24.63 ms | 1303 |     19 µs | **model — fusable** |
|  7 | `_v4_csa_attention_pool_sparse_bwd_partial`     |  22.00 ms |    3 |   7333 µs | attention (plan-5) |
|  8 | `vec_elem<bf16->fp32>` (pre-reduce promo)       |  20.99 ms | 1215 |     17 µs | mixed — see §3 |
|  9 | `deep_ep::combine`                              |  15.82 ms |   16 |    989 µs | comm |
| 10 | `ck_tile::GroupedGemm` (MoE fc1 RowxCol)        |  14.65 ms |   16 |    916 µs | GEMM |
| 11 | `ck_tile::GroupedGemm` (MoE fc1 ColxRow)        |  12.89 ms |   16 |    806 µs | GEMM |
| 12 | `multi_tensor<scale>` (grad-scaling)            |  10.96 ms |  321 |     34 µs | optimizer |
| 13 | `ck_tile::GroupedGemm` (MoE fc2 RowxCol)        |   9.75 ms |   16 |    609 µs | GEMM |
| 14 | `vec_elem<mul_fp32>` (fp32 broadcast)           |   9.59 ms |  109 |     88 µs | **model — fusable** |
| 15 | `hipBLASLt_gemm<Ailk>` (FC-out / out-proj)      |   9.10 ms |   47 |    194 µs | GEMM |
| 16 | `hipBLASLt_gemm<Ailk>` (Q / KV down-proj)       |   8.96 ms |   33 |    272 µs | GEMM |
| 17 | `elem_unroll<mul_bf16>` (unroll bf16 mul)       |   8.31 ms |  122 |     68 µs | **model — fusable** |
| 18 | `deep_ep::combine` (variant 2)                  |   8.30 ms |   16 |    519 µs | comm |
| 19 | `deep_ep::dispatch`                             |   8.16 ms |   16 |    510 µs | comm |
| 20 | `reduce<l2norm_bf16>` (grad-norm clip)          |   7.76 ms |   12 |    647 µs | optimizer |
| 21 | `hipBLASLt_gemm<Alik>` (transpose-A variant)    |   7.39 ms |   33 |    224 µs | GEMM |
| 22 | `_v4_csa_attention_pool_sparse_fwd_kernel`      |   7.16 ms |    3 |   2386 µs | attention (plan-5) |
| 23 | `multi_tensor<l2norm>`                          |   6.72 ms |  321 |     21 µs | optimizer |
| 24 | `elem_unroll<mul_fp32>`                         |   6.51 ms |  121 |     54 µs | **model — fusable** |
| 25 | `_v4_attention_fwd_kernel`                      |   5.95 ms |    8 |    744 µs | attention (plan-5) |
| 26 | `elem_unroll<copy>` (bf16 direct copy)          |   5.78 ms |   47 |    123 µs | **model — fusable** |
| 27 | `vec_elem<mul_bf16>` (AUnary scalar broadcast)  |   5.71 ms |   12 |    476 µs | **model — fusable** |
| 28 | `vec_elem<add_fp32>` (fp32 add)                 |   5.20 ms |   36 |    144 µs | **model — fusable** |
| 29 | `_stack_grouped_weight_fwd_kernel`              |   4.99 ms |   16 |    312 µs | plan-6 P34 |
| 30 | `hipBLASLt_gemm<Ailk>` (Indexer einsum)         |   4.88 ms |   16 |    305 µs | GEMM (eager — Indexer keeps it that way per P41) |

## 3. Categorised fusion candidates

### 3.1 In-model elementwise residual (~30 ms / iter total)

| # | bucket | per-iter | upstream source | proposed phase | est. savings | confidence |
| ---: | --- | ---: | --- | --- | ---: | --- |
| 1 | `vec_elem<bf16_copy>` (1303×) + `elem_unroll<copy>` (47×) | ~30 ms | `permute(0, 3, 1, 2, 4).contiguous()` materialising `gathered_k_v` for CSA sparse top-K branch; V4 attention output projection contiguous; misc `.contiguous()` after MoE token combine | **P42** (fold into V4 attention / CSA FWD inputs) | ~25 ms | **high** (1303 launches concentrate in 4-6 call sites; well-defined source) |
| 2 | `vec_elem<mul_fp32>` (109×) + `elem_unroll<mul_bf16>` (122×) + `elem_unroll<mul_fp32>` (121×) + `vec_elem<add_fp32>` (36×) | ~30 ms | (a) Eager V4 router post-logits path (P39 descope leftovers — `softmax * scale -> denom -> scatter`) ~15 ms; (b) `Compressor.forward` APE elementwise (per-position-encoding) chain ~5 ms; (c) Q/KV down-projection `* scale` + `+ bias` chain ~5 ms; (d) misc post-RMSNorm broadcast ~5 ms | **P43** (re-attempt P39 with 50-iter A/B + Compressor APE fusion) | ~10 ms | medium (P39 microbench wins but proxy-noise-bound — needs longer A/B) |
| 3 | `vec_elem<mul_bf16>` (AUnary, 12×) | ~5.7 ms | V4 attention output projection `out * scale` per-head broadcast; only 12 launches but 0.48 ms each — high per-launch cost suggests broadcast over `[B, H, S, head_dim]` not yet inside the V4 attention FWD epilogue | **P44** (fold into V4 attention FWD epilogue) | ~3-5 ms | medium (need to confirm source via External-id correlation) |
| 4 | `vec_elem<bf16->fp32>` (1215×, 20.99 ms total) — model portion only | ~10 ms (rough) | bf16 -> fp32 promotion before fp32 elementwise inside HC `compute_weights` Sinkhorn boundary + Indexer `_causal_mask` cast + router post-logits cast.  Most launches absorbed by plan-6 P36/P37/P38; the remainder is concentrated in the descoped P39 router path. | folded into P43 | ~3-5 ms | low (intrinsic to dtype contract; hard to remove without changing math) |

**Total in-model fusable: ~40-50 ms / iter** — would push iter
time from 510.6 ms toward **~460-470 ms**.

### 3.2 Optimizer-step residual (~242 ms / iter total — out of model)

| # | bucket | per-iter | proposed plan | est. savings | confidence |
| ---: | --- | ---: | --- | ---: | --- |
| 1 | `vec_elem<add_bf16>` (Adam ε-add) | 170.99 ms | **plan-7 P0a** | ~150 ms | high (TE/Apex `AdamFunctorMasterParamRemainder` emits a separate ε-add functor; folding it into the master Adam multi-tensor kernel is a known optimisation pattern) |
| 2 | `multi_tensor<adam_master>` | 45.92 ms | plan-7 P0b | (covered by P0a) | high |
| 3 | `multi_tensor<scale>` (grad-scaling pre-allreduce) | 10.96 ms | plan-7 P0c | ~5 ms | medium |
| 4 | `reduce<l2norm_bf16>` + `multi_tensor<l2norm>` (grad-norm clip) | 7.76 + 6.72 = 14.48 ms | plan-7 P0d | ~10 ms | high |

**Total optimizer-step fusable: ~165 ms / iter** — would push
iter time from 510.6 ms toward **~345 ms** (within striking
distance of the plan-6 roadmap target of 310 ms).

### 3.3 Out-of-scope buckets

* **V4 attention BWD** (`_v4_attention_bwd_dkv` + `_v4_attention_bwd_dq` = 64.77 ms / 16 launches): Already at the plan-5 P32 split-kernel optimum.  The shared-Q/K/V-load follow-up
  (cooperative groups / persistent kernel layout) is plan-7
  attention scope, not plan-6 elementwise.
* **DeepEP comm** (`combine` + `dispatch` + `cached_notify_combine` = 47.94 ms / 48 launches): Already overlapped via comm stream where possible; the residual is critical-path.
* **MoE GEMM** (`ck_tile::GroupedGemm` 4 variants = 46.45 ms / 64 launches): At cuBLAS / hipBLASLt peak; not a Triton fusion target.

## 4. Proposed phase sequencing (plan-7 starter set)

| phase | scope | target | est. iter-time delta | risk |
| --- | --- | --- | ---: | --- |
| **P41** (this phase) | Indexer post-einsum tail | bandwidth-bound `relu + mul + sum + mask` | -2 to -3 ms | low (P38 tail was the only bandwidth-bound half) |
| **P42** | V4 attention / CSA FWD input gather absorbs `.contiguous()` + permute | `bf16_copy` + `elem_unroll<copy>` | -20 to -25 ms | medium (modifies plan-5 V4 attention kernels) |
| **P43** | V4 router post-logits + Compressor APE elementwise | `vec_elem<mul_fp32>` + `elem_unroll<mul_bf16>` + `vec_elem<add_fp32>` | -5 to -10 ms | medium (P39 already descoped — needs better A/B methodology) |
| **P44** | V4 attention FWD epilogue (`out * scale + sinks`) | `vec_elem<mul_bf16>` AUnary | -3 to -5 ms | low (small change to V4 attention FWD kernel) |
| **plan-7 P0** | Custom Triton fused Adam + grad-clip + scale kernel | `vec_elem<add_bf16>` + `multi_tensor<adam_master>` + `reduce<l2norm_bf16>` + `multi_tensor<scale>` + `multi_tensor<l2norm>` | -150 to -170 ms | high (TE / Apex coordination; out-of-model) |

The plan-7 starter set targets the 310-ms-per-iter goal pinned in
`plan-6/01-roadmap.md` (was missed at P40 final 510.6 ms because
plan-6 only covered in-model elementwise residual).

## 5. Methodology notes

* **Trace inventory** ran `develop/profile/_tools/analyze_p40_trace.py`
  with `--steady-step 6`; the top-30 list is the JSON `top_30`
  field, decompressed by `print_top30.py` (helper, not committed).
* **Per-call-site attribution** for the elementwise buckets (which
  Python source line each launch comes from) is **NOT** in this
  inventory — that requires the forensic `External-id` correlation
  pass from R9.3.  P42 / P43 task-list refinement runs the forensic
  attribution before designing the fusion.
* **EP8 proxy A/B noise floor** is ±1-3 ms / iter (from the P39
  descope analysis); any phase whose microbench prediction is below
  this threshold must ship with `--train_iters 50+` and aggregate
  across at least 3 proxy runs to defeat noise.
