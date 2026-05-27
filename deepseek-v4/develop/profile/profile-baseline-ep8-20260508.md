# Plan-5 P28 — V4-Flash EP=8 baseline trace (seq=4096)

> Generated 2026-05-08T21:27:45 from `output/amd/tas-mi355x-20260509/p28_profile_baseline_pp1_ep8_seq4096/tensorboard/primus-megatron-exp[p28_profile_baseline_pp1_ep8_seq4096]-rank[0].1778291996232776892.pt.trace.json`.

## Key findings (TL;DR)

- **GPU is essentially fully busy.** Wall-clock GPU active = 8.69 s of 8.72 s steady iter ≈ **99.7 %**. CPU-bound floor (1 − GPU active) = **0.3 %**. The pre-trace hypothesis that small-kernel-launch tail is the bottleneck **does NOT hold at V4-Flash production widths** — kernel launch overhead is not the gating factor.
- **Multi-stream overlap factor = 1.87×.** Σ kernel dur across HIP streams (16.28 s) ÷ wall-clock GPU active (8.69 s). HIP runs at least two compute streams in parallel for ≈ half the iter — confirms the chrome-trace top-N kernel `% step` numbers can sum > 100 %.
- **Top kernel by far is an `aten::sum` fp32 reduce.** `at::native::reduce_kernel<512, 1, ReduceOp<float, sum_functor<float, float, float>>>` (the single dominant template instantiation) accounts for **7.61 s (87.3 % of step, 87 % of Σ kernel dur)** across **717 launches** at avg **10.62 ms per launch** (i.e. each call is a multi-millisecond fp32 reduction over a *large* tensor — not a small-op-tail issue). All `sum_functor<*>` variants combined: 7.62 s across 1525 launches. Hypothesis: bias-gradient `sum-over-tokens` in MoE expert backward (256 experts × moe_ffn_hidden_size=2048 across the 8-microbatch GBS) and / or fp32 master-grad accumulation in `DistributedOptimizer`. P29 task-list refinement must root-cause this kernel and decide between `torch.compile` fusion vs Triton fused expert-bias-grad kernel vs FP8 master-grad.
- **V4 Triton attention kernels (BWD heavy).** dense / HCA: 3.90 s (44.7 %) — BWD ≫ FWD (~5×). CSA: 4.19 s (48.1 %) — BWD ≫ FWD (~26×). Plan-5 P30 / P31 should focus on **BWD performance** (atomic-add density, recompute-free LSE merge for HCA, in-kernel `topk_idxs` gather for CSA) rather than FWD autotune.
- **Comm time is negligible** (DeepEP + c10d ≪ 1 % of iter). Plan-5 P32 (overlap / comm-stream tuning) should be **de-scoped** unless a structural change materially raises comm cost.
- **HBM headroom is generous.** Steady peak ≈ 195 GiB / 287 GiB ≈ 68 % at Sq=4096, MBS=1, GBS=8 (recompute off). Plan-5 P31's HBM-saving in-kernel `topk_idxs` gather is no longer headroom-driven; its motivation reduces to the BWD speed-up that comes from cutting the wrapper-side gather + scatter-add.

## Run config provenance

| key | value |
|---|---|
| commit | `578496b3` |
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
| 3 | 8792.5 | 77.8 | 11.16452 |
| 4 | 8638.5 | 79.2 | 10.94444 |
| 5 | 8628.6 | 79.3 | 10.38231 |
| 6 | 8716.6 | 78.5 | 10.07856 |
| 7 | 9556.0 | 71.6 | 9.815662 |
| 8 | 8697.1 | 78.7 | 9.447337 |
| 9 | 8735.4 | 78.3 | 9.288926 |
| 10 | 8690.7 | 78.7 | 9.258618 |

Steady (iter ≥ 5): **8837.4 ms / iter**, **77.52 TFLOP/s/GPU**.

## GPU vs CPU active / idle %

Steady iter window: 8.72 s of trace time.

**`GPU active`** below is the wall-clock union of kernel intervals across all HIP / ROCm compute streams (the time when at least one kernel is in flight). The `kernel-time sum` row is the per-stream `Σ dur` that the chrome-trace top-level kernel table sums up — when streams overlap, `kernel-time sum > GPU active` (the ratio is the **multi-stream overlap factor**: > 1.0 means at least two streams ran kernels in parallel for some fraction of the iter).

| metric | value | % of iter |
|---|---:|---:|
| GPU active (union over streams) | 8.69 s | 99.7 % |
| GPU idle (1 − active) | 23.95 ms | 0.3 % |
| Σ kernel dur (across streams) | 16.28 s | 186.8 % |
| multi-stream overlap factor | **1.87×** | (Σ kernel dur ÷ GPU active) |

**CPU-bound floor (1 − GPU active)** ≈ **0.3 %** of iter time. This is the headline number for plan-5 P29 (small-op fusion).

## Top-30 kernels by total time (steady iter window)

Total kernel time in steady window: 16.28 s.

| rank | kernel | count | total | self avg | % step |
|---:|---|---:|---:|---:|---:|
| 1 | `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native:…` | 717 | 7.61 s | 10.62 ms | 87.3 % |
| 2 | `_v4_csa_attention_bwd_kernel` | 3 | 4.03 s | 1.34 s | 46.3 % |
| 3 | `_v4_attention_bwd_kernel` | 5 | 3.26 s | 652.09 ms | 37.4 % |
| 4 | `_v4_attention_fwd_kernel` | 5 | 634.44 ms | 126.89 ms | 7.3 % |
| 5 | `_v4_csa_attention_fwd_kernel` | 3 | 156.03 ms | 52.01 ms | 1.8 % |
| 6 | `void primus_turbo::deep_ep::intranode::cached_notify_combine<8>(void**, int*, …` | 16 | 78.29 ms | 4.89 ms | 0.9 % |
| 7 | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kern…` | 643 | 60.19 ms | 93.6 µs | 0.7 % |
| 8 | `void multi_tensor_apply_kernel<TensorListMetadata<5, false>, transformer_engin…` | 321 | 46.13 ms | 143.7 µs | 0.5 % |
| 9 | `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 9 | 30.90 ms | 3.43 ms | 0.4 % |
| 10 | `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_ke…` | 1398 | 22.97 ms | 16.4 µs | 0.3 % |
| 11 | `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat3…` | 1191 | 21.86 ms | 18.4 µs | 0.3 % |
| 12 | `void primus_turbo::deep_ep::intranode::cached_notify_dispatch<8>(int const*, i…` | 8 | 19.99 ms | 2.50 ms | 0.2 % |
| 13 | `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_…` | 20 | 15.50 ms | 774.9 µs | 0.2 % |
| 14 | `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kern…` | 204 | 15.40 ms | 75.5 µs | 0.2 % |
| 15 | `void ck_tile::kentry<1, ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocal…` | 16 | 14.48 ms | 905.2 µs | 0.2 % |
| 16 | `void ck_tile::kentry<1, ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocal…` | 16 | 13.28 ms | 830.1 µs | 0.2 % |
| 17 | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<…` | 883 | 13.24 ms | 15.0 µs | 0.2 % |
| 18 | `void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<fl…` | 762 | 12.17 ms | 16.0 µs | 0.1 % |
| 19 | `void multi_tensor_apply_kernel<TensorListMetadata<2, false>, transformer_engin…` | 321 | 10.99 ms | 34.2 µs | 0.1 % |
| 20 | `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native:…` | 26 | 10.29 ms | 395.8 µs | 0.1 % |
| 21 | `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kern…` | 2528 | 9.97 ms | 3.9 µs | 0.1 % |
| 22 | `void ck_tile::kentry<1, ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocal…` | 16 | 9.88 ms | 617.7 µs | 0.1 % |
| 23 | `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native:…` | 720 | 9.72 ms | 13.5 µs | 0.1 % |
| 24 | `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDS…` | 47 | 9.48 ms | 201.6 µs | 0.1 % |
| 25 | `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDS…` | 33 | 9.29 ms | 281.5 µs | 0.1 % |
| 26 | `void primus_turbo::deep_ep::intranode::notify_dispatch<8>(int const*, int*, in…` | 8 | 8.58 ms | 1.07 ms | 0.1 % |
| 27 | `void primus_turbo::deep_ep::intranode::dispatch<8, 1024, true>(HIP_vector_type…` | 16 | 8.53 ms | 533.4 µs | 0.1 % |
| 28 | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kern…` | 122 | 8.35 ms | 68.4 µs | 0.1 % |
| 29 | `void primus_turbo::deep_ep::intranode::combine<hip_bfloat16, 8, 1024, true>(hi…` | 16 | 8.28 ms | 517.3 µs | 0.1 % |
| 30 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<floa…` | 172 | 7.92 ms | 46.0 µs | 0.1 % |

## Kernel launch count + average launch interval (steady iter)

Total kernels launched in steady window: **15126**.
Median inter-launch interval: **4.3 µs** (p50); **71.9 µs** (p90); **541.9 µs** (p99).

| inter-launch (µs) | count | % |
|---|---:|---:|
| 0–10 | 9114 | 60.3 % |
| 10–50 | 4139 | 27.4 % |
| 50–100 | 829 | 5.5 % |
| 100–500 | 869 | 5.7 % |
| 500–5000 | 150 | 1.0 % |
| ≥5000 | 24 | 0.2 % |

## Module-level CPU op-time attribution (steady iter)

PyTorch profiler emits one `cpu_op` event for every aten / module call, **including all nested children**. So summing per-event `dur` double-counts: a top-level `DeepseekV4HybridLayer.forward` event already includes every aten op inside it. The table below shows the raw sum per pattern bucket — useful for spotting which **module subtree** dominates (top-level `Module.forward` rows like `DeepseekV4HybridLayer`, `DeepseekV4MoE`, `DeepseekV4Attention` are the meaningful numbers; the catch-all `other` row is bloated by nested aten ops and should NOT be read as 'CPU work outside V4').

| module pattern | events | Σ event dur (nests) | % iter |
|---|---:|---:|---:|
| other | 157846 | 956.85 s | 10977.9 % |
| DeepseekV4HybridLayer | 8 | 983.07 ms | 11.3 % |
| DeepseekV4MoE | 8 | 937.89 ms | 10.8 % |
| Optimizer | 2060 | 146.13 ms | 1.7 % |
| linear/matmul | 1307 | 61.94 ms | 0.7 % |
| DeepseekV4Attention | 8 | 21.08 ms | 0.2 % |
| v4_attention (Triton) | 55 | 15.88 ms | 0.2 % |
| LayerNorm/RMSNorm | 247 | 10.19 ms | 0.1 % |
| c10d/comm | 91 | 8.07 ms | 0.1 % |
| v4_csa_attention (Triton) | 9 | 3.31 ms | 0.0 % |
| DualRoPE/RoPE | 21 | 2.66 ms | 0.0 % |
| softmax | 143 | 2.06 ms | 0.0 % |

## Comm time (steady iter)

| kind | total | % iter |
|---|---:|---:|
| deepep | 0.0 µs | 0.0 % |
| nccl/c10d | 12.85 ms | 0.1 % |
| **total comm** | **12.85 ms** | **0.1 %** |

## Ranked bottleneck list + per-phase improvement budgets

Bottlenecks are ranked by **% of steady iter wall time** (not Σ kernel dur — that double-counts overlapping streams). The X / Y / Z / W per-phase budgets are the post-phase TARGETS that plan-5's `01-roadmap.md` will adopt after this report is reviewed.

| # | bottleneck | current cost | % iter | proposed budget after phase |
|---|---|---:|---:|---|
| 1 | `aten::sum` fp32 reduce kernel (top-1 template: 717 launches × ~10.62 ms) | 7.61 s | 87.3 % | **X1** = post-P29 target — root-cause + fuse / move to bf16 master / replace with Triton fused bias-grad reduce |
| 2 | V4 Triton CSA attention kernel time (cr == 4, BWD-dominated) | 4.19 s | 48.1 % | **Z** = post-P31 target — in-kernel `topk_idxs` gather + K-tile prefetch |
| 3 | V4 Triton attention kernel time (cr ∈ {0, 128}, BWD-dominated) | 3.90 s | 44.7 % | **Y** = post-P30 target — autotune BWD blocks, persistent-kernel sweep, HCA LSE merge |
| 4 | small-op kernel-launch tail (CPU-bound floor) | 23.95 ms | 0.3 % | **X2** = (de-scoped — see below) |
| 5 | comm time (DeepEP + c10d) | 12.85 ms | 0.1 % | **W** = (de-scoped — see below) |

### Per-phase de-scope decisions

Plan-5's de-scope rule: any bottleneck row < 10 % of step time gets its phase de-scoped. The data above is the input.

| phase | decision | rationale |
|---|---|---|
| P29 | **KEEP — RESCOPE** | CPU-bound floor is 0.3 % (≪ 10 % rule), so the original P29 mandate (small-op kernel-launch fusion via torch.compile or Triton-fused Compressor / Indexer / MoE-router chains) is **de-scoped**. P29 is **redirected** to root-cause + eliminate the dominant `aten::sum` fp32 reduce (87.3 % of step, 87 % of Σ kernel dur — the single largest line on the chrome-trace top-N table). Likely fix: identify whether it is bias-gradient sum-over-tokens in expert BWD or fp32 master-grad accumulation in `DistributedOptimizer`, and either fuse into a Triton kernel or move the reduction to bf16 / FP8. |
| P30 | **KEEP** | V4 Triton attention (dense + HCA) kernel time = 44.7 % of step (≥ 10 % rule). P30 must prioritise **BWD** (currently ~5 × FWD): BLOCK_M / BLOCK_N retune for head_dim=512, persistent-kernel sweep, HCA LSE-merge variant to cut the per-call cost. |
| P31 | **KEEP — RESCOPE** | V4 Triton CSA kernel time = 48.1 % of step (≥ 10 % rule). But HBM headroom is generous (~ 95 GiB free at peak), so **the original P31 motivation (cut the wrapper-side gather to fit Sq=4096) is no longer needed** — Sq=4096 already fits. P31 is **redirected** to BWD-speedup tasks: in-kernel `topk_idxs` gather to cut wrapper-side `torch.gather` + scatter-add overhead, K-tile prefetch in BWD, autotune BLOCK_K for K_topk=512. |
| P32 | **DE-SCOPE** | Comm time = 0.1 % of step (≪ 10 % rule). DeepEP + c10d are essentially free at single-node EP=8. Plan-5 P32 (pipeline / comm / optimizer overlap, recompute knobs) is **de-scoped** unless a P29 or P30 / P31 outcome materially raises comm cost (e.g. cross-node EP, or a structural change that re-introduces `overlap_grad_reduce` complexity). |

### Proposed plan-5 retarget (post-P28)

Plan-5's roadmap should adopt the P28 retarget on review:

- **P29** — `aten::sum` fp32 reduce: root-cause (likely MoE bias-grad sum-over-tokens or DistributedOptimizer fp32 master-grad accumulation), then fuse / replace. Budget X1: kill ≥ 50 % of the 7.6 s reduce kernel time.
- **P30** — V4 Triton dense / HCA attention BWD performance. Budget Y: ≥ 25 % BWD speed-up via BLOCK retune + persistent-kernel + HCA LSE merge.
- **P31** — V4 Triton CSA attention BWD performance (in-kernel `topk_idxs` gather + K-tile prefetch). Budget Z: ≥ 25 % CSA BWD speed-up.
- **P32** — DE-SCOPED. The comm / overlap budget is already won.

Combined target: **plan-5 final ≥ 110 TFLOP/s/GPU steady at Sq=4096 EP=8 single-node** (40 %+ over the 78 TFLOP/s/GPU baseline pinned in this report). Final perf gate (`G35`) lives in `03-test-strategy.md`.
