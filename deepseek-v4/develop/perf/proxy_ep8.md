# Proxy EP8 End-To-End Performance

This table tracks end-to-end training performance for the Plan-5
V4-Flash EP8 proxy.

## Test Shape


| key              | value                                                                                                                                                  |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Host / container | `mi355-gpu-14` / `dev_primus_wenx_693`                                                                                                                 |
| Proxy            | V4-Flash production widths, 8-layer slice                                                                                                              |
| Parallelism      | TP=1, PP=1, EP=8                                                                                                                                       |
| Batch / sequence | MBS=1, GBS=8, `S=4096`                                                                                                                                 |
| Layers           | 8                                                                                                                                                      |
| Compress ratios  | `[0,0,4,128,4,128,4,0]`                                                                                                                                |
| Perf knobs       | `use_v4_triton_attention=True`, `use_v4_triton_csa_attention=True`, `use_turbo_deepep=True`, `use_turbo_grouped_mlp=True`, `use_turbo_attention=False` |


## Results


| Run            | Main delta                                     | Iter time (ms) | TFLOP/s/GPU | vs baseline | Source                                         |
| -------------- | ---------------------------------------------- | -------------- | ----------- | ----------- | ---------------------------------------------- |
| Baseline (P28) | Proxy baseline                                 | 8837.4         | 77.52       | 1.00x       | `profile-baseline-ep8-20260508.md`             |
| P29            | Compiled Sinkhorn                              | ~8630          | 79.1        | 1.02x       | `profile-after-p29-ep8-20260509.md`, P29 smoke |
| P30a           | Dense SWA K-loop pruning (cr=0 only)           | 6437.2         | 106.3       | 1.37x       | trace `1778313739342936814`, P30 smoke         |
| P30b           | Dense + HCA SWA K-loop pruning (cr=0 + cr=128) | 4943.4         | 138.4       | 1.79x       | `profile-after-p30-ep8-20260509.md`, P30 smoke |
| P31            | CSA in-kernel top-K gather/scatter (cr=4)      | 4317.0         | 158.5       | 2.04x       | `profile-after-p31-ep8-20260509.md`, P31 smoke |
| P31b           | CSA dense-local + sparse head-block BWD split  | 964.8          | 709.3       | 9.15x       | trace `1778324637328675032`, P31b smoke        |
| P32            | CSA FWD split (local + sparse + LSE merge)     | 890.5          | 768.4       | 9.92x       | trace `1778476971738245137`, `profile-after-p32-ep8-20260511.md` |
| **P32 RoPE-fix** | dual-RoPE cast cos/sin to bf16 (no fp32 upcast) | **665.0**    | **1029.8**  | **13.29x** | `tas-mi355x-20260514/p32_postropefix_shipped` smoke iter 8 |
| **P32 final**  | RoPE fix + split BWD + segreduce CSA BWD (defaults ON) | **603.3** | **1134.3** | **14.64x** | `tas-mi355x-20260514/p32_final_postropefix_defaults` smoke iter 10 |
| **P33**        | TFLOP/s closed-form fix (SWA visible-pair pruning + HC fn matmul; no runtime change) | **603.3** | **444.2** | **14.64x** | same trace as `P32 final`; recomputed via plan-6 P33 patch (`primus.backends.megatron.patches.deepseek_v4_flops_patches`) |
| **P34**        | `_stack_grouped_linear_weight` Triton FWD/BWD fusion (default ON) | **530.85** | **507.2** | **16.65x** | `progress/p34/runs/triton_on.iter_lines.txt` iter-10 throughput; eager fallback at same revision = 580.65 ms / 463.2 TFLOP/s/GPU |


Notes:

- Iter time and TFLOP/s/GPU use the steady / post-warmup numbers
recorded by the proxy smoke or trace report.
- P30b is the current Plan-5 P30 close-out number; P30a is kept because
it explains why the first dense-only pruning left two cr=128 BWD
kernels as 600 ms+ outliers.
- P31 uses the profiler steady window for the table headline. The final
  10-iter smoke after the `BLOCK_K=64` experiment was reverted reports
  `4312.3/4331.7 ms` and `158.7/158.0 TFLOP/s/GPU` on iter 10.
- P31b uses the post-profiler steady iter 10 line from the EP8 proxy run:
  `964.8/1114.5 ms` and `709.3/648.0 TFLOP/s/GPU`. The profiler window
  includes overhead on iter 7, so the table headline uses the clean
  post-profiler instantaneous value. Trace kernels show
  `_v4_csa_attention_pool_sparse_bwd_kernel` at **80.8 ms / 3 launches**
  and `_v4_csa_attention_pool_fwd_kernel` at **123.1 ms / 3 launches**.
- P32 ships the CSA FWD split (local SWA + sparse pool + LSE merge) as
  the new default; V4 attention BWD stays monolithic and CSA BWD stays
  on the gather + atomic dpool path. The bench-only split BWD and
  segmented-reduction CSA BWD paths remain available via
  `PRIMUS_V4_ATTN_BWD_USE_SPLIT=1` / `PRIMUS_V4_CSA_BWD_SEGREDUCE=1`
  (they win the standalone microbench but lose ~190 ms and ~40 ms /
  iter respectively in EP8 because of doubled HBM traffic competing
  with MoE work). P32 iter 10 is the post-profiler steady value:
  `890.5/1037.1 ms` and `768.4/703.8 TFLOP/s/GPU`. Trace kernels show
  `_v4_csa_attention_pool_sparse_fwd_kernel` at **33.6 ms / 3
  launches** (plus ~17 ms of `_v4_attention_fwd_kernel` for the new
  CSA local FWD; previously `_v4_csa_attention_pool_fwd_kernel` was
  **123.1 ms / 3 launches**) and `_v4_csa_attention_pool_sparse_bwd_kernel`
  at **72.5 ms / 3 launches** (vs P31b 80.8 ms).
- **P32 RoPE-fix** (2026-05-14) — `apply_interleaved_partial_rope` was
  silently upcasting Q/K to fp32 because `cos = position_ids.float() *
  inv_freq` produced an fp32 tensor and `bf16 * fp32 = fp32`. Every
  V4 attention kernel in the EP8 proxy was paying **2x HBM traffic
  for fp32 Q/K** plus running the Triton kernel against the slower
  fp32-dtype-specialised code path; in-process `cuda.Event` timings
  before the fix showed dense FWD = 5.88 ms / hca FWD = 6.87 ms
  (matches trace), and the isolated bench at fp32 reproduces those
  numbers exactly (5.65 / 6.73 ms) — proving the dtype upcast was
  the *only* source of the 1.8-7x microbench-vs-proxy gap. The fix
  is a one-line cast of `cos/sin` to `x.dtype` after the unsqueeze.
  Post-fix in-process timings drop to dense FWD = 0.82 ms and hca
  FWD = 0.97 ms, matching bench bf16 (0.78 / 0.93 ms). Loss numerics
  preserved to bf16 precision (max delta `< 1e-3` relative across
  10 iters).
- **P32 final** — after the RoPE bf16 fix, the operator-microbench
  winners (split BWD + CSA BWD segmented-reduction) also win the EP8
  proxy by ~14 % (603 vs 665 ms / iter), exactly the gap the
  microbench had predicted all along. Both env flags now default ON
  (`PRIMUS_V4_ATTN_BWD_USE_SPLIT=1`, `PRIMUS_V4_CSA_BWD_SEGREDUCE=1`),
  monolithic V4 BWD and gather+atomic CSA BWD remain reachable for
  debugging by setting either flag to `0`. The total P32 win over
  P28 baseline is **14.64x** (8837 ms → 603 ms / iter) and over
  P31b is **1.60x** (965 ms → 603 ms / iter).
- **P33** (2026-05-14, no runtime change) — plan-6 P33 corrects two
  closed-form gaps in `deepseek_v4_flops_patches.compute_v4_flops`:
  (a) the `attn_scores` term now counts only causal-visible
  `(q, k)` pairs surviving the SWA + pool + sparse top-K masks
  rather than the legacy `S_eff^2 / 2` upper bound (16x over-count
  on the dense / HCA local branches at `swa=128, S_eff=16384`),
  and (b) a new `hc` row covers the HyperConnection
  `HyperMixer.fn` + `HyperHead.fn` matmuls that were previously
  ignored (8x per-layer mixer + trunk head, ~1.25 TFLOP/iter at
  V4-Flash proxy shape — small but non-zero so future closure-of-loop
  comparisons stay strict).  The corrected total drops from
  5468 TFLOP/iter (legacy / P32 final headline) to 2144 TFLOP/iter
  (P33), and TFLOP/s/GPU drops correspondingly from 1134.3 to
  **444.2** at the same 603.3 ms iter time.  This is the honest
  number going forward; all plan-6 perf comparisons are gauged
  against this denominator.  The `vs baseline` column stays
  **14.64x** because the iter-time speedup vs P28 is unchanged —
  only the TFLOP/s headline moves.  Per-component recomputed
  breakdown is logged at rank 0 on the first
  `num_floating_point_operations` call (look for
  `[Patch:megatron.deepseek_v4.flops_reporting]` lines).
- **P34** (2026-05-14) — plan-6 P34 replaces the eager
  `torch.stack(weights, dim=0).transpose(1, 2).contiguous()` chain
  inside `PrimusTurboGroupedMLP._stack_grouped_linear_weight` with a
  single Triton kernel that does a fused `[K, N] -> [N, K]` tile-level
  transpose with per-expert int64 pointer dispatch. Default ON via
  `PRIMUS_STACK_GROUPED_WEIGHT_TRITON=1`; setting it to `0` falls
  back to the bit-identical eager chain. Microbench at V4-Flash EP=8
  widths shows **6.0x FWD / 3.9x BWD** at fc1 (E=32, K=4096, N=4096,
  bf16; 0.470 ms / 0.599 ms triton vs 2.821 ms / 2.329 ms eager) and
  **5.3x FWD / 3.2x BWD** at fc2 (E=32, K=4096, N=2048, bf16; 0.280 ms
  / 0.411 ms triton vs 1.495 ms / 1.314 ms eager). EP8 proxy A/B on
  the same script (`progress/p34/run_baseline_trace_ep8_p34.sh`) with
  the env flag toggled gives **steady-state iter time 580.65 ms
  (eager) -> 530.85 ms (Triton), -49.8 ms / -8.6 %**, matching the
  microbench prediction of 16 fc1 × 2.35 ms + 16 fc2 × 1.22 ms ≈ 57 ms
  / iter to within profiler noise. `lm_loss[10]` is bit-identical
  between the two paths (9.258817) because the operation is a pure
  layout transform. The vs-baseline ratchet vs P28 jumps to
  **16.65x** (8837.4 ms / 530.85 ms = 16.65). Plan-4 / plan-5
  release-tier `pytest.mark.slow` ratchet stayed green (92 passed,
  304 deselected in 115.88 s).
