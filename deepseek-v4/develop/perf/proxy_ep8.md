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
| **P35**        | `apply_interleaved_partial_rope` Triton FWD/BWD fusion (default ON) | **526.7** | **513.3** | **16.78x** | `progress/p35/runs/triton_on.iter_lines.txt` iter-10; eager fallback at same revision (`PRIMUS_ROPE_TRITON=0`) = 531.7 ms / 507.1 TFLOP/s/GPU |
| **P36**        | `sinkhorn_normalize` Triton FWD/BWD fusion (default ON; replaces P29 compiled) | **515.0** | **520.4** | **17.16x** | `progress/p36/runs/triton_on.iter_lines.txt` iter-10; compiled fallback at same revision (`PRIMUS_SINKHORN_TRITON=0`) = 526.2 ms / 509.3 TFLOP/s/GPU |
| **P37**        | `HyperMixer.compute_weights` tail Triton FWD/BWD fusion (default ON) | **512.1** | **521.4** | **17.26x** | `progress/p37/runs/triton_on.iter_lines.txt` iter-10; eager fallback at same revision (`PRIMUS_HC_TRITON=0`) = 514.9 ms / 519.4 TFLOP/s/GPU |
| **P38**        | `Indexer.forward` scoring Triton fusion (DESCOPED, default OFF) | **512.1** | **521.4** | **17.26x** | unchanged from P37; V4-Flash microbench shows 0.72x FWD / 0.08x BWD regression vs cuBLAS / hipBLASLt eager `einsum`. Kernel checked in behind `PRIMUS_INDEXER_TRITON=1` for small-shape paths + future tuning. |
| **P39**        | V4 router post-logits Triton FWD/BWD fusion (DESCOPED, default OFF) | **513.1** | **521.4** | **17.22x** | 10-iter smoke `progress/p39/`; microbench wins on V4's `sqrtsoftplus` (1.56x FWD / 1.22x BWD) but the ~1 ms / iter gain submerges in EP=8 NCCL noise (`PRIMUS_V4_ROUTER_TRITON=1` -> 514.5 ms iters 4-10 mean vs `=0` -> 513.1 ms; lm_loss bit-identical iter-by-iter). |
| **P40 final**  | Plan-6 close-out: cumulative bake-off with all plan-6 default-on knobs (P34/P35/P36/P37); P38/P39 default OFF | **510.6** (best 509.5) | **524.9** (best 525.9) | **17.31x** (best 17.34x) | `progress/p40/p40-summary.md` §3; 15-iter clean smoke (`p40_bakeoff_plan6_defaults`); steady-state iters 8-15 mean. Plan-6 contribution vs P32 final: **-92.7 ms / iter (-15.4 %)** at +18.2 % throughput (`444.2 -> 524.9 TFLOP/s/GPU`). |


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
- **P35** (2026-05-14) — plan-6 P35 collapses the 9-op eager chain in
  `apply_interleaved_partial_rope` (slice / reshape / 4 broadcast muls /
  stack / reshape / cat) into a single Triton kernel that does one
  contiguous write with the rotation baked in. Default ON via
  `PRIMUS_ROPE_TRITON=1`; `=0` falls back to the eager body (kept in
  tree as the G38 reference). Microbench (V4-Flash EP=8 widths, bf16):
  Q shape (B=1, S=4096, H=64, head_dim=512, rd=64) → **2.96x FWD /
  2.81x BWD** (0.148 ms / 0.187 ms triton vs 0.437 ms / 0.524 ms eager
  at 3.6 TB/s effective HBM BW); K shape → **2.33x FWD / 1.51x BWD**.
  EP8 proxy A/B (`run_baseline_trace_ep8_p35.sh` with the env toggled)
  gives steady-state iter time **531.7 ms (eager) -> 526.7 ms
  (Triton), -5.0 ms / -0.94 %**, matching the microbench-predicted
  5.7 ms / iter to within profiler noise. `lm_loss[10]` is
  bit-identical (9.258817) between paths because the operation is a
  pure analytic rotation. The vs-baseline ratchet vs P28 nudges to
  **16.78×**. Plan-4 / plan-5 release-tier `pytest.mark.slow` ratchet
  stayed green (94 passed, 331 deselected in 72.90 s; the +2 vs P34
  are the G38 release-tier Q + K shape tests).
- **P36** (2026-05-14) — plan-6 P36 replaces the plan-5 P29
  `torch.compile` Sinkhorn-Knopp body in
  `hyper_connection.sinkhorn_normalize` with a hand-rolled Triton
  FWD/BWD kernel pair.  The full `1 + 2*(n_iters - 1) = 39`
  alternating row/col normalize trajectory runs in registers per row
  of the leading axis (V4-Flash `K=4`: 16 fp32 / row); BWD reads a
  cached FWD-trajectory HBM buffer (~10 MiB / call, negligible vs
  ~170 GiB rank footprint) and walks the analytic VJP backward.
  Default ON via `PRIMUS_SINKHORN_TRITON=1`; `=0` falls back to the
  P29 compiled body (still reachable via `use_compiled=True`).
  Routing precedence: `PRIMUS_SINKHORN_TRITON != "0" > use_compiled
  > eager`.  Microbench at V4-Flash K=4 widths (B=1, S=4096, K=4,
  bf16): **13.62x FWD / 14.81x BWD vs eager** (0.043 ms / 0.101 ms
  triton vs 0.587 ms / 1.502 ms eager); **6.26x FWD / 6.15x BWD vs
  P29 compiled** (0.270 ms / 0.623 ms compiled).  EP8 proxy A/B
  (`run_deepseek_v4_flash_proxy.sh` with `PRIMUS_SINKHORN_TRITON`
  toggled) gives iter-10 instantaneous **526.2 ms (compiled
  fallback) -> 515.0 ms (Triton), -11.2 ms / -2.1 %**, matching the
  microbench-predicted 12.0 ms / iter savings (16 calls × 0.75 ms
  saved / call) within profiler noise.  `lm_loss[10]` is bf16-bit-
  identical between paths (9.258826 Triton vs 9.258817 compiled
  fallback; diff `9e-6`, well below the 1e-3 bf16 floor).  Unlike
  P35 where the savings overlapped with prior BWD GPU work, P36
  lives on the serial path through `HyperConnection.compute_weights`
  and surfaces ~1:1 as wall-clock saving.  The vs-baseline ratchet
  vs P28 jumps to **17.16×**.  Plan-4 / plan-5 release-tier
  `pytest.mark.slow` ratchet stayed green (95 passed, 357 deselected
  in 73.27 s; the +1 vs P35 is the G39 release-tier V4-Flash test).
  Plan-5 G32 boundary kept observable via an `autouse=True`
  `monkeypatch.setenv("PRIMUS_SINKHORN_TRITON", "0")` fixture in
  `test_v4_p29_compiled_sinkhorn.py` (otherwise the default-on
  Triton path would silently hijack all `use_compiled=True` calls
  and the cache-hit assertion would fail).
- **P37** (2026-05-14) — plan-6 P37 collapses the 7-9 ATen
  elementwise ops in `HyperMixer.compute_weights` between the
  `_packed_logits` GEMM and the `sinkhorn_normalize` call (3 slices
  + 3 fused-multiply-adds + 2 sigmoid + 1 softmax + 2 eps adds)
  into a single Triton FWD kernel + single Triton BWD kernel.
  Saves `(sigmoid(pre_logit), sigmoid(post_logit),
  softmax(comb_logit))` as fp32 state for backward; BWD walks the
  analytic VJP per-element and uses host-side `torch.sum` to
  reduce `d_base` / `d_scale` partials (avoids cross-block
  atomic adds).  The matmul inside `_packed_logits` and the
  `collapse` / `expand` matmul-adjacent glue stay eager.  Default
  ON via `PRIMUS_HC_TRITON=1`; `=0` falls back to the eager body.
  Microbench at V4-Flash widths (B=1, S=4096, K=4, bf16 output,
  fp32 internal): **2.34x FWD / 1.47x BWD** (0.044 ms / 0.276 ms
  triton vs 0.102 ms / 0.405 ms eager).  EP=8 proxy A/B
  (`run_baseline_trace_ep8_p37.sh` with the env toggled) gives
  iters 4-10 mean (excl. profile-end iter 7) **519.4 ms (eager
  tail) -> 516.1 ms (Triton), -3.3 ms / -0.64 %**; iter-10
  instantaneous **514.9 -> 512.1 ms**, matching the microbench-
  predicted 3 ms / iter savings (16 calls × 0.19 ms saved / call).
  `lm_loss[10]` is bf16-bit-identical between paths (9.258826 both
  A and B).  The vs-baseline ratchet vs P28 nudges to **17.26×**
  (8837.4 / 512.1).  Plan-4 / plan-5 release-tier `pytest.mark.slow`
  ratchet stayed green (the +1 vs P36 is the G40 release-tier
  V4-Flash test).
- **P38** (2026-05-14) — plan-6 P38 (`Indexer.forward` scoring
  Triton fusion) **descoped per the explicit clause in
  `plan-6/02-phase-details.md` §"Task list refinement"**.  The
  eager `einsum + relu + mul + sum + causal_mask` chain at V4-Flash
  widths (B=1, S=4096, P=1024, H=8, Hd=128) is dominated by a
  cuBLAS / hipBLASLt batched-matmul that runs at ~28 TFLOP/s on
  MI355.  The generic Triton kernel under-utilises tensor cores
  (BLOCK_S=BLOCK_P=32 vs cuBLAS's 128x128 tile) and the BWD's
  three `tl.atomic_add` calls per program (on `dq` / `dk` / `dw`)
  create ~12x contention.  Microbench at V4-Flash widths:
  **FWD 0.424 ms (triton) vs 0.306 ms (eager) -> 0.72x regression;
  BWD 6.457 ms (triton) vs 0.489 ms (eager) -> 0.08x regression
  (12x slower).**  Small-shape paths (B=2, S=128, P=32) still win
  (3.35x FWD speedup) -- the kernel is checked in for future
  tuning + small-shape paths via `PRIMUS_INDEXER_TRITON=1` (default
  `"0"`).  The proxy iter time vs P37 is therefore unchanged
  (still ~512.1 ms / iter steady-state).  No EP=8 A/B trace
  collected because the V4-Flash microbench already shows the
  regression dominates; running the proxy with the Triton path on
  would just measure the regression end-to-end.  Plan-4 / plan-5
  release-tier ratchet stayed green (the +1 vs P37 is the G41
  release-tier V4-Flash test).
- **P39** (2026-05-15) — plan-6 P39 (V4 router post-logits Triton
  fusion shared between learned topk + hash routers) **descoped to
  default-OFF** following the same precedent as P38.  The kernel
  collapses the 7-op eager chain (`score_fn + gather + sum.clamp.div
  + scaling + sparse scatter to probs + sparse scatter to
  routing_map`) into one Triton FWD + one BWD kernel; both kernels
  build the dense `[N, E]` output / gradient tile entirely in
  registers (no store-then-load round trip -- a coherence bug we
  hit + fixed during development).  `score_function: tl.constexpr`
  emits 3 specialised binaries (softmax / sqrtsoftplus / sigmoid).
  Microbench at V4-Flash widths (N=4096, E=256, K=8, bf16) shows
  the kernel does win on V4's production score function:
  **`sqrtsoftplus` -> 1.56x FWD / 1.22x BWD** (0.046 ms / 0.150 ms
  triton vs 0.072 ms / 0.183 ms eager).  But `softmax` BWD
  regresses ~30% (eager `softmax_backward` is an Inductor-fused
  kernel that beats a generic Triton chain).  EP=8 proxy A/B
  (10-iter smoke with `PRIMUS_V4_ROUTER_TRITON` toggled): iters 4-10
  mean **513.1 ms (eager) -> 514.5 ms (Triton), +1.4 ms within the
  ~±2-3 ms NCCL / dispatch noise band**.  lm_loss is bit-identical
  iter-by-iter (every step prints the exact same 6-digit decimal:
  11.16446, 10.94438, 10.38210, 10.07792, 9.814991, 9.446677,
  9.287911, 9.257534), confirming full forward+backward parity.
  The microbench gain (~1 ms / iter aggregate at 16 router calls
  per iter on `sqrtsoftplus`) is submerged in the proxy variance,
  so the kernel ships behind `PRIMUS_V4_ROUTER_TRITON=1` (default
  `"0"`); bit-identity makes the env knob flip safe for future
  re-enablement.  The proxy iter time vs P38 is unchanged at
  ~512-513 ms / iter steady-state.  Plan-4 / plan-5 release-tier
  ratchet stayed green (the +1 vs P38 would be the G42 release-tier
  router test if release-tier coverage is added; the 21 G42 fast +
  3 shape-variant skipped tests already pass).
- **P40 final** (2026-05-15) — plan-6 close-out.  15-iter EP=8
  proxy bake-off with all plan-6 default-on flags enabled
  (`PRIMUS_STACK_GROUPED_WEIGHT_TRITON=1`, `PRIMUS_ROPE_TRITON=1`,
  `PRIMUS_SINKHORN_TRITON=1`, `PRIMUS_HC_TRITON=1`) and the two
  descoped knobs off (`PRIMUS_INDEXER_TRITON=0`,
  `PRIMUS_V4_ROUTER_TRITON=0`) -- the plan-6 production state.
  Steady-state iters 8-15 (post-warmup clean window) show
  **mean 510.6 ms / iter, 524.9 TFLOP/s/GPU; best iter 13 at
  509.5 ms / 525.9 TFLOP/s/GPU**.  Peak HBM / rank: 172.3 GiB
  (59.84 %).  Cumulative speedup vs the P28 anchor:
  `8837.4 / 510.6 = 17.31x` mean (best iter 17.34x).  Plan-6
  contribution vs the plan-5 P32 final iter time (603.3 ms):
  **-92.7 ms / iter saved (-15.4 %)**; vs the P33-corrected
  TFLOP/s denominator (444.2 TFLOP/s/GPU): **+18.2 % throughput
  (444.2 -> 524.9)**.  See `progress/p40/p40-summary.md` for the
  full close-out: per-phase iter-time decomposition,
  roadmap-target gap analysis (510.6 ms vs the original 385 ms
  target — the 125 ms shortfall sits in V4 attention kernels,
  not elementwise fusion, and is plan-7 scope), plan-7 follow-up
  candidates inherited from P37 (`collapse` / `expand` /
  `HyperHead.forward`), P38 (tensor-core-friendly Indexer tiles),
  and P39 (Inductor-tier softmax BWD; dispatcher input fusion).
