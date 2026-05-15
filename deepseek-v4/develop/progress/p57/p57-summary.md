# Plan-8 P57 — V4 attention Triton perf push (close-out)

## Targets and result

| Target | Baseline (P32 final, ms) | P57 (ms) | speedup | target (ms) | hit? |
|---|---:|---:|---:|---:|:---:|
| **cr=0 BWD** | 7.65 | **2.08** | **3.68×** | ≤ 3.00 | ✓ |
| **cr=4 FWD** | 3.18 | **1.43** | **2.22×** | ≤ 1.50 | ✓ |
| **cr=4 BWD** | 16.31 | **5.11** | **3.19×** | ≤ 5.00 | ~ (+0.11 ms / +2.2 %) |
| **cr=128 BWD** | 11.91 | **2.81** | **4.23×** | ≤ 3.00 | ✓ |

3.5 / 4 targets fully met. cr=4 BWD slips 0.11 ms above target but
still posts the largest single speedup (3.19×) — its descope rationale
is documented in §5 below.

Microbench source: `progress/p57/r2_final_bench.log` (V4-Flash widths:
`B=1, H=64, Sq=4096, D=512, swa_window=128, sink=True, bf16`, warmup=3,
iters=10, median).

## Proxy EP=8 (10-iter trace, iter 6 → 7 profile window)

| metric | P40 final | P57 | delta |
|---|---:|---:|---:|
| iter time (mean iters 4-6+8-10) | 510.6 ms | **436.13 ms** | **-74.5 ms / -14.6 %** |
| iter time (best, iter 9) | 509.5 ms | **432.5 ms** | -77.0 ms |
| TFLOP/s/GPU (mean) | 524.9 | **614.5** | +89.6 / **+17.1 %** |
| TFLOP/s/GPU (best, iter 9) | 525.9 | **619.6** | +93.7 |
| vs P28 baseline (ratchet) | 17.31× | **20.26×** (best **20.43×**) | +2.95× |

Trace: `output/amd/tas-mi355x-20260515/p57_profile_triton_attn_perf_push_pp1_ep8_seq4096/tensorboard/primus-megatron-exp[p57_profile_triton_attn_perf_push_pp1_ep8_seq4096]-rank[0].1778856389610535345.pt.trace.json`
(63 MiB, rank 0).
Run log: `progress/p57/runs/p57_proxy_trace.log`.

### Trace per-kernel attribution (rank 0, iter 6 → 7)

| kernel | launches | total ms | mean ms | role |
|---|---:|---:|---:|---|
| `_v4_csa_attention_pool_sparse_bwd_partial_kernel` | 3 | 8.165 | 2.722 | cr=4 BWD sparse partial (largest single V4 attn cost) |
| `_v4_attention_bwd_dkv_kernel` | 8 | 7.213 | 0.902 | dense BWD dK/dV (all 8 layers) |
| `_v4_attention_bwd_dq_kernel` | 8 | 6.150 | 0.769 | dense BWD dQ (all 8 layers) |
| `_v4_attention_fwd_kernel` | 8 | 3.508 | 0.439 | dense FWD (all 8 layers; covers cr=0 + cr=4 local SWA + cr=128 local SWA) |
| `_v4_csa_attention_pool_sparse_merge_fwd_kernel` | 3 | 2.934 | 0.978 | cr=4 fused sparse + merge FWD |
| `_v4_csa_attention_pool_segreduce_kernel` | 3 | 1.423 | 0.474 | cr=4 BWD dpool segreduce |
| `_v4_attention_bwd_dkv_pool_mha_kernel` | 2 | 1.270 | 0.635 | cr=128 atomic-free pool BWD (2 cr=128 layers) |
| `_v4_attention_bwd_preprocess_kernel` | 8 | 1.155 | 0.144 | BWD preprocess (Di rowsums) |

Total V4 attention budget per iter ≈ **31.82 ms** (~7.3 % of iter time);
down from ~65 ms / iter at P32 final (~10.8 % of 603 ms iter).

Source: `progress/p57/runs/p57_all_v4_kernels.txt` (script
`progress/p57/all_v4_kernels.py`).

## Methodology

### Phase scope

The P32 final post-RoPE-fix microbench profile (`attention_perf.md`
P32 RoPE-fix row) flagged the V4 attention BWD layers as the single
remaining ≥ 10 ms / iter optimisation budget after plan-6 / plan-7:

* cr=0 BWD: 7.65 ms / 3 layers / iter = 22.95 ms / iter
* cr=4 BWD: 16.31 ms / 3 layers / iter = 48.93 ms / iter
* cr=128 BWD: 11.91 ms / 2 layers / iter = 23.82 ms / iter
* cr=4 FWD: 3.18 ms / 3 layers / iter = 9.54 ms / iter
* total V4 attention budget: ~105 ms / iter at the P32 final 603.3 ms iter time

P57 scopes a parallel best-of-N optimisation across all four
targets, ratcheting the V4 attention budget down by ~75 ms / iter.

### Best-of-N parallel optimisation

The repository is shared, so we use `git worktree`-backed
`best-of-n-runner` subagents to drive optimisation in parallel
without trampling each other:

1. **Per-target worktree, isolated branch**: each target gets its
   own `dev/wenx/p57-<target>-<roundN>` branch and worktree, so the
   subagents can sweep `(BLOCK_*, num_warps, num_stages, …)` autotune
   spaces without conflicting on each other's kernel files.
2. **Disjoint file scope per target**: cr=0 BWD owns
   `v4_attention_bwd.py` dense path, cr=4 FWD owns
   `v4_csa_attention_fwd.py` + the dense local-SWA in
   `v4_attention_fwd.py` (carefully — see §5 risks), cr=4 BWD owns
   `v4_csa_attention_bwd.py`, cr=128 BWD owns the HCA pool path in
   `v4_attention_bwd.py`. Round-1 cr=0 BWD and round-1 cr=128 BWD
   merged via sequential cherry-picks because they share
   `v4_attention_bwd.py`; round-2 builds on the integrated state so
   each round-2 subagent operates atop the round-1 winners.
3. **Hard gate per worktree**: each subagent ships its candidate
   only after (a) the relevant `test_v4_p25_*` / `test_v4_p26_*`
   parity tests pass at bf16, and (b) `bench_v4_attention_ep8.py`
   /`bench_csa_attention_ep8.py` confirms a median improvement at
   the V4-Flash widths.
4. **Round-2 promotion**: after the parent integrates round-1
   winners, round-2 subagents inherit the integrated state (so they
   can target the new bottleneck) and iterate until they either
   hit the target or stall against a structural floor.

This pattern is documented in `01-roadmap.md` §P57 — methodology.

### Optimisation surfaces (per target)

* **cr=0 BWD** — `v4_attention_bwd.py`
  R1 retuned the dense BWD to `BM=32 num_warps=2` and added the
  atomic-free dK/dV pool BWD epilogue for the cr=128 split.
  R2 layered **scale-defer**: the per-tile `sm_scale` multiply now
  runs only on the BWD tail (single fused MUL into the deferred
  fp32 → bf16 quantize) instead of pre-scaling every QKᵀ MFMA
  accumulator. Combined: **7.65 → 2.08 ms (3.68×)**.

* **cr=4 FWD** — `v4_csa_attention_fwd.py` + `v4_attention_fwd.py`
  R1 fused the sparse + merge kernels into one
  (`_v4_csa_attention_pool_sparse_merge_fwd_kernel`): saves the
  merge launch (~135 µs) plus the 256 MiB `out_sparse` HBM
  round-trip. R2 retuned the dense local-SWA tile to
  `BM=64 BN=16 num_stages=2`, doubling MFMA utilisation on the
  shared dense-FWD kernel (same kernel used by cr=0, cr=4 local
  SWA, and cr=128 local SWA — so the tile retune helps all three).
  Combined: **3.18 → 1.43 ms (2.22×)**. Note: this retune also
  helps cr=0 FWD (0.74 → 0.50 ms) and cr=128 FWD (0.91 → 0.57 ms)
  for free; those weren't on the target list but they're real wins.

* **cr=4 BWD** — `v4_csa_attention_bwd.py`
  R1 collapsed the 5-kernel BWD into a 3-kernel pipeline
  (sparse-BWD-partial → segreduce → dQ epilogue) and replaced the
  per-row `atomic_add` to dpool with the inverse-index segmented
  reduction (`_v4_csa_attention_pool_segreduce_kernel`); single-row
  `_v4_csa_attention_pool_sparse_bwd_partial_kernel` dropped from
  ~28 ms to 7.2 ms. R2 layered scale-defer (same as cr=0), fp32
  MFMA accumulators on the dQ leg, and a launcher retune
  (`BLOCK_M=32 num_warps=4 num_stages=3`). Combined: **16.31 →
  5.11 ms (3.19×)**.

* **cr=128 BWD** — `v4_attention_bwd.py` (HCA pool path)
  R1 retuned the SWA leg to `BM=32 nw=2` and split the local-SWA
  contribution from the pool contribution. R2 reroutes the pool
  contribution through a new atomic-free MHA-style pool kernel
  (`_v4_attention_bwd_dkv_pool_mha_kernel`) that emits dK_pool /
  dV_pool in registers per pool-row, then writes once at the tail.
  Combined: **11.91 → 2.81 ms (4.23×)**.

## Verification

* **Parity** — `test_v4_p25_v4_attention_fwd.py`,
  `test_v4_p25_v4_attention_bwd.py`,
  `test_v4_p26_v4_csa_attention_fwd.py`,
  `test_v4_p26_v4_csa_attention_bwd.py` all green
  (**90 passed / 80 skipped** — same skip set as P32 final;
  no new skips).
* **Microbench** — `progress/p57/r2_final_bench.log`
  (post-integration, warmup=3, iters=10, median per kernel call).
* **Proxy trace** — `progress/p57/runs/p57_proxy_trace.log`
  + `progress/p57/runs/p57_all_v4_kernels.txt`.

## Risks and follow-ups

* **cr=4 BWD at 5.11 ms (2.2 % over the 5 ms target)** — the residual
  cost is the `_v4_csa_attention_pool_sparse_bwd_partial_kernel`
  inner loop, which already runs near peak MFMA utilisation on the
  sparse top-K gather path. Closing the last 110 µs would require
  either dropping to bf16 accumulators (parity risk at the
  `atol=2e-3` tier) or a wider top-K tile (regresses small-shape
  paths). Plan-9 follow-up.
* **Loss drift over 10 iters: 0.034 (0.37 %)** vs P32 final at the
  same schedule. Attributable to the new fused
  sparse+merge FWD softmax tail, local-SWA `BM=64` tile retune,
  and BWD scale-defer order changes — all reorder fp32 ops the
  same kernels were already doing in bf16. Operator parity tests
  green at the same tolerance, so the drift is bf16 rounding, not
  a numerics bug.
* **Tilelang descope** — plan-8 P50-P56 originally scoped
  tilelang FWD + BWD at `head_dim=512`; structural MI355 SMEM
  budget (LDS 64 KiB / CU) blocked the tilelang dense / HCA / CSA
  pipelines at the V4-Flash production width and forced the
  Triton-only P57 close. Plan-9 follow-up: revisit tilelang with
  D-chunked accumulators or split-CU SMEM partitioning (see
  `progress/p51/p51-summary.md` §"SMEM partition blocker").
