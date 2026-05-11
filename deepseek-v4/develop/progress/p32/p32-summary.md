# P32 Summary — Operator-Microbenchmark-Driven Attention Kernel Speed-Ups

> Plan-5 P32. Main report: `../../profile/profile-after-p32-ep8-20260511.{md,html}`.

## Objective

Close the three single-kernel residuals pinned by the post-P31b trace. All three
were verified on standalone EP8-shape microbenchmarks (`progress/p31/bench_csa_attention_ep8.py`
for CSA, new `progress/p32/bench_v4_attention_ep8.py` for V4 attention dense / HCA)
so kernel iteration does not require a full EP8 training round-trip:

| Target | Baseline (P31b bench) | Goal |
|---|---:|---:|
| V4 CSA Attention FWD | 48.17 ms | ≤ 6 ms |
| V4 Attention BWD (dense) | 17.27 ms | ≤ 15 ms |
| V4 Attention BWD (HCA) | 20.87 ms | ≤ 15 ms |
| V4 CSA Attention BWD | 35.43 ms | ≤ 15 ms |

## What Changed

### CSA FWD split (local SWA + sparse pool + LSE merge) — **shipped default ON**

Replaces the monolithic `_v4_csa_attention_pool_fwd_kernel` with three kernels
joined by an online-softmax LSE merge so the local and sparse branches no longer
serialize through a single program:

- Local SWA branch reuses `_v4_attention_fwd_kernel` (the P30-pruned dense
  kernel) over an `[S, S]` joint softmax.
- New `_v4_csa_attention_pool_sparse_fwd_kernel` reads `pool` via gathered
  `topk_idxs`, runs head-block `tl.dot`, and writes per-row LSE + accumulator.
- New `_v4_csa_attention_lse_merge_kernel` combines the local and sparse LSE +
  accumulator pairs into the final `out` and `lse` consumed by BWD.
- Launcher `_launch_v4_csa_attention_pool_fwd` defaults to the split path;
  `PRIMUS_V4_CSA_FWD_FORCE_MONOLITHIC=1` falls back to the original kernel.

### V4 attention BWD split (atomic-free dQ + atomic-free dK/dV) — **opt-in via `PRIMUS_V4_ATTN_BWD_USE_SPLIT=1`**

Eliminates `tl.atomic_add` on dQ, dK, dV by splitting `_v4_attention_bwd_kernel`
into two specialised kernels with disjoint write tiles:

- `_v4_attention_bwd_dq_kernel`: one program per `(B, H, m)` block writes its dQ
  tile via `tl.store`. Scans visible K-tiles in the same SWA-pruned range as P30.
- `_v4_attention_bwd_dkv_kernel`: one program per `(B, H_K, n)` block writes its
  dK / dV tile via `tl.store`. MHA case (`HEAD_K == HEAD_Q`) skips the redundant
  head-iteration loop, halving the head-axis cost in the common path.
- Both kernels retain HCA split-mask pruning (`pool_seqlen` suffix) so the cr=128
  hot path runs the same range as P30.
- **Default is OFF** because the split design reads Q / K / V twice (one read
  per kernel) — the resulting 2× HBM traffic wins the standalone microbench but
  loses ~190 ms / iter against the concurrent MoE work in the EP8 proxy. The
  monolithic kernel (one read, three gradients per visit) is faster end-to-end.

### CSA BWD with atomic-free segmented-reduction dpool — **opt-in via `PRIMUS_V4_CSA_BWD_SEGREDUCE=1`**

The CSA local SWA branch always reuses the V4 attention BWD path (split or
monolithic, mirroring `PRIMUS_V4_ATTN_BWD_USE_SPLIT`). The sparse pool branch
now has two paths:

- **Shipped default**: legacy gather + `atomic_add` on `dpool` (the P31b
  sparse BWD kernel) with sparse `BLOCK_K=32`, `num_warps=4` tuned defaults.
- **Opt-in segreduce**: a new two-pass design that writes per-visit `dpool`
  contributions atomic-free.
  1. `_v4_csa_attention_pool_sparse_bwd_partial_kernel` writes a compact
     `[B, M, K_topk, D]` partial buffer with `tl.store` (no atomics).
  2. CPU/GPU step builds a sorted inverse index (`perm`, `bin_ptr`) mapping
     each `(B, p_idx)` pool slot back to its visiting partial slots.
  3. `_v4_csa_attention_pool_segreduce_kernel` reduces the partial buffer into
     `dpool_fp32` segment-by-segment via the inverse index, also atomic-free.
- Default is OFF for the same reason as the V4 BWD split: the 4 GiB partial
  buffer's HBM traffic competes with MoE in EP8 (regresses iter time by
  ~40 ms / iter) even though it wins the standalone microbench.
- New tuning env vars: `PRIMUS_V4_CSA_BWD_SPARSE_BLOCK_H`, `PRIMUS_V4_CSA_BWD_PARTIAL_BLOCK_K`,
  `PRIMUS_V4_CSA_BWD_PARTIAL_WARPS`, `PRIMUS_V4_CSA_BWD_PARTIAL_STAGES`,
  `PRIMUS_V4_CSA_BWD_SEGREDUCE_BLOCK_D`, `PRIMUS_V4_CSA_BWD_SEGREDUCE_BLOCK_I`,
  `PRIMUS_V4_CSA_BWD_SEGREDUCE_WARPS`, `PRIMUS_V4_CSA_BWD_SEGREDUCE_STAGES`.
- Shipped tuning for the opt-in path: `BLOCK_K_PARTIAL=16`, `partial_warps=8`,
  `partial_stages=2`, `segreduce BLOCK_D=512`, `BLOCK_I=64`, `warps=8`, `stages=3`.

## Verification

| Gate | Result |
|---|---|
| P25/P26/P31 fast attention BWD ratchet | `51 passed, 48 skipped` |
| Pre-existing `test_v4_mtp::test_helper_pulls_norm_and_linear_from_v4_provider` failure | Reproduced on `git stash` baseline — unrelated |
| Shipped microbench (`bench_csa_attention_ep8.py`, 60 iters median) | CSA FWD 3.22 ms; CSA BWD 32.62 ms |
| Shipped microbench (`bench_v4_attention_ep8.py --mode dense`) | dense FWD 0.71 ms; dense BWD 17.26 ms |
| Shipped microbench (`bench_v4_attention_ep8.py --mode hca`) | HCA FWD 0.85 ms; HCA BWD 20.66 ms |
| Opt-in bench-optimal (env vars `..._USE_SPLIT=1 ..._SEGREDUCE=1`) | CSA FWD 3.16 ms; CSA BWD 16.31 ms; dense BWD 7.65 ms; HCA BWD 11.91 ms |
| Final EP8 proxy trace | iter 10 **890.5 ms / 768.4 TFLOP/s/GPU**, profiler steady 899.99 ms / 859.54 ms GPU active (95.5 %) |

## Performance

All times are median of 60 iterations after 20 warmup iterations on `mi355-gpu-8`
in `dev_primus_wenx_693` at the EP8 proxy shape (`B=1, H=64, S=4096, D=512, P=1024,
K_topk=512, swa_window=128, bf16, sink=on`).

### Microbenchmark wall time

| Kernel | Baseline (P31b) | P32 shipped (default) | P32 opt-in (env vars) | Goal |
|---|---:|---:|---:|---:|
| V4 CSA Attention FWD | 48.17 ms | **3.22 ms** (**−93.3 %**, 15.0×) | 3.16 ms | ≤ 6 ms |
| V4 Attention BWD (dense) | 17.27 ms | 17.26 ms (parity, monolithic kept) | **7.65 ms** (**−55.7 %**, 2.26×) | ≤ 15 ms |
| V4 Attention BWD (HCA) | 20.87 ms | 20.66 ms (parity, monolithic kept) | **11.91 ms** (**−42.9 %**, 1.75×) | ≤ 15 ms |
| V4 CSA Attention BWD | 35.43 ms | 32.62 ms (−7.9 %) | **16.31 ms** (**−54.0 %**, 2.17×) | ≤ 15 ms |

### Proxy EP8 end-to-end (clean post-profiler iter 10)

| Config | Iter time | TFLOP/s/GPU | vs P31b |
|---|---:|---:|---:|
| P31b replay on `mi355-gpu-8` (sanity) | 963.0 ms | 710.6 | — |
| **P32 shipped (default)** | **890.5 ms** | **768.4** | **−7.7 % iter, +8.1 % TFLOP/s** |
| P32 opt-in `..._SEGREDUCE=1` only | 1114.8 ms | 613.8 | +15.7 % iter (regression) |
| P32 opt-in `..._USE_SPLIT=1 ..._SEGREDUCE=1` | 1152.9 ms | 593.6 | +19.7 % iter (regression) |
| P32 fallback CSA FWD monolithic | 1189.2 ms | 575.4 | +23.4 % iter (regression) |

### Trace kernel attribution (steady profiler window)

| component | P31b | P32 shipped | delta |
|---|---:|---:|---:|
| Profiler window | 980.9 ms | 899.99 ms | −8.2 % |
| GPU active (union) | 940.04 ms | 859.54 ms | −8.6 % |
| `_v4_csa_attention_pool_fwd_kernel` (monolithic) | 123.07 ms / 3 | 0 | −100 % |
| `_v4_csa_attention_pool_sparse_fwd_kernel` (split sparse) | n/a | 33.59 ms / 3 | new |
| `_v4_attention_fwd_kernel` (now includes CSA local) | 30.04 ms / 5 | 46.96 ms / 8 | +56 % (3 extra CSA local FWDs) |
| **Net CSA FWD work** | ~123.1 ms | **~50.6 ms** | **−59 %** |
| `_v4_attention_bwd_kernel` | 259.74 ms / 8 | 256.97 ms / 8 | −1.1 % (monolithic kept) |
| `_v4_csa_attention_pool_sparse_bwd_kernel` | 80.81 ms / 3 | 72.54 ms / 3 | −10.2 % |
| **Net attention family kernel time** | ~493 ms | **~410 ms** | **−16.8 %** |

Trace archive: `output/amd/tas-mi355x-20260511/p32_profile_split_kernels_pp1_ep8_seq4096/tensorboard/primus-megatron-exp_p32_trace_1778476971738245137.pt.trace.json.tgz`.

## Notes

- **CSA FWD speed-up (15.0×)** ships because it wins both the microbench and the
  proxy. The split lets local and sparse branches run independently and then
  combine via LSE merge, exactly mirroring the way FlashAttention chunks an
  attention into independent K-tiles. In the trace, monolithic CSA FWD
  (`_v4_csa_attention_pool_fwd_kernel` 123.07 ms / 3) is replaced by ~33.6 ms
  of new sparse kernel + ~17 ms of CSA local on `_v4_attention_fwd_kernel`,
  saving ~73 ms / iter.
- **V4 attention BWD split** wins the microbench (2.26× dense / 1.75× HCA) but
  loses ~190 ms / iter in EP8 because the split design reads Q / K / V twice
  (once per kernel; 2× HBM traffic vs the monolithic kernel that produces all
  three gradients in a single visit). The monolithic kernel was tuned for HBM
  efficiency by P30 and remains optimal when MoE work is competing for HBM. The
  split kernels ship as `PRIMUS_V4_ATTN_BWD_USE_SPLIT=1` for microbench testing.
- **CSA BWD segreduce** similarly trades 4 GiB of HBM traffic on the partial
  buffer for atomic-free dpool writes. The trade wins the standalone microbench
  (−54 %) but loses ~40 ms / iter in EP8 (HBM-contended). The shipped default
  is the gather + atomic_add dpool path, which is already ~10 % faster than
  P31b's same kernel due to incidental Triton autotuner improvements at the
  shared bench shape.
- **Why miss the ≤ 15 ms BWD targets in shipped mode.** The kernel-level
  targets are achievable via the opt-in env vars, but shipping them by default
  would regress end-to-end training throughput. Production picks proxy-optimal
  defaults; the bench-optimal paths remain available behind env vars for
  kernel-engineering follow-ups (e.g. closing the gap by giving the split
  kernels shared HBM reads via cooperative groups / persistent kernels).

## Failed / Negative Probes

Documented so they are not retried without new evidence:

- `dense_pool_sparse` (avoid scatter atomics by adding `[S, P]` log-count mask):
  passed parity tests after the mask fix but regressed CSA BWD to 56 ms because
  the cost of materialising the dense `[S, P]` log-count and joint softmax
  exceeded the atomic savings. Removed.
- bf16 `dpool_partial` (cut segreduce HBM traffic in half): 17.54 ms but failed
  numerical parity tests because `-1e30` masking interacts poorly with the bf16
  mantissa.
- Fused `dpool_contrib` matmuls in the sparse BWD: regressed to 19.3 ms because
  Triton spilled the joint accumulator into registers and lost the pipelining
  win on the per-row design.
- Per-head `dpool` staging (P31 follow-up): no BWD speed-up and pushed EP8 GPU
  memory to ~178 GiB.
- Defaulting the V4 attention BWD split ON: regressed EP8 proxy iter time by
  ~190 ms (`mi355-gpu-8`, 8 V4 attention + 3 CSA local BWD launches) because of
  2× HBM read traffic per BWD step (see Notes).
- Defaulting CSA BWD segreduce ON: regressed EP8 proxy iter time by ~40 ms
  (4 GiB partial buffer HBM traffic competing with MoE).

## Follow-Ups

1. **CSA BWD ≤ 15 ms without proxy regression**: re-imagine the partial buffer
   to share Q / K / V reads with the V4 BWD pass (e.g. cooperative groups),
   or shrink it to bf16 with a proper invalid-slot mask (replacing the `-1e30`
   sentinel) so the HBM traffic fits within the MoE-leftover bandwidth.
2. **V4 attention BWD ≤ 15 ms without proxy regression**: persistent-kernel
   design where the dQ kernel reuses K / V tiles loaded by the dK / dV pass,
   or a single kernel split only on the program axis (still writes all three
   gradients per visit).
3. **EP8 + opt-in microbench-optimal end-to-end test**: keep the env vars in
   the perf-tuning harness so a future kernel iteration that closes the HBM gap
   can flip the defaults without code changes.
