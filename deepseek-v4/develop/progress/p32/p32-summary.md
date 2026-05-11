# P32 Summary — Operator-Microbenchmark-Driven Attention Kernel Speed-Ups

> Plan-5 P32. Final post-P32 EP8 trace + report is pending; this hand-off captures
> the microbenchmark wins on `mi355-gpu-8` / `dev_primus_wenx_693`.

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

### CSA FWD split (local SWA + sparse pool + LSE merge)

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

### V4 attention BWD split (atomic-free dQ + atomic-free dK/dV)

Eliminates `tl.atomic_add` on dQ, dK, dV by splitting `_v4_attention_bwd_kernel`
into two specialised kernels with disjoint write tiles:

- `_v4_attention_bwd_dq_kernel`: one program per `(B, H, m)` block writes its dQ
  tile via `tl.store`. Scans visible K-tiles in the same SWA-pruned range as P30.
- `_v4_attention_bwd_dkv_kernel`: one program per `(B, H_K, n)` block writes its
  dK / dV tile via `tl.store`. MHA case (`HEAD_K == HEAD_Q`) skips the redundant
  head-iteration loop, halving the head-axis cost in the common path.
- Both kernels retain HCA split-mask pruning (`pool_seqlen` suffix) so the cr=128
  hot path runs the same range as P30.
- `PRIMUS_V4_ATTN_BWD_FORCE_MONOLITHIC=1` falls back to the legacy kernel.

### CSA BWD split + atomic-free segmented-reduction dpool

Local SWA branch reuses the new `_v4_attention_bwd_dq_kernel` + `_v4_attention_bwd_dkv_kernel`
with CSA's joint `lse / D`. Sparse pool branch now has two paths, both
atomic-free on `dpool`:

- **Default** (`PRIMUS_V4_CSA_BWD_SEGREDUCE=1`): two-pass segmented reduction.
  1. `_v4_csa_attention_pool_sparse_bwd_partial_kernel` writes per-visit
     `dpool` contributions to a compact `[B, M, K_topk, D]` partial buffer with
     `tl.store` (no atomics).
  2. A CPU/GPU step builds a sorted inverse index (`perm`, `bin_ptr`) mapping
     each `(B, p_idx)` pool slot back to its visiting `(B, m, k)` partial slots.
  3. `_v4_csa_attention_pool_segreduce_kernel` reduces the partial buffer into
     `dpool_fp32` segment-by-segment, also with `tl.store` (no atomics).
- **Fallback** (`PRIMUS_V4_CSA_BWD_SEGREDUCE=0`): legacy gather + `atomic_add`
  on `dpool` retained for parity. Sparse-branch `BLOCK_K=32`, `num_warps=4`
  shipped as defaults after a sweep.
- New tuning env vars: `PRIMUS_V4_CSA_BWD_SPARSE_BLOCK_H`, `PRIMUS_V4_CSA_BWD_PARTIAL_BLOCK_K`,
  `PRIMUS_V4_CSA_BWD_PARTIAL_WARPS`, `PRIMUS_V4_CSA_BWD_PARTIAL_STAGES`,
  `PRIMUS_V4_CSA_BWD_SEGREDUCE_BLOCK_D`, `PRIMUS_V4_CSA_BWD_SEGREDUCE_BLOCK_I`,
  `PRIMUS_V4_CSA_BWD_SEGREDUCE_WARPS`, `PRIMUS_V4_CSA_BWD_SEGREDUCE_STAGES`.
- Shipped tuning: `BLOCK_K_PARTIAL=16`, `partial_warps=8`, `partial_stages=2`,
  `segreduce BLOCK_D=512`, `BLOCK_I=64`, `warps=8`, `stages=3`.

## Verification

| Gate | Result |
|---|---|
| P25/P26/P31 fast attention BWD ratchet | `51 passed, 48 skipped` |
| Pre-existing `test_v4_mtp::test_helper_pulls_norm_and_linear_from_v4_provider` failure | Reproduced on `git stash` baseline — unrelated |
| `progress/p31/bench_csa_attention_ep8.py` (CSA FWD + BWD, 60 iters) | FWD 3.16 ms median, BWD 16.31 ms median |
| `progress/p32/bench_v4_attention_ep8.py` dense | FWD 0.73 ms, BWD 7.65 ms |
| `progress/p32/bench_v4_attention_ep8.py` HCA | FWD 0.91 ms, BWD 11.91 ms |
| Final EP8 proxy trace | **pending** (see follow-up below) |

## Performance

All times are median of 60 iterations after 20 warmup iterations on `mi355-gpu-8`
in `dev_primus_wenx_693` at the EP8 proxy shape (`B=1, H=64, S=4096, D=512, P=1024,
K_topk=512, swa_window=128, bf16, sink=on`).

| Kernel | Baseline (P31b) | P32 | Delta | Target | Status |
|---|---:|---:|---:|---:|---|
| V4 CSA Attention FWD | 48.17 ms | **3.16 ms** | -93.4 % (**15.2×**) | ≤ 6 ms | **MET** |
| V4 Attention BWD (dense) | 17.27 ms | **7.65 ms** | -55.7 % (**2.26×**) | ≤ 15 ms | **MET** |
| V4 Attention BWD (HCA) | 20.87 ms | **11.91 ms** | -42.9 % (**1.75×**) | ≤ 15 ms | **MET** |
| V4 CSA Attention BWD | 35.43 ms | **16.31 ms** | -54.0 % (**2.17×**) | ≤ 15 ms | missed by 1.3 ms |

Microbenchmark JSON: `progress/p32/p32_csa_postseg.json`, `p32_v4_attn_dense_final.json`,
`p32_v4_attn_hca_final.json`. Baseline JSON: `progress/p32/baseline_csa.json`,
`baseline_dense.json`, `baseline_hca.json`.

## Notes

- **CSA FWD speed-up (15.2×)** primarily comes from removing the monolithic
  joint-softmax dependency: under the old kernel each program had to merge
  local+sparse contributions in one online softmax across the full visible
  range, which made the sparse `K_topk=512` work serialise behind the SWA
  prefix. The split lets local and sparse branches run independently and then
  combine via LSE merge, exactly mirroring the way FlashAttention chunks an
  attention into independent K-tiles.
- **V4 attention BWD speed-up** comes from atomic elimination. The old kernel
  parallelised over `(B, H, n)` and used `tl.atomic_add` on dQ for every visible
  K-tile; the split kernels each own disjoint write tiles. MHA fast-path further
  drops the kvgroup head loop in the dense `cr=0` case.
- **CSA BWD speed-up** combines the split kernels (local branch ≈ 7.6 ms) with
  the new segmented-reduction dpool path (≈ 8.6 ms), eliminating the dpool
  atomic contention that was the dominant cost (~15 ms) in the gather + atomic
  path.
- **Why CSA BWD missed by 1.3 ms.** Profiling shows the residual time is HBM
  bandwidth on the partial buffer (`B*M*K_topk*D*4 = 4 GiB` write + read in
  fp32). bf16 partial (`PRIMUS_V4_CSA_BWD_PARTIAL_DTYPE=bf16`) cut it to 17.5 ms
  but failed numerical parity tests due to `-1e30` masking interacting with the
  bf16 mantissa, so fp32 partial is shipped. Multi-stream overlap of local +
  sparse kernels (`PRIMUS_V4_CSA_BWD_STREAM_OVERLAP=1`) was prototyped but stays
  default-off because both kernels are HBM-bandwidth-bound on MI355 and
  concurrent execution did not reduce median wall time.

## Failed / Negative Probes

Documented so they are not retried without new evidence:

- `dense_pool_sparse` (avoid scatter atomics by adding `[S, P]` log-count mask):
  passed parity tests after the mask fix but regressed CSA BWD to 56 ms because
  the cost of materialising the dense `[S, P]` log-count and joint softmax
  exceeded the atomic savings. Disabled by default.
- bf16 `dpool_partial`: 17.54 ms but failed parity tests, see note above.
- Fused `dpool_contrib` matmuls in the sparse BWD: regressed to 19.3 ms because
  Triton spilled the joint accumulator into registers and lost the pipelining
  win on the per-row design.
- Per-head `dpool` staging (P31 follow-up): no BWD speed-up and pushed EP8 GPU
  memory to ~178 GiB.

## Follow-Ups

1. **Final EP8 proxy trace + report**: `develop/profile/profile-after-p32-ep8-<YYYYMMDD>.{md,html}`
   and update `develop/perf/attention_perf.md` + `proxy_ep8.md` per rule R2.5.
2. **CSA BWD ≤ 15 ms**: deferred. Options on the table: (a) bf16 partial buffer
   with an explicit invalid-slot mask instead of `-1e30`, (b) folding partial
   write + segreduce into one kernel using `tl.scan` / cooperative groups (Triton
   gating on MI355 still rough), (c) running segreduce on a smaller subset of
   pool slots after observing the sparsity histogram per layer.
