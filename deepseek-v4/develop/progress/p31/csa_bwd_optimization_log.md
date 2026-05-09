# P31 CSA BWD Optimization Log

This log tracks the follow-up optimization work after the first P31
in-kernel `topk_idxs` gather/scatter implementation. The target from the
follow-up request is to move CSA backward toward **< 50 ms** at the proxy EP8
shape.

## Benchmark Harness

Added `bench_csa_attention_ep8.py` so kernel experiments no longer need a full
EP8 training launch.

Default shape:

| B | H | S | D | P | K_topk | SWA | dtype | sink |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 64 | 4096 | 512 | 1024 | 512 | 128 | bf16 | on |

The script times forward separately from backward. Backward timing excludes
forward execution: the forward pass is used only to build the autograd context
and saved LSE/output, then `torch.autograd.grad` is timed.

Command:

```bash
python deepseek-v4/develop/progress/p31/bench_csa_attention_ep8.py \
  --warmup 1 \
  --iters 3 \
  --json-out deepseek-v4/develop/progress/p31/csa_bench_ep8_latest.json
```

Baseline result after reverting failed tuning experiments:

| Case | FWD mean | BWD mean | Peak memory |
|---|---:|---:|---:|
| EP8 real shape, unique top-k (original pool BWD) | 48.44 ms | 1433.02 ms | 3.52 GiB |
| EP8 real shape, dense-local split BWD | 48.33 ms | 35.43 ms | 3.52 GiB |
| EP8 real shape, random top-k | 48.54 ms | 1433.74 ms | 3.52 GiB |
| EP8 real shape, sorted top-k | 48.44 ms | 1410.54 ms | 3.52 GiB |
| Local-only (`K_topk=0`) | 0.88 ms | 18.01 ms | 3.50 GiB |
| Sparse reduced (`K_topk=128`) | 21.62 ms | 749.56 ms | 3.51 GiB |

Profiler on the default shape:

| Event | CUDA time |
|---|---:|
| `_v4_csa_attention_pool_bwd_kernel` | 1.384 s |
| `_v4_csa_attention_pool_fwd_kernel` | 48.45 ms |
| `_v4_attention_bwd_preprocess_kernel` | 0.14 ms |

The benchmark now times backward-only by creating the autograd context with a
forward pass, synchronizing, and then timing only `torch.autograd.grad`.

## Reference Scan

Local `aiter` references reviewed:

- `deepseek-v4/aiter/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/bwd.py`
  uses split `dQ` and `dK/dV` backward paths plus fused atomic variants.
- `deepseek-v4/aiter/aiter/ops/triton/_triton_kernels/attention/mha_fused_bwd.py`
  uses block-level `tl.dot`, staggered inner loops, relaxed atomics, and XCD
  remapping to reduce contention.
- `deepseek-v4/aiter/aiter/ops/triton/utils/_triton/pid_preprocessing.py`
  provides XCD remap helpers for CDNA GPUs.
- Sparse/top-k attention examples in `aiter` are forward-oriented; there is no
  drop-in sparse backward for this CSA layout.

External references reviewed:

- Triton FlashAttention kernels and Dao-AILab FlashAttention Triton code:
  backward is organized around block matmuls and split `dQ` / `dK,dV` work, not
  one CTA per query row.
- Native sparse attention Triton repositories mostly cover forward or block
  sparse layouts; they do not solve per-query arbitrary top-k backward directly.
- Triton issue discussions around attention backward atomics highlight that
  atomics from MMA layouts can be expensive, but the current CSA sparse branch
  is dominated before reaching a useful tensor-core layout.

## Experiments

### E1 — Per-head `dpool` staging

Change:

- Temporarily changed `dpool_fp32` from `[B, P, D]` to `[B, H, P, D]`.
- The BWD kernel atomically accumulated into the head-private slice and the
  launcher reduced over `H` after the kernel.

Correctness:

- P31 fast pool/topk tests: `8 passed`.
- P31 release pool/topk tests: `8 passed`.

Performance:

- EP8 proxy trace stayed at `_v4_csa_attention_pool_bwd_kernel ~= 3.49 s / 3
  launches`, essentially unchanged from the P31 baseline.
- Peak training memory increased from roughly 139 GiB to roughly 178 GiB.

Decision:

- Reverted. The dominant cost is not cross-head `dpool` contention.

### E2 — `num_warps=8` for pool BWD

Change:

- Temporarily changed `_v4_csa_attention_pool_bwd_kernel` launch from
  `num_warps=4` to `num_warps=8`.

Correctness:

- P31 fast pool/topk tests: `8 passed`.

Performance:

- Microbenchmark BWD mean stayed at ~1433 ms, within noise of `num_warps=4`.

Decision:

- Reverted to `num_warps=4`.

### E3 — Top-k scaling probe

Change:

- No kernel change. Used the new benchmark to vary `K_topk`.

Result:

- `K_topk=0`: BWD ~18 ms.
- `K_topk=128`: BWD ~750 ms.
- `K_topk=512`: BWD ~1433 ms.

Decision:

- The target cannot be reached by tuning the current per-row sparse branch.
  The next viable optimization must restructure sparse BWD.

### E4 — Sort `topk_idxs` by pool id

Change:

- No kernel change. Added a benchmark-only `--sort-topk` option to test whether
  sorted pool ids improve gather/scatter locality.

Performance:

- BWD moved from ~1433 ms to ~1411 ms, only ~1.6 % faster before counting the
  real training cost of sorting `[B, S, K_topk]`.

Decision:

- Do not wire sorting into the wrapper. The locality gain is too small to pay
  for an extra sort kernel in the real model.

### E5 — Skip `dpool` write-back

Change:

- Added a temporary/default-off `PRIMUS_V4_CSA_BWD_SKIP_DPOOL=1` diagnostic
  path that keeps sparse `dq` recompute but skips the `dpool` atomic add.

Performance:

- BWD moved from ~1433 ms to ~554 ms.

Decision:

- `dpool` atomics are expensive, but even deleting them entirely cannot reach
  the <50 ms target. The sparse compute/local organization also had to change.

### E6 — Per-row sparse `tl.dot`

Change:

- Replaced the per-row sparse branch's `tl.sum` reductions with `tl.dot` for
  `q @ pool.T`, `dout @ pool.T`, and sparse `dq`.

Performance:

- With `dpool` skipped, BWD regressed from ~554 ms to ~663 ms.

Decision:

- Reverted. The per-row `(1 x D) @ (K x D)` shape does not map well enough to
  MFMA on this path; it needs head/block-level tiling.

### E7 — Split sparse head-block BWD

Change:

- Added `_v4_csa_attention_pool_sparse_bwd_kernel`, grouped heads together, and
  computed sparse `qk`, `dp`, `dq`, and `dpool` with `tl.dot`.
- The old pool BWD kernel was first kept for the local branch only.

Performance:

- BWD improved from ~1433 ms to ~571 ms.
- Profiler showed the new sparse kernel was only ~18 ms; the old local-only
  per-row kernel still cost ~503 ms.

Decision:

- Keep the sparse head-block kernel, but replace the local branch too.

### E8 — Dense local branch + sparse head-block split

Change:

- For the local SWA part, call the optimized dense `_v4_attention_bwd_kernel`
  with CSA's joint `lse` and `D=(dout*out).sum(-1)`.
- Then run `_v4_csa_attention_pool_sparse_bwd_kernel` to atomic-add sparse
  `dq` and `dpool`.

Correctness:

- P31 pool/topk fast tests: `8 passed`.
- P31 pool/topk release tests: `8 passed`.

Performance:

- Corrected benchmark BWD-only mean: **35.43 ms**.
- Profiler split: dense local `_v4_attention_bwd_kernel` **16.48 ms** +
  sparse `_v4_csa_attention_pool_sparse_bwd_kernel` **17.83 ms**.

Decision:

- This meets the <50 ms CSA backward-kernel target on the standalone EP8-shape
  benchmark and is now the default path. Set
  `PRIMUS_V4_CSA_BWD_SPLIT_SPARSE=0` to fall back to the old pool BWD path.

### Profiling note

`rocprof-compute` exists in the container, but currently fails before profiling
because its Python dependencies (`plotly`, `dash`, `textual`, and others from
`/opt/rocm-7.2.0/libexec/rocprofiler-compute/requirements.txt`) are missing.
The kernel-level attribution above therefore uses `torch.profiler` for this
timebox.

## Next Candidate Designs

1. **Pool-owned `dpool` reduction.** Build or derive an inverted top-k index so
   each program owns `(pool_id, d_tile)` and accumulates all query rows that
   selected that pool slot without random global atomics.
2. **Forward-side split.** CSA FWD is now slower than BWD on the microbench
   (about 48 ms), so the next kernel target should be a similar head-block
   sparse FWD redesign.
3. **Run full EP8 smoke/trace.** The benchmark target is met, but end-to-end
   proxy numbers still need a follow-up run before updating `proxy_ep8.md`.
