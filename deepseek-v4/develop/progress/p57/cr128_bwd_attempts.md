# Plan-8 P57 — V4 HCA BWD (cr=128) optimisation attempts

Branch: `dev/wenx/p57-cr128-bwd-attempt1`
Baseline (`dev/wenx/deepseek-v4` @ `112d0dd1`):
- cr=128 HCA BWD median: **11.90 ms** (15.95 TF, ~10% MFMA peak)
- cr=0 dense BWD median: **7.66 ms**

All measurements at V4-Flash production widths (`B=1, H=64, Sq=4096,
D=512, P=32, swa_window=128, sink=True, bf16, MQA HK=1`) on
`mi355-gpu-8` / `dev_primus_wenx_693`.

Bench command:
```
PRIMUS_V4_ATTN_BWD_USE_SPLIT=1 PRIMUS_V4_CSA_BWD_SEGREDUCE=1 \
  python deepseek-v4/develop/progress/p32/bench_v4_attention_ep8.py \
  --mode {hca|dense} --warmup 3 --iters 10
```

## Attempt 1 — Pool dKV split into separate kernel

Hypothesis: in the original dKV kernel the single pool n-block program
iterates `H × Sq/BLOCK_M = 64 × 128 = 8192` inner iterations versus
`H × SWA_WINDOW/BLOCK_M = 64 × 4 = 256` for a local n-block program —
**32× more work**. A dedicated pool kernel parallelised over m-blocks
should eliminate that long pole.

Two sub-attempts:

### Attempt 1a — pool grid `(Sq/BLOCK_M, B*HQ) = (128, 64) = 8192 progs`
Each pool program handles `1 m-block × 1 query head` and atomic-adds
`(BLOCK_N, BLOCK_DMODEL) = (32, 512)` fp32 cells into the shared pool
slice. Atomic contention per cell: 8192-way (B × HQ × M_BLOCKS).

Result: HCA = 12.73 ms (**+0.83 ms regression**). Atomic contention
dominates vs the original 1-program pool branch.

### Attempt 1b — pool grid `(Sq/BLOCK_M, B) = (128, 1) = 128 progs`
Each program handles `1 m-block × 64 heads` (head loop inside) and
atomic-adds the accumulated pool tile **once** at the end. Atomic
contention per cell: 128-way.

Result: HCA = 12.59 ms (**+0.69 ms regression**). The pool kernel
itself takes ~3.5 ms (vs ~3 ms when it was fused into the dKV kernel
and overlapped with the 128 local programs); separating it into its
own launch serialises what was previously parallel.

**Verdict**: pool kernel separation is a structural lateral move on
its own (kept the implementation, gated by
`PRIMUS_V4_ATTN_BWD_HCA_POOL`). The split only becomes a win once
paired with the smaller-tile dKV (attempt 4) where `BLOCK_N=16` makes
the original in-kernel pool branch even worse.

## Attempt 2 — MQA head-split parallelism for dKV

Added `NUM_HEAD_GROUPS` constexpr to `_v4_attention_bwd_dkv_kernel`
(MQA path only). With 64 heads the launcher splits the head loop into
`HEAD_Q / NUM_HEAD_GROUPS` chunks, lifting the dKV grid from
`(128, 1) = 128` programs to `(128, 1, 8) = 1024` programs (with
NUM_HEAD_GROUPS=8). Each program owns a unique slice of query heads
but they all collide on the MQA dK/dV slice, so writes use
`tl.atomic_add` instead of `tl.store` when NUM_HEAD_GROUPS > 1.

Sweep `NUM_HEAD_GROUPS ∈ {1, 2, 4, 8, 16}` on dense (cr=0) BWD with
`num_warps=8 num_stages=1` (original launch params):

| NUM_HEAD_GROUPS | dense BWD median |
|---:|---:|
| 1 | 7.68 ms |
| 2 | 7.67 ms |
| 4 | 7.69 ms |
| 8 | 7.67 ms |
| 16 | 7.69 ms |

**Verdict**: head-split is essentially neutral on MI355 at H=64. The
local dKV kernel is **not parallelism-limited** at the original launch
config — adding programs and atomic_add overhead cancels out the
parallelism gain. Kept the constexpr for future re-tuning but default
to 1 (no atomics).

## Attempt 3 — Launch parameter tuning (`num_warps`, `num_stages`)

Re-bench the original kernel with `num_warps ∈ {4, 8, 16}` and
`num_stages ∈ {1, 2, 3}`:

### Dense (cr=0) BWD — dKV kernel sweep

| num_warps | num_stages | dense BWD median |
|---:|---:|---:|
| 4 | 1 | **5.99 ms** |
| 4 | 2 | 6.82 ms |
| 4 | 3 | 8.37 ms |
| 8 | 1 | 7.67 ms (orig default) |
| 8 | 2 | 8.32 ms |
| 8 | 3 | 8.45 ms |
| 16 | 1 | 14.10 ms |
| 16 | 2 | 14.90 ms |

**Sweet spot**: dKV `num_warps=4 num_stages=1`. The original
``num_warps=8`` over-occupies VGPR/AGPR per program and starves the
SIMDs; ``num_warps=4`` halves the VGPR per program, doubling
occupancy.

### HCA (cr=128) BWD — dQ + pool kernel sweep (with dKV nw=4 ns=1)

| dQ (nw, ns) | pool (nw, ns) | HCA BWD median |
|---|---|---:|
| (4, 2) | (4, 2) | **8.56 ms** |
| (4, 2) | (4, 1) | 8.60 ms |
| (4, 1) | (4, 1) | 9.08 ms |
| (4, 1) | (8, 1) | 9.16 ms |
| (8, 1) | (4, 1) | 10.65 ms |

**Sweet spot (initial)**: dQ `(4, 2)`, dKV `(4, 1)`, pool `(4, 2)`.

## Attempt 4 — Tile size (`BLOCK_M`, `BLOCK_N`)

Sweep BLOCK_M / BLOCK_N (with the attempt-3 launch params):

| BLOCK_M | BLOCK_N | dense BWD median |
|---:|---:|---:|
| 16 | 16 | 3.92 ms |
| 16 | 32 | 3.88 ms |
| 16 | 64 | 10.39 ms |
| 32 | 16 | 3.37 ms |
| 32 | 32 | 4.62 ms (attempt-3 default) |
| 32 | 64 | 9.69 ms |
| **64** | **16** | **3.16 ms** |
| 64 | 32 | 5.13 ms |
| 64 | 64 | 7.19 ms |
| 128 | 16 | 8.80 ms |

**Sweet spot**: `BLOCK_M=64 BLOCK_N=16`. Three reasons:
1. ``BLOCK_N=16`` doubles the dKV grid from 128 → 256 programs (full
   MI355 occupancy at H=64 MQA);
2. ``BLOCK_M=64`` keeps the per-program work balanced (64 heads × 4
   m-blocks per local n-block);
3. The (64 × 16) output tile lays out cleanly across two MFMA
   ``32x16x16`` tiles per warp, reducing scheduling overhead.

Pool kernel `BLOCK_M` decoupled from dKV via
`PRIMUS_V4_ATTN_BWD_POOL_BLOCK_M` (defaults to dKV ``BLOCK_M=64`` —
keeps atomic contention to 64-way for the pool dK/dV tile).

## Attempt 5 — Re-tune pool kernel for new BM/BN

With the new ``BM=64 BN=16`` defaults, re-sweep the pool kernel:

| pool (BM, nw, ns) | HCA BWD median |
|---|---:|
| (32, 4, 1) | 6.88 ms |
| (32, 4, 2) | 6.80 ms |
| (32, 8, 1) | 6.90 ms |
| **(64, 4, 1)** | **5.10 ms** |
| (64, 4, 2) | 6.13 ms |
| (64, 8, 1) | 6.67 ms |

**Sweet spot**: pool kernel ``BM=64 num_warps=4 num_stages=1``. The
attempt-3 ``num_stages=2`` was the right call when dKV was at
``num_warps=8``; with the SMEM/VGPR pressure budget freed by dKV
``num_warps=4``, the pool kernel ``num_stages=1`` is now optimal.

## Attempt 6 — Skip boundary masks via EXACT_TILES_M/N constexpr

Add constexpr flags so the launcher can tell the kernels to skip the
``tl.where(offs_m < seqlen_q, ...)`` / ``tl.where(offs_n < seqlen_k, ...)``
boundary masks when ``Sq % BLOCK_M == 0`` and ``Sk % BLOCK_N == 0``.
At V4-Flash widths both are exact.

Result: HCA stays at 5.14 ms, dense at 3.18 ms — Triton already DCEs
most of the redundant work, so the explicit constexpr is a minor
codegen tidy-up rather than a perf win. Kept the change because it
makes the inner loop minimal for production widths and is correctness-
preserving for non-divisible widths.

## Final result (after attempts 1+2+3+4+5+6)

| Mode | Baseline | Final | Delta | Speedup |
|---|---:|---:|---:|---:|
| cr=0 dense BWD | 7.66 ms | **3.17 ms** | -4.49 ms | **2.42×** |
| cr=128 HCA BWD | 11.90 ms | **5.14 ms** | -6.76 ms | **2.31×** |

- Parity: 33 / 33 PASS (32 skipped, baseline has same 33/32 split).
- cr=0 BWD regression budget: -4.49 ms (well within 0.5 ms budget — net
  **improvement**).
- cr=128 target was ≤ 3.0 ms (3.96×). Achieved 2.31×; the remaining
  ~2 ms gap to the 3 ms target is in the dKV / pool kernel compute
  efficiency (currently ~30% of MFMA peak vs the original ~10%, but
  the kernel is now memory- and atomic-bound on the pool path, not
  parallelism-bound).

## Per-kernel breakdown (HCA, BM=64 BN=16)

| Component | Wall time | Notes |
|---|---:|---|
| `_v4_attention_bwd_preprocess_kernel` (D scalar) | < 0.1 ms | unchanged |
| `_v4_attention_bwd_dq_kernel` | ~1.0 ms | nw=4 ns=2; HCA adds 1 pool iter (~0.2 ms over dense) |
| `_v4_attention_bwd_dkv_kernel` (local SWA) | ~2.4 ms | nw=4 ns=1; 256 programs × 4 m-blocks × 64 heads |
| `_v4_attention_bwd_dkv_pool_kernel` | ~1.7 ms | nw=4 ns=1; 64 programs × 64 heads, atomic_add at end |
| **HCA total** | **5.14 ms** | dq + dkv + pool sequential |

## Future optimisation directions (not done in P57)

- Run dKV local + pool kernels on different CUDA streams to overlap
  (tested — added ~4 ms because per-call stream allocation and the
  HIP runtime serialised the launches; would need a cached stream).
- Fuse pool work into the dKV kernel via an extra grid dim (would
  need atomic_add on the local cells too — net trade-off unclear).
- Persistent dKV kernel that iterates multiple n-blocks per program to
  amortise K/V loads (low priority — K/V load is already L2-cached
  across the 256 dKV programs in the current design).
- Hand-tuned MFMA selection / k_pack / matrix operand fetch via the
  AMD-specific Triton dialect.
