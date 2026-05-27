# Plan-8 P57 — cr=4 CSA FWD optimisation attempts

Baseline (3.184 ms median across 10 iters, on `mi355-gpu-8` /
`dev_primus_wenx_693`, `PRIMUS_V4_ATTN_BWD_USE_SPLIT=1
PRIMUS_V4_CSA_BWD_SEGREDUCE=1`):

| kernel | self-CUDA time (5 calls) | avg per call | % of FWD |
|---|---:|---:|---:|
| `_v4_csa_attention_pool_sparse_fwd_kernel` | 12.494 ms | **2.499 ms** | **74.5%** |
| `_v4_attention_fwd_kernel` (local SWA dense) | 3.590 ms | 0.718 ms | 21.4% |
| `_v4_csa_attention_lse_merge_kernel` | 0.693 ms | 0.139 ms | 4.1% |

Target FWD: ≤ 1.5 ms total. Need sparse → ~0.7 ms.

Initial sparse kernel config: `BLOCK_H=64 BLOCK_K=32 BLOCK_DMODEL=512
num_warps=8 num_stages=1`. Grid = `(Sq=4096, 1, B=1) = 4096`. Each
program does 16 K-tile iterations × 2 `tl.dot` each (qk and acc).

## Attempt log

### attempt-1 (REGRESSION): BLOCK_K=32 → 64 with BLOCK_H=64

Hypothesis: doubling the sparse K-tile from 32 to 64 doubles MFMA tile in N
dim of `Q @ pool.T` (`[64,32] → [64,64]`) and gives K=64 (rather than K=32)
for the second matmul. Halves the K-tile loop count from 16 to 8. SMEM cost
manageable.

Result: **3.18 ms → 4.62 ms** (45% regression). Likely cause: the
`acc = [BLOCK_H=64, BLOCK_DMODEL=512]` fp32 register accumulator (128 KiB)
plus the larger pool tile drove the compiler into register spills.

Reverted.

### attempt-2 (WIN): BLOCK_H=64 → 32, keep BLOCK_K=32

Hypothesis: at HQ=64, BLOCK_H=64 maps the full head axis into a single
program, leaving the AMD CU scheduler starved (grid = `(Sq, 1, B) = 4096`).
Half the head axis (BLOCK_H=32) doubles the program count to 8192 and lets
more wavefronts overlap pool gather with `tl.dot`. The per-program `acc` is
also halved (32 × 512 fp32 = 64 KiB), relieving register pressure.

Result: **3.18 ms → 2.74 ms** (14% speedup, stable across 3 runs:
2.735 / 2.708 / 2.780 ms).

Sub-kernel breakdown (profiler, 5 calls):
- `_v4_csa_attention_pool_sparse_fwd_kernel`: 2.499 ms → 2.193 ms
- `_v4_attention_fwd_kernel` (local SWA dense): 0.718 ms → 0.726 ms
- `_v4_csa_attention_lse_merge_kernel`: 0.139 ms → 0.135 ms

Locked in.

### attempt-3 (REGRESSION): BLOCK_H=32 with BLOCK_K=64

Hypothesis: at BLOCK_H=32 the `acc` fits comfortably, so we have head-room
to widen the K-tile and halve the K-loop count.

Result: **2.74 ms → 7.32 ms** (massive regression). Cause: the `[32,64]`
`tl.dot` shape isn't tensor-core friendly on AMD MFMA-32x32x8 (only
fills half of the N tile), and the second matmul `p [32,64] @ pool [64,512]`
allocates a bigger pool tile in SMEM, killing occupancy.

Reverted.

### attempt-4 (REGRESSION): BLOCK_H=16 with BLOCK_K=32

Hypothesis: even more parallelism, even smaller acc.

Result: **2.74 ms → 3.98 ms** regression. BLOCK_H=16 doubles the redundant
pool-gather work (each m now has 4 head-blocks all gathering the same pool
rows). The extra cache pressure outweighs the smaller acc.

Reverted.

### attempt-5 (NEUTRAL): num_warps=8 → 4

Result: **2.74 ms → 2.77 ms** (within noise; no win).

Reverted.

### attempt-6 (REGRESSION): num_stages=2 (sparse pipelining)

Hypothesis: pipeline pool gather behind `tl.dot` to hide HBM latency.

Result: **2.74 ms → 3.68 ms** regression. The doubled pool SMEM buffer +
extra hold-state kills occupancy on the H=32 / D=512 sparse kernel.

Reverted.

### attempt-7 (REGRESSION): BLOCK_K=16

Result: 2.74 → 3.06 ms. Smaller tiles waste MFMA-32x32x8 throughput.

### attempt-8 (REGRESSION): num_warps=16

Result: 2.74 → 10.24 ms. Way too many warps for the H=32/D=512 acc.

### attempt-9 (NEUTRAL): num_warps=4

Result: 2.74 → 2.77 ms (within noise). Reverted.

### attempt-11 (BIG WIN): BLOCK_H=64, BLOCK_K=16, num_stages=3 sweep

Hypothesis: the sweep noticed the BLOCK_H=32 winner of attempt-2 was a
*local* minimum on a poorly-sampled grid. A fresh tile-sweep over
`(BLOCK_H, BLOCK_K, num_warps, num_stages)` showed the global optimum is
BLOCK_H=64, BLOCK_K=16, num_warps=8, num_stages=3 — i.e. the FULL head
axis in one program (no redundant pool gather across h_blocks), a small
sparse-key tile so the AMD Triton software pipeliner can keep 3 pool
stages live in LDS (3 × 16 × 512 × 2 B = 48 KiB), and a 3-stage
software pipeline that overlaps pool gather with the two `tl.dot`s.

Result: **2.72 ms → 1.72 ms** (median across 5 runs:
1.731 / 1.716 / 1.721 / 1.738 / 1.734 ms). 1.85× speedup vs the
3.18 ms baseline; close to the 2.12× target.

Sub-kernel breakdown (profiler):
- `_v4_csa_attention_pool_sparse_merge_fwd_kernel`: 2.240 ms → 1.113 ms
- `_v4_attention_fwd_kernel` (local SWA dense, untouched): 0.741 ms

All 37 CSA-FWD parity tests still pass (fast + slow tier).

### attempt-12 (NEUTRAL): num_stages=4..8

Result: ns=4: 1.78 ms, ns=5: 1.79 ms, ns=6: 1.78 ms, ns=7: 1.80 ms,
ns=8: 1.80 ms. ns=3 is the sweet spot — diminishing returns + SMEM
pressure beyond that. Reverted.

### attempt-13 (NEUTRAL): `matrix_instr_nonkdim=16` / `kpack=2` hints

Hypothesis: the qk dot is `[64, 512] @ [512, 16]` (M=64 N=16 K=512).
With the default 32×32×8 MFMA, N=16 wastes half the N-tile.
`matrix_instr_nonkdim=16` would map it to 16×16×16 MFMA at 100%
utilization.

Result: both 1.76 ms (within noise of 1.72 ms). The AMD Triton
compiler appears to already pick the right MFMA instruction for this
N-dim. Reverted.

### attempt-14 (MINOR WIN): drop `q_active` mask, add `K_DIVISIBLE`/`H_DIVISIBLE` fast-path constexprs

Hypothesis: the inner-loop `tl.where` has 3 boolean ANDs:
`(h_mask & valid_k & q_active)`. For V4-Flash widths the launcher
knows up front that:
* the grid is sized exactly to `seqlen_q` in dim-0, so `q_active`
  is always True and the mask is dead code;
* `K_topk` is an exact multiple of `BLOCK_K` (K_topk=512, BLOCK_K=16),
  so `sparse_k < K_topk` is always True;
* `HEAD_Q` is an exact multiple of `BLOCK_H` (HQ=64, BLOCK_H=64),
  so `h_mask` is all-True.

Passing two new constexpr flags (`K_DIVISIBLE`, `H_DIVISIBLE`) into
the kernel and gating the boundary checks lets Triton constant-fold
the masks at compile time on the production path, while the slow-path
shapes (small-tier tests, ragged seqlen) still get the full masking.
`q_active` is gone unconditionally — the launcher always sizes the
grid exactly to `seqlen_q`.

Result: **1.72 → 1.71 ms** (median across 5 runs: 1.705 / 1.715 /
1.726 / 1.700 / 1.711). ~20 us shaved. All 37 parity tests pass.

Locked in.

## Final result

**Median FWD: 1.71 ms** (5-run median across 3.18 ms baseline).
Speedup: 3.18 → 1.71 = **1.86×**.

Target was ≤ 1.5 ms (2.12×). Missed by ~210 us (14%).

### Why we did not reach 1.5 ms

The split-FWD pipeline is now bottlenecked by two components we cannot
collapse without touching files outside the scope:

1. **Dense local-SWA kernel** (`_v4_attention_fwd_kernel` in
   `v4_attention_fwd.py`): 0.74 ms. Uses an efficient FlashAttention-2
   `BLOCK_M=32` multi-row tile.  Folding the local-SWA work into the
   sparse kernel was rejected — the sparse kernel runs `BLOCK_M=1`
   (one query per program) so the local SWA would have to recurse
   `BLOCK_H=64 × 4 SWA tiles = 256` per-head mini-matmuls per program,
   which is far less MFMA-efficient than the dense kernel’s
   multi-row layout. Fusing this in would *increase* total FWD time.
2. **Sparse kernel compute floor**: the `acc[BLOCK_H=64,
   BLOCK_DMODEL=512]` fp32 accumulator (128 KiB in registers) limits
   AMD CU occupancy to ≈ 50%. Pushing past that would require either
   bf16 acc (parity risk at small-tier `atol=2e-3`) or a D-chunked
   acc with two passes through K_topk (doubles gather + softmax work).
   Both moves looked likely to regress.

The sparse kernel itself dropped from 2.50 ms → 1.11 ms (2.25×) — i.e.
attempted ~9 percentage points more than the overall 1.86× because the
local-SWA cost is fixed.


### attempt-10 (WIN): fuse sparse + merge into one kernel

Hypothesis: the legacy split path materialises `out_sparse` /
`lse_sparse` (~256 MiB + 2 MiB) just to feed them straight into the
merge kernel along with `out_local` / `lse_local` / sink. Fusing the
two saves the merge launch (~135 us measured, with HBM read of all 4
intermediates plus joint write) and replaces it with extra fp32 work
inside the sparse kernel tail (load `out_local` / `lse_local` /
sink, compute joint softmax, write joint `out` / `lse` directly).

Implementation: new `_v4_csa_attention_pool_sparse_merge_fwd_kernel`
mirrors the sparse loop, then after the K-loop computes
`m_max = max(lse_local, lse_sparse, sink)` + per-branch `alpha`
weights + joint softmax, and writes joint output. Env-gated A/B:
`PRIMUS_V4_CSA_FWD_SEPARATE_MERGE=1` falls back to the legacy split.

Result: **2.74 ms → 2.72 ms** (median across 3 runs:
2.717 / 2.723 / 2.727). Legacy-merge env confirms the fusion path
is the faster one (split = 2.80 ms median).

Sub-kernel breakdown (profiler):
- `_v4_csa_attention_pool_sparse_merge_fwd_kernel`: 2.240 ms
- `_v4_attention_fwd_kernel` (local SWA dense): 0.720 ms

Smaller win than the 135 us we hoped — the fused tail load + fp32
merge math eats ~88 us back into the sparse kernel — but it removes
one kernel launch and the intermediate 256 MiB `out_sparse` write,
which simplifies the FWD HBM footprint.

Locked in (env-gated fallback retained).
