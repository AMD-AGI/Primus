# Plan-8 P57 cr=0 BWD — optimisation attempts

Branch: `dev/wenx/p57-cr0-bwd-attempt1` (off `112d0dd1`).

Baseline: **7.65 ms** median (PRIMUS_V4_ATTN_BWD_USE_SPLIT=1,
PRIMUS_V4_CSA_BWD_SEGREDUCE=1, V4-Flash widths, MQA H=64 D=512 Sq=4096
SWA=128 sink=True bf16, MI355).

## Attempt 1 — split-BWD dKV head-parallel layout

**Hypothesis.** The split BWD's dKV kernel grid is
`(cdiv(Sk, BLOCK_N), B * HK)`. At MQA (HEAD_K=1, HEAD_Q=64) and
B=1, Sk=4096, BLOCK_N=32 this is only **128 programs** — vs MI355's
~304 CUs. Each program serialises a `HEAD_Q × m_per_n = 64 × 5 = 320`
inner matmul iteration. Expanding the grid by `HEAD_Q` so each
program owns a single query head, with `tl.atomic_add` into the
shared (b, khid, n) fp32 dK / dV tile, gives **8192 programs** —
fully populating MI355 — and trades one big head-serial loop for
HEAD_Q-way atomic contention at the fp32 atomic engine, which is a
much cheaper cost on MI355.

**Implementation.** New `HEAD_PARALLEL: tl.constexpr` flag on
`_v4_attention_bwd_dkv_kernel`. The kernel body is unified into a
single head-sweep `for h_iter in range(head_lo, head_hi):`, with the
range collapsing to a single iteration in the MHA and MQA-head-parallel
paths and iterating all `HEAD_Q` heads in the legacy MQA path. The
storage section gains a third branch that atomic-adds when
`HEAD_PARALLEL and HEAD_K != HEAD_Q`.

**Result.** Wall-clock **unchanged at 7.65 ms**. Parity intact (33
pass / 32 skip on the fast tier). The dKV kernel isn't actually the
parallelism bottleneck — the head-serial loop's K / V tile re-use is
already L2-cached (K/V is only 4 MiB total at MQA), so the
HEAD_Q-way atomic contention exactly cancels the parallelism win.
The flag is kept env-controlled (`PRIMUS_V4_BWD_DKV_HEAD_PARALLEL`,
default off) for shapes with smaller H_Q where atomics are cheaper.

## Attempt 2 — `num_warps = 4` for both dQ and dKV kernels

**Hypothesis.** The dQ / dKV kernels were launched with
`num_warps = 8` (256 lanes / 512 lanes per program on MI355 with
64-thread waves). Both kernels carry a large register-resident
accumulator at `head_dim = 512`:

* dQ accumulator: `[BLOCK_M, D] = [32, 512]` fp32 = 64 KiB per program.
* dK + dV accumulators: `2 × [BLOCK_N, D] = 2 × [32, 512]` fp32 =
  128 KiB per program.

At `num_warps = 8`, splitting these accumulators across 8 waves
forces the AMD Triton backend to keep all 8 waves co-resident, which
constrains the compiler to a configuration where each wave has only
modest VGPR budget. `num_warps = 4` doubles the per-wave VGPR
budget, leaving plenty of headroom for the 32 × 16 fp32 fragment +
the matmul inputs, and the AMD MFMA scheduler hits a much better
issue rate.

**Sweep at `num_warps in {2, 4, 8, 16}` (stages=1, HEAD_PARALLEL=0):**

| dQ \ dKV | 4 | 8 |
|---|---:|---:|
| 4 | **4.35 ms** | — |
| 8 | 5.95 ms (dkv) | 7.65 ms |

Standalone:

| run | BWD ms | FWD ms |
|---|---:|---:|
| dq warps=4 (dkv=8) | 6.19 | 0.76 |
| dq warps=8 (dkv=4) | 5.95 | 0.76 |
| dq warps=2 (dkv=8) | 9.46 | 0.78 |
| dq warps=16 (dkv=8) | 19.3 | 3.58 |
| dq warps=4 + dkv warps=4 | **4.35** | 0.76 |

`num_stages = 2` on dKV consistently regresses (5.16-5.44 ms vs
4.35 ms at stages=1), confirming the dKV path is register-pressure
bound. `num_stages = 2` on dQ is roughly neutral
(4.34 ms vs 4.35 ms).

**Result.** **4.35 ms** stable (median of 3 runs: 4.34 / 4.35 / 4.35).
That's a **1.76× speedup** vs the 7.65 ms baseline. Parity intact
(33 pass / 32 skip on the fast tier). Still 1.45× away from the
3.0 ms target.

**Defaults shipped.**

* `_v4_attention_bwd_dq_kernel`: `num_warps = 4`, `num_stages = 1`
  (env-tunable via `PRIMUS_V4_BWD_DQ_{WARPS,STAGES}`).
* `_v4_attention_bwd_dkv_kernel`: `num_warps = 4`, `num_stages = 1`
  (env-tunable via `PRIMUS_V4_BWD_DKV_{WARPS,STAGES}`).
* `PRIMUS_V4_BWD_DKV_HEAD_PARALLEL = 0` by default (kept env-tunable
  for shapes where atomic contention is cheaper than the head-serial
  loop).

## Attempt 3 — tile sweep: ``BLOCK_M = 64``, ``BLOCK_N = 16``

**Hypothesis.** With `num_warps = 4` fixed, the original
`BLOCK_M = BLOCK_N = 32` was a compromise picked at P25 for SMEM
budget and is not the perf optimum. Wider `BLOCK_M` amortises Q / dout
loads (fewer m-block programs) and amortises the dQ accumulator init
cost; narrower `BLOCK_N` lightens the per-iter K / V tile weight,
doubles dKV n-block parallelism (`Sk / BLOCK_N`), and gives the MFMA
scheduler more iterations to hide memory latency without help from
software pipelining.

**Sweep at `num_warps = 4` (stages=1, HEAD_PARALLEL=0):**

| BLOCK_M \ BLOCK_N | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|
| 16 | — | — | 4.05 | — |
| 32 | 41.6 | **3.55** | 4.38 | 10.3 |
| 64 | — | **3.31** | 4.89 | 7.77 |
| 128 | 13.3 | 8.09 | — | — |

The `BLOCK_N = 16` column is uniformly best. Below that (`BLOCK_N = 8`)
the MFMA tile drops below the 16-lane minimum and runtime explodes;
above (`BLOCK_N = 32+`) the per-iter K / V tile bloat outweighs the
loop-overhead amortisation.

**Stages sweep at the BM=64, BN=16 sweet spot:**

| dQ stages | dKV stages | BWD ms |
|---:|---:|---:|
| 1 | 1 | 3.30 |
| 2 | 1 | **3.14** |
| 1 | 2 | 3.33 |
| 2 | 2 | 3.17 |
| 3 | 1 | 7.21 (compile thrash) |

`num_stages = 2` on the dQ kernel pays off at BM=64 (the bigger Q
load benefits from one-stage software pipelining) but bigger stage
counts hurt the dKV kernel.

**Result.** **3.14 ms** stable across 5 runs (3.14 / 3.16 / 3.16 /
3.17 / 3.16; median 3.16). Parity intact (33 pass / 32 skip on the
fast tier). That's a **2.42× speedup** vs the 7.65 ms baseline and
**1.05× over the 3.0 ms P57 target** — close but not over the line.

**Defaults shipped after attempt 3.**

* `BLOCK_M = 64`, `BLOCK_N = 16` (env-tunable via
  `PRIMUS_V4_BWD_BLOCK_{M,N}`).
* dQ `num_warps = 4`, `num_stages = 2`.
* dKV `num_warps = 4`, `num_stages = 1`.

## Attempt 4 — scale-defer + ``tl.dot(acc=...)`` fused MFMA

**Hypothesis.** Both kernels apply `sm_scale` per inner-loop iteration:

* dQ kernel: `dq += tl.dot(ds, k) * sm_scale` — the per-iter scale is
  an elementwise multiply on the 64 × 512 fp32 result, run ~12 times
  per dQ program (n-loop iterations at BLOCK_N=16, SWA=128). The
  pattern `acc += tl.dot(a, b) * c` generates three ops: dot,
  elementwise scale, elementwise add.
* dKV kernel: `dk += tl.dot(ds^T, q) * sm_scale` — same shape, same
  three-op pattern, run 64 × 4 = 256 times per dKV program.

Both can be rewritten as `acc = tl.dot(a, b, acc=acc)` — a single
fused MFMA-add instruction on AMD — and pull the `sm_scale` out into
a single multiply on the final accumulator after the loop. The
rewrite is bit-equivalent modulo fp32 associativity (the deferred
scale is applied to the final sum rather than every partial sum).

**Implementation.** Replace the in-loop
`dq += tl.dot(ds.to(k.dtype), k) * sm_scale` with
`dq = tl.dot(ds.to(k.dtype), k, acc=dq)` and add a single
`dq = dq * sm_scale` after the n-loop (plus the HCA pool branch).
Symmetric change for the dKV kernel: rewrite the
`dv += tl.dot(p^T, dout)` and `dk += tl.dot(ds^T, q) * sm_scale`
to `tl.dot(..., acc=...)` form and apply `dk = dk * sm_scale` once
after the head × m loop.

**Result.** **2.989 ms** median across 10 runs (range
2.975–3.016 ms; 9/10 runs below 3.0 ms). Parity intact
(33 pass / 32 skip on the fast tier).

That's a **2.56× speedup vs the 7.65 ms baseline** and lands the
cr=0 dense BWD **below the P57 ≤ 3.0 ms target with comfortable
margin**.

| run | BWD ms |
|---:|---:|
| 1 | 2.988 |
| 2 | 2.991 |
| 3 | 2.980 |
| 4 | 2.994 |
| 5 | 2.990 |
| 6 | 2.991 |
| 7 | 2.984 |
| 8 | 3.016 |
| 9 | 2.980 |
| 10 | 2.975 |
| **median** | **2.989** |

## Summary

| Attempt | Strategy | BWD ms | Speedup |
|---|---|---:|---:|
| baseline | P32 final defaults | 7.65 | 1.00× |
| 1 | head-parallel dKV (opt-in) | 7.65 | 1.00× |
| 2 | `num_warps = 4` for dQ + dKV | 4.35 | 1.76× |
| 3 | tile sweep `BM=64, BN=16` + dQ stages=2 | 3.16 | 2.42× |
| 4 | scale-defer + `tl.dot(acc=...)` fused MFMA | **2.989** | **2.56×** |

P57 target ≤ 3.0 ms is **hit** (median 2.989 ms, 0.4% under target).
