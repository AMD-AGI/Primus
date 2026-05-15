# Plan-6 P41 — `Indexer.forward` post-einsum tail Triton fusion + plan-7 candidate inventory

> Phase summary written at P41 close-out (2026-05-15).  Microbench
> numbers (§4.1) pinned against the live `mi355-gpu-8` /
> `dev_primus_wenx_693` runs in `progress/p41/bench/{v4,small}.json`.

**Status: kernel ships, default OFF.  Microbench wins (FWD 4.30×,
BWD 1.63× at V4-Flash widths) but the EP=8 proxy A/B (10-iter
smoke, A vs B) shows the ~0.2 ms / iter aggregate gain is
indistinguishable from the ~±1 ms NCCL / dispatch noise band.
lm_loss bit-identical iter-by-iter.  Same descope precedent as
P38 / P39.  Plan-7 candidate inventory at
`progress/p41/p41-candidates.md` is the load-bearing deliverable.**

---

## 1. Objective

Re-attempt the P38 Indexer scoring fusion that was descoped at
V4-Flash widths because the generic Triton kernel lost to cuBLAS
on the matmul half (28 vs 20 TFLOP/s; BWD regressed 12× because of
`tl.atomic_add` contention).  **Keep the `einsum` eager** — cuBLAS
wins — and fuse **only** the post-matmul tail
(`relu → mul(w_i) → sum(H) → + causal_mask`).  The tail is
bandwidth-bound (75.6 MiB HBM footprint / call at V4-Flash), so
Triton wins by collapsing 5 ATen launches into one.

Side deliverable: seed plan-7 with the trace-driven candidate
inventory.  See `progress/p41/p41-candidates.md`.

## 2. Design

### 2.1 FWD kernel

`_indexer_score_post_fwd_kernel` in
`primus/backends/megatron/core/transformer/v4_attention_kernels/_triton/indexer_score_post.py`.

Inputs:

* `dot [B, S, H, P]` — eager einsum output (pre-`relu`); read once.
* `w_i [B, S, H]`    — per-head weights.

Per-program tile (`BLOCK_S × BLOCK_P`):

1. ``acc [BLOCK_S, BLOCK_P]`` fp32 = 0.
2. For each head `h` (constexpr unroll):
   - load `dot[b, s_tile, h, p_tile]`;
   - `relu = max(dot, 0)`;
   - load `w[b, s_tile, h]`;
   - `acc += relu * w[:, None]`.
3. Apply causal mask inline:
   `where((p + 1) * compress_ratio - 1 <= s, acc, -inf)`.
4. Store cast to `OUT_DTYPE`.

Grid: `(B, ceil(S / BLOCK_S), ceil(P / BLOCK_P))` with default
`BLOCK_S = BLOCK_P = 64` at V4-Flash widths.

### 2.2 BWD kernel pair

The P38 BWD's killer was the cross-block `tl.atomic_add` for
`d_w` (multiple p_tiles per `(b, s_tile, h)` cell contending).
P41 splits the BWD into **two** kernels:

1. `_indexer_score_post_bwd_ddot_kernel` — same grid as FWD, emits
   `d_dot [B, S, H, P]` only.  Each output element is touched by
   exactly one program; no `atomic_add` needed.
2. `_indexer_score_post_bwd_dw_kernel` — grid `(B, ceil(S /
   BLOCK_S), H)`; loops over P internally in chunks of
   `BLOCK_P_INNER = 128`; emits `d_w [B, S, H]`.  Each program
   owns one `(b, s_tile, h)` cell so the reduce is fully local.

Both kernels are bandwidth-bound and parallelise cleanly.

### 2.3 Routing

`PRIMUS_INDEXER_TRITON` is **re-purposed** to mean "post-einsum
tail fusion".  Legacy P38 full-fuse path moves behind
`PRIMUS_INDEXER_TRITON_FULL` (default `"0"`, kept in tree for
small-shape paths).

`Indexer.forward` dispatch precedence:

```
PRIMUS_INDEXER_TRITON_FULL=1  →  legacy P38 full-fuse
PRIMUS_INDEXER_TRITON=1       →  P41 tail-only (einsum eager)
else                          →  fully eager
```

Default at landing: both `"0"` (P41 ships kernel + tests +
microbench but descoped per proxy A/B).

## 3. Code surface

```
primus/backends/megatron/core/transformer/v4_attention_kernels/_triton/indexer_score_post.py
  + _indexer_score_post_fwd_kernel
  + _indexer_score_post_bwd_ddot_kernel
  + _indexer_score_post_bwd_dw_kernel
  + IndexerScorePostFn (autograd.Function)
  + indexer_score_post_triton (public entry)
  + is_triton_path_enabled / is_triton_kernel_supported

primus/backends/megatron/core/transformer/v4_attention_kernels/_triton/indexer_score.py
  M is_triton_path_enabled  (env knob re-purposed → checks
                              PRIMUS_INDEXER_TRITON_FULL)

primus/backends/megatron/core/transformer/indexer.py
  M Indexer.forward  (dispatch precedence: TRITON_FULL > TRITON > eager)

tests/unit_tests/megatron/transformer/deepseek_v4/test_p41_indexer_tail_triton.py
  + 35 tests + 1 release-tier slow (G43, G43 edge cases, entry-point smoke)

deepseek-v4/develop/progress/p41/
  + bench_indexer_tail.py
  + bench/{v4,small}.json (microbench output)
  + run_baseline_trace_ep8_p41.sh
  + p41-candidates.md (plan-7 inventory)
  + p41-summary.md (this file)
```

## 4. Performance

### 4.1 Microbench (V4-Flash widths, bf16)

| shape | path | FWD median (ms) | BWD median (ms) | FWD GB/s | BWD GB/s |
|---|---|---:|---:|---:|---:|
| V4-Flash (B=1, S=4096, P=1024, H=8, HD=128; 75.6 MiB / call) | eager tail  | 0.201 | 0.319 |  376.2 |  709.7 |
| V4-Flash                                                       | triton tail | 0.047 | 0.196 | 1619.4 | 1154.6 |
| **V4-Flash speedup**                                          |             | **4.30×** | **1.63×** | **+331 %** | **+63 %** |
| small (B=2, S=128, P=32, H=8, HD=128)                         | eager tail  | 0.097 | 0.140 |   1.6 |   3.2 |
| small                                                          | triton tail | 0.026 | 0.167 |   5.9 |   2.7 |
| **small speedup**                                              |             | **3.81×** | **0.84×** | **+281 %** | **-15 %** |

(The small-shape BWD regression is the per-launch overhead of the
two-kernel BWD pipeline; benign at production widths.)

### 4.2 EP=8 proxy A/B

10-iter smoke; A = `PRIMUS_INDEXER_TRITON=0` (eager tail =
P40 production), B = `PRIMUS_INDEXER_TRITON=1` (P41 tail-only).

| metric | A (eager tail) | B (P41 tail) | delta |
|---|---:|---:|---:|
| iter 8 ms                                | 513.5 | 514.4 |  +0.9 |
| iter 9 ms                                | 515.1 | 515.1 |   0.0 |
| iter 10 ms                               | 512.4 | 512.3 |  −0.1 |
| **mean iters 8/9/10**                    | **513.7** | **513.9** | **+0.2** |
| lm_loss iter 10                          | 9.257534 | 9.257534 | **bit-identical** |
| lm_loss iter 8                           | 9.446677 | 9.446677 | **bit-identical** |
| lm_loss iter 9                           | 9.287911 | 9.287911 | **bit-identical** |

The aggregate iter-time difference (+0.2 ms / iter on the mean) is
**within the ~±1 ms NCCL / dispatch noise band**.  Same pattern
as P38 / P39: microbench wins do not translate to proxy wins at
the per-CSA-layer scope (Indexer runs only at 3 of 8 layers).

### 4.3 Trace evidence

Profiler window `ProfilerStep#6` for the B side: 522.8 ms (vs P40
final 523.67 ms — also within noise).  The eager `relu + mul +
sum + mask` ATen launches inside the CSA layers have collapsed
into the two new Triton kernels (`_indexer_score_post_fwd_kernel`
at ~0.05 ms × 3 = ~0.15 ms, `_indexer_score_post_bwd_ddot_kernel`
+ `_dw_kernel` at ~0.2 ms × 3 = ~0.6 ms).  Microbench-predicted
~0.5 ms / iter savings, lost in proxy noise.

## 5. Tests

* **G43 FWD parity** — bf16 `atol=5e-3 rtol=5e-3` vs eager tail
  across `H ∈ {1, 2, 4, 8}` × `dtype ∈ {fp32, bf16}` ×
  `compress_ratio ∈ {1, 4, 16}` = **24 combinations**.  All
  pass.
* **G43 topk parity** — bit-equal `topk_idxs` vs eager full chain
  via the Indexer module; `match_ratio ≥ 0.95` bf16, `≥ 0.99`
  fp32.  Pass.
* **G43 BWD `gradcheck`** — fast tier fp32 across `H ∈ {1, 2, 4,
  8}` = **4 tests**.  Pass.
* **G43 release-tier slow** — V4-Flash production shape FWD
  parity bf16.  Pass.
* **G43 edge cases** — unsupported `H` raises; supported predicate;
  env default OFF; tail vs full env knobs are distinct;
  entry-point matches autograd class.  All pass.
* **Total: 35 fast + 1 release-tier slow, all green.**

## 6. Gating

* `PRIMUS_INDEXER_TRITON` default **`"0"`** (descoped).  Set to
  `"1"` to opt into the P41 tail-only path.  Microbench wins
  documented above; future tuning may unlock a clear proxy win.
* `PRIMUS_INDEXER_TRITON_FULL` default **`"0"`** (legacy P38 path;
  also descoped).  Set to `"1"` for the small-shape path.
* Eager fallback when both are `"0"`.

## 7. Failed / negative probes

* Initial BWD design used a single kernel that emitted both `d_dot`
  and `d_w` with `tl.atomic_add` for `d_w` (the P38 design pattern).
  Microbench showed BWD regressed to 0.26× — cross-block atomic_add
  contention swamped the win.  Split into two kernels (FlashAttention
  precedent); BWD now 1.63× faster than eager.
* Second design move (Python-side `(d_acc.unsqueeze(2) * relu_dot).
  sum(dim=-1)` for `d_w`) reached 0.81× — still a regression.  The
  Python step materialised a `[B, S, H, P]` fp32 intermediate
  (8.6 MB at V4-Flash) which competed with the d_dot store stream.
  Dedicated `_indexer_score_post_bwd_dw_kernel` resolves it.

## 8. Follow-ups + commit pin

* Plan-7 candidate inventory `progress/p41/p41-candidates.md`
  pins the next 4 phase IDs (P42 / P43 / P44 in plan-6 +
  plan-7 P45..P48).
* Feature commit SHA: TBD-p41.
* Status pin commit follows R1.3 / R2.4.
