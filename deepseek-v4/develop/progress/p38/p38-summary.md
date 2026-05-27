# Plan-6 P38 — `Indexer.forward` scoring Triton fusion (descoped)

> Phase summary written at P38 close-out (2026-05-14). Microbench
> numbers (§4.1) pinned against the live `mi355-gpu-8` /
> `dev_primus_wenx_693` runs in
> `progress/p38/bench/{v4,small}.json`.

**Status: descoped per the explicit clause in
`plan-6/02-phase-details.md` §"Task list refinement" -- the
post-P37 trace's per-CSA-layer Indexer cost is dominated by a
cuBLAS / hipBLASLt batched-matmul that beats the generic Triton
kernel.  The kernel is checked in (with `PRIMUS_INDEXER_TRITON=0`
default) so it stays available for future tuning + small-shape
paths.**

---

## 1. Objective

Fuse the ``einsum + relu + mul + sum + causal_mask`` scoring chain
in `Indexer.forward` into one Triton FWD + one Triton BWD kernel.
Eager body:

```python
relu = F.relu(torch.einsum("bshd,bpd->bshp", q_i, k_icomp))
scores = (relu * w_i.unsqueeze(-1)).sum(dim=2)
mask = self._causal_mask(S, P, scores.device, scores.dtype)
scores = scores + mask.unsqueeze(0)
```

That chain decomposes into ~7 ATen kernels (einsum + relu + mul +
sum + mask alloc + mask add + dtype cast).  The CSA Indexer is
invoked at 3 of 8 layers in the V4-Flash 8-layer proxy slice
(`compress_ratio == 4` layers).

---

## 2. Design

The Triton kernel does:

* Per ``(b, s_tile, p_tile)`` program: load `BLOCK_S * HD` queries +
  `BLOCK_P * HD` keys + `BLOCK_S` weights once per head;
* Inner unrolled loop over heads (`H: tl.constexpr`):
    - `dot = q @ k.T` via `tl.dot`;
    - `relu` per element;
    - multiply by `w_h`;
    - accumulate into `acc [BLOCK_S, BLOCK_P]` fp32.
* After the head loop, materialise the causal mask inline
  (`tl.where((p+1)*compress_ratio - 1 <= s, acc, -inf)`) -- no
  ``[S, P]`` mask tensor allocated;
* Store the result cast to ``OUT_DTYPE``.

BWD uses the FlashAttention-style trick of **recomputing** the
per-tile `relu` mask in the backward pass (no saved-for-backward
mask state).  Per program:

* Re-load q, k, w; re-derive `dot` and `relu_mask`;
* Apply the analytic VJP per element;
* `tl.atomic_add` into `dq`, `dk`, `dw` (necessary because multiple
  programs touch the same output rows / columns).

---

## 3. Performance microbench

Bench: `progress/p38/bench_indexer.py` (`--mode {v4, small}`, raw
JSON at `progress/p38/bench/{v4,small}.json`).  `iters=10, warmup=3,
n_input_copies=4, l2_flush_mb=512`, bf16 throughout.

| shape | path | FWD median (ms) | BWD median (ms) | FWD TFLOP/s | BWD TFLOP/s |
|---|---|---:|---:|---:|---:|
| V4-Flash (B=1, S=4096, P=1024, H=8, Hd=128; 8.6 GFLOP / call) | eager  | 0.306 | 0.489 |  28.1 |  52.7 |
| V4-Flash                                                      | triton | 0.424 | 6.457 |  20.2 |   4.0 |
| **V4-Flash speedup**                                          |        | **0.72x (regression)** | **0.08x (regression)** |   |   |
| small (B=2, S=128, P=32, H=8, Hd=128)                         | eager  | 0.176 | 0.256 |   0.1 |   0.2 |
| small                                                          | triton | 0.053 | 0.226 |   0.3 |   0.2 |
| **small speedup**                                              |        | **3.35x** | **1.14x** |   |   |

### Why the Triton path loses at V4-Flash widths

1. The eager `torch.einsum("bshd,bpd->bshp", q, k)` is a batched
   matmul with `(S, P, Hd) = (4096, 1024, 128)`.  cuBLAS / hipBLASLt
   picks an MFMA-friendly tile (typically 128x128 / 64x128) and
   runs at ~30 TFLOP/s on MI355.  The generic Triton kernel uses
   ``BLOCK_S = BLOCK_P = 32`` -- bound by the register footprint of
   the per-tile q + k + acc state across the head-unrolled loop --
   which under-utilises tensor cores by ~3x.
2. The BWD does **three** `tl.atomic_add` calls per program:
   `dq[b, s, h, :]`, `dk[b, p, :]`, `dw[b, s, h]`.  At V4-Flash
   widths the grid is `(1, 128, 32) = 4096` programs; the `dk`
   atomic-add contention is severe because all 128 s_tiles touch
   the same 32 p_tiles' worth of `dk` rows.

### Where it does win

Small-shape paths (B=2, S=128, P=32) win because the eager body
launches 7 separate CUDA kernels with overhead ~10-15 µs each, and
the Triton kernel collapses that into one launch.  The CSA layers
in the V4-Flash proxy are at S=4096, so this win does not apply
end-to-end.

---

## 4. Descope rationale

The plan-6 P38 task list **explicitly** contains:

> Task list refinement — measure the per-layer Indexer scoring cost
> on the post-P37 trace; if it is < 3 ms / iter, P38 is descoped.

At V4-Flash widths the Indexer is **not** > 3 ms / iter -- the
eager body is fast enough (~0.3 ms / call x 3 CSA layers = ~0.9 ms
/ iter total) that fusing it into a sub-optimal Triton kernel
**regresses** end-to-end iter time by ~5-6 ms (3 layers x ~2 ms
BWD regression).

The kernel is therefore landed **with the env default flipped to
"0"** so the eager path runs unchanged in production.  Setting
``PRIMUS_INDEXER_TRITON=1`` opts in for benchmarking / future
tuning -- per the plan-5 P32 RoPE-bug precedent on default-off
landings.

---

## 5. Code surface

| Path | Role |
| --- | --- |
| `primus/backends/megatron/core/transformer/v4_attention_kernels/_triton/indexer_score.py` | Triton kernels + `IndexerScoreFn` autograd wrapper + dispatcher predicates.  Default-off env knob. |
| `primus/backends/megatron/core/transformer/indexer.py` | `Indexer.forward` routes through Triton when `PRIMUS_INDEXER_TRITON=1`.  Eager body verbatim otherwise. |
| `tests/unit_tests/megatron/transformer/deepseek_v4/test_p38_indexer_triton.py` | G41 parity + topk parity + release-tier slow test |
| `deepseek-v4/develop/progress/p38/bench_indexer.py` | Microbench: V4-Flash + small |
| `deepseek-v4/develop/progress/p38/bench/{v4,small}.json` | Bench raw JSON |
| `run_deepseek_v4_flash_proxy.sh` | `PRIMUS_INDEXER_TRITON=${PRIMUS_INDEXER_TRITON:-0}` (default OFF) |

---

## 6. Tests

* `tests/unit_tests/megatron/transformer/deepseek_v4/test_p38_indexer_triton.py`
  * `TestG41ForwardParity` -- FWD parity vs eager, ``H ∈ {1,2,4,8}``,
    ``dtype ∈ {fp32, bf16}``.
  * `TestG41TopKParity` -- post-`topk` indices match between Triton
    and eager path via `Indexer.forward` env-toggle.
  * `TestG41BackwardParity` -- BWD parity vs eager autograd
    (`q.grad`, `k.grad`, `w.grad` all match within fp32 tolerance).
  * `TestG41ReleaseTier` -- V4-Flash shape (`B=1, S=4096, P=1024,
    H=8, Hd=128`, bf16).  Marked `slow`.
  * `TestG41EdgeCases` -- unsupported H raises;
    `is_triton_kernel_supported` predicate parity.

16 fast + 1 slow all green.

---

## 7. Follow-ups

* **Tensor-core-friendly tile sizes.** The BWD kernel's `tl.dot`
  operates at `[BLOCK_S, HD] @ [HD, BLOCK_P]` per head; at
  BLOCK_S=32, BLOCK_P=32, HD=128 we under-utilise the MFMA pipe.
  Bumping to BLOCK_S=64, BLOCK_P=64 raises register pressure to ~16
  KiB / warp which would need a different launcher (smaller H
  unroll or fewer concurrent programs).
* **Split FWD-only fusion.** A FWD-only kernel (BWD via autograd-
  recorded eager ops) might still win at V4-Flash if the eager
  einsum kernel overhead is high.  Deferred.
* **Eager-better profile annotation.** Add a one-shot trace
  capturing the eager Indexer call so future plan iterations can
  spot any regression in the eager path that would re-open this
  fusion's value.  Deferred to plan-6 close-out.

---

## 8. Commit pin

(Filled in at commit time.)

```
commit <SHA>
Date:   2026-05-14
Author: …

backend(deepseek-v4)[plan-6][P38]: Indexer.forward scoring Triton FWD/BWD (descoped, default-off)
```
