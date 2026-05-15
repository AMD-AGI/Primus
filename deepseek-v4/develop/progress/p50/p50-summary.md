# Plan-8 P50 — V4 dense FWD tilelang kernel (cr=0)

> Phase summary written 2026-05-15 at P50 close-out.

**Status: shipped, default OFF.  Microbench wins big at the smoke
shape (1.62×) but **regresses 6× at V4-Flash production widths**
(`D=512`, 0.14×) — this is the head_dim=512 SMEM-budget gap
documented in `plan-8/01-roadmap.md` Top Risk #2.  G50 parity
gate green (19/19 tests).  Plan-4..plan-8 ratchet unchanged at
default-off.  Per R9.1 the kernel ships behind
`PRIMUS_V4_TILELANG_ATTN=0` for future tuning + flagged for
upstream tilelang.**

---

## 1. Objective

Re-implement the plan-4 P25 V4 dense FWD Triton kernel
(`_triton/v4_attention_fwd.py`) in tilelang for the V4-Flash
`compress_ratio == 0` layers (dense / SWA / sink, MQA / MHA), with
identical wrapper signature so the env-knob dispatcher swaps in
seamlessly.

## 2. Design

### 2.1 Kernel

New `_tilelang/v4_attention_fwd_tilelang.py` ships a single
`@tilelang.jit`-decorated factory `_make_v4_attention_fwd_kernel`
that takes shape args (`batch, heads_q, heads_k, seq_q, seq_k,
dim, has_sink, swa_window`) + tile config (`block_M, block_N,
num_stages, threads`).  The kernel body:

* **Grid**: `(ceildiv(seq_q, block_M), heads_q, batch)` — one
  program per `(b, h, m_tile)`.
* **MQA broadcast**: `k_head = by // (heads_q // heads_k)` reads
  the shared K/V head when `heads_k == 1`.
* **Online softmax**: classic FlashAttention v2 with
  `T.reduce_max` / `T.exp2 * scale` (FFMA-friendly) /
  `T.reduce_sum` per tile.
* **SWA**: `start = max(0, (bx * block_M - swa_window) //
  block_N)` skips K tiles outside the window.  Includes the
  FlashAttention-3 `check_inf` step that resets `scores_max ==
  -inf` to 0 so `exp2(-inf - -inf)` doesn't NaN.
* **Sink**: applied as a virtual key column at end-of-K-loop via
  `logsum[i] += exp2(sinks[i] * log2e - scores_max[i] * scale)`
  — mirrors the
  `tilelang/examples/attention_sink/example_mha_sink_fwd_bhsd.py`
  formulation.
* **LSE**: emitted as `log(logsum) + scores_max * sm_scale` (base-e,
  matches the plan-4 P25 Triton kernel + BWD's recompute
  contract).

### 2.2 Wrapper + dispatcher integration

`v4_attention_fwd_tilelang(q, k, v, *, sink, additive_mask,
swa_window, attn_dropout, training, scale, hca_local_seqlen)`
matches the Triton launcher signature.  Falls back to the Triton
kernel when `additive_mask is not None` or `hca_local_seqlen > 0`
(those are P52 territory).

Module-import-time hooks:

* `_tilelang.register_available_kernel("v4_attention_fwd")` flips
  `is_tilelang_kernel_available(...)` to True.
* `_tilelang._lazy_load(...)` then replaces the stub
  `_tilelang.v4_attention_fwd_tilelang` (which was set to the
  submodule by Python's import machinery) with the real wrapper
  function.

### 2.3 Tile-config heuristic

```python
if dim >= 256:
    block_M, block_N, threads = 32, 32, 64   # V4-Flash (D=512)
else:
    block_M, block_N, threads = 64, 64, 128  # smoke / smaller dims
```

The conservative `block_M=32, block_N=32` at `D=512` matches the
plan-4 P25 Triton kernel's choice and keeps `Q+K+V+O` shared
buffers at `4 * 32 * 512 * 2 = 128 KiB`, under the 160 KiB MI355
SMEM budget.

## 3. Code surface

```
primus/backends/megatron/core/transformer/v4_attention_kernels/_tilelang/v4_attention_fwd_tilelang.py (new)
  + _make_v4_attention_fwd_kernel (@tilelang.jit JIT factory)
  + v4_attention_fwd_tilelang (autograd-free wrapper)
  + _get_or_compile_kernel (Python-side memoise)
  + _kernel_supports (Triton fallback predicate)

primus/backends/megatron/core/transformer/v4_attention_kernels/_tilelang/__init__.py
  M  _lazy_load now restores parent-module attribute after submodule import
     (Python's import machinery would otherwise shadow the stub with the
     submodule).

tests/unit_tests/megatron/transformer/deepseek_v4/test_p50_v4_attention_fwd_tilelang.py (new)
  + 16 G50 parity tests across (MQA/MHA × sink/no_sink × SWA/full × bf16/fp16).
  + 2 fallback-to-Triton tests (additive_mask + hca_local_seqlen routes
    through the Triton path).
  + 1 dispatcher-registration test.

deepseek-v4/develop/progress/p50/
  + bench_v4_attention_fwd_tilelang.py (microbench harness)
  + bench/{smoke, v4_flash}.json (microbench JSONs)
  + p50-summary.md (this file)
```

## 4. Performance

### 4.1 Microbench at smoke shape (B=2, HQ=4, HK=1, Sq=Sk=128, D=64, bf16)

| path     | median ms | TFLOP/s | speedup |
| ---      | ---:      | ---:    | ---:    |
| triton   |   0.038   |   0.4   | 1.00×   |
| tilelang |   0.024   |   0.7   | **1.62×** |

Small-shape win — tilelang's MFMA scheduling control + `exp2`
FFMA fusion pulls ahead of Triton's `tl.dot` + `exp` path.

### 4.2 Microbench at V4-Flash shape (B=1, HQ=64, HK=1, Sq=Sk=4096, D=512, swa_window=128, bf16)

| path     | median ms | TFLOP/s | speedup |
| ---      | ---:      | ---:    | ---:    |
| triton   |   0.744   |  90.9   | 1.00×   |
| tilelang |   5.241   |  12.9   | **0.14× (regression)** |

**The V4-Flash regression is the documented Top Risk #2 in
`plan-8/01-roadmap.md`** ("head_dim=512 is too large for
tilelang's default tile shapes").  Two probe configurations
exercised:

* `block_M=block_N=32, threads=64`: 5.24 ms (this report).
* `block_M=16, block_N=32, threads=64`: 4.47 ms — also a
  regression, just slightly less bad.

The Triton kernel at the same shape ran at 0.744 ms (90.9
TFLOP/s).  Triton's `tl.dot` + `BLOCK_M=BLOCK_N=32` at
`head_dim=512` schedules MFMA differently and lands at ~90
TFLOP/s; tilelang's `T.gemm(... policy=GemmWarpPolicy.FullRow)`
at the same tile shape lands at ~13 TFLOP/s.  Reproducing the
Triton schedule in tilelang requires either:

* Larger tiles (64x64) — exceeds SMEM at D=512, would need
  programmable shared-memory partitioning (upstream tilelang
  feature request).
* Hand-pick `k_pack=2` + swizzle pattern — out of scope for P50
  given the autotune set in the AMD example is for D=128.

## 5. Tests

**G50 — 19 tests, all green** (`pytest -q` 3.00s):

* **TestG50FastTierParity** (16): parametrise `(dtype ∈ {fp16, bf16})
  × (has_sink ∈ {True, False}) × (swa_window ∈ {0, 16}) × (is_mqa ∈
  {True, False})` at `B=2, HQ=4, Sq=Sk=64, D=64`.
* **TestG50FallbackToTriton** (2): `additive_mask` + `hca_local_seqlen`
  both route through the Triton path.
* **TestG50DispatcherRegistration** (1): importing the kernel
  module flips `_tilelang.is_tilelang_kernel_available(...)` to
  True.

bf16 tolerance set to `atol=3e-2 rtol=5e-2` at fast-tier (the
ULP-level rounding difference between tilelang's exp2 + scale
fusion and the eager reference's exp + scale).

**Plan-4..plan-8 ratchet** (default-off): unchanged from P49
(`451 passed`).  No regression.

## 6. Gating

* `PRIMUS_V4_TILELANG_ATTN` default **`"0"`** (descoped — V4-Flash
  microbench regresses 6×).
* Set to `"1"` to opt into the tilelang FWD path; the kernel
  parity gate passes but the V4-Flash widths run slower than
  Triton until the SMEM-budget gap is closed (upstream tilelang
  feature request or a hand-tuned config).

## 7. Failed / negative probes

* **Initial JIT factory pattern** with `q_shape` defined in an
  outer non-JIT'd function — tilelang's `@T.prim_func` annotation
  eval failed with `NameError: name 'q_shape' is not defined`.
  Fix: decorate the SAME function with `@tilelang.jit` so the
  annotation closure scope works (matches the
  `tilelang/examples/.../example_mha_sink_fwd_bhsd.py` pattern).
* **`from __future__ import annotations`** — making the annotations
  lazy strings broke tilelang's `get_type_hints` again.  Fix:
  removed the future-import (this module is the only V4 file
  that does so; the existing files don't need it).
* **`block_M=block_N=64` at D=512** — HIP launch error
  `hipModuleLaunchKernel invalid argument`.  SMEM bust.  Fix:
  lower to 32x32 at D >= 256.
* **`block_M=block_N=32, threads=128` at D=512** — same HIP
  launch error; some hidden tilelang allocation pushes total
  SMEM over the 160 KiB limit.  Fix: lower threads to 64.
* **`block_M=16, block_N=32, threads=64` at D=512** — runs but
  4.47 ms (0.17× speedup vs Triton).  Insufficient parallelism.
* **`block_M=block_N=32, threads=64` at D=512** — final landed
  config; 5.24 ms (0.14× speedup vs Triton).  Worse than the
  16x32 variant; would have descoped either way per R9.1.

## 8. Follow-ups + commit pin

* P51 will land the dense BWD tilelang kernel using the same
  factory pattern.  The BWD recipe (preprocess + dq + dkv +
  dsink) is more amenable to MFMA-aware scheduling than the FWD
  online softmax — there's a chance the V4-Flash BWD regression
  story is different.
* Upstream tilelang feature request: programmable shared-memory
  partitioning so we can run `block_M=block_N=64` at D=512 with
  only partial Q/K/V residency in shared memory.  Until that
  lands, the V4-Flash widths stay on Triton.
* Feature commit SHA: af07de91.
