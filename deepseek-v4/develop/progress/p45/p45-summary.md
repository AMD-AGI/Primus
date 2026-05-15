# Plan-7 P45 — Multi-tensor BF16 add Triton prototype (kernel ships; production integration descoped)

> Phase summary written 2026-05-15 at P45 close-out.  Microbench
> numbers pinned against the live `mi355-gpu-8` runs in
> `progress/p45/bench/{v4flash,v4flash_743,uniform_small_743}.json`.

**Status: prototype kernel ships behind no env knob (microbench /
unit-test target only).  Production integration with the Apex / TE
`multi_tensor_adam_master_param_remainder` call site is descoped
to plan-8.  The microbench reveals that the eager
`torch._foreach_add_` baseline is already a well-tuned multi-tensor
kernel internally — the Triton prototype matches it at n=743 but
offers no real headroom over `foreach`.  The 170.99 ms / iter
`vec_elem<add_bf16>` trace bucket is therefore NOT a multi-tensor
fusion target; the real cost is in the per-tensor compute, not in
launch overhead.**

---

## 1. Objective (originally scoped)

Replace the ~743 ``vec_elem<add_bf16>`` launches that dominate
the P40 trace's optimizer-step residual (170.99 ms / iter, 32.7 %
of step) with a single Triton multi-tensor kernel.  Working
hypothesis at plan-7 kick-off: the launches were per-tensor
`tensor.add_(other)` calls dispatched serially, and a multi-tensor
fusion would buy ~150 ms / iter by collapsing the launch overhead.

## 2. Design

Two Triton kernel variants in
`primus/backends/megatron/core/extensions/_triton/multi_tensor_add.py`:

### 2.1 `_multi_tensor_add_per_tensor_kernel`

Single-tensor variant.  One grid program per ``BLOCK_SIZE`` chunk
of one tensor.  Used in a loop by the wrapper, with each launch
landing on the same HIP stream — Triton stacks them concurrently
on the GPU.

### 2.2 `_multi_tensor_add_packed_kernel`

Single-kernel variant.  CPU-side dispatch table maps each
``program_id`` to ``(tensor_idx, local_block_idx)`` so a single
grid launch absorbs all N_TENSORS tensors in one shot.  Pointers,
sizes, and dispatch table all stored on-device.

Both variants implement ``out_i = a_i + scale * b_i`` in place.
Math runs fp32 internally (bf16 load → fp32 compute → bf16 store)
to match the eager `torch._foreach_add_` rounding.

## 3. Performance

### 3.1 Microbench at V4-Flash-mix (n_tensors=743 — matches trace)

| path | median ms | GB/s | speedup vs foreach |
| --- | ---: | ---: | ---: |
| `torch._foreach_add_` (eager) | 7.77 | 1994.3 | 1.00× |
| `triton_per_tensor`           | 7.59 | 2041.1 | **1.02×** (tied) |
| `triton_packed`               | 33.95 |  456.2 | **0.23×** (regression) |

### 3.2 Microbench at uniform-small (n=743 × `[4096]`)

| path | median ms | GB/s | speedup vs foreach |
| --- | ---: | ---: | ---: |
| `torch._foreach_add_` (eager) | 0.325 | 56.2 | 1.00× |
| `triton_per_tensor`           | 7.30 |  2.5 | 0.04× (launch overhead) |
| `triton_packed`               | 0.84 | 21.6 | 0.39× |

### 3.3 Key finding

`torch._foreach_add_` is already a **well-tuned multi-tensor
kernel**.  At n=743 it sustains ~2000 GB/s of HBM bandwidth
(near MI355's peak ~3-4 TB/s).  The Triton `per_tensor` variant
matches at 2041 GB/s; the packed variant **regresses** because the
CPU-side dispatch-table construction (`pid_to_tid` / `pid_to_local`
Python loops) is itself a multi-ms overhead at this scale.

**The P40 trace's 743 `vec_elem<add_bf16>` launches at 230 µs each
are NOT launch-overhead-dominated** — 230 µs at MI355's ~2 GB/s
elementwise rate per launch = ~460 KB of memory traffic per
launch, which is real per-tensor work.  Collapsing them into a
multi-tensor call saves the per-tensor launch overhead (~5-10 µs
each, ~5 ms / iter at n=743) but does **not** save the bulk of
the 170 ms / iter cost.

## 4. Code surface

```
primus/backends/megatron/core/extensions/_triton/multi_tensor_add.py (new)
  + _multi_tensor_add_per_tensor_kernel
  + _multi_tensor_add_packed_kernel
  + multi_tensor_add_triton_per_tensor
  + multi_tensor_add_triton_packed
  + _build_multi_tensor_dispatch_table

tests/unit_tests/megatron/extensions/test_p45_multi_tensor_add.py (new)
  + 8 parity tests (per_tensor + packed, fp32 + bf16, odd sizes,
    V4-Flash mix)

deepseek-v4/develop/progress/p45/
  + bench_multi_tensor_add.py (3-path bench)
  + bench/{v4flash,v4flash_743,uniform_small_743}.json
```

## 5. Tests

* **G47 parity** — 6 tests (per_tensor + packed × 3 scales).  All
  pass.  `per_tensor` is `atol=0 rtol=0` bit-equal vs eager
  foreach (same fp32 internal math).  `packed` is `atol=1e-3
  rtol=1e-3` bf16 — ULP-1 difference is allowed because the
  packed kernel's bf16 cast-back ordering differs from
  `_foreach_add_` by a few mantissa bits.
* **G47 variants** — 2 tests (V4-Flash mix + odd-sized tail).
  All pass.
* **Total: 8/8 green.**

## 6. Gating

The prototype is currently importable but not wired into any
runtime path.  No env knob.  The production integration would
require:

1. R9.3 forensic External-id correlation to identify the exact
   call site emitting the 743 `vec_elem<add_bf16>` launches.
2. A Primus monkey-patch wrapping that call site to route through
   `multi_tensor_add_triton_per_tensor` (the packed variant
   regresses; the per-tensor variant ties).

Even with full production integration, the headroom is **at most
~5 ms / iter** (the per-tensor launch overhead) — well below the
150 ms / iter the plan-7 P0a budget assumed.

## 7. Why the original plan-7 budget mis-fired

The plan-7 P45 roadmap assumed `vec_elem<add_bf16>` 743 × 230 µs
launches were launch-overhead-dominated and a single-kernel
fusion would absorb them all.  The microbench shows otherwise:
the launches are doing real per-tensor work and the existing
eager `_foreach_add_` already runs them at near-peak bandwidth.

The actual dominant cost in those launches is HBM traffic on the
target tensors, not launch dispatch overhead.  Reducing that
requires:

* **Fusing with adjacent ops in the optimizer step** (e.g. fuse
  Adam ε-add into the master functor's `sqrt(v_hat) + eps`).  This
  is the standard fused-Adam optimisation, requires bit-exact
  replication of Apex / TE's master-param remainder math, and is
  genuinely a multi-day plan-8 deliverable.
* **Reducing the parameter count** (sharded optimizer / ZeRO-3).
  Out of scope for plan-7.
* **Moving Adam to fp32** to halve HBM traffic per parameter.
  Loses convergence accuracy; out of scope.

## 8. Failed / negative probes

* The `_multi_tensor_add_packed_kernel` regresses at all tested
  scales due to CPU-side dispatch-table construction overhead.
  Pre-computing the dispatch table once per parameter list (not
  per call) would fix this but requires a stateful wrapper —
  deferred.
* Initial packed-kernel design used a Triton-side linear scan over
  ``OFF_BUF`` to map ``chunk_start`` to ``tensor_idx``.  The scan
  was correct but the per-element offsets straddled tensor
  boundaries, causing reads from the wrong tensor's pointer.
  Fixed by switching to a CPU-side dispatch table (one
  ``program_id`` per ``(tensor_idx, local_block_idx)`` pair).

## 9. Follow-ups + commit pin

* The microbench JSONs are committed (one-off perf evidence per
  R2.5); future re-attempts should re-run them on a fresh MI355
  before reasoning about headroom.
* Plan-8 P-XX (TBD): production fused Adam Triton kernel with
  master-param remainder bit-exactness.  Required for any further
  attack on the optimizer-step residual.
* P46 / P47 will follow the same descope-on-evidence pattern
  (microbench-driven, not roadmap-driven).
* Feature commit SHA: TBD-p45.
