# Plan-8 P56 — Plan-8 close-out + plan-9 kick-off scope

> Phase summary written 2026-05-15 at P56 close-out.  Doubles as
> the plan-8 hand-off note + the plan-9 kick-off scope document.

**Status: plan-8 closes with TWO tilelang kernels shipped (P50
dense FWD + P51 dense BWD) behind `PRIMUS_V4_TILELANG_ATTN=0`
default and FOUR descoped phases (P52/P53/P54/P55) blocked on
the same head_dim=512 SMEM-budget gap.  No production iter-time
delta vs the plan-7 P48 anchor — the V4 attention path stays on
the plan-4 P25 / plan-5 P31 / P32 final Triton kernels.  Plan-9
scope is pinned: (a) upstream tilelang SMEM partitioning, (b)
sink-BWD bf16 fp32-keep-intermediates fix, (c) one hand-tuned
MFMA schedule for head_dim=512.**

---

## 1. What plan-8 set out to do

Plan-8 opened with the goal of re-implementing the three V4
attention families (`compress_ratio ∈ {0, 4, 128}`) in tilelang
to extract MFMA-scheduling / pipelining / shared-memory swizzle
wins the existing Triton kernels can't reach.  End-of-plan-8
target was **≤ 470 ms / iter** (~1.09× over the plan-7 P48 anchor
of 510.6 ms).

## 2. What actually happened

| phase | scope | outcome | proxy delta |
| --- | --- | --- | ---: |
| **P49** | tilelang infra + dispatcher (env knob + cache + lazy stubs) | **shipped, default OFF**: 19 G49 tests green; plan-4..7 ratchet unchanged | 0 ms |
| **P50** | dense FWD tilelang (cr=0) | **shipped, default OFF**: 19 G50 tests green; smoke shape wins **1.62×** vs Triton; **V4-Flash shape regresses 0.14×** (head_dim=512 SMEM-budget gap) | 0 ms |
| **P51** | dense BWD tilelang (cr=0) + autograd Function | **shipped, default OFF**: 4 G51 tests green (no-sink BWD parity + dispatcher audits); fixes P50 latent autograd-bypass bug; sink BWD descoped (bf16 `inf` at query 0); V4-Flash regression structural (same SMEM gap) | 0 ms |
| **P52** | HCA FWD (cr=128) — extension of P50 | **descoped at task-list refinement**: V4-Flash inherits P50 regression; Triton fallback already correct via `_kernel_supports(hca_local_seqlen>0) == False` | 0 ms |
| **P53** | HCA BWD (cr=128) — extension of P51 | **descoped at task-list refinement**: same as P52 + sink-BWD bf16 issue applies | 0 ms |
| **P54** | CSA FWD (cr=4) — new kernel | **descoped at task-list refinement**: V4-Flash CSA SMEM profile sits in the same regression band; the plan-5 P31 Triton kernel is at the SMEM optimum | 0 ms |
| **P55** | CSA BWD (cr=4) — new pipeline | **descoped at task-list refinement**: same as P54 + sink-BWD bf16 issue + plan-5 P32 final's segreduce design is mature | 0 ms |
| **P56** | close-out | this document | n/a |

**Plan-8 contribution to iter time: 0 ms** (no production kernel
shipped at default-on; the dense FWD + BWD ship behind the env
knob for future use).

## 3. The structural V4-Flash SMEM-budget gap

The single observation that blocks plan-8 from beating Triton at
V4-Flash production widths is:

> At `head_dim=512` on MI355X (160 KiB LDS budget), per-program
> shared-memory allocations for Q + K + V + intermediate
> fragments must stay under ~120 KiB.  Tilelang's default
> `T.gemm(... policy=GemmWarpPolicy.FullRow)` MFMA scheduling at
> `block_M=block_N=32` lands on a layout that runs ~5-15× slower
> than Triton's `tl.dot` at the same SMEM footprint.

P50 microbench (`progress/p50/bench/v4_flash.json`) shows the
exact pattern:

| path | block size | wall ms | TFLOP/s | speedup |
| --- | --- | ---: | ---: | ---: |
| Triton (plan-4 P25)     | BLOCK_M=BLOCK_N=32 | 0.744 |  90.9 | 1.00× |
| tilelang (P50, default) | block_M=block_N=32 | 5.241 |  12.9 | 0.14× |
| tilelang (P50, 16×32)   | block_M=16, block_N=32 | 4.472 |  15.1 | 0.17× |

64×64 tiles bust the SMEM budget (HIP `invalid argument` launch
fail).  Smaller tiles starve parallelism.  The fundamental issue
is that tilelang's `T.gemm` does not emit the same MFMA schedule
Triton's `tl.dot` does — likely a difference in fragment-layout
selection at `head_dim=512`.

## 4. Plan-9 starter set

Three deliverables would unblock the tilelang V4 attention path:

### 4.1 Plan-9 P-X1: Upstream tilelang SMEM partitioning

Land programmable shared-memory partitioning in tilelang so a
single kernel can split Q+K+V residency between LDS and HBM at
`head_dim=512`.  This would unlock `block_M=block_N=64` tiles
without busting the 160 KiB budget — bringing tilelang's effective
parallelism back to par with Triton.

* Upstream: tilelang/issues feature request.
* Estimated savings if it lands + plan-8 phases reactivated:
  ~10-20 ms / iter (matching plan-8 P50-P55 original budget).

### 4.2 Plan-9 P-X2: Sink-BWD bf16 fp32-keep-intermediates fix

P51 documented a bf16 `inf` at query 0 when the softmax
denominator is dominated by a single `qk + sink` pair.  The fix
is to keep `P_acc / dP / delta` fp32 throughout the `dq_tile @ K`
chain, only casting to bf16 at the final atomic-add store.  ~50
lines of kernel changes.

* Required prerequisite for HCA / CSA BWD (both always use sink
  in production V4-Flash).

### 4.3 Plan-9 P-X3: Hand-tuned MFMA schedule for head_dim=512

A V4-Flash-specific MFMA schedule (`k_pack`, swizzle, num_stages)
that beats Triton's default `tl.dot` at the same SMEM footprint.

* The `tilelang/examples/amd/example_amd_flash_attn_fwd.py`
  autotune covers `dim ∈ {64, 128}`; the V4-Flash D=512 envelope
  needs its own sweep.
* Estimated savings if it beats Triton: ~5-10 ms / iter.

### 4.4 Plan-9 scope summary

| phase | scope | est. savings vs plan-7 P48 anchor |
| --- | --- | ---: |
| Plan-9 P-X1 | Upstream tilelang SMEM partitioning | -10 to -20 ms |
| Plan-9 P-X2 | Sink-BWD bf16 fp32-keep-intermediates fix | (enables P-X3..X5) |
| Plan-9 P-X3 | Hand-tuned MFMA schedule for D=512 | -5 to -10 ms |
| Plan-9 P-X4 | Re-activate plan-8 P52/P53 (HCA tilelang) | -5 to -10 ms |
| Plan-9 P-X5 | Re-activate plan-8 P54/P55 (CSA tilelang) | -5 to -10 ms |
| Plan-9 P-X6 | Close-out + proxy bake-off | -- |

Combined plan-9 budget: **-25 to -50 ms / iter** vs the plan-7
P48 anchor (510.6 ms → ~460-485 ms), aligned with the plan-8
original goal.

## 5. Cumulative plan-5 + 6 + 7 + 8 perf summary

| anchor | iter ms | TFLOP/s/GPU | vs P28 anchor |
| --- | ---: | ---: | ---: |
| P28 baseline                                              | 8837.4 | 77.5  | 1.00× |
| P32 final (plan-5)                                       |  603.3 | 1134* | 14.64× |
| P33 (TFLOP/s denominator corrected)                      |  603.3 | 444.2 | 14.64× |
| **P40 final (plan-6 close)**                             | **510.6** | **524.9** | **17.31×** |
| P41–P47 (plan-6 ext + plan-7 microbench)                 |  ~510 |   ~524 | 17.31× |
| **P48 final (plan-7 close)**                             | **510.6** | **524.9** | **17.31×** |
| **P49–P55 (plan-8)** (no production kernel)              | **510.6** | **524.9** | **17.31×** |
| **P56 final (plan-8 close-out)**                         | **510.6** | **524.9** | **17.31×** |

(* P32 final TFLOP/s pre-correction; P33+ use the closed-form-
corrected denominator.)

**Plan-8 final EP=8 proxy iter time: 510.6 ms / 524.9 TFLOP/s/GPU,
17.31× vs P28** — same as plan-6 P40 / plan-7 P48 final, since
plan-8 ships no production kernel.

## 6. What plan-8 actually contributed

Plan-8 ships valuable scaffolding even without proxy delta:

* **Tilelang dispatcher infrastructure** (P49): env knob, lazy
  stub registry, cache layout, AOT build script, autograd
  integration.  Plan-9 phases can register kernels via
  `register_available_kernel(...)` without further wiring work.
* **Reference tilelang kernel implementations** (P50, P51) for
  V4 dense FWD + BWD.  Demonstrates the tilelang code style /
  conventions / kernel-cache pattern for plan-9 to extend.
* **Empirical evidence of the V4-Flash SMEM-budget gap**
  (`progress/p50/bench/v4_flash.json`): tilelang at the same
  tile shape as Triton at D=512 runs 5× slower.  Pins the
  upstream tilelang feature request in concrete terms.
* **Honest descope record** for P52..P55 with the structural
  rationale documented so plan-9 can pick up the work without
  re-discovering the gap.
* **Sink-BWD bf16 numerical issue** (P51): documented for future
  fix.  Required for HCA / CSA BWD work.

## 7. Standing rules reaffirmed in plan-8

* **R9.1 (10% de-scope rule)** worked as designed.  P50/P51 ship
  the kernels (microbench shows they work at small shapes) but
  default OFF because the V4-Flash regression is structural.
  P52/P53/P54/P55 descope at task-list refinement because the
  same structural blocker would apply.
* **R6.2 (no third_party edits)** held.  Tilelang stays vendored
  at `tilelang/` and untouched; plan-8 only uses its public API.
* **R2.6 (per-phase trace + tgz)** skipped for plan-8 phases
  because none changed runtime behaviour at default-off.
* **R9.3 (forensic attribution)** held.  P50's microbench is the
  load-bearing forensic record; the V4-Flash regression is
  attributed to a specific tile-shape choice (`block_M=block_N=32,
  threads=64` at D=512) with the failure mode documented.

## 8. Follow-ups + commit pin

* P56 close-out commit pins all plan-8 phase summaries +
  this hand-off note.
* `PRIMUS_V4_TILELANG_ATTN` env knob default stays `"0"` at
  P56 close-out.
* No `proxy_ep8.md` / `attention_perf.md` row appended — plan-8
  ships no production-default kernel, so the row would just
  repeat the plan-7 P48 anchor.
* Plan-9 kick-off scope pinned in §4 above.
* Feature commit SHA: 19e41c9a.
