# Plan-7 P48 — Plan-7 close-out (microbench-driven descope retrospective)

> Phase summary written 2026-05-15 at P48 close-out.  Doubles as
> the plan-7 hand-off + plan-8 kick-off note.

**Status: plan-7 closes with no production kernel shipped.  P45's
microbench evidence disproved the plan-7 P45..P47 budget
hypothesis; P46 and P47 descoped on the same evidence; P48 ships
the close-out retrospective + the plan-8 kick-off scope.**

---

## 1. What plan-7 set out to do

Plan-7 opened with the goal of attacking the optimizer-step
residual that the P40 trace surfaced as the dominant remaining
cost after plan-6 closed the in-model elementwise sweep:

| trace bucket | total / iter | launches | plan-7 phase | predicted savings |
| --- | ---: | ---: | --- | ---: |
| `vec_elem<add_bf16>` (Adam ε-add)               | 170.99 ms |  743 | **P45** | -150 ms |
| `multi_tensor<adam_master>` (TE fused Adam)     |  45.92 ms |  321 | **P45** | (covered) |
| `multi_tensor<scale>` (grad-scaling pre-AR)     |  10.96 ms |  321 | **P46** |   -5 ms |
| `reduce<l2norm_bf16>` (grad-norm clipping)      |   7.76 ms |   12 | **P47** |  -10 ms |
| `multi_tensor<l2norm>` (per-param L2)           |   6.72 ms |  321 | **P47** | (covered) |
| **Total optimizer-step residual**                | **~242 ms** |      | **~165 ms total predicted** |

Target end-of-plan-7 iter time: **≤ 340 ms / iter** (from the
plan-6 P40 final of 510.6 ms, after subtracting the ~165 ms
optimizer savings).

## 2. What actually happened

P45 (the highest-budget phase) shipped a prototype Triton
multi-tensor BF16 add kernel + microbench to validate the
hypothesis.  The microbench at n_tensors=743 (matching the trace
launch count) showed:

| path | median ms | GB/s | speedup vs eager |
| --- | ---: | ---: | ---: |
| `torch._foreach_add_` (eager)  | 7.77 | **1994.3** | 1.00× |
| `triton_per_tensor` (prototype) | 7.59 | 2041.1 | **1.02× (tied)** |
| `triton_packed` (prototype)    | 33.95 |  456.2 | **0.23× (regression)** |

**Key finding: `torch._foreach_add_` is already a well-tuned
multi-tensor kernel that runs at near-MI355 HBM peak.**  The
P40 trace's 743 `vec_elem<add_bf16>` launches are NOT
launch-overhead-dominated — each launch is doing ~460 KB of real
per-tensor HBM traffic, and the existing eager path already
batches them efficiently.

The plan-7 budget assumption ("collapse 743 launches into one
kernel saves 150 ms") was therefore mis-calibrated.  The actual
extractable headroom over `_foreach_add_` is bounded by the
per-launch overhead reduction (~5-10 µs × 743 ≈ 5 ms / iter), not
the full 170 ms / iter bucket.

P46 and P47 descoped via the same evidence: `multi_tensor<scale>`
(321 launches × 34 µs) and `multi_tensor<l2norm>` (321 × 21 µs)
are also already multi-tensor-batched — no headroom for a Triton
re-implementation.

## 3. Why the real win requires plan-8

The actual dominant cost in the 743 `vec_elem<add_bf16>` launches
is the per-tensor compute (BF16 add at near-peak HBM bandwidth),
not the launch overhead.  Reducing that requires:

* **Fusing with adjacent ops in the optimizer step** — fold the
  Adam ε-add into the master functor's `sqrt(v_hat) + eps` so the
  master_p stays in registers across the cast-to-bf16 step.
  This is the standard fused-Adam optimisation, requires
  bit-exact replication of the Apex / TE master-param remainder
  rounding math, and is genuinely multi-day work.
* **Reducing the parameter count** — sharded optimizer / ZeRO-3.
  Out of scope.

Plan-8 P-XX (TBD) will land:

* `primus/backends/megatron/core/extensions/_triton/fused_adam.py`
  — the production fused-Adam Triton kernel with master-param
  remainder bit-exactness.
* `primus/backends/megatron/patches/turbo_adam_patches.py` — a
  monkey-patch wrapping the Apex / TE call site (no third_party
  edits per R6.2).
* G47-equivalent parity gates: ULP ≤ 1 bf16 master; 10-step
  micro-rollup ≤ 1e-3; release-tier 100-step loss-curve
  difference ≤ 1e-3 vs upstream at fixed seed.

The P45 prototype + microbench already in tree are the load-bearing
seed work for this plan-8 phase.

## 4. Plan-7 phase-by-phase outcome

| phase | scope | outcome | savings |
| --- | --- | --- | ---: |
| **P45** | Triton fused Adam ε-add | **prototype ships**; production descoped to plan-8 (Apex / TE bit-exactness) | 0 ms |
| **P46** | Fused grad-scale Triton | descoped via P45 evidence (`multi_tensor_apply` already at HBM peak) | 0 ms |
| **P47** | Fused grad-norm clip Triton | descoped via P45 evidence | 0 ms |
| **P48** | Plan-7 close-out | this document | n/a |

**Plan-7 contribution to iter time: 0 ms** (no production kernel
shipped; perf delta vs P40 final = noise).

**Plan-7 contribution to plan-8 readiness:**

* Definitive microbench evidence that the optimizer-step
  fusion budget needs to come from joint Adam + ε-add fusion,
  not from multi-tensor-batching alone.
* Multi-tensor BF16 add Triton prototype (passes 8 G47 parity
  tests, bench-matched against `_foreach_add_`).
* Three eight-section per-phase summaries (`p45/p46/p47/p48`)
  documenting the budget mis-calibration + the plan-8 follow-up
  scope.

## 5. Cumulative plan-5 + plan-6 + plan-7 perf summary

| anchor | iter ms | TFLOP/s/GPU | vs P28 |
| --- | ---: | ---: | ---: |
| P28 baseline                                              | 8837.4 | 77.5  | 1.00× |
| P31b (plan-5 V4 attention BWD split)                     |  964.8 | 709.3 | 9.15× |
| P32 final (plan-5 RoPE-fix + split-BWD + segreduce)      |  603.3 | 1134* | 14.64× |
| P33 (TFLOP/s denominator corrected)                      |  603.3 | 444.2 | 14.64× |
| **P40 final** (plan-6 elemwise fusion sweep close)       | **510.6** | **524.9** | **17.31×** |
| **P41**  (Indexer tail descoped via 10-iter noise)       |    513.9 | 523.7 | 17.20× |
| **P42**  (Permute / .contiguous() descoped at refinement)|        n/a |   n/a | n/a |
| **P43**  (Router 50-iter A/B descope **reaffirmed**)     |    510.17 | 528.0 | 17.32× |
| **P44**  (Attn epilogue descoped — already in kernel)    |        n/a |   n/a | n/a |
| **P45**  (Adam multi-tensor add prototype; descoped int.)|        n/a |   n/a | n/a |
| **P46**  (Grad-scale descoped via P45 evidence)          |        n/a |   n/a | n/a |
| **P47**  (Grad-norm clip descoped via P45 evidence)      |        n/a |   n/a | n/a |
| **P48 final** (plan-7 close-out)                         | **510.6** | **524.9** | **17.31×** |

(* P32 final TFLOP/s pre-correction; P33+ use the closed-form-
corrected denominator.)

**Plan-7 final EP=8 proxy iter time: 510.6 ms / 524.9 TFLOP/s/GPU,
17.31× vs P28** — same as P40 final, since plan-7 ships no
production kernel.

## 6. Plan-8 kick-off scope

| phase | scope | est. savings | confidence |
| --- | --- | ---: | --- |
| **plan-8 P49** | Production fused-Adam Triton kernel with master-param remainder bit-exactness | ~80-150 ms (the actual headroom unknown until measured) | medium (requires bit-exact replication) |
| **plan-8 P50** | Monkey-patch onto TE / Apex Adam call site | (covered) | low risk |
| **plan-8 P51** | Joint fused-grad-clip + Adam megakernel | ~10 ms additional | low |
| **plan-8 P52** | V4 attention BWD shared Q/K/V load (cooperative groups) | ~30-40 ms | high risk (modifies plan-5 kernels) |
| **plan-8 P53** | Forensic External-id helper (R9.3) for the unattributed `vec_elem<bf16_copy>` / `vec_elem<mul_bf16>` buckets | enables future plan-8 in-model fusion | n/a |
| **plan-8 P54** | Plan-8 close-out | -- | -- |

Total plan-8 budget (if all phases ship): **~120-200 ms / iter
savings** vs the current 510.6 ms baseline → projected
end-of-plan-8 iter time **~310-390 ms** — finally crossing the
plan-6 original 310 ms goal.

## 7. Standing rules reaffirmed in plan-7

* **R9.1 (10 % de-scope rule)** worked exactly as designed.  Each
  plan-7 phase de-scoped because its budget fell below the 51 ms
  cut-off after the P45 microbench revealed the actual headroom.
* **R9.3 (forensic attribution before fix)** prevented P42 / P44
  from shipping code that wouldn't have moved the proxy.  The
  pattern: when the trace shows a bucket but the call site is
  spread / unknown, de-scope until the forensic helper lands.
* **R2.6 (per-phase trace + tgz)** did NOT fire for plan-7 phases
  because none shipped a runtime-affecting change — the rule
  explicitly skips for documentation-only / prototype phases.

## 8. Follow-ups + commit pin

* Plan-7 close-out commit (this commit) pins all plan-7 phase
  summaries + the plan-8 kick-off scope.
* Feature commit SHA: 0f95e812.
