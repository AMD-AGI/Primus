# Plan-6 P43 — V4 router post-logits 50-iter A/B re-attempt (descope reaffirmed)

> Phase summary written 2026-05-15 at P43 close-out.  Compressor
> APE elementwise fusion sub-task descoped at task-list refinement.

**Status: descope reaffirmed.  The P39 router post-logits Triton
kernel ships in tree (already from P39), default OFF.  P43's
contribution is the *measurement methodology* — 50-iter × 3-run
aggregate cleanly converts the P39 "descoped due to noise" verdict
into "descoped because the proxy A/B measures a clean +0.62 ms /
iter regression at the 0.19 ms noise floor".  No code change.**

---

## 1. Objective

Re-attempt the plan-6 P39 router post-logits Triton fusion that
was descoped because the 10-iter EP=8 proxy A/B sat inside the
±1-3 ms NCCL / dispatch noise band.  Hypothesis: a 50-iter run
aggregated across 3 independent invocations drops the noise floor
by `sqrt(3 × 5) ≈ 3.87 ×` to ~0.5 ms — enough to either confirm
or rule out the ~1 ms / iter aggregate gain the microbench
predicts for `score_fn=sqrtsoftplus` (V4 production).

Side deliverable: the Compressor APE elementwise tail Triton
fusion (`x.reshape(B, P, ratio, D) * ape → sum(2) → + bias`).

## 2. Methodology

* 50-iter EP=8 smoke (no profiler) per
  `progress/p43/run_smoke_p43_router_ab.sh`.
* 3 independent invocations per side, alternating
  `PRIMUS_V4_ROUTER_TRITON ∈ {0, 1}`.
* Steady-iter window: discard iters 1-5 (compile + warmup),
  keep iters 6-50 = 45 samples per run.  3 runs → 135 samples
  per side.
* Aggregator: `progress/p43/aggregate_router_ab.py` (reads
  `rank-7` debug.log for iter times; rank-0 only logs every Nth
  iter, rank-7 logs every iter).

## 3. Performance

### 3.1 Per-run breakdown (steady iters 6-50, n=45 each)

| TRITON | RUN | mean ms | median ms |
| ---: | ---: | ---: | ---: |
| 0 (eager) | 1 | 509.69 | 509.40 |
| 0         | 2 | 509.32 | 508.80 |
| 0         | 3 | 509.64 | 509.00 |
| 1 (triton)| 1 | 509.84 | 509.50 |
| 1         | 2 | 510.08 | 509.90 |
| 1         | 3 | 510.60 | 509.90 |

### 3.2 Aggregated 3-run summary (n=135 each)

| side | mean | median | stdev | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| **A** eager  (`PRIMUS_V4_ROUTER_TRITON=0`) | **509.553 ms** | 509.100 | 2.103 | 506.70 | 516.40 |
| **B** triton (`PRIMUS_V4_ROUTER_TRITON=1`) | **510.174 ms** | 509.600 | 2.351 | 506.60 | 518.10 |

**Delta B − A = +0.621 ms / iter** (Triton slower than eager).

Combined noise floor estimate
(`(stdev_a + stdev_b) / (2 × sqrt(n_min))`) = **±0.192 ms** —
the regression is **3.2× the noise floor**, statistically
significant.

## 4. Why the microbench wins do not surface

The V4-Flash microbench (`progress/p39/bench/v4_sqrtsoftplus.json`)
shows:

* FWD: 0.072 ms (eager) → 0.046 ms (triton) = **1.56× speedup**
* BWD: 0.183 ms (eager) → 0.150 ms (triton) = **1.22× speedup**

Per-call savings = 0.059 ms (FWD) + 0.033 ms (BWD) = **~0.09 ms /
call**.  At 8 layers × 2 micro-batch steps = 16 calls per iter,
total predicted savings = **~1.5 ms / iter**.

In the actual proxy run the savings are **−0.62 ms / iter** —
a **2.1 ms swing** vs the microbench prediction.  Three plausible
causes (no forensic attribution yet; future P43 follow-up):

1. **Triton kernel JIT cache miss.**  The `score_function:
   tl.constexpr` switch emits 3 specialised binaries; each
   first-call JIT-compile costs ~200 ms.  With 16 calls per iter
   × 50 iters, the JIT cost amortises but the first iter pays.
   Iter 1's overhead is excluded from the steady window so this
   is unlikely to explain a 0.6 ms / iter regression at iters
   6-50.
2. **Eager `score_fn` already fused by Inductor.**  PyTorch's
   `torch.nn.functional.softplus` is registered with an Inductor
   fused-kernel rewriter; the eager body may be executing **one**
   kernel for `score_fn`, not the 7 ops the source code suggests.
   The microbench bypasses Inductor; the proxy runs through it.
3. **Cache thrashing.**  The Triton kernel allocates a fresh
   `[N, E] = [4096, 256]` output per call (4 MiB).  Repeated
   allocation + dealloc across 16 calls / iter exercises the
   caching allocator's hot path more aggressively than the eager
   in-place ops do.

The +0.62 ms regression is small but real.  Future re-attempts
should land Inductor-vs-microbench profiling first to attribute
the discrepancy.

## 5. Compressor APE sub-task — descoped at task-list refinement

The Compressor APE elementwise chain
(`score.reshape + score + ape → softmax → mul(kv) → sum(win)`)
runs once per Compressor.forward call.  V4-Flash 8-layer slice
invokes Compressor at:

* 3 CSA layers (`compress_ratio == 4`): main + indexer mini =
  3 × 2 = 6 calls.
* 2 HCA layers (`compress_ratio == 128`): main = 2 calls.

Total: **8 calls / iter (FWD) + 8 BWD = 16 calls**.  Eager
elementwise chain emits ~5 launches per call (reshape view-only
+ add + softmax 3-sub + mul + sum) ≈ 80 launches / iter ≈
4-5 ms / iter at the per-launch overhead floor.

A Triton FWD+BWD fusion would absorb ~80 launches into 16 (one
per Compressor call × FWD/BWD) — savings ≈ 3-4 ms / iter.  Below
the 10 % R9.1 cut-off (510 ms × 10 % = 51 ms).

**Descoped** to keep P43 focused on the measurement
methodology improvement.  The Compressor APE candidate stays
in `progress/p41/p41-candidates.md` for a future re-attempt
(plan-8) with the same forensic External-id attribution.

## 6. Tests

No code change.  Plan-4/5/6 ratchet stays green by construction.

## 7. Gating

* `PRIMUS_V4_ROUTER_TRITON` stays at default `"0"` (descope
  re-affirmed).
* Kernel ships in tree (from P39) for future tuning + small-
  shape paths.

## 8. Failed / negative probes

* The 50-iter × 3-run aggregate showed a **+0.62 ms regression**
  at the 0.19 ms noise floor — the Triton router post-logits
  path is ~0.1 % slower than eager at proxy scale.
* The microbench → proxy discrepancy (1.5 ms predicted gain vs
  0.62 ms measured regression = 2.1 ms swing) is unattributed.
  Future re-attempts should run an Inductor-vs-microbench
  forensic before considering another flip-default-on attempt.
* Compressor APE elementwise fusion descoped at task-list
  refinement — below the R9.1 10 % cut-off, deferred to plan-8.

## 9. Follow-ups + commit pin

* P44 (next): V4 attention FWD epilogue (`out * scale + sinks`)
  absorbed into the FWD kernel.
* Plan-7 P45..P48: optimizer-step fusion sweep (the dominant
  residual at this point).
* Feature commit SHA: TBD-p43.
