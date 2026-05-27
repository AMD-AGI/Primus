# Plan-6 P40 — Close-out

> Plan-6 close-out summary written 2026-05-15.  Cumulative perf
> numbers pinned against the `mi355-gpu-8` / `dev_primus_wenx_693`
> bake-off run from `2026-05-14 18:14`
> (`PRIMUS_EXP_NAME=p40_bakeoff_plan6_defaults`, 15-iter smoke with
> all plan-6 default knobs on).

---

## 1. Objective

P40 is the plan-6 hand-off:

* Pin the cumulative plan-6 EP=8 proxy iter time and TFLOP/s/GPU
  against a clean 15-iter bake-off with **every plan-6 default-on
  flag enabled** (P34, P35, P36, P37) and the two descoped knobs
  off (P38, P39).
* Record one row per shipped fusion in `develop/perf/elem_fusion.md`
  (the elementwise-fusion ledger introduced for plan-6).
* Append `P40 final` row to `develop/perf/proxy_ep8.md` so the
  proxy headline tracks the cumulative speedup vs the P28 anchor.
* Surface all six plan-6 env knobs in `run_deepseek_v4_flash_proxy.sh`
  with the production default for each (mirrors the plan-5 P32
  final precedent).
* Verify every plan-6 phase (P33..P40) has commit-SHA pins in
  `progress/status.md` per rule R2.4.
* Write this eight-section summary per rule R2.1.

---

## 2. Plan-6 deliverables shipped

| Phase | Kernel / change                                       | Default | Status                |
| ----- | ----------------------------------------------------- | ------- | --------------------- |
| P33   | TFLOP/s closed-form correction (SWA pair pruning + HC fn matmul) | n/a   | shipped (no runtime change; corrected denominator) |
| P34   | `_stack_grouped_linear_weight` Triton FWD/BWD         | **ON**  | shipped, 49.8 ms / iter win |
| P35   | `apply_interleaved_partial_rope` Triton FWD/BWD       | **ON**  | shipped, 5.0 ms / iter win  |
| P36   | `sinkhorn_normalize` Triton FWD/BWD                   | **ON**  | shipped, 11.2 ms / iter win |
| P37   | `HyperMixer.compute_weights` tail Triton FWD/BWD      | **ON**  | shipped, 2.9 ms / iter win  |
| P38   | `Indexer.forward` scoring Triton FWD/BWD              | off     | descoped: V4-Flash microbench regresses (eager cuBLAS einsum at ~28 TFLOP/s, BWD `atomic_add` contention) |
| P39   | V4 router post-logits Triton FWD/BWD                  | off     | descoped: microbench wins but the ~1 ms / iter gain submerges in EP=8 dispatch noise |
| P40   | close-out — perf docs, cumulative bake-off, status pinning | n/a | (this phase) |

**Four default-on kernels** ship as the plan-6 production state.
**Two opt-in kernels** (P38, P39) ship behind env knobs for
future tuning + small-shape paths; lm_loss bit-identical to eager
so the knobs can be flipped without correctness risk.

---

## 3. P40 final bake-off (cumulative plan-6 perf)

15-iter EP=8 proxy smoke with all plan-6 defaults on:

| iter | iter time (ms) | TFLOP/s/GPU | lm_loss     |
|-----:|---------------:|------------:|------------:|
|   3  |          677.7 |       395.4 | 1.116224E+01 |
|   4  |          515.8 |       519.5 | 1.093397E+01 |
|   5  |          510.8 |       524.6 | 1.034111E+01 |
|   6  |          520.1 |       515.3 | 9.957450E+00 |
|   7  |          512.4 |       523.0 | 9.516789E+00 |
|   8  |          511.3 |       524.2 | 8.916406E+00 |
|   9  |          512.8 |       522.6 | 8.451833E+00 |
|  10  |          510.9 |       524.5 | 8.100058E+00 |
|  11  |          510.0 |       525.5 | 7.852410E+00 |
|  12  |          510.2 |       525.3 | 7.266212E+00 |
|  13  |          509.5 |       525.9 | 7.307085E+00 |
|  14  |          509.9 |       525.6 | 7.066420E+00 |
|  15  |          510.1 |       525.4 | 6.950458E+00 |

**Steady-state means (iters 8-15, post-warmup, clean steady window):**

| Metric | Value |
| --- | --- |
| Mean iter time (iters 8-15) | **510.6 ms** |
| Best iter time (iter 13)    | **509.5 ms** |
| Mean TFLOP/s/GPU (iters 8-15) | **524.9** |
| Best TFLOP/s/GPU (iter 13)    | **525.9** |
| Peak HBM / rank | 172.3 GiB (59.84 %) |

**Cumulative speedup vs the plan-5 P28 anchor (8837.4 ms / iter):**

`8837.4 / 510.6 = 17.31x` (mean iters 8-15)
`8837.4 / 509.5 = 17.34x` (best, iter 13)

**Cumulative throughput vs P28 anchor (77.5 TFLOP/s/GPU):**

`524.9 / 77.5 = 6.77x` (mean)
`525.9 / 77.5 = 6.79x` (best)

---

## 4. Per-phase iter-time decomposition

| Step | iter time (ms) | delta vs prev (ms) | TFLOP/s/GPU | speedup vs P28 |
|------|---:|---:|---:|---:|
| P28 baseline                            | 8837.4 |     -- |  77.5 |  1.00x |
| P29 compiled Sinkhorn                   | ~8630  |  -207  |  79.1 |  1.02x |
| P30a dense SWA K-loop pruning           | 6437.2 | -2193  | 106.3 |  1.37x |
| P30b dense+HCA SWA K-loop pruning       | 4943.4 | -1494  | 138.4 |  1.79x |
| P31  CSA in-kernel top-K gather/scatter | 4317.0 |  -626  | 158.5 |  2.04x |
| P31b CSA BWD dense-local + sparse split |  964.8 | -3352  | 709.3 |  9.15x |
| P32  CSA FWD split + LSE merge          |  890.5 |   -74  | 768.4 |  9.92x |
| P32 RoPE-fix bf16 cast                  |  665.0 |  -225  |1029.8 | 13.29x |
| P32 final (defaults ON)                 |  603.3 |   -62  |1134.3*| 14.64x |
| P33  TFLOP/s closed-form correction     |  603.3 |     0  | 444.2 | 14.64x |
| P34  `_stack_grouped_linear_weight`     |  530.85|   -72  | 507.2 | 16.65x |
| P35  `apply_interleaved_partial_rope`   |  526.7 |    -4  | 513.3 | 16.78x |
| P36  `sinkhorn_normalize`               |  515.0 |   -12  | 520.4 | 17.16x |
| P37  `HyperMixer.compute_weights` tail  |  512.1 |    -3  | 521.4 | 17.26x |
| P38  Indexer (descoped, default OFF)    |  512.1 |     0  | 521.4 | 17.26x |
| P39  V4 router post (descoped, default OFF) | 513.1 |   +1  | 521.4 | 17.22x |
| **P40 final (plan-6 close-out)**        | **510.6** | **-2.5** | **524.9** | **17.31x** |

(* TFLOP/s/GPU at P32 final is the pre-correction denominator;
P33 onward use the closed-form-corrected denominator, so the
column is not directly comparable across the P32/P33 boundary.
The iter-time speedup column is the apples-to-apples one.)

**Plan-6 contribution (delta vs P32 final iter time):**
`603.3 - 510.6 = 92.7 ms / iter saved (15.4 %)`.

**Plan-6 contribution (delta vs P33-corrected baseline at 603.3 ms /
444.2 TFLOP/s/GPU):** `444.2 -> 524.9 TFLOP/s/GPU = +18.2 %
throughput`.

---

## 5. Roadmap target check

`plan-6/01-roadmap.md` originally set a "iter time <= 385 ms" close-
out target.  The plan-6 close-out arrives at **510.6 ms / iter
(best 509.5)** instead.  Gap rationale:

* The original target assumed P38 (Indexer) and P39 (router post-
  logits) would each contribute 20-40 ms / iter savings.  Both
  descoped because the eager paths already map to cuBLAS / Inductor-
  fused kernels that beat a generic Triton chain at V4-Flash widths.
* The remaining 125 ms gap to the 385 ms target sits in the four
  long-tail kernels still on the critical path (V4 attention
  FWD/BWD across `cr ∈ {0, 4, 128}`).  These are plan-7 scope:
  attention-kernel-level work (tensor-core-friendly tiling, SMEM
  budgeting, sparse-pool sharing across layers), not elementwise
  fusion.

The plan-6 small-op-fusion bucket is fully harvested; further iter-
time wins move to the attention kernel itself.

---

## 6. Code surface (plan-6 close-out)

| Path | Role |
| --- | --- |
| `run_deepseek_v4_flash_proxy.sh` | All 6 plan-6 env knobs surfaced (P34 / P35 / P36 / P37 default ON; P38 / P39 default OFF) with the descope rationale inline |
| `develop/perf/elem_fusion.md` | One row per fusion (P34..P39); R2.5 format `<ms> ms \| <tflops>`; cumulative perf summary appended |
| `develop/perf/proxy_ep8.md` | `P33..P39 + P40 final` rows pinned to the EP=8 proxy bake-off iter-time + TFLOP/s/GPU |
| `develop/progress/status.md` | Phase 33..40 cells `[x]` with commit SHAs (rule R2.4) |
| `develop/progress/p3X/p3X-summary.md` | Eight-section per-phase summary (rule R2.1) for every P34..P40 |
| `develop/progress/p40/p40-summary.md` | This file |

---

## 7. Follow-ups (plan-7 candidates)

| Source phase | Follow-up | Notes |
| --- | --- | --- |
| P37 | `HyperMixer.collapse` / `expand` matmul-adjacent residue | matmul-dominated; small returns unless attention BWD compresses first |
| P37 | `HyperHead.forward` shares rsqrt + linear + scale + base + sigmoid | could re-use `_hc_compute_tail_fwd_kernel` with `K_steps=1` |
| P38 | Tensor-core-friendly tile sizes for the Indexer Triton kernel | requires `tl.dot` shape work; FWD-only fusion at small shapes may win |
| P38 | Inline causal mask without the `tl.where(p+1)*compress_ratio - 1` chain | reduces register pressure on small Sq |
| P39 | Score-fn-specialised softmax BWD using Inductor's share-reduce trick | unlocks default-on for non-V4 softmax configs |
| P39 | Dispatcher input fusion (router scatter -> `permute(probs)` -> `index_select(routing_map)` -> dispatcher input) | removes materialised `[N, E]` routing_map |

These are plan-7 scope: the elementwise-fusion bucket is exhausted
for plan-6.

---

## 8. Commit pin

```
commit b08975bc
Date:   2026-05-15

docs(deepseek-v4)[plan-6][P40]: plan-6 close-out -- proxy bake-off + perf docs + status pinning
```
