# Plan-7 — Optimizer-step elementwise / reduce fusion

Plan-7 attacks the dominant residual that the P40 trace exposed
**after** the plan-6 in-model elementwise sweep completed at
P41-P44: the Adam optimizer step + grad-scale + grad-norm clip
elementwise chain that consumes ~242 ms (~46 %) of every V4-Flash
EP=8 proxy step.

The candidate inventory lives in
`../progress/p41/p41-candidates.md` §3.2.

## At a glance

| phase | scope | target | est. savings |
| --- | --- | --- | ---: |
| **P45** | Custom Triton fused Adam kernel | absorbs BF16 ε-add + master multi-tensor | ~150 ms |
| **P46** | Fused grad-scale kernel | absorbs `multi_tensor<scale>` | ~5 ms |
| **P47** | Fused grad-norm clip kernel | absorbs `reduce<l2norm_bf16>` + `multi_tensor<l2norm>` | ~10 ms |
| **P48** | Plan-7 close-out | proxy bake-off + perf docs + status pinning | -- |

End-of-plan-7 EP8 proxy steady-iter target: **≤ 340 ms / iter,
≥ 790 TFLOP/s/GPU (P33-corrected denominator)**.  This finally
crosses the plan-6 original goal of 310 ms set in `plan-6/01-roadmap.md`.

## Files

- `01-roadmap.md` — phase overview, dependency graph, milestones, top risks.
- `02-phase-details.md` — per-phase task breakdown.
- `03-test-strategy.md` — gate matrix (G47..G50).

## Out of scope (plan-7)

- New attention kernel work (plan-5 + plan-6 P42/P44 absorbed
  the model-side attention residual; further attention work
  belongs to a future plan-8).
- FP8 / FP4 / mxfp4 (separate plan).
- Long-context (1M-token) / multi-node EP / HF state-dict adapter
  (same as plan-5/6).
- Convergence run (plan-7 runs 10-iter smokes only).
