# Plan-7 P47 — Fused grad-norm clip Triton kernel (descoped via P45 evidence)

> Phase summary written 2026-05-15 at P47 close-out.

**Status: descoped via P45 evidence.  No kernel shipped.**

## 1. Objective (originally scoped)

Fuse the L2-norm reduce + max-with-existing-norm + clip-scale
derivation + apply-clip chain into a 3-kernel pipeline.  Budget
at plan-7 kick-off: ~10 ms / iter saved (`reduce<l2norm_bf16>`
7.76 ms + `multi_tensor<l2norm>` 6.72 ms = 14.48 ms total).

## 2. Why the descope

P45's microbench evidence (`progress/p45/p45-summary.md` §3)
established that the `torch._foreach_*` / `multi_tensor_apply`
path runs at near-MI355 HBM peak.  The trace's
`reduce<l2norm_bf16>` (7.76 ms / 12 launches × 647 µs each) and
`multi_tensor<l2norm>` (6.72 ms / 321 × 21 µs) are not launch-
overhead-dominated:

* The 647 µs per `reduce<l2norm_bf16>` launch is doing a real
  cross-tensor reduce — likely the global gradient L2 norm
  computation that runs once per `clip_grad_norm` call.
* The 321 `multi_tensor<l2norm>` launches per iter are the
  per-parameter-group L2 partials, already multi-tensor-batched.

A Triton re-implementation:

* For the per-parameter L2 partial reduce — would tie with
  `multi_tensor_apply` per P45's evidence.
* For the cross-tensor global reduce — would need a 3-kernel
  pipeline with inter-kernel sync, no obvious win vs the
  upstream implementation.

Per R9.1 the combined budget (14.48 ms = 2.8 % of step) is below
the 10 % cut-off, and per P45's evidence the actual extractable
win is much smaller than the budget.

## 3. Code surface

No code change.  Documentation-only deliverable:

```
deepseek-v4/develop/progress/p47/
  + p47-summary.md (this file)
```

## 4. Follow-ups

* Plan-8 fused-Adam Triton kernel could absorb grad-norm-clip
  into a fused optimizer-step megakernel.  Joint design point
  with P46.
