# Plan-7 P46 — Fused grad-scale Triton kernel (descoped via P45 evidence)

> Phase summary written 2026-05-15 at P46 close-out.

**Status: descoped via P45 evidence.  No kernel shipped.**

## 1. Objective (originally scoped)

Absorb the per-parameter `multi_tensor<scale>` calls into a single
Triton kernel.  Budget at plan-7 kick-off: ~5 ms / iter saved
(10.96 ms / iter total residual at the P40 anchor).

## 2. Why the descope

P45's microbench evidence (`progress/p45/p45-summary.md` §3)
established that:

* `torch._foreach_*` (and by extension `multi_tensor_apply`) is
  already a well-tuned multi-tensor kernel that runs at near-MI355
  HBM peak (~2000 GB/s on bf16 elementwise).
* The 321 `multi_tensor<scale>` launches per iter visible in the
  P40 trace are the **multi-tensor batches** for the 321
  parameter groups in V4-Flash — already collapsed, not individual
  per-tensor calls.
* The per-launch cost (34 µs) is dominated by HBM traffic on the
  target tensors, not launch dispatch overhead.

A Triton re-implementation would replicate `multi_tensor_apply`'s
existing multi-tensor batching with no headroom to extract.  Per
R9.1 the budget (~10.96 ms = 2.1 % of step) is well below the
10 % cut-off, and per P45's evidence the actual extractable win
is much smaller than the budget (~0.5 ms / iter at best).

## 3. Code surface

No code change.  Documentation-only deliverable:

```
deepseek-v4/develop/progress/p46/
  + p46-summary.md (this file)
```

## 4. Follow-ups

* Plan-8 fused-Adam Triton kernel (the production replacement for
  the TE / Apex multi-tensor optimizer path) would naturally
  absorb grad-scale into the same kernel.  No separate P46-style
  effort needed.
