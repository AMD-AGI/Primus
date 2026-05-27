# Plan-8 P52 — HCA FWD tilelang (cr=128) — descoped at task-list refinement

> Phase summary written 2026-05-15 at P52 close-out.

**Status: descoped at task-list refinement per R9.1 / R9.3.  P50
demonstrated a 6× regression vs Triton at V4-Flash widths
(`head_dim=512` SMEM-budget gap); P52 extends the same kernel
with the HCA split-mask K-loop + pool-only additive bias and
inherits the same structural blocker.  HCA call sites continue
to route through the existing Triton path via the wrapper's
`_kernel_supports(...)` fallback (P50 design).**

---

## 1. Objective (originally scoped)

Extend the P50 dense FWD tilelang kernel with the `hca_local_seqlen
> 0` parameter so HCA layers (cr=128) can route through tilelang.
The kernel would:

* For the first `hca_local_seqlen` queries, hit kernel-native SWA on
  the local KV (first `hca_local_seqlen` keys).
* For the trailing queries, hit the compressed pool half of K/V with
  a caller-supplied `[Sq_local, Sk_pool]` additive bias.
* Apply the joint softmax + per-head sink at end-of-K.

## 2. Why the descope

Three independent observations:

### 2.1 P50 already demonstrated the V4-Flash regression

P50 microbench at V4-Flash widths (`progress/p50/bench/v4_flash.json`):

* Triton:   0.744 ms (90.9 TFLOP/s)
* Tilelang: 5.241 ms (12.9 TFLOP/s) = **0.14× speedup**

The HCA extension adds a constexpr branch + a second mask matrix
load to the same kernel; it does not change the underlying MFMA
schedule choice that causes the regression.  Per R9.1 a sub-10%
expected improvement does not warrant the implementation cost.

### 2.2 The Triton fallback already handles HCA

P50's `_kernel_supports(...)` predicate explicitly falls back to
Triton when `hca_local_seqlen > 0`:

```python
def _kernel_supports(*, sink, swa_window, additive_mask, hca_local_seqlen):
    if additive_mask is not None:
        return False  # P52 territory
    if hca_local_seqlen > 0:
        return False  # P52 territory
    return True
```

So HCA call sites ALREADY work correctly via the Triton path even
with `PRIMUS_V4_TILELANG_ATTN=1`.  No production-state regression
from descoping P52.

### 2.3 Plan-9 + upstream tilelang are the right scope

Closing the V4-Flash SMEM-budget gap requires upstream tilelang
support for programmable shared-memory partitioning (allowing
64×64 tiles at D=512 with partial Q/K/V residency).  That's a
plan-9 deliverable, not a plan-8 phase.

## 3. Code surface

No code change.  Documentation-only deliverable:

```
deepseek-v4/develop/progress/p52/
  + p52-summary.md (this file)
```

The plan-8 P52 status row in `progress/status.md` is marked `[-]`
per the R2.2 de-scope convention.

## 4. Follow-ups + commit pin

* When the V4-Flash SMEM gap closes (upstream tilelang or
  hand-tuned config), revisit P52 + ship the HCA extension via
  the constexpr `hca_local_seqlen` branch.
* P53 (HCA BWD) inherits the same descope rationale.
* Feature commit SHA: 19e41c9a.
