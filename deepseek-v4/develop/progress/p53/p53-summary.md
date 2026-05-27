# Plan-8 P53 — HCA BWD tilelang (cr=128) — descoped at task-list refinement

> Phase summary written 2026-05-15 at P53 close-out.

**Status: descoped at task-list refinement per R9.1 / R9.3 +
P52 precedent.  P51 demonstrated the same head_dim=512
SMEM-budget gap on the BWD side; the HCA BWD extension would
inherit the structural blocker.  HCA BWD call sites continue
to route through the existing Triton path via the wrapper's
`_kernel_supports(...)` fallback (P51 design).**

---

## 1. Objective (originally scoped)

Extend the P51 dense BWD tilelang kernel with the
`hca_local_seqlen > 0` parameter so HCA layers (cr=128) can
route through the tilelang BWD.  `dq` covers both local + pool
branches; two-pass `dkv_kernel` invocation for `dk_local /
dv_local` then `dk_pool / dv_pool`.

## 2. Why the descope

Same three observations as P52:

### 2.1 P51 already demonstrated the V4-Flash blocker

P51's BWD inherits P50's SMEM-budget gap (Q+K+V+dO+intermediate
fragments balloon past the 160 KiB MI355 budget at D=512).
The HCA BWD adds a second pass over K/V which makes the SMEM
situation strictly worse.  Per R9.1 the expected improvement is
negative.

### 2.2 The Triton fallback already handles HCA BWD

P51's `_kernel_supports(...)` predicate explicitly falls back
to Triton when `hca_local_seqlen > 0`.  So HCA BWD call sites
ALREADY work correctly via the Triton path even with
`PRIMUS_V4_TILELANG_ATTN=1`.

### 2.3 Sink BWD bf16 numerical issue (P51) compounds

P51 documented a bf16 `inf` at query 0 in the sink-BWD case.
HCA layers always use sink in production, so HCA BWD would
need the sink-BWD fp32-keep-intermediates fix from P51 BEFORE
the kernel extension can even run.  That's two coupled
deliverables; both are deferred to plan-9.

## 3. Code surface

No code change.  Documentation-only:

```
deepseek-v4/develop/progress/p53/
  + p53-summary.md (this file)
```

The plan-8 P53 status row in `progress/status.md` is marked
`[-]` per R2.2.

## 4. Follow-ups + commit pin

* Plan-9: revisit when (a) the SMEM gap closes and (b) the
  sink-BWD bf16 fix lands.
* Feature commit SHA: 19e41c9a.
