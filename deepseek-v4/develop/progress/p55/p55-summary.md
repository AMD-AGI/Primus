# Plan-8 P55 — CSA BWD tilelang (cr=4) — descoped at task-list refinement

> Phase summary written 2026-05-15 at P55 close-out.

**Status: descoped at task-list refinement per R9.1 / R9.3 +
P54 precedent.  CSA BWD inherits the same SMEM-budget gap +
adds the 3-kernel pipeline complexity (dq + dkv_local +
dpool_segreduce).  CSA BWD call sites continue to route
through the plan-5 P32 final Triton CSA BWD (with both
gather+atomic and segreduce variants behind
`PRIMUS_V4_CSA_BWD_SEGREDUCE`).**

---

## 1. Objective (originally scoped)

3-kernel BWD pipeline mirroring plan-5 P32 final's design:

* `dq_kernel` — re-materialise both branches' softmax from saved
  LSE; walk back through local + sparse + sink chains; emit dQ.
* `dkv_local_kernel` — dK_local / dV_local.  MQA-aware.
* `dpool_segreduce_kernel` — segreduce variant for dpool (sorted-
  inverse index, same pattern as plan-5 P32
  `PRIMUS_V4_CSA_BWD_SEGREDUCE`).

## 2. Why the descope

Three independent observations:

### 2.1 P51 demonstrated dense BWD SMEM blocker

P51's BWD inherits P50's V4-Flash SMEM-budget gap.  CSA BWD adds
TWO additional accumulators (sparse branch local fragments) +
the sorted-inverse-index segreduce pass; the SMEM situation is
strictly worse than dense.

### 2.2 Sink BWD bf16 numerical issue (P51) compounds

P51 documented a bf16 `inf` at query 0 in the sink-BWD case.
CSA layers always use sink in production, so CSA BWD would need
the sink-BWD fp32-keep-intermediates fix from P51 BEFORE the
kernel pipeline can even run.

### 2.3 Plan-5 P32 final segreduce design is mature

Plan-5 P32 final shipped TWO CSA BWD variants behind the
`PRIMUS_V4_CSA_BWD_SEGREDUCE` env knob: gather+atomic (default)
and segmented-reduction.  Both run at the SMEM-budget optimum
for the V4-Flash CSA shape envelope.  Reproducing that with
tilelang would land in the same 5-15× regression band P50/P51
exhibited.

## 3. Code surface

No code change.  Documentation-only:

```
deepseek-v4/develop/progress/p55/
  + p55-summary.md (this file)
```

The plan-8 P55 status row in `progress/status.md` is marked
`[-]` per R2.2.

## 4. Follow-ups + commit pin

* Plan-9: revisit CSA BWD tilelang once (a) the upstream SMEM
  partitioning lands, (b) the sink-BWD bf16 fix is in place,
  and (c) the CSA FWD tilelang (P54 follow-up) is shipped + wins
  at V4-Flash widths.
* Feature commit SHA: TBD-p55.
