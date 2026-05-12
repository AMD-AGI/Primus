# Execution Strategy Selection

**Status**: v1 (single-node, decision-tree driven)
**Stage**: end of `OPTIMIZE_LOOP.REPLAN` (Strategy Select step in `replan.md` §2 step ⑥)
**Consumed by**: Re-Plan Stage Worker (and `pilot.tools.tune_single.replan` Strategy Select hook)
**Anchors**: `replan.md` · `axis_taxonomy.md`

Once Re-Plan has scored a CandidatePool, exactly **one** strategy decides how those candidates actually get to EXECUTE. This skill specifies the decision tree, the parameter defaults, and the fallback when calibration distrust signals trip.

---

## 1. Strategies

| Strategy | When to use | Mechanics |
|---|---|---|
| **Champion-Challenger** | Mixed pool with `cluster_shared` or `strongly_local` axes present, or model `confidence < 0.6`. | Run the champion AND each candidate in lockstep; promote on a 2%+ delta verified by the rerun. Safe but expensive. |
| **Per-Plan** | All candidates `weakly_local`, model `confidence ≥ 0.6`. | Run candidates without rerunning the champion; rely on the existing champion measurement for the baseline. Cheapest. |
| **Per-Plan + Pruning** | Many `strongly_local` candidates, model `confidence ≥ 0.7`. | Per-Plan with early-stop: if first 20% of iters lag champion by > 5%, kill the run. Avoids burning budget on obvious losers. |
| **Successive Halving** | Pool size > 4 AND budget is ample (`budget_remaining ≥ 4 × est_cost_gpu_h(pool)`). | Run all candidates at 25% budget, drop bottom 50%, double budget on survivors, repeat. Classic SH ratio (η = 2). |

---

## 2. Decision tree

```
pool_size = len(candidates)
has_cluster_shared = any(c.axis_meta.type == 'cluster_shared' for c in pool)
has_structural     = any(c.axis_meta.type == 'structural'     for c in pool)
confidence         = engine_report.confidence  # 0..1
budget_remaining   = target.budget.total_gpu_h - budget_used.gpu_h
est_cost           = sum(c.est_cost_gpu_h for c in pool)

if drift_alarm:                                 # see §4
    → Champion-Challenger                       # forced
elif has_cluster_shared:
    → Champion-Challenger                       # blast radius too large to skip champion rerun
elif pool_size > 4 and budget_remaining > 4 * est_cost:
    → Successive_Halving
elif has_structural or confidence < 0.6:
    → Champion-Challenger
elif confidence < 0.7:
    → Per-Plan
else:
    → Per-Plan + Pruning
```

---

## 3. Parameter defaults

| Symbol | Default | Strategy | Definition |
|---|---|---|---|
| `top_k` | `min(max_candidates, len(candidates))` | all | Number of candidates that actually reach EXECUTE. |
| `noise_band_pct` | **2.0** (%) | Champion-Challenger | Delta below which a result is `NOISE_BAND` and does NOT promote. Matches `settle.md ε_promote = 0.02`. |
| `prune_after_iter_pct` | **20** (%) | Per-Plan + Pruning | Fraction of `train_iters` after which pruning may fire. |
| `prune_regression_pct` | **5.0** (%) | Per-Plan + Pruning | If candidate iter time > champion × (1 + 0.05) at the prune checkpoint, kill it. |
| `halving_eta` | **2** | Successive Halving | Drop ratio per stage. |
| `halving_rungs` | **3** | Successive Halving | Stages until full budget. |

These are the **canonical defaults**; implementations source them from this document.

---

## 4. Calibration-distrust fallback

If `pilot/state/calibration_state.yaml` shows `drift_alarm: true`, every round forces **Champion-Challenger** regardless of the decision tree, until the alarm clears. Rationale: when the Execution Model's predictions are demonstrably wrong, Per-Plan loses its main advantage (trusting the prediction) and we MUST verify each candidate against the live champion.

---

## 5. Output

Strategy Select MUST populate `CandidatePool.selection` per `replan.md` §7:

```yaml
selection:
  strategy:   Champion-Challenger | Per-Plan | Per-Plan+Pruning | Successive_Halving
  pick_top_k: <int>
  selected:   [<id>...]
  rejected:
    - id:     <candidate id>
      reason: budget | duplicate | constraint | exhausted_neighborhoods | noise_band
```

A rejected candidate stays in `CandidatePool.candidates` for audit; only `selected` ids are submitted by EXECUTE.

---

## 6. Single-node v1 simplification

`pilot.tools.tune_single` ships a degenerate `Per-Plan` selector that simply takes `pool.candidates[:max_candidates]` (priority-sorted). It is the correct behavior for the typical single-node case (small pool, weakly_local axes dominant). The decision-tree branches above light up once the engine grows multi-node CandidatePool generation; the canonical thresholds in this document are already in force so the upgrade is purely additive.
