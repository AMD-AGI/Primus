# Settle (Convergence)

**Status**: v1 (single-node, PlanGraph-aware)
**Stage**: `OPTIMIZE_LOOP.SETTLE`
**Consumed by**: `pilot.tools.tune_single.settle` (canonical implementation), any Settle Stage Worker.

This file is the **single authoritative source** for the numeric thresholds and decision rules that decide whether a tuning round promotes a new champion, stays put, backtracks, or stops the loop. Implementations MUST source their constants from this document, not from in-code defaults.

---

## 1. Inputs

| Input | Source | Required |
|---|---|---|
| `round_results` | the candidates executed in the current round (`run_history` slice with `score`s) | yes |
| `plan_graph` | current `PlanGraph` (see `plan_graph.md`); contains `champion`, `frontier`, `metadata` | yes |
| `target_vector` | active `TargetVector` from `tuning_state.target` | optional (defaults: minimize `median_iter_time_ms`) |
| `budget_used` | `tuning_state.budget_used.{rounds, gpu_h, wallclock_h}` | yes |

---

## 2. Score function

The Settle skill compares plans by a **single scalar score** derived from the primary metric:

| `target.primary.metric` | `score(measurement)` |
|---|---|
| `median_iter_time_ms` (default; lower is better) | `1000.0 / median_iter_time_ms` (throughput proxy, higher is better) |
| `median_tflops` | `median_tflops` directly |
| `throughput_steps_per_s` | the same field |

A measurement is **scoreable** iff:

1. `measurement.status ∈ {completed, success, pass}`
2. `measurement.loss_finite == True`
3. The chosen metric is a finite positive number.

Otherwise the measurement is treated as `dead` and contributes only to `dead_rate_in_subtree`.

---

## 3. Promotion rules (per-round)

Let `cur` = current champion plan, `cur_score` = score of `cur.measurement`, `best`, `best_score` = the highest-scoring scoreable plan in `round_results`.

| Rule | Condition | Action on PlanGraph |
|---|---|---|
| **R1 Promote** | `cur_score is None` OR `best_score > cur_score × (1 + ε_promote)` | `champion = best.id`; old champion → `shelved`; append `{round, id: best.id}` to `champion_history`; reset `rounds_since_promotion = 0` |
| **R2 Marginal gain** | `cur_score < best_score ≤ cur_score × (1 + ε_promote)` | champion unchanged; `best` → `shelved` (kept as backtrack candidate); `rounds_since_promotion += 1` |
| **R3 No improvement** | `best_score ≤ cur_score` AND at least one scoreable result | champion unchanged; surviving plans go to `shelved`; `rounds_since_promotion += 1` |
| **R4 All regressed** | every result is `dead` (failure or non-scoreable) | trigger Backtrack (see §5); not a stop reason on its own |

`gain := best_score / cur_score − 1` (or `0` when `cur_score is None`).

---

## 4. Default thresholds (sourced by `tune_single.py`)

These are the **canonical defaults**. Implementations MAY tighten them when budget is short (see §6), but MUST NOT loosen without a documented override.

| Symbol | Default | Definition | Override knob |
|---|---|---|---|
| `ε_promote` | **0.02** (2%) | Min relative score gain required to promote new champion. | `target.budget.epsilon_promote` |
| `ε_stop` | **0.005** (0.5%) | Per-round gain below which a round counts as "stagnant". | `target.budget.epsilon_stop` |
| `stagnation_rounds` | **2** | Number of consecutive stagnant rounds that triggers STOP. | `target.budget.stagnation_rounds` |
| `dead_rate_backtrack` | **0.50** (50%) | Subtree dead-rate above which Backtrack fires (after 2 rounds). | `target.budget.dead_rate_backtrack` |
| `dead_rate_window_rounds` | **2** | Consecutive rounds the dead-rate threshold must hold. | n/a (tied to dead_rate_backtrack) |
| `explore_period_K` | **3** | Force one Explore round every K rounds. | `target.budget.explore_period_K` |

> **Rationale for ε_promote = 2%**: smaller than this is well inside per-iter noise on production training jobs (observed σ ≈ 0.3–0.7% on 30-iter windows in `pilot/state/sessions/.../deepseek_v2_lite_fp8_tuning_20260508_summary.md`). Promoting on smaller deltas leads to *NOISE_BAND* champions that don't survive re-runs.
>
> **Rationale for ε_stop = 0.5%**: the loop is allowed to keep trying until two rounds in a row deliver less than half a percent — beyond that, the marginal cost of another GPU·h is not worth the expected gain.

---

## 5. Escape-local-optima mechanisms

| Mechanism | Trigger | Action | Counter reset |
|---|---|---|---|
| **Backtrack** | `dead_rate_in_subtree[champion] > dead_rate_backtrack` for `dead_rate_window_rounds` rounds, OR `rounds_since_promotion ≥ 2` and every round result is below `ε_stop` | Demote champion to `shelved`; new champion = highest-priority node from `frontier \ {dead}` (typically the second-best shelved); `rounds_since_promotion = 0`; emit `{backtrack_reason}` in the Settle return. | `rounds_since_promotion` resets to 0; champion lineage continues from the chosen shelved node. |
| **Diversification bonus** | every Re-Plan scoring (see `replan.md` §3) | `novelty_bonus *= 1.20` when the candidate covers an axis not yet seen as `derived_axis` under the current parent. | n/a (per-candidate flag) |
| **Periodic Exploration Round** | `rounds_since_explore ≥ explore_period_K` AND a non-empty `shelved` set exists | Next Re-Plan derives ONLY from `shelved` (not from champion); the result is **never** treated as a stop signal even if regressive; `rounds_since_explore = 0` on exit. | Resets on next non-explore round if a promotion happens. |

The three mechanisms are not mutually exclusive: Backtrack handles sudden death, Diversification handles boiled-frog stagnation, Periodic Exploration is the long-term safety net.

---

## 6. Stop conditions

Settle returns `stop=True` when **any** holds:

1. **Target met**: all `target.constraints` are satisfied AND the primary metric did not improve in the most recent round.
2. **Stagnation**: depends on whether a `PlanGraph` is wired in.
   - **With PlanGraph (canonical, single-node v1+)**: `plan_graph.metadata.rounds_since_promotion ≥ stagnation_rounds` *after this round* (i.e. the engine predicts the counter's post-bump value: `+1` if not promoted, `0` if promoted). This is **per-round**, not per-candidate, so multi-candidate rounds aren't artificially counted as multiple "stagnant rounds".
   - **Legacy fallback** (no PlanGraph): the last `stagnation_rounds` entries of `run_history` all delivered `gain_vs_champion < ε_stop` AND no node in `frontier \ {dead}` has predicted priority > 1.0 (per `replan.md` §3 formula).
3. **Budget exhausted**: `budget_used.rounds ≥ target.budget.max_rounds` OR `budget_used.gpu_h ≥ target.budget.total_gpu_h` OR `budget_used.wallclock_h ≥ target.budget.wallclock_h`.
4. **No more candidates**: the most recent Re-Plan emitted an empty `CandidatePool`.

### 6.1 Periodic Exploration exemption

The Periodic Exploration Round (§5) **does not** count toward stagnation: if `is_explore_round=True` was passed to Settle, condition 2 is forced off for that round even if its `gain < ε_stop`. The engine signals this via `derive_policy == "explore"` (set by Re-Plan when `plan_graph.should_explore_round()` fires).

### 6.2 Shelved-reprieve (stop-deferral)

If Settle returns `stop=True` from condition 2 but **both** of the following hold, the engine **defers** the stop for one more round:
- `frontier` still contains at least one `shelved` node (i.e. there is still something to explore), AND
- the loop has not yet completed a Periodic Exploration Round (`plan_graph.metadata.explore_rounds_completed == 0`).

This guarantees at least one explore round actually gets to run before the loop declares stagnation. Once an explore round has fired and the loop is still stagnant, stop sticks. The deferral is implemented in the run-session driver (it inspects the PlanGraph), not in Settle itself; Settle's `stop` field still reports the un-deferred decision.

### 6.3 Backtrack signaling

When `PlanGraph` is provided and the round did not promote, Settle also checks `plan_graph.should_backtrack()` (which fires when `dead_rate_in_subtree[champion] > 0.5`, configurable). On fire, Settle emits a recommendation:

```yaml
backtrack:
  fired:        true
  reason:       "subtree dead-rate 1.00 > 0.50"
  new_champion: <plan_id-from-pick_backtrack_target>
```

Settle **does not** mutate the PlanGraph itself; the engine that owns the loop applies `plan_graph.promote(new_champion)` to rebase. Settle sets `stop=False` when Backtrack fires, since the rebase is a fresh chance.

---

## 7. Output schema

```yaml
status: success                # / failed
champion:        {<plan record>}
champion_id:     str
promoted:        bool
gain:            float          # best_score / cur_score - 1
stop:            bool
reason:          str            # "promoted new champion" / "kept champion" / "stagnation" /
                                 # "budget exhausted" / "backtrack triggered" / "no scoreable candidates"
plan_graph_delta:               # for audit; the engine applies these atomically
  champion_change:   bool
  promoted_to:       str | null
  shelved_added:     [plan_id...]
  dead_added:        [plan_id...]
  rounds_since_promotion: int
  rounds_since_explore:   int
  backtrack:
    fired:           bool
    reason:          str | null
    new_champion:    str | null
```

---

## 8. Single-node v1 status

For `cluster.yaml mode=single` running `pilot.tools.tune_single.settle`:

- **R1**, **R2**, **R3** promotion rules: implemented.
- **Stagnation** stop condition with both PlanGraph (canonical) and legacy paths: implemented (§6.2).
- **Periodic Exploration Round (K-cadence)** + explore-round stagnation exemption + shelved-reprieve: implemented (§6.1, §6.2).
- **Backtrack** rescue (signal from Settle, rebase applied by run-session): implemented (§6.3).
- **Diversification bonus** (novelty + stability): implemented inside `replan.md` §3 priority formula; no Settle hook needed.

`dead_rate_in_subtree` is now computed transitively from the PlanGraph (`pg._recompute_dead_rates`) every time a node is recorded.
