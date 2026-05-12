# PlanGraph (Search-Space Tree)

**Status**: v1 (single-node)
**Stage**: persistent state, written at every Stage exit inside `OPTIMIZE_LOOP`
**Consumed by**: `pilot.tools.tune_single` (single-node engine), Re-Plan / Settle / Diagnose Stage Workers
**Anchors**: `settle.md`, `replan.md`
**Schema**: `pilot/schemas/plan_graph.schema.json`
**On-disk path**: `pilot/state/plan_graphs/<session_id>.yaml` (also referenced from `tuning_state.plan_graph_ref`)

This document is the **single authoritative source** for the tree data structure that backs Pilot's search guarantees. The flat `run_history` array is a *log*, not a *search space*; PlanGraph is the search space.

---

## 1. Why a tree, not a list

A flat list dedups by plan-id only and loses *derivation relations*: which baseline a candidate evolved from, which axis was moved, whether a runner-up may still be revivable. This causes two well-known failure modes:

1. **Premature convergence**: each round picks the best and discards runners-up, but a runner-up may open a different bottleneck door (e.g. PIPELINE_BOUND vs COMPUTE_BOUND). Greedy is forever stuck.
2. **Repeated search**: history dedupes by `plan_id`, but axes already tried at a similar value get retried after a re-derivation chain.

PlanGraph fixes both by maintaining (a) a parent pointer per node, (b) a `frontier` of derivable nodes, and (c) an `exhausted_neighborhoods` set keyed on `(parent, axis, value)`.

---

## 2. Node lifecycle

```
                            ┌──────────────┐
       Re-Plan emits ──────▶│ running       │
                            └──────┬───────┘
                                   │ Execute returns
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
        ┌──────────┐         ┌───────────┐        ┌─────────┐
        │ completed │         │ shelved   │        │ dead    │
        └────┬──────┘         └────┬──────┘        └─────────┘
             │                    ▲                permanent
   ★ champion │                    │                no derivation
             │  Settle demotion    │ Settle keeps     allowed
             ▼                    │ as backtrack
        ┌──────────┐               │ candidate
        │  shelved │───────────────┘
        └──────────┘
```

| Status | Semantics | Can derive new candidates? | Tps stored? |
|---|---|---|---|
| `running` | Submitted, not yet observed. | No (transient). | No |
| `completed` | Finished, scoreable, currently or formerly the champion. | Yes via `frontier` if champion or shelved. | Yes |
| `shelved` | Completed, was a runner-up; kept alive for Backtrack / Periodic Exploration. | Yes (default explore source). | Yes |
| `dead` | Permanent prune. Failed (`OOM` / `HANG` / `INVALID_CONFIG` / `NUMERICAL`) or otherwise unrecoverable. | **No** (forever). | Optional (last observed if any) |

The transition `completed → shelved` is the only "step down"; `dead` is one-way.

---

## 3. Required fields per node

Matches `schemas/plan_graph.schema.json` (descriptive copy here for the Worker; the schema is the binding contract).

```yaml
nodes:
  <plan_id>:
    parent:        <plan_id> | null      # null only for the BASELINE root
    status:        completed | shelved | dead | running
    tps:           <float>               # primary-metric score; absent if dead
    bottleneck:    COMM_BOUND | PIPELINE_BOUND | MEMORY_BOUND | COMPUTE_BOUND | UNKNOWN
    champion_at:   [<round_id>...]       # rounds at which this node was champion
    derived_axis:                        # the (axis, value) move that produced this node
      axis:        <axis_name>           # from skills/workflow/axis_taxonomy.md §2
      value:       <yaml-serializable>
      type:        weakly_local | strongly_local | structural | cluster_shared
    reason:        <free text>           # e.g. "OOM at step 47" / "promoted r2"
    measurement_ref: <path>              # pointer back into state/runs/<run_id>/
```

`derived_axis = null` only on the root (baseline). Multi-axis moves are encoded by composing several derived_axis entries via `derived_axis.composite = [{axis,value,type}, ...]`.

---

## 4. Top-level fields

```yaml
schema_version: "1.0"
session_id:     <str>
created_at:     <iso8601>
champion:       <plan_id>                # current baseline (pointer into nodes)
champion_history:                        # full timeline; lets Settle audit promotions
  - {round: 0, id: baseline}
  - {round: 2, id: r2_p4}
nodes:          {<plan_id>: <node>...}
frontier:       [<plan_id>...]           # currently derivable (= champion ∪ shelved)
exhausted_neighborhoods:                 # see §5
  - around: <plan_id>
    axis:   <axis_name>
    tried:  [<value>, ...]
metadata:
  rounds_since_promotion: <int>          # consumed by Settle stagnation check
  rounds_since_explore:   <int>          # explore-round trigger
  dead_rate_in_subtree:                  # plan_id → fraction in [0, 1]
    <plan_id>: <float>
```

`frontier` is **always re-derived** at write time from `{n.id | n.status ∈ {completed (champion), shelved}}`; never appended to without a corresponding status flip.

---

## 5. exhausted_neighborhoods semantics

Re-Plan MUST consult this set before emitting a candidate. A candidate is rejected if any of the following matches:

| Axis type | Match rule |
|---|---|
| `structural` | `parent == around` AND `axis == axis` AND `value` is an **exact** match (e.g. `mbs=14` differs from `mbs=15`). |
| `strongly_local` | exact match. |
| `weakly_local` | bucketed match for numeric values: within ±25% of any `tried` entry. For booleans / enums: exact. |
| `cluster_shared` | exact match scoped by `cluster_id`. |

The radius rules are also documented in `axis_taxonomy.md` §3. The implementation MUST consult `axis_taxonomy.md` for the axis type (`type` is also stored on the node itself for fast checks but the catalog is authoritative).

**Adding a row**: after a candidate runs (regardless of completed / dead), append `{around: parent, axis: derived_axis.axis, tried: [...,value]}` (or merge into an existing row).

---

## 6. PlanGraph operations (the engine's API surface)

These are the methods `pilot.tools.tune_single` (and any future Re-Plan / Settle worker) MUST call instead of mutating the dict directly:

| Operation | Effect |
|---|---|
| `add_node(plan_id, parent, derived_axis, status='running')` | Create a node; append to `nodes`; status starts `running`. |
| `record_result(plan_id, status, tps, bottleneck, reason)` | Transition `running → completed|dead`; update `frontier` if becomes `completed`. |
| `shelve(plan_id, reason)` | Transition `completed → shelved`; keep in `frontier`. |
| `promote(plan_id, round_id)` | Old champion → `shelved`; new champion = `plan_id`; append `{round, id}` to `champion_history`; reset `rounds_since_promotion`. |
| `mark_exhausted(around, axis, value)` | Append `(value)` to the matching row in `exhausted_neighborhoods` (create row if absent). |
| `is_exhausted(parent, axis, value, axis_type)` | Read-only check used by Re-Plan; honors §5 radius. |
| `frontier_excluding_dead()` | Returns shelved + current champion, sorted by `tps` desc; the search candidates Re-Plan can derive from. |
| `subtree_dead_rate(plan_id)` | Fraction of children-of-children-...-of `plan_id` that are `dead`. Used by Settle Backtrack rule. |
| `bump_explore_counter()` / `reset_explore_counter()` | Maintain `rounds_since_explore`. `reset_explore_counter` also bumps `explore_rounds_completed` (used by Settle's stop-deferral, see `settle.md` §6.2). |
| `bump_promotion_counter()` / `reset_promotion_counter()` | Maintain `rounds_since_promotion`. |
| `should_explore_round()` | Read-only: `rounds_since_explore ≥ explore_period_K`. Re-Plan calls this at the top of each round to switch its derivation source from champion → shelved. |
| `should_backtrack(dead_rate_threshold=0.5)` | Read-only: `dead_rate_in_subtree[champion] > threshold`. Settle calls this when no promotion happens. |
| `pick_backtrack_target()` | Returns the highest-tps `shelved` node in `frontier` (≠ current champion). Used by Settle to populate `backtrack.new_champion`. |

All operations are **pure on a deepcopy**: they return a new graph and never mutate in place. Callers persist the returned graph via the same atomic write protocol as `tuning_state.yaml`.

---

## 7. Persistence + atomicity

- One file per session: `pilot/state/plan_graphs/<session_id>.yaml`.
- Atomic write: write to `.tmp` then `os.replace`, identical to `state.checkpoint`.
- The file is referenced from `tuning_state.plan_graph_ref`.
- After every Settle exit the engine MUST also rewrite `frontier` and `dead_rate_in_subtree` (since both are derived).

---

## 8. Single-node v1 status

For `cluster.yaml mode=single`:

- ✅ **Day 1**: PlanGraph built alongside `run_history`. `tune_single.run_session` calls `add_node` / `record_result` / `promote` / `shelve` / `mark_exhausted`. Re-Plan honors `is_exhausted` and applies novelty + stability bonuses from the graph.
- ✅ **Day 2**: Backtrack wired into Settle (`should_backtrack` + `pick_backtrack_target`). When the champion's subtree dead-rate exceeds the threshold and no promotion happens this round, Settle emits a Backtrack recommendation and the run-session driver calls `promote(new_champion)` to rebase.
- ✅ **Day 3**: Periodic Exploration Round (`explore_period_K=3`). Re-Plan picks derivation source = highest-tps `shelved` node when `should_explore_round()` fires. Stop-deferral keeps the loop alive until at least one explore round runs (`settle.md` §6.2).
- ⏸ **Future (parking lot)**: cross-session reuse — load `plan_graph.yaml` from a prior session that ran on the same `(model_id, cluster_id)` and pre-populate `exhausted_neighborhoods` (zero-cost migration: avoid re-trying knobs already explored).

---

## 9. Cross-references

- Promotion / demotion thresholds: `settle.md` §3–§5.
- Priority formula consuming `derived_axis`: `replan.md` §3.
- Axis types + neighborhood radius rules: `axis_taxonomy.md` §1, §3.
- Schema: `pilot/schemas/plan_graph.schema.json`.
- README design source: §7 and §8.9.
