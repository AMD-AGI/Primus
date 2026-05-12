# Re-Plan (Candidate Generation)

**Status**: v1 (single-node, engine + legacy paths)
**Stage**: `OPTIMIZE_LOOP.REPLAN`
**Consumed by**: `pilot.tools.tune_single.replan` (canonical implementation), Re-Plan Stage Worker
**Anchors**: `plan_graph.md` · `axis_taxonomy.md` · `settle.md`
**Schema**: `pilot/schemas/candidate_pool.schema.json`

Re-Plan's job: given a `DiagnosisReport` and a `PlanGraph`, emit a **priority-weighted `CandidatePool`** that Strategy Select will trim before EXECUTE. This file specifies the priority formula, the derivation-source rules, the per-bottleneck Skill mapping, and the legacy fallback for tracedless callers.

---

## 1. Inputs

| Input | Source | Required |
|---|---|---|
| `diagnosis` | output of `pilot.tools.diagnose.run` (or `tune_single.diagnose`) | yes |
| `base_plan` | the Primus exp YAML being tuned | yes |
| `cluster` | the `cluster.yaml` loaded as a dict | yes |
| `plan_graph` | current `PlanGraph` per `plan_graph.md` | yes (single-node v1: a synthetic one from `run_history` is acceptable until full wiring lands) |
| `history` | flat `run_history` (legacy dedupe path) | optional |
| `round_id` | current round number | yes |
| `max_candidates` | top-K after Strategy Select | yes |
| `train_iters` | iter budget per candidate (engine pool: typically 20) | yes |

---

## 2. Seven-step generation flow

```
① Pick derivation source from PlanGraph
       default      = champion
       on stagnation = shelved (per settle.md §5 Periodic Exploration)
② Skill mapping by bottleneck (§4)
       COMM     → optimization/comm/*
       PIPELINE → optimization/pipeline/*
       MEMORY   → optimization/memory/*
       COMPUTE  → optimization/compute/*
       MOE      → optimization/moe/*
③ For each engine-emitted axis × value pair:
       translate via pilot.tools._axis_translator (catalog mapping
       axis → trainer-YAML key | structural diff | env var key)
④ Constraint check (constraint.check / check_env / estimate_mem)
       drop violators; record them in `rejected` for audit
⑤ exhausted_neighborhoods dedupe (per plan_graph.md §5)
       drop matches; record them in `rejected`
⑥ Strategy Select (per execution_strategy.md)
       axis taxonomy x confidence x budget → pick K
⑦ Output top-K into CandidatePool, sorted by priority desc
```

---

## 3. Priority formula (single source of truth)

For each candidate `c` derived from parent `p` along axis `a` with proposed value `v`:

```
priority(c) = gain_mid(a, v) × max(confidence_floor, confidence(c)) / cost_proxy(type(a))
            × novelty_bonus(c)
            × stability_bonus(parent(c))
```

| Symbol | Definition | Source |
|---|---|---|
| `gain_mid(a, v)` | `(lo + hi) / 2 / 100` where `(lo, hi)` is `expected_gain_band_pct` on the engine's axis entry; the engine guarantees a `[lo, hi]` band per axis. | `diagnose.md` §3 (engine output) |
| `confidence(c)` | `engine_report.confidence` (a per-DiagnosisReport scalar in `[0,1]`) | `diagnose.md` §5 |
| `confidence_floor` | **0.05** — protects against `confidence=0` zeroing out a sensible candidate. | this file |
| `cost_proxy(type)` | per-axis-type table (next sub-section) | `axis_taxonomy.md` §1 |
| `novelty_bonus(c)` | **1.20** if `c` covers an axis not yet present as `derived_axis.axis` under any sibling of `c` under the same parent; **1.00** otherwise. | `settle.md` §5 (Diversification mechanism) |
| `stability_bonus(parent)` | **1.10** if `parent.champion_at` has ≥ 2 entries (two-round consecutive champion); **1.00** otherwise. | this file |

### 3.1 cost_proxy table (canonical)

| Axis type | `cost_proxy` | Rationale |
|---|---|---|
| `weakly_local` | **1.0** | cheapest, no structural impact, no env-baseline change |
| `strongly_local` | **1.2** | still per-run but bigger blast radius (e.g. recompute, NCCL_MIN_NCHANNELS) |
| `structural` | **2.0** | invalidates memory predictions; `constraint.check` must rerun |
| `cluster_shared` | **3.0** | changes the env baseline; forces Champion-Challenger (per `execution_strategy.md`) |

These constants are sourced by `pilot.tools.tune_single._AXIS_COST_PROXY`; the symbol is treated as a constant and any change must update this table first.

### 3.2 Special-purpose priorities (not derived from the formula)

| Candidate kind | Priority | Why a flat priority |
|---|---|---|
| `env_suspect` not shadowed by any candidate_axis | `0.6 × confidence` | no `expected_gain_band_pct` (the engine only knows it as a *suspect*); a 60%-of-confidence weight balances "worth a 30-step probe" against the safer band-driven candidates. |
| Control rerun (rerun champion with current structure, no override) | **0.05** | Below every real candidate by design — only used when nothing else qualifies, to catch run-to-run noise. |

These constants live alongside the cost_proxy table in the implementation; see `tune_single.py` `_DEFAULT_ENV_SUSPECT_PRIORITY` / `_DEFAULT_CONTROL_PRIORITY`.

---

## 4. Skill mapping by bottleneck

Re-Plan reads only the **subtree** matching the diagnosed bottleneck. The Worker MUST NOT pull other subtrees (per the Worker reading-scope rule in `orchestration.md`).

| `engine_report.bottleneck` | Worker reading scope | Typical axes the scope contributes |
|---|---|---|
| `COMM_BOUND` | `skills/optimization/comm/*` + `skills/env/rccl.md` | `overlap_grad_reduce`, `overlap_param_gather`, `NCCL_BUFFSIZE`, `NCCL_MIN_NCHANNELS`, `RCCL_MSCCL_ENABLE`, `turbo_deepep_use_comm_stream`, MoE alltoall overlap |
| `PIPELINE_BOUND` | `skills/optimization/pipeline/*` | `pipeline_model_parallel_size`, `virtual_pipeline_model_parallel_size`, `micro_batch_size`, `global_batch_size` |
| `MEMORY_BOUND` | `skills/optimization/memory/*` + `skills/env/alloc.md` | `recompute_granularity`, `recompute_method`, `recompute_num_layers`, `optimizer_offload`, `PYTORCH_HIP_ALLOC_CONF` |
| `COMPUTE_BOUND` | `skills/optimization/compute/*` + `skills/env/hsa.md` | `micro_batch_size`, `tensor_model_parallel_size`, `attention_backend`, `MOE_PERMUTE_FUSION`, `turbo_deepep_num_cu` |
| MoE-flavored (any of above with `meta.moe == true`) | merge in `skills/optimization/moe/*` | `routing`, `dispatch`, `load_balance`, with the **two HARD CONSTRAINTS** (`moe_shared_expert_overlap=false`, `moe_router_force_load_balancing=true`) per `axis_taxonomy.md` §2.3 |

The trace-driven engine (`pilot.tools.diagnose`) already names the axes it wants in `candidate_axes[]`; Re-Plan's job is to honor that list verbatim, not to second-guess it. Skill mapping is for **fallback / supplementation**, not override.

---

## 5. Derivation source rules

| State of PlanGraph | Source for `derived_from.primary` | Source for `derived_from.secondary` | Policy tag |
|---|---|---|---|
| Normal (champion exists, no stagnation) | `champion` | `[]` | `exploit` |
| Stagnant (`rounds_since_promotion ≥ 1` AND last gain `< ε_promote`) | `champion` | first 1–2 entries of `frontier \ {champion, dead}` sorted by `tps` desc | `explore_exploit` |
| Periodic Exploration Round triggered (per `settle.md` §5) | the highest-`tps` `shelved` node | none (champion is excluded this round) | `explore` |
| Backtrack triggered (per `settle.md` §5) | the new champion picked by Backtrack | n/a | `exploit` (resets from the rescued branch) |

---

## 6. Legacy fallback (no trace, no engine_report)

When `diagnosis.meta.engine_report` is absent (SMOKE / BASELINE failure paths, or rocprof unavailable), Re-Plan falls back to a small hand-written rule set. This path MUST emit candidates with `axis_meta.source = "legacy.axis_name"` so downstream auditing can distinguish them.

Hand-written rules (kept tight to avoid drift):

| `diagnosis.candidate_axes` contains | Emitted candidate (priority) |
|---|---|
| `micro_batch_size` | `mbs+1` (priority **1.0**); if `mbs > 1`, also `mbs-1` (priority **0.95**) — both keep `gbs` valid against `world / (tp×pp×ep)` |
| `tensor_model_parallel_size` | `tp×2` AND `tp÷2` (priority **0.85**), keeping `world % (next_tp × pp × ep) == 0` |
| `recompute` OR `bottleneck ∈ {MEMORY, OOM}` | `recompute_granularity = full`, `recompute_method = block`, `recompute_num_layers ≥ 1` (priority **0.90**) |
| (always) | control rerun, priority **0.50** (legacy priority for a no-override re-run) |

The legacy "control rerun" priority of 0.50 differs from the engine-path 0.05 because in the legacy path the control rerun is often the only valid candidate (e.g. SMOKE failure with no actionable axes); on the engine path the control is purely a noise check.

---

## 7. Output schema (CandidatePool)

Pilot writes `pilot/state/candidate_pools/<session>_r<N>.yaml`. The shape conforms to `schemas/candidate_pool.schema.json`. Key fields:

```yaml
schema_version: "1.0"
round_id:        <int>
status:          ready | empty
diagnosis:       <DiagnosisReport reference or inline>
derived_from:
  primary:       <plan_id>
  secondary:     [<plan_id>...]
policy:          exploit | explore | explore_exploit
candidates:                      # sorted by priority desc, length ≤ max_candidates
  - id:            <hashed id, see §8>
    round_id:      <int>
    priority:      <float, the formula's output>
    reason:        <free text; "axis=value :: rationale">
    overrides:     {trainer YAML override key → value}
    env_overrides: {env var → str value}
    axis_meta:
      axis:        <axis name from axis_taxonomy.md>
      value:       <yaml-serializable>
      type:        <axis type>
      channel:     trainer_override | structural | env | control
      expected_gain_band_pct: [<lo>, <hi>]
      rationale:   <str>
      source:      engine.candidate_axes | engine.env_suspect | engine.control |
                   legacy.axis_name
selection:                       # output of execution_strategy.md
  strategy:        Champion-Challenger | Per-Plan | Per-Plan+Pruning | Successive_Halving
  pick_top_k:      <int>
  selected:        [<id>...]
  rejected:                       # for audit
    - {id, reason: <"constraint" | "exhausted_neighborhoods" | "below_budget" | ...>}
priority_formula:  "<the formula text, for self-documentation>"
source:            engine | legacy
```

---

## 8. Candidate id

Stable, human-readable, deterministic — collisions across rounds would corrupt PlanGraph. The reference rule (used by `tune_single._candidate_id`):

```
id = f"r{round_id}_c{idx}_{sha1('|'.join(sorted(f'{k}={v}' for k,v in overrides_for_id.items()))[:8]}"
```

where `overrides_for_id = plan_overrides ∪ {f'env::{k}': v for k, v in env_overrides.items()}`, ensuring env-only candidates also produce unique ids.

---

## 9. Cross-references

- Score / promotion: `settle.md`.
- Strategy Select (Champion-Challenger / Per-Plan / Successive Halving): `execution_strategy.md`.
- Axis catalog + radius rules: `axis_taxonomy.md`.
- Engine report shape: `diagnose.md` §3.
- README design source: §3.1 (Re-Plan sub-flow), §7.4, §8.10.
