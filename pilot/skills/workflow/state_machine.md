# State Machine

**Status**: PREFLIGHT row v1; remaining stages stub
**Read by**: Orchestrator (per `prompts/orchestrator.md` reading scope)

This document is the **sole authority** for Orchestrator routing decisions. Each stage row defines: trigger / inputs / produces / valid next states / on_fail mapping / counts_against_budget.

---

## States

```
PREFLIGHT → PROJECTION → SMOKE → BASELINE → CORRECTNESS → OPTIMIZE_LOOP → REPORT → LEARN
                                                                  │
                  OPTIMIZE_LOOP substates: OBSERVE → DIAGNOSE → REPLAN → EXECUTE
                                           → CORRECTNESS_LITE → SETTLE
                                           ↳ ENV_SWEEP (conditional, after SETTLE)
                                                                  │
Terminal: REPORT (success) · ABORT (failure escalation) · HANDOFF (context overflow)
```

---

## PREFLIGHT  (v1, fully specified)

| Field | Value |
|-------|-------|
| **Purpose** | Produce / refresh `ClusterProfile`; validate `env_baseline`; surface bad nodes. |
| **Authoritative protocol** | `skills/workflow/preflight.md` |
| **Worker prompt** | `prompts/worker/preflight.md` |
| **Tool** | `pilot.tools.preflight.{run, env_probe}` |
| **Subagent stage?** | Yes — heavy Skill scope, ~15K context peak. |
| **Counts against budget?** | **No** (infrastructure stage; never consumes a tuning round). |

### Entry conditions

The Orchestrator enters PREFLIGHT when **any** of:

| Trigger | `reason` arg | Cache action |
|---------|--------------|--------------|
| New session, no cached profile for `cluster_id` | `bootstrap` | Full collect |
| Cached profile exists, `age <= 7d`, no invalidation event | (none — skip stage entirely; Orchestrator advances to PROJECTION using cache) | Cache hit |
| Cached profile exists, `age > 7d` | `bootstrap` (delta) | Delta-refresh (steps 4+5) |
| Reentry from `HANG` (NCCL/IB timeout) | `reentry_hang` | env_probe only |
| Reentry from `CLUSTER` (node down / driver) | `reentry_cluster` | Full re-collect |
| Diagnose flagged stale profile mid-loop | `reentry_stale` | Delta-refresh |
| User `--force` | `force` | Full re-collect |

### Inputs

`PreflightRequest` (see `skills/workflow/preflight.md` §3): `cluster_id` (required), `reason`, `target_version?`, `force?`, `delta_only?`, `max_wallclock_s`, `blacklist_path`, `reentry_context?`.

### Produces

- **Required**: `ClusterProfile` artifact at `state/cluster_profiles/<cluster_id>_<version>.yaml` (schema: `cluster_profile.schema.json`).
- **Optional**: `BlacklistProposal` at `state/blacklist_proposals/<ts>.yaml` (schema: `blacklist.schema.json`).
- **Side effect on Orchestrator state**: when `env_baseline.version` changed, mark `calibration_state.version = new` and reset `effective_n = min(prior, 5)` (see §S1).

### Valid next states

| `SubagentResult.status` | Next state | Conditions |
|-------------------------|-----------|------------|
| `success` | `PROJECTION` | All gates passed; `env_baseline.status = validated`. |
| `tentative` | `PROJECTION` | At least one soft gate fired (peak 50–70%, `env_baseline.status = tentative`). Orchestrator should bump `ENV_SWEEP` priority for round 1 and consider re-PREFLIGHT after the first full round. |
| `failed` (`failure.kind = HANG`) | `PREFLIGHT` (retry once) → `ABORT` | One automatic retry with `reason=force` and `env_baseline.source = vendor_default`. Second failure → ABORT + escalate. |
| `failed` (`failure.kind = CLUSTER`) | `ABORT + escalate` | Humans must restore the cluster; not a self-healing case. |
| `failed` (`failure.kind = TOOL_ERROR`) | `ABORT + escalate` | Pilot implementation gap; not a tuning issue. |
| `failed` (`failure.kind = TIMEOUT`) | `ABORT + escalate` | Preflight should never take this long; investigate. |
| `failed` (other) | `ABORT + escalate` | Defer to humans. |

**No path back to PREFLIGHT from itself except the single retry above** — prevents infinite loops.

### Cache-hit fast path

When the Orchestrator can prove a fresh cached profile exists (read `state/cluster_profiles/_index.yaml`, check `age` and recent invalidation events), it **must not** spawn the Worker. Instead:

1. Set `tuning_state.cluster_profile_ref = <cached_path>`.
2. Append a synthetic `stage_history` entry: `{stage: PREFLIGHT, status: skipped_cache_hit, headline: "<n> nodes, peak <x>, cached <age>"}`.
3. Advance directly to `PROJECTION`.

This is the only stage with a "skip via cache" affordance.

### Reentry semantics from in-loop failures

When a downstream stage emits `failure.kind = HANG | CLUSTER | STRUCTURAL_INVALIDATION` per §12.2:

| failure.kind from downstream | Reentry target | reason arg | Mark prior profile |
|------------------------------|----------------|------------|-------------------|
| `HANG` | `PREFLIGHT` | `reentry_hang` | env_baseline only |
| `CLUSTER` | `PREFLIGHT` | `reentry_cluster` | Full superseded |
| `STRUCTURAL_INVALIDATION` | `PROJECTION` | (not preflight) | — |

`counts_against_budget=false` for all PREFLIGHT reentries.

---

## PROJECTION

**Status**: single-node v1 implemented via `pilot.tools.profiler.run`.

Inputs: ClusterProfile + Model Spec + base Primus exp YAML.
Produces: GPU inventory and per-config estimates used to seed the first candidate pool.

Valid next states:

| Result | Next state | Conditions |
|--------|------------|------------|
| `success` | `SMOKE` | At least one config estimate produced. |
| `failed(CLUSTER)` | `ABORT` | No GPU visible or torch unavailable on the training node. |
| `failed(TOOL_ERROR)` | `ABORT` | Profiler implementation/runtime error. |

---

## SMOKE

**Status**: single-node v1 implemented by `pilot.tools.tune_single run`.

SMOKE submits the base Primus exp with a short `train_iters` override, observes the terminal `RunSnapshot`, and fails fast on OOM, hang, NCCL/RCCL, CUDA/HIP, Python exception, or NaN/Inf loss.

Valid next states:

| Result | Next state |
|--------|------------|
| `success` | `BASELINE` |
| hard runtime symptom | `ABORT` or `OPTIMIZE_LOOP.DIAGNOSE` |

---

## BASELINE

**Status**: single-node v1 implemented by `pilot.tools.tune_single run`.

Runs the base plan for the configured short tuning window, writes the first `run_history` entry, and sets it as `champion`.

Valid next state: `CORRECTNESS` for explicit reference checks, or directly `OPTIMIZE_LOOP.DIAGNOSE` in the single-node runner.

---

## CORRECTNESS / CORRECTNESS_LITE

**Status**: single-node v1 implemented by `pilot.tools.observe.compare_loss`.

The first implementation is a short-window numeric health gate: finite loss, no hard symptoms, and latest loss delta within a configured percent of the reference snapshot/scalar. It is not yet a full token-aligned T0/T1/T2 curve comparison.

---

## OPTIMIZE_LOOP

**Status**: single-node v1 implemented by `pilot.tools.tune_single.{diagnose,replan,settle,run}`.

Loop order:

```
OBSERVE -> DIAGNOSE -> REPLAN -> EXECUTE -> CORRECTNESS_LITE -> SETTLE
```

Single-node v1 rules:

- `DIAGNOSE` uses `RunSnapshot` metrics and symptoms only.
- `REPLAN` emits a small priority pool over `micro_batch_size`, `tensor_model_parallel_size`, and recompute settings.
- `EXECUTE` runs candidates through `submit.run` in foreground mode.
- `SETTLE` promotes only if score improves by `epsilon_promote`; repeated gains below `epsilon_stop` stop the loop.

---

## REPORT / LEARN

**Status**: single-node v1.

`pilot.tools.report build --tuning-state <path>` summarizes champion overrides, baseline/champion improvement, run count, and referenced artifacts. `pilot.tools.knowledge.write` writes review-only drafts under `pilot/state/knowledge_drafts/`; it never writes directly to `skills/knowledge/`.

---

## Global rules (apply to every stage)

1. **`counts_against_budget` semantics**: only `OPTIMIZE_LOOP.SETTLE` increments `round_id` and `budget_used.rounds`. Every reentry path documented above is `counts_against_budget=false`.
2. **`SubagentResult.status` is the single source of routing truth** — Orchestrator never inspects artifact contents to decide transitions.
3. **`suggested_transition` is advisory** — Orchestrator may override via this state machine when `on_fail` rules dictate.
4. **All transitions are checkpoint-bounded** — checkpoint *before* the transition, never after.
