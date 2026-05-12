# Diagnose (Bottleneck Classification)

**Status**: v1
**Stage**: `OPTIMIZE_LOOP.DIAGNOSE`
**Worker scope**: this file + `skills/optimization/<bottleneck>/SKILL.md` + `skills/workflow/axis_taxonomy.md`
**Tool boundary**: `pilot.tools.diagnose run` (rule engine; no LLM call)
**Authoritative schemas**: `schemas/run_snapshot.schema.json` (input), `schemas/diagnosis_report.schema.json` (output), `schemas/cluster_profile.schema.json` (peak reference)

---

## 1. Purpose

Map one terminal `RunSnapshot` (plus the `ClusterProfile` and the effective `Plan` that produced it) into a `DiagnosisReport` that tells Re-Plan **what bottleneck class to optimize for, which axes to vary, and which strategy to use**.

The Diagnose Worker NEVER changes a Plan and NEVER spawns runs. It is a pure classifier; everything actionable is downstream.

## 2. Inputs

| Required | Source | Used for |
|---|---|---|
| `RunSnapshot` (terminal) | `state/runs/<run_id>/snapshots/<ts>.yaml` | symptoms, iter_time, tflops, loss curve |
| `ClusterProfile` (current) | `state/cluster_profiles/.../<id>_<class>_<ver>.yaml` | hardware peak, intra/inter-node bandwidth |
| Effective `Plan` | `state/runs/<run_id>/plan.effective.yaml` | parallelism (TP/PP/EP/CP), MBS/GBS/seq, env_diff, MoE flags |

| Optional | Source | Used for |
|---|---|---|
| Champion `RunSnapshot` | last successful run referenced from `tuning_state.champion_id` | regression detection, deltas |
| `PlanGraph` | `state/sessions/.../plan_graph.yaml` | exhausted_neighborhood dedup hint |
| Profiler trace | `state/profiles/<run_id>.json` (when present) | comm_ratio / bubble_ratio / overlap_ratio |

If profiler trace is absent (current Megatron path), the engine falls back to **TFLOPs-vs-peak** as the comm/compute discriminator.

## 3. Output

A `DiagnosisReport` (schema: `diagnosis_report.schema.json`) with these required and key optional fields:

```yaml
snapshot_id: <run_id>
bottleneck: COMM_BOUND | PIPELINE_BOUND | MEMORY_BOUND | COMPUTE_BOUND
confidence: 0.0..1.0          # see §6
evidence: [<one rule per element, with the threshold and observed value>]
recommended_skills:           # paths into skills/optimization/...
  - skills/optimization/<bottleneck>/SKILL.md
  - skills/optimization/<bottleneck>/<specific>.md     # if any rule fired
env_suspect:                  # may be empty; non-empty triggers EnvSweep before structural Re-Plan
  - {flag: NCCL_BUFFSIZE, reason: ..., hint: skills/env/rccl.md#buffsize}
candidate_axes:               # consumed by Re-Plan; type drives strategy
  - {axis: turbo_deepep_num_cu, type: weakly_local, candidates: [80]}
  - {axis: micro_batch_size,    type: structural,   candidates: [12, 16]}
suggested_strategy: Per-Plan  # one of Champion-Challenger | Per-Plan | Successive_Halving
```

In addition to the schema-required fields, the engine emits a `meta` object with the raw signals (`tflops_pct_of_peak`, `bubble_ratio_estimate`, `mem_pct`, etc.) so reviewers and tests can audit how the verdict was reached. The `meta` block is informational; the agent must not consume it as ground truth.

## 4. Bottleneck taxonomy

Exactly one **primary** bottleneck per snapshot, picked by the rules in §5. A snapshot may also have **secondary** observations recorded in `evidence`, but they don't change the verdict.

| Class | Definition | Typical fix family |
|---|---|---|
| `COMM_BOUND` | The achievable throughput is bounded by collective bandwidth/latency (allreduce, alltoall, broadcast, allgather). | overlap, bucket size, fewer ranks per collective, alternative parallelism layout |
| `PIPELINE_BOUND` | Pipeline bubble dominates: many ranks idle waiting for an upstream stage. Only meaningful when `pp > 1`. | virtual pipeline (vpp), microbatch count, stage balance |
| `MEMORY_BOUND` | Run is mem-fragile: OOM fired, or `mem_pct > MEM_TIGHT_PCT`, or fragmentation high. The constraint is "fit, then go fast"; speed is secondary. | recompute (selective/full), offload, smaller MBS, expandable_segments |
| `COMPUTE_BOUND` | Achievable throughput is close to hardware peak; remaining headroom is intrinsic kernel efficiency. | bigger MBS, kernel tuning (FA, fused FFN), better dtype, parallelism re-shape |

When `RunSnapshot.status != completed` and a `failure_kind` is detected, the engine SHORT-CIRCUITS to the `failure_routing` table in §7 instead of using the bottleneck taxonomy. Bottleneck classification only applies to **terminal-success** runs.

## 5. Decision rules (eval order; first match wins)

The engine evaluates rules **top-to-bottom**. Each rule checks one condition; the first one that holds determines the bottleneck. Confidence is computed in §6.

> **Symbol convention.** All thresholds live in §10; rules below name them so this section reads as English, not magic numbers.

### Rule R0 — Failure short-circuit
```
if snapshot.status in {failed, killed, hung, unknown}:
    return failure_routing(snapshot, plan)   # see §7
```

### Rule R1 — Memory tightness (precedes everything else)
```
if rule.OOM_DETECTED:                                       # symptoms.oom_detected == true
    bottleneck = MEMORY_BOUND
    confidence = 0.95
    recommended_skills += memory/recompute.md, memory/offload.md
elif rule.MEM_FRAGMENTATION:                                # mem_reserved/mem_alloc > 1.4 (when present)
    bottleneck = MEMORY_BOUND
    confidence = 0.7
    env_suspect += PYTORCH_HIP_ALLOC_CONF
elif rule.MEM_TIGHT:                                        # peak_mem_pct > MEM_TIGHT_PCT (default 0.92)
    bottleneck = MEMORY_BOUND
    confidence = 0.8
```

Rationale: an OOM-prone run is unsafe to "speed up" further; we must first relieve mem pressure even if compute is also low.

### Rule R2 — Pipeline bubble (only meaningful when pp > 1)
```
if plan.pp > 1 and rule.BUBBLE_HIGH:
    bottleneck = PIPELINE_BOUND
    confidence = 0.85
```

`bubble_ratio` source priority:
1. profiler trace if present;
2. fallback estimate `(pp - 1) / (mbs_count_in_one_step + pp - 1)` where `mbs_count = gbs / (mbs * dp)`.

The fallback overestimates bubble for vpp-on plans; the engine flags this with a warning in `meta.bubble_ratio_source`.

### Rule R3 — Communication boundedness (the case the auto-heuristic missed)

Two paths, same conclusion:

**R3a (profiler available).**
```
if profile.comm_ratio > COMM_HIGH:
    bottleneck = COMM_BOUND
    confidence = 0.90
    recommended_skills += comm/<class>.md (alltoall|allreduce|allgather)
```

**R3b (profiler-less, the current Megatron path).**
```
peak = ClusterProfile.compute.peak_tflops_<dtype>          # bf16 or fp8 depending on plan
util = snapshot.tflops_steady / peak
if util < COMPUTE_PEAK_LOW and any_of([
        plan.uses_alltoall   (ep > 1 OR moe_routed_experts > 0),
        plan.uses_allreduce_heavy (tp > 1 AND tp_in_attention),
        plan.cp > 1
    ]):
    bottleneck = COMM_BOUND
    confidence = 0.75 + 0.05 if util < COMPUTE_PEAK_VERY_LOW else 0.75
```

`COMPUTE_PEAK_LOW` defaults to **0.30** (30% of measured peak). `COMPUTE_PEAK_VERY_LOW` to **0.15**. These are intentionally generous because FP8 path realism is ~10–25% of vendor peak on first pass.

This is the rule that classifies our `deepseek_v2_lite` baseline (240 / 1219 ≈ 19.6%, EP=8 → alltoall heavy → COMM_BOUND, confidence 0.80).

### Rule R4 — Compute headroom
```
if util >= COMPUTE_PEAK_HIGH (default 0.55):
    bottleneck = COMPUTE_BOUND
    confidence = 0.85
elif util in [COMPUTE_PEAK_LOW, COMPUTE_PEAK_HIGH):
    bottleneck = COMPUTE_BOUND        # default verdict when no other rule fires
    confidence = 0.55                  # lower confidence; encourages exploration
```

### Rule R5 — Regression vs champion (overrides R3/R4 only)
```
if champion_present and current.iter_ms > champion.iter_ms * REGRESSION_PCT:
    bottleneck = REGRESSION
    confidence = 0.95
    candidate_axes = [whatever axis_diff(plan, champion_plan) introduced]
    skip_replan_strategy = true        # report "this candidate is dead, do not derive from it"
```

`REGRESSION` is reported as a 5th synthetic class in `evidence` and `meta.bottleneck_extended`; the `bottleneck` field falls through to whichever R1–R4 rule fires next. This keeps the schema enum closed.

## 6. Confidence

```
confidence = base_for_rule
           - 0.10 if measurement_noise_cv > 0.05      # CV across reported iters
           - 0.10 if snapshot.iters_observed < 5       # too few samples
           + 0.05 if all symptoms clean (no nccl, cuda, python, hang)
clamped to [0.0, 1.0]
```

Two-decimal rounded. The schema requires the field; the engine MUST emit it even when degraded.

## 7. Failure routing (R0 short-circuit table)

| `RunSnapshot` status / symptom | `bottleneck` | `confidence` | `next` (suggested_transition.to) | `counts_against_budget` |
|---|---|---|---|---|
| `failed` AND oom_detected | `MEMORY_BOUND` | 0.95 | `OPTIMIZE_LOOP.REPLAN` (mark dead) | false |
| `failed` AND nccl_error | `COMM_BOUND` | 0.85 | `PREFLIGHT` (env_probe) | false |
| `failed` AND cuda_error | n/a (CLUSTER) | 1.00 | `PREFLIGHT` (full re-collect) | false |
| `failed` AND python_error AND traceback contains `INVALID_CONFIG` keywords | `INVALID_CONFIG` | 0.95 | `OPTIMIZE_LOOP.REPLAN` (mark dead) | false |
| `failed` AND loss_nan_or_inf | `NUMERICAL` | 1.00 | `ABORT` (escalate to human) | false |
| `hung` (snapshot.status hung OR symptoms.hang_suspected with `silent_for_s > hang_threshold_s`) | `HANG` | 0.90 | `OPTIMIZE_LOOP.REPLAN` (mark dead) | false |
| `killed` by orchestrator | `CANCELLED` | 1.00 | `OPTIMIZE_LOOP.REPLAN` (skip) | false |
| `unknown` | n/a | 0.30 | `WAIT` (re-snapshot) | false |

`bottleneck` values not in the schema enum (`INVALID_CONFIG`, `NUMERICAL`, `HANG`, `CANCELLED`, `REGRESSION`) are emitted into `meta.bottleneck_extended`; the schema-visible `bottleneck` field carries the closest legal mapping (`MEMORY_BOUND` for OOM, `COMM_BOUND` for nccl/hang, `COMPUTE_BOUND` as last-resort). The agent MUST read `meta` when `meta.bottleneck_extended != bottleneck`.

## 8. Re-entry triggers

A diagnosis can decide it has nothing useful to say without first re-collecting upstream artifacts. These edges DO NOT consume round budget (per pilot/README §12).

| Trigger | Re-entry target | Reason |
|---|---|---|
| `ClusterProfile.collected_at` older than `STALE_PROFILE_DAYS` (default 7 days) | `PREFLIGHT` (`reason=reentry_stale`) | hardware drift; thresholds in §5 may use wrong peak |
| `ClusterProfile.peak_tflops` differs from current plan's expected dtype peak by > 30% | `PREFLIGHT` (`reason=reentry_dtype_mismatch`) | profile measured for wrong dtype family |
| `Plan.modules.pre_trainer.overrides.{TP,PP,EP,CP}` not representable in `ClusterProfile.gpus_per_node × nodes_total` | `PROJECTION` | plan structurally invalid against current cluster |

The engine emits `suggested_transition.to = PREFLIGHT|PROJECTION|REPLAN` accordingly; the orchestrator decides whether to honor it.

## 9. `candidate_axes` — production rules

Each axis emitted by Diagnose carries a `type` from `axis_taxonomy.md` and a small `candidates` list to scan in Re-Plan. The engine emits axes only when **all three** hold:

1. The axis is **defined** in `skills/workflow/axis_taxonomy.md` (engine refuses unknown axes).
2. The axis is **legal** under the current plan's other constraints (e.g. `tp×pp×ep ≤ world_size`; `gbs % (mbs×dp) == 0`).
3. The axis is **not exhausted** in this neighborhood — i.e. `(parent_plan_id, axis, value)` not present in `PlanGraph.exhausted_neighborhoods`.

Axis-emission table by bottleneck:

| Bottleneck | First-line axes (`weakly_local` / `strongly_local`) | Holdout axes (`structural`) |
|---|---|---|
| COMM | `turbo_deepep_use_comm_stream`, `turbo_deepep_num_cu`, `NCCL_BUFFSIZE`, `NCCL_MIN_NCHANNELS`, `RCCL_MSCCL_ENABLE`, `overlap_grad_reduce`, `overlap_param_gather`, `moe_shared_expert_overlap` | EP layout (`ep`), `cp`, `tp` (only when comm-class is `allreduce`) |
| PIPELINE | `microbatch_count`, `vpp`, `recompute_method` | `pp` itself (only with constraint check) |
| MEMORY | `recompute_granularity`, `recompute_method`, `PYTORCH_HIP_ALLOC_CONF`, `MOE_BUFFER_PCT`, `optimizer_offload` | `mbs ↓`, `seq_length ↓` |
| COMPUTE | `mbs ↑`, `gradient_accumulation_fusion`, `MOE_PERMUTE_FUSION`, `attention_kernel`, FP8 toggles | `tp` (only when fits with cohort) |

Each axis row also includes `expected_gain_band: [pct_low, pct_high]` and `est_cost_gpu_h`; these feed Re-Plan's priority formula in pilot/README §7.4.

## 10. Threshold table (single source of truth)

| Symbol | Default | Meaning | Source |
|---|---|---|---|
| `OOM_DETECTED` | snapshot.symptoms.oom_detected | binary | observe |
| `MEM_FRAGMENTATION` | mem_reserved / mem_alloc > 1.4 | float ratio | profiler (optional) |
| `MEM_TIGHT_PCT` | 0.92 | peak GPU mem fraction | snapshot.metrics.history (computed) |
| `BUBBLE_HIGH` | 0.15 | profiler.bubble_ratio OR fallback estimate | profiler / model |
| `COMM_HIGH` | 0.25 | profiler.comm_ratio | profiler |
| `COMPUTE_PEAK_LOW` | 0.30 | tflops_steady / peak_tflops | snapshot + cluster_profile |
| `COMPUTE_PEAK_VERY_LOW` | 0.15 | confidence-bump threshold | derived |
| `COMPUTE_PEAK_HIGH` | 0.55 | enough utilization to call it COMPUTE_BOUND | derived |
| `REGRESSION_PCT` | 1.05 | iter_ms vs champion | derived |
| `STALE_PROFILE_DAYS` | 7 | re-entry trigger age | derived |
| `MEASUREMENT_NOISE_CV` | 0.05 | CV across reported iter_time_ms | derived |

All defaults can be overridden via `pilot/state/thresholds/diagnose.yaml` (an optional file the engine reads if present). This decouples policy tuning from code.

## 11. `suggested_strategy` mapping

| Pattern in `candidate_axes` | Strategy | Why |
|---|---|---|
| ≥ 1 `cluster_shared` axis present | `Champion-Challenger` | the env baseline must be re-validated against the champion before structural changes |
| All axes `weakly_local` AND ≤ 4 axes total | `Per-Plan` | cheap; explore each independently |
| Mixed types AND total candidate count > 6 | `Successive_Halving` | budget-aware pruning; halve worst half each round |
| Only `structural` axes (no env axes available) | `Per-Plan` | with explicit warning that bigger structural moves are coming |

The engine emits one strategy and a one-sentence `meta.strategy_rationale`.

## 12. CLI contract

```
python -m pilot.tools.diagnose run \
    --snapshot     state/runs/<id>/snapshots/<ts>.yaml \
    --cluster-profile state/cluster_profiles/.../<id>.yaml \
    --plan         state/runs/<id>/plan.effective.yaml \
    [--champion-snapshot state/runs/<champ_id>/snapshots/<ts>.yaml] \
    [--plan-graph    state/sessions/<sid>/plan_graph.yaml] \
    [--profile       state/profiles/<id>.json] \
    [--thresholds    state/thresholds/diagnose.yaml] \
    [--out           state/sessions/<sid>/<NN>_diagnose.yaml]
```

Exit codes match the rest of `pilot.tools.*`:

| Code | Meaning |
|---|---|
| 0 | success (DiagnosisReport printed to stdout AND optionally written to `--out`) |
| 1 | stage-failed (R0 path returned a failure_kind that requires escalation) |
| 2 | usage error (missing/invalid input) |
| 3 | TOOL_ERROR (NotImplementedError / dep missing / schema validation failed on output) |

When called from the Stage Worker, the wrapper translates the exit code into a `SubagentResult` envelope (per pilot/README §8.11).

## 13. Worker protocol

The DIAGNOSE Worker, when invoked by the Orchestrator:

1. Loads this file + `axis_taxonomy.md` + the `optimization/<bottleneck>/SKILL.md` for its provisional guess.
2. Calls `pilot.tools.diagnose run` with the four inputs above; trusts its output as the source of truth for `bottleneck`, `confidence`, `evidence`, `env_suspect`, `candidate_axes`, `suggested_strategy`.
3. If the engine returns `meta.bottleneck_extended in {INVALID_CONFIG, NUMERICAL, HANG, CANCELLED, REGRESSION}`, the Worker MUST set `SubagentResult.suggested_transition.to` per §7 instead of REPLAN.
4. Writes `<NN>_diagnose.yaml` to the session dir; references it in `tuning_state.stage_history`; returns a `SubagentResult` whose `summary.headline` is `<bottleneck> conf=<x.xx> via <rule_id>` (≤ 200 tokens, schema-checked).

The Worker DOES NOT consult the LLM beyond reading the recommended skills — the engine output is deterministic given fixed inputs. This is intentional: DIAGNOSE is the most reproducibility-critical stage.

## 14. Limitations and follow-up

- Bubble estimate falls back to a closed-form when no profiler trace is present; it overestimates for `vpp > 1`. Track `meta.bubble_ratio_source` and treat `=fallback` outputs as 0.7× weight.
- Memory fragmentation rule depends on profiler telemetry; without it, only OOM and `mem_pct` rules apply.
- `comm_ratio` rule (R3a) is profiler-gated; until profiler is wired, R3b carries the COMM determination.
- Engine produces axes; it does NOT score them numerically — that's Re-Plan's job (`expected_gain × confidence × novelty / cost`).

## 15. Reference: signal extraction from `RunSnapshot`

The engine derives the following quantities from `RunSnapshot` (input contract in `run_snapshot.schema.json`):

| Quantity | Computation | Notes |
|---|---|---|
| `iter_ms_steady_median` | `median(metrics.history.iter_time_ms[2:])` | drop first 2 iters (kernel compile) |
| `tflops_steady_median` | `median(metrics.history.tflops[2:])` | same drop |
| `iter_ms_cv` | `stdev(...) / mean(...)` over the same slice | feeds `MEASUREMENT_NOISE_CV` |
| `iters_observed` | `len(metrics.history.iters)` | gates confidence |
| `silent_for_s` | `progress.silent_for_s` | hang detection in §7 |
| `mem_pct` | from train.log lines `usage_ratio: <X>%` (parsed) | snapshot does NOT carry mem yet; engine re-tails the log via `--snapshot-log` if needed |
| `failure_kind` | `pilot.tools.constraint.diagnose_failure(snapshot)` | reuses existing helper |
