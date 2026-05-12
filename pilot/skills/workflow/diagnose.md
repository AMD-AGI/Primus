# Diagnose (Trace-driven Bottleneck Classification)

**Status**: v2
**Stage**: `OPTIMIZE_LOOP.DIAGNOSE`
**Worker scope**: this file + `skills/workflow/trace_analysis.md` + `skills/workflow/axis_taxonomy.md` + `skills/optimization/<bottleneck>/SKILL.md`
**Tool boundary**: `pilot.tools.diagnose run` — pure deterministic rule engine (no LLM)
**Authoritative schemas**: `schemas/diagnosis_report.schema.json` (output)

---

## 1. Purpose

Map one terminal training run into a `DiagnosisReport` that tells Re-Plan
**what bottleneck class to optimize for, which axes to vary, and which
strategy to use**. The Diagnose Worker NEVER changes a Plan and NEVER
spawns runs — it is a pure classifier; everything actionable is downstream.

The engine is deterministic: given the same inputs, the output is
bit-exactly reproducible.

Primary input is `<run_dir>/trace_analysis.json` produced by
`pilot.tools.trace_analyze` (per `skills/workflow/trace_analysis.md`),
which carries the four mutually-exclusive cost slices (pure_compute /
pure_comm / overlap / bubble), per-bucket and per-phase roll-ups, the
per-stream pipeline timeline, and a cross-bucket full kernel list.

Secondary inputs (`RunSnapshot`, `Plan`, `ClusterProfile`,
`champion_snapshot`) are **optional** — they enrich evidence (peak
comparison, regression detection, plan-axis legality) but never decide
the verdict on their own.

## 2. Inputs

| Required | Source | Used for |
|---|---|---|
| `trace_analysis.json` (one rank, steady iter) | `<run_dir>/trace_analysis.json` | every primary signal: cost_breakdown, ratios, kernels_full, phases, pipeline_timeline |

| Optional | Source | Used for |
|---|---|---|
| `RunSnapshot` (terminal) | `state/runs/<run_id>/snapshots/<ts>.yaml` | failure routing (§7), iter_ms cross-check, regression detection |
| `Plan` (effective) | `state/runs/<run_id>/plan.effective.yaml` | axis legality (`tp×pp×ep×cp ≤ world`, MoE flag mutex), `pp` for PIPELINE_BOUND |
| `ClusterProfile` | `state/cluster_profiles/.../*.yaml` | tflops % of peak (informational only; not a verdict driver) |
| `champion_snapshot` | per `tuning_state.champion_id` | regression vs champion |

If only `trace_analysis.json` is provided, the engine still produces a
fully usable verdict + first-line axes; structural axes (`tp`, `pp`,
`ep`, `mbs`) are emitted ONLY when a `Plan` is also provided so that
legality can be checked.

## 3. Output

A `DiagnosisReport`. All fields are populated; the trace evidence the
engine used appears under `meta.trace_evidence`:

```yaml
schema_version: "1.0"
snapshot_id: <run_id>
bottleneck: COMPUTE_BOUND     # one of COMM/PIPELINE/MEMORY/COMPUTE_BOUND
confidence: 0.85
evidence:
  - "R-COMP-A: compute_gemm_ratio=0.802 ≥ 0.70 (gemm-dominated steady iter)"
  - "R-OVRLP-A: pure_comm_pct=0.073 with overlap_pct=0.001 → comm cannot hide behind compute (no separate stream)"
recommended_skills:
  - skills/optimization/compute/SKILL.md
  - skills/optimization/comm/moe_overlap.md
env_suspect:
  - {flag: "turbo_deepep_use_comm_stream", reason: "MoE EP comm rides the compute stream", hint: "set true and re-profile"}
candidate_axes:
  - {axis: turbo_deepep_use_comm_stream, type: weakly_local, candidates: [true],   expected_gain_band_pct: [3, 8], rationale: "MoE comm and GEMM serialize on stream 0"}
  - {axis: turbo_deepep_num_cu,          type: weakly_local, candidates: [80, 96], expected_gain_band_pct: [1, 5], rationale: "MoE EP dispatch 7.2% of iter"}
  - {axis: micro_batch_size,             type: structural,   candidates: [12, 16], expected_gain_band_pct: [2, 6], rationale: "GEMM-dominated; bigger batch amortizes launch"}
suggested_strategy: Per-Plan
suggested_transition:
  to: OPTIMIZE_LOOP.REPLAN
  reason: "compute-bound with one obvious comm-overlap improvement; explore both"
  counts_against_budget: false
meta:
  rule_id: R-COMP-A
  bottleneck_extended: COMPUTE_BOUND
  trace_evidence:
    iter_wallclock_ms: 31999.30
    cost_breakdown: {pure_compute: 0.9205, pure_comm: 0.0727, overlap: 0.0007, bubble: 0.0061}
    bucket_top: [{compute_gemm_fp8_grouped: 0.5927}, {compute_gemm: 0.2096}, {comm_moe_dispatch: 0.0724}]
    streams_observed: 2
    phases: {fwd_pct: 0.038, bwd_pct: 0.960, optim_pct: 0.0002}
```

## 4. Bottleneck taxonomy

| Class | Definition | Typical fix family |
|---|---|---|
| `COMM_BOUND` | Iter wallclock is dominated by communication that doesn't overlap with compute. | overlap, dedicated stream, smaller-radius collectives, parallelism re-shape |
| `PIPELINE_BOUND` | Pipeline bubble dominates; many ranks idle waiting for upstream. Only meaningful when `pp > 1`. | virtual pipeline (vpp), microbatch count, stage rebalance |
| `MEMORY_BOUND` | OOM fired or memory tight enough to constrain choice. "Fit, then go fast." | recompute, offload, smaller MBS, expandable_segments |
| `COMPUTE_BOUND` | Iter wallclock is dominated by compute; comm and bubble are small. Headroom is intrinsic kernel efficiency or batch math. | bigger MBS, kernel tuning (FA, fused FFN), better dtype, parallelism re-shape |

A 5th synthetic class `REGRESSION` may appear in `meta.bottleneck_extended`
(only when a champion is provided AND `iter_ms > champion_iter_ms × REGRESSION_PCT`).
The schema-visible `bottleneck` field still falls through to the closest
COMM/PIPELINE/MEMORY/COMPUTE.

## 5. Decision rules (eval order; first match wins, with refinements)

The engine evaluates rules **top-to-bottom**. Each rule maps directly to
fields in `trace_analysis.json` so operators can audit the verdict by
opening the report side-by-side.

> **Symbol convention.** All thresholds live in §10; rules below name them
> so this section reads as English, not magic numbers.

### Rule R0 — Failure short-circuit (when `RunSnapshot` is present)
```
if snapshot.status in {failed, killed, hung, unknown}:
    return failure_routing(snapshot, plan)   # see §7
```

### Rule R1 — Memory tightness (precedes everything else)
```
if rule.OOM_DETECTED:                # snapshot.symptoms.oom_detected
    bottleneck = MEMORY_BOUND ; conf = 0.95
elif rule.MEM_TIGHT (mem_pct > MEM_TIGHT_PCT):
    bottleneck = MEMORY_BOUND ; conf = 0.80
```
Memory comes from `RunSnapshot` (or its train.log); the trace itself
doesn't carry it. R1 wins over the trace-driven rules — an OOM-prone
run is unsafe to "speed up" before relieving pressure.

### Rule R2 — Pipeline bubble (only when `pp > 1`)
```
if plan.pp > 1 and trace.bubble_pct ≥ BUBBLE_HIGH:
    bottleneck = PIPELINE_BOUND ; conf = 0.85
elif plan.pp > 1 and trace.bubble_pct ≥ BUBBLE_MED:
    bottleneck = PIPELINE_BOUND ; conf = 0.70
```
`bubble_pct` comes straight from `trace_analysis.cost_breakdown.bubble_pct`.
PP-less plans never hit R2.

### Rule R3 — Communication boundedness

The trace gives us THREE complementary measures, all of which can imply
COMM_BOUND. We require any TWO of three to fire (or the strongest one
alone with `comm_ratio ≥ COMM_VERY_HIGH`):

```
A. trace.ratios.comm_ratio ≥ COMM_HIGH
B. trace.cost_breakdown.pure_comm_pct ≥ PURE_COMM_HIGH
C. trace.cost_breakdown.longest_serialized_comm_ms ≥ SERIAL_COMM_HIGH
   AND longest_serial_comm / iter_wallclock ≥ 0.10

if A and trace.ratios.comm_ratio ≥ COMM_VERY_HIGH:
    bottleneck = COMM_BOUND ; conf = 0.90 ; rule_id = R3-A
elif sum([A, B, C]) ≥ 2:
    bottleneck = COMM_BOUND ; conf = 0.80 ; rule_id = R3-AB|R3-AC|R3-BC
elif A:
    bottleneck = COMM_BOUND ; conf = 0.70 ; rule_id = R3-A-weak
```

Sub-classification (drives `recommended_skills`):
- `comm_moe_dispatch_ratio > comm_collective_ratio` → MoE-EP class →
  `skills/optimization/comm/moe_dispatch.md`
- `comm_collective_ratio` dominant AND plan has `tp > 1` → TP-allreduce
  class → `skills/optimization/comm/tp_allreduce.md`
- `comm_collective_ratio` dominant AND plan has `cp > 1` → CP-class →
  `skills/optimization/comm/cp_overlap.md`

### Rule R4 — Compute boundedness (default verdict)
```
if trace.compute_gemm_ratio ≥ COMPUTE_GEMM_HIGH AND trace.bubble_pct < BUBBLE_LOW:
    bottleneck = COMPUTE_BOUND ; conf = 0.85 ; rule_id = R4-A
elif trace.compute_ratio ≥ COMPUTE_HIGH AND trace.comm_ratio < COMM_LOW:
    bottleneck = COMPUTE_BOUND ; conf = 0.75 ; rule_id = R4-B
else:
    bottleneck = COMPUTE_BOUND ; conf = 0.55 ; rule_id = R4-DEFAULT
```

The default (R4-DEFAULT) is intentionally low-confidence — it tells
Re-Plan to explore broadly rather than commit to a strategy.

### Rule R5 — Regression vs champion (overlay; runs AFTER R1..R4)
```
if champion present and trace.iter_wallclock_ms > champion.iter_ms × REGRESSION_PCT:
    meta.bottleneck_extended = REGRESSION
    confidence = 0.95
    candidate_axes = axis_diff(plan, champion_plan)   # only what changed
    suggested_transition.to = OPTIMIZE_LOOP.REPLAN
    suggested_transition.hint = "this candidate is dead; do not derive from it"
```
The schema `bottleneck` field stays whatever R1..R4 picked.

## 6. Confidence

```
confidence = base_for_rule
           - 0.10 if iter_wallclock too short (< 100 ms; sample ≪ steady)
           - 0.10 if trace.warnings is non-empty
           + 0.05 if R3 fires with all 3 sub-signals (A AND B AND C)
clamped to [0.0, 1.0], rounded to two decimals
```

## 7. Failure routing (R0 short-circuit table)

| Snapshot status / symptom | `bottleneck` | `confidence` | `next` (suggested_transition.to) | `counts_against_budget` |
|---|---|---|---|---|
| `failed` AND oom_detected | `MEMORY_BOUND` | 0.95 | `OPTIMIZE_LOOP.REPLAN` (mark dead) | false |
| `failed` AND nccl_error | `COMM_BOUND` | 0.85 | `PREFLIGHT` (env_probe) | false |
| `failed` AND cuda_error | n/a (CLUSTER) | 1.00 | `PREFLIGHT` (full re-collect) | false |
| `failed` AND python_error AND traceback ⊃ `INVALID_CONFIG` keywords | n/a (INVALID_CONFIG) | 0.95 | `OPTIMIZE_LOOP.REPLAN` (mark dead) | false |
| `failed` AND loss_nan_or_inf | n/a (NUMERICAL) | 1.00 | `ABORT` (escalate to human) | false |
| `hung` | n/a (HANG) | 0.90 | `OPTIMIZE_LOOP.REPLAN` (mark dead) | false |
| `killed` by orchestrator | n/a (CANCELLED) | 1.00 | `OPTIMIZE_LOOP.REPLAN` (skip) | false |
| `unknown` | n/a | 0.30 | `WAIT` (re-snapshot) | false |

Synthetic classes (`INVALID_CONFIG`, `NUMERICAL`, `HANG`, `CANCELLED`)
go in `meta.bottleneck_extended`; the schema-visible `bottleneck` carries
the closest legal mapping.

## 8. Re-entry triggers

| Trigger | Re-entry target | Reason |
|---|---|---|
| `ClusterProfile.collected_at` older than `STALE_PROFILE_DAYS` (default 7 days) | `PREFLIGHT` (`reason=reentry_stale`) | hardware drift |
| `Plan.modules.pre_trainer.overrides.{TP,PP,EP,CP}` not representable in cluster | `PROJECTION` | structurally invalid plan |
| `trace.warnings` contains `iter_boundary_fallback` (means `ProfilerStep` markers were missing AND we fell back to wallclock) | `WAIT` (`reason=reentry_unreliable_trace`) | iter window is approximate; verdict deferred |

## 9. `candidate_axes` — production rules (trace-evidence driven)

Each axis emitted by Diagnose carries a `type` from `axis_taxonomy.md`,
a small `candidates` list to scan in Re-Plan, an `expected_gain_band_pct`
band, and a `rationale` referencing the trace evidence that produced it.

The engine emits axes only when **all three** hold:
1. The axis is **defined** in `skills/workflow/axis_taxonomy.md`.
2. The axis is **legal** under the current plan's other constraints
   (only checked when `Plan` is provided; if not, `structural` axes are
   omitted).
3. The axis is **not exhausted** (only checked when a `PlanGraph` is
   provided).

### 9.1 Axis-emission table (trace-driven)

| Trace evidence pattern | candidate_axis | type | expected_gain_band_pct |
|---|---|---|---|
| `comm_moe_dispatch_ratio ≥ 0.05` AND `overlap_pct < 0.02` AND only 1–2 GPU streams observed AND comm_moe_dispatch shares stream with compute_gemm | `turbo_deepep_use_comm_stream` (true) | weakly_local | [3, 8] |
| `comm_moe_dispatch_ratio ≥ 0.05` (always when MoE EP active) | `turbo_deepep_num_cu` ([64, 80, 96]) | weakly_local | [1, 5] |
| `comm_collective_ratio ≥ 0.05` AND `overlap_pct < 0.02` (grad-reduction not overlapped) | `overlap_grad_reduce` (true) | weakly_local | [1, 4] |
| `comm_collective_ratio ≥ 0.05` AND `overlap_pct < 0.02` AND plan has DP-zero param-gather | `overlap_param_gather` (true) | weakly_local | [1, 3] |
| `compute_fp8_prep_ratio ≥ 0.03` (FP8 amax/cast hot) AND ≥ 50% of those kernels are amax/scale | `MOE_PERMUTE_FUSION` (true) | strongly_local | [2, 6] |
| `compute_attention_ratio ≥ 0.05` AND attention p99 > 1.5 × p50 | `attention_kernel` ([flash, sdpa]) | weakly_local | [1, 5] |
| `compute_other_ratio ≥ 0.05` AND `compute_other.kernel_count > 10000` AND avg_ms < 0.05 (kernel-launch overhead) | `gradient_accumulation_fusion` (true) | weakly_local | [1, 3] |
| `compute_gemm_ratio ≥ 0.70` AND `bubble_pct < 0.02` AND `comm_ratio < 0.10` (compute-bound, headroom from batch math) | `micro_batch_size` (up; e.g. [current+2, current+4]) | structural | [2, 6] |
| `comm_ratio ≥ 0.20` AND `compute_gemm_ratio < 0.40` AND plan.ep > 1 (EP-comm dominant) | `expert_model_parallel_size` (down) | structural | [3, 10] |
| `bubble_pct ≥ 0.10` AND plan.pp > 1 (PP bubble) | `virtual_pipeline_model_parallel_size` (up; e.g. [2, 4]) | structural | [3, 8] |
| `mem_pct > MEM_TIGHT_PCT` (from snapshot) | `recompute_granularity` (selective) | strongly_local | [1, 5] |
| `mem_pct > 0.95` (very tight) | `micro_batch_size` (down) | structural | [N/A — fit, not speed] |

When a rule is gated by `plan.<X>` but no `Plan` was passed, the rule
is skipped silently and a one-line warning is added to
`meta.skipped_axes` so the operator knows to provide a plan next time.

### 9.2 `env_suspect`

| Trigger | Flag | Hint |
|---|---|---|
| comm_moe_dispatch shares stream with compute_gemm | `turbo_deepep_use_comm_stream` | "set true and re-profile" |
| `comm_collective_ratio ≥ 0.05` AND `RCCL_MSCCL_ENABLE` not currently true | `RCCL_MSCCL_ENABLE` | "try true; algorithm pick can change collective shape" |
| `mem_pct ≥ 0.92` AND `PYTORCH_HIP_ALLOC_CONF` not set to `expandable_segments` | `PYTORCH_HIP_ALLOC_CONF` | "set `expandable_segments:True`" |

`env_suspect` items always also produce a `candidate_axes` entry of
matching type (`weakly_local`); the dual emission lets Re-Plan choose
between Champion-Challenger (env-only) and Per-Plan (mixed).

## 10. Threshold table (single source of truth)

| Symbol | Default | Meaning | Source |
|---|---|---|---|
| `OOM_DETECTED` | binary | snapshot.symptoms.oom_detected | observe |
| `MEM_TIGHT_PCT` | 0.92 | peak GPU mem fraction (when computable from snapshot) | snapshot |
| `BUBBLE_HIGH` | 0.10 | trace.cost_breakdown.bubble_pct | trace |
| `BUBBLE_MED` | 0.05 | trace.cost_breakdown.bubble_pct | trace |
| `BUBBLE_LOW` | 0.02 | trace.cost_breakdown.bubble_pct | trace |
| `COMM_HIGH` | 0.20 | trace.ratios.comm_ratio | trace |
| `COMM_VERY_HIGH` | 0.30 | trace.ratios.comm_ratio | trace |
| `COMM_LOW` | 0.10 | trace.ratios.comm_ratio | trace |
| `PURE_COMM_HIGH` | 0.10 | trace.cost_breakdown.pure_comm_pct | trace |
| `SERIAL_COMM_HIGH` | 5 ms | trace.cost_breakdown.longest_serialized_comm_ms | trace |
| `COMPUTE_HIGH` | 0.80 | trace.ratios.compute_ratio | trace |
| `COMPUTE_GEMM_HIGH` | 0.70 | trace.ratios.compute_gemm_ratio | trace |
| `OVERLAP_LOW` | 0.02 | trace.cost_breakdown.overlap_pct | trace |
| `REGRESSION_PCT` | 1.05 | iter_ms vs champion | derived |
| `STALE_PROFILE_DAYS` | 7 | re-entry trigger age | derived |

All defaults can be overridden via `pilot/state/thresholds/diagnose.yaml`
(an optional file the engine reads if present). This decouples policy
tuning from code.

## 11. `suggested_strategy` mapping

| Pattern in `candidate_axes` | Strategy | Why |
|---|---|---|
| ≥ 1 `cluster_shared` axis present | `Champion-Challenger` | the env baseline must be re-validated |
| All axes `weakly_local` AND ≤ 4 axes total | `Per-Plan` | cheap; explore each independently |
| Mixed types AND total candidate count > 6 | `Successive_Halving` | budget-aware pruning |
| Only `structural` axes (no env axes available) | `Per-Plan` | with explicit warning |

## 12. CLI contract

```
python -m pilot.tools.diagnose run \
    --trace-analysis state/runs/<id>/trace_analysis.json   \  # PRIMARY input (required)
    [--snapshot         state/runs/<id>/snapshots/<ts>.yaml] \
    [--plan             state/runs/<id>/plan.effective.yaml] \
    [--cluster-profile  state/cluster_profiles/.../<id>.yaml] \
    [--champion-snapshot state/runs/<champ_id>/snapshots/<ts>.yaml] \
    [--plan-graph       state/sessions/<sid>/plan_graph.yaml] \
    [--thresholds       state/thresholds/diagnose.yaml] \
    [--out              state/sessions/<sid>/<NN>_diagnose.json]
```

Exit codes:

| Code | Meaning |
|---|---|
| 0 | success (DiagnosisReport printed to stdout AND optionally written to `--out`) |
| 1 | stage-failed (R0 path returned a failure_kind that requires escalation) |
| 2 | usage error (missing/invalid input — including `--trace-analysis` not found) |
| 3 | TOOL_ERROR (NotImplementedError / dep missing / schema validation failed on output) |

## 13. Worker protocol

The DIAGNOSE Worker, when invoked by the Orchestrator:

1. Loads this file + `axis_taxonomy.md` + the
   `optimization/<bottleneck>/SKILL.md` for its provisional guess.
2. Calls `pilot.tools.diagnose run` with the required `--trace-analysis`
   plus any optional inputs that are available in the run dir; trusts
   the engine's output as the source of truth for `bottleneck`,
   `confidence`, `evidence`, `env_suspect`, `candidate_axes`,
   `suggested_strategy`.
3. If the engine returns `meta.bottleneck_extended in {INVALID_CONFIG,
   NUMERICAL, HANG, CANCELLED, REGRESSION}`, the Worker MUST set
   `SubagentResult.suggested_transition.to` per §7 instead of REPLAN.
4. Writes `<NN>_diagnose.json` to the session dir; references it in
   `tuning_state.stage_history`; returns a `SubagentResult` whose
   `summary.headline` is `<bottleneck> conf=<x.xx> via <rule_id>`.

The Worker DOES NOT consult the LLM beyond reading the recommended
skills — the engine output is deterministic given fixed inputs. This is
intentional: DIAGNOSE is the most reproducibility-critical stage.

## 14. Limitations and follow-up

- **Per-collective sub-classification is partial**. RCCL collapses all
  collectives into one device kernel name on ROCm; the per-collective
  type (`alltoall` vs `allreduce` vs `allgather`) lives in
  `args.Collective name`. `pilot.tools.trace_analyze` v0.5 does not
  parse this yet (planned for next round). Until then, R3 sub-class is
  derived from `comm_moe_dispatch_ratio` vs `comm_collective_ratio` plus
  the plan's `tp/pp/ep/cp` knobs.
- **Memory rules require the `RunSnapshot`**. The chrome trace itself
  doesn't carry mem_pct. When only `--trace-analysis` is provided, R1
  is skipped silently.
- **Champion regression check requires `--champion-snapshot`**. Without
  it, R5 is skipped.
- The engine **does not score** axes numerically; that's Re-Plan's job
  (`expected_gain × confidence × novelty / cost`).

## 15. Iteration log

| Round | Goal | Outcome |
|---|---|---|
| v2 | trace-driven decision tree on `trace_analysis.json`; every verdict carries 1–3 trace evidence lines; axes are produced from observed kernel patterns | shipped |
| todo | per-collective sub-buckets via `args.Collective name` (tell TP-allreduce apart from DP-allreduce apart from EP-alltoall) | open |
| todo | calibrate `expected_gain_band_pct` from observed Re-Plan outcomes (Re-Plan writes back actual gains so future Diagnose updates the bands) | open |
