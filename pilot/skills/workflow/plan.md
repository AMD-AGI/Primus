# Plan Structure

**Status**: v1 (single-node)
**Consumed by**: every Stage Worker that produces or consumes a Plan (Re-Plan emits; EXECUTE consumes; Diagnose annotates)
**Anchors**: `axis_taxonomy.md` · `replan.md` · `plan_graph.md`
**Schema**: `pilot/schemas/plan.schema.json`

A `Plan` is the unit of execution: one Plan → one `submit.run` invocation → one `RunSnapshot`. The Plan only encodes the **diff** against `base_plan + env_baseline`, never the full state — this keeps audit / reproduce / rollback fast.

---

## 1. Top-level fields

```yaml
plan_id:        r3_p2                   # see replan.md §8 for id rules
parent_baseline: r2_p1                  # plan_id this was derived from
parallelism:                            # one row per parallelism axis (all structural)
  tp:  4
  pp:  2
  dp:  16
  ep:  8
  vpp: 2
  cp:  1                                # context parallel; 1 = disabled
runtime:                                # per-run knobs
  mbs:                  2
  gbs:                  1024
  seq_length:           4096
  train_iters:          30
  recompute_granularity: selective
  recompute_method:      block
  recompute_num_layers:  2
comm:                                   # comm-specific knobs (subset of runtime, kept separate for legibility)
  bucket_size_mb:       64
  overlap_grad_reduce:  true
  overlap_param_gather: true
env:                                    # diff-only; baseline lives in ClusterProfile.env_baseline
  baseline_ref:         mi300x-16node-v3
  diff:
    NCCL_MIN_NCHANNELS:   16
    NCCL_BUFFSIZE:        16777216
    PYTORCH_HIP_ALLOC_CONF: "expandable_segments:True,max_split_size_mb:512"
predicted:                              # Execution Model estimate; informational, not a verdict
  tps:        18500
  mem_peak_gb: 165
  comm_ratio: 0.22
  confidence: 0.78
generated_by:
  bottleneck: PIPELINE_BOUND
  strategy:   skills/optimization/pipeline/vpp.md
```

---

## 2. env.diff merge rules

When EXECUTE materializes the actual env for the launcher, the rule is:

```
effective_env = ClusterProfile.env_baseline.rccl
              ∪ ClusterProfile.env_baseline.hsa
              ∪ ClusterProfile.env_baseline.alloc
              ∪ ClusterProfile.env_baseline.threading
              ∪ Plan.env.diff           # this overrides anything above
```

A Plan MAY only override env knobs that are recorded in `axis_taxonomy.md` §2.3 / §2.5. Unknown env knobs are rejected by `constraint.check_env`.

`Plan.env.baseline_ref` MUST match the active `ClusterProfile.env_baseline.version`; otherwise EXECUTE refuses to launch with `failure.kind = STRUCTURAL_INVALIDATION` (a stale env baseline is a structural change in disguise).

---

## 3. predicted block sourcing

`predicted` is populated by the **Execution Model** at Re-Plan time:

| Field | Source | Comment |
|---|---|---|
| `tps` | `T_step` estimate from `execution-model/compute.md`, `communication.md`, `pipeline.md` | inverted to throughput |
| `mem_peak_gb` | `M_param + M_grad + M_optim + M_act + M_buffer` per `execution-model/memory.md` | used by `constraint.estimate_mem` to early-reject |
| `comm_ratio` | `T_comm / T_step` | only used for `confidence` weighting |
| `confidence` | engine's per-axis band overlap + cluster-profile freshness | the same scalar that feeds `replan.md` §3 priority formula |

If the Execution Model is offline or untrusted (`pilot/state/calibration_state.yaml.drift_alarm == true`), `predicted` MAY be omitted; Re-Plan then falls back to flat priority (no `gain × confidence` scaling). See `execution_strategy.md` §4.

---

## 4. generated_by

For every Plan the engine MUST record:

| Field | What it stores | Why |
|---|---|---|
| `bottleneck` | the bottleneck class the candidate was generated to address | lets Settle / LEARN explain "why we tried this" |
| `strategy` | the Skill path most directly responsible for the move | enables `knowledge.write` to bind precipitated cases back to the right Skill |
| `derived_axis` (mirror of `PlanGraph.nodes[<id>].derived_axis`) | the (axis, value, type) tuple that produced this Plan from `parent_baseline` | de-duped via `plan_graph.md` §5 |

---

## 5. Single-node v1 simplification

The single-node engine (`pilot.tools.tune_single`) does **not** fully populate `predicted` (the Execution Model is calibration-tier `tentative`); it stores `predicted = null` and falls back to `replan.md` §3 with `confidence = engine_report.confidence`. The rest of the schema is honored verbatim.
