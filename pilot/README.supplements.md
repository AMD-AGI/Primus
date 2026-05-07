# Primus Pilot — Supplementary Chapters (v2 supplements)

**Status**: Draft, to be merged into `README.md`

> This file drafts 4 chapters that the main v2 doc points at but does not expand. The §S1–§S4 numbering indicates "supplement"; once finalized, see the "Insert at" hint at the top of each section for the suggested merge point.

| § | Topic | Priority | Proposed insert point |
|---|-------|----------|----------------------|
| S1 | Execution Model Calibration | P0 | After §6, as §6.x subsection / or standalone §6A |
| S2 | Correctness Reference & Scale-aware Tolerance | P0 | Expansion of §3.2 CORRECTNESS, as §3.4 |
| S3 | Parallel Execution & Resource Protocol | P1 | Between §11 and §12, as new §11A |
| S4 | LEARN Governance & Knowledge Lifecycle | P1 | Expansion of §3.2 LEARN, as §11B (adjacent to integration chapter) |

---

## §S1. Execution Model Calibration (where `expected_gain` / `confidence` come from)

> **Problem**: the §7.4 priority formula depends on `expected_gain(c)` and `confidence(c)`. The §6 formula `T_step = T_comp + T_comm + T_bubble - T_overlap` is structural; it contains free parameters (`η_comp`, `BW_eff`, `α_overlap`, `β_bubble`) that the hardware peaks in ClusterProfile alone cannot fill in. If we guess these parameters, the §7 search structure degenerates into "random walk with metadata."
>
> **What this section does**: defines the calibration parameter set, the online update protocol, the confidence computation, and drift detection.

### S1.1 Parameterized Execution Model

Augment the §6 formulas with calibration parameters (**bold** = to be learned):

```
T_comp_pred(plan)    = (model_flops / num_gpus) / (peak_tflops × η_comp)
T_comm_pred(plan)    = Σ_i  msg_size_i / (BW_peak_i × η_comm_i)
T_bubble_pred(plan)  = (pp - 1) / (pp - 1 + M) × T_comp_pred × β_bubble
T_overlap_pred(plan) = α_overlap × min(T_comm_overlappable, T_comp_spare)
T_step_pred          = T_comp_pred + T_comm_pred + T_bubble_pred - T_overlap_pred

Mem_pred(plan)       = M_param + M_grad + M_optim + γ_act × M_act_theory + δ_buffer

calibration_params = {
    η_comp,                       # actual compute efficiency (incl. kernel / mem limits)
    η_comm[allreduce|alltoall|allgather|reduce_scatter],
    α_overlap,                    # how much of theoretical overlap is realized
    β_bubble,                     # bubble correction (warmup / unbalanced stages)
    γ_act,                        # actual activation memory / formula estimate
    δ_buffer,                     # workspace / fragmentation constant
}
```

**Why this decomposition**: each parameter corresponds to a known class of "theory vs reality" gap, and they are roughly orthogonal — updates do not contaminate each other.

**Scope**: `calibration_params` is bound to the triple `(cluster_id, model_family, framework_version)`. Switching cluster / model family / backend re-calibrates from scratch; same cluster across different `model_size` shares.

### S1.2 Calibration State (State-Layer persistence)

Add to the §8 schema set (recommended location: a standalone file next to ClusterProfile, to avoid polluting it):

```yaml
# state/calibration_state.yaml
calibration_id: mi300x-16node__llama__megatron-0.7
binding:
  cluster_id: mi300x-16node
  cluster_profile_version: mi300x-16node-v3
  model_family: llama_dense        # same family shares
  framework: megatron
  framework_version: "0.7.x"

params:                            # posterior means
  eta_comp: 0.71
  eta_comm:
    allreduce: 0.62
    alltoall:  0.48
    allgather: 0.55
    reduce_scatter: 0.58
  alpha_overlap: 0.73
  beta_bubble: 1.08
  gamma_act: 1.12
  delta_buffer_gb: 4.5

posterior:                         # used for confidence computation
  kind: gaussian                   # gaussian / bootstrap
  cov_diag:                        # per-parameter posterior variance
    eta_comp: 0.0009
    alpha_overlap: 0.0036
    ...
  effective_n: 47                  # effective sample count (with decay)

residuals:                         # last K=20 prediction errors (for drift detection)
  - {plan_id: r2_p4, predicted_tps: 17600, observed_tps: 17820, rel_err: +0.012}
  - {plan_id: r3_p1, predicted_tps: 18800, observed_tps: 17400, rel_err: -0.074}
  ...

freshness:
  last_updated: 2026-04-21T15:32:00Z
  observations_since_reset: 47
  drift_alarm: false               # see S1.5
```

### S1.3 Online update protocol (called after each Execute)

**Trigger**: after each plan completes Execute and Snapshot is written (after [Observe] in the §3.3 inner swimlane). **Do not** update from EnvSweep short runs (30–50 step has poor SNR).

```python
# tools/calibrate.py (new)
def calibrate.update(snapshot: Snapshot, plan: Plan,
                     state_path: str = 'state/calibration_state.yaml') -> CalibrationState:
    """
    Update calibration_params from one (predicted, observed) pair.
    Uses log-space weighted recursive least squares (WRLS):
      - log space: keeps η/α/β positive and multiplicative
      - weighted: recent observations weighted higher (exponential decay λ=0.9)
      - recursive: O(1) update, no full history retained
    """

    # 1) Residual decomposition: attribute total step_time error to each parameter
    components = decompose_step_time(snapshot, plan)
    # components = {
    #     T_comp_obs, T_comm_obs[type], T_bubble_obs, T_overlap_obs
    # }
    # Source: profiler trace (event categorization rules in trace.md);
    # if no profile (short run), fall back to total T_step residual,
    # updating only eta_comp + delta_buffer.

    # 2) Per-parameter WRLS update (with decay)
    for param, observed_ratio in residuals_per_param(components, plan):
        state.params[param] = wrls_update(
            prior_mean = state.params[param],
            prior_var  = state.posterior.cov_diag[param],
            observation = observed_ratio,
            obs_noise  = SENSOR_NOISE[param],   # supplied by profiling/trace.md
            decay      = 0.9,
        )

    # 3) Append to residuals (ring buffer, cap=20)
    state.residuals.append({...}); state.residuals = state.residuals[-20:]

    # 4) Drift detection (see S1.5)
    state.freshness.drift_alarm = detect_drift(state.residuals)

    state.checkpoint()
    return state
```

**Degradation paths** (when trace data is unavailable):

| Data availability | Updatable params | Frozen |
|-------------------|------------------|--------|
| Full profile trace | All | — |
| Step time + mem peak only | `η_comp` (mixed with `η_comm`, lumped into effective comp) + `γ_act` + `δ_buffer` | `α_overlap`, `β_bubble` not separable |
| Step time only | `η_comp_aggregate` (no per-collective split) | Rest frozen |
| OOM / failed | No param update; `γ_act` flagged as underestimated, more conservative next time |

### S1.4 Concrete computation of `expected_gain` / `confidence`

```python
# tools/predict.py (new)
def predict.gain(parent_plan, candidate_plan, calibration_state):
    """
    Returns: (predicted_tps, expected_gain_pct, confidence)
    """
    mu_step_parent = T_step_pred(parent_plan,    calibration_state.params)
    mu_step_cand   = T_step_pred(candidate_plan, calibration_state.params)

    predicted_tps = global_batch_tokens / mu_step_cand
    expected_gain = (mu_step_parent - mu_step_cand) / mu_step_parent

    # confidence: propagate parameter posterior variance to T_step via delta method
    var_step_cand = sum(
        (∂T_step / ∂param)**2 × posterior.cov_diag[param]
        for param in calibration_params
    )
    sigma_step = sqrt(var_step_cand)
    cv_step    = sigma_step / mu_step_cand     # coefficient of variation

    # confidence ∈ [0,1]: low CV → high confidence
    confidence = 1.0 / (1.0 + 5.0 × cv_step)

    # Discount further the farther we are from recent observations in axis space
    novelty_penalty = axis_distance(candidate_plan, recent_observations)
    confidence *= max(0.3, 1 - 0.4 × novelty_penalty)

    return predicted_tps, expected_gain, confidence
```

**Semantics of `confidence`** (when used as a multiplier in §7.4 priority):

| confidence range | Meaning | Re-Plan behavior |
|------------------|---------|------------------|
| > 0.7 | Model trustworthy | Use priority normally |
| 0.4 – 0.7 | Model advisory | Switch to Successive Halving, run multiple candidates in parallel |
| < 0.4 | Model untrustworthy | **Fall back to Champion-Challenger physical experiment**; predictions only used as OOM safety net |

`execution_strategy.md` implements strategy selection per these ranges.

### S1.5 Drift detection and invalidation

Calibration is not "fit once forever" — cluster upgrades, driver changes, model-family switches all invalidate old parameters. Detection protocol:

| Signal | Threshold | Action |
|--------|-----------|--------|
| Last 5 `rel_err` same sign | mean(\|rel_err\|) > 10% | `drift_alarm = true`; this Re-Plan **stops consuming `confidence`** — effectively degrades to strategy=Champion-Challenger |
| `cluster_profile.version` changed | any change | Mark entire calibration_state stale; `effective_n` decays to 5 to re-warmup |
| `framework_version` major bump | major segment changed | same as above |
| New model family | binding miss | Start new file; first Re-Plan caps `confidence` at 0.3 (conservative baseline) |
| `observations_since_reset > 200` AND largest child subtree dead_rate > 50% | compound | Trigger Re-Calibrate sub-flow: refresh micro-bench against latest BASELINE |

**Fallback after drift**: `expected_gain` is still produced by the formula, but `confidence < 0.4` — the §7.4 priority naturally down-weights, search degrades to physical-experiment-dominant. **Do not turn the Execution Model off entirely** — it still does the irreplaceable OOM estimation job.

### S1.6 Cold-start: what to do the first time

When a new (cluster, model_family, framework) triple has no observations:

1. **Use single-node profiling from the PROJECTION stage** (already in §3.2): backfill measured `T_comp` / `T_comm` (if ≥ 2 GPUs), giving zero-shot estimates of `η_comp` / `η_comm[*]`.
2. **For the rest**, use cluster-class defaults (`alpha_overlap=0.7`, `beta_bubble=1.05`, `gamma_act=1.10`, `delta_buffer_gb=4.0`), provided by `skills/execution-model/cluster_priors.md` (**new Skill**).
3. **After the first BASELINE completes, immediately run `calibrate.update()`**, lifting `effective_n` from 0 to 1.
4. **For the first 3 rounds**: cap all candidates' `confidence` at 0.6 (avoid cold-start over-trust).

### S1.7 Evaluation and regression

Add 3 rows to `§10 Evaluation metrics`:

| Dimension | Metric | Target |
|-----------|--------|--------|
| Model quality | Median rel-err of `T_step_pred` vs `T_step_obs` | ≤ 15% |
| Model quality | p90 rel-err of `Mem_pred` vs `Mem_obs` | ≤ 20% |
| Model quality | Calibration drift false-alarm rate (drift_alarm misfire / total) | ≤ 10% |

Methodology: use historical sessions' (predicted, observed) pairs as hold-out test.

### S1.8 Cross-references to existing §X

- `predict.gain()` replaces the "given" `expected_gain` / `confidence` in §7.4 priority.
- `calibration_state.yaml` joins §4.1 directory tree (sibling of `tuning_state.yaml`).
- §5 Tool interface table adds: `calibrate.update`, `predict.gain`, `predict.mem`.
- §8 schema set adds `calibration_state.schema.json`.
- §13.2 subagent boundary table: **Calibrate need not be a subagent** (O(1) math op, no large Skill reading).

---

## §S2. Correctness Reference & Scale-aware Tolerance (where references come from)

> **Problem**: §3 / §12.1 places "align with reference loss curve" right after BASELINE, but (1) on first bring-up the reference does not exist yet; (2) loss curves naturally morph with mbs / gas / scale — naive comparison flags "normal statistical noise" as "numerical bug."
>
> **What this section does**: define the reference source hierarchy, the scale-aware tolerance formula, and which round triggers which tier of gate.

### S2.1 Three-tier reference

| Tier | Source | Trust | When built | When used |
|------|--------|-------|------------|-----------|
| **T0 Anchor** | Single-node / dual-node deterministic FP32/BF16, 200 step, fixed seed, fixed data | Highest (treated as ground truth) | Before bring-up's first PROJECTION | To validate Tier-1 |
| **T1 Reference** | The BASELINE curve that passed the CORRECTNESS gate | High | At BASELINE, after T0 validation | LITE checks within OPTIMIZE_LOOP |
| **T2 Local** | Smoothed last-100-step curve of current round's champion | Medium | Maintained rolling in OPTIMIZE_LOOP | Detect dramatic regression (weak signal beyond hard gates) |

**Promotion rule**: T0 → T1 via the "equivalence-normalized" test in §S2.3. T1 cannot self-promote (prevents drift accumulation).

**First-bring-up protocol**:

```
PROJECTION
  └─ T0 Anchor must exist (CLI flag pointer / auto small-scale build) before start
       │
       ▼
  single-node profiling
       │
       ▼
SMOKE (tiny scale × 100 step)
       │
       ├── correctness_lite_gate(against=T0)   ← first numerical gate
       ▼
BASELINE
       │
       ▼
CORRECTNESS (full scale, align with T0 → promote to T1)
       │
       ▼
OPTIMIZE_LOOP (all subsequent rounds use T1)
```

T0 has fixed cost (typically < 1 GPU·h) and reuses across sessions, indexed by `(model_family, dataset_id, seed)`.

### S2.2 Scale-aware tolerance: noise model

**Core observation**: under different parallel configs, the loss differs primarily because **effective batch size changes the grad noise scale**, not because of numerical bugs.

```
σ_loss(scale) ≈ σ_loss(reference) × sqrt(EBS_ref / EBS(scale))

where EBS = global_batch_tokens × num_dp_replicas (effective batch size)
```

Step-wise loss tolerance is given as multiples of σ:

```
tolerance_step(s)        = k_step × σ_loss(scale at step s) + ε_systematic
window_tolerance(s, w)   = k_window × σ_loss(scale) / sqrt(w) + ε_systematic
                           (w-step rolling mean shrinks std by 1/√w)
```

| Param | Default | Source |
|-------|---------|--------|
| `k_step` | 4.0 (4σ) | step-wise strict gate |
| `k_window` | 3.0 (3σ) | window-mean, tighter |
| `w` | 50 (50-step rolling) | aligns with §3.1 SMOKE step order |
| `ε_systematic` | 0.005 | tolerates float-precision diff, kernel numerical-path diff, etc. |

`σ_loss(reference)` is **measured at T0 build time** (run same seed 3 times, take cross-run std), saved into the reference file.

### S2.3 Equivalence normalization (make different scales comparable)

Comparing raw loss is wrong (different EBS → different absolute values). Normalize to "per-token NLL" + "EBS-adjusted expected mean":

```
loss_per_token = loss / log(vocab_size)             # roughly normalized to [0,1]
expected_mean(s | scale)  = T1_mean(s) + Δ_EBS(EBS_ref → EBS_scale)

where Δ_EBS comes from grad-noise-scale literature (Smith et al. / OpenAI scaling laws):
for the same token count, doubling EBS makes the loss at ~1/√2 the step index match.

Implementation: align the x-axis on "tokens consumed", not "step idx".
```

Practical implication: **align comparison on tokens consumed, not step idx**. This automatically aligns out the mbs/gas effects.

### S2.4 Gate hierarchy (which trigger uses which tier)

| Gate | Triggered at | Tier | Tolerance | Failure action |
|------|--------------|------|-----------|----------------|
| **smoke_correctness** | After SMOKE | T0 | step-wise k=5σ (loosest, 100 step is noisy) | Back to PROJECTION, rebuild initial plan |
| **baseline_correctness** | After BASELINE | T0 | tokens-aligned window-mean k=3σ + grad_norm range | ABORT + escalate (numerical correctness broken) |
| **lite_correctness** | Every N rounds in OPTIMIZE_LOOP / on dramatic change | T1 | window-mean k=3σ | Mark plan dead, back to RE_PLAN (do not abort the whole session) |
| **regression_signal** | Every Snapshot | T2 | window-mean k=2σ (most sensitive) | Warn only, non-blocking — feeds the §S1 drift detector |

**Choice of N**: default `lite_correctness` every 3 rounds (aligned with the §7.6 explore-round period to amortize cost). Forced trigger when mbs / pp / recompute changes dramatically.

### S2.5 Schema additions

**New ReferenceCurve schema** (§8.x):

```yaml
# state/references/<model_family>__<dataset>__<seed>.yaml
reference_id: llama_dense_8b__c4_subset__seed42
tier: T0                           # T0 / T1
created_at: 2026-04-15T10:00:00Z
binding:
  model_family: llama_dense
  model_size_b: 8
  dataset_id: c4_subset_v3
  seed: 42
  precision: bf16

config:                            # smallest config used at build time
  parallelism: {tp: 1, pp: 1, dp: 8, ep: 1}
  mbs: 4
  gbs: 256
  recompute: none

trajectory:                        # tokens-aligned (NOT step-aligned)
  - {tokens: 1.0e6, loss_per_token: 0.842, grad_norm: 0.51}
  - {tokens: 2.0e6, loss_per_token: 0.781, grad_norm: 0.48}
  ...

noise:                             # cross-run std from 3 same-seed runs
  loss_per_token_std_at_1e7_tokens: 0.0042
  grad_norm_std_at_1e7_tokens: 0.018

promoted_to_t1:                    # T0 -> T1 promotion log
  - {at: 2026-04-15T11:30:00Z, by_session: pilot_run_20260415_a1, baseline_plan: r0_p0}
```

**FailureReport extension** (§8.8): when `failure_kind=NUMERICAL`, append `gate_tier` and `tokens_at_failure` so escalation has direct context on which comparison failed.

### S2.6 Edge cases

| Situation | Handling |
|-----------|----------|
| Fully custom model / data / seed; no T0 available | Pilot refuses to enter OPTIMIZE_LOOP; require user to commission a T0 first (`pilot reference build`) |
| MoE stochastic routing (same seed still varies) | T0 uses deterministic routing; T1 uses stochastic routing — increased noise absorbed by σ |
| User intentionally changed model arch (e.g. head_dim) | binding mismatch → force T0 rebuild |
| Loss spike with obviously non-numerical cause (e.g. LR warmup) | Tolerance formula does not apply; explicitly skip first N tokens via `skills/correctness/known_artifacts.md` (**new Skill**) |

### S2.7 Cross-references to existing chapters

- §3.1 main diagram: split CORRECTNESS into "BASELINE_CORRECTNESS (T0)" and "LITE_CORRECTNESS (T1)" state-machine nodes.
- §4.2 skills/ adds a `correctness/` subtree: `SKILL.md` / `tolerance.md` / `tier_promotion.md` / `known_artifacts.md`.
- §5 Tool interface adds: `reference.build`, `reference.compare` (replaces the current implementation of `observe.compare_loss`, signature upgraded).
- §10 evaluation metrics adds: "numerical regression false-negative ≤ 5%, false-positive ≤ 10%."
- §12.2 failure paths: refine the `NUMERICAL` row into 4 (one per gate); different tier failures take different transitions.

---

## §S3. Parallel Execution & Resource Protocol (filling the §3.1 "3 plans in parallel" gap)

> **Problem**: §3.1 says Champion-Challenger / Successive Halving runs "K plans in parallel," and §9 assumes P1/P2/P3 run 50 step in parallel. But: (1) how do you split 16 nodes? (2) does an OOM plan crash other plans? (3) what does this look like on Slurm/k8s? (4) how do short-run metrics extrapolate to full scale?
>
> **What this section does**: define the resource shard model, isolation contract, scheduling protocol, and extrapolation rules.

### S3.1 Three execution modes

```
        cluster_size = N nodes
        ┌──────────────────────────────────────────┐
        │ A) FullScale     ─ 1 plan × N nodes      │  BASELINE / final validation
        │ B) Sharded       ─ K plan × N/K nodes    │  Tuning Loop short runs
        │ C) TimeMux       ─ K plan serial × N     │  long-step / strong-struct-diff plans
        └──────────────────────────────────────────┘
```

| Mode | Suitable for | Pros | Cons |
|------|--------------|------|------|
| **FullScale** | BASELINE, CORRECTNESS, final config validation | Real data, no extrapolation | Serial, slow |
| **Sharded** | 50-step short runs in Tuning Loop (the common case) | Fast, parallel | Needs extrapolation + shape constraints (see S3.4) |
| **TimeMux** | Plans differ in structure (pp / world_size); or fewer than 3 candidates not worth slicing | Simple, no extrapolation | Total wall time long |

Mode-selection rule (in `skills/workflow/execute.md`):

```python
def choose_execution_mode(candidates, cluster_size):
    if stage in ['BASELINE', 'CORRECTNESS', 'FINAL_VALIDATION']:
        return 'FullScale'

    # Candidates differ in world_size → cannot shard
    target_ws = candidates[0].world_size
    if not all(c.world_size == target_ws for c in candidates):
        return 'TimeMux'

    # Too few candidates: sharding wastes nodes
    if len(candidates) < 3:
        return 'TimeMux'

    # Per-shard node count below the minimum credible shard
    shard_size = cluster_size // len(candidates)
    if shard_size < min_credible_shard(candidates[0]):  # see S3.4
        return 'TimeMux'

    return 'Sharded'
```

### S3.2 Resource contract for Sharded mode

Split the cluster into K **topology-aware shards**, one candidate plan per shard:

```
Shard partitioning principles (priority order):
  1. IB locality: keep nodes under the same leaf switch in one shard (avoid cross-spine interference)
  2. Homogeneity: same GPU model, same NIC model within a shard
  3. Equal size: node count differs by at most 1
  4. Fault avoidance: skip blacklisted nodes in §S3.5
```

**Upgraded `submit.run()` signature** (§5 Tool interface extension):

```python
submit.run(
    plans: List[Plan],
    mode: Literal['FullScale', 'Sharded', 'TimeMux'],
    shard_strategy: ShardStrategy = TopologyAware(),
    isolation: IsolationLevel = 'node',  # 'node' / 'rack' / 'cluster'
    timeout_per_plan_s: int = 1800,
    early_stop: EarlyStopPolicy = ...,
) -> List[RunResult]
```

**Slurm implementation**:

```bash
# Sharded: heterogeneous job step
sbatch \
  --job-name=pilot_r3 \
  --het-group=0 --nodes=4 --nodelist=node[01-04] : \
  --het-group=1 --nodes=4 --nodelist=node[05-08] : \
  --het-group=2 --nodes=4 --nodelist=node[09-12] : \
  --het-group=3 --nodes=4 --nodelist=node[13-16] \
  pilot_shard_runner.sh
```

**k8s implementation**: one PodGroup per shard (gang scheduling); shards anchored via nodeAffinity.

### S3.3 Failure-isolation contract

| Failure type | Blast radius | Other shards' behavior | Blacklist action |
|--------------|--------------|------------------------|------------------|
| OOM | Self shard's process only | Unaffected, continue | No blacklist |
| HANG (NCCL timeout) | Self shard's communicator only | Unaffected | Temp-blacklist that shard's nodes for 30 min (protect next) |
| Node down / GPU ECC | Whole shard fails | Unaffected | Permanently blacklist that node (write `state/blacklist.yaml`) |
| Cross-shard interference (fabric jitter) | Multiple shards regress simultaneously | Detected ≥ 2 same-direction regressions → ABORT whole batch | Trigger `PREFLIGHT` re-check |

**Cross-shard anomaly detection**:

```python
def cross_shard_anomaly(shard_results):
    # At least 2 shards' tps deviates from its own prediction by > 15% (same direction)
    deviations = [(r.tps - r.predicted_tps) / r.predicted_tps for r in shard_results]
    same_sign_outliers = sum(1 for d in deviations if d < -0.15)
    return same_sign_outliers >= 2
```

Hit → write `FailureReport(failure_kind=CLUSTER, root_cause=cross_shard_interference)`, take §12.2's PREFLIGHT path; this round batch does not consume budget.

### S3.4 Sharded extrapolation (the easiest pitfall)

**Core tension**: sharded runs give "tps on N/K nodes" but settle decisions need "tps on N nodes." The two are not always linear.

The doc establishes **3 extrapolation classes** (in `skills/workflow/execute_extrapolation.md`, **new Skill**):

| Axis changed by Plan | Cross-scale behavior | Sharded → full credibility | Recommendation |
|----------------------|----------------------|----------------------------|----------------|
| env / mbs / recompute / bucket only (**no struct change**) | Communication pattern unchanged; nearly linear | High (err < 5%) | Sharded OK, extrapolate directly |
| TP / VPP / EP (**comm structure changes**) | Cross-node traffic varies with world_size | Medium (err 5–15%) | Sharded for screening + top-2 to FullScale rerun |
| PP / DP (**comm group size changes**) | Bubble and allreduce both heavily depend on world_size | Low (err can exceed 20%) | Direct TimeMux; do not shard |

**`min_credible_shard()`** (decides whether sharding is allowed):

```python
def min_credible_shard(plan):
    # Must at least preserve the plan's smallest comm group
    return max(plan.tp * plan.pp,  # one full model-parallel group
               2)                   # at least 2 nodes preserve IB comm
```

**Prediction-consistency self-check**: after sharded runs complete, use §S1 calibration to back-compute "what tps would this plan get on full scale," and compare to "the corresponding prediction at round 0 BASELINE." If they differ by > 20%, **force top-2 to FullScale rerun** — otherwise the settle decision is untrustworthy.

### S3.5 Node health and blacklist

```yaml
# state/blacklist.yaml
nodes:
  - id: node07
    reason: ECC_uncorrectable_at_2026-04-20
    severity: permanent             # permanent / temporary
    expires_at: null
  - id: node12
    reason: NCCL_hang_repeated_3x
    severity: temporary
    expires_at: 2026-04-21T16:00:00Z
```

**Integration**: `preflight.run()` reads blacklist on start; excludes those nodes from `ClusterProfile.nodes`. `submit.run()`'s shard partitioning respects blacklist.

**Auto-blacklist conditions** (any one):

- ≥ 3 NCCL hangs within a single session (temporary, 4h)
- ECC uncorrectable / GPU not present (permanent)
- Cross-session cumulative failure rate > 30% (temporary, 24h)

Blacklisting is a side effect of §12.2 `CLUSTER` failure; does not consume round budget.

### S3.6 Checkpoint protocol for TimeMux

Time-multiplexing K plans on the same node group should avoid each plan re-warming up (compile / shape cache loss). Convention:

| Item | Reused across plans? | Notes |
|------|---------------------|-------|
| Python process | **Not reused** (isolation guarantee) | Independent process per plan |
| Megatron / TorchTitan compile cache | Reused when same backend / same model_arch | Shared via `TORCH_INDUCTOR_CACHE_DIR` |
| Optimizer state | Not reused | Re-init per plan |
| Dataloader prefetch buffer | Not reused | — |

Empirically, compile-cache reuse saves 30–60s/plan of warmup — non-trivial fraction of a 50-step short run.

### S3.7 Evaluation metric additions

`§10` adds:

| Dimension | Metric | Target |
|-----------|--------|--------|
| Parallel efficiency | Sharded wall time / TimeMux wall time | ≤ 0.4 (≥ 2.5× speedup) |
| Isolation quality | Cross-shard interference false-call rate (FullScale rerun overturn rate) | ≤ 5% |
| Node health | Blacklist false-positive (re-passes PREFLIGHT after rehab) | ≤ 10% |

### S3.8 Cross-references to existing chapters

- §3.1 main diagram: `Execute` node internally branches into the three modes.
- §5 Tool interface: `submit.run()` signature upgrade (mode / shard_strategy / isolation).
- §8 schema adds: `shard_plan.schema.json`, `blacklist.schema.json`.
- §12.1 Guardrails adds: node blacklist, cross-shard interference detection, shard isolation contract.
- §13.2 subagent boundary: Execute is still **not** a subagent, but a single shard's observe can be spawned independently.

---

## §S4. LEARN Governance & Knowledge Lifecycle (closing the "LLM writes its own knowledge" gap)

> **Problem**: §3.1 / §4.2 writes LEARN as "best/failure cases written back to `skills/knowledge/`" — the only Skill ← State reverse flow. If we let LLMs write and then read with no human review, in 3 months `knowledge/cases.md` will be full of low-quality observations, contradictory hints, and outdated "experience."
>
> **What this section does**: split LEARN into a "draft → review → merge" three-stage pipeline, define the draft schema, conflict detection, aging, and retraction.

### S4.1 Three-stage pipeline

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Stage A: Draft │ →  │  Stage B: Triage │ →  │  Stage C: Merge  │
│  (auto, by LLM) │    │  (human / curator)│    │  (git commit)    │
│                 │    │                  │    │                  │
│ state/knowledge_│    │ scripts/curate.py│    │ skills/knowledge/│
│   drafts/<sid>/ │    │ + manual review  │    │   *.md (versioned│
└─────────────────┘    └──────────────────┘    └──────────────────┘
                                ▲
                                │
                       ┌──────────────────┐
                       │  conflict / age  │
                       │  signals         │
                       └──────────────────┘
```

**Critical invariant**: `skills/knowledge/` only changes via git PR; the runtime process **has no write access** — LLMs can only write to `state/knowledge_drafts/`.

### S4.2 Stage A: Draft (LLM-emitted)

**Trigger**: when a session enters LEARN, emit a draft for each of the following:

| Draft kind | Trigger condition | Template |
|------------|-------------------|----------|
| `final_best_case` | session normally STOPs and `final_tps / baseline_tps ≥ 1.2` | templates/case.md |
| `failure_pattern` | An axis subtree has dead-rate > 30% | templates/anti_pattern.md |
| `env_recipe` | EnvSweep found ≥ 5% improvement env diff | templates/env_recipe.md |
| `model_calibration_drift` | §S1 reported drift_alarm and human confirmed real drift | templates/calibration_note.md |

**Draft schema** (unified envelope; new in §8 schema set):

```yaml
# state/knowledge_drafts/<session_id>/<draft_id>.yaml
draft_id: drft_20260421_a3_b2
draft_kind: final_best_case        # / failure_pattern / env_recipe / calibration_note
created_at: 2026-04-21T18:00:00Z
created_by:
  session_id: pilot_run_20260418_a3
  agent: claude-3.7-sonnet
  pilot_version: 0.4.2

binding:                           # related (cluster, model_family, framework)
  cluster_class: mi300x_8gpu       # cluster_class is more general than cluster_id (cross-cluster migration)
  model_family: llama_dense
  model_size_range: [70, 140]      # billions
  framework: megatron
  framework_version_range: ["0.6", "0.8"]

claim:                             # the regularity this draft wants to precipitate
  headline: "Llama 70B+ on MI300X-8GPU class, prefer pp=4 over pp=8 (bubble dominates after pp>4)"
  detail: |
    At 16 nodes (128 GPU) scale, pp=8 gives bubble_ratio median 0.21,
    pp=4 + vpp=2 gives bubble_ratio 0.09; net tps 14–18% higher.
  applicability:
    - "node_count ∈ [8, 32]"
    - "global_batch_tokens ≥ 4M"
  not_applicable_when:
    - "model_size_b > 200"          # no data
    - "framework=torchtitan"        # known to behave differently

evidence:                          # must have evidence; LLM guess not allowed
  - kind: round_result
    ref: state/checkpoints/r2/plan_graph.yaml#nodes.r2_p4
    summary: "pp=4 vpp=2 config tps=17600 vs pp=8 config tps=15400"
    cost_gpu_h: 1.2
  - kind: cross_session
    ref: knowledge/cases.md#llama70b-mi300x-2026-04-15  # historical case corroborates
    summary: "Same conclusion in prior session on same model"

confidence:                        # LLM self-assessment (Stage B reassesses)
  self_assessed: 0.72
  signals:
    - "single session evidence"
    - "consistent with one prior case"
    - "no contradicting case"

conflicts_with:                    # potential conflicts the LLM proactively flags
  - knowledge_id: case-llama70b-2026-03-12
    note: "Prior conclusion was pp=8 + ep_overlap, but that was v0.5 backend; no longer applicable"

review_status: pending             # pending / accepted / rejected / superseded
```

**Anti-patterns (drafts auto-rejected at Stage A)**:

| Anti-pattern | Why auto-reject | Reason |
|--------------|-----------------|--------|
| `evidence` empty or all `kind: llm_inference` | Auto-reject | Must have round_result or cross_session reference |
| `claim.headline` length > 200 chars | Auto-reject | Encourage atomic claims; complex conclusions become multiple drafts |
| `binding` empty or all `*` | Auto-reject | "Universal regularities" without applicability are usually noise |
| Direct contradiction with `skills/knowledge/anti-patterns.md` and no new evidence | Auto-reject | Prevent overriding known counter-examples |

### S4.3 Stage B: Triage (curator + human)

`curator` is a standalone agent / CLI tool (**not** part of the Pilot main flow); it periodically scans `state/knowledge_drafts/` and outputs the **review queue**.

**Auto signals** (curator scores each draft):

| Signal | + / − |
|--------|-------|
| `confirmation_count` (independent sessions emitting same claim) | +0.2/each, cap +0.6 |
| `applicability` fully overlaps an existing knowledge entry | -0.3 (duplicate) |
| Conflicts with existing knowledge (without explicit `supersedes`) | -0.5 |
| From final_best_case AND `final_tps / baseline > 1.5` | +0.2 |
| From failure_pattern AND dead subtree > 5 nodes | +0.2 |
| `evidence` spans ≥ 2 clusters | +0.3 |
| `evidence` only 1 session | -0.2 |

`triage_score = self_assessed_confidence + sum(signals)`

**Tiers**:

| triage_score | Disposition |
|--------------|-------------|
| ≥ 1.0 | Mark `auto_acceptable`; one-click PR (still needs reviewer approve) |
| 0.5 – 1.0 | Enter **human review queue** (cap 10 per batch) |
| < 0.5 | Mark `low_signal`; archive without merging (kept under drafts/, GC'd in 30 days) |

**Curator CLI** (`tools/curate.py`):

```bash
# List pending
pilot curate list --since 2026-04-15

# Bulk auto-accept high-score drafts
pilot curate auto-accept --threshold 1.2 --dry-run
pilot curate auto-accept --threshold 1.2 --apply  # creates PR

# Single-draft review
pilot curate show drft_20260421_a3_b2
pilot curate accept drft_20260421_a3_b2 --merge-into knowledge/cases.md
pilot curate reject drft_20260421_a3_b2 --reason "evidence too thin"
pilot curate supersede drft_20260421_a3_b2 \
    --replaces case-llama70b-2026-03-12 \
    --merge-into knowledge/cases.md
```

### S4.4 Stage C: Merge (git-ified)

Accepted drafts become PRs into `skills/knowledge/`. **Strong constraints**:

1. **PRs may only be authored by curator CLI** (signed with `Pilot-Curated-By: <user>`); manual commits rejected (prevents bypass).
2. **Each knowledge entry must contain**:
   - `knowledge_id` (immutable ULID)
   - `applicability` / `not_applicable_when`
   - `evidence_refs` (back-pointers to the frozen draft copy)
   - `created_at` / `last_confirmed_at`
   - `confirmation_count`
3. **PR requires ≥ 1 reviewer approval** (regardless of auto_acceptable).
4. **CI checks** (`scripts/lint_knowledge.py`):
   - `applicability` not fully congruent with existing knowledge (else require supersede)
   - No broken `evidence_refs`
   - `claim` does not contradict anti-patterns

**Final knowledge entry shape** (a single block in `skills/knowledge/cases.md`):

```yaml
# skills/knowledge/cases.md (YAML front-matter blocks)
---
knowledge_id: 01HV9X8K7N2QD3F4G5H6J7K8L9
title: "Llama 70B+ on MI300X-8GPU: prefer pp=4 over pp=8"
applicability:
  - "model_family=llama_dense, model_size_b∈[70,140]"
  - "cluster_class=mi300x_8gpu, node_count∈[8,32]"
  - "framework=megatron, version∈[0.6, 0.8]"
not_applicable_when:
  - "model_size_b > 200"
  - "framework=torchtitan"
recommendation:
  hint: "Default starting point: pp=4 + vpp=2"
  rationale_skill: skills/optimization/pipeline/vpp.md
created_at: 2026-04-21T18:30:00Z
last_confirmed_at: 2026-04-21T18:30:00Z
confirmation_count: 1
evidence_refs:
  - state/archive/drafts/drft_20260421_a3_b2.yaml
status: active                     # active / superseded / retired
---

Detailed narrative ...
```

### S4.5 Aging and retraction

**Auto-aging scan** (weekly CI job):

| Condition | Action |
|-----------|--------|
| `last_confirmed_at` > 90 days AND `confirmation_count = 1` | Mark `stale`; LEARN does not read it next time; pinned to bottom of main library |
| New knowledge entry's `applicability` fully covers an old one | Suggest reviewer to `supersede` the old one |
| 3 consecutive sessions on same binding contradict the entry | Auto-open issue; reviewer decides retire |
| `framework_version` falls outside `applicability` range (external upgrade) | Mark `version_drift`; await reconfirm |

**Retraction (retire)**: knowledge is not deleted; only marked `status: retired`; original text preserved for audit replay.

### S4.6 LLM read protocol (avoid using bad knowledge)

LEARN is the write side; **`skills/knowledge/` reading by Diagnose / Re-Plan** also needs a protocol (in `skills/knowledge/SKILL.md`):

1. **Priority**: active > stale; do not read retired / superseded.
2. **Binding match requirement**: every `applicability` clause must hit current (cluster, model, framework); if `not_applicable_when` hits → skip.
3. **Multiple matches**: sort by `confirmation_count desc, last_confirmed_at desc`, take top-3.
4. **Conflict handling**: if top-3 contain mutually-contradictory `recommendation.hint` (same axis, opposite directions), **discard all + record the conflict** — fall back to model-only decision without knowledge.

### S4.7 Cross-cluster migration (`cluster_class` abstraction)

`cluster_id` is a specific cluster (mi300x-16node, mi300x-32node); `cluster_class` is the equivalence class of "same generation GPU + same generation NIC + same GPUs/node" (mi300x_8gpu).

**Why**: knowledge is mostly sensitive to GPU/node ratio and NIC type, but **far more elastic to total node count than to per-node config**. Binding to `cluster_class` lets knowledge transfer across same-class different scales.

**`cluster_class` derivation rules** (`skills/knowledge/cluster_class.md`, **new Skill**):

```python
def cluster_class(profile: ClusterProfile) -> str:
    return f"{profile.gpu_arch}_{profile.gpus_per_node}gpu_{profile.nic_class}"
    # e.g. mi300x_8gpu_cx7
```

Different `cluster_class` values **do not auto-migrate** knowledge; a human reviewer can explicitly promote.

### S4.8 Multi-tenancy coordination (concurrent sessions)

| Scenario | Protocol |
|----------|----------|
| Multiple sessions emit drafts concurrently | `state/knowledge_drafts/<session_id>/` is naturally isolated; no conflict |
| Multiple sessions read same `skills/knowledge/` | Read-only; no conflict |
| Curator merges PR while upstream changed (another PR landed) | Standard git rebase; CI re-runs lint |
| Multiple drafts within one cycle make the same claim | Curator detects similarity (embedding + headline lev distance) and merges into single entry with `confirmation_count=N` |

### S4.9 Evaluation metric additions

`§10` adds:

| Dimension | Metric | Target |
|-----------|--------|--------|
| Knowledge quality | Fraction superseded / retired within 6 months of merge | ≤ 20% |
| Knowledge quality | Re-Plan hit rate after consulting knowledge (recommended config achieves ≥ predicted gain) | ≥ 60% |
| Governance throughput | Average draft → merge time | ≤ 7 days |
| Governance throughput | Reviewer load (manual reviews / week) | ≤ 20 |

### S4.10 Cross-references to existing chapters

- §3.2 LEARN: no longer "directly writes back to skills/knowledge/" — instead "writes draft + submits to review queue."
- §4.1 directory tree adds: `state/knowledge_drafts/`, `state/archive/drafts/`, `scripts/curate.py`.
- §4.2 skills/: under knowledge/, new `SKILL.md` (read protocol), `cluster_class.md`, `drafts_lifecycle.md`.
- §5 Tool interface: `knowledge.write` **only writes drafts/**; cannot write skills/.
- §8 schema adds: `knowledge_draft.schema.json`, `knowledge_entry.schema.json`.
- §12.1 Guardrails adds: "skills/knowledge/ write-permission isolation," "draft schema validation," "conflict detection."

---

## Unified cross-reference index

Modifications to the main doc are concentrated below for ease of review and merging.

| Section | Modification | Original location |
|---------|--------------|-------------------|
| §S1 / §S3 | Add Tool interfaces | §5 Tool interface table |
| §S1 / §S2 / §S3 / §S4 | Add schemas | §8 Data structures |
| §S2 | Split state-machine nodes | §3.1 main diagram + §3.2 narrative |
| §S1 / §S2 / §S4 | Add Skill subtrees | §4.2 skills/ directory |
| §S1 / §S2 / §S3 / §S4 | Add evaluation metrics | §10 Evaluation metrics |
| §S1 / §S2 / §S3 / §S4 | Add Guardrails / failure paths | §12.1 / §12.2 |
| §S3 | Adjust Subagent boundary | §13.2 boundary table |

## Rollout priority (recommended)

| Stage | Must land | Defer |
|-------|-----------|-------|
| MVP (5–8 round short tasks) | §S2 (reference + tolerance), §S4 partial (draft schema + write-permission isolation) | §S1 (cold-start defaults suffice), §S3 (use TimeMux), §S4 full pipeline |
| Core (10+ round / multi-model) | §S1 (full calibration), §S3 (Sharded + blacklist), §S4 full pipeline | §S4 cross-cluster migration |
| Full (multi-day / multi-team) | All | — |

Each stage does not block the next stage's extension points (schemas reserve backward compatibility).

---

## One-line positioning

§S1 gives the search engine a **calibrated drivetrain** (so the priority formula is more than a pretty form), §S2 gives the numerical gate **evidence-grounded tolerance** (so CORRECTNESS stops false-positive / false-negative), §S3 gives parallel execution a **resource + isolation contract** (so the §3.1 "3 plans in parallel" line actually means something), §S4 gives knowledge precipitation **a turnstile** (so LEARN does not silently degrade into a noise dump).
