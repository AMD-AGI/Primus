# Preflight Workflow

**Status**: detailed-design v1
**Read by**: Preflight Stage Worker
**Domain**: stage-protocol authority
**Anchors**: README §3.1 (stage 1) · §5 Tools · §8.1 ClusterProfile · §12.2 failure paths · §S3.5 Blacklist · §S1 Calibration

This document is the **single authoritative protocol** for the PREFLIGHT stage. Everything else (tool CLI, worker prompt, schema fields) is derived from it.

---

## 1. Purpose & Position

PREFLIGHT is the **first stage** of the state machine and the **only producer** of `ClusterProfile` (§8.1). Its outputs are *cluster_shared* — produced once, reused across every job that targets the same cluster within the freshness window.

**What it answers** (in order):

1. Is the cluster usable at all? (basic reachability + driver health)
2. What does the hardware actually deliver? (measured peaks, not vendor spec)
3. What is the safe `env_baseline` to start from? (3-tier validated)
4. Which nodes are degraded and must be excluded? (Blacklist write-back)

**What it does NOT do**: model-specific profiling (that's PROJECTION), bottleneck analysis (DIAGNOSE), or env tuning beyond baseline (ENV_SWEEP).

---

## 2. Triggers — when to enter PREFLIGHT

| Trigger | Source | Cache action | Counts against budget? |
|---------|--------|--------------|------------------------|
| New session, no cached `ClusterProfile` for `cluster_id` | Bootstrap | Full collect | No (preflight is infra) |
| Cached profile exists, `age <= 7d`, no invalidation event | Bootstrap | **Skip stage**, load from cache, advance to PROJECTION | No |
| Cached profile exists, `age > 7d` | Bootstrap | Delta-refresh (env + RCCL only) | No |
| Reentry: prior `HANG` failure (NCCL/IB timeout) | §12.2 `HANG` | Re-run env_probe only | No (counts_against_budget=false) |
| Reentry: prior `CLUSTER` failure (node down / driver) | §12.2 `CLUSTER` | Full re-collect, mark prior superseded | No |
| Reentry: Diagnose found `cluster_profile.age > 7d` mid-loop | DiagnosisReport `env_suspect.cluster_freshness` | Delta-refresh | No |
| User `--force` flag | CLI | Full re-collect | No |
| Driver/firmware/RCCL upgrade detected at bootstrap | `/sys/...` mtime check | Full re-collect | No |

**Important**: PREFLIGHT never consumes a tuning round. The state machine treats it as orthogonal infrastructure.

---

## 3. Inputs

PREFLIGHT (and every other Pilot tool) consumes **two** classes of input:

### 3.1 Environmental input — `cluster.yaml` (mandatory, see `AGENTS.md` §4)

A single file conforming to `schemas/cluster_config.schema.json`, resolved via `--cluster-config <path>` / `PILOT_CLUSTER_CONFIG` / `./cluster.yaml`. It declares **only** which runtime mode Pilot is operating inside:

```yaml
# Example: prepared single-node container
schema_version: "1.0"
cluster_id: mi355x-localhost
mode: single
runtime: { image_label: "rocm7.2-torch2.10" }
```

```yaml
# Example: attached to a user-owned SLURM allocation
schema_version: "1.0"
cluster_id: mi355x-prod
mode: slurm
slurm:
  job_id: 12345          # user already ran `salloc -N 4`
  rdzv_port: 29400       # optional, default 29400
runtime: { image_label: "rocm7.2-torch2.10" }
```

The Worker is **not allowed** to discover the cluster any other way. No env-var sniffing, no ssh probing, no docker introspection. The contract is: *if `cluster.yaml` is wrong, the user fixes it; the Worker reports `failure.kind=CLUSTER` and exits.*

The three fast-fail checks at PREFLIGHT entry (also covered in `AGENTS.md` §4):

1. `cluster.yaml` parses and validates against the schema.
2. If `mode=slurm`: `scontrol show job <slurm.job_id>` returns `JobState=RUNNING`. Stale allocation → `failure.kind=CLUSTER` with message *"slurm allocation is no longer running; please re-salloc and update cluster.yaml"*.
3. At least one GPU is visible inside the current process. None visible → `failure.kind=CLUSTER` with message *"no GPU visible; check container GPU passthrough"*.

These three checks **do not consume tuning rounds**.

### 3.2 Stage-specific input — the PreflightRequest

Passed by the Orchestrator (or assembled from CLI flags) on top of `cluster.yaml`:

```yaml
preflight_request:
  target_version: null                # optional; if set, attempt to produce this version
  force: false                        # full re-collect even if cache fresh
  delta_only: false                   # explicit delta-refresh (env + RCCL only)
  max_wallclock_s: 1800               # 30 min hard cap
  blacklist_path: state/blacklist.yaml
  reason: bootstrap                   # bootstrap | reentry_hang | reentry_cluster | reentry_stale | force
  reentry_context:                    # populated when reason=reentry_*
    prior_failure_kind: HANG          # HANG | CLUSTER | null
    prior_session_id: cursor_dryrun_001
    prior_profile_ref: state/cluster_profiles/mi300x-16node_v3.yaml
```

`cluster_id` is **not** here — it comes from `cluster.yaml`. The Worker **must not** invent inputs the Orchestrator did not pass.

---

## 4. The 5-step measurement protocol

Each step has explicit pass criteria, a hard time cap, and a documented degradation mode. The Worker executes them in order; an early failure short-circuits remaining steps.

### How `cluster.yaml` drives the measurement layout

Before Step 1, the Worker resolves a `LaunchPlan` from `cluster.yaml`:

| `cluster.yaml` mode | Worker action for Steps 1-5 |
|---------------------|------------------------------|
| `single` | Single-process discovery (Step 1). Steps 2-5 run as `torchrun --nnodes=1 --nproc-per-node=$(visible_gpus)` inside the current container. No peer nodes touched. |
| `slurm` | Step 1 calls `scontrol show job <slurm.job_id>` to enumerate nodelist (after subtracting blacklist). Steps 2-5 run as `srun --jobid=<slurm.job_id> -N <nnodes> --ntasks-per-node=1 python -m pilot.tools._preflight_node_entry …`, which on each node spawns a per-node `torchrun --nnodes=<nnodes> --node-rank=$SLURM_NODEID …`. Rendezvous endpoint = `<head_host>:<slurm.rdzv_port>` where `head_host` is the first hostname of the (filtered) nodelist; `rdzv_id = pf_<slurm.job_id>`. |

The Worker never reads `cluster.yaml` again after this resolution; the `LaunchPlan` is the only thing passed downstream.

### Step 1 — Topology discovery  *(time cap: 30s)*

| Source | Output | Pass criteria |
|--------|--------|---------------|
| `rocm-smi --showtoponuma` / `nvidia-smi topo -m` per node | `nodes`, `gpus_per_node`, intra-node link type (xGMI / NVLink), inter-node link type (IB / RoCE) | `gpus_per_node ≥ 1` on every reachable node; topology consistent across nodes (same link types) |

Reads `state/blacklist.yaml`; **excludes** blacklisted nodes from the discovery set before the rest of the protocol.

**Failure**: `nodes_reachable < ceil(0.5 × requested)` → `failure.kind = CLUSTER`, return immediately. No partial profile.

### Step 2 — Compute peak measurement  *(time cap: 60s/node, parallel)*

| Probe | Output field | Pass threshold |
|-------|--------------|----------------|
| Fixed-shape GEMM (M=N=K=8192, BF16) | `compute.peak_tflops_bf16` | ≥ 70% of vendor spec → validated; 50–70% → tentative; < 50% → fail |
| FP8 GEMM (if hardware supports) | `compute.peak_tflops_fp8` | same thresholds |
| HBM stream (rocBLAS / cuBLAS scope) | `compute.hbm_bandwidth_gbs` | ≥ 80% of nominal HBM3 BW |
| HBM capacity probe | `compute.hbm_capacity_gb` | matches vendor spec ±2% |

If a single node fails: blacklist proposal (temporary, reason `compute_peak_low`), continue with remaining nodes. If `nodes_healthy < floor(0.8 × requested)`: `status = tentative`, `nodes_healthy / nodes_total` recorded.

### Step 3 — Interconnect peak measurement  *(time cap: 90s, sample-based)*

Sample 4 node-pairs (or all if `nodes ≤ 4`):

| Link | Probe | Output | Pass threshold |
|------|-------|--------|----------------|
| Intra-node (xGMI/NVLink) | In-node 8-GPU ring AllReduce 1GB | `interconnect.intra_node.bandwidth_gbs` | ≥ 70% of link spec |
| Inter-node (IB/RoCE) | `ib_send_bw -d <hca>` between pair | `interconnect.inter_node.bandwidth_gbs` (per-GPU effective) | ≥ 60% of nominal port BW |

Pair-to-pair variance > 25% → record `interconnect.uniformity = degraded`, surface in summary; do not fail.

### Step 4 — RCCL collective baseline curves  *(time cap: 5min)*

Schema 2.0: measured at **three independent scopes**, each via a dedicated `srun` invocation orchestrated by `_dispatch_run_slurm` (single-node mode populates `intra_node` only). Each scope runs all 5 collectives — `AllReduce`, `AllGather`, `ReduceScatter`, `Broadcast`, `AllToAll` — at sizes `[1, 16, 64, 256]` MB (AllToAll defaults to `[1, 16, 64]` due to per-rank N× memory cost):

| Scope | Layout | Isolates | srun pattern |
|-------|--------|----------|--------------|
| `intra_node` | each node runs its own local 8-GPU PG, **in parallel** | xGMI/NVLink only | `-N <nnodes> --ntasks-per-node=1` → each task launches local `torchrun --nnodes=1 --nproc-per-node=8` |
| `inter_node` | 1 GPU/node × N nodes single ring | inter-node fabric (RoCE/IB) only | `-N <nnodes> --ntasks-per-node=1` → `_preflight_node_entry --nnodes=N --nproc-per-node=1` |
| `world` | full N×gpus_per_node ring | actual training topology | `-N <nnodes> --ntasks-per-node=1` → `_preflight_node_entry --nnodes=N --nproc-per-node=8` |

`intra_node` uses **per-node columnar** layout (each node's local-PG curve recorded separately under its hostname); `inter_node` and `world` use single-ring arrays. Authoritative shape: `pilot/schemas/cluster_profile.schema.json` §`rccl_baseline`.

```yaml
rccl_baseline:
  intra_node:
    world_size: 8
    nnodes_measured: 16
    collectives:
      allreduce:
        sizes_mb: [1, 16, 64, 256]
        per_node_bw_gbs:
          smc01: [25.1, 182.2, 306.3, 377.2]
          # ... 15 more nodes
        per_node_latency_us:
          smc01: [73.1, 161.1, 383.4, 1245.3]
          # ...
        roll_up:
          median_bw_gbs: [25.0, 181.5, 305.8, 376.1]
          min_bw_gbs:    [22.3, 175.0, 290.0, 281.5]
          max_bw_gbs:    [25.2, 182.2, 306.3, 377.2]
          stddev_pct:    [3.2,  1.1,   1.5,   9.8]
          slow_nodes_at_max_size: [smc04]
      # allgather / reduce_scatter / broadcast / alltoall: same shape

  inter_node:
    world_size: 16
    nnodes: 16
    nproc_per_node: 1
    collectives:
      allreduce:
        sizes_mb:   [1, 16, 64, 256]
        bw_gbs:     [3.2, 18.5, 23.1, 25.4]
        latency_us: [580, 1580, 5060, 18400]
      # ...

  world:
    world_size: 128
    nnodes: 16
    nproc_per_node: 8
    collectives:
      allreduce:
        sizes_mb:   [1, 16, 64, 256]
        bw_gbs:     [1.8, 11.4, 19.7, 22.3]
        latency_us: [1020, 2580, 5940, 20800]
      # ...
```

**Pass criteria**:
- `intra_node.collectives.allreduce.roll_up.median_bw_gbs[256MB] ≥ 50%` of theoretical xGMI/NVLink ring BW.
- `intra_node.collectives.allreduce.roll_up.stddev_pct[256MB] ≤ 10%` (cross-node uniformity).
- `world.collectives.allreduce.bw_gbs[256MB] ≥ 50%` of `min(intra_node, inter_node)` ring BW.

Any miss → top-level `status = tentative` and add `rccl_underperformance` to env_baseline notes (env_probe in step 5 may correct this). `slow_nodes_at_max_size` populates blacklist proposals (PREFLIGHT only flags; it does not auto-blacklist).

### Step 5 — Env probe (3-tier safe-probe per `profiling/env_probe.md`)  *(time cap: 8min)*

Candidate `env_baseline` source order:

1. `cluster_class` preset from `skills/env/presets.md` (e.g. `mi300x_8gpu`)
2. Prior validated `env_baseline` of same `cluster_class` from another `ClusterProfile`
3. Vendor defaults (last resort, marked `unsafe_fallback`)

3 tiers, each gates the next:

| Tier | Probe | Time cap | Pass criteria | On fail |
|------|-------|----------|---------------|---------|
| T1 Connectivity | `nccl-tests` 2-node sanity AllReduce 1GB | 30s | Returns within 30s, no NaN | T2 skipped, env_baseline = `unsafe_fallback`, `failure.kind = HANG` (prior cluster default likely broken) |
| T2 Micro-bench | RCCL ar/a2a curves under candidate env | 90s | Within 10% of step-4 baseline curves | Mark this candidate `unsafe`, fall back to vendor defaults, `status = tentative` |
| T3 Multi-node short run | 1 node × 100 step toy training | 6min | Completes, `loss_finite=true`, `tps` non-zero | Same as T2 fall-back; record failure for blacklist |

Output:

```yaml
env_baseline:
  version: mi300x-16node-v4              # auto: <cluster_class>-<node_count>-v<incr>
  status: validated | tentative | unsafe_fallback
  cluster_class: mi300x_8gpu
  source: preset | inherited | vendor_default
  rccl: { ... validated values ... }
  hsa:  { ... }
  alloc:{ ... }
  threading: { ... }
  notes: []                              # human-readable warnings
```

`status = unsafe_fallback` is a yellow flag — PROJECTION may proceed, but ENV_SWEEP gets higher priority in the first OPTIMIZE_LOOP round.

---

## 5. Cache & versioning

**Cache layout**:

```
state/cluster_profiles/
├── <cluster_id>_<version>.yaml         # one per validated profile
└── _index.yaml                         # cluster_id → latest_version + age
```

**Cache key**: `(cluster_id, env_baseline.version)`. The `version` string is auto-incremented on full re-collect; preserved across delta-refresh.

**Freshness rules**:

| Condition | Action |
|-----------|--------|
| `now - collected_at <= 7d` AND no invalidation event | Cache hit; return cached profile, skip stage |
| `now - collected_at > 7d` | Delta-refresh: re-run steps 4 + 5 only, increment version |
| Driver/firmware/RCCL change detected | Full re-collect (steps 1–5), new version, prior superseded |
| Prior failure within 4h: kind=HANG | Re-run step 5 (env_probe) only, version++ if env_baseline changed |
| Prior failure within 4h: kind=CLUSTER | Full re-collect (steps 1–5), new version |

**Cross-job reuse**: a job at session bootstrap reads `_index.yaml`. If a fresh profile exists for its `cluster_id`, the Worker is **not even spawned** — Orchestrator advances directly to PROJECTION using the cached `cluster_profile_ref`.

---

## 6. Blacklist integration  (§S3.5)

**Read** (start of every PREFLIGHT run):

- Open `state/blacklist.yaml`
- Filter out nodes with `severity=permanent` OR `severity=temporary AND expires_at > now`
- Use the filtered set as the topology discovery starting set (Step 1)

**Write** (during PREFLIGHT):

A node is **proposed** for the blacklist when any of:

| Condition | Severity | TTL |
|-----------|----------|-----|
| Step 1: unreachable | temporary | 4h |
| Step 2: compute peak < 50% spec | temporary | 4h |
| Step 2: HBM capacity mismatch | permanent | — |
| Step 3: IB BW < 30% nominal | temporary | 4h |
| Step 5 T3: short run hangs (NCCL timeout) | temporary | 4h |
| ECC uncorrectable in dmesg | permanent | — |

Proposals are written to `state/blacklist_proposals/<ts>.yaml`. **Promotion to actual `state/blacklist.yaml` requires either (a) `auto_blacklist_threshold` met in §S3.5 — typically ≥ 3 hangs in 24h cumulative — or (b) explicit Orchestrator approval**. The Worker never edits `blacklist.yaml` directly.

---

## 7. Output contract

`SubagentResult` returned to Orchestrator:

```yaml
stage: PREFLIGHT
status: success | tentative | failed
artifacts:
  - kind: ClusterProfile
    ref: state/cluster_profiles/<cluster_id>_<version>.yaml
  - kind: BlacklistProposal           # only when proposals were generated
    ref: state/blacklist_proposals/<ts>.yaml
summary:
  headline: "<n_healthy>/<n_total> nodes, peak BF16 <x> TFLOPs (<pct>%), env_baseline=<version> <status>"
  key_metrics:
    nodes_healthy: 16
    nodes_total: 16
    peak_tflops_bf16: 1280
    peak_pct_of_spec: 0.985
    ib_bw_gbs: 392
    rccl_ar_256mb_gbs: 192
    env_baseline_version: mi300x-16node-v4
    env_baseline_status: validated
    blacklist_proposals: 0
  warnings: []                          # short, human-readable; e.g. "node mi300x-04 thermal margin low"
suggested_transition:
  to: PROJECTION
  reason: "cluster ready"               # or "tentative; consider rerun" / "fatal; abort"
cost:
  gpu_h: 0.6
  wallclock_s: 1700
  tool_calls: 12
failure: null                           # populated when status=failed
```

**`status` semantics**:
- `success` — all 5 steps passed every gate; downstream proceeds normally.
- `tentative` — at least one gate softened (e.g. peak 50–70%, env_baseline = unsafe_fallback). Downstream proceeds, but Orchestrator should bump ENV_SWEEP priority and consider an early re-PREFLIGHT after 1 round.
- `failed` — step 1 fatal or all candidates failed Tier 1. `failure.kind ∈ {CLUSTER, HANG, UNKNOWN}`.

---

## 8. Failure paths (mapped to §12.2)

| Internal cause | failure.kind | escalate_to_orchestrator | Orchestrator transition |
|----------------|--------------|--------------------------|-------------------------|
| Step 1: < 50% nodes reachable | CLUSTER | true | `ABORT` (humans: bring nodes back) |
| Step 1: blacklist + remaining < requested floor | CLUSTER | true | `ABORT` or wait + retry |
| Step 2/3: peaks too low cluster-wide | UNKNOWN | true | `ABORT + escalate` (hardware not delivering) |
| Step 4: RCCL underperforms but step 5 T2 recovers | (none, status=tentative) | false | `PROJECTION` |
| Step 5 T1 fail | HANG | true | `ABORT` or retry once with vendor defaults |
| Tool stub raises `NotImplementedError` | TOOL_ERROR | true | `ABORT` (implementation gap) |
| Wallclock exceeds `max_wallclock_s` | TIMEOUT | true | `ABORT` (preflight should not take this long) |

Worker never retries internally — single attempt, return verdict. Orchestrator decides retry policy.

---

## 9. Calibration interaction  (§S1)

When PREFLIGHT writes a new `env_baseline.version`, the Orchestrator (in `apply_result`) must:

1. Mark `calibration_state.version = env_baseline.version`
2. Set `calibration_state.effective_n = min(prior_effective_n, 5)` to force re-warmup of the WRLS estimator
3. Persist via `state.checkpoint()`

This is the only stage that can invalidate calibration. Document it here so the Worker doesn't have to know — but the Orchestrator's `apply_result` mapping does.

---

## 10. Hard constraints (Worker self-check)

- Total wallclock < 30 min first-time, < 10 min delta-refresh, < 2 min cache-hit (no real work)
- GPU·h cost < 1.0 (preflight is infrastructure, not a tuning round)
- Worker context peak < 15K tokens (read `profiling/*` subtree only; do **not** read `optimization/*` or `execution-model/*`)
- `summary.headline` < 200 tokens
- Never write to `cluster_profiles/_index.yaml` until all artifacts persisted (atomic-rename pattern)

---

## 11. Cross-references

| What you might also need | Where |
|--------------------------|-------|
| Measurement-level details (how to sweep RCCL, how to read rocm-smi) | `skills/profiling/preflight.md`, `skills/profiling/network.md` |
| Env safe-probe 3-tier exact thresholds | `skills/profiling/env_probe.md` |
| Per-cluster-class env presets | `skills/env/presets.md` |
| `cluster.yaml` field-by-field schema (universal tool input) | `schemas/cluster_config.schema.json` |
| Universal tool input contract & lifecycle | `AGENTS.md` §4 |
| ClusterProfile field-by-field schema | `schemas/cluster_profile.schema.json` |
| Blacklist schema and §S3.5 | `schemas/blacklist.schema.json`, `README.supplements.md` §S3.5 |
| Where Orchestrator decides cache-hit vs re-run | `skills/workflow/state_machine.md` (PREFLIGHT entry rules) |
| Worker role prompt (what "you are" reads as) | `prompts/worker/preflight.md` |
| CLI surface | `tools/preflight.py` |
