---
name: preflight
description: Cluster baseline collection and env probe for Primus tuning. Use the first time Pilot runs against a cluster, or after any cluster change (driver upgrade, ROCm bump, fabric reconfig). Produces output/pilot/cluster-<id>.md with GEMM peak, intra/inter-node collective bandwidth, validated env baseline, and slow-node flags. Triggers: preflight, cluster baseline, cluster profile, GEMM peak, RCCL/NCCL bandwidth, env probe, allreduce baseline, alltoall baseline, hardware peak, scaling baseline, slow node detection.
---

# Preflight — Cluster Baseline + Env Probe

The first thing every tuning session needs is *what this cluster can actually do*: GEMM peak, collective bandwidths, the validated env baseline. Without it, `execution-model` predictions are meaningless and `bottleneck-diagnose` thresholds are arbitrary. This skill collects all of it once per cluster (per env-baseline version) and writes one markdown file the rest of Pilot reads.

Preflight runs as a **subagent invocation** (it does several short benchmarks and parses outputs). The main agent kicks it off and only ever sees the final summary table.

## Inputs

| Slot | How |
|---|---|
| `cluster_id` | A short, stable label (e.g. `mi300x-16node-amd-aig`, `mi355x-105-ainic`). The user names the cluster; pilot uses this in the output filename. |
| `mode` | `single` or `slurm`. Pilot does not allocate; the user prepares the env. |
| `nodes`, `gpus_per_node` | From the user / from `sinfo` |
| `image` | The Primus container image (default `rocm/primus:v26.2`) |
| `force_refresh` | Optional. Force re-collect even if a recent file exists. |

## Outputs

A single file at `output/pilot/cluster-<cluster_id>.md` with these sections:

1. **Header** — cluster_id, version tag, collected_at, nodes_total, nodes_healthy, gpus_per_node
2. **Compute peak** — GEMM bf16 / fp8 peak, HBM capacity / bandwidth, per-node variance
3. **Interconnect** — intra-node (xGMI/NVLink), inter-node (IB/AINIC), bandwidth + uniformity
4. **RCCL/NCCL baseline** — intra_node / inter_node / world for AllReduce, AllGather, ReduceScatter, AllToAll at sizes [1, 16, 64, 256] MB
5. **Slow nodes** — flagged but not auto-blacklisted
6. **Env baseline** — the validated golden env (one of the presets from `env-catalog`, plus probed values)
7. **Notes** — known issues / version pinning / things to revisit

## Workflow

### Step 1: Check for cached baseline

If `output/pilot/cluster-<cluster_id>.md` exists and is < 7 days old (and `force_refresh` is not set):

- Read its header.
- If `version` and `nodes_total` match the current request, return its summary directly. **Skip the rest.**
- Otherwise tell the user the cached baseline is stale (driver/cluster changed) and proceed to re-collect.

### Step 2: Sanity-check the runtime

Before launching anything, verify the prerequisites are met. Pilot does NOT allocate.

| Mode | Check |
|---|---|
| `single` | container is up (`docker ps` / `podman ps`), at least 1 GPU visible (`rocm-smi` or `nvidia-smi`) |
| `slurm` | `sinfo` returns the partition; if a job_id is provided, `scontrol show job <id>` returns `JobState=RUNNING`; nodes are reachable |

If a check fails, surface a clear error and stop. Do not try to "fix" the runtime.

For SLURM environments, also use the `.cursor/skills/slurm-idle-node-check` skill to filter to actually-idle, healthy nodes before running benchmarks. For containerized AMD AIG environments, see `.cursor/skills/slurm-xiaoming-dev-container`.

### Step 3: Compute peak (GEMM)

Run a short GEMM benchmark on each node. The standard tool is `rocblas-bench` / `cuBLAS bench` invoked inside the Primus container. Approximate matrix shape: large square BF16 GEMM at the dimension your typical model sees (e.g. `M=N=K=8192` for hidden=8192).

For each node:

| Metric | How |
|---|---|
| `peak_tflops_bf16` | sustained BF16 GEMM throughput (median across 5 runs) |
| `peak_tflops_fp8` | sustained FP8 if hardware supports |
| `hbm_capacity_gb` | from `rocm-smi` / `nvidia-smi` |
| `hbm_bandwidth_gbs` | optional STREAM-style HBM bench |

Record `per_node_variance_pct = (max-min)/median × 100`. Variance > 3% means at least one node is degraded.

### Step 4: Interconnect peak

```bash
# Intra-node: per-pair p2pBandwidthLatencyTest equivalent inside one node
# Inter-node: ib_send_bw / ib_write_bw between one GPU on each of two nodes
```

Record the median per-pair / per-link bandwidth. Compare with the published spec (xGMI ≈ 800 GB/s, NVLink 4 ≈ 900 GB/s, IB HDR ≈ 25 GB/s/link, AINIC variable).

### Step 5: RCCL / NCCL collective baseline

This is the most important section — `execution-model` reads from it. Run `rccl-tests` (AMD) or `nccl-tests` (NVIDIA) at three scopes for each collective:

| Scope | World size | Procs/node | Why |
|---|---|---|---|
| `intra_node` | 8 (per-node, parallel across all nodes) | 8 | xGMI/NVLink only |
| `inter_node` | N (one GPU per node, single ring) | 1 | isolates IB |
| `world` | N × 8 (full ring) | 8 | actual training topology |

Collectives to measure: `allreduce`, `allgather`, `reduce_scatter`, `broadcast`, `alltoall` (alltoall typically only at intra_node and world).

Sizes: `[1, 16, 64, 256] MB` (extend to 1024 MB if MoE is in scope).

For each (scope, collective, size) record `bw_gbs` and `latency_us`. For `intra_node` (per-node parallel), also keep `per_node_bw_gbs` and roll it up into `median / min / max / stddev_pct` so slow nodes pop out.

A node whose `intra_node.AR.min_bw_gbs[256MB]` is < 0.7 × median is added to `slow_nodes_at_max_size`. **Flag, do not auto-blacklist** — the user decides.

### Step 6: Env probe

For each preset in `env-catalog` matching the hardware class:

1. **Connectivity check (fail-fast)**: tiny AllReduce (2 GPUs across 2 nodes if multi-node) with the candidate env. If it doesn't connect within 30s, mark `tentative` and try the next preset variant.
2. **Micro-bench validation**: re-run `rccl-tests intra_node + inter_node` with the candidate env; require ≥ 90% of the probed peak from Step 5.
3. **Multi-node smoke**: 2-node, 100-step toy training (Primus' tiny example) to confirm no hang / crash.

The first preset that passes all three becomes the validated `env_baseline` (`status: validated`). Anything that passes only step 1+2 is `status: tentative` — usable but warn the user.

### Step 7: Write `output/pilot/cluster-<cluster_id>.md`

```markdown
# Cluster Baseline — <cluster_id>

- **Version**: <cluster_class>-<nodes>node-v<n>
- **Collected at**: <iso8601>
- **Mode**: single | slurm
- **Nodes**: <healthy>/<total>; gpus_per_node: <m>

## Compute peak (per-node median)

- BF16 GEMM peak: <X> TFLOPS  (per-node variance: <Y>%)
- FP8 GEMM peak:  <X> TFLOPS  (when supported)
- HBM capacity:   <Z> GB
- HBM bandwidth:  <Z> GB/s

## Interconnect

- Intra-node: xGMI / NVLink, ~<X> GB/s effective per pair
- Inter-node: IB / AINIC, ~<X> GB/s per-GPU effective, uniformity: uniform | heterogeneous

## RCCL/NCCL baseline (median bw_gbs at sizes [1, 16, 64, 256] MB)

| scope       | AR        | AG        | RS        | AT          |
|-------------|-----------|-----------|-----------|-------------|
| intra_node  | …         | …         | …         | …           |
| inter_node  | …         | …         | …         | (n/a)       |
| world       | …         | …         | …         | …           |

(Latency table: same shape, in µs.)

## Slow nodes (flagged, NOT blacklisted)

- node<X>: intra_node AR @ 256MB = <bw> GB/s vs cluster median <bw_med> (Δ -<%>)

## Env baseline (validated)

```yaml
version: <cluster_class>-<nodes>node-v<n>
status: validated | tentative
rccl:
  NCCL_IB_HCA: "..."
  NCCL_NET_GDR_LEVEL: 4
  NCCL_IB_GID_INDEX: 3
  NCCL_SOCKET_IFNAME: ...
hsa:
  HSA_FORCE_FINE_GRAIN_PCIE: 1
  GPU_MAX_HW_QUEUES: 2
alloc:
  PYTORCH_HIP_ALLOC_CONF: "expandable_segments:True"
threading:
  OMP_NUM_THREADS: 8
```

## Notes

- ROCm version: <X>; driver: <Y>
- Known issues: ...
- Re-run when: driver upgrade, fabric topology change, > 1 month old
```

### Step 8: Return summary to the parent

The main agent gets a 5-line summary:

```markdown
## Cluster Preflight (cluster_id: <id>)

- nodes: <healthy>/<total>, gpus_per_node: <m>
- bf16 peak: <X> TFLOPS, hbm: <Z> GB
- intra/inter/world AR @256MB: <a>/<b>/<c> GB/s
- env baseline: <preset_tag>, status: validated, slow_nodes: [<node>...]
- file: output/pilot/cluster-<id>.md  (Read on demand)
```

Anything more detailed is in the file.

## How `tuning-loop` consumes this

```
Stage 0 of tuning-loop:
  Read header of output/pilot/cluster-<id>.md
  Extract: peak_tflops_bf16, hbm_capacity_gb, BW_eff[scope][coll][size], env_baseline
  Cache these in chat.
  Discard the rest until needed.
```

`execution-model` formulas read `peak_tflops_*` and `BW_eff[*]`. `optimize-*` env-sweeps read `env_baseline` to decide which flags to vary. `bottleneck-diagnose` reads `hbm_capacity_gb` for the MEMORY_BOUND threshold.

## What to collect (canonical fields list — single-source for the file)

This is the definitive list. If you cannot collect a field, write `n/a` (don't omit the row).

| Section | Field |
|---|---|
| Header | `cluster_id`, `version`, `cluster_class`, `collected_at`, `mode`, `nodes_total`, `nodes_healthy`, `gpus_per_node` |
| Compute | `peak_tflops_bf16`, `peak_tflops_fp8`, `hbm_capacity_gb`, `hbm_bandwidth_gbs`, `per_node_variance_pct` |
| Interconnect | `intra_node.type`, `intra_node.bandwidth_gbs`, `inter_node.type`, `inter_node.bandwidth_gbs`, `inter_node.uniformity` |
| RCCL/NCCL | for each (scope, coll, size): `bw_gbs` and `latency_us` (+ per-node breakdown for intra_node) |
| Slow nodes | list of node names that fall < 0.7× median on any roll-up |
| Env baseline | full `env_baseline` block + `version` + `status` (validated / tentative) |
| Notes | ROCm/driver versions, known issues |

This list mirrors the legacy `ClusterProfile` schema content but lives as a markdown checklist, not a JSON Schema.

## Important Notes

- **Pilot does not allocate**: the SLURM job / docker container must be up before preflight runs. If it isn't, surface an error and stop.
- **`output/pilot/cluster-<id>.md` is the only persistence**. No JSON file, no YAML state. The next session just reads it.
- **Slow nodes are flagged, not blacklisted**. The user decides whether to drop them; auto-blacklist is the wrong default.
- **The env baseline preset comes from `env-catalog`** — preflight only validates / minor-tweaks the preset, it doesn't invent a new env from scratch.
- **Re-run when the cluster changes**: ROCm / driver / firmware bump, fabric reshuffling, or 7+ days old. Stale baselines silently bias every subsequent prediction.
- **Tentative status is OK to start tuning**, but the user should know — note it in the tuning-loop session header.
- **Don't mix preflight runs with tuning runs**. Preflight should be its own subagent; the parent only sees the final 5-line summary.
