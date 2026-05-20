---
name: env-catalog
description: Single-source catalog of NCCL / RCCL / HSA / HIP / allocator / threading environment variables for AMD ROCm and NVIDIA CUDA training. Use whenever the agent needs to look up what an env flag does, its default, safe range, known pitfalls, or which bottleneck it is relevant to. Triggers: NCCL_*, RCCL_*, HSA_*, HIP_*, GPU_MAX_HW_QUEUES, PYTORCH_HIP_ALLOC_CONF, PYTORCH_CUDA_ALLOC_CONF, expandable_segments, OMP_NUM_THREADS, MKL_NUM_THREADS, NCCL_IB_HCA, NCCL_BUFFSIZE, MSCCL, env preset, env baseline.
---

# Env Catalog — One Place to Look Up Every Flag

This is the **only** place that defines what each env flag means, its safe range, its known pitfalls, and the cluster-class presets we have validated. The `optimize-*` skills point to flag names; full descriptions live here. When you add a new flag, add it here once.

The catalog is split by domain: RCCL/NCCL, HSA/HIP, allocator, threading, presets.

## How to use this catalog

1. The current cluster's **env baseline** is in `output/pilot/cluster-<cluster_id>.md` (produced by `preflight`). That is the golden default.
2. A candidate plan ships only `env_diff` (diff vs baseline). Look up each diff key in this catalog before approving the candidate.
3. Each row tells you: name → default → safe range → which bottleneck flips it → pitfalls.

## RCCL / NCCL (communication)

| Flag | Default | Safe range / values | Used for | Pitfalls |
|---|---|---|---|---|
| `NCCL_IB_HCA` | unset → autodetect | comma list of `<dev>:<port>` from preflight | force right NICs | wrong list = traffic on wrong card; *must* match preflight env_baseline |
| `NCCL_NET_GDR_LEVEL` | `2` | `0`–`5`; `4` recommended on AMD IB | GPUDirect RDMA | level 5 needs proper kernel module; can hang on misconfig |
| `NCCL_IB_GID_INDEX` | `0` | usually `3` on RoCEv2 | GID for IB | wrong GID → fall back to TCP, 10× slower |
| `NCCL_SOCKET_IFNAME` | autodetect | `bond0` / `ens9np0` / cluster-specific | bootstrap & fallback iface | wrong iface → connect timeout |
| `NCCL_BUFFSIZE` | `4 MB` | `4M` – `64M`; `16M` typical | Tier-0 COMM | very large (≥ 128M) eats ~1–4 GB HBM/rank |
| `NCCL_MIN_NCHANNELS` | autodetect | `8` / `16` / `32` | parallel rings | over-tuned → no benefit, more overhead |
| `NCCL_MAX_NCHANNELS` | autodetect | usually leave default | parallel rings | rarely needs override |
| `NCCL_ALGO` | autodetect | `Ring` / `Tree` / `CollnetDirect` | force algorithm | overrides hurt more often than help |
| `NCCL_PROTO` | autodetect | `Simple` / `LL` / `LL128` | small msg path | `Simple` sometimes wins for small p2p in PP |
| `NCCL_P2P_LEVEL` | `LOC` | `LOC` / `NVL` / `PIX` / `SYS` | constrain P2P scope | misuse can disable P2P entirely |
| `NCCL_DEBUG` | unset | `WARN` / `INFO` (debug only) | diagnostics | `INFO` floods log, slows runs |
| `RCCL_MSCCL_ENABLE` | `0` | `0` / `1` (AMD-only) | algorithmic AllReduce | enables MSCCL path; may conflict with custom `NCCL_ALGO` |
| `NCCL_IB_TIMEOUT` | `20` | `18`–`23` (log scale) | tolerate slow links | too low → spurious timeouts |
| `NCCL_IB_RETRY_CNT` | `7` | `5`–`12` | retransmits | too high masks real link issues |
| `NCCL_CHECKS_DISABLE` | `0` | leave `0` for prod | sanity checks | `1` only if you know what you're doing |

Notes:

- On AMD ROCm, the variable names are still `NCCL_*` (RCCL exposes the same surface). `RCCL_*`-prefixed flags are AMD-specific extensions.
- The preflight env-probe already sets `NCCL_IB_HCA`, `NCCL_NET_GDR_LEVEL`, `NCCL_IB_GID_INDEX`, `NCCL_SOCKET_IFNAME` to validated values — **do not override** these in `env_diff` unless the cluster baseline tag has changed.

## HSA / HIP (AMD runtime)

| Flag | Default | Safe range | Used for | Pitfalls |
|---|---|---|---|---|
| `HSA_FORCE_FINE_GRAIN_PCIE` | `0` | `0` / `1` | Tier-0 COMPUTE / COMM | enable for better PCIe granularity on most setups |
| `HSA_NO_SCRATCH_RECLAIM` | `0` | typically `1` | reduce scratch thrash | leaving `0` can cause perf jitter |
| `HSA_ENABLE_SDMA` | `1` | `0` / `1` | SDMA copies | `0` only as workaround for known SDMA bugs |
| `GPU_MAX_HW_QUEUES` | `2` | `2` / `4` / `8` | concurrent kernel queues | `> 8` rarely helps |
| `HIP_VISIBLE_DEVICES` | unset | comma list | mask GPUs | use only outside Primus' device assignment |
| `HIP_LAUNCH_BLOCKING` | `0` | leave `0` | sync launches | `1` cripples perf — debug only |
| `NVTE_CK_USES_BWD_V3` | varies | `0` / `1` | TE backward kernel selection | model-/version-specific; sweep when COMPUTE_BOUND |

## Allocator

| Flag | Default | Safe range | Used for | Pitfalls |
|---|---|---|---|---|
| `PYTORCH_HIP_ALLOC_CONF` | unset | `expandable_segments:True[,max_split_size_mb:N][,garbage_collection_threshold:F]` | Tier-0 MEMORY | requires ROCm ≥ 6.0 for `expandable_segments`; older versions crash |
| `PYTORCH_CUDA_ALLOC_CONF` | unset | same shape, NVIDIA | Tier-0 MEMORY | same contract |
| `MALLOC_TRIM_THRESHOLD_` | system | `131072` | host-side fragmentation | rarely needed in containers |
| `MALLOC_MMAP_THRESHOLD_` | system | `131072` | host-side mmap path | rarely needed |

`PYTORCH_HIP_ALLOC_CONF` recipe presets (paste into `env_diff`):

| Recipe | Value |
|---|---|
| Default fragmentation fix | `expandable_segments:True` |
| Mixed-size workloads | `expandable_segments:True,max_split_size_mb:512` |
| Aggressive GC | `expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.8` |

## Threading / NUMA

| Flag | Default | Safe range | Used for | Pitfalls |
|---|---|---|---|---|
| `OMP_NUM_THREADS` | system | `4` / `8` / `16` | Tier-0 COMPUTE | very large → contention; usually `cores_per_rank / 2` |
| `MKL_NUM_THREADS` | system | match `OMP_NUM_THREADS` | MKL paths | mismatch with OMP can cause oversubscription |
| `OMP_PROC_BIND` | unset | `close` / `spread` | pin threads | usually leave default |
| `KMP_AFFINITY` | unset | `granularity=fine,compact,1,0` | Intel MKL pinning | overlaps with OMP_PROC_BIND |
| `numactl --cpunodebind, --membind` | unset | local NUMA node id | DRAM locality | mismatch with GPU NUMA = perf cliff |

## GLOO (CPU collective bootstrap)

| Flag | Default | Safe range | Used for | Pitfalls |
|---|---|---|---|---|
| `GLOO_SOCKET_IFNAME` | autodetect | match `NCCL_SOCKET_IFNAME` | rendezvous | wrong iface → init hang |

## Per-cluster-type presets (validated combinations)

These are starting points for `env_baseline` when bringing up a new cluster of a known class. The actual baseline lives in `output/pilot/cluster-<cluster_id>.md`; this section is the seed.

### MI300X 8-GPU/node, ROCm ≥ 6.2, IB

```yaml
rccl:
  NCCL_IB_HCA: "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1"   # adjust per cluster
  NCCL_NET_GDR_LEVEL: 4
  NCCL_IB_GID_INDEX: 3
  NCCL_SOCKET_IFNAME: bond0
hsa:
  HSA_FORCE_FINE_GRAIN_PCIE: 1
  GPU_MAX_HW_QUEUES: 2
alloc:
  PYTORCH_HIP_ALLOC_CONF: "expandable_segments:True"
threading:
  OMP_NUM_THREADS: 8
gloo:
  GLOO_SOCKET_IFNAME: bond0
```

### MI355X 8-GPU/node, AINIC fabric (ionic devices)

```yaml
rccl:
  NCCL_IB_HCA: "ionic_0:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1,ionic_9:1"
  NCCL_NET_GDR_LEVEL: 4
  NCCL_IB_GID_INDEX: 3
  NCCL_SOCKET_IFNAME: ens9np0
hsa:
  HSA_FORCE_FINE_GRAIN_PCIE: 1
  GPU_MAX_HW_QUEUES: 4
alloc:
  PYTORCH_HIP_ALLOC_CONF: "expandable_segments:True,max_split_size_mb:512"
threading:
  OMP_NUM_THREADS: 8
gloo:
  GLOO_SOCKET_IFNAME: ens9np0
extras:
  USING_AINIC: 1
```

### NVIDIA H100 8-GPU/node, NVLink, IB

```yaml
nccl:
  NCCL_IB_HCA: "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1"
  NCCL_NET_GDR_LEVEL: 5
  NCCL_IB_GID_INDEX: 3
  NCCL_SOCKET_IFNAME: bond0
alloc:
  PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True"
threading:
  OMP_NUM_THREADS: 8
```

If the cluster does not match any preset, run `preflight.env_probe` to derive the baseline empirically.

## Incompatibility / dangerous-combo matrix

| Combo | Result | Mitigation |
|---|---|---|
| `expandable_segments:True` + ROCm < 6.0 | crash | upgrade ROCm or omit |
| `RCCL_MSCCL_ENABLE=1` + custom `NCCL_ALGO` override | undefined | choose one |
| `NCCL_NET_GDR_LEVEL=5` + missing peer mem driver | startup hang | drop to 4 |
| `HIP_LAUNCH_BLOCKING=1` in production | severe perf loss | debug-only |
| `NCCL_DEBUG=INFO` + long run | log file blowup | only short debug runs |
| `OMP_NUM_THREADS` = total cores | host contention | use `≤ cores_per_rank / 2` |
| `GPU_MAX_HW_QUEUES > 8` | wasted resources | cap at 8 |
| `NCCL_BUFFSIZE ≥ 128M` + tight HBM | OOM risk | re-check memory budget |

`run-and-profile` should reject any candidate whose `env_diff` matches a row in this table.

## Important Notes

- **Add new flags here, only here.** All `optimize-*` skills reference flags by name; the source of truth is this file.
- **Always express env changes as `env_diff` vs the cluster baseline**, never as a full env.
- **The cluster baseline is preflight-derived**, not hardcoded. The presets above are seeds, not authority.
- **When unsure of safe range, default to leaving the flag alone**. The Primus team's defaults are usually right.
- **Do not override `NCCL_IB_HCA` / `NCCL_SOCKET_IFNAME` casually**. Misrouting traffic to the wrong NIC silently degrades perf without crashing — very hard to debug.
- **Cluster upgrades invalidate the baseline.** When ROCm / driver / firmware moves, re-run `preflight` and bump the env_baseline version tag.
