# Node-smoke test instruction

A lightweight, distributed-rendezvous-free preflight check that runs on every node in parallel under SLURM. It produces a **single PASS/FAIL verdict per node** plus SLURM-ready `passing_nodes.txt` / `failing_nodes.txt` you can pipe straight into `srun --nodelist=` / `--exclude=`.

Use it to **screen a cluster fast and exclude bad nodes before launching a real training job**. A bad GPU, NIC, wedged driver, or leaked process on any node surfaces as a node FAIL — without a single global rendezvous, so a stuck node can't wedge its peers.

- **Recommended launcher**: `runner/primus-cli slurm srun -- direct -- node_smoke ...` (auto-resolves the distributed env, applies `slurm.*` config defaults, same pattern as `train` / `benchmark`). The shorter `runner/primus-cli direct -- node_smoke ...` (bare `srun` + `direct`) is equivalent and handy for ad-hoc runs.
- **Companion tool**: [`preflight`](./preflight.md) — the heavier diagnostic with a global rendezvous and inter-node bandwidth tests. The recommended workflow is **node-smoke first, preflight second** (see [§10](#10-comparison-with-the-full-preflight)).

---

## 1. What it does

Node-smoke answers one question fast: **"which nodes are healthy enough to run anything?"** Because training jobs allocate whole nodes, a single degraded GPU (or NIC, or wedged driver) takes an entire node out of rotation. Node-smoke checks each node independently and emits a per-node verdict plus a ready-to-use exclude list, so you can prune broken nodes before committing a large job to a global rendezvous.

It deliberately does **not** measure cross-node bandwidth — that's what [`preflight`](./preflight.md) is for.

---

## 2. How it works

- **Per-node and independent** — every node runs the checks on its own. No `MASTER_ADDR`, no `MASTER_PORT`, no global `torch.distributed` rendezvous, so a stuck node cannot wedge its peers.
- **Per-GPU isolation** — each GPU's checks run in their own Python subprocess with a hard timeout (`--per-gpu-timeout-sec`, default 15 s). A stuck `torch.cuda.set_device()` (which `signal.alarm` cannot interrupt because it sits inside a driver syscall) is `SIGKILL`'d from the parent without affecting the rest of the node's checks.
- **Local-only RCCL** — the optional Tier 2 all-reduce uses `torch.multiprocessing.spawn` over `tcp://127.0.0.1`. No cross-node communication.
- **Rank-0 aggregation** — `NODE_RANK==0` polls for the expected number of per-node JSONs (with a timeout), computes cluster-wide drift, writes the Markdown report + pass/fail lists, and returns non-zero if any node FAILs or never reports.

---

## 3. Prerequisites

| Prerequisite                        | How                                                                                                                                                                                                                                                                                                                                                                                              |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Python venv on a shared filesystem  | Same venv used by `primus-cli direct -- preflight` (see [`preflight-without-container.md`](./preflight-without-container.md) §2).                                                                                                                                                                                                                                                                 |
| `VENV_ACTIVATE` exported            | `export VENV_ACTIVATE=~/envs/preflight/.venv/bin/activate` (optional inside the container path).                                                                                                                                                                                                                                                                                                 |
| Inside an existing SLURM allocation | One task per node. Recommended: `runner/primus-cli slurm srun -N "$SLURM_NNODES" --ntasks-per-node=1 -- direct -- node_smoke ...`. Equivalent bare form: `srun ... --ntasks-per-node=1 runner/primus-cli direct -- node_smoke ...`. Either way the `direct -- node_smoke` path auto-selects `--single`, so each task spawns one Python process and per-GPU subprocesses are launched internally. |

No `MASTER_ADDR`, no `MASTER_PORT`, no global rendezvous required.

---

## 4. Quick start

**Git clone the Primus repository to a shared filesystem that all nodes can read.**

```bash
git clone --recurse-submodules https://github.com/AMD-AGI/Primus.git
cd Primus
```

**Note: remember to set up the Python virtual environment and NCCL / fabric environment variables as described in [§3 Prerequisites](#3-prerequisites).**

> ⚠ **Set the NCCL / RCCL environment first** if you plan to run with `--tier2-perf` (the local 8-GPU RCCL all-reduce). Even though the smoke test never opens a cross-node rendezvous, the Tier 2 RCCL step calls `dist.init_process_group(backend="nccl", ...)`, and RCCL **enumerates every transport at init** (XGMI / PCIe P2P + IB + sockets). A misconfigured `NCCL_IB_HCA` / `NCCL_SOCKET_IFNAME` / `NCCL_IB_GID_INDEX` can stall init or make the all-reduce silently fall back to a slow path. The launcher's `base_env.sh` auto-detects these, **but auto-detect sometimes picks the wrong values inside a container** (devices masked by the network namespace, frontend NICs picked up instead of fabric NICs, etc.), so check them and set them explicitly if auto-detection is wrong.
>
> Minimum-viable checklist before running with `--tier2-perf`:
>
> ```bash
> # Pin the RDMA / RoCE training NICs the container can actually see.
> # On a bare-metal host the auto-detect in base_env.sh usually picks
> # the right set; inside a container or on a multi-role node, list
> # them explicitly. Use the same set you would pass to a training job.
> export NCCL_IB_HCA="rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7"
>
> # Pick the RoCE v2 GID index for your fabric:
> #   - Mellanox / Broadcom: typically 3 (base_env.sh default).
> #   - Pensando Pollara (AINIC): 1.
> export NCCL_IB_GID_INDEX=3
>
> # The bootstrap socket interface. Auto-detect prefers the first
> # non-loopback interface from `hostname -I`; override when that
> # picks a frontend NIC instead of the data-plane interface.
> export NCCL_SOCKET_IFNAME=eno0
> export GLOO_SOCKET_IFNAME=eno0
> ```
>
> See [`preflight-without-container.md` §4 Cluster-specific NCCL configuration](./preflight-without-container.md#4-cluster-specific-nccl-configuration) for the canonical Broadcom / Pensando Pollara values (the same `NCCL_*` set is used by both tools). If you skip `--tier2-perf`, the RCCL step is not executed and none of the above applies — Tier 1 (host limits, RDMA roll-call, leaked-process detection, etc.) does not depend on RCCL.
>
> Quick verification: `runner/primus-cli direct --dry-run -- node_smoke --tier2-perf` prints the resolved `NCCL_*` block under "Environment Variables" so you can confirm the values before launching for real.

Recommended — through the `primus-cli slurm srun` wrapper (auto-resolves `MASTER_ADDR`/`NNODES`/`NODE_RANK`, applies `slurm.*` config defaults):

```bash
export VENV_ACTIVATE=~/envs/preflight/.venv/bin/activate

# Basic Tier 1 check (~5 s/GPU, ~30 s total)
runner/primus-cli slurm srun -N "$SLURM_NNODES" --ntasks-per-node=1 \
    -- direct -- node_smoke

# Tier 1 + Tier 2 perf sanity (GEMM TFLOPS, HBM GB/s, local 8-GPU RCCL)
runner/primus-cli slurm srun -N "$SLURM_NNODES" --ntasks-per-node=1 \
    -- direct -- node_smoke --tier2-perf

# Then re-run training, excluding any node the smoke test failed:
srun --exclude=$(paste -sd, output/preflight/failing_nodes.txt) ... your-real-job
```

Equivalent with bare `srun` (works the same; useful when composing with custom `srun` flags):

```bash
srun -N "$SLURM_NNODES" --ntasks-per-node=1 \
    runner/primus-cli direct -- node_smoke

srun -N "$SLURM_NNODES" --ntasks-per-node=1 \
    runner/primus-cli direct -- node_smoke --tier2-perf
```

Single-node sanity check (no SLURM):

```bash
runner/primus-cli direct -- node_smoke
```

> **Both forms produce the same workload.** The wrapper form is recommended because it resolves the distributed env once on the launching node and propagates it via `--env`, and applies any `slurm.`* config defaults (partition / time / etc.). The `direct` keyword between the two `--`s is **mandatory** to take the no-container path — without it the wrapper routes through the container path. See [`preflight-without-container.md` § Wrapper vs. bare-srun](./preflight-without-container.md#wrapper-vs-bare-srun) for the precedence table.

---

## 5. What's checked

A `level='fail'` finding in any check FAILs the node. Everything else is reported as info / warn.

### Tier 1 — always runs (~5 s/GPU)

**Per-GPU liveness** (each GPU in its own subprocess with a hard timeout):

- `torch.cuda.set_device(i)` — proves the device is bindable (a stale / wedged GPU often fails here).
- 256 MB allocation.
- Tiny 2048² bf16 GEMM with an `isfinite()` check on the result.

**Host / GPU / network inventory** (no rendezvous):

- **dmesg recent-error scan** — greps the last `--dmesg-minutes` (default 15) of `dmesg` for known patterns (`xid`, `gpu reset`, `hung_task`, `mce:`, `amdgpu.*error`, ...). Matches are surfaced in the report.
- **A. Software-stack fingerprint** — kernel / OS / Python, ROCm version, amdgpu kernel-module version, PyTorch / `torch.version.hip` / RCCL versions, and per-IB-device firmware + HCA model. Used for cluster drift detection.
- **B. NIC / RDMA roll-call** — per-port state read from `/sys/class/infiniband` (works inside containers; no `ibv_devinfo` / `ibstat` dependency). Many clusters expose more RDMA ports than the training job uses, so the hard-fail rules only run against the *training-NIC* subset, selected by this precedence:
  1. `--rdma-nic-allowlist` (`NCCL_IB_HCA` syntax: comma-separated `device[:port]`, `^...` denylist, `=dev` exact-match).
  2. `NCCL_IB_HCA` env (same syntax) — so the smoke test and the training launch agree by construction.
  3. Heuristic: auto-exclude any port whose `phys_state` is `Disabled` or `Sleep` (admin-disabled).
  4. Fallback: every IB port must be ACTIVE / LinkUp.

  **Hard-fail rules** (on the included set only): port not ACTIVE / not LinkUp; active port with zero RoCE v2 GIDs (RoCE) or zero valid GIDs (IB); included-NIC count ≠ `--expected-rdma-nics N` (when set). If *every* discovered port gets excluded, the node still fails — a node with zero training NICs cannot participate in inter-node training. Excluded ports stay visible in the report for diagnostics but don't contribute to the FAIL signal.
- **C. Host limits / system tunables** — `RLIMIT_MEMLOCK` below `--ulimit-l-min-gb` (default 32 GiB) → "RDMA pin will fail under load"; `/dev/shm` below `--shm-min-gb` (default 8 GiB) → "NCCL shared-mem may fail". NUMA node count, CPU count, and `cpu0` scaling governor are collected for drift detection.
- **Foreign / leaked process detection** — foreign PIDs holding a GPU FAIL the node by default (the most common cause of "training fails to launch on a healthy-looking node"). Allowed by default: `gpuagent,rocm-smi-daemon,amd-smi,dcgm-exporter`. See the [container note in §6.5](#65-allow--extend-the-foreign-process-whitelist) — running inside a container almost always needs `--allow-foreign-procs`.
- **rocm-smi self-latency** — a `rocm-smi --version` call slower than `--rocm-smi-timeout-sec` (default 5 s) is a node FAIL; a wedging amdgpu driver typically hangs `rocm-smi` for 30–60 s before the GPU itself stops responding.

### Tier 2 — optional perf sanity (`--tier2-perf`)

Per-GPU steady-state metrics, with iteration counts aligned to the preflight `--quick` preset so smoke and preflight numbers are directly comparable. It's a single switch — you cannot enable just one half.

- **GEMM TFLOPS** — 8192³ bf16 `torch.matmul`; FAIL below `--gemm-tflops-min` (default 600).
- **HBM GB/s** — 512 MB device-to-device `torch.Tensor.copy_` (counts read + write); FAIL below `--hbm-gbs-min` (default 2000; a healthy MI300X is ≈ 4500–5000).
- **Local 8-GPU RCCL all-reduce GB/s** — algorithmic bandwidth `2·S·(P-1)/P / t / 1e9` at `--rccl-size-mb` (default 64 MB); FAIL below `--rccl-gbs-min` (default 100). Local only, no cross-node traffic.

---

## 6. Examples (by configuration knob)

> **Convention used below.** The examples are written with bare `srun` for brevity. Anywhere you see `srun <flags> runner/primus-cli direct -- node_smoke ...`, the equivalent wrapper form is `runner/primus-cli slurm srun <flags> -- direct -- node_smoke ...`. Pick whichever matches your habits; both target the same launcher.

### 6.1 Hard-fail on partial NIC enumeration

Catches "7 of 8 RDMA NICs visible" — a common cause of crashes after RoCE init. The count is compared against the *training-NIC* set (after the selector chain), so frontend / storage RoCE NICs do not inflate it.

```bash
srun -N "$SLURM_NNODES" --ntasks-per-node=1 \
    runner/primus-cli direct -- node_smoke --tier2-perf --expected-rdma-nics 8
```

### 6.2 Pin the training-NIC set explicitly

When auto-detection picks the wrong ports, name the training NICs directly (otherwise `NCCL_IB_HCA` env is used; otherwise admin-disabled ports are auto-excluded):

```bash
srun -N "$SLURM_NNODES" --ntasks-per-node=1 \
    runner/primus-cli direct -- node_smoke --tier2-perf \
    --rdma-nic-allowlist 'rocep158s0:1,rocep190s0:1,rocep206s0:1,rocep222s0:1,rocep28s0:1,rocep62s0:1,rocep79s0:1,rocep96s0:1'
```

### 6.3 Tighten Tier 2 perf thresholds

Reject GPUs that come in below your acceptance bar. Defaults: GEMM 600 TFLOPS, HBM 2000 GB/s, local RCCL 100 GB/s.

```bash
srun -N "$SLURM_NNODES" --ntasks-per-node=1 \
    runner/primus-cli direct -- node_smoke --tier2-perf \
        --gemm-tflops-min 700 --hbm-gbs-min 4500 --rccl-gbs-min 180
```

### 6.4 Tighten host limits

Fail nodes whose `RLIMIT_MEMLOCK` or `/dev/shm` is too small for production training.

```bash
srun -N "$SLURM_NNODES" --ntasks-per-node=1 \
    runner/primus-cli direct -- node_smoke \
        --ulimit-l-min-gb 64 --shm-min-gb 16
```

### 6.5 Allow / extend the foreign-process whitelist

By default, leaked / foreign processes holding a GPU FAIL the node. Allowed by default: `gpuagent,rocm-smi-daemon,amd-smi,dcgm-exporter`.

```bash
# Add a site-specific monitoring agent to the whitelist
srun ... runner/primus-cli direct -- node_smoke \
    --allowed-procs gpuagent,rocm-smi-daemon,amd-smi,dcgm-exporter,my-monitor

# Don't fail at all on foreign processes (still reported in the markdown)
srun ... runner/primus-cli direct -- node_smoke --allow-foreign-procs
```

> ⚠ **Running node_smoke inside a container almost always trips this check.** `amd-smi process --json` reports `name="N/A"` for kernel/system PIDs like `gpuagent` whose `/proc/<host_pid>/comm` it cannot read, and the fallback name resolution inside `node_smoke` also fails because the container's `/proc` typically does not expose host PIDs (private PID namespace without `--pid=host`, or a `hidepid=2` mount). The unresolved name doesn't match the allowlist, so the check fires and the node FAILs — even though the only "foreign" processes are well-known system daemons holding zero HBM.
>
> **In the container path, pass `--allow-foreign-procs`:**
>
> ```bash
> srun ... runner/primus-cli direct -- node_smoke --tier2-perf --allow-foreign-procs
> ```
>
> The processes are still listed in `smoke_report.md` under "Busy GPUs / leaked processes", so a real leak is still visible; only the FAIL verdict is downgraded.
>
> **Narrower alternative** — add the literal sentinel `N/A` to the allowlist so the check still catches leaks with resolvable names (e.g. a leftover `python` rank):
>
> ```bash
> srun ... runner/primus-cli direct -- node_smoke --tier2-perf \
>     --allowed-procs gpuagent,rocm-smi-daemon,amd-smi,dcgm-exporter,N/A
> ```
>
> Name resolution runs first, so whenever a real name *can* be resolved it overrides `N/A` and the normal allowlist applies — the `N/A` entry only matches PIDs whose name genuinely could not be recovered.
>
> **Root-cause fix** (preferred long-term): grant the container access to host PIDs so names resolve and the report shows `gpuagent` etc. instead of `N/A`. Typical fixes: launch with `--pid=host` (Docker / Podman); mount `/proc` without `hidepid=2`; or loosen `ptrace_scope` / grant `CAP_SYS_PTRACE`.

### 6.6 Require specific tools

Make missing CLI tools a hard FAIL (default: warn-only).

```bash
srun ... runner/primus-cli direct -- node_smoke --require-tools amd-smi,rocm-smi,lsof
```

### 6.7 Skip dmesg scan (containers with no privileges)

```bash
srun ... runner/primus-cli direct -- node_smoke --skip-dmesg
```

### 6.8 Custom dump path

Keep one report per smoke run instead of overwriting the default location.

```bash
srun -N "$SLURM_NNODES" --ntasks-per-node=1 \
    runner/primus-cli direct -- node_smoke --tier2-perf \
        --dump-path /shared/smoke-archive/$(date +%Y%m%d-%H%M%S)
```

### 6.9 Re-aggregate from existing per-node JSONs (no re-run)

The primus-cli wrapper always runs both phases (per-node run + rank-0 aggregate). To *only* re-render the report from JSONs collected earlier, use the standalone `aggregate` subcommand — it reads the existing `<dump>/smoke/*.json` without re-running the per-node step:

```bash
python -m primus.tools.preflight.node_smoke aggregate \
    --dump-path output/preflight --expected-nodes 6 --wait-timeout-sec 5
```

### 6.10 Silent mode (for CI)

Suppresses wrapper stdout, but the **final report path is still printed** and stderr / exit code are preserved.

```bash
srun ... runner/primus-cli direct --silent -- node_smoke --tier2-perf
```

### 6.11 Combined "production-ready screen"

A representative one-shot for a production cluster screen:

```bash
srun -N "$SLURM_NNODES" --ntasks-per-node=1 \
    runner/primus-cli direct --silent -- node_smoke --tier2-perf \
        --expected-rdma-nics 8 \
        --gemm-tflops-min 700 --hbm-gbs-min 4500 --rccl-gbs-min 180 \
        --ulimit-l-min-gb 64 --shm-min-gb 16 \
        --require-tools amd-smi,rocm-smi,lsof \
        --dump-path /shared/smoke-archive/$(date +%Y%m%d-%H%M%S)
```

---

## 7. Outputs

All written under `--dump-path` (default `output/preflight/`).

| File                      | Purpose                                                                                                              |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `smoke/<short-host>.json` | Per-node verdict + every collected metric. One file per node.                                                        |
| `smoke_report.md`         | Human-readable cluster report (status table, drift sections, perf summary, failing-node detail).                     |
| `passing_nodes.txt`       | Newline-separated short hostnames. Pipe into `srun --nodelist=`.                                                     |
| `failing_nodes.txt`       | Newline-separated short hostnames. Pipe into `srun --exclude=`.                                                      |
| `expected_nodes.txt`      | Auto-populated from `scontrol show hostnames "$SLURM_JOB_NODELIST"`. Lets the report name nodes that never reported. |

Read the cluster verdict at a glance:

```bash
head -10 output/preflight/smoke_report.md
```

Feed bad nodes into a re-run:

```bash
srun --exclude=$(paste -sd, output/preflight/failing_nodes.txt) ... your-real-job
```

---

## 8. Understanding the report

`smoke_report.md` renders in a stable order. Each section short-circuits to a placeholder (e.g. `*All nodes match.*`, `*No NIC issues.*`) on a healthy cluster, so a clean report stays short. In order:

1. **Status table** — one row per node: `node_rank`, hostname, PASS/FAIL, duration, top fail reason.
2. **Stack drift across cluster** — per fingerprint key, outliers vs the cluster majority (catches "1 of N nodes on a different RCCL build").
3. **NIC firmware drift across cluster** — per-IB-device firmware drift.
4. **NIC / RDMA roll-call issues** — every offending node + port (included set only).
5. **NIC port-count summary** — cluster-majority *training-NIC* count and any node that disagrees (catches partial-NIC degradation even without `--expected-rdma-nics`).
6. **NIC excluded ports (informational)** — ports the selector chain dropped, grouped by source. Does not contribute to FAIL.
7. **Host limits issues** — per-node hard-limit violations.
8. **GPU visibility issues** — nodes where torch couldn't see the GPUs, or amd-smi sees more GPUs than torch (stale ROCm / wedged driver).
9. **GPU low-level outliers (PCIe link / HBM)** — per-GPU outliers vs the cluster majority on PCIe width/speed and HBM total.
10. **XGMI link issues** — any non-XGMI GPU pair (intra-node collectives silently fall back to PCIe).
11. **Cluster clock + time daemons** — wall-clock spread plus per-node time-daemon health.
12. **Tooling self-latency (`rocm-smi --version`)** — slow / timed-out tool calls (precursor to a wedged driver).
13. **Tooling availability** — inventory of `amd-smi` / `rocm-smi` / `lsof` per node, plus which Tier 1 checks have no working tool on each node.
14. **Busy GPUs / leaked processes** — foreign PIDs holding GPUs at smoke start.
15. **GPU pre-touch HBM usage outliers** — GPUs with non-trivial HBM in use *before* smoke touched the device.
16. **GPU compute-activity outliers** — GPUs above `--gpu-activity-warn-pct` at smoke start (warn-only).
17. **Tier 2 perf summary** (only when at least one node ran Tier 2) — per-node GEMM TFLOPS / HBM GB/s as `min / median / max`, plus local RCCL GB/s.
18. **Failing nodes — full reasons** (only when there are failing nodes) — every fail reason, expanded per node.

---

## 9. Configuration reference

### 9.1 Common knobs (cheat sheet)

| Flag                     | Default                                          | When you'd change it                                                                  |
| ------------------------ | ------------------------------------------------ | ------------------------------------------------------------------------------------- |
| `--tier2-perf`           | off                                              | Always on for production screens — adds GEMM TFLOPS, HBM GB/s, local RCCL all-reduce. |
| `--gemm-tflops-min N`    | 600                                              | Site-specific acceptance bar.                                                         |
| `--hbm-gbs-min N`        | 2000                                             | Site-specific acceptance bar (MI300X healthy ≈ 4500–5000).                            |
| `--rccl-gbs-min N`       | 100                                              | Site-specific acceptance bar.                                                         |
| `--expected-rdma-nics N` | unset                                            | Hard-fail on partial NIC enumeration.                                                 |
| `--ulimit-l-min-gb GB`   | 32                                               | Raise for production training profiles.                                               |
| `--shm-min-gb GB`        | 8                                                | Raise for large-batch / many-rank profiles.                                           |
| `--allow-foreign-procs`  | off                                              | Co-tenant clusters, shared GPUs, or the container path (see [§6.5](#65-allow--extend-the-foreign-process-whitelist)). |
| `--allowed-procs LIST`   | `gpuagent,rocm-smi-daemon,amd-smi,dcgm-exporter` | Add site-specific monitoring agents.                                                  |
| `--require-tools LIST`   | `""`                                             | Fail-fast if a CLI tool is missing in PATH.                                           |
| `--skip-dmesg`           | off                                              | Inside unprivileged containers.                                                       |
| `--dump-path DIR`        | `output/preflight`                               | Archive each run separately.                                                          |
| `--silent` (launcher)    | off                                              | CI / scripted runs.                                                                   |

### 9.2 Full `node_smoke` (per-node) flags

Authoritative source: `python -m primus.tools.preflight.node_smoke run --help`.

| Flag | Default | Purpose |
|---|---|---|
| `--dump-path` | `output/preflight` | Output directory. |
| `--expected-gpus N` | auto | Override GPU count (auto-detected from `LOCAL_WORLD_SIZE` / `GPUS_PER_NODE` / `torch.cuda.device_count()`). |
| `--per-gpu-timeout-sec` | 15 | Hard timeout per per-GPU subprocess. |
| `--tier2-perf` | off | Enable Tier 2 perf sanity (per-GPU GEMM TFLOPS + HBM GB/s + node-local RCCL all-reduce). Single switch. |
| `--gemm-tflops-min` | 600 | Tier 2 GEMM threshold. |
| `--hbm-gbs-min` | 2000 | Tier 2 HBM threshold. |
| `--rccl-size-mb` | 64 | Local RCCL message size. |
| `--rccl-gbs-min` | 100 | Local RCCL bandwidth threshold. |
| `--rccl-timeout-sec` | 120 | Hard timeout for the RCCL phase. |
| `--skip-dmesg` | off | Skip dmesg scan (e.g. inside containers). |
| `--dmesg-minutes` | 15 | dmesg `--since` window. |
| `--expected-rdma-nics N` | auto (report-only) | When set, a mismatch between the included (training-NIC) count and N becomes a node FAIL. |
| `--rdma-nic-allowlist LIST` | unset | Explicit training-NIC selector in `NCCL_IB_HCA` syntax (`device[:port],...`, `^...` denylist, `=dev` exact-match). Wins over `NCCL_IB_HCA` env. When neither is set, ports whose `phys_state` is `Disabled` / `Sleep` are auto-excluded. |
| `--ulimit-l-min-gb GB` | 32 | `RLIMIT_MEMLOCK` threshold (0 disables). |
| `--shm-min-gb GB` | 8 | `/dev/shm` size threshold (0 disables). |
| `--rocm-smi-timeout-sec SEC` | 5.0 | Hard timeout for the `rocm-smi --version` self-latency canary; hitting it is a node FAIL. |
| `--hbm-busy-threshold-gib GiB` | 2.0 | FAIL if any GPU has ≥ this much HBM in use before smoke touches the device. Boundary inclusive. |
| `--allow-foreign-procs` | off | Do NOT FAIL on foreign processes holding a GPU. They are still reported. |
| `--allowed-procs LIST` | `gpuagent,rocm-smi-daemon,amd-smi,dcgm-exporter` | Process names OK to find holding the GPU. Set to `""` to disable the whitelist. |
| `--gpu-activity-warn-pct PCT` | 20.0 | Warn (does NOT fail) if any GPU's `gfx_activity_pct` exceeds this at smoke start. |
| `--require-tools LIST` | `""` (warn-only) | CLI tools that MUST be in PATH (`amd-smi`, `rocm-smi`, `lsof`); anything missing becomes a hard node FAIL. |
| `--no-clean-dump-path` | off | Do NOT auto-wipe stale per-node JSONs / aggregator outputs from `--dump-path` on rank 0 at startup. |

### 9.3 Standalone `aggregate` flags

The primus-cli wrapper runs the aggregator automatically on rank 0 and fills `--expected-nodes` / `--expected-nodelist-file` from SLURM. These matter only when you invoke `python -m primus.tools.preflight.node_smoke aggregate` yourself (see [§6.9](#69-re-aggregate-from-existing-per-node-jsons-no-re-run)).

| Flag | Default | Purpose |
|---|---|---|
| `--dump-path` | `output/preflight` | Same as `run`. |
| `--expected-nodes N` | none | If fewer JSONs land within `--wait-timeout-sec`, missing nodes are added as FAIL placeholders. |
| `--wait-timeout-sec` | 60 | Polling timeout. |
| `--rocm-smi-warn-sec SEC` | 1.0 | Flag (warn-only) any node where `rocm-smi --version` took longer than this. |
| `--clock-skew-warn-sec SEC` | 30.0 | Warn when wall-clock spread across nodes exceeds this many seconds (includes srun launch jitter). |
| `--hbm-busy-threshold-gib GiB` | 2.0 | Mirrors the `run` default; labels the "GPU pre-touch HBM usage outliers" section. |
| `--gpu-activity-warn-pct PCT` | 20.0 | Mirrors the `run` default; labels the "GPU compute-activity outliers" section. |
| `--expected-nodelist-file FILE` | none | One short hostname per line. Missing nodes get their real short hostname in the report and `failing_nodes.txt`. The wrapper auto-populates this from `scontrol show hostnames "$SLURM_JOB_NODELIST"` under SLURM. |

### 9.4 Launcher-level knobs (`primus-cli direct`)

Consumed by `primus-cli-direct.sh` **before** the `--` separator (not forwarded to the `node_smoke` Python tool):

| Flag | Purpose |
|---|---|
| `--silent` | Redirect launcher + tool stdout to `/dev/null`. Launcher errors (`LOG_ERROR` / `LOG_WARN` on stderr) and the log file are preserved. Exit code propagated. |
| `--debug` | Verbose launcher logging. |
| `--dry-run` | Print the resolved configuration and command without executing. |
| `--env KEY=VALUE` | Inject an env var into the Python process. |

> **Run vs. aggregate.** The primus-cli `node_smoke` subcommand always runs the per-node checks on every rank, then aggregates on rank 0 — which is what you want ~100% of the time, so there is no `--aggregate-only` wrapper flag. For the rare single-phase cases, call the standalone CLI directly: `python -m primus.tools.preflight.node_smoke run ...` (per-node only, no report) or `... aggregate ...` (report only, from existing JSONs).

---

## 10. Comparison with the full `preflight`

| Aspect | `node_smoke` | full `preflight` |
|---|---|---|
| Rendezvous | None — every node independent | Global `torch.distributed` |
| Wall clock | ~30–60 s for 6 nodes (Tier 1+2) | Minutes; scales with N for inter-node tests |
| Granularity | Per-node PASS/FAIL | Per-rank measurements (no auto-fail by default) |
| GEMM | Hard threshold per GPU | Reports per-GPU numbers, no auto-fail |
| HBM bandwidth | Yes (D2D `copy_`) | Not measured |
| Inter-node all-reduce / all-to-all | Not tested (intentionally) | Yes |
| Drift detection | Yes (versions, NIC firmware, port count) | No |
| Host limits / RDMA roll-call | Yes (hard fail) | Reported via `collect_*_info` only |
| Output format | Per-node JSON + cluster md + SLURM-ready txt | Markdown + PDF |

Use `node_smoke` to **screen** a cluster fast and exclude bad nodes. Use the full [`preflight`](./preflight.md) when you want **deep cross-node measurements** (inter-node bandwidth matrix, ring-P2P, etc.). The recommended sequence is node-smoke first, then `preflight --quick` on the surviving nodes.

---

## 11. Troubleshooting

| Symptom                                                                       | Likely cause / fix                                                                                                                                                                            |
| ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `[ERROR] [direct] VENV_ACTIVATE is set but file does not exist: ...`          | Fix the path (`export VENV_ACTIVATE=~/envs/preflight/.venv/bin/activate`), or `unset VENV_ACTIVATE` to fall back to system / container Python.                                                |
| Every node FAILs with `gpu_processes: ... name='N/A'`                         | Container `/proc` can't resolve host PID names. See [§6.5](#65-allow--extend-the-foreign-process-whitelist): pass `--allow-foreign-procs`, or grant host-PID visibility.                       |
| Some nodes never produce a JSON                                               | The aggregator names them in `failing_nodes.txt` via `expected_nodes.txt`. If `scontrol` was unavailable, they appear as `<missing-N>`.                                                       |
| Tier 2 perf numbers below threshold on a known-good node                      | Almost always insufficient CPU cores on `srun` — pass `-c <cores-per-node>` so RCCL proxy threads have CPU.                                                                                   |
| Re-run on a smaller nodelist still shows the previously removed nodes as PASS | Default behavior cleans stale JSONs on rank 0. If you passed `--no-clean-dump-path`, either remove it or `rm -rf output/preflight` between runs.                                              |

---

## 12. See also

- [`preflight.md`](./preflight.md) — the heavier `preflight` tool with a global rendezvous and inter-node bandwidth tests.
- [`preflight-without-container.md`](./preflight-without-container.md) — running `preflight` directly on the host (no container), including the shared venv + NCCL setup.
- [`primus/cli/subcommands/node_smoke.py`](../../primus/cli/subcommands/node_smoke.py) — the primus-cli subcommand wiring (two-phase dispatch: per-rank run + rank-0 aggregate).
- [`primus/tools/preflight/node_smoke/cli.py`](../../primus/tools/preflight/node_smoke/cli.py) — canonical flag definitions and per-node / aggregate phase bodies.
