# Node-Smoke Test — Quick-Start Instructions

A short get-started guide for the per-node preflight smoke test. For the full design / aggregator section reference / implementation history, see [node-smoke.md](./node-smoke.md).

---

## 1. What it does

A lightweight, distributed-rendezvous-free preflight check that runs on every node in parallel under SLURM. It produces a **single PASS/FAIL verdict per node** plus SLURM-ready `passing_nodes.txt` / `failing_nodes.txt` you can pipe straight into `srun --nodelist=` / `--exclude=`.

Use it to **screen a cluster fast and exclude bad nodes before launching a real training job**. A bad GPU, NIC, wedged driver, or leaked process on any node will surface as a node FAIL — without a single global rendezvous, so a stuck node can't wedge its peers.

---

## 2. Prerequisites


| Prerequisite                        | How                                                                                                                                                                                                                                                                                                                                                                                              |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Python venv on a shared filesystem  | Same venv used by `primus-cli direct -- preflight` (see `[preflight-direct.md](./preflight-direct.md)` §2).                                                                                                                                                                                                                                                                                      |
| `VENV_ACTIVATE` exported            | `export VENV_ACTIVATE=~/envs/preflight/.venv/bin/activate` (optional inside the container path).                                                                                                                                                                                                                                                                                                 |
| Inside an existing SLURM allocation | One task per node. Recommended: `runner/primus-cli slurm srun -N "$SLURM_NNODES" --ntasks-per-node=1 -- direct -- node_smoke ...`. Equivalent bare form: `srun ... --ntasks-per-node=1 runner/primus-cli direct -- node_smoke ...`. Either way the `direct -- node_smoke` path auto-selects `--single`, so each task spawns one Python process and per-GPU subprocesses are launched internally. |


No `MASTER_ADDR`, no `MASTER_PORT`, no global rendezvous required.

---

## 3. Quick start

> ⚠ **Set the NCCL / RCCL environment first** if you plan to run with `--tier2-perf` (the local 8-GPU RCCL all-reduce). Even though the smoke test never opens a cross-node rendezvous, the Tier 2 RCCL step calls `dist.init_process_group(backend="nccl", ...)`, and RCCL **enumerates every transport at init** (XGMI / PCIe P2P + IB + sockets). A misconfigured `NCCL_IB_HCA` / `NCCL_SOCKET_IFNAME` / `NCCL_IB_GID_INDEX` can stall init or make the all-reduce silently fall back to a slow path. The launcher's `base_env.sh` auto-detects these via `get_nccl_ib_hca.sh` + `get_ip_interface.sh`, **but auto-detect sometimes picks the wrong values inside a container** (devices masked by the network namespace, frontend NICs picked up instead of fabric NICs, etc.) so you usually want to check these settings and set them explicitly if auto-detection is wrong.
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
> See `[preflight-direct.md` § 4 Cluster-specific NCCL configuration](./preflight-direct.md#4-cluster-specific-nccl-configuration) for the canonical Broadcom / Pensando Pollara values (the same `NCCL_*` set is used by both tools). If you skip `--tier2-perf`, the RCCL step is not executed and none of the above applies — Tier 1 (host limits, RDMA roll-call, leaked-process detection, etc.) does not depend on RCCL.
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

> **Both forms produce the same workload.** The wrapper form is recommended because it resolves the distributed env once on the launching node and propagates it via `--env`, and applies any `slurm.`* config defaults (partition / time / etc.). See `[preflight-direct.md` § Wrapper vs. bare-srun](./preflight-direct.md#wrapper-vs-bare-srun) for the precedence table.

---

## 4. More examples (by configuration knob)

> **Convention used below.** The examples in this section are written with bare `srun` for brevity. Anywhere you see `srun <flags> runner/primus-cli direct -- node_smoke ...`, the equivalent wrapper form is `runner/primus-cli slurm srun <flags> -- direct -- node_smoke ...`. Pick whichever matches your habits; both target the same launcher.

### 4.1 Hard-fail on partial NIC enumeration

Catches "7 of 8 RDMA NICs visible" — common cause of crashes after RoCE init.

```bash
srun -N "$SLURM_NNODES" --ntasks-per-node=1 \
    runner/primus-cli direct -- node_smoke --tier2-perf --expected-rdma-nics 8
```

### 4.2 Tighten Tier 2 perf thresholds

Reject GPUs that come in below your acceptance bar. Defaults: GEMM 600 TFLOPS, HBM 2000 GB/s, local RCCL 100 GB/s.

```bash
srun -N "$SLURM_NNODES" --ntasks-per-node=1 \
    runner/primus-cli direct -- node_smoke --tier2-perf \
        --gemm-tflops-min 700 --hbm-gbs-min 4500 --rccl-gbs-min 180
```

### 4.3 Tighten host limits

Fail nodes whose `RLIMIT_MEMLOCK` or `/dev/shm` is too small for production training.

```bash
srun -N "$SLURM_NNODES" --ntasks-per-node=1 \
    runner/primus-cli direct -- node_smoke \
        --ulimit-l-min-gb 64 --shm-min-gb 16
```

### 4.4 Custom dump path

Keep one report per smoke run instead of overwriting the default location.

```bash
srun -N "$SLURM_NNODES" --ntasks-per-node=1 \
    runner/primus-cli direct -- node_smoke --tier2-perf \
        --dump-path /shared/smoke-archive/$(date +%Y%m%d-%H%M%S)
```

### 4.5 Allow / extend the foreign-process whitelist

By default, leaked / foreign processes holding a GPU FAIL the node (most common cause of "training fails to launch on a healthy-looking node"). Allowed by default: `gpuagent,rocm-smi-daemon,amd-smi,dcgm-exporter`.

```bash
# Add a site-specific monitoring agent to the whitelist
srun ... runner/primus-cli direct -- node_smoke \
    --allowed-procs gpuagent,rocm-smi-daemon,amd-smi,dcgm-exporter,my-monitor

# Don't fail at all on foreign processes (still reported in the markdown)
srun ... runner/primus-cli direct -- node_smoke --allow-foreign-procs
```

#### Containers: `name='N/A'` false positives → use `--allow-foreign-procs`

> ⚠ **Running node_smoke inside a container almost always trips this check.** `amd-smi process --json` reports `name="N/A"` for kernel/system PIDs like `gpuagent` whose `/proc/<host_pid>/comm` it cannot read, and the fallback `_resolve_proc_name(pid)` inside `node_smoke` then also fails because the container's `/proc` typically does not expose host PIDs (private PID namespace without `--pid=host`, or a `hidepid=2` mount). The unresolved name doesn't match the allowlist (`gpuagent,rocm-smi-daemon,...`), so the check fires and the node FAILs — even though the only "foreign" processes are well-known system daemons holding zero HBM.
>
> **In the container path, pass `--allow-foreign-procs`:**
>
> ```bash
> srun ... runner/primus-cli direct -- node_smoke --tier2-perf --allow-foreign-procs
> ```
>
> The processes are still listed in `smoke_report.md` under "Busy GPUs / leaked processes" so a real leak is still visible; only the FAIL verdict is downgraded.
>
> **Narrower alternative** if you want the check to still catch leaks with resolvable names (e.g. a leftover `python` rank), add the literal sentinel `N/A` to the allowlist:
>
> ```bash
> srun ... runner/primus-cli direct -- node_smoke --tier2-perf \
>     --allowed-procs gpuagent,rocm-smi-daemon,amd-smi,dcgm-exporter,N/A
> ```
>
> The annotator runs `_resolve_proc_name` first, so whenever a real name *can* be resolved (on the host, or after fixing `/proc` visibility) it overrides "N/A" and the normal allowlist applies. The `N/A` entry only matches PIDs whose name genuinely could not be recovered — strictly narrower than `--allow-foreign-procs`.
>
> **Root-cause fix** (preferred long-term): grant the container access to host PIDs so `_resolve_proc_name` works and the report shows real names (`gpuagent`, etc.) instead of `N/A`. Typical fixes:
>
> - Launch with `--pid=host` (Docker / Podman) so host PIDs are directly addressable.
> - Mount `/proc` without `hidepid=2`.
> - Loosen `ptrace_scope` or grant `CAP_SYS_PTRACE`.
>
> Once any of those is in place, `_resolve_proc_name` finds the names, the default allowlist matches them, and you no longer need `--allow-foreign-procs`.

### 4.6 Require specific tools

Make missing CLI tools a hard FAIL (default: warn-only).

```bash
srun ... runner/primus-cli direct -- node_smoke --require-tools amd-smi,rocm-smi,lsof
```

### 4.7 Skip dmesg scan (containers with no privileges)

```bash
srun ... runner/primus-cli direct -- node_smoke --skip-dmesg
```

### 4.8 Re-aggregate from existing per-node JSONs (no re-run)

Useful when you only want to refresh the markdown report, or when you've collected JSONs separately.

```bash
# From any node, no allocation needed if you're just reading local files.
# The primus-cli wrapper always runs both phases, so use the standalone
# aggregate subcommand for "aggregate only" -- it reads the existing
# <dump>/smoke/*.json without re-running the per-node smoke step.
python -m primus.tools.preflight.node_smoke aggregate \
    --dump-path output/preflight --expected-nodes 6 --wait-timeout-sec 5
```

### 4.9 Silent mode (for CI)

Suppresses wrapper stdout, but the **final report path is still printed** and stderr / exit code are preserved.

```bash
srun ... runner/primus-cli direct --silent -- node_smoke --tier2-perf
```

### 4.10 Combined "production-ready screen"

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

## 5. Outputs

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

## 6. Common knobs (cheat sheet)


| Flag                         | Default                                          | When you'd change it                                                                  |
| ---------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------- |
| `--tier2-perf`               | off                                              | Always on for production screens — adds GEMM TFLOPS, HBM GB/s, local RCCL all-reduce. |
| `--gemm-tflops-min N`        | 600                                              | Site-specific acceptance bar.                                                         |
| `--hbm-gbs-min N`            | 2000                                             | Site-specific acceptance bar (MI300X healthy ≈ 4500–5000).                            |
| `--rccl-gbs-min N`           | 100                                              | Site-specific acceptance bar.                                                         |
| `--expected-rdma-nics N`     | unset                                            | Hard-fail on partial NIC enumeration.                                                 |
| `--ulimit-l-min-gb GB`       | 32                                               | Raise for production training profiles.                                               |
| `--shm-min-gb GB`            | 8                                                | Raise for large-batch / many-rank profiles.                                           |
| `--allow-foreign-procs`      | off                                              | Co-tenant clusters or shared GPUs.                                                    |
| `--allowed-procs LIST`       | `gpuagent,rocm-smi-daemon,amd-smi,dcgm-exporter` | Add site-specific monitoring agents.                                                  |
| `--require-tools LIST`       | `""`                                             | Fail-fast if a CLI tool is missing in PATH.                                           |
| `--skip-dmesg`               | off                                              | Inside unprivileged containers.                                                       |
| `--dump-path DIR`            | `output/preflight`                               | Archive each run separately.                                                          |
| `--silent` (wrapper)         | off                                              | CI / scripted runs.                                                                   |
| `--aggregate-only` (wrapper) | off                                              | Re-render report without re-running per-node checks.                                  |


For the full flag list and the aggregator subcommand, see `python -m primus.tools.preflight.node_smoke run --help` and `... aggregate --help`, or `[node-smoke.md](./node-smoke.md)` §"Configuration knobs".

---

## 7. Troubleshooting


| Symptom                                                                       | Likely cause / fix                                                                                                                                                                            |
| ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `[ERROR] [direct] VENV_ACTIVATE is set but file does not exist: ...`          | Fix the path (`export VENV_ACTIVATE=~/envs/preflight/.venv/bin/activate`), or `unset VENV_ACTIVATE` to fall back to system / container Python.                                                |
| Every node FAILs with `gpu_processes: ... name='N/A'`                         | Should no longer happen after the `/proc/<pid>/comm` fallback fix. If it does, check that `/proc/<pid>/comm` is readable on the node (`hidepid` mount?). Workaround: `--allow-foreign-procs`. |
| Some nodes never produce a JSON                                               | Aggregator names them in `failing_nodes.txt` via `expected_nodes.txt`. If `scontrol` was unavailable, they'll appear as `<missing-N>`.                                                        |
| Tier 2 perf numbers below threshold on a known-good node                      | Almost always insufficient CPU cores on `srun` — pass `-c <cores-per-node>` so RCCL proxy threads have CPU.                                                                                   |
| Re-run on a smaller nodelist still shows the previously removed nodes as PASS | Default behavior cleans stale JSONs on rank 0. If you passed `--no-clean-dump-path`, either remove it or `rm -rf output/preflight` between runs.                                              |


---

## 8. See also

- `[node-smoke.md](./node-smoke.md)` — full design, aggregator sections, configuration reference, implementation history.
- `[preflight-direct.md](./preflight-direct.md)` — the heavier `preflight` tool with global rendezvous and inter-node bandwidth tests.
- `[primus/cli/subcommands/node_smoke.py](../primus/cli/subcommands/node_smoke.py)` — the primus-cli subcommand wiring (two-phase dispatch: rank-N run + rank-0 aggregate).
- `[primus/tools/preflight/node_smoke/cli.py](../primus/tools/preflight/node_smoke/cli.py)` — canonical flag definitions and per-node / aggregate phase bodies.
