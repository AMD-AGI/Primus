# Run Preflight Without a Container

> ⚠ **Run the [node-smoke test](./node-smoke-test-instruction.md) first.** `preflight` opens a global `torch.distributed` rendezvous, so a single sick node (wedged driver, leaked rank holding HBM, partial NIC enumeration, time-sync drift, etc.) can stall the whole job for up to `--dist-timeout-sec` seconds — long before any cross-node bandwidth number is produced. The node-smoke test catches those exact failure modes *without* a rendezvous in ~30–60 s and emits a SLURM-ready `failing_nodes.txt` you can pipe straight into `srun --exclude=`. Treat node-smoke as a hard prerequisite; only run `preflight` on the nodes node-smoke marked PASS. See [§0 "Which test should I run?"](#0-which-test-should-i-run) for the side-by-side comparison and the recommended 3-step workflow.

This guide explains how to run Primus's `[preflight](./preflight.md)` cluster-diagnostic tool **directly on the host** (no Docker / Podman), via the standard Primus launcher.

**Git clone the Primus repository to a shared filesystem that all nodes can read.**

```bash
git clone --recurse-submodules https://github.com/AMD-AIG-AIMA/Primus.git
cd Primus
git checkout dev/preflight-direct-test
```

**Recommended (through the primus-cli SLURM wrapper):**

```
runner/primus-cli slurm srun -N <NNODES> --ntasks-per-node=1 -- direct -- preflight [PREFLIGHT_ARGS...]
```

**Equivalent (bare srun, useful when composing with custom srun flags):**

```
srun -N <NNODES> --ntasks-per-node=1 runner/primus-cli direct -- preflight [PREFLIGHT_ARGS...]
```

Both forms produce the **same workload** on the same ranks. The wrapper form is recommended because it auto-resolves `MASTER_ADDR` / `MASTER_PORT` / `NNODES` / `NODE_RANK` / `GPUS_PER_NODE` once on the launching node and passes them to every rank via `--env`, applies any `slurm.`* config defaults (partition / time / etc.) from your YAML, and is the same pattern used for `train` / `benchmark` / `node_smoke`. See [§ Wrapper vs. bare-srun](#wrapper-vs-bare-srun) below for the exact precedence / caveats.

`primus-cli direct` activates an optional Python virtualenv (`VENV_ACTIVATE`), auto-derives the distributed environment variables (`NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `GPUS_PER_NODE`) from `SLURM_*` when running inside a SLURM allocation, and then launches the `preflight` Python subcommand via `torchrun` (one worker per GPU). It is the recommended entry point when:

- You're running on a SLURM cluster but cannot (or don't want to) use the container-based path.
- Your nodes share a Python virtual environment on a network-mounted filesystem.
- You want a single-node sanity check with no extra configuration.

---

## 0. Which test should I run?

Primus ships **two** complementary cluster screens. Pick the right one — and ideally run them in this order.


| Aspect            | `node-smoke` (start here)                                                          | `preflight` (this doc)                                              |
| ----------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Purpose           | "Which nodes are healthy enough to run anything?"                                  | "What is the actual cross-node performance on the surviving nodes?" |
| Rendezvous        | None — every node independent                                                      | Global `torch.distributed` rendezvous                               |
| Wall clock        | ~30–60 s for 6 nodes (Tier 1+2)                                                    | A few minutes; scales with N for inter-node tests                   |
| Granularity       | Per-node PASS/FAIL                                                                 | Per-rank perf measurements                                          |
| Safety            | A stuck node cannot wedge its peers                                                | A single hung NIC can stall the whole rendezvous                    |
| Output            | Per-node JSON + cluster md + SLURM-ready `passing_nodes.txt` / `failing_nodes.txt` | Markdown + PDF perf report                                          |
| Entry point       | `primus-cli direct -- node_smoke`                                                  | `primus-cli direct -- preflight` (this doc)                         |
| Quick-start guide | `[node-smoke-test-instruction.md](./node-smoke-test-instruction.md)`               | This doc, §3+                                                       |


### Recommended workflow

> **Before running any of the commands below, complete the one-time setup:**
>
> 1. **Python virtualenv** on a shared filesystem — see [§2 Set up the Python virtual environment](#2-set-up-the-python-virtual-environment), then point the launcher at it via `export VENV_ACTIVATE=...` (details in [§2 → Tell the launcher where the venv is](#tell-the-launcher-where-the-venv-is)).
> 2. **NCCL / fabric environment variables** — usually the defaults in `base_env.sh` are fine, but multi-NIC nodes may need `NCCL_IB_HCA` / `NCCL_IB_GID_INDEX` / `NCCL_SOCKET_IFNAME` overrides. See [§4 Cluster-specific NCCL configuration](#4-cluster-specific-nccl-configuration) for known-good values per fabric (Broadcom, Pensando Pollara/AINIC).

Through the `primus-cli slurm srun -- direct --` wrapper (recommended):

```bash
# 1) Prune broken nodes with node-smoke (fast, no rendezvous).
runner/primus-cli slurm srun -N "$SLURM_NNODES" --ntasks-per-node=1 \
    -- direct -- node_smoke --tier2-perf

# 2) Re-allocate excluding the bad nodes, and run preflight --quick
#    for a fast cross-node sanity check.
runner/primus-cli slurm srun -N <good-nnodes> -c 128 --gpus-per-node=8 \
    --ntasks-per-node=1 \
    --exclude=$(paste -sd, output/preflight/failing_nodes.txt) \
    -- direct -- preflight --quick

# 3) Optional: full preflight on the same set if --quick numbers
#    look off, or if you want the full bandwidth matrix.
runner/primus-cli slurm srun -N <good-nnodes> -c 128 --gpus-per-node=8 \
    --ntasks-per-node=1 \
    --exclude=$(paste -sd, output/preflight/failing_nodes.txt) \
    -- direct -- preflight
```

Equivalent with bare `srun` (works identically; useful when scripting around custom srun flags that don't compose with the wrapper):

```bash
srun -N "$SLURM_NNODES" --ntasks-per-node=1 \
    runner/primus-cli direct -- node_smoke --tier2-perf

srun -N <good-nnodes> -c 128 --gpus-per-node=8 --ntasks-per-node=1 \
    --exclude=$(paste -sd, output/preflight/failing_nodes.txt) \
    runner/primus-cli direct -- preflight --quick

srun -N <good-nnodes> -c 128 --gpus-per-node=8 --ntasks-per-node=1 \
    --exclude=$(paste -sd, output/preflight/failing_nodes.txt) \
    runner/primus-cli direct -- preflight
```

Why this ordering matters:

- A single broken node can stall a `torch.distributed.init_process_group()` for `--dist-timeout-sec` seconds (default 120), so feeding a known-good list to preflight is much faster.
- `node-smoke` catches things preflight cannot — leaked / foreign processes, wedged drivers, partial NIC enumeration, time-sync drift, RDMA roll-call issues — that produce *misleading* preflight failures.
- `preflight --quick` adds the cross-node bandwidth signal that `node-smoke` deliberately does not measure.

---

## 1. Prerequisites

- A working AMD ROCm installation on every node.
- Network reachability between nodes (Ethernet for bootstrap, RDMA / InfiniBand recommended for perf tests).
- A Python ≥ 3.10 virtual environment **on a shared filesystem** that all nodes can read (the same path is sourced on every node).
- The Primus repository checked out somewhere readable from every node.

---

## 2. Set up the Python virtual environment

The environment must live on a path visible from every node (e.g. NFS-mounted home, Lustre, or any shared filesystem). All nodes will `source` the same activation script.

You can use any tool you like; `uv` is the fastest. Either of the following works.

### What you actually need to install

The `preflight` and `node-smoke` tools deliberately use **only a small subset** of Primus's full dependency tree. You do **not** need to install the entire `requirements.txt` — that pulls in trainer / dataset / experiment-tracking packages that neither tool ever imports.


| Package              | Required for                                                                          | Skip when                                                                                  |
| -------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `torch` (ROCm build) | Both tools — perf measurements (`torch.matmul`, `torch.distributed`, `torch.cuda.`*). | Never (mandatory).                                                                         |
| `markdown2`          | `preflight` PDF report only (Markdown → HTML).                                        | You always pass `--disable-pdf`, or you only run `node-smoke` (which never produces PDFs). |
| `weasyprint`         | `preflight` PDF report only (HTML → PDF).                                             | Same as above.                                                                             |
| `matplotlib`         | `preflight --plot` only (per-test bandwidth bar charts).                              | You don't pass `--plot`.                                                                   |


Everything else in the preflight / node-smoke code path is Python stdlib (`os`, `subprocess`, `socket`, `argparse`, `dataclasses`, `json`, `time`, ...) — no extra installs needed.

### Option A — `uv` (recommended), minimal install

```bash
mkdir -p ~/envs/preflight
cd ~/envs/preflight

uv venv --python 3.12
source .venv/bin/activate

# 1) ROCm-built PyTorch (pin to your ROCm version; rocm7.1 shown here)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1 --no-cache-dir

# 2) Optional: only if you want preflight PDF reports (omit to use --disable-pdf)
uv pip install markdown2 weasyprint

# 3) Optional: only if you want preflight --plot bar charts
uv pip install matplotlib
```

### Option B — `python -m venv`, minimal install

```bash
mkdir -p ~/envs/preflight
python3.12 -m venv ~/envs/preflight/.venv
source ~/envs/preflight/.venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1 --no-cache-dir
pip install markdown2 weasyprint   # optional, for preflight PDFs
pip install matplotlib             # optional, for preflight --plot
```

### Option C — full Primus runtime (only if you also want the rest of Primus)

```bash
cd /path/to/Primus
uv pip install -r requirements.txt   # or: pip install -r requirements.txt
```

This installs every Primus runtime dependency (trainer, dataset loaders, experiment trackers, ...). Use only if you're going to run more than just preflight / node-smoke from this environment.

### Per-tool minimum install matrix

If you want the absolute smallest footprint, install only what your intended invocations need:


| Invocation                                       | `torch`  | `markdown2`                       | `weasyprint`                      | `matplotlib` |
| ------------------------------------------------ | -------- | --------------------------------- | --------------------------------- | ------------ |
| `node-smoke` (any flags)                         | required | —                                 | —                                 | —            |
| `preflight --host --gpu --network --disable-pdf` | required | —                                 | —                                 | —            |
| `preflight --host --gpu --network` (with PDF)    | required | required                          | required                          | —            |
| `preflight --quick --disable-pdf`                | required | —                                 | —                                 | —            |
| `preflight --quick` (with PDF)                   | required | required                          | required                          | —            |
| `preflight ... --plot`                           | required | required (unless `--disable-pdf`) | required (unless `--disable-pdf`) | required     |


### Tell the launcher where the venv is

`primus-cli direct` reads the `**VENV_ACTIVATE**` environment variable. When set, it sources the path before launching the Python process; when unset, it is a no-op (the container path, which uses the container's bundled Python, leaves this unset):

```bash
export VENV_ACTIVATE=~/envs/preflight/.venv/bin/activate
```

`VENV_ACTIVATE` is the only optional environment variable specific to the direct flow. Everything else has a sensible default; distributed-env variables (`NNODES`, `NODE_RANK`, `MASTER_ADDR`, ...) are auto-derived from SLURM when not pre-exported.

---

## 3. Run preflight

### Single node (no SLURM)

```bash
export VENV_ACTIVATE=~/envs/preflight/.venv/bin/activate

# Info report only (fast)
runner/primus-cli direct -- preflight --host --gpu --network

# Info + perf report
runner/primus-cli direct -- preflight

# Perf report only
runner/primus-cli direct -- preflight --perf-test
```

When SLURM is not detected the script defaults to `NNODES=1`, `NODE_RANK=0`, `MASTER_ADDR=localhost`. Any of those can be overridden by exporting them before calling the script.

### Multi-node without SLURM (parallel SSH)

When no scheduler is available (bare-metal, cloud VMs, lab nodes), launch
`primus-cli direct` on each node yourself via SSH. The script works
identically — you just pre-export the distributed variables that SLURM
would normally provide.

#### Requirements

- All nodes share the same filesystem (or at least the same Primus checkout + venv path).
- Nodes can reach each other on a **data-plane** network interface (not the management NIC).
- SSH key-based access to each node from the launching host.

#### Required environment variables


| Variable             | Description                                                            |
| -------------------- | ---------------------------------------------------------------------- |
| `NNODES`             | Total number of nodes                                                  |
| `NODE_RANK`          | This node's rank (`0` through `NNODES-1`)                              |
| `MASTER_ADDR`        | IP of rank-0 node **on the data-plane interface**                      |
| `MASTER_PORT`        | Rendezvous port (default `1234`; increment between concurrent runs)    |
| `GPUS_PER_NODE`      | GPUs per node (default `8`)                                            |
| `NCCL_SOCKET_IFNAME` | Data-plane NIC name (e.g. `enp159s0np0`) — **critical for multi-node** |
| `GLOO_SOCKET_IFNAME` | Same as `NCCL_SOCKET_IFNAME`                                           |
| `VENV_ACTIVATE`      | Path to virtualenv `activate` script                                   |


> **Warning**: `NCCL_SOCKET_IFNAME` auto-detection often picks a management interface
> (e.g. `enp28s0np0`, `eno8303`) instead of the high-bandwidth data NIC. For multi-node
> runs this causes `init_process_group` to hang or NCCL to fail silently. Always set it
> explicitly.

#### Identifying the correct data-plane interface

```bash
# On any node, find the interface whose IP matches the MASTER_ADDR subnet:
ip -4 addr show | grep "10.245.134"
# → enp159s0np0  inet 10.245.134.129/24

# Or check which interface routes to the master:
ip route get 10.245.134.129 | awk '{print $5; exit}'
### Multi-node via SLURM

`primus-cli direct` auto-detects a SLURM allocation (via `SLURM_JOB_ID`) and derives all distributed variables from `SLURM_*` automatically. **Pre-exported values always win**, so the same launcher script also works inside the `primus-cli slurm srun ... -- direct -- ...` chain (where `slurm-entry` has already set these via `--env`):

| Variable        | Resolved as                                                          |
| --------------- | -------------------------------------------------------------------- |
| `NNODES`        | `NNODES` → `SLURM_NNODES` → `SLURM_JOB_NUM_NODES` → `1`              |
| `NODE_RANK`     | `NODE_RANK` → `SLURM_NODEID` → `SLURM_PROCID` → `0`                  |
| `MASTER_ADDR`   | `MASTER_ADDR` (if not empty / not `localhost`) → first hostname from `scontrol show hostnames "$SLURM_NODELIST"` |
| `MASTER_PORT`   | `MASTER_PORT` → `1234`                                               |
| `GPUS_PER_NODE` | `GPUS_PER_NODE` → `8`                                                |

Run it as a single task per node (the script invokes `torchrun` internally, which spawns one worker per GPU):

> **Verify NCCL / network env first.** The script sets sensible `NCCL_`* defaults via `base_env.sh`, but auto-detection can pick the wrong device on multi-NIC nodes. Always confirm `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`, `NCCL_SOCKET_IFNAME`, and `GLOO_SOCKET_IFNAME` (set to the same value as `NCCL_SOCKET_IFNAME`) are correct for your fabric, and `export` overrides before running. See [§4](#4-cluster-specific-nccl-configuration) for cluster-specific values.

```bash
export VENV_ACTIVATE=~/envs/preflight/.venv/bin/activate

# export NCCL_IB_HCA=rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eno0
# export GLOO_SOCKET_IFNAME=eno0

# Recommended: through the primus-cli SLURM wrapper.
runner/primus-cli slurm srun -t 00:45:00 -N 4 -c 128 --gpus-per-node=8 \
    --nodelist <nodes> --ntasks-per-node=1 \
    -- direct -- preflight --perf-test

# Or, equivalently, with bare srun:
srun -t 00:45:00 -N 4 -c 128 --gpus-per-node=8 --nodelist <nodes> \
    --ntasks-per-node=1 \
    runner/primus-cli direct -- preflight --perf-test
```



#### Wrapper vs. bare-srun

Both forms target the **same** `primus-cli-direct.sh` launcher and produce identical workloads. The difference is only in how the SLURM context is constructed:


| Aspect                                   | `primus-cli slurm srun -- direct --` (recommended)                                                                                                             | Bare `srun ... primus-cli direct --`                                                                                                |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `MASTER_ADDR` resolution                 | Resolved **once** on the launching node via `scontrol show hostnames "$SLURM_NODELIST" | head -n1`, then propagated to every rank via `--env MASTER_ADDR=...`. | Each rank re-derives it inside `primus-cli-direct.sh` STEP 4.7 from `SLURM_`* (same result, more `scontrol` calls).                 |
| `NNODES` / `NODE_RANK` / `GPUS_PER_NODE` | Set explicitly by `slurm-entry.sh` via `--env`.                                                                                                                | Derived from `SLURM_NNODES` / `SLURM_NODEID` / `SLURM_PROCID` inside `direct.sh`.                                                   |
| `slurm.*` config defaults                | Honored (partition, time, ntasks-per-node, etc. from the active YAML).                                                                                         | Not consulted — you pass every flag explicitly to `srun`.                                                                           |
| Default wall-time                        | `-t 4:00:00` is auto-added if you don't pass `--time`.                                                                                                         | None — `srun` uses the cluster default (may reject the job).                                                                        |
| `direct` keyword                         | **Required**: `primus-cli slurm srun ... -- direct -- <cmd>`. Without `direct`, the wrapper routes through the **container** path.                             | N/A — there's only one path.                                                                                                        |
| `--ntasks-per-node=1`                    | **Not auto-added**. Pass it on the CLI (before the first `--`) or set it in the `slurm.`* config.                                                              | **Not auto-added**. Pass it as an `srun` flag.                                                                                      |
| Best for                                 | Production / repeatable runs. Same pattern as `train` / `benchmark` / `node_smoke`.                                                                            | Ad-hoc runs where you want to compose with arbitrary `srun` flags (`--nodelist=$(...)`, `--exclude=...` from a runtime file, etc.). |


For the rest of this doc the examples use bare `srun` for brevity, but every example also works with the wrapper form by substituting `srun <flags> runner/primus-cli direct --` → `runner/primus-cli slurm srun <flags> -- direct --`.

### Key `srun` flags


| Flag                  | Why it's necessary                                                                                                                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-c 128`              | Allocate all CPU cores per task. Without this, SLURM may default to 1 core, which starves the RCCL network proxy threads and can cause >30× slowdown on perf tests. Set this to your node's core count. |
| `--gpus-per-node=8`   | Grants GPU device access (`/dev/kfd`, `/dev/dri`). Required for non-container execution.                                                                                                                |
| `--ntasks-per-node=1` | One launcher invocation per node; `primus-cli direct` then spawns 8 workers per node via `torchrun`.                                                                                                    |
| `-t 00:45:00`         | Wall-clock limit. Full perf tests on 8N usually finish well under 10 min.                                                                                                                               |


> Tip — check core count: `srun -N 1 --gpus-per-node=8 bash -c 'nproc'`

---

## 4. Cluster-specific NCCL configuration

`primus-cli direct` sources `runner/helpers/envs/base_env.sh`, which sets sensible defaults for `NCCL_`* and auto-detects `NCCL_IB_HCA` / `NCCL_SOCKET_IFNAME`. Pre-exported values from your shell take precedence, so the standard pattern is:

```bash
export VAR=value
runner/primus-cli direct -- preflight ...
```

### Broadcom NICs (no AINIC)

Most clusters fall here. The defaults from `base_env.sh` are usually fine, but the two values most commonly worth overriding are:

```bash
export NCCL_CROSS_NIC=1     # default in base_env.sh is 0
export NCCL_PXN_DISABLE=0   # default in base_env.sh is 1

srun -t 00:45:00 -N 4 -c 128 --gpus-per-node=8 --nodelist <nodes> \
    --ntasks-per-node=1 \
    runner/primus-cli direct -- preflight --perf-test
```

### Pensando Pollara (AINIC) RDMA

```bash
export USING_AINIC=1
export NCCL_IB_GID_INDEX=1   # AINIC uses index 1 (default in base_env.sh is 3)
export NCCL_PXN_DISABLE=0

srun -t 00:45:00 -N 4 -c 128 --gpus-per-node=8 --nodelist <nodes> \
    --ntasks-per-node=1 \
    runner/primus-cli direct -- preflight
```

> `primus-cli direct` *does* accept `--env KEY=VALUE` on its own command line (placed before `--`), in addition to the conventional `export`/`srun --export=` approaches.

---

## 5. Launcher flags vs. preflight flags

Anything you place **after** the `--` separator is forwarded verbatim to the `preflight` Python tool. The launcher (`primus-cli-direct.sh`) consumes a small set of flags **before** `--`. The one most users care about is `--silent`.

### Launcher-only flags (before `--`)


| Flag              | Effect                                                                                                                                                                                                                                                                                                                                                                      |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--silent`        | Back-pocket knob: redirect the launcher's and the Python tool's `stdout` to `/dev/null`. Launcher errors (`LOG_ERROR` / `LOG_WARN`, written to `stderr`) are preserved so real failures still surface; the log file under `logs/` captures everything. Exit code is propagated unchanged. **Not recommended** for normal use — you lose live progress; prefer the log file. |
| `--debug`         | Verbose launcher logging (`PRIMUS_LOG_LEVEL=DEBUG`). Forwarded to the Python tool as `--debug` too.                                                                                                                                                                                                                                                                         |
| `--dry-run`       | Print the resolved configuration and final `torchrun` / `python3` command without executing.                                                                                                                                                                                                                                                                                |
| `--single`        | Force `python3` instead of `torchrun`. `node_smoke` auto-selects this; for `preflight` you usually want the default (`torchrun`).                                                                                                                                                                                                                                           |
| `--env KEY=VALUE` | Inject an env var into the Python process (in addition to anything `export`-ed in the shell).                                                                                                                                                                                                                                                                               |
| `--log_file PATH` | Redirect the captured tee log to a specific path (default: `logs/log_<timestamp>.txt`).                                                                                                                                                                                                                                                                                     |


See `runner/primus-cli direct --help` for the full set.

### Forwarded `preflight` flags (after `--`, most common)

See [Preflight](./preflight.md) for the full list. The most common are:

- Mode selection: `--host`, `--gpu`, `--network`, `--perf-test`, `--tests`, `--quick`
- Test tuning: `--comm-sizes-mb`, `--intra-comm-sizes-mb`, `--inter-comm-sizes-mb`, `--intra-group-sizes`, `--inter-group-sizes`, `--ring-p2p-sizes-mb`
- Reporting: `--dump-path`, `--report-file-name`, `--disable-pdf`, `--plot`
- Reliability: `--comm-cleanup-delay-sec`, `--dist-timeout-sec`

If you do not pass `--report-file-name`, `preflight` auto-generates a unique one of the form:

```
preflight-${NNODES}N-YYYYMMDD-HHMMSS
```

This guarantees that each run lands in its own files and prevents stale leftovers from earlier runs from being mistaken for fresh output. The auto-name logic now lives in the Python tool itself, so every call site (host `srun ... primus-cli direct`, `primus-cli slurm ... -- direct`, `primus-cli slurm ... -- container`) gets the same fresh name.

### Examples

The examples below all assume one of the two equivalent shell-prefix conventions. Pick whichever matches your habits — every example block in this section works with either definition:

```bash
export VENV_ACTIVATE=~/envs/preflight/.venv/bin/activate

# Recommended: through the primus-cli SLURM wrapper. Auto-resolves
# MASTER_ADDR/NNODES/NODE_RANK once on the launching node and propagates
# them via --env; honors slurm.* config defaults.
SRUN="runner/primus-cli slurm srun -t 00:45:00 -N 4 -c 128 --gpus-per-node=8 --ntasks-per-node=1 --nodelist <nodes> --"
# Then in every example below, replace `$SRUN runner/primus-cli direct --`
# with just `$SRUN direct --`. (The wrapper expects the entry-mode keyword
# `direct` as the first token after the inner `--`.)

# Equivalent: bare srun. NNODES/NODE_RANK/MASTER_ADDR get derived inside
# primus-cli-direct.sh's STEP 4.7 directly from SLURM_*; same net effect.
SRUN="srun -t 00:45:00 -N 4 -c 128 --gpus-per-node=8 --ntasks-per-node=1 --nodelist <nodes>"
```

The examples in this section use the **bare-srun** form below for brevity (since `$SRUN runner/primus-cli direct -- preflight` reads naturally as one command line). To use the wrapper form instead, substitute `$SRUN runner/primus-cli direct --` → `$SRUN direct --` after exporting `SRUN` to the wrapper variant.

#### A. Mode selection

```bash
# Default: info report + every perf test
$SRUN runner/primus-cli direct -- preflight

# Info-only (fast, no torch.distributed rendezvous)
$SRUN runner/primus-cli direct -- preflight --host --gpu --network --disable-pdf

# Perf-only, every test
$SRUN runner/primus-cli direct -- preflight --perf-test

# Fast pre-launch sanity preset (gemm + intra-AR + inter-AR @ 64,1024 MB,
# full intra-node group, full N-node inter group, low warmup/iter)
$SRUN runner/primus-cli direct -- preflight --quick
```

> **Note**: Mixing perf-mode flags (`--perf-test` / `--tests` / `--quick`) with info selectors (`--host` / `--gpu` / `--network`) makes preflight drop the info selectors with a `WARN`. Run two invocations if you want both reports.

#### B. Test selection (`--tests`)

```bash
# Only GEMM
$SRUN runner/primus-cli direct -- preflight --tests gemm

# Only the inter-node bandwidth tests
$SRUN runner/primus-cli direct -- preflight --tests inter-allreduce,inter-alltoall

# Only the inter-node ring P2P
$SRUN runner/primus-cli direct -- preflight --tests inter-ring-p2p

# Combine: GEMM + inter-AR with overridden sizes/groups
$SRUN runner/primus-cli direct -- preflight \
    --tests gemm,inter-allreduce \
    --comm-sizes-mb 64,1024 \
    --inter-group-sizes all
```

Valid `--tests` tokens: `gemm`, `intra-allreduce`, `intra-alltoall`, `inter-allreduce`, `inter-alltoall`, `inter-p2p`, `inter-ring-p2p`, `all`. Unknown tokens fail fast (before NCCL init).

#### C. Message sizes

```bash
# One CSV applied to both intra and inter
$SRUN runner/primus-cli direct -- preflight --tests intra-allreduce,inter-allreduce \
    --comm-sizes-mb 8,128

# Different sizes for intra vs inter (override wins over --comm-sizes-mb)
$SRUN runner/primus-cli direct -- preflight --tests intra-allreduce,inter-allreduce \
    --comm-sizes-mb 8,128 --intra-comm-sizes-mb 4,32

# Inter-only override (also covers inter-p2p when enabled)
$SRUN runner/primus-cli direct -- preflight --tests inter-allreduce,inter-p2p \
    --comm-sizes-mb 8,128 --inter-comm-sizes-mb 16,512
```

#### D. Group sizes

```bash
# Custom intra-node group sizes (each must divide LOCAL_WORLD_SIZE)
$SRUN runner/primus-cli direct -- preflight \
    --tests intra-allreduce \
    --intra-group-sizes 4,8

# Custom inter-node groups: 2-node pairs and the full N-node group
$SRUN runner/primus-cli direct -- preflight \
    --tests inter-allreduce \
    --inter-group-sizes 2,all
```

> Note: for `inter-alltoall` only, every requested per-group node count is internally clamped to **16** (real-world MoE training rarely dispatches across more nodes; see `[preflight.md` §5.2](./preflight.md#52-group-sizes) for the rationale). The other inter-node tests use the requested sizes unchanged. So on a 128-node cluster, `--tests inter-alltoall --inter-group-sizes all` actually runs at 16-node sub-groups, while `--tests inter-allreduce --inter-group-sizes all` runs at 128 nodes as written.

#### E. Ring P2P sizes

```bash
$SRUN runner/primus-cli direct -- preflight \
    --tests inter-ring-p2p \
    --ring-p2p-sizes-mb 5,20,80
```

#### F. Plotting

```bash
# Generate per-test bandwidth bar charts under <dump-path>/<test>/*.png
$SRUN runner/primus-cli direct -- preflight \
    --tests intra-allreduce,inter-allreduce --plot
```

#### G. Reliability knobs

```bash
# Bump the per-phase cleanup delay. Default 2.0 is sufficient at every
# cluster size for the comm shapes preflight exercises (inter-alltoall
# is internally capped at 16 nodes; see preflight.md §5.2). Only bump
# this on very flaky networks or unusual kernel TIME_WAIT settings.
$SRUN runner/primus-cli direct -- preflight --quick --comm-cleanup-delay-sec 5

# Fail fast if torch.distributed rendezvous can't complete in 30s
$SRUN runner/primus-cli direct -- preflight --perf-test --dist-timeout-sec 30
```

> Operating clusters at ≥ 128 nodes? See `[preflight.md` §7](./preflight.md#7-running-on-very-large-clusters--64-nodes) for the recommended OS-level tunings (`tcp_tw_reuse`, wider `ip_local_port_range`) and per-test invocation patterns. With the §5.2 inter-alltoall cap in place, a default invocation is safe at every cluster size; the §7.2 sysctls remain best-practice for any RDMA workload.

#### H. Reporting & output layout

```bash
# Quick info-only check on 4 nodes, no PDF
$SRUN runner/primus-cli direct -- preflight --host --gpu --network --disable-pdf \
    --report-file-name info-4N

# Perf test only, silenced (CI-friendly), explicit name. Note that --silent
# is consumed by primus-cli-direct.sh and must appear BEFORE the `--`
# separator; everything after `--` is forwarded to the preflight Python tool.
$SRUN runner/primus-cli direct --silent -- preflight --perf-test \
    --report-file-name nightly-4N-perf

# Archive each run under its own directory
$SRUN runner/primus-cli direct -- preflight --quick \
    --dump-path /shared/preflight-archive/$(date +%Y%m%d-%H%M%S)
```

#### I. Backward-compat aliases

These still work and are equivalent to flags above. Use them only when retrofitting older scripts.

```bash
# Same as --host --gpu --network
$SRUN runner/primus-cli direct -- preflight --check-host --check-gpu --check-network

# Same as --inter-group-sizes all AND drops inter-p2p
$SRUN runner/primus-cli direct -- preflight --perf-test --no-split-nodes-subgroup
```

#### J. Combined "production-ready" pre-launch screen

```bash
# 1) Smoke first to prune broken nodes (note: --silent goes BEFORE `--`)
srun -N "$SLURM_NNODES" --ntasks-per-node=1 \
    runner/primus-cli direct --silent -- node_smoke --tier2-perf

# 2) Quick perf sanity on the survivors
srun -N <good-nnodes> -c 128 --gpus-per-node=8 --ntasks-per-node=1 \
    --exclude=$(paste -sd, output/preflight/failing_nodes.txt) \
    runner/primus-cli direct --silent -- preflight --quick \
        --comm-cleanup-delay-sec 5 --dist-timeout-sec 60 \
        --report-file-name screen-$(date +%Y%m%d-%H%M%S)
```

---

## 6. Outputs

Reports are written to `--dump-path` (default: `output/preflight/`), with the basename from `--report-file-name` and a `_perf` suffix for performance reports:


| File              | Produced by                                     | Notes                     |
| ----------------- | ----------------------------------------------- | ------------------------- |
| `<name>.md`       | `--host --gpu --network` (or default selection) | Info report               |
| `<name>.pdf`      | same, unless `--disable-pdf`                    | Info report PDF           |
| `<name>_perf.md`  | `--perf-test`                                   | Perf report (GEMM + comm) |
| `<name>_perf.pdf` | same, unless `--disable-pdf`                    | Perf report PDF           |


Only **rank 0** writes the report. After preflight completes, the Python tool prints the absolute path of every report file it produced to stdout. Under `--silent` these prints go to `/dev/null` along with everything else (one of the trade-offs of using `--silent`); without `--silent` the announcement is visible live. Sample output:

```
[Primus:Preflight] Report: /home/.../Primus/output/preflight/preflight-4N-20260428-201925.md
[Primus:Preflight] Report: /home/.../Primus/output/preflight/preflight-4N-20260428-201925_perf.md
```

---

## 7. Environment variable reference

Variables read by `primus-cli direct` itself:


| Variable        | Required | Default                                                                      | Purpose                                                                                                                |
| --------------- | -------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `VENV_ACTIVATE` | no       | —                                                                            | Path to the venv `bin/activate` script. Unset = no-op (use system / container Python). Set + missing file = fail-fast. |
| `NNODES`        | no       | `1` (or auto-derived from `SLURM_NNODES` / `SLURM_JOB_NUM_NODES`)            | Number of nodes. Pre-exported always wins.                                                                             |
| `NODE_RANK`     | no       | `0` (or auto-derived from `SLURM_NODEID` / `SLURM_PROCID`)                   | This node's rank. Pre-exported always wins.                                                                            |
| `GPUS_PER_NODE` | no       | `8`                                                                          | GPUs per node                                                                                                          |
| `MASTER_ADDR`   | no       | `localhost` (or first host from `scontrol show hostnames "$SLURM_NODELIST"`) | Rendezvous host. Pre-exported always wins.                                                                             |
| `MASTER_PORT`   | no       | `1234`                                                                       | Rendezvous port                                                                                                        |


Variables consumed downstream by `primus-cli direct` / `base_env.sh` (set them via `export`):


| Variable             | Default in `base_env.sh` | When to override                                  |
| -------------------- | ------------------------ | ------------------------------------------------- |
| `NCCL_SOCKET_IFNAME` | auto-detected            | Force a specific Ethernet interface for bootstrap |
| `NCCL_IB_HCA`        | auto-detected            | Force specific RDMA HCAs                          |
| `NCCL_IB_GID_INDEX`  | `3`                      | `1` on AINIC clusters                             |
| `NCCL_CROSS_NIC`     | `0`                      | `1` for multi-rail IB fabrics                     |
| `NCCL_PXN_DISABLE`   | `1`                      | `0` to enable PXN multi-hop NIC sharing           |
| `USING_AINIC`        | unset                    | `1` on Pensando Pollara clusters                  |
| `NCCL_DEBUG`         | unset                    | `INFO` for verbose NCCL logging                   |


---

## 8. Troubleshooting

### `[ERROR] [direct] VENV_ACTIVATE is set but file does not exist: ...`

`VENV_ACTIVATE` was set in the environment but the path it points at doesn't exist on this node. This is a fail-fast guard to prevent a silent fallback to system Python (which usually has the wrong `torch` / no ROCm). Either fix the path:

```bash
export VENV_ACTIVATE=~/envs/preflight/.venv/bin/activate
```

… or unset it to fall back to the container / system Python:

```bash
unset VENV_ACTIVATE
```

If the path looks right but the file still appears missing, confirm the venv lives on a filesystem visible from the node SLURM scheduled you onto.

### `[Primus:Preflight] FAIL: No GPUs detected`

The Python process inside the venv can't find ROCm. Diagnose with:

```bash
srun --nodes=1 --nodelist=<node> bash -c '
echo "=== PATH ==="; echo $PATH
echo "=== LD_LIBRARY_PATH ==="; echo $LD_LIBRARY_PATH
echo "=== rocm-smi ==="; rocm-smi --showid 2>&1
echo "=== Python torch check ==="
source ~/envs/preflight/.venv/bin/activate
python3 -c "import torch; print(\"hip:\", torch.version.hip); print(\"available:\", torch.cuda.is_available()); print(\"count:\", torch.cuda.device_count())"
'
```

If `LD_LIBRARY_PATH` is empty, set it explicitly:

```bash
export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}
```

### Report announcement points at stale files

This shouldn't happen with the current Python tool — the auto-generated unique report name (`preflight-${NNODES}N-<timestamp>`) ensures every run gets a fresh path. If you explicitly pass `--report-file-name X`, you're responsible for choosing a name that doesn't collide with prior runs.

### Slow perf tests (~30× expected)

Almost always a symptom of insufficient CPU cores. Pass `-c <cores-per-node>` to `srun` so RCCL's network proxy threads have CPU to spawn on. Verify with `srun -N 1 --gpus-per-node=8 bash -c 'nproc'`.

### Using `conda` instead of venv

`primus-cli direct` does `source "$VENV_ACTIVATE"`, which works for venv/uv but not directly for conda. Two options:

1. Create a venv inside the conda env and point `VENV_ACTIVATE` at that venv's activate script.
2. Write a small shim activate script (e.g. `~/envs/conda-shim.sh`) that activates conda and the desired env, then point `VENV_ACTIVATE` at it:
  ```bash
   # ~/envs/conda-shim.sh
   source "$HOME/miniconda3/etc/profile.d/conda.sh"
   conda activate <env_name>
  ```

### "Address already in use" during perf tests

This error occurs when peak simultaneous ESTAB sockets per node during an `ncclCommInit` exhausts the kernel ephemeral-port pool, so the next outgoing `bind()` walks the entire range without finding an allocatable port. (Despite the name and the `TIME_WAIT` framing in the kernel docs, accumulated `TIME_WAIT` count does *not* gate this for NCCL inter-node OOB — see `[preflight.md` §7.1](./preflight.md#71-why-address-already-in-use-used-to-surface-at-scale) for the mechanism and the empirical evidence.)

Preflight has two complementary defenses:

1. The **inter-node alltoall sub-group is internally capped at 16 nodes** (see `[preflight.md` §5.2](./preflight.md#52-group-sizes)) — the only test that, at large scale, can push peak ESTAB anywhere near the per-node ephemeral pool. The cap eliminates this failure mode by construction.
2. A **global barrier + `--comm-cleanup-delay-sec` sleep** (default 2 s) is inserted after every comm destroy, primarily for cross-rank synchronization across the destroy → setup transition.

If you still see `Address already in use` (e.g. on a network with an unusually narrow ephemeral-port range), the directly relevant **OS-level tuning** is widening that range — best-practice for any RDMA host:

```bash
# Widen the ephemeral port range from ~28k to ~64k. This is the only
# OS knob that directly addresses the binding constraint.
sudo sysctl -w net.ipv4.ip_local_port_range="1024 65535"

# General hygiene for hosts running mixed RDMA + repeated outgoing
# TCP workloads (NCCL inter-node OOB by itself doesn't benefit from
# this -- see preflight.md §7.2 for why).
sudo sysctl -w net.ipv4.tcp_tw_reuse=1
```

As a fallback, raise the per-phase delay:

```bash
# Bump the per-phase delay (default 2 s) on a particularly stressed
# network. Rarely needed in practice with the §5.2 alltoall cap.
runner/primus-cli direct -- preflight --comm-cleanup-delay-sec 5
```

See `[preflight.md` §7](./preflight.md#7-running-on-very-large-clusters--64-nodes) for the full explanation, persistence, and recommended large-cluster invocation patterns (split tests into separate runs, etc.).

If the error occurs at `init_process_group` (before tests even start), it typically means a previous job left port 29500 in `TIME_WAIT`. Either wait ~60 s or use a different port:

```bash
export MASTER_PORT=29501
```

### Capturing full output

The launcher already writes a complete log to `logs/log_<timestamp>.txt` (configurable via `--log_file PATH`), even under `--silent`. If you also want a copy at the call site, redirect there:

```bash
srun ... runner/primus-cli direct -- preflight --perf-test \
    2>&1 | tee preflight-$(date +%Y%m%d-%H%M%S).log
```

---

## 9. Automated node bisection (finding the bad node in an NCCL hang)

When a cluster-wide preflight run hangs or fails, use
`[tools/preflight_bisect/bisect.py](../tools/preflight_bisect/bisect.py)` to
run `preflight --perf-test` on smaller Slurm node subsets until suspect nodes
are isolated.

### Prerequisites

1. Working non-container preflight setup from the sections above, with
  `VENV_ACTIVATE` exported from a shared filesystem path.
2. Run from the SLURM login/head node, where both `scontrol` and `srun` are
  available.
3. Run from inside a Slurm allocation, or provide a Slurm nodelist explicitly.

### Example from inside an allocation

```bash
export VENV_ACTIVATE=~/envs/preflight/.venv/bin/activate
mkdir -p output

python tools/preflight_bisect/bisect.py \
    --nodelist "$SLURM_NODELIST" \
    --output-dir "output/bisect-$(date +%Y%m%d-%H%M%S)" \
    --trial-timeout-sec 600 \
    --slurm-time 00:15:00 \
    --preflight-env USING_AINIC=1 \
    --preflight-env NCCL_IB_GID_INDEX=1 \
    --preflight-env NCCL_CROSS_NIC=1 \
    --preflight-env NCCL_PXN_DISABLE=0 \
    2>&1 | tee output/bisect-latest.log
```

Adjust the `--preflight-env` lines to match your cluster. Per-trial logs and a
final `summary.txt` are written under `--output-dir`.

> Note: Set `--trial-timeout-sec` high enough for a healthy subset to finish.
> Too small a timeout can turn slow-but-good trials into false failures, causing
> the bisection to explore extra paths.
>
> Note: `--preflight-env KEY=VALUE` values are concatenated into a single
> `srun --export=ALL,...` argument, so values must not contain commas or
> whitespace. Keep comma-containing values as normal exported environment
> variables.

---

## 10. See also

- [Preflight](./preflight.md) — full reference for the `preflight` subcommand and its flags
- [CLI User Guide](./cli/PRIMUS-CLI-GUIDE.md) — container-based and `primus-cli slurm` workflows
- `[runner/primus-cli-direct.sh](../runner/primus-cli-direct.sh)` — the direct launcher itself (`primus-cli direct` dispatches here)
- `[primus/tools/preflight/](../primus/tools/preflight/)` — preflight implementation
- `[tools/preflight_bisect/bisect.py](../tools/preflight_bisect/bisect.py)` — bisect wrapper for narrowing down failing nodes in multi-node preflight runs
