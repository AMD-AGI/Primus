# Run Preflight Without a Container

This guide explains how to run Primus's [`preflight`](./preflight.md) cluster-diagnostic tool **directly on the host** (no Docker / Podman), using the helper script:

```
runner/run_preflight_direct.sh
```

This script wraps `primus-cli direct -- preflight ...` and fills in the distributed environment variables (`NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `GPUS_PER_NODE`) that `primus-cli direct` does **not** derive from SLURM on its own. It is the recommended entry point when:

- You're running on a SLURM cluster but cannot (or don't want to) use the container-based path.
- Your nodes share a Python virtual environment on a network-mounted filesystem.
- You want a single-node sanity check with no extra configuration.

---

## 0. Which test should I run?

Primus ships **two** complementary cluster screens. Pick the right one — and ideally run them in this order.

| Aspect | `node-smoke` (start here) | `preflight` (this doc) |
|---|---|---|
| Purpose | "Which nodes are healthy enough to run anything?" | "What is the actual cross-node performance on the surviving nodes?" |
| Rendezvous | None — every node independent | Global `torch.distributed` rendezvous |
| Wall clock | ~30–60 s for 6 nodes (Tier 1+2) | A few minutes; scales with N for inter-node tests |
| Granularity | Per-node PASS/FAIL | Per-rank perf measurements |
| Safety | A stuck node cannot wedge its peers | A single hung NIC can stall the whole rendezvous |
| Output | Per-node JSON + cluster md + SLURM-ready `passing_nodes.txt` / `failing_nodes.txt` | Markdown + PDF perf report |
| Wrapper | `runner/run_node_smoke_direct.sh` | `runner/run_preflight_direct.sh` (this doc) |
| Quick-start guide | [`node-smoke-test-instruction.md`](./node-smoke-test-instruction.md) | This doc, §3+ |

### Recommended workflow

```bash
# 1) Prune broken nodes with node-smoke (fast, no rendezvous).
srun -N "$SLURM_NNODES" --ntasks-per-node=1 \
    bash runner/run_node_smoke_direct.sh --tier2-perf

# 2) Re-allocate excluding the bad nodes, and run preflight --quick
#    for a fast cross-node sanity check.
srun -N <good-nnodes> -c 128 --gpus-per-node=8 --ntasks-per-node=1 \
    --exclude=$(paste -sd, output/preflight/failing_nodes.txt) \
    runner/run_preflight_direct.sh --quick

# 3) Optional: full preflight on the same set if --quick numbers
#    look off, or if you want the full bandwidth matrix.
srun -N <good-nnodes> -c 128 --gpus-per-node=8 --ntasks-per-node=1 \
    --exclude=$(paste -sd, output/preflight/failing_nodes.txt) \
    runner/run_preflight_direct.sh
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

| Package | Required for | Skip when |
|---|---|---|
| `torch` (ROCm build) | Both tools — perf measurements (`torch.matmul`, `torch.distributed`, `torch.cuda.*`). | Never (mandatory). |
| `markdown2` | `preflight` PDF report only (Markdown → HTML). | You always pass `--disable-pdf`, or you only run `node-smoke` (which never produces PDFs). |
| `weasyprint` | `preflight` PDF report only (HTML → PDF). | Same as above. |
| `matplotlib` | `preflight --plot` only (per-test bandwidth bar charts). | You don't pass `--plot`. |

Everything else in the preflight / node-smoke code path is Python stdlib (`os`, `subprocess`, `socket`, `argparse`, `dataclasses`, `json`, `time`, ...) — no extra installs needed.

### Option A — `uv` (recommended), minimal install

```bash
mkdir -p ~/envs/preflight
cd ~/envs/preflight

uv venv --python 3.12
source .venv/bin/activate

# 1) ROCm-built PyTorch (pin to your ROCm version; rocm7.1 shown here)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1

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

pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1
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

| Invocation | `torch` | `markdown2` | `weasyprint` | `matplotlib` |
|---|---|---|---|---|
| `node-smoke` (any flags) | required | — | — | — |
| `preflight --host --gpu --network --disable-pdf` | required | — | — | — |
| `preflight --host --gpu --network` (with PDF) | required | required | required | — |
| `preflight --quick --disable-pdf` | required | — | — | — |
| `preflight --quick` (with PDF) | required | required | required | — |
| `preflight ... --plot` | required | required (unless `--disable-pdf`) | required (unless `--disable-pdf`) | required |


### Tell the script where the venv is

`run_preflight_direct.sh` requires the **`VENV_ACTIVATE`** environment variable to point at the venv's `bin/activate` script:

```bash
export VENV_ACTIVATE=~/envs/preflight/.venv/bin/activate
```

This is the only environment variable the wrapper insists on. Everything else has a sensible default.

---

## 3. Run preflight

### Single node (no SLURM)

```bash
export VENV_ACTIVATE=~/envs/preflight/.venv/bin/activate

# Info report only (fast)
runner/run_preflight_direct.sh --host --gpu --network

# Info + perf report
runner/run_preflight_direct.sh

# Perf report only
runner/run_preflight_direct.sh --perf-test
```

When SLURM is not detected the script defaults to `NNODES=1`, `NODE_RANK=0`, `MASTER_ADDR=localhost`. Any of those can be overridden by exporting them before calling the script.

### Multi-node via SLURM

The script auto-detects a SLURM allocation (via `SLURM_JOB_ID`) and derives all distributed variables from `SLURM_*` automatically:

| Wrapper variable | Derived from                                                |
| ---------------- | ----------------------------------------------------------- |
| `NNODES`         | `SLURM_NNODES` → `SLURM_JOB_NUM_NODES` → `NNODES` → `1`     |
| `NODE_RANK`      | `SLURM_NODEID` → `SLURM_PROCID` → `NODE_RANK` → `0`         |
| `MASTER_ADDR`    | First hostname from `scontrol show hostnames "$SLURM_NODELIST"` |
| `MASTER_PORT`    | `MASTER_PORT` → `1234`                                       |
| `GPUS_PER_NODE`  | `GPUS_PER_NODE` → `8`                                        |

Run it as a single task per node (the script invokes `torchrun` internally, which spawns one worker per GPU):

> **Verify NCCL / network env first.** The script sets sensible `NCCL_*` defaults via `base_env.sh`, but auto-detection can pick the wrong device on multi-NIC nodes. Always confirm `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`, `NCCL_SOCKET_IFNAME`, and `GLOO_SOCKET_IFNAME` (set to the same value as `NCCL_SOCKET_IFNAME`) are correct for your fabric, and `export` overrides before running. See [§4](#4-cluster-specific-nccl-configuration) for cluster-specific values.

```bash
export VENV_ACTIVATE=~/envs/preflight/.venv/bin/activate

# export NCCL_IB_HCA=rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eno0
# export GLOO_SOCKET_IFNAME=eno0

srun -t 00:45:00 -N 4 -c 128 --gpus-per-node=8 --nodelist <nodes> \
    --ntasks-per-node=1 \
    runner/run_preflight_direct.sh --perf-test
```

### Key `srun` flags

| Flag                   | Why it's necessary                                                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `-c 128`               | Allocate all CPU cores per task. Without this, SLURM may default to 1 core, which starves the RCCL network proxy threads and can cause >30× slowdown on perf tests. Set this to your node's core count. |
| `--gpus-per-node=8`    | Grants GPU device access (`/dev/kfd`, `/dev/dri`). Required for non-container execution.                                         |
| `--ntasks-per-node=1`  | One wrapper invocation per node; the wrapper spawns 8 workers per node via `torchrun`.                                          |
| `-t 00:45:00`          | Wall-clock limit. Full perf tests on 8N usually finish well under 10 min.                                                       |

> Tip — check core count: `srun -N 1 --gpus-per-node=8 bash -c 'nproc'`

---

## 4. Cluster-specific NCCL configuration

`primus-cli direct` (which this wrapper invokes) sources `runner/helpers/envs/base_env.sh`, which sets sensible defaults for `NCCL_*` and auto-detects `NCCL_IB_HCA` / `NCCL_SOCKET_IFNAME`. Pre-exported values from your shell take precedence, so the standard pattern is:

```bash
export VAR=value
runner/run_preflight_direct.sh ...
```

### Broadcom NICs (no AINIC)

Most clusters fall here. The defaults from `base_env.sh` are usually fine, but the two values most commonly worth overriding are:

```bash
export NCCL_CROSS_NIC=1     # default in base_env.sh is 0
export NCCL_PXN_DISABLE=0   # default in base_env.sh is 1

srun -t 00:45:00 -N 4 -c 128 --gpus-per-node=8 --nodelist <nodes> \
    --ntasks-per-node=1 \
    runner/run_preflight_direct.sh --perf-test
```

### Pensando Pollara (AINIC) RDMA

```bash
export USING_AINIC=1
export NCCL_IB_GID_INDEX=1   # AINIC uses index 1 (default in base_env.sh is 3)
export NCCL_PXN_DISABLE=0

srun -t 00:45:00 -N 4 -c 128 --gpus-per-node=8 --nodelist <nodes> \
    --ntasks-per-node=1 \
    runner/run_preflight_direct.sh
```

> **Note**: This wrapper does not accept `--env KEY=VALUE` on its own command line. Set environment variables with `export` before invoking the script, or pass them via `srun --export=` if you prefer SLURM-managed propagation. (The `primus-cli-direct-preflight.sh` variant does support `--env`; see the [reference guide](https://github.com/AMD-AGI/Primus/blob/dev/fuyuajin/preflight-without-container/docs/run-preflight-without-container.md) for that flow.)

---

## 5. Wrapper flags vs. preflight flags

Most arguments are forwarded verbatim to `preflight`; only one flag is consumed by the wrapper itself.

### Wrapper-only flags

| Flag       | Effect                                                                                                                                                          |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--silent` | Redirect the wrapper's and `primus-cli`'s `stdout` to `/dev/null`. `stderr` is preserved so real errors still surface, and the final report file path is still printed at the end (via a saved fd). Exit code is propagated unchanged. |

### Forwarded `preflight` flags (most common)

See [Preflight](./preflight.md) for the full list. The most common are:

- Mode selection: `--host`, `--gpu`, `--network`, `--perf-test`, `--tests`, `--quick`
- Test tuning: `--comm-sizes-mb`, `--intra-comm-sizes-mb`, `--inter-comm-sizes-mb`, `--intra-group-sizes`, `--inter-group-sizes`, `--ring-p2p-sizes-mb`
- Reporting: `--dump-path`, `--report-file-name`, `--disable-pdf`, `--plot`
- Reliability: `--comm-cleanup-delay-sec`, `--dist-timeout-sec`

If you do not pass `--report-file-name`, the wrapper auto-generates a unique one of the form:

```
preflight-${NNODES}N-YYYYMMDD-HHMMSS
```

This guarantees that each run lands in its own files and prevents stale leftovers from earlier runs from being mistaken for fresh output.

### Examples

The examples below all assume:

```bash
export VENV_ACTIVATE=~/envs/preflight/.venv/bin/activate
SRUN="srun -t 00:45:00 -N 4 -c 128 --gpus-per-node=8 --ntasks-per-node=1 --nodelist <nodes>"
```

#### A. Mode selection

```bash
# Default: info report + every perf test
$SRUN runner/run_preflight_direct.sh

# Info-only (fast, no torch.distributed rendezvous)
$SRUN runner/run_preflight_direct.sh --host --gpu --network --disable-pdf

# Perf-only, every test
$SRUN runner/run_preflight_direct.sh --perf-test

# Fast pre-launch sanity preset (gemm + intra-AR + inter-AR @ 64,1024 MB,
# full intra-node group, full N-node inter group, low warmup/iter)
$SRUN runner/run_preflight_direct.sh --quick
```

> **Note**: Mixing perf-mode flags (`--perf-test` / `--tests` / `--quick`) with info selectors (`--host` / `--gpu` / `--network`) makes preflight drop the info selectors with a `WARN`. Run two invocations if you want both reports.

#### B. Test selection (`--tests`)

```bash
# Only GEMM
$SRUN runner/run_preflight_direct.sh --tests gemm

# Only the inter-node bandwidth tests
$SRUN runner/run_preflight_direct.sh --tests inter-allreduce,inter-alltoall

# Only the inter-node ring P2P
$SRUN runner/run_preflight_direct.sh --tests inter-ring-p2p

# Combine: GEMM + inter-AR with overridden sizes/groups
$SRUN runner/run_preflight_direct.sh \
    --tests gemm,inter-allreduce \
    --comm-sizes-mb 64,1024 \
    --inter-group-sizes all
```

Valid `--tests` tokens: `gemm`, `intra-allreduce`, `intra-alltoall`, `inter-allreduce`, `inter-alltoall`, `inter-p2p`, `inter-ring-p2p`, `all`. Unknown tokens fail fast (before NCCL init).

#### C. Message sizes

```bash
# One CSV applied to both intra and inter
$SRUN runner/run_preflight_direct.sh --tests intra-allreduce,inter-allreduce \
    --comm-sizes-mb 8,128

# Different sizes for intra vs inter (override wins over --comm-sizes-mb)
$SRUN runner/run_preflight_direct.sh --tests intra-allreduce,inter-allreduce \
    --comm-sizes-mb 8,128 --intra-comm-sizes-mb 4,32

# Inter-only override (also covers inter-p2p when enabled)
$SRUN runner/run_preflight_direct.sh --tests inter-allreduce,inter-p2p \
    --comm-sizes-mb 8,128 --inter-comm-sizes-mb 16,512
```

#### D. Group sizes

```bash
# Custom intra-node group sizes (each must divide LOCAL_WORLD_SIZE)
$SRUN runner/run_preflight_direct.sh \
    --tests intra-allreduce \
    --intra-group-sizes 4,8

# Custom inter-node groups: 2-node pairs and the full N-node group
$SRUN runner/run_preflight_direct.sh \
    --tests inter-allreduce \
    --inter-group-sizes 2,all
```

#### E. Ring P2P sizes

```bash
$SRUN runner/run_preflight_direct.sh \
    --tests inter-ring-p2p \
    --ring-p2p-sizes-mb 5,20,80
```

#### F. Plotting

```bash
# Generate per-test bandwidth bar charts under <dump-path>/<test>/*.png
$SRUN runner/run_preflight_direct.sh \
    --tests intra-allreduce,inter-allreduce --plot
```

#### G. Reliability knobs

```bash
# Increase inter-phase cleanup delay (helpful at 128+ nodes / 1000+ ranks)
$SRUN runner/run_preflight_direct.sh --quick --comm-cleanup-delay-sec 5

# Fail fast if torch.distributed rendezvous can't complete in 30s
$SRUN runner/run_preflight_direct.sh --perf-test --dist-timeout-sec 30
```

#### H. Reporting & output layout

```bash
# Quick info-only check on 4 nodes, no PDF
$SRUN runner/run_preflight_direct.sh --host --gpu --network --disable-pdf \
    --report-file-name info-4N

# Perf test only, silenced (CI-friendly), explicit name
$SRUN runner/run_preflight_direct.sh --silent --perf-test \
    --report-file-name nightly-4N-perf

# Archive each run under its own directory
$SRUN runner/run_preflight_direct.sh --quick \
    --dump-path /shared/preflight-archive/$(date +%Y%m%d-%H%M%S)
```

#### I. Backward-compat aliases

These still work and are equivalent to flags above. Use them only when retrofitting older scripts.

```bash
# Same as --host --gpu --network
$SRUN runner/run_preflight_direct.sh --check-host --check-gpu --check-network

# Same as --inter-group-sizes all AND drops inter-p2p
$SRUN runner/run_preflight_direct.sh --perf-test --no-split-nodes-subgroup
```

#### J. Combined "production-ready" pre-launch screen

```bash
# 1) Smoke first to prune broken nodes
srun -N "$SLURM_NNODES" --ntasks-per-node=1 \
    bash runner/run_node_smoke_direct.sh --silent --tier2-perf

# 2) Quick perf sanity on the survivors
srun -N <good-nnodes> -c 128 --gpus-per-node=8 --ntasks-per-node=1 \
    --exclude=$(paste -sd, output/preflight/failing_nodes.txt) \
    runner/run_preflight_direct.sh --silent --quick \
        --comm-cleanup-delay-sec 5 --dist-timeout-sec 60 \
        --report-file-name screen-$(date +%Y%m%d-%H%M%S)
```

---

## 6. Outputs

Reports are written to `--dump-path` (default: `output/preflight/`), with the basename from `--report-file-name` and a `_perf` suffix for performance reports:

| File                                | Produced by                | Notes                          |
| ----------------------------------- | -------------------------- | ------------------------------ |
| `<name>.md`                         | `--host --gpu --network` (or default selection) | Info report                    |
| `<name>.pdf`                        | same, unless `--disable-pdf` | Info report PDF                |
| `<name>_perf.md`                    | `--perf-test`              | Perf report (GEMM + comm)      |
| `<name>_perf.pdf`                   | same, unless `--disable-pdf` | Perf report PDF                |

Only **rank 0** writes the report. After preflight completes, the wrapper prints the absolute path of every report file it produced — even under `--silent`. Sample output:

```
[run_preflight_direct] Report: /home/.../Primus/output/preflight/preflight-4N-20260428-201925.md
[run_preflight_direct] Report: /home/.../Primus/output/preflight/preflight-4N-20260428-201925_perf.md
```

---

## 7. Environment variable reference

Variables read by `run_preflight_direct.sh` itself:

| Variable          | Required | Default                  | Purpose                                         |
| ----------------- | -------- | ------------------------ | ----------------------------------------------- |
| `VENV_ACTIVATE`   | yes      | —                        | Path to the venv `bin/activate` script          |
| `PRIMUS_CLI`      | no       | `<repo>/runner/primus-cli` | Path to the `primus-cli` entry point            |
| `NNODES`          | no       | `1` (or from SLURM)      | Number of nodes                                 |
| `NODE_RANK`       | no       | `0` (or from SLURM)      | This node's rank                                |
| `GPUS_PER_NODE`   | no       | `8`                      | GPUs per node                                   |
| `MASTER_ADDR`     | no       | `localhost` (or from SLURM) | Rendezvous host                                 |
| `MASTER_PORT`     | no       | `1234`                   | Rendezvous port                                 |

Variables consumed downstream by `primus-cli direct` / `base_env.sh` (set them via `export`):

| Variable               | Default in `base_env.sh` | When to override                                  |
| ---------------------- | ------------------------ | ------------------------------------------------- |
| `NCCL_SOCKET_IFNAME`   | auto-detected            | Force a specific Ethernet interface for bootstrap |
| `NCCL_IB_HCA`          | auto-detected            | Force specific RDMA HCAs                          |
| `NCCL_IB_GID_INDEX`    | `3`                      | `1` on AINIC clusters                             |
| `NCCL_CROSS_NIC`       | `0`                      | `1` for multi-rail IB fabrics                     |
| `NCCL_PXN_DISABLE`     | `1`                      | `0` to enable PXN multi-hop NIC sharing           |
| `USING_AINIC`          | unset                    | `1` on Pensando Pollara clusters                  |
| `NCCL_DEBUG`           | unset                    | `INFO` for verbose NCCL logging                   |

---

## 8. Troubleshooting

### `[run_preflight_direct][ERROR] VENV_ACTIVATE is not set`

Export it before invoking the script:

```bash
export VENV_ACTIVATE=~/envs/preflight/.venv/bin/activate
```

If the path is correct but you're still getting "Virtualenv activate script not found", confirm the venv lives on a filesystem visible from the node SLURM scheduled you onto.

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

### Wrapper announces stale report files

This shouldn't happen with the current script — the auto-generated unique report name (`preflight-${NNODES}N-<timestamp>`) ensures every run gets a fresh path. If you explicitly pass `--report-file-name X`, you're responsible for choosing a name that doesn't collide with prior runs.

### Slow perf tests (~30× expected)

Almost always a symptom of insufficient CPU cores. Pass `-c <cores-per-node>` to `srun` so RCCL's network proxy threads have CPU to spawn on. Verify with `srun -N 1 --gpus-per-node=8 bash -c 'nproc'`.

### Using `conda` instead of venv

The script does `source "$VENV_ACTIVATE"`, which works for venv/uv but not directly for conda. Two options:

1. Create a venv inside the conda env and point `VENV_ACTIVATE` at that venv's activate script.
2. Edit the script to replace the `source "$VENV_ACTIVATE"` line with:

   ```bash
   source "$HOME/miniconda3/etc/profile.d/conda.sh"
   conda activate <env_name>
   ```

   You'll still need `VENV_ACTIVATE` set to anything non-empty so the existence check passes, or remove that guard.

### "Address already in use" during perf tests

This error occurs when NCCL/RCCL sockets are still in `TIME_WAIT` state from a recently destroyed process group. Preflight mitigates this with a barrier + sleep between test phases (`--comm-cleanup-delay-sec`, default 2s).

If the error still occurs at very large scale (128+ nodes / 1000+ ranks):

```bash
# Increase the inter-phase delay
runner/run_preflight_direct.sh --perf-test --comm-cleanup-delay-sec 5
```

If it occurs at `init_process_group` (before tests even start), it typically means a previous job left port 29500 in `TIME_WAIT`. Either wait ~60s or use a different port:

```bash
export MASTER_PORT=29501
```

### Capturing full output

`--silent` only suppresses interactive output; if you also want a persistent log file, redirect at the call site:

```bash
srun ... runner/run_preflight_direct.sh --perf-test \
    2>&1 | tee preflight-$(date +%Y%m%d-%H%M%S).log
```

---

## 9. See also

- [Preflight](./preflight.md) — full reference for the `preflight` subcommand and its flags
- [CLI User Guide](./cli/PRIMUS-CLI-GUIDE.md) — container-based and `primus-cli slurm` workflows
- [`runner/run_preflight_direct.sh`](../runner/run_preflight_direct.sh) — the wrapper itself
- [`primus/tools/preflight/`](../primus/tools/preflight/) — preflight implementation
- AMD-AGI reference guide: [run-preflight-without-container.md](https://github.com/AMD-AGI/Primus/blob/dev/fuyuajin/preflight-without-container/docs/run-preflight-without-container.md)
