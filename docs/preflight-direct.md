# Run Preflight Without a Container

This guide explains how to run Primus's [`preflight`](./preflight.md) cluster-diagnostic tool **directly on the host** (no Docker / Podman), using the helper script:

```
runner/run_preflight_direct.sh
```

This script wraps `primus-cli direct -- preflight …` and fills in the distributed environment variables (`NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `GPUS_PER_NODE`) that `primus-cli direct` does **not** derive from SLURM on its own. It is the recommended entry point when:

- You're running on a SLURM cluster but cannot (or don't want to) use the container-based path.
- Your nodes share a Python virtual environment on a network-mounted filesystem.
- You want a single-node sanity check with no extra configuration.

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

### Option A — `uv` (recommended)

```bash
mkdir -p ~/envs/preflight
cd ~/envs/preflight

uv venv --python 3.12
source .venv/bin/activate

# Install ROCm-built PyTorch (pin to your ROCm version; rocm7.1 shown here)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1

# Install the rest of Primus's runtime requirements (matplotlib, markdown2, weasyprint, etc.)
cd /path/to/Primus
uv pip install -r requirements.txt
```

### Option B — `python -m venv`

```bash
mkdir -p ~/envs/preflight
python3.12 -m venv ~/envs/preflight/.venv
source ~/envs/preflight/.venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1
cd /path/to/Primus
pip install -r requirements.txt
```

### What gets installed

The preflight tool needs only a small subset of Primus's full dependency tree. The packages that matter are:

| Package      | Used in                                        | Purpose                                     |
| ------------ | ---------------------------------------------- | ------------------------------------------- |
| `torch`      | most files in `primus/tools/preflight/`        | GPU compute, `torch.distributed`, profiler  |
| `matplotlib` | `square_gemm.py`, `intra_node_comm.py`, …      | Plotting (`--plot`)                         |
| `markdown2`  | `utility.py`                                   | Markdown → HTML for the report              |
| `weasyprint` | `utility.py`                                   | HTML → PDF for the report                   |

If you skip `weasyprint`/`markdown2` you can still get the Markdown report; just pass `--disable-pdf` (see below).

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

```bash
export VENV_ACTIVATE=~/envs/preflight/.venv/bin/activate

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
| `-t 00:45:00`          | Wall-clock limit. Full perf tests on 8N usually finish well under 30 min.                                                       |

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

- Selection: `--host`, `--gpu`, `--network`, `--perf-test`
- Reporting: `--dump-path`, `--report-file-name`, `--disable-pdf`
- Perf-test extras: `--plot`

If you do not pass `--report-file-name`, the wrapper auto-generates a unique one of the form:

```
preflight-${NNODES}N-YYYYMMDD-HHMMSS
```

This guarantees that each run lands in its own files and prevents stale leftovers from earlier runs from being mistaken for fresh output.

### Examples

```bash
# Quick info-only check on 4 nodes, no PDF
srun -N 4 -c 128 --gpus-per-node=8 --ntasks-per-node=1 \
    runner/run_preflight_direct.sh --host --gpu --network --disable-pdf

# Perf test only, silenced (CI-friendly), explicit name
srun -N 4 -c 128 --gpus-per-node=8 --ntasks-per-node=1 \
    runner/run_preflight_direct.sh --silent --perf-test \
        --report-file-name nightly-4N-perf

# Custom output directory
srun -N 4 -c 128 --gpus-per-node=8 --ntasks-per-node=1 \
    runner/run_preflight_direct.sh --perf-test \
        --dump-path /shared/preflight-archive
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
