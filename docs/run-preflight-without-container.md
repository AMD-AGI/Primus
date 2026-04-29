# Run Primus preflight without docker container

## clone the Primus repository

```bash
git clone --recurse-submodules https://github.com/AMD-AIG-AIMA/Primus.git
cd Primus
git checkout dev/fuyuajin/preflight-without-container
git submodule update --init --recursive
```

## Create python environment on a shared file system

### Required Python Packages

Based on the imports across all files in primus/tools/preflight/, here are the third-party (non-stdlib) packages needed:

| Package    | Used in                                                        | Purpose                                           |
|------------|----------------------------------------------------------------|---------------------------------------------------|
| torch      | Most files (square_gemm.py, intra_node_comm.py, inter_node_comm.py, etc.) | GPU compute, distributed communication (torch.distributed), profiler |
| matplotlib | square_gemm.py, intra_node_comm.py, inter_node_comm.py, inter_node_comm_p2p.py | Plotting benchmark results (charts)               |
| markdown2  | utility.py                                                     | Converting the Markdown report to HTML            |
| weasyprint | utility.py                                                     | Converting HTML to PDF for the final report       |

There is a list of package in the project's top-level `requirements.txt`.

### Create virtual environment and install packages

The environment is shared between all nodes in the cluster. So we need to create a virtual environment and install packages on a shared file system. Python venv or uv is recommended to create a virtual environment and install packages. Here is an example of creating a virtual environment and installing packages using uv:

#### Basic installation

```bash
mkdir -p ~/envs/preflight
cd ~/envs/preflight

uv venv --python 3.12
source .venv/bin/activate

uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1
```

#### Full installation (for plotting and PDF reports)

```bash
cd /path/to/Primus
uv pip install -r requirements.txt
```

### System level package dependencies

Preflight will run performance tests for the intra-node and inter-node communication. The packages for networking and communication are required. These should be already installed on the cluster.


## Run preflight on the cluster
### With cluster that does not use Pensando Pollara (AINIC) for RDMA
If you're using Broadcom NICs, you can use a command like this
```bash
# set the environment variable VENV_PATH to the path of the virtual environment activate script
export VENV_PATH=~/envs/preflight/.venv/bin/activate

# run preflight with slurm
# it will collect basic information and performance tests on the cluster.
srun -t 00:45:00 -N 4 -c 128 --gpus-per-node=8 --nodelist <nodes> \
  runner/primus-cli-direct-preflight.sh \
  --env NCCL_CROSS_NIC=1 --env NCCL_PXN_DISABLE=0 \
  -- preflight --perf-test --report-file-name preflight-report-4N
```

### With cluster that does use Pensando Pollara (AINIC) for RDMA
```bash
# set the environment variable VENV_PATH to the path of the virtual environment activate script
export VENV_PATH=~/envs/preflight/.venv/bin/activate

# run preflight with slurm
# it will collect basic information and performance tests on the cluster.
# note that these are setting cluster-specific environment variables for a cluster that uses AINIC.
srun -t 00:45:00 -N 4 -c 128 --gpus-per-node=8 --nodelist <nodes> \
  runner/primus-cli-direct-preflight.sh \
  --env USING_AINIC=1 --env NCCL_IB_GID_INDEX=1 --env NCCL_PXN_DISABLE=0 \
  -- preflight --report-file-name preflight-report-4N
```

### Key srun flags explained

| Flag | Why it is necessary |
|------|-----|
| `-c 128` | Allocates all CPU cores per task — without this, Slurm may default to 1 core, starving RCCL network proxy threads and causing 35x slowdown |
| `--gpus-per-node=8` | Grants GPU device access (`/dev/kfd`, `/dev/dri`) — required for non-container mode |
| `--env NCCL_CROSS_NIC=1` | Enables cross-NIC traffic for multi-rail IB fabrics — the runner default (`=0`) restricts to single-NIC paths |
| `--env NCCL_PXN_DISABLE=0` | Enables PXN (peer exchange over NIC) for multi-hop NIC sharing |

> **Note:** Set `-c` to the number of CPU cores per node on your cluster (e.g. `-c 128` for 128-core nodes). You can check with: `srun -N 1 --gpus-per-node=8 bash -c 'nproc'`

## Troubleshooting
### Using other virtual environment tools
Currently, our script uses this pattern `source <env_name>/bin/activate` to activate the virtual environment across all nodes. Specifically it picks up the `VENV_PATH` to run `source "$VENV_PATH"`.

However, if you use `conda`, the command is `conda activate <env_name>`. As a result, you may need to replace this [line](https://github.com/AMD-AGI/Primus/blob/dev/fuyuajin/preflight-without-container/runner/primus-cli-direct-preflight.sh#L375) to have something like this
```
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate py_3.11
```
instead of 
```
source "$VENV_PATH"
```
NOTE: you may still need to export `VENV_PATH` (by running `export VENV_PATH=conda`) so the script can enter the `if` statement in the conditional to be able to run `conda activate py_3.11`.

### Can't detect GPUs
You may see
```bash
[Primus:Preflight] FAIL: No GPUs detected (torch.cuda.is_available/device_count)
```
Run
```bash
srun --nodes=1 --nodelist=<nodes> bash -c '
echo "=== PATH ==="
echo $PATH
echo "=== LD_LIBRARY_PATH ==="
echo $LD_LIBRARY_PATH
echo "=== rocm-smi ==="
rocm-smi --showid 2>&1
echo "=== Python torch check ==="
source ~/envs/preflight/.venv/bin/activate
python3 -c "import torch; print(\"hip:\", torch.version.hip); print(\"available:\", torch.cuda.is_available()); print(\"count:\", torch.cuda.device_count())"
'
```
If `LD_LIBRARY_PATH` is empty, try setting it explicitly.
```bash
# Most common ROCm install location:
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```
If you still have issue, try
```bash
srun -t 00:30:00 -N 4 -c 128 --gpus-per-node=8 --nodelist <nodes> \
  runner/primus-cli-direct-preflight.sh \
  -- preflight --report-file-name preflight-report-4N \
  2>&1 | tee preflight-report-4N.log
```

## Automated node bisection (finding the bad node in an NCCL hang)

When a cluster-wide preflight run hangs or fails, you can automatically bisect
the nodelist to isolate the node(s) responsible instead of manually halving
the nodelist and re-running preflight.

Each trial runs `preflight --perf-test` on a subset of nodes via
[`runner/primus-cli-direct-preflight.sh`](../runner/primus-cli-direct-preflight.sh).
Subsets that pass are pruned immediately. Subsets that fail or time out are
split in half, and the two sibling subsets are launched in parallel by default
(up to 2 concurrent trials) until a singleton suspect is isolated.

### Prerequisites

1. Working non-container preflight setup from the sections above, with
   `VENV_PATH` exported from a shared filesystem path.
2. Run from the Slurm login/head node, where both `scontrol` and `srun` are
   available.
3. `scontrol show hostnames "<nodelist>"` must resolve the node expression you
   plan to bisect.

### Example template

```bash
PARTITION="<your-partition>"
OUTPUT_DIR="<your-output-dir>"
LOG_FILE="<your-log-file>"
NODELIST="${SLURM_NODELIST:-<your-nodelist>}"

python tools/preflight_bisect/bisect.py \
    --nodelist "$NODELIST" \
    --partition "$PARTITION" \
    --output-dir "$OUTPUT_DIR/bisect-$(date +%Y%m%d-%H%M%S)" \
    --trial-timeout-sec 600 \
    --slurm-time 00:15:00 \
    --preflight-env USING_AINIC=1 \
    --preflight-env NCCL_IB_GID_INDEX=1 \
    --preflight-env NCCL_CROSS_NIC=1 \
    --preflight-env NCCL_PXN_DISABLE=0 \
    2>&1 | tee "$LOG_FILE"
```

Set `PARTITION`, `OUTPUT_DIR`, and `LOG_FILE` for your environment. If you are
already inside a Slurm allocation, `SLURM_NODELIST` is used automatically;
otherwise set `NODELIST` explicitly. Adjust the `--preflight-env` lines to
match your cluster.

The script runs `preflight --perf-test` on shrinking subsets via `srun`. By
default, when a subset fails or hangs, its two sibling halves are launched in
parallel (up to 2 concurrent trials total), while passing subsets are pruned
immediately and are not split further. Any non-zero exit or trial timeout is
treated as a failure. Per-trial logs and a final `summary.txt` (including
`SUSPECT_NODES: ...`) are written under `--output-dir`.
