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

```bash
# Use a shared file system accessible from all nodes (e.g. NFS home, /shared)
mkdir -p ~/envs/preflight
cd ~/envs/preflight

uv venv --python 3.12
source .venv/bin/activate

# install torch for rocm
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1

# install other packages
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

## Automated node bisection

The bisection tool now targets the new wrapper. See
[preflight-direct.md](./preflight-direct.md#9-automated-node-bisection-finding-the-bad-node-in-an-nccl-hang).
