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

There are a list of package in the project's top-level `requirements.txt`.

### Create virtual environment and install packages

The environment is shared between all nodes in the cluster. So we need to create a virtual environment and install packages on a shared file system. Python venv or uv is recommended to create a virtual environment and install packages. Here is an example of creating a virtual environment and installing packages using uv: 

```bash
# /mnt/vast is a shared file system. All nodes in the cluster can access this directory.
mkdir -p /mnt/vast/fuyuan/envs/test
cd /mnt/vast/fuyuan/envs/test

uv venv --python 3.12
source .venv/bin/activate

# install torch for rocm
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1

# install other packages
cd /mnt/vast/fuyuan/Primus
uv pip install -r requirements.txt
```

### System level package dependencies

Preflight will run performance tests for the intra-node and inter-node communication. The packages for networking and communication are required. These should be already installed on the cluster.


## Run preflight on the cluster

```bash
# set the environment variable VENV_PATH to the path of the virtual environment activate script
export VENV_PATH=/mnt/vast/fuyuan/envs/test/.venv/bin/activate

# set other environment variables for cluster
# these are for the vultr cluster. They are cluster specific.
export USING_AINIC=1
export NCCL_PXN_DISABLE=0
export NCCL_IB_GID_INDEX=1

# run preflight with slurm
# it will collect basic information and performance tests on the cluster.
srun -N 4 --nodelist chi2874,chi2875,chi2810,chi2812 runner/primus-cli-direct-preflight.sh -- preflight --report-file-name preflight-report-4N-0226
```

## Troubleshooting
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
source /mnt/vast/fuyuan/envs/test/.venv/bin/activate
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
srun -t 00:30:00 -p gpus -N 4 --gpus-per-node=8  --nodelist <nodes> runner/primus-cli-direct-preflight.sh -- preflight  --report-file-name preflight-report-4N 2>&1 | tee preflight_32N_gpu-per-node.log
```
