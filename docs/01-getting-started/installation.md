# Installation and setup

This guide covers supported platforms, prerequisites, and how to set up the training environment: **container (recommended)** and **bare metal**, plus **multi-node distributed training** (Slurm recommended).

---

## Supported platforms


| Requirement | Notes                                                                                                       |
| ----------- | ----------------------------------------------------------------------------------------------------------- |
| **OS**      | Linux (ROCm-supported distributions per AMD documentation).                                                 |
| **ROCm**    | **≥ 7.0** recommended.                                                                                      |
| **GPUs**    | AMD Instinct™ **MI300X**, **MI325X**, **MI355X** (or other ROCm-supported Instinct SKUs your site supports). |


---

## Prerequisites


| Prerequisite                                                  | Purpose                                       |
| ------------------------------------------------------------- | --------------------------------------------- |
| **AMD Instinct GPUs**                                         | Training and benchmarks execute on GPU.       |
| **ROCm drivers and user-space stack**                         | Required for HIP, RCCL, and ML frameworks.    |
| **Docker ≥ 24.0** (or Podman with compatible GPU passthrough) | Container mode and reproducible environments. |
| **git**                                                       | Clone the repository and submodules.          |


### Quick environment checks

```bash
rocm-smi
docker --version
```

`rocm-smi` should list your GPUs; `docker --version` should report **24.0** or newer.

---

## Container setup (recommended)

AMD publishes training Docker images monthly, providing a consistent, ready-to-run environment optimized for AMD GPUs. It is recommended to use the AMD-published training Docker images together with this Primus-LM repository to run your training jobs. The images support pre-training and post-training workflows with multiple backends including Megatron-LM, TorchTitan, and JAX MaxText, alongside ROCm-optimized components.

Check the AMD-published training Docker images here:

- For Megatron-LM and TorchTitan backends: [https://hub.docker.com/r/rocm/primus/tags](https://hub.docker.com/r/rocm/primus/tags)
- For MaxText backend: [https://hub.docker.com/r/rocm/jax-training/tags](https://hub.docker.com/r/rocm/jax-training/tags)

### 1. Pull the image

```bash
# For Megatron-LM and TorchTitan backends
docker pull rocm/primus:v26.4
# For MaxText backend
docker pull rocm/jax-training:maxtext-v26.4-jax0.9.1-te2.12.0
```

### 2. Clone the repository

Submodules are required for third-party backends and tools:

```bash
git clone --recurse-submodules https://github.com/AMD-AGI/Primus.git
cd Primus
# checkout the branch for the specific release
git checkout release/v26.4
git submodule update --init --recursive
```

### 3. Run a verification benchmark

From the repository root:

```bash
./primus-cli container --image rocm/primus:v26.4 -- \
  benchmark gemm --M 4096 --N 4096 --K 4096
```

A successful run validates the GPU stack and Primus CLI wiring without launching a full training job.

---

## Bare-metal (host) setup

> **The [container setup](#container-setup-recommended) above is strongly recommended.** The AMD-published training Docker image is the tested, reproducible, and best-supported path. Build the full stack on a bare-metal host only when containers are not an option (for example, due to policy or operational constraints).

### What to expect

Reproducing the training environment on the host means building the **same stack the Docker image ships**, mostly from source. This is a **long, build-heavy process** that requires:

- Several **source-built kernel libraries** (Flash Attention, TransformerEngine, AITER, Primus-Turbo, Grouped GEMM, causal-conv1d, Mamba) compiled against ROCm.
- A **machine with many CPU cores, ample RAM, and tens of GB of free disk**.
- A long time (expect a **multi-hour first build**).

### What a complete host environment needs

| Layer                   | What it provides                                                                 | How it is installed              |
| ----------------------- | -------------------------------------------------------------------------------- | -------------------------------- |
| Kernel/hardware         | AMD GPU driver (amdgpu KMD) and device access (`/dev/kfd`, `/dev/dri`)            | OS/administrator (root, one-time) |
| OS libraries            | Build toolchain and runtime libraries (`g++`, `git`, RDMA, hwloc, etc.)          | `apt` (root, one-time)           |
| ROCm user-space         | `rocm-sdk-devel` + device wheels—**no system-wide ROCm install required**       | `pip` (TheRock wheels, in `venv`) |
| Deep learning framework | ROCm-enabled PyTorch (`torch`, `torchvision`, `torchaudio`, `apex`)              | `pip` (TheRock wheels, in `venv`) |
| Accelerated kernels     | Flash Attention, TransformerEngine, AITER, Primus-Turbo, Grouped GEMM, Mamba     | build from source (in `venv`)    |
| Multi-node communications | UCX, OpenMPI, rocSHMEM, AMD AINIC—only for distributed (multi-node) training  | build from source or `apt`       |
| Primus + Python dependencies | Primus, submodules, and training libraries (datasets, transformers, wandb, etc.) | `git` + `pip` (in `venv`)   |

### General approach

1. **System packages (root, one-time):** install the build toolchain and, for multi-node, the RDMA/networking libraries via `apt`. The GPU kernel driver must already be loaded.
2. **Python virtual environment (no root):** create a `venv`, then install ROCm and PyTorch from AMD's TheRock multi-arch wheels—this replaces a system ROCm install and keeps everything unprivileged.
3. **Build the accelerated kernels from source** against the ROCm in `venv` and your GPU architecture (`gfx942` for MI300X/MI325X, `gfx950` for MI350X/MI355X).
4. **Install Primus and its Python dependencies**, then persist the required environment variables (ROCm paths, `NVTE_*` flags) in your `venv` activation script.
5. **(Optional) Build the multi-node communication stack** (UCX, OpenMPI, rocSHMEM) only if you require RDMA-based distributed training.

### Detailed instructions

Follow the full, step-by-step guide here, which includes the exact pinned versions, environment variables, and automated install scripts:

- **[Installing the Primus training environment on a host (no Docker)](../../docs_deprecated/install-on-host.md)**

### Verify

After the build, validate the environment and run a benchmark directly on the host (no container):

```bash
./primus-cli direct -- benchmark gemm --M 4096 --N 4096 --K 4096
```

---

## Multi-node distributed training (Slurm recommended)

For training jobs that span **multiple nodes**, we recommend using **[Slurm](https://slurm.schedmd.com/)**. Slurm is a cluster workload manager and job scheduler: it allocates nodes and GPUs, places your job on them, launches one task per node, and injects the topology information (node list, node count, per-node rank) that distributed PyTorch needs. `primus-cli` has a built-in **`slurm` mode** that wraps `srun`/`sbatch` and wires this topology into the training launcher for you.

### Cluster baseline

Before launching distributed jobs, ensure every participating node has:

- The **same software stack**—use the **same container image** on all nodes (recommended), or an identical bare-metal install (see sections above).
- A **shared filesystem** for code, datasets, checkpoints, and logs (e.g. NFS/Lustre), mounted at the same path on every node.
- **Working inter-node networking**: for best performance, use a high-speed RDMA fabric (InfiniBand, RoCE, or AMD AINIC) and ensure RCCL can select the right interface.

### Setting up Slurm

Setting up Slurm itself is a cluster-administration task and is outside Primus's scope. Follow the official documentation:

- [Slurm Quick Start (users)](https://slurm.schedmd.com/quickstart.html)
- [Slurm Quick Start Administrator Guide (install & configure)](https://slurm.schedmd.com/quickstart_admin.html)

Once `sinfo` and `srun` work on your login node, Primus can submit jobs to it. If you don't administer the cluster, your site administrator typically provides the partition, account, and reservation names you need.

### Launching with `primus-cli slurm`

The Slurm wrapper uses a single `--` separator:

- **Before the `--`**: put the launcher (`srun` or `sbatch`, default `srun`) and Slurm flags (`-N`, `-p`, `--nodelist`, `--account`, `--qos`, `--reservation`, …).
- **After the `--`**: put the Primus command to run (`train` / `benchmark` / …). It runs inside the container image (see [Selecting the container image](#selecting-the-container-image) below).

```bash
cd /path/to/Primus

# Pretrain on 2 nodes via srun
./primus-cli slurm srun -N 2 \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
```

Add the global `--dry-run` flag before the launcher to print the exact command without executing it:

```bash
./primus-cli --dry-run slurm srun -N 2 \
  -- train pretrain --config <your-config>.yaml
```

> See `runner/README.md` and the [CLI reference](../02-user-guide/cli-reference.md) for the full set of launcher flags and scenarios.

#### Selecting the container image

The image used for the job is resolved according to the following priority order (highest first):

1. **`DOCKER_IMAGE` environment variable**—overrides everything else. This is the simplest way to switch images and it propagates to all nodes:

```bash
export DOCKER_IMAGE=rocm/primus:v26.4
./primus-cli slurm srun -N 2 \
  -- train pretrain --config <your-config>.yaml
```

2. **`--image` CLI flag**—place it immediately after the `--`, before the Primus command (ignored if `DOCKER_IMAGE` is set):

```bash
./primus-cli slurm srun -N 2 \
  -- --image rocm/primus:v26.4 train pretrain --config <your-config>.yaml
```

3. **Config file default**—`container.options.image` in `runner/.primus.yaml` (or your `~/.primus.yaml`), which is set to `rocm/primus:v26.4` by default.

### Distributed environment variables

Primus launches training with `torchrun`, which needs to know the cluster topology. These are the key variables:


| Variable        | Role                                                      | Default     |
| --------------- | -------------------------------------------------------- | ----------- |
| `MASTER_ADDR`   | Hostname/IP of rank 0; all ranks rendezvous here.        | `localhost` |
| `MASTER_PORT`   | Port on the master used for rendezvous.                  | `1234`      |
| `NNODES`        | Number of nodes in the job.                              | `1`         |
| `NODE_RANK`     | Index of this node (0-based, unique per node).           | `0`         |
| `GPUS_PER_NODE` | GPUs (processes) to launch per node.                     | `8`         |

The total number of training processes (world size) is `NNODES × GPUS_PER_NODE`.

**Under Slurm, the values of these variables are derived automatically.** `primus-cli slurm` reads Slurm's own variables and sets the Primus ones for you: `NNODES` from `SLURM_NNODES`, `NODE_RANK` from `SLURM_NODEID`, and `MASTER_ADDR` from the first host in `SLURM_NODELIST` (with `MASTER_PORT` defaulting to `1234`). You normally only need to set `GPUS_PER_NODE` if your nodes don't have 8 GPUs. You can still override `MASTER_PORT` (e.g. to avoid a port clash) via `--env`.

### Without Slurm: Kubernetes or `parallel-ssh`

Slurm is recommended but not required. The mechanism underneath is simple: **set the same distributed environment variables on every node, point them all at the same `MASTER_ADDR`, give each node a unique `NODE_RANK`, and run the same `primus-cli direct` command on each node.** Any tool that can run a command across nodes works—for example Kubernetes (e.g. a `PyTorchJob` / indexed Job) or `parallel-ssh`/`pdsh`.

For a 2-node job you would run, on the master (rank 0):

```bash
export NNODES=2 GPUS_PER_NODE=8 NODE_RANK=0 MASTER_ADDR=<node0-host> MASTER_PORT=1234
./primus-cli direct -- train pretrain --config <your-config>.yaml
```

and on the worker (rank 1), the same command with `NODE_RANK=1` and the same `MASTER_ADDR`:

```bash
export NNODES=2 GPUS_PER_NODE=8 NODE_RANK=1 MASTER_ADDR=<node0-host> MASTER_PORT=1234
./primus-cli direct -- train pretrain --config <your-config>.yaml
```

With Kubernetes, inject these as container environment variables (deriving `NODE_RANK` from the pod's index); with `parallel-ssh`, pass them per host. The training command itself is identical on every node.

### Other important considerations

- **Cluster validation.** Run the built-in preflight check across your nodes before a long job: `./primus-cli slurm srun -N <N> -- preflight`.
- **Networking/RCCL.** On RDMA fabrics, make sure the correct interface is selected (Primus should auto-detect this; if it doesn't, set `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA`). Use `NCCL_DEBUG=INFO` (passed via `--env`) to diagnose hangs at startup.
- **RDMA limits.** High-performance networking usually needs locked-memory limits raised (`ulimit -l unlimited`) and sometimes hugepages—configured by your admin.
- **`MASTER_PORT` must be free** on the master node and reachable from all workers; firewalls between nodes will cause rendezvous timeouts.
- **Hugging Face access.** If your configuration downloads gated models or tokenizers, export `HF_TOKEN` (and ensure it is propagated to all nodes and into the containers).

---

## Post-installation verification checklist (for all setup approaches)


| Step                 | Check                                                                                                   |
| -------------------- | ------------------------------------------------------------------------------------------------------- |
| **ROCm**             | `rocm-smi` shows expected GPUs and no driver errors.                                                    |
| **Container engine** | `docker run --rm ... rocm/primus:v26.4` (or your site’s GPU test) succeeds.                             |
| **GEMM benchmark**   | `./primus-cli` **container** or **direct** benchmark completes (see sections above).                    |
| **Preflight**        | Run preflight diagnostics: `./primus-cli direct -- preflight` (single node) or `./primus-cli slurm srun -N <N> -- preflight` (cluster).         |


If your training pulls models or tokenizers from Hugging Face Hub, configure tokens (for example `HF_TOKEN`) in the environment or container flags as required by your configuration.

---

## Related documentation

- [Overview](./overview.md)
- [Quickstart](./quickstart.md)
- [CLI reference](../02-user-guide/cli-reference.md)
