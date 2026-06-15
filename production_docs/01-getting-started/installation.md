# Installation and setup

This guide covers supported platforms, prerequisites, and three deployment patterns: **container (recommended)**, **bare metal**, and **Slurm**.

---

## Supported platforms


| Requirement | Notes                                                                                                       |
| ----------- | ----------------------------------------------------------------------------------------------------------- |
| **OS**      | Linux (ROCm-supported distributions per AMD documentation).                                                 |
| **ROCm**    | **≥ 7.0** recommended.                                                                                      |
| **GPUs**    | AMD Instinct **MI300X**, **MI325X**, **MI355X** (or other ROCm-supported Instinct SKUs your site supports). |


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

AMD publishes training docker images monthly, which provides consistent, ready-to-run environment optimized for AMD GPUs. It is recommended to use the AMD published training Docker images together with this Primus-LM repository to run your training jobs. It supports pre-training and post-training workflows with multiple backends including Megatron-LM, TorchTitan, and JAX MaxText, alongside ROCm-optimized components.

Check the AMD published training Docker images here:

- For Megatron-LM and TorchTitan backends: [https://hub.docker.com/r/rocm/primus/tags](https://hub.docker.com/r/rocm/primus/tags)
- For MaxText backend: [https://hub.docker.com/r/rocm/jax-training/tags](https://hub.docker.com/r/rocm/jax-training/tags)

### 1. Pull the image

```bash
# For Megatron-LM and TorchTitan backends
docker pull rocm/primus:v26.3
# For MaxText backend
docker pull rocm/jax-training:v26.3
```

### 2. Clone the repository

Submodules are required for third-party backends and tools:

```bash
git clone --recurse-submodules https://github.com/AMD-AIG-AIMA/Primus.git
cd Primus
# checkout the branch for the specific release
git checkout release/v26.3
git submodule update --init --recursive
```

### 3. Run a verification benchmark

From the repository root:

```bash
./primus-cli container --image rocm/primus:v26.3 -- \
  benchmark gemm --M 4096 --N 4096 --K 4096
```

A successful run exercises the GPU stack and Primus CLI wiring without launching a full training job.

---

## Bare-metal setup

Use bare-metal mode when you manage Python, ROCm, and frameworks directly on the host.

### 1. Clone the repository

```bash
git clone --recurse-submodules https://github.com/AMD-AIG-AIMA/Primus.git
cd Primus
```

### 2. Install Python dependencies


| File                       | Use                                                                       |
| -------------------------- | ------------------------------------------------------------------------- |
| `requirements.txt`         | PyTorch-oriented backends (Megatron-LM, TorchTitan, and related tooling). |
| `requirements-jax.txt`     | JAX / MaxText paths.                                                      |
| `requirements-torchft.txt` | Optional fault-tolerance extras for Torch-based runs.                     |


```bash
pip install -r requirements.txt
# For JAX / MaxText:
pip install -r requirements-jax.txt
# Optional:
pip install -r requirements-torchft.txt
```

Representative packages pulled by these files include **loguru**, **wandb**, **pre-commit**, **nltk**, **matplotlib**, **torchao**, **datasets**, **mlflow**, **pyrsmi**, and other helpers declared in the requirement files.

Ensure your **PyTorch** and **ROCm** builds match AMD’s compatibility matrix for your GPU and driver version.

### 3. Run a verification benchmark

```bash
./primus-cli direct -- benchmark gemm --M 4096 --N 4096 --K 4096
```

---

## Slurm cluster setup

### Baseline

- Install the same ROCm stack (and optionally the same container image) on **all** nodes that participate in distributed jobs.
- Shared filesystems for code and experiment outputs are typical; configure according to your site.

### Environment variables

Distributed jobs commonly rely on variables such as:


| Variable        | Role                                    |
| --------------- | --------------------------------------- |
| `MASTER_ADDR`   | Hostname or IP of rank 0.               |
| `NNODES`        | Number of nodes in the job.             |
| `NODE_RANK`     | Index of this node (0-based).           |
| `GPUS_PER_NODE` | GPUs visible per node for the launcher. |


Exact names may vary with your scheduler integration; align with your cluster’s Primus or PyTorch launch scripts.

### Example: Slurm with container mode

Two nodes, container image on the worker command line:

```bash
./primus-cli slurm srun -N 2 -- \
  benchmark gemm --M 4096
```

Adjust partition, account, GPU GRES, and bind mounts to match your site. For production training, mount datasets, caches, and artifact directories as needed.

---

## Post-installation verification checklist


| Step                 | Check                                                                                                   |
| -------------------- | ------------------------------------------------------------------------------------------------------- |
| **ROCm**             | `rocm-smi` shows expected GPUs and no driver errors.                                                    |
| **Container engine** | `docker run --rm ... rocm/primus:v26.2` (or your site’s GPU test) succeeds.                             |
| **GEMM benchmark**   | `./primus-cli` **container** or **direct** benchmark completes (see sections above).                    |
| **Preflight**        | Run preflight diagnostics against your cluster when available (`primus/tools/preflight/` in-repo docs). |


If training pulls models or tokenizers from Hugging Face Hub, configure tokens (for example `HF_TOKEN`) in the environment or container flags as required by your config.

---

## Related documentation

- [Overview](./overview.md)
- [Quickstart](./quickstart.md)
- [CLI reference](../02-user-guide/cli-reference.md)

