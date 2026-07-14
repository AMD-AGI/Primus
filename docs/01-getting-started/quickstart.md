# Quickstart (about five minutes)

This guide runs a **small Megatron-LM pretraining example** with **mock data** so you can validate the stack without preparing a full dataset. The same sample YAML works across **direct**, **container**, and **Slurm** modes.

> **Recommended: run this example inside AMD-published training Docker images.** AMD publishes ready-to-run ROCm training images (`rocm/primus` for Megatron-LM and TorchTitan, `rocm/jax-training` for MaxText) with all dependencies and the complete training software stack already installed and validated. Using them means you don't have to build or tune the environment yourself, and—most important for multi-node jobs—**every node runs an identical, tested environment**. That consistency helps avoid version-skew and configuration issues that often occur with per-host installations. Host-based installation is supported, but is recommended only for advanced users (see [Installation and setup](./installation.md)).

See [Installation and setup](./installation.md) for prerequisites and environment setup.

---

## Prerequisites

- AMD ROCm drivers (version ≥ 7.0 recommended)
- Docker (version ≥ 24.0) with ROCm support
- ROCm-compatible AMD GPUs (e.g., Instinct MI300 series)
- Proper permissions for Docker and GPU device access

---

## Option 1: Clone the repository and run training in a container (recommended)

### Step 1: Pull the container image

Check the AMD published training Docker images:

- Megatron-LM and TorchTitan backends: <https://hub.docker.com/r/rocm/primus/tags>
- MaxText backend: <https://hub.docker.com/r/rocm/jax-training/tags>

```bash
# For Megatron-LM and TorchTitan backends
docker pull rocm/primus:v26.3
# For MaxText backend
docker pull rocm/jax-training:maxtext-v26.4-jax0.9.1-te2.12.0
```

### Step 2: Clone the repository

```bash
git clone --recurse-submodules https://github.com/AMD-AGI/Primus.git
cd Primus
# checkout the branch for the specific release
git checkout release/v26.3
git submodule update --init --recursive
```

### Step 3: Run training inside container

Run the training from the repository root. If your configuration downloads weights or tokenizers from Hugging Face Hub, pass `HF_TOKEN` into the container:

```bash
./primus-cli container --image rocm/primus:v26.3 \
  --env HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml
```

---

## Option 2: Install Primus from a wheel and run training in a container

### Step 1: Install Primus as a Python package

Install Primus in a virtual environment:

```bash
python -m venv primus-env
source primus-env/bin/activate
pip install "primus==26.3.1" --no-deps --extra-index-url https://amd-agi.github.io/Primus/simple/
```

> **Note:** This installs only the Primus CLI into your virtual environment (under `site-packages`), without other dependencies. Third-party submodules are downloaded on the first run of the container, and the complete training software stack is provided in the AMD-published Docker images. You can launch `primus-cli` from any directory.

### Step 2: Run training in a container using the pip-installed Primus

```bash
primus-cli container --image rocm/primus:v26.3 \
  --env HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
  --volume /path/to/your/data:/data -- --log_file /data/run.log \
  -- train pretrain --config /data/your/config.yaml
```

> **Note:** `--volume` mounts a local data directory into the container. `--log_file` writes the training log there; if omitted, logs go to the Primus install directory (`site-packages/primus/logs` by default).

---

## Expected output

You should see the backend initialize distributed processes, load the training configuration, and emit **iteration-level logs** (with loss, throughput, step index, etc.). Exact fields depend on the backend and logging configuration; a typical pattern resembles:

```
... [INFO] starting training ...
... iteration      1 | loss: 10.xxx | ...
... iteration      2 | loss:  9.xxx | ...
```

Let the job run briefly to confirm stability; stop with `Ctrl+C` when satisfied.

---

## Same configuration, three execution modes

Use one configuration file: `examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml`.

| Mode | Example command |
|------|------------------|
| **Container** | `./primus-cli container --image rocm/primus:v26.3 -- train pretrain --config examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml` |
| **Direct** | `./primus-cli direct -- train pretrain --config examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml` |
| **Slurm** | `./primus-cli slurm srun -N <nodes> ... -- train pretrain --config examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml` |

Replace `<nodes>` and Slurm resource flags with values appropriate for your cluster. **Container** and **Slurm** runs execute inside a Docker image (Slurm dispatches through the container launcher on each node); **Direct** runs execute inside a Docker container (you start the container yourself and run the command) or directly on the host if the training environment is already set up.

> **Multi-node networking:** Primus auto-detects RDMA settings (`NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`, …) on each node. If auto-detection selects the wrong NIC, or your fabric needs specific values (RoCE `NCCL_IB_GID_INDEX`, AMD AINIC, etc.), override them via `--env` or your config. See [Multi-node networking](../04-technical-guides/multi-node-networking.md).

### Selecting the container image

For container and Slurm runs, Primus resolves which Docker image to use in the following order (**highest priority first**):

1. **`DOCKER_IMAGE` environment variable**—if set, it overrides every other source (including `--image`):

```bash
export DOCKER_IMAGE=rocm/primus:v26.3
./primus-cli container -- train pretrain --config examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml
```

2. **`--image` command-line argument**—the usual per-run override, passed as a container mode argument (before `--`):

```bash
./primus-cli container --image rocm/primus:v26.3 -- train pretrain --config examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml
```

3. **`image` field in a config file** (`container.options.image`)—used when neither of the above is set:

```yaml
container:
  options:
    image: "rocm/primus:v26.3"
```

Primus loads a **single** config file—the first that exists among `--config <file>`, then `~/.primus.yaml`, then the shipped `runner/.primus.yaml` (these files are **not** merged). Because `runner/.primus.yaml` ships with a default image, a bare `./primus-cli container -- ...` works out of the box.

>**Check the logs to make sure actual image being used is the one you wanted.**


---

## Command structure

`primus-cli` parses **global options**, a **mode** (`direct`, `container`, `slurm`, …), optional **mode-specific arguments**, then a **`--` separator** followed by the **subcommand and its arguments** (for example `train` or `benchmark`).

```
primus-cli [global-options] <mode> [mode-args] -- <command> [command-args...]
```

Example:

```text
primus-cli  container  --image rocm/primus:v26.3  --  train pretrain --config path/to/experiment.yaml
            ^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            mode        mode args                     command + args
```

---

## Next steps

| Topic | Document |
|-------|----------|
| Full CLI flags and subcommands | [CLI reference](../02-user-guide/cli-reference.md) |
| YAML presets, overrides, and composition | [Configuration system](../02-user-guide/configuration-system.md) |
| Pretraining workflows and backend notes | [Pretraining workflows](../02-user-guide/pretraining.md) |
| Terminology | [Glossary](./glossary.md) |
