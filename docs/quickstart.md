# Quickstart

This guide helps you set up Primus and run your first LLM training or benchmarking job on AMD ROCm-based GPUs—whether on a single node, across a cluster, or inside a container.

---

## 🚀 1.Installation / Environment Setup

Primus requires **Python 3.10+** and an ROCm-enabled environment (**ROCm 6.3+**, ROCm 6.4 recommended).
The easiest way to get a ready-to-use environment is via AMD’s pre-built ROCm container images, which already include Python and PyTorch.

There are two typical usage modes:

- **Single node (development / testing)**: run inside a pre-built AMD ROCm container.
- **Slurm cluster (multi-node training)**: submit jobs via Slurm, which launches containers automatically. No need to `docker run` manually.

### Option A: Start a development container
```bash
# Example: pull AMD ROCm + PyTorch base image
docker pull rocm/megatron-lm:v25.8_py310
```

Run the container:
```bash
# Launch container with GPU and shared memory
docker run -it --rm --device=/dev/kfd --device=/dev/dri \
    --group-add video --ipc=host --shm-size 16G \
    -v /mnt/data:/data \
    rocm/megatron-lm:v25.8_py310 bash
```

Inside the container, clone and install Primus:


```bash
# Clone with submodules
git clone --recurse-submodules git@github.com:AMD-AIG-AIMA/Primus.git

# Or initialize submodules if already cloned
git submodule update --init --recursive

cd primus

# Install Python dependencies (suggest using a fresh conda or venv)
pip install -r requirements.txt
```

> 💡 Advanced users can also use a fresh **conda** or **venv** on bare metal if not running inside Docker/Podman.


### Option B: Use on a Slurm cluster

On Slurm clusters, you don’t need to start containers manually.
Primus jobs will run inside the default ROCm image (`rocm/megatron-lm:v25.8_py310`) unless another image is specified.



---

## ⚡ 2. Your First Primus Job (Single Node, inside container)

Once inside a container (Option A), you can launch a quick Llama3-8B pretraining example using Primus CLI:

```bash
primus-cli direct -- train pretrain --config ./examples/configs/llama3_8B-pretrain.yaml
```

- `direct` mode runs on the current host with ROCm GPUs.
- Edit the config YAML to adjust GPU count, data path, or batch size.

---

## 🖥️ 3. Distributed Training (Slurm, auto-container)

For multi-node jobs, just use **Slurm**—Primus scripts handle containers automatically.

### Interactive run with `srun` (4 nodes):

```bash
primus-cli slurm srun -N 4 -p AIG_Model -- train pretrain --config ./examples/megatron/configs/llama3_8B-pretrain.yaml
```

### Batch job with `sbatch` (8 nodes):

```bash
primus-cli slurm sbatch -N 8 -p AIG_Model -- train pretrain --config ./examples/megatron/configs/llama3_8B-pretrain.yaml
```

---

## 🐳 4. Training Inside a Container (Docker/Podman)

Primus can also be launched directly in containers via the CLI.
This is useful for mounting datasets, configs, or setting custom environment variables.

```bash
primus-cli container --mount /mnt/data:/data -- train pretrain --config ./examples/megatron/configs/llama3_8B-pretrain.yaml --data_path ./data
```

Tips:
- `--mount host_dir:container_dir` must point to existing host paths.(e.g., datasets, configs, output).

---

## 🧪 5. Benchmarking & Preflight

Primus includes **benchmarking** and **preflight** checks.

### 5.1 Preflight (cluster/env check)

```bash
primus-cli direct -- preflight --config preflight.yaml
```

### 5.2 GEMM Benchmark

```bash
primus-cli direct -- benchmark gemm --m 4096 --n 4096 --k 4096
```

See [Benchmark Suite](./benchmark/overview.md) for more details.

---

## 🛠️ 6. Troubleshooting

- Ensure ROCm and PyTorch/rccl versions match your GPU hardware.
- Check [FAQ](./faq.md) for common issues (installation, GPU visibility, Slurm configs).
- For advanced CLI usage, see:
  - [CLI Usage](./cli.md)
  - [Slurm & Container Usage](./usage/slurm_container.md)

---

🎉 **Congratulations!**
You’ve successfully run your first Primus job.
Next, explore the docs for supported models, advanced features, and performance tuning tips.

---

_Last updated: 2025-09-23_
