# Quickstart

This quickstart guide will help you set up Primus and run your first large language model (LLM) training or benchmarking job on AMD ROCm-based GPUs—whether on a single node, across a cluster, or inside a container.

---

## 🚀 1. Installation

Primus is best used in an ROCm-enabled environment (**ROCm 6.3+ recommended**).

```bash
# Clone Primus repository
git clone https://github.com/amd/primus.git
cd primus

# Install Python dependencies (suggest using a fresh conda or venv)
pip install -r requirements.txt

# (Optional) Install ROCm-specific dependencies
# Example (PyTorch ROCm wheels):
pip install torch==2.3.0+rocm6.3 --extra-index-url https://download.pytorch.org/whl/rocm6.3

# Install Primus itself
pip install .
```

---

## ⚡ 2. Your First Primus Job (Single Node)

Run a quick Llama3-8B pretraining example using Primus CLI:

```bash
primus-cli direct -- train --config ./examples/configs/llama3_8B-pretrain.yaml
```

- `direct` mode runs on the current host with ROCm GPUs.
- Edit the config YAML to adjust GPU count, data path, or batch size.

---

## 🖥️ 3. Distributed Training on a Slurm Cluster

Primus integrates with **Slurm** for multi-node training.

### Interactive run with `srun` (4 nodes):

```bash
primus-cli slurm srun -N 4 -p AIG_Model -- train --config ./examples/configs/llama3_8B-pretrain.yaml
```

### Batch job with `sbatch` (8 nodes):

Create a script `run_slurm_job.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=primus-llama
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --partition=AIG_Model
#SBATCH --output=primus-%j.out

module load rocm/6.4

primus-cli direct -- train --config /mnt/data/exp.yaml
```

Submit it:

```bash
sbatch run_slurm_job.sh
```

---

## 🐳 4. Training Inside a Container (Docker/Podman)

Primus supports containerized workflows:

```bash
primus-cli container --mount /mnt/data:/data -- train --config /data/exp.yaml --data-path /data
```

- Use `--mount` to bind host directories (e.g., datasets, configs, output).
- Use `--env` to pass environment variables:
  ```bash
  primus-cli container --env MASTER_ADDR=10.1.1.1 -- train --config exp.yaml
  ```

---

## 🧪 5. Benchmarking and Preflight

Primus includes **benchmarking** and **preflight** checks.

### Preflight (cluster/env sanity check):

```bash
primus-cli direct -- preflight --config preflight.yaml
```

### GEMM Benchmark:

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

_Last updated: 2025-09-17_
