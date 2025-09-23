# CLI Usage

Primus provides a **unified command-line interface (CLI)** for launching training, benchmarking, and system validation tasks.
It follows a **two-phase CLI design**:
1. **Outer launcher** (`primus-cli <mode>`) — chooses execution context (host / Slurm / container).
2. **Inner Primus CLI** (`train | benchmark | preflight`) — executes the actual operation with configs & args.

This design ensures a consistent developer/user experience across local, containerized, and multi-node Slurm environments.

---


## 🌳 Command Tree

```
primus-cli
 ├── slurm        # Launch via Slurm (srun / sbatch)
 │    ├── train       # LLM training
 │    ├── benchmark   # GEMM / RCCL / end-to-end tests
 │    └── preflight   # Cluster sanity check
 │
 ├── container    # Launch in Docker/Podman container
 │    ├── train
 │    ├── benchmark
 │    └── preflight
 │
 └── direct       # Launch directly on host
      ├── train
      ├── benchmark
      └── preflight
```

---

## 📖 Unified Launcher Usage

```
Primus Unified Launcher CLI

Usage:
    primus-cli <mode> [mode-args] -- [primus-commands and args]

Description:
    The unified entry point for all Primus workflows: distributed launch, containerization, and direct host training.
    Use '--' to separate mode-specific arguments from Primus Python CLI commands (train, benchmark, etc).

Supported modes:
    slurm      Launch distributed workflows via Slurm cluster (srun/sbatch)
    container  Launch workflows in a managed Docker/Podman container
    direct     Launch workflows directly on the current host or container

Mode-specific help:
    primus-cli slurm --help         Show usage for slurm mode
    primus-cli container --help     Show usage for container mode (single node or debugging)
    primus-cli direct --help        Show usage for direct mode (host/container)
```
- **Modes** define *how* to launch (Slurm / container / host).
- **Commands** define *what* to run (`train`, `benchmark`, `preflight`).

---

## 📚 Available Subcommands

| Command      | Description                                        |
|--------------|----------------------------------------------------|
| `train`      | Launch LLM training with Megatron/Titan backends   |
| `benchmark`  | Run operator / RCCL / end-to-end perf tests        |
| `preflight`  | Run environment & cluster sanity check             |

- `train` requires an experiment YAML/TOML config (with optional CLI overrides).
- `benchmark` and `preflight` are fully configured by command-line flags.

---

## ⚙️ Configuration & Overrides

Primus supports **YAML/TOML experiment configs** with CLI overrides.

Example:

```yaml
# exp.yaml
modules:
  pre_trainer:
    framework: megatron
    config: pre_trainer.yaml

    # model to run
    model: llama3.1_8B.yaml
    overrides:
      train_iters: 50
      micro_batch_size: 2
      global_batch_size: 128
```

You can override fields at runtime:

```bash
primus-cli direct -- train pretrain --config exp.yaml --global_batch_size=2048
```

---

## 🖥️ Direct Workflow

Direct mode runs Primus directly on the host (or inside an existing container).
It requires ROCm and dependencies to be installed on the system.

```bash
# Run training directly on host
primus-cli direct -- train pretrain --config ./exp.yaml

# Run GEMM benchmark
primus-cli direct -- benchmark gemm --m 4096 --n 4096 --k 8192 --dtype bf16

# Run preflight check
primus-cli direct -- preflight all
```

Recommended for debugging or single-node experiments.
For multi-node workloads, use slurm mode instead.

---

## 🐳 Container Workflow

In container mode, Primus handles **volume mounts**, **environment variables**, and **Python path setup** automatically.

```bash
primus-cli container \
  --image rocm/megatron-lm:v25.8_py310 \
  --mount /mnt/data:/data \
  -- train pretrain --config /data/exp.yaml
```

- `--image`: container image (default: `rocm/megatron-lm:v25.8_py310`)
- `--mount`: bind mount (`host_dir:container_dir`)


---

## 🖥️ Slurm Workflow

Primus integrates seamlessly with **Slurm clusters**.

### Interactive (srun)

```bash
primus-cli slurm srun -N 4 -p AIG_Model -- train --config ./exp.yaml
```

### Batch (sbatch)

```bash
primus-cli slurm sbatch -N 8 -p AIG_Model \
  --time 02:00:00 \
  -- train --config ./exp.yaml
```

- Slurm job scripts will internally set up container + envs.
- Users only need to specify nodes, partition, and time.

---

## 🧩 Advanced Features

- **Patch args injection**:
  Export extra arguments from YAML or file via `PRIMUS_PATCH_ARGS_FILE`:
  ```bash
  export PRIMUS_PATCH_ARGS_FILE=patch_args.yaml
  primus-cli direct -- train --config exp.yaml
  ```

- **Backend path overrides**:
  - `MEGATRON_PATH`, `TORCHTITAN_PATH`, `BACKEND_PATH` allow custom backend locations.

- **Slurm-env propagation**:
  Use `--env VAR=VALUE` to explicitly forward environment variables into Slurm jobs.
  Primus also propagates a small set of essential variables by default
  (e.g. `NCCL_SOCKET_IFNAME`, `HSA_NO_SCRATCH_RECLAIM`) to ensure consistency across nodes.

  ```bash
  primus-cli slurm sbatch -N 16 -p AIG_Model -- \
    --env HSA_NO_SCRATCH_RECLAIM=1 \
    train pretrain --config examples/megatron/configs/llama3_405B.yaml
  ```

---

## 🚀 Example Workflows

### 1. Single-node pretraining

```bash
primus-cli direct -- train pretrain --config examples/megatron/configs/llama3_8B.yaml
```

### 2. Multi-node training with Slurm

```bash
primus-cli slurm sbatch -N 16 -p AIG_Model -- \
  train pretrain --config examples/megatron/configs/llama3_405B.yaml
```

### 3. Benchmark GEMM performance

```bash
primus-cli direct -- benchmark gemm --m 4096 --n 4096 --k 8192 --dtype bf16
```

### 4. Run cluster sanity check

```bash
primus-cli slurm srun -N 2 -p AIG_Model -- preflight all
```

---

## 📌 Design Principles

- **One CLI, one ecosystem** — consistent UX across host, Slurm, and containers
- **Three pillars** — `train`, `benchmark`, `preflight` unify workflows
- **Low learning curve** — config-driven, minimal boilerplate
- **Extensible** — new backends and commands can be added seamlessly

---

_Last updated: 2025-09-23_
