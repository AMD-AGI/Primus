# Primus CLI Examples - Quick Start Guide

This document explains how to use Primus CLI example scripts for model training.

## 📚 Overview

Primus provides three training mode example scripts, **we highly recommend using Direct and Slurm modes**:

| Script | Mode | Use Case | Rating |
|--------|------|----------|--------|
| `run_pretrain_cli.sh` | Direct | Run directly on host, no container overhead | ⭐⭐⭐ |
| `run_slurm_pretrain_cli.sh` | Slurm | Cluster environment, multi-node training | ⭐⭐⭐ |
| `run_local_pretrain_cli.sh` | Container | Use Docker/Podman container, environment isolation | ⭐ |

---

## 1️⃣ Direct Mode

**Use Case**: Quick testing and training in pre-configured environments ⭐⭐⭐ **Recommended**

### Usage

```bash
# Use default config (Llama3.1 8B BF16)
bash examples/run_pretrain_cli.sh

# Use custom config
export EXP=my_experiments/custom_config.yaml
bash examples/run_pretrain_cli.sh

# Pass additional arguments
bash examples/run_pretrain_cli.sh \
  --train_iters 10 \
  --micro_batch_size 4 \
  --global_batch_size 128
```

> 💡 **Tip**: The `run_pretrain_cli.sh` script contains complete usage examples and instructions. Check the script header comments for more scenarios.

### Default Configuration

- Default: `examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml`
- Override with `EXP` environment variable

---

## 2️⃣ Slurm Mode

**Use Case**: Slurm-managed cluster environments, multi-node distributed training ⭐⭐⭐ **Recommended**

### Usage

```bash
# Basic usage (single node, default config)
bash examples/run_slurm_pretrain_cli.sh

# Specify number of nodes
NNODES=4 bash examples/run_slurm_pretrain_cli.sh

# Custom config file
EXP=my_experiments/custom_config.yaml bash examples/run_slurm_pretrain_cli.sh
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EXP` | `examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml` | Experiment config file |
| `NNODES` | `1` | Number of nodes to use |
| `MASTER_PORT` | `12345` | Master node port |
| `LOG_DIR` | `./output` | Log output directory |

### Examples

#### Scenario 1: Single Node Quick Test

```bash
# Use default config, single node training
bash examples/run_slurm_pretrain_cli.sh
```

#### Scenario 2: Standard Multi-Node Training

```bash
# 4-node training with node list
bash $PRIMUS_PATH/runner/primus-cli slurm srun \
  -N 4 \
  --nodelist "node[01-04]" \
-- train pretrain --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml

# Or use environment variables
NNODES=4 EXP=my_config.yaml bash examples/run_slurm_pretrain_cli.sh
```

#### Scenario 3: Pass Additional Training Parameters

```bash
# Override batch sizes and iterations
bash examples/run_slurm_pretrain_cli.sh \
  --micro_batch_size 4 \
  --global_batch_size 128 \
  --train_iters 10
```

#### Scenario 4: Use Container Image + Clean Up

```bash
# Use custom Docker image in Slurm mode
bash $PRIMUS_PATH/runner/primus-cli slurm srun \
  -N 2 \
  --nodelist "node[01-02]" \
-- container \
  --image docker.io/rocm/primus:v25.10 \
  --clean \
-- train pretrain --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
```

**Explanation**:
- `container` subcommand is used to specify container-related options
- `--image`: Specify Docker image
- `--clean`: Clean up container after training
- First `--` separates Slurm options
- Second `--` separates container options from training command

#### Scenario 5: Environment Variables + Container Mode

```bash
# Set debugging environment variables
bash $PRIMUS_PATH/runner/primus-cli slurm srun \
  -N 4 \
  --nodelist "node[01-04]" \
-- container \
  --image docker.io/rocm/primus:v25.10 \
-- \
  --env NCCL_DEBUG=INFO \
  --env TORCH_DISTRIBUTED_DEBUG=DETAIL \
-- train pretrain --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
```

**Explanation**:
- After first `--`: `container` subcommand and image options
- After second `--`: environment variables (`--env` can be used multiple times)
- After third `--`: training command

#### Scenario 6: Large-Scale Training + Full Configuration

```bash
#!/bin/bash
# large_scale_training.sh - Large-scale Llama3 70B training

# Experiment configuration
export EXP=experiments/llama3_70b.yaml
export LOG_DIR=/shared/experiments/llama3_70b_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"

# Cluster configuration
export NNODES=32
export MASTER_PORT=29500

# Slurm + Container + Environment variables complete example
bash $PRIMUS_PATH/runner/primus-cli slurm srun \
  -N $NNODES \
  --nodelist "gpu[001-032]" \
  --gpus-per-node=8 \
  --ntasks-per-node=8 \
-- container \
  --image docker.io/rocm/primus:v25.10 \
  --volume /shared/datasets:/data \
  --volume /shared/checkpoints:/checkpoints \
-- \
  --env NCCL_DEBUG=INFO \
  --env NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3 \
  --env TORCH_DISTRIBUTED_DEBUG=OFF \
  --env CUDA_DEVICE_MAX_CONNECTIONS=1 \
  --env HF_TOKEN \
-- train pretrain --config $EXP 2>&1 | tee "$LOG_DIR/training.log"
```

**Configuration Details**:
- 32 nodes, 8 GPUs per node (256 GPUs total)
- Mount shared dataset and checkpoint directories
- Configure NCCL communication parameters
- `HF_TOKEN` passed from host environment (for model downloading)
- Logs output to both file and terminal

#### Scenario 7: MaxText/JAX Training

```bash
# MaxText backend training (using JAX image)
export EXP=examples/maxtext/config.yaml
export NNODES=8

bash $PRIMUS_PATH/runner/primus-cli slurm srun \
  -N $NNODES \
  --nodelist "gpu[01-08]" \
-- container \
  --image docker.io/rocm/jax-training:maxtext-v25.9 \
-- \
  --env JAX_COORDINATOR_IP \
  --env JAX_COORDINATOR_PORT \
  --env XLA_FLAGS="--xla_gpu_enable_async_all_gather=true" \
  --env NVTE_FUSED_ATTN=1 \
-- train pretrain --config $EXP
```

#### Scenario 8: Specific GPU Nodes + Debug Mode

```bash
# Specify high-performance GPU nodes + enable debugging
LOG_DIR=/tmp/debug_run \
bash $PRIMUS_PATH/runner/primus-cli slurm srun \
  -N 2 \
  --nodelist "mi300x[01-02]" \
  --constraint="mi300x" \
-- \
  --env NCCL_DEBUG=INFO \
  --env TORCH_DISTRIBUTED_DEBUG=DETAIL \
  --env PRIMUS_LOG_LEVEL=DEBUG \
-- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml \
  --dry-run
```

#### Scenario 9: Simplified Version (Using Script Defaults)

If your needs are simple, you can edit the `run_slurm_pretrain_cli.sh` script:

```bash
# 1. Edit the script to modify defaults
nano examples/run_slurm_pretrain_cli.sh

# 2. Modify these lines:
# export NNODES=${NNODES:-4}              # Default 4 nodes
# export EXP=${EXP:-"your_default_config.yaml"}

# 3. For container mode, change Scenario 1 to:
# bash $PRIMUS_PATH/runner/primus-cli slurm srun -N $NNODES \
# -- container \
#   --image docker.io/rocm/primus:v25.10 \
# -- \
#   --env NCCL_DEBUG=INFO \
# -- train pretrain --config $EXP $* 2>&1 | tee $LOG_FILE

# 4. Run directly
bash examples/run_slurm_pretrain_cli.sh
```

---

## 🔧 Advanced Usage

### Passing Additional Arguments

All scripts support passing additional arguments to the `primus train` command:

```bash
# Direct mode - Override batch sizes and iterations
bash examples/run_pretrain_cli.sh \
  --train_iters 10 \
  --micro_batch_size 4 \
  --global_batch_size 128

# Slurm mode - Add checkpoint interval
bash examples/run_slurm_pretrain_cli.sh \
  --checkpoint-interval 100 \
  --log-level DEBUG
```

### Using NUMA Binding for Performance (Direct Mode)

```bash
# Enable NUMA binding
bash $PRIMUS_PATH/runner/primus-cli direct \
  --numa \
  -- train pretrain --config $EXP
```

### Slurm Advanced Options

```bash
# Specify GPU type and resources
bash $PRIMUS_PATH/runner/primus-cli slurm srun \
  -N 8 \
  --nodelist "gpu[01-08]" \
  --gpus-per-node=8 \
  --ntasks-per-node=8 \
  --constraint="mi300x" \
  --exclusive \
-- train pretrain --config $EXP
```

---

## 🐳 Container Mode (Optional)

**Note**: If you already have a pre-configured environment, we recommend using Direct or Slurm mode. Container Mode is primarily for environment isolation or special image requirements.

### Basic Usage

```bash
# PyTorch training
bash examples/run_local_pretrain_cli.sh

# MaxText/JAX training
BACKEND=MaxText bash examples/run_local_pretrain_cli.sh

# Custom image
DOCKER_IMAGE=my-registry.com/custom:v1.0 \
bash examples/run_local_pretrain_cli.sh
```

### Detailed Configuration

For detailed container configuration options, see: `bash $PRIMUS_PATH/runner/primus-cli container --help`

---

## 📝 FAQ

### Q: How do I choose which script to use?

**A**:
- 🏃 **Single machine testing/development**: Use `run_pretrain_cli.sh` (Direct Mode) - **Recommended**
- 🖥️ **Multi-node training**: Use `run_slurm_pretrain_cli.sh` (Slurm Mode) - **Recommended**
- 🐳 **Environment isolation/special images**: Use `run_local_pretrain_cli.sh` (Container Mode) - Optional

### Q: How do I set the experiment config in Direct Mode?

**A**: Specify the config file via the `EXP` environment variable:

```bash
# Method 1: Export environment variable
export EXP=examples/megatron/exp_pretrain.yaml
bash examples/run_pretrain_cli.sh

# Method 2: Inline setting
EXP=config.yaml bash examples/run_pretrain_cli.sh
```

### Q: How do I view logs in Slurm mode?

**A**: Logs are saved to `LOG_DIR/log_slurm_pretrain.txt`:

```bash
# View logs in real-time
tail -f ./output/log_slurm_pretrain.txt

# Or specify log directory
LOG_DIR=/tmp/my_logs bash examples/run_slurm_pretrain_cli.sh
tail -f /tmp/my_logs/log_slurm_pretrain.txt
```

### Q: How do I use specific GPU nodes on Slurm?

**A**: Use the `--nodelist` option:

```bash
# Method 1: Use environment variable (recommended for simple scenarios)
NNODES=4 bash examples/run_slurm_pretrain_cli.sh

# Method 2: Call CLI directly with node list
bash $PRIMUS_PATH/runner/primus-cli slurm srun \
  -N 4 \
  --nodelist "mi300x[01-04]" \
  --constraint="mi300x" \
-- train pretrain --config $EXP
```

### Q: How do I pass additional training parameters?

**A**: Add parameters after the script:

```bash
# Direct mode - Pass batch size parameters
bash examples/run_pretrain_cli.sh \
  --train_iters 10 \
  --micro_batch_size 4

# Slurm mode - Pass checkpoint parameters
bash examples/run_slurm_pretrain_cli.sh \
  --checkpoint-interval 500 \
  --enable-profiling
```

### Q: How do I validate the configuration is correct?

**A**: Use the `--dry-run` parameter:

```bash
# Direct mode
bash examples/run_pretrain_cli.sh --dry-run

# Slurm mode
bash examples/run_slurm_pretrain_cli.sh --dry-run
```

---

## 📚 References

- [Primus CLI Full Documentation](../runner/README.md)
- [Configuration File Examples](../examples/)
- [Troubleshooting Guide](../docs/troubleshooting.md)

---

## 🎯 Quick Reference

```bash
# ===== Direct Mode (Recommended for single machine testing) =====
# Use default config
bash examples/run_pretrain_cli.sh

# Use custom config
EXP=config.yaml bash examples/run_pretrain_cli.sh

# Pass additional arguments
bash examples/run_pretrain_cli.sh --train_iters 10 --micro_batch_size 4

# ===== Slurm Mode (Recommended for multi-node training) =====
# Single node
bash examples/run_slurm_pretrain_cli.sh

# Multi-node
NNODES=4 bash examples/run_slurm_pretrain_cli.sh

# Specify node list
bash $PRIMUS_PATH/runner/primus-cli slurm srun \
  -N 4 --nodelist "node[01-04]" \
-- train pretrain --config $EXP

# ===== Container Mode (Optional) =====
bash examples/run_local_pretrain_cli.sh
```

---

**Last Updated**: 2026-01-12
**Version**: v1.1
