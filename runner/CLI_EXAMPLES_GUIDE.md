# Primus CLI Examples - Quick Start Guide

This document provides a quick start guide for using **Primus CLI commands directly** (no wrapper scripts), focusing on:

- **Direct mode**: single-node training
- **Slurm mode**: multi-node distributed training

## 📚 Overview

Primus CLI provides two primary training modes:

| Mode | Command | Use Case | Rating |
|------|---------|----------|--------|
| Direct | `primus-cli direct` | Single-node training on local host | ⭐⭐⭐ |
| Slurm | `primus-cli slurm` | Multi-node distributed training on a cluster | ⭐⭐⭐ |

> **Note**: For containerized environments, use `primus-cli container` (or `slurm ... -- container ...` in Slurm mode).

---

## 1️⃣ Direct Mode - Single Node Training

**Best for**: Development, testing, single-GPU/multi-GPU training on a single machine.

### Basic Usage

```bash
# From anywhere:
cd /path/to/Primus

# Use default config (Llama3.1 8B BF16)
bash runner/primus-cli direct \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml

# Pass training arguments
bash runner/primus-cli direct \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml \
  --train_iters 100 \
  --micro_batch_size 2 \
  --global_batch_size 32
```

### Default Configuration

- **Config**: `examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml`
- **Override**: Set `EXP` environment variable

### CLI Feature Scenarios

The examples below focus on **Primus CLI features and runner behavior** (not training/hyperparameter tuning).

#### Scenario 1: Print the command without running (`--dry-run`)

Use this to validate parsing, config loading, and the final launcher command.

```bash
bash runner/primus-cli direct \
  --dry-run \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
```

#### Scenario 2: Pass environment variables via runner (`--env`)

```bash
bash runner/primus-cli direct \
  --env NCCL_DEBUG=INFO \
  --env PRIMUS_LOG_LEVEL=DEBUG \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
```

#### Scenario 3: Save logs to a file (`--log_file`)

```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_direct.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
```

#### Scenario 4: Enable NUMA binding (`--numa`)

```bash
bash runner/primus-cli direct \
  --numa \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
```

#### Scenario 5: Apply patch scripts before running (`--patch`)

Patch scripts are executed in order before launching the Python entrypoint.

```bash
bash runner/primus-cli direct \
  --patch runner/helpers/patches/00_hello_world.sh \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
```

#### Scenario 6: Run a custom Python entrypoint (`--script`) / single-process mode (`--single`)

Useful for debugging or for backends that require a non-`torchrun` launcher.

```bash
# Single process (python3), custom script, plus script args after '--'
bash runner/primus-cli direct \
  --single \
  --script runner/helpers/examples/hello_world.py \
  -- --arg1 val1
```

---

## 2️⃣ Slurm Mode - Multi-Node Training

**Best for**: Large-scale distributed training across multiple nodes in a cluster environment.

### Basic Usage

```bash
cd /path/to/Primus

# Multi-node training (example: 4 nodes)
NNODES=4
EXP=examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml

bash runner/primus-cli slurm srun \
  -N "$NNODES" \
  --nodelist "node[01-04]" \
-- train pretrain --config "$EXP"
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EXP` | `examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml` | Training config file |
| `NNODES` | `1` | Number of nodes |
| `MASTER_PORT` | `12345` | Master node port |
| `LOG_DIR` | `./output` | Log directory |

### Use Case Scenarios

#### Scenario 1: Large Language Model Pre-training

Train 70B+ parameter models across multiple nodes:

```bash
#!/bin/bash
# llama3_70b_pretrain.sh - Llama3 70B pre-training

export EXP=configs/llama3_70B-BF16.yaml
export NNODES=16
export LOG_DIR=/shared/logs/llama3_70b_$(date +%Y%m%d_%H%M%S)

# 16 nodes x 8 GPUs = 128 GPUs
bash runner/primus-cli slurm srun \
  -N $NNODES \
  --nodelist "gpu[01-16]" \
  --gpus-per-node=8 \
  --constraint="mi300x" \
-- train pretrain \
  --config $EXP \
  --train_iters 50000 \
  --global_batch_size 2048 \
  --tensor_model_parallel 8 \
  --pipeline_model_parallel 4 \
  2>&1 | tee $LOG_DIR/training.log
```

#### Scenario 2: Fault-Tolerant Training with Checkpointing

Long-running training with frequent checkpointing:

```bash
# Configure checkpoint intervals for fault recovery
bash runner/primus-cli slurm srun \
  -N 8 \
  --nodelist "gpu[01-08]" \
-- train pretrain \
  --config "$EXP" \
  --save_interval 500 \
  --checkpoint_dir /shared/checkpoints/exp_001 \
  --train_iters 100000 \
  --load_checkpoint /shared/checkpoints/exp_001/latest
```

#### Scenario 3: Mixed Precision Training at Scale

FP8 training for improved throughput:

```bash
export EXP=configs/llama3_405B-FP8.yaml
export NNODES=32

# 32 nodes for 405B model with FP8
bash runner/primus-cli slurm srun \
  -N $NNODES \
  --nodelist "gpu[001-032]" \
-- train pretrain \
  --config $EXP \
  --fp8 \
  --global_batch_size 4096
```

#### Scenario 4: Multi-Node Performance Benchmarking

Compare scaling efficiency across different node counts:

```bash
# Benchmark scaling: 1, 2, 4, 8 nodes
for N in 1 2 4 8; do
  echo "Testing with $N nodes..."
  bash runner/primus-cli slurm srun \
    -N "$N" \
    --nodelist "gpu[01-$N]" \
  -- train pretrain \
    --config "$EXP" \
    --train_iters 100 \
    --profile \
    --log_dir /tmp/benchmark_${N}nodes
done
```

#### Scenario 5: Containerized Multi-Node Training

Use custom Docker images with specific dependencies:

```bash
# Train with custom ROCm image
bash runner/primus-cli slurm srun \
  -N 4 --nodelist "node[01-04]" \
-- container \
  --image docker.io/rocm/primus:custom-v25.10 \
  --clean \
-- train pretrain \
  --config configs/custom_model.yaml \
  --train_iters 50000
```

#### Scenario 6: Debug Mode with Environment Variables

Debug distributed training issues:

```bash
# Enable verbose debugging
bash runner/primus-cli slurm srun \
  -N 2 --nodelist "node[01-02]" \
-- \
  --env NCCL_DEBUG=INFO \
  --env TORCH_DISTRIBUTED_DEBUG=DETAIL \
  --env PRIMUS_LOG_LEVEL=DEBUG \
-- train pretrain \
  --config $EXP \
  --train_iters 10 \
  --dry-run
```

#### Scenario 7: Heterogeneous GPU Allocation

Target specific GPU types in a mixed cluster:

```bash
# Use only MI300X nodes
bash runner/primus-cli slurm srun \
  -N 8 \
  --constraint="mi300x" \
  --exclusive \
-- train pretrain --config $EXP

# Use A100 nodes for comparison
bash runner/primus-cli slurm srun \
  -N 8 \
  --constraint="a100" \
  --exclusive \
-- train pretrain --config $EXP
```

#### Scenario 8: Long-Running Production Training

Production training with logging and monitoring:

```bash
#!/bin/bash
# production_training.sh - Production training workflow

export EXP=configs/production/llama3_70b.yaml
export NNODES=16
export LOG_DIR=/shared/production/runs/$(date +%Y%m%d_%H%M%S)
export CHECKPOINT_DIR=/shared/production/checkpoints/llama3_70b

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

bash runner/primus-cli slurm srun \
  -N $NNODES \
  --nodelist "gpu[001-016]" \
  --gpus-per-node=8 \
  --exclusive \
-- train pretrain \
  --config $EXP \
  --train_iters 200000 \
  --save_interval 1000 \
  --eval_interval 500 \
  --checkpoint_dir $CHECKPOINT_DIR \
  --log_dir $LOG_DIR \
  --tensorboard \
  2>&1 | tee $LOG_DIR/training.log
```

#### Scenario 9: Multi-Stage Training Pipeline

Progressively increase model size or training complexity:

```bash
# Stage 1: Warm-up with smaller batch
bash runner/primus-cli slurm srun \
  -N 4 \
  --nodelist "gpu[01-04]" \
-- train pretrain \
  --config "$EXP" \
  --train_iters 1000 \
  --global_batch_size 512 \
  --lr 1e-4

# Stage 2: Main training with larger batch
bash runner/primus-cli slurm srun \
  -N 8 \
  --nodelist "gpu[01-08]" \
-- train pretrain \
  --config "$EXP" \
  --load_checkpoint /path/to/stage1_checkpoint \
  --train_iters 50000 \
  --global_batch_size 2048 \
  --lr 3e-4

# Stage 3: Fine-tuning with smaller learning rate
bash runner/primus-cli slurm srun \
  -N 8 \
  --nodelist "gpu[01-08]" \
-- train pretrain \
  --config "$EXP" \
  --load_checkpoint /path/to/stage2_checkpoint \
  --train_iters 10000 \
  --global_batch_size 2048 \
  --lr 1e-5
```

---

## 🔧 Advanced Tips

### Performance Optimization

```bash
# Direct Mode: Enable NUMA binding
bash runner/primus-cli direct \
  --numa \
  -- train pretrain \
  --config "$EXP"

# Slurm Mode: Optimize NCCL settings
bash runner/primus-cli slurm srun \
  -N 8 \
  --nodelist "gpu[01-08]" \
-- \
  --env NCCL_IB_HCA=mlx5_0,mlx5_1 \
  --env NCCL_SOCKET_IFNAME=ib0 \
-- train pretrain --config "$EXP"
```

### Quick Validation

```bash
# Dry run to validate configuration
bash runner/primus-cli direct -- train pretrain --config "$EXP" --dry-run
bash runner/primus-cli slurm srun -N 2 --nodelist "gpu[01-02]" -- train pretrain --config "$EXP" --dry-run

# Short training run for smoke testing
bash runner/primus-cli direct -- train pretrain --config "$EXP" --train_iters 5
bash runner/primus-cli slurm srun -N 2 --nodelist "gpu[01-02]" -- train pretrain --config "$EXP" --train_iters 5
```

---

## 📝 FAQ

### Q: When should I use Direct Mode vs Slurm Mode?

**A**:
- **Direct Mode**: Single-node training, models < 13B parameters, development/testing, quick experiments
- **Slurm Mode**: Multi-node training, models > 13B parameters, production training, large-scale pre-training

### Q: How do I monitor training progress?

**A**:
```bash
# View logs in real-time
tail -f ./output/log_slurm_pretrain.txt

# Monitor GPU utilization
watch -n 1 rocm-smi
```

### Q: How do I resume from a checkpoint?

**A**:
```bash
# Direct Mode
bash runner/primus-cli direct \
  -- train pretrain \
  --config "$EXP" \
  --load_checkpoint /path/to/checkpoint

# Slurm Mode
bash runner/primus-cli slurm srun \
  -N 4 \
  --nodelist "gpu[01-04]" \
-- train pretrain \
  --config "$EXP" \
  --load_checkpoint /path/to/checkpoint
```

### Q: How do I adjust parallelism settings?

**A**:
```bash
# Tensor parallel + Pipeline parallel
bash runner/primus-cli slurm srun \
  -N 4 \
  --nodelist "gpu[01-04]" \
-- train pretrain \
  --config "$EXP" \
  --tensor_model_parallel 4 \
  --pipeline_model_parallel 2 \
  --data_parallel 8
```

---

## 📚 References

- [Primus CLI Full Documentation](./README.md)
- [Configuration Examples](../examples/)
- [Troubleshooting Guide](../docs/troubleshooting.md)

---

**Last Updated**: 2026-01-12
**Version**: v1.2
