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
# From the Primus repo root:
export PRIMUS_PATH="$(pwd)"

# Use default config (Llama3.1 8B BF16)
bash "$PRIMUS_PATH/runner/primus-cli" direct \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml

# Pass training arguments
bash "$PRIMUS_PATH/runner/primus-cli" direct \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml \
  --train_iters 100 \
  --micro_batch_size 2 \
  --global_batch_size 32
```

### Default Configuration

- **Config**: `examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml`
- **Override**: Set `EXP` environment variable

### Use Case Scenarios

#### Scenario 1: Quick Model Validation

Test a new model configuration before launching expensive multi-node training:

```bash
# Run 10 iterations to validate setup
bash "$PRIMUS_PATH/runner/primus-cli" direct \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml \
  --train_iters 10 \
  --eval_iters 5 \
  --log-level DEBUG
```

#### Scenario 2: Hyperparameter Tuning

Experiment with different batch sizes and learning rates:

```bash
# Test different batch size
EXP=configs/llama3_8B.yaml \
bash "$PRIMUS_PATH/runner/primus-cli" direct \
  -- train pretrain \
  --config "$EXP" \
  --micro_batch_size 4 \
  --global_batch_size 64 \
  --lr 3e-4

# Test with gradient accumulation
bash "$PRIMUS_PATH/runner/primus-cli" direct \
  -- train pretrain \
  --config "$EXP" \
  --micro_batch_size 2 \
  --global_batch_size 128 \
  --gradient_accumulation_steps 32
```

#### Scenario 3: Small-Scale Pre-training

Train smaller models (< 10B parameters) on single node with 8 GPUs:

```bash
# Llama3 8B on 8x MI300X GPUs
EXP=configs/llama3_8B-BF16.yaml \
bash "$PRIMUS_PATH/runner/primus-cli" direct \
  -- train pretrain \
  --config "$EXP" \
  --train_iters 10000 \
  --save_interval 1000 \
  --eval_interval 500
```

#### Scenario 4: Fine-tuning and Continued Pre-training

Resume from checkpoint for fine-tuning:

```bash
# Resume from checkpoint
bash "$PRIMUS_PATH/runner/primus-cli" direct \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml \
  --load_checkpoint /path/to/checkpoint \
  --train_iters 5000 \
  --lr 1e-5
```

#### Scenario 5: Performance Profiling

Profile training performance with NUMA binding:

```bash
# Enable NUMA binding for optimal memory access
bash "$PRIMUS_PATH/runner/primus-cli" direct \
  --numa \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml \
  --profile \
  --train_iters 50
```

#### Scenario 6: Dataset Preprocessing Validation

Verify dataset loading and preprocessing:

```bash
# Dry run to check data pipeline
bash "$PRIMUS_PATH/runner/primus-cli" direct \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml \
  --train_iters 1 \
  --dry-run \
  --log-level DEBUG
```

---

## 2️⃣ Slurm Mode - Multi-Node Training

**Best for**: Large-scale distributed training across multiple nodes in a cluster environment.

### Basic Usage

```bash
export PRIMUS_PATH="$(pwd)"

# Multi-node training (example: 4 nodes)
NNODES=4
EXP=examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml

bash "$PRIMUS_PATH/runner/primus-cli" slurm srun \
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
bash $PRIMUS_PATH/runner/primus-cli slurm srun \
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
bash "$PRIMUS_PATH/runner/primus-cli" slurm srun \
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
bash $PRIMUS_PATH/runner/primus-cli slurm srun \
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
  bash "$PRIMUS_PATH/runner/primus-cli" slurm srun \
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
bash $PRIMUS_PATH/runner/primus-cli slurm srun \
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
bash $PRIMUS_PATH/runner/primus-cli slurm srun \
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
bash $PRIMUS_PATH/runner/primus-cli slurm srun \
  -N 8 \
  --constraint="mi300x" \
  --exclusive \
-- train pretrain --config $EXP

# Use A100 nodes for comparison
bash $PRIMUS_PATH/runner/primus-cli slurm srun \
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

bash $PRIMUS_PATH/runner/primus-cli slurm srun \
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
bash "$PRIMUS_PATH/runner/primus-cli" slurm srun \
  -N 4 \
  --nodelist "gpu[01-04]" \
-- train pretrain \
  --config "$EXP" \
  --train_iters 1000 \
  --global_batch_size 512 \
  --lr 1e-4

# Stage 2: Main training with larger batch
bash "$PRIMUS_PATH/runner/primus-cli" slurm srun \
  -N 8 \
  --nodelist "gpu[01-08]" \
-- train pretrain \
  --config "$EXP" \
  --load_checkpoint /path/to/stage1_checkpoint \
  --train_iters 50000 \
  --global_batch_size 2048 \
  --lr 3e-4

# Stage 3: Fine-tuning with smaller learning rate
bash "$PRIMUS_PATH/runner/primus-cli" slurm srun \
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
bash "$PRIMUS_PATH/runner/primus-cli" direct \
  --numa \
  -- train pretrain \
  --config "$EXP"

# Slurm Mode: Optimize NCCL settings
bash "$PRIMUS_PATH/runner/primus-cli" slurm srun \
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
bash "$PRIMUS_PATH/runner/primus-cli" direct -- train pretrain --config "$EXP" --dry-run
bash "$PRIMUS_PATH/runner/primus-cli" slurm srun -N 2 --nodelist "gpu[01-02]" -- train pretrain --config "$EXP" --dry-run

# Short training run for smoke testing
bash "$PRIMUS_PATH/runner/primus-cli" direct -- train pretrain --config "$EXP" --train_iters 5
bash "$PRIMUS_PATH/runner/primus-cli" slurm srun -N 2 --nodelist "gpu[01-02]" -- train pretrain --config "$EXP" --train_iters 5
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
bash "$PRIMUS_PATH/runner/primus-cli" direct \
  -- train pretrain \
  --config "$EXP" \
  --load_checkpoint /path/to/checkpoint

# Slurm Mode
bash "$PRIMUS_PATH/runner/primus-cli" slurm srun \
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
bash "$PRIMUS_PATH/runner/primus-cli" slurm srun \
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

- [Primus CLI Full Documentation](../runner/README.md)
- [Configuration Examples](../examples/)
- [Troubleshooting Guide](../docs/troubleshooting.md)

---

**Last Updated**: 2026-01-12
**Version**: v1.2
