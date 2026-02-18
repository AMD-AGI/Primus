# 🎓 Post-Training with Primus

This guide demonstrates how to perform post-training (fine-tuning) using **Megatron Bridge** within the **Primus** framework. It covers both **Supervised Fine-Tuning (SFT)** and **Low-Rank Adaptation (LoRA)** methods for customizing pre-trained models.

---

## 📚 Table of Contents

- [🎓 Post-Training with Primus](#-post-training-with-primus)
  - [📚 Table of Contents](#-table-of-contents)
  - [🎯 Overview](#-overview)
  - [⚙️ Supported Backends](#️-supported-backends)
  - [🔧 Post-Training Methods](#-post-training-methods)
  - [🚀 Quick Start](#-quick-start)
    - [Prerequisites](#prerequisites)
    - [Basic Usage](#basic-usage)
  - [📝 Configuration Examples](#-configuration-examples)
    - [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
    - [LoRA Fine-Tuning](#lora-fine-tuning)
  - [🖥️ Single Node Training](#️-single-node-training)
    - [Direct Mode](#direct-mode)
    - [Container Mode](#container-mode)
  - [📊 Hardware-Specific Configurations](#-hardware-specific-configurations)
    - [MI300X Configurations](#mi300x-configurations)
    - [MI355X Configurations](#mi355x-configurations)
  - [🎨 Customizing Training Parameters](#-customizing-training-parameters)
  - [💡 Best Practices](#-best-practices)
  - [🔍 Troubleshooting](#-troubleshooting)

---

## 🎯 Overview

Post-training (fine-tuning) allows you to adapt pre-trained foundation models to specific tasks or domains. Primus supports two primary fine-tuning approaches:

- **Supervised Fine-Tuning (SFT)**: Full fine-tuning that updates all model parameters
- **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning that only trains lightweight adapter modules

---

## ⚙️ Supported Backends

Post-training in Primus uses the **Megatron Bridge** backend:

| Backend         | Description                                                     |
| --------------- | --------------------------------------------------------------- |
| Megatron Bridge | Bridge implementation for fine-tuning Megatron-based models    |

---

## 🔧 Post-Training Methods

| Method | Memory Usage | Training Speed | Use Case                              |
| ------ | ------------ | -------------- | ------------------------------------- |
| **SFT**  | High         | Slower         | Maximum performance, full adaptation  |
| **LoRA** | Low          | Faster         | Resource-efficient, quick iteration   |

**Key Differences:**
- **SFT** updates all model parameters, requiring more memory and compute
- **LoRA** trains only low-rank adapter matrices, significantly reducing resource requirements

---

## 🚀 Quick Start

### Prerequisites

- AMD ROCm drivers (≥ 7.0)
- Docker (≥ 24.0) with ROCm support (recommended)
- AMD Instinct GPUs (MI300X, MI355X, etc.)
- Pre-trained model checkpoint (optional, for continued training)

```bash
# Quick verification
rocm-smi && docker --version
```

### Basic Usage

The general command structure for post-training:

```bash
./runner/primus-cli <mode> train posttrain --config <config_file>
```

**Example commands:**

```bash
# SFT with direct mode
./runner/primus-cli direct train posttrain \
  --config ./examples/megatron_bridge/configs/MI355X/qwen3_32b_sft_posttrain.yaml

# LoRA with direct mode
./runner/primus-cli direct train posttrain \
  --config ./examples/megatron_bridge/configs/MI355X/qwen3_32b_lora_posttrain.yaml
```

---

## 📝 Configuration Examples

### Supervised Fine-Tuning (SFT)

Full fine-tuning configuration example for **Qwen3 32B** on **MI355X**:

```yaml
work_group: ${PRIMUS_TEAM:amd}
user_name: ${PRIMUS_USER:root}
exp_name: ${PRIMUS_EXP_NAME:qwen3_32b_sft_posttrain}
workspace: ${PRIMUS_WORKSPACE:./output}

modules:
  post_trainer:
    framework: megatron_bridge
    config: sft_trainer.yaml
    model: qwen3_32b.yaml

    overrides:
      # Parallelism configuration
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
      context_parallel_size: 1
      sequence_parallel: false

      # Fine-tuning method
      peft: "none"  # Full fine-tuning
      
      # Training configuration
      train_iters: 200
      global_batch_size: 8
      micro_batch_size: 1
      seq_length: 8192

      # Optimizer configuration
      finetune_lr: 5.0e-6
      min_lr: 0.0
      lr_warmup_iters: 50

      # Precision
      precision_config: bf16_mixed
```

**Configuration location:** `examples/megatron_bridge/configs/MI355X/qwen3_32b_sft_posttrain.yaml`

### LoRA Fine-Tuning

Parameter-efficient fine-tuning configuration for **Qwen3 32B** on **MI355X**:

```yaml
work_group: ${PRIMUS_TEAM:amd}
user_name: ${PRIMUS_USER:root}
exp_name: ${PRIMUS_EXP_NAME:qwen3_32b_lora_posttrain}
workspace: ${PRIMUS_WORKSPACE:./output}

modules:
  post_trainer:
    framework: megatron_bridge
    config: sft_trainer.yaml
    model: qwen3_32b.yaml

    overrides:
      # Parallelism configuration
      tensor_model_parallel_size: 1  # LoRA requires less parallelism
      pipeline_model_parallel_size: 1
      context_parallel_size: 1
      sequence_parallel: false

      # Fine-tuning method
      peft: lora  # LoRA fine-tuning
      
      # Training configuration
      train_iters: 200
      global_batch_size: 32
      micro_batch_size: 4
      seq_length: 8192

      # Optimizer configuration
      finetune_lr: 1.0e-4  # Higher LR for LoRA
      min_lr: 0.0
      lr_warmup_iters: 50

      # Precision
      precision_config: bf16_mixed
      
      # Recompute configuration
      recompute_granularity: full
      recompute_method: uniform
      recompute_num_layers: 1
```

**Configuration location:** `examples/megatron_bridge/configs/MI355X/qwen3_32b_lora_posttrain.yaml`

---

## 🖥️ Single Node Training

### Direct Mode

Best for local development or when running directly on bare metal with ROCm installed.

**SFT Example:**
```bash
./runner/primus-cli direct train posttrain \
  --config ./examples/megatron_bridge/configs/MI355X/qwen3_32b_sft_posttrain.yaml
```

**LoRA Example:**
```bash
./runner/primus-cli direct train posttrain \
  --config ./examples/megatron_bridge/configs/MI355X/qwen3_32b_lora_posttrain.yaml
```

**MI300X Examples:**
```bash
# SFT on MI300X
./runner/primus-cli direct train posttrain \
  --config ./examples/megatron_bridge/configs/MI300X/qwen3_32b_sft_posttrain.yaml

# LoRA on MI300X
./runner/primus-cli direct train posttrain \
  --config ./examples/megatron_bridge/configs/MI300X/qwen3_32b_lora_posttrain.yaml
```

### Container Mode

Recommended for environment isolation and dependency management.

**Pull Docker image:**
```bash
docker pull docker.io/rocm/primus:v26.1
```

**SFT Example:**
```bash
./runner/primus-cli container --image rocm/primus:v26.1 \
  train posttrain \
  --config ./examples/megatron_bridge/configs/MI355X/qwen3_32b_sft_posttrain.yaml
```

**LoRA Example:**
```bash
./runner/primus-cli container --image rocm/primus:v26.1 \
  train posttrain \
  --config ./examples/megatron_bridge/configs/MI355X/qwen3_32b_lora_posttrain.yaml
```

---

## 📊 Hardware-Specific Configurations

### MI300X Configurations

Available configurations for AMD Instinct MI300X GPUs:

| Model      | Method | Config File                                 | TP | GBS | MBS | Seq Len |
| ---------- | ------ | ------------------------------------------- | -- | --- | --- | ------- |
| Qwen3 32B  | SFT    | `MI300X/qwen3_32b_sft_posttrain.yaml`       | 2  | 8   | 2   | 8192    |
| Qwen3 32B  | LoRA   | `MI300X/qwen3_32b_lora_posttrain.yaml`      | 1  | 32  | 2   | 8192    |

**Example:**
```bash
./runner/primus-cli direct train posttrain \
  --config ./examples/megatron_bridge/configs/MI300X/qwen3_32b_sft_posttrain.yaml
```

### MI355X Configurations

Available configurations for AMD Instinct MI355X GPUs:

| Model        | Method | Config File                                 | TP | GBS | MBS | Seq Len |
| ------------ | ------ | ------------------------------------------- | -- | --- | --- | ------- |
| Qwen3 32B    | SFT    | `MI355X/qwen3_32b_sft_posttrain.yaml`       | 1  | 8   | 1   | 8192    |
| Qwen3 32B    | LoRA   | `MI355X/qwen3_32b_lora_posttrain.yaml`      | 1  | 32  | 4   | 8192    |

**Legend:**
- **TP**: Tensor Parallelism Size
- **GBS**: Global Batch Size
- **MBS**: Micro Batch Size (per GPU)
- **Seq Len**: Sequence Length

**Example:**
```bash
./runner/primus-cli direct train posttrain \
  --config ./examples/megatron_bridge/configs/MI355X/qwen3_32b_lora_posttrain.yaml
```

---

## 🎨 Customizing Training Parameters

Key parameters you can customize in the YAML configuration:

### Parallelism Settings
```yaml
tensor_model_parallel_size: 1    # Number of GPUs for tensor parallelism (1-8)
pipeline_model_parallel_size: 1  # Number of GPUs for pipeline parallelism
context_parallel_size: 1         # Context parallelism for long sequences
sequence_parallel: false         # Enable sequence parallelism
```

### Training Hyperparameters
```yaml
train_iters: 200              # Total training iterations
global_batch_size: 8          # Global batch size (8-32 depending on config)
micro_batch_size: 1           # Batch size per GPU (1-4 depending on config)
seq_length: 2048              # Sequence length (2048-8192 depending on model)
eval_interval: 30             # Evaluate every N iterations
save_interval: 50             # Save checkpoint every N iterations
```

### Learning Rate Configuration
```yaml
finetune_lr: 1.0e-4          # Initial learning rate
min_lr: 0.0                  # Minimum learning rate
lr_warmup_iters: 50          # Number of warmup iterations
lr_decay_iters: null         # Learning rate decay iterations
```

### Fine-Tuning Method
```yaml
peft: lora                   # Options: "lora" or "none" (for full SFT)
packed_sequence: false       # Enable packed sequences for efficiency
```

### Precision Configuration
```yaml
precision_config: bf16_mixed  # Options: bf16_mixed, fp16_mixed, fp32
```

### Memory Optimization
```yaml
recompute_granularity: full   # Options: full, selective, null
recompute_method: uniform     # Recompute strategy
recompute_num_layers: 1       # Number of layers to recompute
```

---

## 💡 Best Practices

### Choosing Between SFT and LoRA

**Use SFT when:**
- You need maximum model performance
- You have sufficient GPU memory
- Training time is not critical
- You want full model adaptation

**Use LoRA when:**
- GPU memory is limited
- You need fast iteration cycles
- Training multiple task-specific adapters
- Parameter efficiency is important

### Parallelism Configuration

**For SFT:**
- Use higher `tensor_model_parallel_size` for large models (e.g., TP=8 for 70B)
- Consider pipeline parallelism for very large models
- Examples: 
  - 32B model: TP=1-2 (MI300X: TP=2, MI355X: TP=1)
  - 70B model: TP=8

**For LoRA:**
- Lower `tensor_model_parallel_size` due to reduced memory
- LoRA can fit larger models with less parallelism
- Examples:
  - 32B model: TP=1
  - 70B model: TP=8 (still requires high TP due to model size)

### Learning Rate Guidelines

- **SFT**: Use lower learning rates (5e-6 to 1e-5)
- **LoRA**: Use higher learning rates (1e-4 to 5e-4)
- Always use warmup for stable training

### Batch Size Recommendations

- Start with `global_batch_size: 8` for SFT development
- LoRA can use higher batch sizes (e.g., 32) due to lower memory usage
- Increase for production: 64, 128, or higher
- Adjust `micro_batch_size` (1-4) based on GPU memory and sequence length
- Longer sequences (8192) may require higher `micro_batch_size` for efficiency

---

## 🔍 Troubleshooting

### Out of Memory (OOM) Errors

**For SFT:**
1. Increase `tensor_model_parallel_size`
2. Reduce `micro_batch_size`
3. Enable gradient checkpointing:
   ```yaml
   recompute_granularity: full
   recompute_method: uniform
   recompute_num_layers: 1
   ```
4. Reduce `seq_length`

**For LoRA:**
1. LoRA should have lower memory usage; verify `peft: lora` is set
2. Reduce `micro_batch_size` if still facing OOM
3. Enable recomputation as above

### Training Instability

1. **Check learning rate**: Reduce if loss is spiking
2. **Increase warmup**: Try `lr_warmup_iters: 100` or higher
3. **Use mixed precision**: Ensure `precision_config: bf16_mixed`
4. **Monitor gradients**: Watch for gradient explosions

### Slow Training Speed

1. **Optimize batch size**: Increase `global_batch_size` if possible
2. **Check parallelism**: Ensure optimal TP/PP configuration
3. **Use container mode**: Docker containers can improve performance
4. **Profile execution**: Use profiling tools to identify bottlenecks

### Configuration Issues

1. **Verify paths**: Ensure config file paths are correct
2. **Check YAML syntax**: Validate indentation and structure
3. **Environment variables**: Set `PRIMUS_WORKSPACE` if needed
4. **Model checkpoint**: Verify pre-trained checkpoint path (if using)

---

## 🎯 Summary Commands

**Quick reference for common post-training tasks:**

```bash
# SFT on MI355X (direct mode)
./runner/primus-cli direct train posttrain \
  --config ./examples/megatron_bridge/configs/MI355X/qwen3_32b_sft_posttrain.yaml

# LoRA on MI355X (direct mode)
./runner/primus-cli direct train posttrain \
  --config ./examples/megatron_bridge/configs/MI355X/qwen3_32b_lora_posttrain.yaml

# SFT on MI300X (container mode)
./runner/primus-cli container --image rocm/primus:v26.1 train posttrain \
  --config ./examples/megatron_bridge/configs/MI300X/qwen3_32b_sft_posttrain.yaml
```

---

**Need help?** Open an issue on [GitHub](https://github.com/AMD-AIG-AIMA/Primus/issues).

**Start fine-tuning with Primus! 🚀**
