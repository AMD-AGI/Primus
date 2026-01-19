# Megatron-Bridge Backend

This backend integrates [NVIDIA Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) with Primus for **post-training tasks**.

## Overview

Megatron-Bridge is a training library for Megatron-based models with bidirectional Hugging Face conversion capability. In Primus, it is specialized for **post-training tasks** such as:

- **Supervised Fine-Tuning (SFT)**: Fine-tune pretrained models on instruction datasets
- **Instruction Tuning**: Adapt models to follow instructions
- **LoRA Fine-Tuning**: Parameter-efficient fine-tuning with Low-Rank Adaptation
- **Chat Model Training**: Fine-tune models for conversational AI
- **HuggingFace Integration**: Seamless conversion between Megatron and HuggingFace formats

**Note**: For pre-training, use the `megatron` or `torchtitan` backends. Megatron-Bridge is optimized for post-training workflows.

### Supported Model Architectures

- Llama (1, 2, 3, 3.1, 3.2, 3.3)
- GPT (NeoX, GPT-2)
- Mistral and Mixtral
- Gemma and Gemma2
- Qwen (2, 2.5, 3)
- And many more

## Architecture

The Megatron-Bridge backend consists of:

```
megatron_bridge/
├── __init__.py                              # Backend registration
├── megatron_bridge_adapter.py              # BackendAdapter implementation
├── argument_builder.py                     # Config → Megatron-Bridge args conversion
├── megatron_bridge_posttrain_trainer.py   # Posttrain trainer (SFT, LoRA)
├── patches/                                # Backend-specific patches
│   └── __init__.py                         # Patch registry
└── README.md                               # This file
```

## Key Components

### MegatronBridgeAdapter

The adapter class that implements the `BackendAdapter` protocol:
- Prepares the Megatron-Bridge environment
- Converts Primus configs to Megatron-Bridge arguments
- Handles recipe loading and HuggingFace conversions
- Provides the posttrain trainer class to Primus

### MegatronBridgeArgBuilder

Responsible for translating Primus configuration to Megatron-Bridge arguments:
- Merges CLI arguments, config files, and defaults
- Supports recipe-based configuration
- Handles distributed training environment
- Computes derived values (e.g., FFN sizes)
- Configures LoRA parameters

### MegatronBridgePosttrainTrainer

The trainer class that executes Megatron-Bridge post-training:
- Loads pretrained checkpoints from Megatron or HuggingFace
- Handles HuggingFace model conversion (bidirectional)
- Supports LoRA and other parameter-efficient fine-tuning methods
- Manages instruction/SFT dataset formatting
- Supports chat templates and prompt formatting
- Exports fine-tuned models to HuggingFace format

## Usage

### Supervised Fine-Tuning (SFT)

```bash
primus train \
    --framework megatron_bridge \
    --config configs/megatron_bridge/llama_sft_posttrain.yaml \
    --load /path/to/pretrained/checkpoint \
    --data_path /path/to/instruction/data
```

### LoRA Fine-Tuning

```bash
primus train \
    --framework megatron_bridge \
    --config configs/megatron_bridge/llama_lora_posttrain.yaml \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 32
```

### Fine-tuning from HuggingFace Model

```bash
primus train \
    --framework megatron_bridge \
    --convert_from_hf \
    --hf_model_name_or_path meta-llama/Llama-3-8B \
    --data_path /path/to/instruction/data \
    --lr 5e-6 \
    --train_iters 5000
```

### Exporting to HuggingFace Format

```bash
# Export after training (or specify convert_to_hf in config)
python -m primus.cli.convert \
    --framework megatron_bridge \
    --load /path/to/finetuned/checkpoint \
    --hf_save_path /path/to/output/hf_model \
    --convert_to_hf
```

## Configuration

Megatron-Bridge supports both traditional argument-based configuration and recipe-based configuration for post-training.

### SFT Configuration Example

```yaml
framework: megatron_bridge

# Load pretrained model
model:
  load: /path/to/pretrained/checkpoint

# Post-training hyperparameters
training:
  micro_batch_size: 1
  global_batch_size: 128
  train_iters: 5000
  lr: 5.0e-6
  min_lr: 5.0e-7

# LoRA configuration (optional)
lora:
  use_lora: true
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.05

# Instruction dataset
data:
  data_path: /path/to/instruction/data
  dataset_format: "alpaca"
  prompt_template: "alpaca"

# Checkpointing
checkpointing:
  save: /path/to/finetuned/checkpoints
  save_interval: 500
```

### LoRA Configuration Example

```yaml
framework: megatron_bridge

# Load from HuggingFace
convert_from_hf: true
hf_model_name_or_path: "meta-llama/Llama-3-8B"

# LoRA parameters
lora:
  use_lora: true
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.05
  lora_target_modules:
    - q_proj
    - v_proj
    - k_proj
    - o_proj

# Training
training:
  micro_batch_size: 2
  global_batch_size: 64
  train_iters: 3000
  lr: 2.0e-4

# Data
data:
  data_path: /path/to/instruction/data
  dataset_format: "alpaca"
```

## Post-Training Features

### Parameter-Efficient Fine-Tuning

**LoRA (Low-Rank Adaptation):**
- Configurable rank, alpha, and dropout
- Target specific modules (attention, MLP)
- Save only adapter weights (much smaller checkpoints)
- Merge adapters with base model for deployment

**Benefits:**
- Much lower memory requirements
- Faster training
- Smaller checkpoint sizes
- Multiple adapters can be trained for different tasks

### Instruction Dataset Support

Supported dataset formats:
- **Alpaca**: Standard instruction-following format
- **Dolly**: Databricks instruction format
- **OASST**: Open Assistant conversation format
- **ShareGPT**: Multi-turn conversation format
- **Custom**: Define your own format

### Prompt Templates

Built-in prompt templates:
- `alpaca`: Alpaca instruction format
- `chatml`: ChatML conversation format
- `llama2`: Llama 2 chat format
- `custom`: Define your own template

### Chat Format Support

For conversational models:
- Multi-turn conversation handling
- Role-based formatting (system, user, assistant)
- Special token handling
- Context length management

## Typical Post-Training Workflow

1. **Prepare Data**: Format your instruction/chat dataset
2. **Load Pretrained Model**: From Megatron checkpoint or HuggingFace
3. **Configure LoRA** (optional): Set rank, alpha, target modules
4. **Fine-tune**: Run training with appropriate hyperparameters
5. **Evaluate**: Test on validation set
6. **Export**: Convert to HuggingFace format for deployment

## Configuration Guidelines

### Hyperparameter Recommendations

**Full Fine-Tuning:**
- Learning Rate: 5e-6 to 1e-5
- Batch Size: 64-128
- Training Steps: 3K-10K
- Warmup: 100-500 steps

**LoRA Fine-Tuning:**
- Learning Rate: 1e-4 to 3e-4 (higher than full fine-tuning)
- LoRA Rank: 8-16 (higher for more capacity)
- LoRA Alpha: 16-32 (usually 2x rank)
- Batch Size: 32-64 (can be smaller due to lower memory)

### Parallelism Strategy

For fine-tuning, you typically need less aggressive parallelism:

```yaml
parallelism:
  tensor_model_parallel_size: 1-2  # Less than pre-training
  pipeline_model_parallel_size: 1  # Often not needed
```

**LoRA fine-tuning** can often fit on a single GPU or use only data parallelism.

## Supported Models

Megatron-Bridge supports numerous model architectures. See the [official documentation](https://github.com/NVIDIA-NeMo/Megatron-Bridge#supported-models) for a complete list.

Popular models for post-training:
- **Llama Family**: Llama 2/3/3.1 (7B, 8B, 13B, 70B)
- **Mistral**: Mistral-7B, Mixtral-8x7B
- **Qwen**: Qwen2, Qwen2.5, Qwen3
- **Gemma**: Gemma 2B/7B

## Examples

See `examples/run_megatron_bridge.sh` for complete examples:

```bash
# Interactive mode
bash examples/run_megatron_bridge.sh

# Command line mode
bash examples/run_megatron_bridge.sh sft      # SFT from checkpoint
bash examples/run_megatron_bridge.sh hf       # SFT from HuggingFace
bash examples/run_megatron_bridge.sh lora     # LoRA fine-tuning
bash examples/run_megatron_bridge.sh export   # Export to HuggingFace
```

## Development Status

**Current Status**: Initial implementation for post-training

**Implemented:**
- Backend structure and registration
- Posttrain trainer framework
- LoRA configuration support
- HuggingFace conversion interface
- Configuration examples
- Patch system
- Documentation

**TODO:**
- Recipe loading implementation
- HuggingFace conversion integration
- Training loop implementation
- LoRA adapter integration
- Instruction dataset loaders
- Prompt template system
- Testing and validation

## References

- [Megatron-Bridge GitHub](https://github.com/NVIDIA-NeMo/Megatron-Bridge)
- [Megatron-Bridge Documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/)
- [Supported Models](https://github.com/NVIDIA-NeMo/Megatron-Bridge#supported-models)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## Notes

- **Pre-training**: Use `megatron` or `torchtitan` backends for pre-training from scratch
- **Post-training**: Use `megatron_bridge` for fine-tuning pretrained models
- **Conversion**: Megatron-Bridge excels at HuggingFace ↔ Megatron conversion
- **LoRA**: Ideal for parameter-efficient fine-tuning on limited resources
