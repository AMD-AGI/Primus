# Megatron-LM Native SFT Examples

This directory contains example configurations for running **Supervised Fine-Tuning (SFT)** using native Megatron-LM with Primus.

## Overview

The `megatron_sft` framework provides SFT capabilities using Megatron-LM's `pretrain()` infrastructure with:

- **Full Fine-tuning**: Train all model parameters
- **LoRA Support**: Parameter-efficient fine-tuning with Low-Rank Adaptation
- **SFT Dataset Support**: JSON/JSONL files and HuggingFace datasets
- **Chat Template**: Automatic chat template application via HuggingFace tokenizers
- **Answer-Only Loss**: Loss masking to only compute loss on assistant responses
- **Sample Packing**: Efficient training by packing multiple samples per sequence

## Quick Start

### Full Fine-tuning

```bash
# Run SFT with default configuration
primus train --config examples/megatron_sft/exp_sft.yaml

# Run Llama3 8B SFT on MI300X
primus train --config examples/megatron_sft/configs/MI300X/llama3_8B-BF16-sft.yaml
```

### LoRA Fine-tuning

```bash
# Run Llama3 8B with LoRA on MI300X
primus train --config examples/megatron_sft/configs/MI300X/llama3_8B-BF16-lora-sft.yaml
```

## Configuration

### Key Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `framework` | Must be `megatron_sft` | `megatron_sft` |
| `tokenizer_type` | Must be `HuggingFaceTokenizer` for chat support | `HuggingFaceTokenizer` |
| `tokenizer_model` | HuggingFace model name or path | `meta-llama/Meta-Llama-3-8B` |
| `finetune_hf_dataset` | HuggingFace dataset name | `Open-Orca/SlimOrca` |
| `train_data_path` | Path to local JSON/JSONL file | `/path/to/train.jsonl` |
| `load` | Pretrained checkpoint to fine-tune | `/path/to/checkpoint` |

### LoRA Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `peft` | PEFT method (`"none"` or `"lora"`) | `"none"` |
| `lora_enable` | Alternative way to enable LoRA | `false` |
| `lora_r` | LoRA rank (dimension of low-rank matrices) | `16` |
| `lora_alpha` | LoRA scaling factor | `32` |
| `lora_dropout` | Dropout probability for LoRA layers | `0.05` |
| `lora_bias` | Bias config (`"none"`, `"all"`, `"lora_only"`) | `"none"` |
| `lora_target_modules` | Module patterns to apply LoRA | See below |

**Default target modules for LoRA:**
- `query`, `key`, `value`, `dense` (attention)
- `linear_qkv`, `linear_proj` (attention projections)
- `linear_fc1`, `linear_fc2` (MLP layers)

## Data Format

### JSON/JSONL Format

The SFT dataset expects conversations in OpenAI format:

```json
{
  "conversations": [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ]
}
```

Or ShareGPT format (automatically converted):

```json
{
  "conversations": [
    {"from": "human", "value": "What is the capital of France?"},
    {"from": "gpt", "value": "The capital of France is Paris."}
  ]
}
```

### Supported HuggingFace Datasets

- `Open-Orca/OpenOrca`
- `Open-Orca/SlimOrca`
- `nvidia/Daring-Anteater`
- `Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered`
- `HuggingFaceH4/ultrachat_200k`

## Example Configurations

### Full Fine-tuning with HuggingFace Dataset

```yaml
modules:
  sft_trainer:
    framework: megatron_sft
    config: sft_trainer.yaml
    model: llama3_8B.yaml
    overrides:
      tokenizer_type: HuggingFaceTokenizer
      tokenizer_model: meta-llama/Meta-Llama-3-8B
      finetune_hf_dataset: Open-Orca/SlimOrca
      load: /path/to/pretrained/checkpoint
      train_iters: 2000
      lr: 2.0e-5
```

### LoRA Fine-tuning

```yaml
modules:
  sft_trainer:
    framework: megatron_sft
    config: sft_trainer.yaml
    model: llama3_8B.yaml
    overrides:
      # Enable LoRA
      peft: lora
      lora_r: 16
      lora_alpha: 32
      lora_dropout: 0.05
      lora_target_modules:
        - query
        - key
        - value
        - dense

      # Other settings
      tokenizer_type: HuggingFaceTokenizer
      tokenizer_model: meta-llama/Meta-Llama-3-8B
      finetune_hf_dataset: Open-Orca/SlimOrca
      load: /path/to/pretrained/checkpoint
      lr: 1.0e-4  # Higher LR for LoRA
```

### SFT with Local Data

```yaml
modules:
  sft_trainer:
    framework: megatron_sft
    config: sft_trainer.yaml
    model: llama3_8B.yaml
    overrides:
      tokenizer_type: HuggingFaceTokenizer
      tokenizer_model: meta-llama/Meta-Llama-3-8B
      train_data_path:
        - /data/sft/train.jsonl
      valid_data_path:
        - /data/sft/valid.jsonl
      load: /path/to/pretrained/checkpoint
```

## Directory Structure

```
examples/megatron_sft/
├── README.md                              # This file
├── exp_sft.yaml                           # Basic SFT example
└── configs/
    └── MI300X/
        ├── llama3_8B-BF16-sft.yaml        # Llama3 8B full fine-tuning
        └── llama3_8B-BF16-lora-sft.yaml   # Llama3 8B with LoRA
```

## LoRA vs Full Fine-tuning

| Aspect | Full Fine-tuning | LoRA |
|--------|------------------|------|
| Memory | High | Low (~10-20% of full) |
| Speed | Slower | Faster |
| Quality | Best | Near full-finetuning quality |
| Trainable Params | 100% | ~0.1-1% |
| Learning Rate | 1e-5 to 5e-5 | 1e-4 to 3e-4 |
| Use Case | Production models | Quick experiments, multi-adapter |

## Notes

1. **Tokenizer**: SFT requires `HuggingFaceTokenizer` to support chat templates
2. **Micro Batch Size**: Must be 1 for sample packing to work correctly
3. **Checkpoint**: Always load a pretrained checkpoint for fine-tuning
4. **LoRA Learning Rate**: Typically 5-10x higher than full fine-tuning
5. **Distributed Optimizer**: May need to be disabled for LoRA (`use_distributed_optimizer: false`)
