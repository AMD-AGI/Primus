# Megatron-LM Native SFT Examples

This directory contains example configurations for running **Supervised Fine-Tuning (SFT)** using native Megatron-LM with Primus.

## Overview

The `megatron_sft` framework provides SFT capabilities using Megatron-LM's `pretrain()` infrastructure with:

- **SFT Dataset Support**: JSON/JSONL files and HuggingFace datasets
- **Chat Template**: Automatic chat template application via HuggingFace tokenizers
- **Answer-Only Loss**: Loss masking to only compute loss on assistant responses
- **Sample Packing**: Efficient training by packing multiple samples per sequence

## Quick Start

```bash
# Run SFT with default configuration
primus train --config examples/megatron_sft/exp_sft.yaml

# Run Llama3 8B SFT on MI300X
primus train --config examples/megatron_sft/configs/MI300X/llama3_8B-BF16-sft.yaml
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

### Data Format

#### JSON/JSONL Format

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

#### Supported HuggingFace Datasets

- `Open-Orca/OpenOrca`
- `Open-Orca/SlimOrca`
- `nvidia/Daring-Anteater`
- `Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered`
- `HuggingFaceH4/ultrachat_200k`

## Example Configurations

### Basic SFT with HuggingFace Dataset

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
├── README.md                 # This file
├── exp_sft.yaml              # Basic SFT example
└── configs/
    └── MI300X/
        └── llama3_8B-BF16-sft.yaml  # Llama3 8B on MI300X
```

## Notes

1. **Tokenizer**: SFT requires `HuggingFaceTokenizer` to support chat templates
2. **Micro Batch Size**: Must be 1 for sample packing to work correctly
3. **Checkpoint**: Always load a pretrained checkpoint for fine-tuning
4. **Learning Rate**: Typically 1e-5 to 5e-5 (lower than pretraining)
