# Native SFT (Supervised Fine-Tuning) Support for Megatron Backend

This document explains how to use native Megatron-LM for supervised fine-tuning (SFT) in Primus, without requiring the Megatron-Bridge wrapper.

## Overview

Primus now supports SFT using the native Megatron-LM backend. This provides:
- Direct integration with Megatron-LM's training infrastructure
- Full control over SFT-specific training logic
- Support for all Megatron parallelism strategies (DP, TP, PP, CP)
- Masked loss computation (only on response tokens, not instructions)

## Key Differences from Megatron-Bridge

| Feature | Native Megatron SFT | Megatron-Bridge SFT |
|---------|---------------------|---------------------|
| **Backend** | `framework: megatron` | `framework: megatron_bridge` |
| **Abstraction Level** | Lower (more control) | Higher (more convenient) |
| **Dependencies** | Megatron-LM only | Megatron-Bridge required |
| **Customization** | Full access to training loop | Limited by Megatron-Bridge API |
| **Data Format** | Megatron native format | Recipe-based format |

## Configuration

### Module Configuration

To use native SFT, create a `post_trainer` module with `framework: megatron`:

```yaml
modules:
  post_trainer:
    framework: megatron  # Use native Megatron, not megatron_bridge
    config: post_trainer.yaml
    model: llama2_7B.yaml
    
    overrides:
      # Required: Mark this as an SFT task
      is_instruction_dataset: true
      
      # SFT-specific settings
      finetune_lr: 5.0e-6
      train_iters: 200
      micro_batch_size: 1
      global_batch_size: 128
      
      # Load pretrained checkpoint
      finetune: true
      load: /path/to/pretrained/checkpoint
      no_load_optim: true
      no_load_rng: true
      
      # Data paths
      train_data_path: /path/to/sft/train/data
      valid_data_path: /path/to/sft/valid/data
```

### Task Detection

The megatron backend automatically detects SFT tasks using explicit markers:
1. `is_instruction_dataset: true` (recommended marker)
2. `is_sft: true` (alternative marker)

**Important**: At least one of these markers must be explicitly set to `true` in your configuration for the SFT trainer to be selected. The backend does NOT automatically infer SFT based on other parameters like `finetune_lr` to avoid incorrect trainer selection for non-SFT fine-tuning tasks.

### Example Configuration

See `examples/megatron/configs/MI300X/llama2_7B-BF16-sft.yaml` for a complete example.

## Data Format

### SFT Dataset Requirements

The SFT trainer expects datasets in Megatron's native format where each batch contains:
- `tokens`: Concatenated instruction + response tokens
- `labels`: Same as tokens (for next-token prediction)
- `loss_mask`: Binary mask (1 for response tokens, 0 for instruction tokens)
- `attention_mask`: Causal attention mask
- `position_ids`: Position indices for each token

### Creating SFT Datasets

**Option 1: Use Megatron's preprocessing tools**
```bash
# Convert instruction-response pairs to Megatron format
python preprocess_data.py \
    --input sft_data.jsonl \
    --output-prefix sft_dataset \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file vocab.json \
    --merge-file merges.txt \
    --append-eod \
    --workers 8
```

**Option 2: Use custom dataset class**
- Implement a custom dataset that inherits from `GPTDataset`
- Generate `loss_mask` by marking instruction tokens as 0, response tokens as 1
- Ensure proper padding and attention masking

### Data Format Example

```python
# Example batch structure for SFT
{
    'tokens': [instruction_tokens + response_tokens],  # Shape: (B, S)
    'labels': [instruction_tokens + response_tokens],  # Shape: (B, S)
    'loss_mask': [0, 0, ..., 1, 1, ...],  # Shape: (B, S), 0=instruction, 1=response
    'attention_mask': None,  # Or causal mask if needed
    'position_ids': [0, 1, 2, ..., S-1]  # Shape: (B, S)
}
```

## Training

### Using Primus CLI

```bash
# Run SFT training
./runner/primus-cli train posttrain \
    --config examples/megatron/configs/MI300X/llama2_7B-BF16-sft.yaml \
    --backend-path /path/to/Megatron-LM

# Or with container
./runner/primus-cli container --image rocm/primus:v25.10 \
    -- train posttrain --config examples/megatron/configs/MI300X/llama2_7B-BF16-sft.yaml
```

### Using Legacy Runtime

```python
from primus.pretrain import launch_pretrain_from_cli

# Load and launch SFT
launch_pretrain_from_cli(args, overrides)
```

## Implementation Details

### Module-Level Trainer

The core SFT logic is in `primus/modules/trainer/megatron/sft_trainer.py`:

- **`MegatronSFTTrainer`**: Inherits from `MegatronTrainer`
- **`get_batch()`**: Loads SFT batches with proper loss masking
- **`loss_func()`**: Computes masked loss (only on response tokens)
- **`forward_step()`**: Executes forward pass with loss masking

### Backend-Level Trainer

The backend wrapper is in `primus/backends/megatron/megatron_sft_trainer.py`:

- **`MegatronSFTTrainer`**: Inherits from `MegatronBaseTrainer`
- Delegates to module-level SFT trainer for forward_step
- Uses Megatron's `pretrain()` infrastructure
- Handles version compatibility and inprocess restart

### Adapter Logic

The `MegatronAdapter` in `primus/backends/megatron/megatron_adapter.py`:

- **`load_trainer_class()`**: Dynamically selects trainer based on config
- **`_is_sft_task()`**: Detects SFT markers in configuration
- Returns `MegatronSFTTrainer` for SFT, `MegatronPretrainTrainer` for pretrain

## Best Practices

### 1. Learning Rate

SFT typically uses much smaller learning rates than pretraining:
```yaml
finetune_lr: 5.0e-6  # vs pretrain lr: 1.0e-4 to 6.0e-4
min_lr: 0.0
lr_warmup_iters: 50
lr_decay_style: cosine
```

### 2. Training Iterations

SFT requires fewer iterations than pretraining:
```yaml
train_iters: 200  # vs pretrain: thousands to millions
eval_interval: 50
save_interval: 50
```

### 3. Batch Sizes

SFT often uses smaller batch sizes:
```yaml
micro_batch_size: 1  # or 2-4
global_batch_size: 128  # vs pretrain: 1024-2048
```

### 4. Sequence Length

SFT typically uses shorter sequences:
```yaml
seq_length: 2048  # vs pretrain: 4096-8192
```

### 5. Checkpoint Loading

Always load a pretrained checkpoint for SFT:
```yaml
finetune: true
load: /path/to/pretrained/checkpoint
no_load_optim: true  # Don't load optimizer state
no_load_rng: true  # Don't load RNG state
```

## Troubleshooting

### Issue: Trainer not detected as SFT

**Solution**: Ensure you explicitly set one of these markers in your config:
```yaml
is_instruction_dataset: true  # Recommended
# OR
is_sft: true
```

Note: The backend does NOT automatically detect SFT from other parameters like `finetune_lr` to avoid incorrect trainer selection for non-SFT fine-tuning tasks. You must explicitly mark your configuration as SFT.

### Issue: Loss not being masked correctly

**Solution**: Check your dataset's loss_mask generation:
- Instruction tokens should have loss_mask = 0
- Response tokens should have loss_mask = 1
- Padding tokens should have loss_mask = 0

### Issue: ModuleNotFoundError for pretrain_gpt

**Solution**: Ensure Megatron-LM is in your Python path:
```bash
export PYTHONPATH=/path/to/Megatron-LM:$PYTHONPATH
```

### Issue: "Backend 'megatron' not registered"

**Solution**: This should not happen with native SFT. If it does:
1. Check that `primus/backends/megatron/__init__.py` properly imports trainers
2. Verify your installation of Primus is complete

## Advanced Topics

### Custom Dataset Providers

You can provide a custom dataset provider for SFT:

```python
# In your training script or patch
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder

def custom_sft_dataset_provider(train_val_test_num_samples):
    # Your custom dataset logic here
    # Must return train_ds, valid_ds, test_ds
    pass
```

### Custom Forward Step

The SFT trainer's forward_step can be customized for specific requirements:

```python
# Override in a subclass
class CustomSFTTrainer(MegatronSFTTrainer):
    def forward_step(self, data_iterator, model, return_schedule_plan=False):
        # Custom forward logic
        pass
```

### Evaluation Metrics

For SFT, you may want custom evaluation metrics beyond perplexity:

```python
# Add custom metrics in eval phase
# - BLEU score
# - ROUGE score  
# - Generation quality metrics
```

## Comparison with Megatron-Bridge

When to use native Megatron SFT:
- ✅ Need full control over training loop
- ✅ Want to customize forward_step, loss, or data loading
- ✅ Debugging or profiling training internals
- ✅ Implementing research experiments

When to use Megatron-Bridge SFT:
- ✅ Want simplicity and convenience
- ✅ Standard SFT/LoRA workflows
- ✅ Recipe-based configuration
- ✅ Don't need to modify training internals

## References

- [Megatron-LM Documentation](https://github.com/NVIDIA/Megatron-LM)
- [Primus Documentation](../../docs/README.md)
- [Primus CLI Guide](../../docs/cli/PRIMUS-CLI-GUIDE.md)
- [Megatron-Bridge Backend](../../primus/backends/megatron_bridge/README.md)
