# Megatron-LM SFT Trainer

This directory contains the implementation of a Supervised Fine-Tuning (SFT) trainer directly based on Megatron-LM (without Megatron-Bridge dependency).

## Overview

The SFT trainer provides:
- Direct integration with Megatron-LM's training infrastructure
- Universal dataset interface supporting HuggingFace datasets
- Multiple conversation format support (Alpaca, ChatML, extensible)
- Proper loss masking for instruction tuning
- Compatible with distributed training
- Stage-based registration system for flexible trainer selection

## Key Components

### 1. MegatronSFTTrainer (`megatron_sft_trainer.py`)
The main trainer class that:
- Inherits from `MegatronBaseTrainer` (handles argument injection, patches)
- Implements SFT-specific training loop
- Provides dataset provider and forward step functions
- Applies loss masking to compute loss only on response tokens

### 2. SFTDataset (`core/datasets/sft_dataset.py`)
Universal dataset interface that:
- Loads data from HuggingFace datasets
- Supports multiple conversation formats (Alpaca, ChatML)
- Handles tokenization and loss mask creation
- Extensible design for adding new formats

### 3. Configuration Files
- `primus/configs/modules/megatron/sft_trainer.yaml` - Base SFT trainer config
- `examples/megatron/configs/MI355X/llama3_8B-BF16-sft.yaml` - Example experiment config

### 4. Stage-Based Registration
The trainer uses Primus's stage-based registration system:
- Registered with `stage="sft"` to differentiate from pretrain
- Allows flexible trainer selection via `stage` parameter in config

## Usage

### 1. Basic SFT Training

Create an experiment configuration file (e.g., `my_sft_config.yaml`):

```yaml
work_group: ${PRIMUS_TEAM:amd}
user_name: ${PRIMUS_USER:root}
exp_name: ${PRIMUS_EXP_NAME:llama3_8B-sft}
workspace: ${PRIMUS_WORKSPACE:./output}

modules:
  sft_trainer:
    framework: megatron
    config: sft_trainer.yaml
    model: llama3_8B.yaml
    
    overrides:
      # Specify stage to use SFT trainer
      stage: sft
      
      # SFT-specific settings
      sft_dataset_name: "tatsu-lab/alpaca"
      sft_conversation_format: "alpaca"
      
      # Training settings
      train_iters: 1000
      micro_batch_size: 1
      global_batch_size: 128
      seq_length: 2048
      
      # Learning rate (lower for fine-tuning)
      lr: 1.0e-5
      min_lr: 0.0
      lr_warmup_iters: 50
      
      # Checkpoint settings
      finetune: true
      load: /path/to/pretrained/checkpoint
      save: /path/to/save/finetuned
      save_interval: 100
      eval_interval: 50
```

### 2. Supported Datasets

The SFT trainer supports multiple data sources:

#### HuggingFace Hub Datasets
Any HuggingFace dataset with instruction-response pairs:

```yaml
# Alpaca format dataset from HuggingFace Hub
sft_dataset_name: "tatsu-lab/alpaca"

# Custom dataset from HuggingFace Hub
sft_dataset_name: "your-org/your-dataset"
```

#### Local JSONL Files (Offline Training)
For offline training, you can use local JSONL (JSON Lines) files:

```yaml
# Local JSONL file
sft_dataset_name: "/path/to/data.jsonl"

# Local JSON array file
sft_dataset_name: "/path/to/data.json"
```

**JSONL Format**: Each line is a separate JSON object:
```jsonl
{"instruction": "What is Python?", "response": "Python is a programming language."}
{"instruction": "Explain AI.", "response": "AI stands for Artificial Intelligence..."}
```

**JSON Format**: Single JSON array of objects:
```json
[
    {"instruction": "What is Python?", "response": "Python is a programming language."},
    {"instruction": "Explain AI.", "response": "AI stands for Artificial Intelligence..."}
]
```

**Supported Field Names**:
- Instruction: `instruction`, `prompt`, or `question`
- Response: `response`, `output`, or `answer`
- Optional: `input` (additional context), `system` (system prompt)

### 3. Conversation Formats

#### Alpaca Format
```python
# Default format
sft_conversation_format: "alpaca"

# Format:
# Below is an instruction that describes a task...
# ### Instruction:
# {instruction}
# ### Response:
# {response}
```

#### ChatML Format
```python
sft_conversation_format: "chatml"

# Format:
# <|im_start|>system
# {system_prompt}<|im_end|>
# <|im_start|>user
# {instruction}<|im_end|>
# <|im_start|>assistant
# {response}<|im_end|>
```

#### OpenAI Messages Format (Multi-Turn Conversations)
```python
sft_conversation_format: "openai"  # or "messages"

# Supports multi-turn conversations with role-content pairs
# Data format:
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "content": "I don't have access to weather data."}
  ]
}

# Formatted output (ChatML-style):
# <|im_start|>system
# You are a helpful assistant<|im_end|>
# <|im_start|>user
# Hello<|im_end|>
# <|im_start|>assistant
# Hi! How can I help?<|im_end|>
# <|im_start|>user
# What's the weather?<|im_end|>
# <|im_start|>assistant
# I don't have access to weather data.<|im_end|>

# Loss is computed ONLY on assistant message content
```

**Key Features of OpenAI Messages Format:**
- ✅ Support for multi-turn conversations
- ✅ Loss computed only on assistant responses
- ✅ System, user, and assistant roles supported
- ✅ Compatible with OpenAI API message format
- ✅ Automatically masks non-assistant content

**Example JSONL with messages:**
```jsonl
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
{"messages": [{"role": "system", "content": "Be helpful"}, {"role": "user", "content": "Help me"}, {"role": "assistant", "content": "I'm here to help!"}]}
```

### 4. Adding Custom Conversation Formats

To add a new format, extend the `ConversationFormatter` class in `sft_dataset.py`:

```python
class MyCustomFormatter(ConversationFormatter):
    def format_conversation(self, instruction, response, input_text=None, system_prompt=None):
        # Your formatting logic here
        full_text = f"User: {instruction}\nAssistant: {response}"
        instruction_length = len(f"User: {instruction}\nAssistant: ")
        return full_text, instruction_length
```

Then register it in `SFTDataset.__init__`:

```python
if formatter == "my_custom":
    self.formatter = MyCustomFormatter()
```

## Dataset Requirements

Your dataset should have fields matching one of these conventions:

### Option 1: Standard fields
- `instruction` or `prompt` or `question`: The instruction/question
- `response` or `output` or `answer`: The expected response
- `input` (optional): Additional input context
- `system` (optional): System prompt

### Option 2: Custom fields
Modify the `__getitem__` method in `SFTDataset` to extract your specific fields.

## Loss Masking

The trainer automatically masks the loss computation to only calculate loss on response tokens:
- Instruction tokens: loss_mask = 0 (no loss)
- Response tokens: loss_mask = 1 (compute loss)

This ensures the model learns to generate appropriate responses without being penalized for the instruction tokens.

## Architecture

The SFT trainer uses Primus's stage-based registration system:

```
Config (stage: sft)
    ↓
BackendAdapter.load_trainer_class(stage="sft")
    ↓
BackendRegistry.get_trainer_class("megatron", stage="sft")
    ↓
MegatronSFTTrainer (megatron_sft_trainer.py)
    ↓ inherits from
MegatronBaseTrainer (megatron_base_trainer.py)
    ↓ handles
    - Argument injection via parse_args patching
    - ROCm compatibility patches
    - Common Megatron initialization
    ↓ inherits from
BaseTrainer (primus/core/trainer/base_trainer.py)
    - Universal training workflow
    - Patch management
```

### Stage-Based Registration

The trainer registration happens in `primus/backends/megatron/__init__.py`:
```python
BackendRegistry.register_trainer_class(MegatronPretrainTrainer, "megatron")           # stage="pretrain" (default)
BackendRegistry.register_trainer_class(MegatronSFTTrainer, "megatron", "sft")        # stage="sft"
```

The stage is selected via configuration:
- **Explicit**: Set `stage: sft` in config overrides
- **Default**: If not specified, defaults to `stage: pretrain`

## Configuration Parameters

Key SFT-specific parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `sft_dataset_name` | HuggingFace dataset name | `tatsu-lab/alpaca` |
| `sft_conversation_format` | Format type (alpaca, chatml) | `alpaca` |
| `lr` | Learning rate | `1.0e-5` |
| `train_iters` | Number of training iterations | `1000` |
| `seq_length` | Maximum sequence length | `2048` |
| `finetune` | Load pretrained checkpoint | `true` |
| `load` | Path to pretrained checkpoint | `null` |
| `save` | Path to save fine-tuned model | `null` |

## Comparison with Megatron-Bridge

| Feature | Megatron-LM SFT Trainer | Megatron-Bridge |
|---------|------------------------|-----------------|
| Dependency | Direct Megatron-LM | Megatron-Bridge layer |
| Dataset Source | HuggingFace | Recipe-based config |
| Format Support | Extensible (Alpaca, ChatML, etc.) | Bridge-specific |
| Integration | Direct pretrain() call | finetune() wrapper |
| Customization | Full control over forward_step | Higher-level abstraction |

## Testing

Run the unit tests:

```bash
python -m unittest tests.unit_tests.backends.megatron.test_megatron_sft_trainer
```

## Future Extensions

Potential areas for extension:
1. More conversation formats (ShareGPT, OpenAI, etc.)
2. Multi-turn conversation support
3. Special token handling for different tokenizers
4. Dataset streaming for large datasets
5. Custom data processing pipelines
6. LoRA/PEFT integration (separate from dataset handling)

## References

- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- HuggingFace Datasets: https://huggingface.co/docs/datasets
- Alpaca Dataset: https://github.com/tatsu-lab/stanford_alpaca
