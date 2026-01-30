# Offline Dataset Support for SFT Trainer

## Overview

The SFT trainer now supports loading datasets from local JSONL and JSON files, enabling offline training without requiring internet access or HuggingFace Hub connectivity.

## Quick Start

### 1. Prepare Your Data

Create a JSONL (JSON Lines) file where each line is a separate JSON object:

**data.jsonl:**
```jsonl
{"instruction": "What is Python?", "response": "Python is a high-level programming language."}
{"instruction": "Explain machine learning.", "response": "Machine learning is a subset of AI..."}
{"instruction": "What is a neural network?", "response": "A neural network is a computing system..."}
```

Or use a JSON array file:

**data.json:**
```json
[
    {"instruction": "What is Python?", "response": "Python is a high-level programming language."},
    {"instruction": "Explain machine learning.", "response": "Machine learning is a subset of AI..."}
]
```

### 2. Update Your Configuration

**config.yaml:**
```yaml
modules:
  trainer:
    framework: megatron
    config: sft_trainer.yaml
    model: llama3_8B.yaml
    
    overrides:
      stage: sft
      
      # Use local JSONL file instead of HuggingFace dataset
      sft_dataset_name: "/path/to/your/data.jsonl"
      
      # Or JSON file
      # sft_dataset_name: "/path/to/your/data.json"
      
      sft_conversation_format: "alpaca"
      # ... other training parameters
```

### 3. Run Training

The trainer will automatically detect the local file and load it:

```bash
python -m primus.cli.subcommands.train --config config.yaml
```

## Supported Field Names

The dataset loader supports multiple field name conventions for flexibility:

| Purpose | Field Names (in order of priority) |
|---------|-------------------------------------|
| Instruction | `instruction`, `prompt`, `question` |
| Response | `response`, `output`, `answer` |
| Input Context (optional) | `input` |
| System Prompt (optional) | `system` |

## Examples

### Example 1: Basic JSONL Format

```jsonl
{"instruction": "Translate to French: Hello", "response": "Bonjour"}
{"instruction": "What is 5 + 3?", "response": "5 + 3 equals 8."}
```

### Example 2: With Input Context

```jsonl
{"instruction": "Summarize this text", "input": "Long text here...", "response": "Summary here..."}
{"instruction": "Answer the question", "input": "Context paragraph...", "response": "The answer is..."}
```

### Example 3: With System Prompt

```jsonl
{"system": "You are a helpful assistant.", "instruction": "Help me", "response": "I'm here to help!"}
```

### Example 4: Alternative Field Names

```jsonl
{"prompt": "What is AI?", "output": "AI stands for Artificial Intelligence."}
{"question": "Define ML", "answer": "ML is Machine Learning."}
```

## File Format Requirements

### JSONL Format
- One JSON object per line
- Each line must be valid JSON
- Empty lines are skipped
- UTF-8 encoding

### JSON Format
- Single JSON array at the root
- Each array element is a data sample
- UTF-8 encoding

## Error Handling

The loader provides clear error messages:

1. **File not found:**
   ```
   FileNotFoundError: JSONL file not found: /path/to/file.jsonl
   ```

2. **Invalid JSON:**
   ```
   JSONDecodeError: Invalid JSON on line 5 in /path/to/file.jsonl
   ```

3. **Wrong JSON structure:**
   ```
   ValueError: JSON file must contain a list of objects, got <class 'dict'>
   ```

## Dataset Splits

For local files, splits are not automatically available. To use train/validation/test splits:

1. Create separate files:
   - `train.jsonl`
   - `validation.jsonl`
   - `test.jsonl`

2. Load them separately by pointing to different files in your configuration.

## Performance Notes

- Local files are loaded entirely into memory as HuggingFace Dataset objects
- For very large datasets (>1GB), consider using HuggingFace's Dataset.from_generator() or memory-mapped formats
- JSONL format is preferred for large datasets as it's easier to stream

## Comparison: Online vs Offline

| Feature | HuggingFace Hub | Local JSONL/JSON |
|---------|-----------------|------------------|
| Internet Required | ✅ Yes | ❌ No |
| Automatic Splits | ✅ Yes | ❌ Manual |
| Streaming Support | ✅ Yes | ⚠️ Limited |
| Setup Complexity | 🟢 Low | 🟢 Low |
| Custom Data | 🟡 Upload Required | 🟢 Direct Use |
| Data Privacy | 🟡 Uploaded to Hub | 🟢 Stays Local |

## Troubleshooting

**Q: My file isn't being detected as local**
- Ensure the file path is absolute or the file exists
- Check file extension is `.jsonl` or `.json`

**Q: Getting "No module named 'datasets'"**
- Install HuggingFace datasets: `pip install datasets`
- This is required even for local files (for internal format compatibility)

**Q: Data fields not being recognized**
- Check that your JSON objects use supported field names
- See "Supported Field Names" table above

**Q: Want to use custom field names?**
- Currently requires code modification
- Future enhancement: configurable field mapping

## Advanced Usage

### Mixed Sources

You can train on HuggingFace datasets and validate on local files:

```python
# In your training script
train_dataset = SFTDataset(
    dataset_name="tatsu-lab/alpaca",  # Online
    tokenizer=tokenizer,
    max_seq_length=2048,
    split="train"
)

val_dataset = SFTDataset(
    dataset_name="/path/to/validation.jsonl",  # Offline
    tokenizer=tokenizer,
    max_seq_length=2048
)
```

### Programmatic Loading

```python
from primus.backends.megatron.core.datasets.sft_dataset import load_jsonl_file

# Load data
data = load_jsonl_file("/path/to/data.jsonl")

# Inspect
print(f"Loaded {len(data)} samples")
print(f"First sample: {data[0]}")

# Filter or transform as needed
filtered_data = [d for d in data if len(d['response']) > 10]
```

## Future Enhancements

Planned improvements:
- ⏳ CSV format support
- ⏳ Parquet format support
- ⏳ Configurable field name mapping
- ⏳ Streaming for large files
- ⏳ Automatic train/val/test splitting from single file
