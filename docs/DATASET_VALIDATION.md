# Dataset Validation and Statistics

## Overview

The `GPTSFTChatDataset` now includes automatic dataset validation and statistics reporting when initialized. This feature helps identify potential issues early and provides insights into your training data.

## Features

### Automatic Validation

When you create a `GPTSFTChatDataset` instance, it automatically:
1. Samples up to 100 examples from the dataset
2. Computes various statistics
3. Reports findings via logging
4. Warns about potential issues

### Statistics Reported

#### 1. **Sequence Length Statistics**
- Average, median, min, max, and standard deviation
- Comparison with `max_seq_length`
- Truncation rate warning (if samples exceed max length)

Example output:
```
Sequence Length Statistics:
  Average: 342.5 tokens
  Median: 298.0 tokens
  Min: 45 tokens
  Max: 1024 tokens
  Std Dev: 156.3 tokens
  Max allowed: 512 tokens
  ⚠️  15/100 samples (15.0%) will be truncated!
```

#### 2. **Loss Mask Statistics**
- Average, median, min, max loss ratio
- Loss ratio = (tokens with loss) / (total tokens)
- Warnings for suspicious ratios

Example output:
```
Loss Mask Statistics:
  Average loss ratio: 45.30%
  Median loss ratio: 43.20%
  Min loss ratio: 12.50%
  Max loss ratio: 89.30%
```

**Warnings:**
- **Low ratio (<10%)**: Most tokens are masked, training may be inefficient
- **High ratio (>90%)**: Few tokens masked, may include instruction tokens

#### 3. **Error Detection**
- Reports errors encountered during sampling
- Shows first 5 errors with details
- Continues execution even if validation fails

## Usage

### Basic Usage

No changes needed - validation runs automatically:

```python
from primus.backends.megatron.sft import GPTSFTChatDataset

dataset = GPTSFTChatDataset(
    file_path="data/training.jsonl",
    tokenizer=tokenizer,
    max_seq_length=4096,
    pad_to_max_length=False,
    max_num_samples=10000,
    seed=1234,
)
# Validation report is automatically printed
```

### Manual Validation

You can also call validation manually:

```python
dataset = GPTSFTChatDataset(...)
dataset.validate_and_report()  # Run validation again
```

### Disable Validation (Not Recommended)

If you need to disable validation for some reason:

```python
# Temporarily suppress logging
import logging
logging.getLogger('primus.backends.megatron.sft.gpt_sft_chat_dataset').setLevel(logging.WARNING)

dataset = GPTSFTChatDataset(...)
# Validation runs but output is suppressed
```

## Example Output

Here's a complete example of what you'll see:

```
2025-01-30 10:30:15 - INFO - Loading SFT dataset from data/training.jsonl
2025-01-30 10:30:16 - INFO - Validating dataset with 100 samples...
2025-01-30 10:30:18 - INFO - ================================================================================
2025-01-30 10:30:18 - INFO - Dataset Validation Report
2025-01-30 10:30:18 - INFO - ================================================================================
2025-01-30 10:30:18 - INFO - Total samples: 50,000
2025-01-30 10:30:18 - INFO - Samples validated: 100/100
2025-01-30 10:30:18 - INFO - 
2025-01-30 10:30:18 - INFO - Sequence Length Statistics:
2025-01-30 10:30:18 - INFO -   Average: 342.5 tokens
2025-01-30 10:30:18 - INFO -   Median: 298.0 tokens
2025-01-30 10:30:18 - INFO -   Min: 45 tokens
2025-01-30 10:30:18 - INFO -   Max: 1024 tokens
2025-01-30 10:30:18 - INFO -   Std Dev: 156.3 tokens
2025-01-30 10:30:18 - INFO -   Max allowed: 4096 tokens
2025-01-30 10:30:18 - INFO - 
2025-01-30 10:30:18 - INFO - Loss Mask Statistics:
2025-01-30 10:30:18 - INFO -   Average loss ratio: 45.30%
2025-01-30 10:30:18 - INFO -   Median loss ratio: 43.20%
2025-01-30 10:30:18 - INFO -   Min loss ratio: 12.50%
2025-01-30 10:30:18 - INFO -   Max loss ratio: 89.30%
2025-01-30 10:30:18 - INFO - ================================================================================
```

## Interpreting Results

### Sequence Length

**Good:**
- Most samples fit within `max_seq_length`
- Reasonable variance (not all same length)
- Average around 50-70% of max length

**Concerning:**
- High truncation rate (>20%)
- Very low average (<10% of max)
- Very high standard deviation

**Action:** If truncation rate is high, consider:
1. Increasing `max_seq_length`
2. Filtering/preprocessing long examples
3. Using packed sequences

### Loss Ratio

**Good:**
- Average around 30-60% (typical for instruction tuning)
- Reasonable variance across samples

**Concerning:**
- Very low (<10%): Training inefficient
- Very high (>90%): May train on instructions

**Action:** If ratio is problematic:
1. Check chat template configuration
2. Verify `assistant_tokens_mask` is working
3. Review data format

## Technical Details

### Implementation

The `validate_and_report()` method:
1. Samples up to 100 examples randomly
2. Loads each sample through `__getitem__`
3. Collects statistics in a try-catch to handle errors
4. Uses numpy for efficient statistics computation
5. Logs results using Python's logging module

### Performance

- Validation samples 100 examples (configurable in code)
- Takes 1-5 seconds for most datasets
- Runs only once during initialization
- Minimal memory overhead

### Error Handling

The validation is designed to never crash your training:
- All operations wrapped in try-catch
- Errors logged but don't stop initialization
- If validation fails completely, logs warning and continues

## Advanced Configuration

### Changing Sample Size

Edit `gpt_sft_chat_dataset.py`:

```python
def validate_and_report(self):
    # Change 100 to desired number
    sample_size = min(200, total_samples)  # Sample 200 instead of 100
```

### Adding Custom Statistics

You can extend `validate_and_report()` to collect additional metrics:

```python
def validate_and_report(self):
    # ... existing code ...
    
    # Add custom statistics
    vocab_usage = []
    for idx in sample_indices:
        sample = self[int(idx)]
        unique_tokens = len(set(sample['tokens'].tolist()))
        vocab_usage.append(unique_tokens)
    
    logger.info(f"Vocabulary Usage:")
    logger.info(f"  Avg unique tokens: {np.mean(vocab_usage):.1f}")
```

## Troubleshooting

### "Dataset is empty!"
- Check that your data file exists and has content
- Verify the file path is correct

### High truncation rate
- Increase `max_seq_length` in config
- Or filter your dataset to remove very long examples

### Very low loss ratio
- Check that your chat template is configured correctly
- Verify `assistant_tokens_mask` is being generated
- Review data format (should be OpenAI messages or ShareGPT)

### Validation errors
- Check the error messages for specific issues
- Verify data format is consistent across all examples
- Ensure tokenizer is properly initialized

## Future Enhancements

Planned improvements:
- [ ] Configurable sample size via parameter
- [ ] Vocabulary coverage analysis
- [ ] Token distribution histograms
- [ ] Export statistics to JSON/CSV
- [ ] Comparison across multiple datasets
- [ ] Integration with TensorBoard

## References

- [GPTSFTChatDataset Source](../primus/backends/megatron/sft/gpt_sft_chat_dataset.py)
- [HuggingFace Chat Templates](https://huggingface.co/docs/transformers/chat_templating)
- [Megatron-LM Documentation](https://github.com/NVIDIA/Megatron-LM)
