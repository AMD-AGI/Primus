# Position IDs Fix for SFT Trainer

## Problem

The SFT trainer was failing at runtime with the following error:

```
TypeError: GPTModel.forward() missing 1 required positional argument: 'position_ids'
```

## Root Cause

The `forward_step` function in `megatron_sft_trainer.py` was calling the model without providing the required `position_ids` parameter:

```python
# Before (incorrect):
output_tensor = model(tokens, attention_mask=None, labels=labels)
```

Megatron-LM's `GPTModel.forward()` requires `position_ids` as a positional argument to properly encode token positions in the sequence.

## Solution

Added position_ids generation before the model forward call:

```python
# Generate position_ids for the model
batch_size, seq_len = tokens.size()
position_ids = torch.arange(seq_len, dtype=torch.long, device=tokens.device)
position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

# Forward pass with position_ids
output_tensor = model(tokens, position_ids, attention_mask=None, labels=labels)
```

## Implementation Details

### Position IDs Structure

- **Shape**: `[batch_size, seq_len]`
- **Content**: Each row contains `[0, 1, 2, ..., seq_len-1]`
- **Data type**: `torch.long`
- **Device**: Same as input tokens (CUDA)

### Example

For a batch of 2 samples with sequence length 5:
```python
position_ids = tensor([
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3, 4]
], device='cuda:0', dtype=torch.int64)
```

### Why Position IDs Are Needed

Position IDs tell the model where each token is located in the sequence. This is essential for:

1. **Positional Embeddings**: The model adds positional information to token embeddings
2. **Attention Masks**: Used in conjunction with attention mechanisms
3. **Sequence Understanding**: Helps the model understand token order and relationships

## Testing

The fix ensures:
- ✅ No more TypeError about missing position_ids
- ✅ Correct position encoding for all tokens
- ✅ Compatible with Megatron-LM's GPTModel interface
- ✅ Minimal code change with no side effects

## Related Code

- **File**: `primus/backends/megatron/megatron_sft_trainer.py`
- **Function**: `forward_step()` (lines 185-191)
- **Change**: Added 4 lines of position_ids generation code

## References

- Megatron-LM GPTModel documentation
- Standard transformer position encoding practices
- PyTorch tensor operations: `torch.arange()`, `unsqueeze()`, `expand()`
