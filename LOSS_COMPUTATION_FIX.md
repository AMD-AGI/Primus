# Loss Computation Fix for SFT Trainer

## Problem

The SFT trainer was encountering a runtime error during loss computation:

```
[rank7]: File "/wekafs/xiaoming/dev/Primus/primus/backends/megatron/megatron_sft_trainer.py", line 247, in <lambda>
```

The error occurred when the loss function tried to process the model output.

## Root Cause

The issue was in the `forward_step` function where the model was called:

```python
# BEFORE (incorrect):
output_tensor = model(tokens, position_ids, attention_mask=None, labels=labels)
```

**Problem**: When GPTModel receives the `labels` parameter, it computes loss internally and returns a loss tensor (scalar or low-dimensional) instead of logits.

**Conflict**: The custom `loss_func` expects logits with shape `[batch_size, seq_len, vocab_size]` to:
1. Shift logits and labels for next-token prediction
2. Flatten tensors
3. Compute cross-entropy loss
4. Apply custom masking (only compute loss on response tokens)

**Result**: Type mismatch - the loss function tried to perform tensor operations on a scalar loss value, causing crashes.

## Solution

Remove the `labels` parameter from the model forward call:

```python
# AFTER (correct):
output_tensor = model(tokens, position_ids, attention_mask=None)
```

**Why This Works**:
- Model returns logits with proper shape `[batch_size, seq_len, vocab_size]`
- Custom `loss_func` can process logits correctly
- Masking is applied properly to compute loss only on response tokens
- Separates model forward pass from loss computation (standard Megatron pattern)

## Implementation Details

### Model Forward Pass (Line 191)
```python
# Forward pass through model (without labels to get logits)
# We compute loss separately with custom masking
output_tensor = model(tokens, position_ids, attention_mask=None)
```

### Loss Function (Lines 194-243)
The loss function now correctly receives logits and:
1. Extracts logits from output_tensor
2. Shifts logits and labels for next-token prediction
3. Flattens tensors for cross-entropy computation
4. Computes cross-entropy loss
5. Applies loss mask (only on response tokens)
6. Handles edge case of all tokens masked
7. Averages across data parallel group

### Return Value (Line 245)
```python
return output_tensor, lambda output: loss_func(loss_mask, output)
```

The lambda receives the output tensor (logits) and passes it to loss_func along with the loss_mask.

## Benefits

✅ **Correct Loss Computation** - Loss is computed on logits, not on pre-computed loss
✅ **Custom Masking** - Can apply mask to compute loss only on response tokens
✅ **Standard Pattern** - Aligns with Megatron's standard SFT approach
✅ **Type Safety** - No more type mismatches between scalar loss and 3D logits
✅ **Flexibility** - Full control over loss computation for SFT-specific needs

## Related Files

- `primus/backends/megatron/megatron_sft_trainer.py` - Main fix location
- `primus/backends/megatron/core/datasets/sft_dataset.py` - Provides loss_mask

## Testing

The fix ensures:
1. ✓ Model returns logits (not loss)
2. ✓ loss_func receives proper 3D tensor shape
3. ✓ Shifting operations work correctly
4. ✓ Flattening operations work correctly
5. ✓ Cross-entropy computation succeeds
6. ✓ Loss masking is applied as intended
7. ✓ Training proceeds without errors

## Standard Megatron SFT Pattern

This fix aligns with the standard pattern used in Megatron-LM for supervised fine-tuning:

1. **Forward Pass**: Call model without labels to get logits
2. **Loss Computation**: Compute loss separately with custom logic
3. **Masking**: Apply masks to compute loss only on desired tokens
4. **Return**: Return logits and a loss function

This separation allows for flexible loss computation strategies, which is essential for SFT where we only want to compute loss on assistant responses, not on instructions or system messages.
