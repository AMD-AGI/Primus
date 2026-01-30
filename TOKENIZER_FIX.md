# Tokenizer Interface Compatibility Fix

## Problem

The SFT dataset was failing with the error:
```
AttributeError: '_HuggingFaceTokenizer' object has no attribute 'convert_tokens_to_ids'
```

This occurred when trying to use Megatron's `_HuggingFaceTokenizer` wrapper with the SFT dataset.

## Root Cause

The original code assumed all tokenizers follow the standard HuggingFace pattern:
```python
tokens = tokenizer.tokenize(text)  # Returns list of string tokens
token_ids = tokenizer.convert_tokens_to_ids(tokens)  # Converts to IDs
```

However, Megatron's `_HuggingFaceTokenizer` wrapper has a different interface:
- The `tokenize()` method may return token IDs directly (list of integers)
- The `convert_tokens_to_ids()` method may not be exposed

## Solution

Added a flexible `_tokenize_text()` helper method that handles multiple tokenizer interfaces:

### 1. Megatron Style (tokenize returns IDs)
```python
result = tokenizer.tokenize(text)
if result and isinstance(result[0], int):
    return result  # Already token IDs
```

### 2. Standard HuggingFace (tokenize + convert)
```python
tokens = tokenizer.tokenize(text)  # Returns strings
token_ids = tokenizer.convert_tokens_to_ids(tokens)
```

### 3. Encode Method (fallback)
```python
token_ids = tokenizer.encode(text, add_special_tokens=False)
```

## Implementation Details

### New Helper Method

```python
def _tokenize_text(self, text: str) -> List[int]:
    """
    Tokenize text and return token IDs.
    
    Handles different tokenizer interfaces:
    - Megatron _HuggingFaceTokenizer (tokenize returns IDs)
    - Standard HuggingFace (tokenize + convert_tokens_to_ids)
    - Encode-based tokenizers (encode method)
    """
    try:
        result = self.tokenizer.tokenize(text)
        
        # Check if already token IDs
        if result and isinstance(result[0], int):
            return result
        
        # Try convert_tokens_to_ids
        if hasattr(self.tokenizer, 'convert_tokens_to_ids'):
            return self.tokenizer.convert_tokens_to_ids(result)
        
        # Try encode as fallback
        if hasattr(self.tokenizer, 'encode'):
            return self.tokenizer.encode(text, add_special_tokens=False)
        
        raise AttributeError("Tokenizer missing required methods")
        
    except (AttributeError, TypeError) as e:
        # Final fallback to encode
        if hasattr(self.tokenizer, 'encode'):
            return self.tokenizer.encode(text, add_special_tokens=False)
        
        raise TypeError(
            f"Tokenizer must have either 'encode()' method or 'tokenize()' "
            f"method that returns token IDs. Got: {type(self.tokenizer)}"
        )
```

### Updated Methods

All tokenization calls now use the helper:

**Before:**
```python
tokens = self.tokenizer.tokenize(text)
token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
```

**After:**
```python
token_ids = self._tokenize_text(text)
```

Updated in:
- `_tokenize_and_mask()` - Single-turn tokenization
- `_tokenize_and_mask_messages()` - Multi-turn tokenization

## Benefits

1. **Compatibility**: Works with Megatron, HuggingFace, and custom tokenizers
2. **Robustness**: Graceful fallback through multiple interface patterns
3. **Error Messages**: Clear error messages if tokenizer is incompatible
4. **No Breaking Changes**: Existing code with standard tokenizers continues to work

## Testing

The fix was tested with three tokenizer patterns:

1. **Megatron-style**: `tokenize()` returns integers directly
2. **Standard**: `tokenize()` + `convert_tokens_to_ids()`
3. **Encode-based**: `encode()` method only

All patterns now work correctly.

## Files Changed

- `primus/backends/megatron/core/datasets/sft_dataset.py`
  - Added `_tokenize_text()` method
  - Updated `_tokenize_and_mask()` 
  - Updated `_tokenize_and_mask_messages()`
  - Removed direct calls to `convert_tokens_to_ids()`

## Related Issues

This fix resolves the runtime error when using:
- Megatron's `_HuggingFaceTokenizer` wrapper
- Custom tokenizers with non-standard interfaces
- Any tokenizer where `tokenize()` returns IDs directly

## Backward Compatibility

✅ **Fully backward compatible** - Standard HuggingFace tokenizers continue to work as before.
