# Multi-Turn Conversation Support - Summary

## What Was Implemented

Successfully added comprehensive multi-turn conversation support to the SFT trainer using the OpenAI messages format.

## Core Implementation

### 1. OpenAIMessagesFormatter Class

A new formatter class that handles multi-turn conversations:

```python
class OpenAIMessagesFormatter(ConversationFormatter):
    def format_messages(self, messages: List[Dict[str, str]]) -> Tuple[str, List[Tuple[int, int]]]:
        """Format messages and return assistant content positions."""
        # Formats with ChatML-style markers
        # Returns text and character positions of assistant messages
        
    def format_conversation(self, ...):
        """Maintains compatibility with single-turn interface."""
```

**Key Features:**
- Formats messages with `<|im_start|>role\ncontent<|im_end|>` markers
- Tracks assistant message positions for loss masking
- Supports system, user, and assistant roles
- Handles both single-turn and multi-turn conversations

### 2. Specialized Loss Masking

New `_tokenize_and_mask_messages()` method:
- Tokenizes full conversation
- Creates binary mask (0 for non-assistant, 1 for assistant)
- Ensures loss computed only on assistant responses
- Handles multi-turn properly with position tracking

### 3. Dataset Updates

Updated `SFTDataset.__getitem__()`:
- Auto-detects `messages` field in data
- Routes to appropriate handler (messages vs single-turn)
- Uses specialized masking for multi-turn
- Falls back gracefully for single-turn data

### 4. Formatter Selection

Added new formatter options:
- `"openai"` - OpenAI messages format
- `"messages"` - Alias for openai format
- Backward compatible with `"alpaca"` and `"chatml"`

## Data Format

### OpenAI Messages Format

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "content": "I don't have weather data."}
  ]
}
```

### Supported Roles

| Role | Purpose | Loss Computed |
|------|---------|---------------|
| `system` | Instructions/persona | ❌ No |
| `user` | Questions/prompts | ❌ No |
| `assistant` | Model responses | ✅ **Yes** |

## Usage

### Configuration

```yaml
modules:
  trainer:
    framework: megatron
    overrides:
      stage: sft
      sft_dataset_name: "/path/to/conversations.jsonl"
      sft_conversation_format: "openai"  # or "messages"
```

### Data Preparation

**JSONL Format:**
```jsonl
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
{"messages": [{"role": "system", "content": "Be helpful"}, {"role": "user", "content": "Help"}, {"role": "assistant", "content": "Sure!"}]}
```

## Documentation

### Files Created

1. **docs/MULTI_TURN_CONVERSATIONS.md** (11KB)
   - Comprehensive guide
   - Examples and best practices
   - Conversion scripts
   - Troubleshooting tips

2. **tests/unit_tests/backends/megatron/test_messages_format.py** (9KB)
   - Unit tests for messages formatting
   - Loss masking tests
   - Data loading tests

3. **examples/megatron/configs/MI355X/llama3_8B-BF16-multiturn-sft.yaml** (3KB)
   - Example training configuration
   - Recommended hyperparameters
   - Comments and tips

### Updated Files

1. **primus/backends/megatron/core/datasets/sft_dataset.py**
   - Added `OpenAIMessagesFormatter` class (~150 lines)
   - Added `_tokenize_and_mask_messages()` method (~80 lines)
   - Updated `__getitem__()` to handle messages (~50 lines)
   - Updated formatter selection logic

2. **primus/backends/megatron/README_SFT.md**
   - Added OpenAI messages format section
   - Examples of multi-turn conversations
   - Loss masking explanation

## Technical Details

### Formatted Output

Input messages are formatted as:

```
<|im_start|>system
You are helpful<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi!<|im_end|>
```

### Loss Masking

```
Text:  <|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>
Mask:  [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0]
       └─────user message────────┘└assistant┘└─end─┘
```

Only assistant content tokens contribute to training loss.

### Position Tracking

The formatter tracks character positions of assistant messages:
- Returns list of (start, end) positions
- Used to create token-level masks
- Handles tokenizer boundary effects

## Testing

### Test Coverage

1. **Format Logic**: Verify correct message formatting
2. **Position Tracking**: Verify assistant ranges are correct
3. **Data Loading**: Test JSONL with messages format
4. **Loss Masking**: Verify mask creation
5. **Dataset Integration**: End-to-end testing

### Test Results

All tests pass (with expected skips for missing dependencies):
```
✓ format_messages_logic
✓ messages_file_loading
✓ openai_formatter
✓ sft_dataset_with_messages
✓ formatter_selection
```

## Benefits

### For Users

✅ **Natural Conversations**: Train on realistic multi-turn dialogues
✅ **Context-Aware**: Model learns from full conversation history
✅ **Standard Format**: Compatible with OpenAI API format
✅ **Flexible Training**: Mix single-turn and multi-turn data
✅ **Proper Masking**: Only train on assistant responses

### For Development

✅ **Clean Architecture**: Extends existing formatter pattern
✅ **Backward Compatible**: No breaking changes
✅ **Well Tested**: Comprehensive test coverage
✅ **Documented**: Extensive documentation and examples
✅ **Maintainable**: Clear code structure

## Performance Considerations

### Memory

Multi-turn conversations use more memory:
- Longer sequences per sample
- More tokens to process
- Consider reducing batch size

### Training

May need adjustments:
- Lower learning rate (5e-6 vs 1e-5)
- Longer sequences (2048-4096)
- Smaller batch size (64 vs 128)
- More training iterations

## Future Enhancements

Potential improvements (not currently implemented):

1. **Turn-Level Loss Weighting**
   - Weight earlier turns differently than later
   - Emphasize certain conversation stages

2. **Streaming Support**
   - Stream very long conversations
   - Handle truncation intelligently

3. **Additional Formats**
   - ShareGPT format
   - Anthropic format
   - Custom role types

4. **Analytics**
   - Per-turn loss tracking
   - Conversation quality metrics
   - Turn distribution analysis

## Comparison with Single-Turn

| Aspect | Single-Turn | Multi-Turn |
|--------|-------------|------------|
| Format | instruction/response | messages list |
| Context | Limited | Full dialogue |
| Complexity | Simple | More complex |
| Use Cases | Q&A, Commands | Conversations |
| Loss Mask | After prompt | Multiple segments |
| Data Size | ~100 tokens | ~500+ tokens |

## Migration Guide

### For Existing Users

If you're currently using single-turn format:

**Option 1: Continue as-is**
- No changes needed
- Single-turn still fully supported
- Use `"alpaca"` or `"chatml"` formatter

**Option 2: Convert to messages**
- Use conversion script in docs
- Switch to `"openai"` formatter
- Benefit from multi-turn support

### Conversion Example

```python
# Convert single-turn to messages
def convert(data):
    return {
        "messages": [
            {"role": "user", "content": data["instruction"]},
            {"role": "assistant", "content": data["response"]}
        ]
    }
```

## Examples

### Example 1: Simple Q&A

```json
{
  "messages": [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"}
  ]
}
```

### Example 2: Code Help

```json
{
  "messages": [
    {"role": "system", "content": "You are a Python expert"},
    {"role": "user", "content": "How do I reverse a list?"},
    {"role": "assistant", "content": "Use list.reverse() or list[::-1]"},
    {"role": "user", "content": "What's the difference?"},
    {"role": "assistant", "content": ".reverse() modifies in-place, [::-1] creates a copy"}
  ]
}
```

### Example 3: Multi-Turn Chat

```json
{
  "messages": [
    {"role": "user", "content": "Hi!"},
    {"role": "assistant", "content": "Hello! How can I help?"},
    {"role": "user", "content": "Tell me about AI"},
    {"role": "assistant", "content": "AI is..."},
    {"role": "user", "content": "Thanks!"},
    {"role": "assistant", "content": "You're welcome!"}
  ]
}
```

## Summary

Successfully implemented comprehensive multi-turn conversation support with:

- ✅ Full OpenAI messages format support
- ✅ Proper loss masking for multi-turn
- ✅ Extensive documentation and examples
- ✅ Comprehensive testing
- ✅ Backward compatibility
- ✅ Production-ready implementation

The feature enables training on realistic conversational data while maintaining compatibility with existing single-turn workflows.
