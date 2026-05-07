# Multi-Turn Conversation Support Guide

## Overview

The SFT trainer now supports multi-turn conversations using the OpenAI messages format. This enables training on complex dialogues where the model learns from multiple exchanges between user and assistant.

## Quick Start

### 1. Prepare Your Data

Create a JSONL file with messages format:

**conversations.jsonl:**
```jsonl
{"messages": [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi! How can I help you today?"}]}
{"messages": [{"role": "system", "content": "You are a helpful coding assistant."}, {"role": "user", "content": "How do I reverse a string in Python?"}, {"role": "assistant", "content": "You can reverse a string using slicing: text[::-1]"}, {"role": "user", "content": "Can you show an example?"}, {"role": "assistant", "content": "Sure! Here's an example:\n\ntext = 'hello'\nreversed_text = text[::-1]\nprint(reversed_text)  # Output: 'olleh'"}]}
{"messages": [{"role": "user", "content": "Tell me a joke"}, {"role": "assistant", "content": "Why did the chicken cross the road?"}, {"role": "user", "content": "Why?"}, {"role": "assistant", "content": "To get to the other side!"}]}
```

### 2. Configure Training

**config.yaml:**
```yaml
modules:
  trainer:
    framework: megatron
    config: sft_trainer.yaml
    model: llama3_8B.yaml
    
    overrides:
      stage: sft
      
      # Use messages format for multi-turn conversations
      sft_dataset_name: "/path/to/conversations.jsonl"
      sft_conversation_format: "openai"  # or "messages"
      
      # Training parameters
      train_iters: 1000
      global_batch_size: 128
      lr: 1.0e-5
```

### 3. Run Training

```bash
python -m primus.cli.subcommands.train --config config.yaml
```

## Message Format Specification

### Basic Structure

Each data sample must have a `messages` field containing a list of message objects:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"}
  ]
}
```

### Supported Roles

| Role | Description | Loss Computed |
|------|-------------|---------------|
| `system` | System prompt/instructions | ❌ No |
| `user` | User messages/questions | ❌ No |
| `assistant` | Assistant responses | ✅ **Yes** |

**Important**: Loss is computed **only** on assistant message content. System messages and user messages are used for context but don't contribute to the loss.

### Message Object Fields

Each message object requires:
- `role`: One of "system", "user", or "assistant"
- `content`: The text content of the message

## Examples

### Example 1: Simple Single-Turn

```json
{
  "messages": [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."}
  ]
}
```

**Formatted Output:**
```
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
2+2 equals 4.<|im_end|>
```

**Loss Mask:** Only `2+2 equals 4.` contributes to loss.

### Example 2: With System Prompt

```json
{
  "messages": [
    {"role": "system", "content": "You are a math tutor. Explain step by step."},
    {"role": "user", "content": "How do I solve x + 5 = 10?"},
    {"role": "assistant", "content": "To solve x + 5 = 10:\n1. Subtract 5 from both sides\n2. x = 10 - 5\n3. x = 5"}
  ]
}
```

**Loss Mask:** Only the assistant's explanation contributes to loss.

### Example 3: Multi-Turn Conversation

```json
{
  "messages": [
    {"role": "system", "content": "You are a friendly chatbot."},
    {"role": "user", "content": "Hi there!"},
    {"role": "assistant", "content": "Hello! How are you doing today?"},
    {"role": "user", "content": "I'm good, thanks! Can you help me with something?"},
    {"role": "assistant", "content": "Of course! I'd be happy to help. What do you need?"},
    {"role": "user", "content": "What's the weather like?"},
    {"role": "assistant", "content": "I don't have access to real-time weather data, but I can help you find weather information if you tell me your location."}
  ]
}
```

**Loss Mask:** All three assistant responses contribute to loss:
1. "Hello! How are you doing today?"
2. "Of course! I'd be happy to help. What do you need?"
3. "I don't have access to real-time weather data..."

### Example 4: Code Assistant

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert Python programmer."},
    {"role": "user", "content": "Write a function to check if a number is prime"},
    {"role": "assistant", "content": "Here's a Python function to check if a number is prime:\n\n```python\ndef is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n```"},
    {"role": "user", "content": "Can you explain how it works?"},
    {"role": "assistant", "content": "Sure! The function works as follows:\n1. Numbers less than 2 are not prime\n2. We check divisibility from 2 to sqrt(n)\n3. If any number divides evenly, it's not prime\n4. If no divisors are found, it's prime"}
  ]
}
```

## Formatted Output

The messages are formatted using ChatML-style markers:

```
<|im_start|>system
{system_content}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
{assistant_content}<|im_end|>
```

This format is commonly used by models like:
- GPT-3.5/GPT-4
- Mistral
- Qwen
- Many open-source chat models

## Loss Masking Details

### How It Works

1. **Full Text Tokenization**: The entire conversation is tokenized as one sequence
2. **Position Tracking**: Assistant message positions are tracked during formatting
3. **Mask Creation**: A binary mask is created:
   - `0` for system and user messages (no loss)
   - `1` for assistant messages (compute loss)
4. **Training**: Model is trained to predict assistant responses given the context

### Example Mask

For conversation:
```
<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>
```

Loss mask (conceptual):
```
[0,0,0,0,0,  1,1,1,1,  0,0]
 user tokens  assistant  end
             content
```

## Converting Existing Data

### From Single-Turn to Messages

**Before (single-turn):**
```json
{"instruction": "What is AI?", "response": "AI is Artificial Intelligence."}
```

**After (messages):**
```json
{
  "messages": [
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "AI is Artificial Intelligence."}
  ]
}
```

### Conversion Script

```python
import json

def convert_to_messages(input_file, output_file):
    """Convert single-turn to messages format."""
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            data = json.loads(line)
            
            messages = []
            
            # Add system prompt if exists
            if 'system' in data:
                messages.append({
                    "role": "system",
                    "content": data['system']
                })
            
            # Add user message
            messages.append({
                "role": "user",
                "content": data.get('instruction', data.get('prompt', ''))
            })
            
            # Add assistant response
            messages.append({
                "role": "assistant",
                "content": data.get('response', data.get('output', ''))
            })
            
            f_out.write(json.dumps({"messages": messages}) + '\n')

# Usage
convert_to_messages('single_turn.jsonl', 'multi_turn.jsonl')
```

## Best Practices

### 1. System Prompts

Use system prompts to set the assistant's behavior:
```json
{"role": "system", "content": "You are a helpful, harmless, and honest assistant."}
```

### 2. Context Length

Be mindful of context length with multi-turn conversations:
- Each turn adds to the sequence length
- Monitor truncation in logs
- Adjust `max_seq_length` if needed

### 3. Data Quality

For multi-turn conversations:
- ✅ Ensure coherent dialogue flow
- ✅ Include relevant context in earlier turns
- ✅ Maintain consistent persona across turns
- ❌ Avoid abrupt topic changes
- ❌ Don't include redundant information

### 4. Training Tips

- Start with smaller learning rate (1e-5 to 5e-6)
- Monitor validation loss per turn if possible
- Consider curriculum learning (single-turn → multi-turn)

## Comparison: Single-Turn vs Multi-Turn

| Feature | Single-Turn | Multi-Turn (Messages) |
|---------|-------------|----------------------|
| Format | `instruction`/`response` | `messages` list |
| Turns | 1 | Multiple |
| Context | Limited | Full conversation |
| Use Case | Q&A, Commands | Dialogues, Conversations |
| Loss Computation | After response marker | All assistant messages |
| Data Complexity | Simple | More complex |
| Training Difficulty | Easier | More challenging |

## Troubleshooting

### Issue: Loss not computed correctly

**Solution**: Verify your messages format:
```python
# Check your data
import json
with open('data.jsonl', 'r') as f:
    sample = json.loads(f.readline())
    print("Has messages?", "messages" in sample)
    print("Number of messages:", len(sample.get("messages", [])))
    print("Roles:", [m["role"] for m in sample.get("messages", [])])
```

### Issue: High loss or poor convergence

**Possible causes**:
1. Too many turns (exceeding context length)
2. Inconsistent dialogue quality
3. Learning rate too high

**Solutions**:
- Reduce number of turns per sample
- Filter low-quality conversations
- Lower learning rate to 1e-6

### Issue: Model only responds with first turn

**Solution**: Ensure you're training with `sft_conversation_format: "openai"`:
```yaml
sft_conversation_format: "openai"  # NOT "alpaca" or "chatml"
```

## Advanced Usage

### Mixed Training

You can train on both single-turn and multi-turn data:

**Option 1: Separate datasets**
```yaml
# Use single-turn for some epochs, then multi-turn
```

**Option 2: Convert all to messages format**
```python
# Convert single-turn to messages (see conversion script above)
```

### Custom Loss Weighting

Future enhancement: Weight different turns differently
```python
# Currently: all assistant messages weighted equally
# Future: first turn = 1.0, second = 0.8, third = 0.6, etc.
```

## Performance Considerations

### Memory Usage

Multi-turn conversations use more memory:
- More tokens per sample
- Longer sequences
- Consider reducing batch size

### Training Speed

Multi-turn training may be slower:
- Longer sequences take more time
- More computation per sample
- But fewer samples needed for same data coverage

## Summary

Multi-turn conversation support enables:
- ✅ Training on complex dialogues
- ✅ Learning conversational patterns
- ✅ Context-aware responses
- ✅ Better chatbot behavior
- ✅ Compatible with OpenAI format

The implementation automatically handles loss masking to ensure the model learns only from assistant responses while using the full conversation context.
