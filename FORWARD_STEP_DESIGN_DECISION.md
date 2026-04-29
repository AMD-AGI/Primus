# Forward Step Design Decision

## Question
"Why not use `from megatron.bridge.training.vlm_step import forward_step` as the forward_step in megatron_sft_trainer?"

## Answer

We chose to implement a custom `forward_step` in the SFT trainer rather than reusing Megatron-Bridge's `forward_step` for the following reasons:

### 1. **Original Design Requirement**

The initial requirement explicitly stated:
> "不要import Megatron-Bridge 的代码"  
> (Don't import Megatron-Bridge code)

The goal was to create a **direct Megatron-LM integration** without external dependencies.

### 2. **Dependency Independence**

**Current Situation:**
- Megatron-Bridge is not installed in the environment
- The `third_party/Megatron-Bridge` directory is empty
- Using Megatron-Bridge would add an external dependency

**Benefits of Independence:**
- ✅ Simpler deployment (fewer dependencies)
- ✅ Direct integration with Megatron-LM
- ✅ No version conflicts with Megatron-Bridge
- ✅ Clearer separation of concerns

### 3. **SFT-Specific Requirements**

The SFT trainer has specific needs that may not be fully covered by a general-purpose `forward_step`:

**SFT-Specific Features:**
- **Custom Loss Masking**: Only compute loss on response tokens, not instruction tokens
- **Flexible Data Format**: Support for multiple conversation formats (Alpaca, ChatML, OpenAI messages)
- **Offline Dataset Support**: Load from local JSONL files
- **Field Flexibility**: Support various field naming conventions
- **Multi-turn Conversations**: Special handling for conversation context

**Our Custom Implementation:**
```python
# Extract tokens with custom fields
tokens = batch['input_ids'].long().cuda()
labels = batch['labels'].long().cuda()
loss_mask = batch['loss_mask'].float().cuda()  # SFT-specific masking

# Custom loss computation with masking
def loss_func(loss_mask, output_tensor):
    # Apply mask (only compute loss on response tokens)
    masked_losses = losses * shift_mask
    # ... SFT-specific logic
```

### 4. **Code Comparison**

**Megatron-Bridge vlm_step.forward_step:**
- Designed for vision-language models (VLM)
- General-purpose forward pass
- May not have SFT-specific masking logic
- Tied to Megatron-Bridge's data format and conventions

**Our Custom forward_step:**
- Designed specifically for SFT
- Custom loss masking for instruction tuning
- Direct control over data format
- Optimized for text-only models
- ~90 lines of well-documented code

### 5. **Maintainability**

**With Custom Implementation:**
- ✅ Clear ownership and control
- ✅ Easy to customize for SFT needs
- ✅ Self-contained and documented
- ✅ No hidden dependencies or surprises

**With Megatron-Bridge Dependency:**
- ❌ External code changes could break our trainer
- ❌ Need to track Megatron-Bridge versions
- ❌ Less control over behavior
- ❌ VLM-specific code may be unnecessary for text SFT

### 6. **Consistency with Architecture**

The Primus framework has both:
1. **Megatron-Bridge trainers** (in `primus/backends/megatron_bridge/`)
   - These DO use `megatron.bridge.training.vlm_step.forward_step`
   - Designed for users who want Megatron-Bridge features

2. **Direct Megatron-LM trainers** (in `primus/backends/megatron/`)
   - These use custom implementations
   - Designed for users who want direct Megatron-LM integration
   - MegatronSFTTrainer falls into this category

## When Should You Use Each?

### Use MegatronSFTTrainer (Direct Megatron-LM):
- ✅ Want minimal dependencies
- ✅ Need SFT-specific customizations
- ✅ Working with text-only models
- ✅ Want full control over training loop
- ✅ Prefer direct Megatron-LM integration

### Use MegatronBridgePosttrainTrainer (Megatron-Bridge):
- ✅ Already using Megatron-Bridge
- ✅ Want Megatron-Bridge features (recipe system, etc.)
- ✅ Working with VLMs (vision-language models)
- ✅ Prefer Megatron-Bridge abstractions
- ✅ Need Megatron-Bridge's post-training utilities

## Could We Switch to Using Megatron-Bridge?

**Yes, but it would require:**

1. **Installing Megatron-Bridge:**
   ```bash
   # Currently third_party/Megatron-Bridge is empty
   # Would need to clone and install Megatron-Bridge
   ```

2. **Changing the Import:**
   ```python
   from megatron.bridge.training.vlm_step import forward_step
   ```

3. **Adapting Our Data Format:**
   - Megatron-Bridge expects specific data structure
   - May need to modify our dataset classes

4. **Testing Compatibility:**
   - Ensure vlm_step.forward_step works for text SFT
   - Verify loss masking behavior matches our needs

5. **Updating Documentation:**
   - Document Megatron-Bridge dependency
   - Update installation instructions

## Recommendation

**Keep the custom implementation** because:

1. ✅ Aligns with original design goal (no Megatron-Bridge dependency)
2. ✅ Provides SFT-specific features and control
3. ✅ Simpler deployment (fewer dependencies)
4. ✅ Consistent with direct Megatron-LM architecture
5. ✅ Well-tested and working code (~90 lines)

**If you want Megatron-Bridge features:**
- Use `MegatronBridgePosttrainTrainer` instead
- It already uses `megatron.bridge.training.vlm_step.forward_step`
- Provides the full Megatron-Bridge feature set

## Summary

The decision to implement a custom `forward_step` was intentional and based on:
- **Independence**: No Megatron-Bridge dependency
- **Control**: SFT-specific customizations
- **Simplicity**: Direct Megatron-LM integration
- **Consistency**: Matches the "direct integration" architecture

However, Primus provides **both options**:
- `MegatronSFTTrainer` (direct, custom) ← Current implementation
- `MegatronBridgePosttrainTrainer` (Megatron-Bridge-based) ← Alternative

Choose based on your needs and preferences!
