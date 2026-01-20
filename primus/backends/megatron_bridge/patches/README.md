# Megatron-Bridge Patches

This directory contains patches required for Megatron-Bridge compatibility with third-party libraries.

## Patches

### 1. `transformers_modeling_utils.py`

**Purpose**: Add missing `pytorch_utils` imports to transformers' `modeling_utils.py`

**Target**: Hugging Face Transformers v4.57-release

**Why needed**: 
- Megatron-Bridge depends on several functions from `transformers.pytorch_utils`
- These functions are not explicitly imported in `modeling_utils.py` in v4.57-release
- Without this patch, Megatron-Bridge will fail with `ImportError`

**Functions added**:
```python
from .pytorch_utils import (
    Conv1D,                          # 1D convolution layer (used in GPT-2, etc.)
    apply_chunking_to_forward,       # Memory-efficient chunked forward pass
    find_pruneable_heads_and_indices,  # Find prunable attention heads
    id_tensor_storage,               # Get tensor storage ID
    prune_conv1d_layer,             # Prune 1D convolution layers
    prune_layer,                    # Prune generic layers
    prune_linear_layer,             # Prune linear layers
)
```

**Usage**:
```bash
# From command line
python transformers_modeling_utils.py /path/to/transformers

# In setup script
python "${PRIMUS_ROOT}/primus/backends/megatron_bridge/patches/transformers_modeling_utils.py" "$TRANSFORMERS_DIR"
```

**Features**:
- ✅ Idempotent: Safe to run multiple times
- ✅ Validates file existence before patching
- ✅ Clear output messages
- ✅ Proper error handling

## Adding New Patches

When adding new patches:

1. Create a standalone Python script with:
   - Clear docstring explaining the purpose
   - Idempotent implementation (safe to re-run)
   - Proper error handling
   - Command-line interface

2. Update this README with:
   - Patch name and purpose
   - Target library/version
   - Why it's needed
   - Usage example

3. Update setup scripts to use the new patch

## Testing Patches

```bash
# Test transformers patch
cd /tmp
git clone https://github.com/huggingface/transformers.git test-transformers
cd test-transformers
git checkout v4.57-release

# Apply patch
python /path/to/Primus/primus/backends/megatron_bridge/patches/transformers_modeling_utils.py .

# Verify
grep -A 8 "from .pytorch_utils import" src/transformers/modeling_utils.py
```

## Maintenance

- Keep patches minimal and focused
- Document why each patch is needed
- Test patches against specific library versions
- Consider upstreaming fixes when appropriate
