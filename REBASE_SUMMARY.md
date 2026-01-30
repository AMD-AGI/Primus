# Rebase Summary: feat/staged-trainer-registry

## Overview

Successfully rebased the SFT trainer implementation to align with the `feat/staged-trainer-registry` branch. The new approach uses a stage-based trainer registration system instead of module name parsing.

## What Changed

### Before (Module-Based)
```python
# Old approach - used module names for selection
BackendRegistry.register_trainer_class("megatron", MegatronPretrainTrainer)
BackendRegistry.register_trainer_class("megatron_sft", MegatronSFTTrainer)

# Adapter parsed module name
def load_trainer_class(self, module_config):
    module_name = module_config.name if module_config else None
    if module_name == "sft_trainer":
        return BackendRegistry.get_trainer_class("megatron_sft")
    else:
        return BackendRegistry.get_trainer_class("megatron")
```

### After (Stage-Based)
```python
# New approach - explicit stage parameter
BackendRegistry.register_trainer_class(MegatronPretrainTrainer, "megatron")           # stage="pretrain" (default)
BackendRegistry.register_trainer_class(MegatronSFTTrainer, "megatron", "sft")        # stage="sft"

# Adapter uses stage parameter
def load_trainer_class(self, stage="pretrain"):
    return BackendRegistry.get_trainer_class(self.framework, stage=stage)
```

## Key Benefits

1. **Cleaner API**: Stage parameter is explicit and consistent across all backends
2. **Better Separation**: Training stages (pretrain, sft, etc.) are first-class concepts
3. **More Flexible**: Easy to add new stages (rlhf, dpo, etc.) without name conflicts
4. **Consistent**: Same pattern works for all backends (megatron, torchtitan, megatron_bridge)

## Changes Made

### Core Framework
- `primus/core/backend/backend_registry.py`: Updated to use `(backend, stage)` tuple keys
- `primus/core/backend/backend_adapter.py`: Changed `load_trainer_class()` signature

### Megatron Backend
- `primus/backends/megatron/__init__.py`: Registered both trainers with stages
- `primus/backends/megatron/megatron_adapter.py`: Simplified to use stage parameter

### Other Backends
- `primus/backends/torchtitan/*`: Updated for compatibility
- `primus/backends/megatron_bridge/*`: Updated for compatibility

### Tests & Documentation
- Removed old module-based test file
- Updated registration tests to use stage-based API
- Updated all documentation (README_SFT.md, IMPLEMENTATION_SUMMARY.md)
- Added stage parameter to example configs

## Usage

To use the SFT trainer, specify `stage: sft` in your config:

```yaml
modules:
  trainer:
    framework: megatron
    config: sft_trainer.yaml
    model: llama3_8B.yaml
    
    overrides:
      stage: sft  # ← This selects the SFT trainer
      sft_dataset_name: "tatsu-lab/alpaca"
      sft_conversation_format: "alpaca"
      # ... other config
```

Without the `stage` parameter, it defaults to `pretrain`.

## Verification

All tests pass:
- ✅ Pretrain trainer registers with `stage="pretrain"`
- ✅ SFT trainer registers with `stage="sft"`
- ✅ Default (no stage) returns pretrain trainer
- ✅ Adapter correctly loads trainers by stage
- ✅ Registration tests updated and passing

## Migration Guide

If you have existing configs that used module names for trainer selection, update them:

**Old approach:**
```yaml
modules:
  sft_trainer:  # Module name was used for selection
    framework: megatron
    # ...
```

**New approach:**
```yaml
modules:
  trainer:  # Module name doesn't matter
    framework: megatron
    overrides:
      stage: sft  # ← Add this to select SFT trainer
    # ...
```

## Next Steps

The stage-based registration system is now in place and ready for:
- ✅ SFT training with Megatron-LM
- 🔄 Future stages (RLHF, DPO, etc.)
- 🔄 Multi-stage training pipelines
- 🔄 Stage-specific configurations

## Related Branch

This rebase aligns with commit `9eb5585` from `feat/staged-trainer-registry` branch.
