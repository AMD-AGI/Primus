# TorchTitan Model Override Patch

## Overview

The `model_override_patches` module provides a mechanism to dynamically override TorchTitan model configuration parameters at runtime without modifying TorchTitan's train_spec registry.

## Purpose

This patch enables users to:
- Override model architecture parameters (e.g., `n_layers`, `dim`, `n_heads`)
- Test different model configurations without creating new train_spec entries
- Quickly iterate on model architectures during development

## Configuration

### Enable the Patch

Add `model_overrides` to your module's `params` section in the Primus YAML config:

```yaml
modules:
  - name: pre_trainer
    framework: torchtitan
    model:
      name: llama3
      flavor: debugmodel
    params:
      # ... other torchtitan params ...
      model_overrides:
        model.n_layers: 8
        model.dim: 2048
        model.n_heads: 16
```

### Nested Format (Auto-Flattened)

You can also use nested format, which will be automatically flattened:

```yaml
params:
  model_overrides:
    model:
      n_layers: 8
      dim: 2048
      n_heads: 16
```

This is equivalent to the flat format above.

## Requirements

1. **Key Prefix**: All override keys MUST start with `"model."` prefix
   - ✅ Valid: `model.n_layers`, `model.dim`, `model.n_heads`
   - ❌ Invalid: `n_layers`, `training.steps`, `optimizer.lr`

2. **Model Name & Flavor**: Must specify both `model.name` and `model.flavor`
   ```yaml
   model:
     name: llama3      # Required
     flavor: debugmodel # Required
   ```

3. **Valid Fields**: Override keys must correspond to existing fields in the model's dataclass

## How It Works

1. **Condition Check**: Patch activates when `params.model_overrides` is present
2. **Validation**: Validates all keys have `model.` prefix
3. **Target Identification**: Extracts `model.name` and `model.flavor` from config
4. **Monkey Patching**: Intercepts `torchtitan.protocols.train_spec.get_train_spec()`
5. **Selective Override**: Only modifies the specified model and flavor
6. **Dynamic Application**: Updates model_args dataclass fields before TorchTitan Trainer initialization

## Example Workflow

```python
# Original train_spec for llama3.debugmodel
# ModelArgs(n_layers=32, dim=4096, n_heads=32)

# With override config:
model_overrides:
  model.n_layers: 8
  model.dim: 2048

# Result after patch:
# ModelArgs(n_layers=8, dim=2048, n_heads=32)
#            ↑ overridden   ↑ overridden   ↑ unchanged
```

## Error Handling

### Invalid Key Prefix
```yaml
model_overrides:
  n_layers: 8  # ❌ Missing 'model.' prefix
```
**Error**: `ValueError: Invalid override keys detected: ['n_layers']`

### Missing Model Name
```yaml
model:
  flavor: debugmodel  # ❌ Missing 'name'
```
**Error**: `ValueError: model.name is required for model override patch`

### Invalid Flavor
```yaml
model:
  name: llama3
  flavor: nonexistent  # ❌ Flavor doesn't exist
```
**Error**: `KeyError: flavor 'nonexistent' not found in model_args for 'llama3'`

### Invalid Field
```yaml
model_overrides:
  model.invalid_field: 123  # ❌ Field doesn't exist in dataclass
```
**Error**: `AttributeError: 'ModelArgs' has no field 'invalid_field'`

## Comparison with Old Implementation

### Old Way (in `pre_trainer.py`)
```python
def __init__(self, *args, **kwargs):
    extra_args = kwargs.pop("extra_args", None)
    # ...
    self.patch_titan_train_spec(model_name, flavor, extra_args)
```

### New Way (Patch System)
```yaml
# Just add to config - patch applies automatically!
params:
  model_overrides:
    model.n_layers: 8
```

## Benefits

1. **Declarative**: Configuration-driven, no code changes needed
2. **Centralized**: Managed via unified patch system
3. **Discoverable**: Registered in patch registry
4. **Testable**: Can be unit tested independently
5. **Consistent**: Follows same pattern as other TorchTitan patches

## Testing

Run unit tests:
```bash
pytest tests/unit_tests/backends/torchtitan/test_model_override_patches.py -v
```

## Related Files

- **Patch Implementation**: `primus/backends/torchtitan/patches/model_override_patches.py`
- **Patch Registration**: `primus/backends/torchtitan/patches/__init__.py`
- **Unit Tests**: `tests/unit_tests/backends/torchtitan/test_model_override_patches.py`
- **Old Implementation**: `primus/modules/trainer/torchtitan/pre_trainer.py` (line 183-268)
