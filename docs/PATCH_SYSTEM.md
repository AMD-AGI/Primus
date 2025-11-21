# Primus Patch System

## Overview

The Primus Patch System is a flexible, version-aware framework for applying runtime patches to backend frameworks (Megatron, TorchTitan, etc.) without modifying their source code.

## Design Goals

1. **Version Compatibility**: Handle breaking changes between framework versions
2. **Model-Specific Fixes**: Apply patches for specific models (DeepSeek, Llama, etc.)
3. **Hotfix Capability**: Fix bugs without waiting for upstream releases
4. **Zero Source Modification**: All patches applied at runtime
5. **Conditional Application**: Smart patch selection based on context

## Architecture

### Core Components

```
primus/core/patches/
├── patch_system.py          # Core patch system
│   ├── Patch                # Base patch class
│   ├── FunctionPatch        # Function/method patching
│   ├── AttributePatch       # Attribute modification
│   ├── ImportPatch          # Import fixes
│   ├── PatchContext         # Runtime context
│   └── PatchRegistry        # Central registry
└── __init__.py

primus/backends/megatron/patches/
├── __init__.py              # Auto-registration
├── compatibility_patches.py # Version compatibility
├── deepseek_patches.py      # DeepSeek-specific
├── llama_patches.py         # Llama-specific
└── performance_patches.py   # Performance optimizations
```

### Patch Types

#### 1. FunctionPatch
Replaces or wraps functions/methods.

```python
from primus.core.patches import FunctionPatch, PatchRegistry, PatchPriority

def patched_function(original_func, *args, **kwargs):
    # Pre-processing
    result = original_func(*args, **kwargs)
    # Post-processing
    return result

PatchRegistry.register(
    FunctionPatch(
        name="my_patch",
        description="Fix for issue XYZ",
        target_module="megatron.training.arguments",
        target_function="parse_args",
        patch_function=patched_function,
        wrap=True,  # Wrap original (True) or replace (False)
        framework="megatron",
        version_range=">=0.7.0,<0.9.0",
        priority=PatchPriority.HIGH,
    )
)
```

#### 2. AttributePatch
Modifies module/class attributes.

```python
from primus.core.patches import AttributePatch

PatchRegistry.register(
    AttributePatch(
        name="fix_default_value",
        description="Override default configuration",
        target_module="megatron.core.config",
        target_attribute="DEFAULT_BATCH_SIZE",
        new_value=32,
        framework="megatron",
    )
)
```

#### 3. ImportPatch
Fixes import issues.

```python
from primus.core.patches import ImportPatch

PatchRegistry.register(
    ImportPatch(
        name="add_missing_import",
        description="Add missing import for compatibility",
        target_module="megatron.training",
        imports={"get_args": "megatron.training.arguments"},
        framework="megatron",
    )
)
```

### Patch Context

Patches are applied based on runtime context:

```python
from primus.core.patches import PatchContext

context = PatchContext(
    framework="megatron",
    framework_version="0.8.0",
    model_name="llama3_70B",
    model_type="gpt",
    config={"num_layers": 80, "hidden_size": 8192},
)
```

### Version Matching

Supports semantic versioning constraints:

```python
version_range=">=0.7.0,<0.9.0"  # 0.7.0 <= version < 0.9.0
version_range="==0.8.0"          # Exactly 0.8.0
version_range="!=0.7.0"          # Not 0.7.0
```

### Priority Levels

```python
class PatchPriority(Enum):
    CRITICAL = 0   # Import fixes, path setup
    HIGH = 10      # Version compatibility
    NORMAL = 50    # Bug fixes
    LOW = 100      # Optimizations
```

## Usage

### 1. Creating a New Patch

```python
# primus/backends/megatron/patches/my_patches.py

from primus.core.patches import FunctionPatch, PatchRegistry, PatchPriority

def fix_my_issue(original_func, *args, **kwargs):
    """
    Fix for specific issue.

    Issue: Description of the problem
    Fix: Description of the solution
    """
    # Your fix here
    return original_func(*args, **kwargs)

PatchRegistry.register(
    FunctionPatch(
        name="my_issue_fix",
        description="Fix for issue #123",
        target_module="megatron.training.training",
        target_function="train_step",
        patch_function=fix_my_issue,
        wrap=True,
        framework="megatron",
        version_range=">=0.8.0",
        models=["llama3_70B", "llama3_405B"],
        priority=PatchPriority.HIGH,
    )
)
```

### 2. Model-Specific Patches

```python
# Only apply for DeepSeek V3
PatchRegistry.register(
    FunctionPatch(
        name="deepseek_v3_fix",
        description="DeepSeek V3 specific fix",
        target_module="megatron.core.models.gpt",
        target_function="forward",
        patch_function=my_fix,
        wrap=True,
        framework="megatron",
        models=["deepseek_v3", "deepseek_v3_671B"],
    )
)
```

### 3. Conditional Patches

```python
class MyCustomPatch(Patch):
    def check_condition(self, context: PatchContext) -> bool:
        """Custom condition logic."""
        if context.config:
            # Only apply if using MoE with >100 experts
            return context.config.get("num_experts", 0) > 100
        return False

    def apply(self, context: PatchContext) -> bool:
        # Your patch logic
        return True

PatchRegistry.register(MyCustomPatch(...))
```

### 4. Integration with Backend

```python
# In MegatronAdapter.prepare_backend()

from primus.backends.megatron.patches import apply_megatron_patches
from primus.core.patches import PatchContext

# Create context
context = PatchContext(
    framework="megatron",
    framework_version=self._detect_megatron_version(),
    model_name=config.model,
    config=config.params,
)

# Apply patches
applied, failed = apply_megatron_patches(context)
```

## Examples

### Example 1: Version Compatibility Fix

```python
# Fix for Megatron 0.7.0 argument parsing issue
def patched_parse_args_v07(original_func, *args, **kwargs):
    import sys
    if "--no-async-tensor-model-parallel-allreduce" in sys.argv:
        sys.argv.remove("--no-async-tensor-model-parallel-allreduce")
        os.environ["MEGATRON_ASYNC_TP_ALLREDUCE"] = "0"
    return original_func(*args, **kwargs)

PatchRegistry.register(
    FunctionPatch(
        name="megatron_v07_parse_args_fix",
        description="Fix argument parsing in Megatron 0.7.0",
        target_module="megatron.training.arguments",
        target_function="parse_args",
        patch_function=patched_parse_args_v07,
        wrap=True,
        framework="megatron",
        version_range=">=0.7.0,<0.8.0",
        priority=PatchPriority.HIGH,
    )
)
```

### Example 2: Model-Specific Fix

```python
# Fix for Llama 3 RoPE scaling
def patched_llama3_rope_scaling(original_func, *args, **kwargs):
    if "max_position_embeddings" in kwargs:
        max_pos = kwargs["max_position_embeddings"]
        if max_pos > 8192:
            kwargs["rope_scaling"] = {
                "type": "yarn",
                "factor": max_pos / 8192,
            }
    return original_func(*args, **kwargs)

PatchRegistry.register(
    FunctionPatch(
        name="llama3_rope_scaling_fix",
        description="Fix RoPE scaling for Llama 3 long context",
        target_module="megatron.core.models.gpt.gpt_model",
        target_function="GPTModel",
        patch_function=patched_llama3_rope_scaling,
        wrap=True,
        framework="megatron",
        models=["llama3_8B", "llama3_70B", "llama3_405B"],
        priority=PatchPriority.HIGH,
    )
)
```

### Example 3: Performance Optimization

```python
# Enable TF32 for Ampere+ GPUs
def enable_tf32():
    import torch
    if torch.cuda.is_available():
        device_capability = torch.cuda.get_device_capability()
        if device_capability[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            return True
    return False

PatchRegistry.register(
    FunctionPatch(
        name="enable_tf32_ampere",
        description="Enable TF32 for Ampere+ GPUs",
        target_module="megatron.training.initialize",
        target_function="initialize_megatron",
        patch_function=lambda orig, *args, **kwargs: (enable_tf32(), orig(*args, **kwargs))[1],
        wrap=True,
        framework="megatron",
        priority=PatchPriority.LOW,
    )
)
```

## Best Practices

1. **Clear Naming**: Use descriptive patch names (e.g., `megatron_v08_init_order_fix`)
2. **Documentation**: Always document the issue and fix in docstrings
3. **Version Constraints**: Be specific about version ranges
4. **Testing**: Test patches across different versions
5. **Rollback**: Implement rollback when possible
6. **Logging**: Add informative log messages
7. **Priority**: Set appropriate priority levels
8. **Conditions**: Use custom conditions for complex scenarios

## Debugging

### List All Patches

```python
from primus.core.patches import PatchRegistry

# List all patches
all_patches = PatchRegistry.list_patches()

# List Megatron patches only
megatron_patches = PatchRegistry.list_patches(framework="megatron")
```

### Check Applicable Patches

```python
from primus.core.patches import PatchContext, PatchRegistry

context = PatchContext(
    framework="megatron",
    framework_version="0.8.0",
    model_name="llama3_70B",
)

applicable = PatchRegistry.get_applicable_patches(context)
for patch in applicable:
    print(f"{patch.name}: {patch.description}")
```

### Patch Status

```python
# After applying patches
for patch in PatchRegistry.list_patches():
    print(f"{patch.name}: {patch.status.value}")
    if patch.error:
        print(f"  Error: {patch.error}")
```

## Contributing

When adding new patches:

1. Create patch in appropriate file (e.g., `llama_patches.py`)
2. Register with `PatchRegistry.register()`
3. Add tests in `tests/unit_tests/core/patches/`
4. Update this documentation
5. Submit PR with clear description of issue and fix

## Future Enhancements

- [ ] Patch dependency system
- [ ] Automatic patch discovery from git issues
- [ ] Patch performance profiling
- [ ] Web UI for patch management
- [ ] Automatic rollback on failure
- [ ] Patch versioning and history
