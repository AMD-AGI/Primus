# Patch System Version Parameters Usage Guide

## Overview

The Primus Patch System supports two version parameters in `PatchContext`:
- `backend_version`: Version of the backend framework (e.g., Megatron, TorchTitan)
- `primus_version`: Version of Primus itself

This document explains when and how to use these parameters.

## 1. `backend_version` - Backend Framework Version

### Purpose

Track the version of the **backend framework** (Megatron-LM, TorchTitan, etc.) to apply version-specific patches for:
- **Compatibility fixes**: Handle breaking API changes between versions
- **Bug workarounds**: Fix known bugs in specific versions
- **Feature detection**: Enable features only available in certain versions

### Usage Scenarios

#### Scenario 1: Version-Specific Compatibility Patches

**Problem**: Megatron 0.7.x has a different argument parsing API than 0.8.x

**Solution**: Use `condition` with `backend_version` to apply patches only to specific versions

```python
@register_patch(
    "megatron.v07.parse_args_fix",
    backend="megatron",
    phase="before_build_args",
    description="Fix argument parsing issue in Megatron 0.7.x",
    condition=lambda ctx: ctx.backend_version and version_matches(ctx.backend_version, "0.7.*"),
)
def fix_megatron_v07_parse_args(ctx: PatchContext):
    """
    Only applies to Megatron 0.7.x versions.

    Issue: --no-async-tensor-model-parallel-allreduce not handled correctly
    Fix: Remove from sys.argv and set via environment variable
    """
    import sys
    if "--no-async-tensor-model-parallel-allreduce" in sys.argv:
        sys.argv.remove("--no-async-tensor-model-parallel-allreduce")
        os.environ["MEGATRON_ASYNC_TP_ALLREDUCE"] = "0"
```

**Version Matching**:
- `"0.7.*"` - Matches 0.7.0, 0.7.1, 0.7.2, etc.
- `"0.8.0"` - Exact match only
- `"*2024-09*"` - Contains pattern (for commit-based versions)

#### Scenario 2: Multi-Version Compatibility

**Problem**: Megatron 0.8.x and 0.9.x both have the same initialization order issue

**Solution**: Use OR condition to match multiple version ranges

```python
@register_patch(
    "megatron.v08.init_order_fix",
    backend="megatron",
    phase="before_train",
    description="Fix initialization order in Megatron 0.8.0+",
    condition=lambda ctx: ctx.backend_version and (
        version_matches(ctx.backend_version, "0.8.*") or
        version_matches(ctx.backend_version, "0.9.*")
    ),
)
def fix_megatron_v08_init_order(ctx: PatchContext):
    """
    Applies to both Megatron 0.8.x and 0.9.x.

    Issue: Distributed backend must be initialized before model parallel
    Fix: Ensure torch.distributed.init_process_group is called first
    """
    import torch.distributed as dist
    if not dist.is_initialized():
        backend = os.getenv("DISTRIBUTED_BACKEND", "nccl")
        dist.init_process_group(backend=backend)
```

#### Scenario 3: Feature Detection

**Problem**: `torch.compile` is only available in Megatron 0.9.0+

**Solution**: Use version comparison to enable features conditionally

```python
@register_patch(
    "megatron.torch_compile",
    backend="megatron",
    phase="before_train",
    description="Enable torch.compile for supported models",
    condition=lambda ctx: ctx.backend_version and ctx.backend_version >= "0.9.0",
)
def enable_torch_compile(ctx: PatchContext):
    """
    Only applies to Megatron 0.9.0 and later.

    Feature: torch.compile support for faster training
    """
    args = ctx.extra.get("args")
    if args and not hasattr(args, "use_torch_compile"):
        args.use_torch_compile = True
```

#### Scenario 4: Bug Workarounds

**Problem**: Megatron 0.8.1 has a specific memory leak bug

**Solution**: Apply targeted fix only to the affected version

```python
@register_patch(
    "megatron.v081.memory_leak_fix",
    backend="megatron",
    phase="before_train",
    description="Fix memory leak in Megatron 0.8.1",
    condition=lambda ctx: ctx.backend_version == "0.8.1",
)
def fix_memory_leak_v081(ctx: PatchContext):
    """
    Only applies to Megatron 0.8.1 (exact version).

    Issue: Memory leak in gradient accumulation
    Fix: Clear gradient buffers explicitly
    """
    # Apply specific fix for 0.8.1
    pass
```

### How `backend_version` is Detected

```python
# In MegatronAdapter
def _detect_megatron_version(self) -> str:
    try:
        import megatron
        if hasattr(megatron, "__version__"):
            return megatron.__version__  # e.g., "0.8.0"
        elif hasattr(megatron, "version"):
            return megatron.version
        else:
            # Try to detect from package metadata
            from importlib.metadata import version
            return version("megatron-lm")
    except Exception:
        return "unknown"

# Usage
apply_megatron_patches(
    backend_version=self._detect_megatron_version(),  # "0.8.0", "0.9.0", etc.
    phase="before_train",
)
```

### Version Matching Patterns

```python
from primus.core.patches.patch_system import version_matches

# Exact match
version_matches("0.8.0", "0.8.0")  # True
version_matches("0.8.1", "0.8.0")  # False

# Wildcard match (prefix)
version_matches("0.8.0", "0.8.*")  # True
version_matches("0.8.1", "0.8.*")  # True
version_matches("0.9.0", "0.8.*")  # False

# Wildcard match (contains)
version_matches("v2024-09-15", "*2024-09*")  # True
version_matches("commit:abc123", "*abc*")    # True
```

## 2. `primus_version` - Primus Framework Version

### Purpose

Track the version of **Primus itself** to:
- **Deprecate old patches**: Remove patches that are no longer needed in newer Primus versions
- **Feature migration**: Handle changes in Primus architecture across versions
- **Backward compatibility**: Support old configurations in newer Primus versions

### Usage Scenarios

#### Scenario 1: Deprecating Old Patches

**Problem**: Primus 2.0 has a new config system, old patches are no longer needed

**Solution**: Skip patches in newer Primus versions

```python
@register_patch(
    "primus.legacy.config_migration",
    backend="megatron",
    phase="after_build_args",
    description="Migrate old config format to new format",
    condition=lambda ctx: ctx.primus_version and ctx.primus_version < "2.0.0",
)
def migrate_legacy_config(ctx: PatchContext):
    """
    Only applies to Primus < 2.0.0.

    In Primus 2.0+, the new config system handles this automatically.
    """
    args = ctx.extra.get("args")
    # Migrate old config format
    pass
```

#### Scenario 2: Feature Migration

**Problem**: Primus 1.5 changed how logging is initialized

**Solution**: Apply different patches based on Primus version

```python
@register_patch(
    "primus.v15.logging_init",
    backend="megatron",
    phase="before_train",
    description="Initialize logging for Primus 1.5+",
    condition=lambda ctx: ctx.primus_version and ctx.primus_version >= "1.5.0",
)
def init_logging_v15(ctx: PatchContext):
    """
    Only applies to Primus 1.5.0 and later.

    Primus 1.5+ uses a new logging system.
    """
    # Use new logging API
    pass
```

#### Scenario 3: Backward Compatibility

**Problem**: Support old Primus 1.x configs in Primus 2.0

**Solution**: Detect Primus version and apply compatibility layer

```python
@register_patch(
    "primus.v2.backward_compat",
    backend="megatron",
    phase="after_build_args",
    description="Backward compatibility for Primus 1.x configs",
    condition=lambda ctx: ctx.primus_version and ctx.primus_version >= "2.0.0",
)
def ensure_backward_compat(ctx: PatchContext):
    """
    Only applies to Primus 2.0+.

    Ensures old 1.x configs still work in 2.0.
    """
    config = ctx.extra.get("config", {})
    # Convert old config keys to new format
    if "old_key" in config:
        config["new_key"] = config["old_key"]
```

### How `primus_version` is Detected

```python
# In Primus
import primus

def get_primus_version() -> str:
    if hasattr(primus, "__version__"):
        return primus.__version__  # e.g., "1.5.0", "2.0.0"
    return "unknown"

# Usage
apply_megatron_patches(
    backend_version="0.8.0",
    primus_version=get_primus_version(),  # "1.5.0", "2.0.0", etc.
    phase="before_train",
)
```

## 3. Combined Usage: Backend + Primus Version

### Scenario: Complex Compatibility Matrix

**Problem**: Megatron 0.8.x works differently with Primus 1.x vs 2.x

**Solution**: Use both versions in condition

```python
@register_patch(
    "megatron.v08.primus_v2_compat",
    backend="megatron",
    phase="before_train",
    description="Megatron 0.8.x compatibility for Primus 2.0",
    condition=lambda ctx: (
        ctx.backend_version and version_matches(ctx.backend_version, "0.8.*") and
        ctx.primus_version and ctx.primus_version >= "2.0.0"
    ),
)
def megatron_v08_primus_v2_compat(ctx: PatchContext):
    """
    Only applies when:
    - Megatron version is 0.8.x
    - Primus version is 2.0.0 or later

    Handles specific incompatibilities between Megatron 0.8.x and Primus 2.0.
    """
    # Apply compatibility fix
    pass
```

## 4. When NOT to Use Version Parameters

### Scenario 1: Universal Patches

If a patch applies to **all versions**, don't use version conditions:

```python
@register_patch(
    "megatron.env.cuda_device_max_connections",
    backend="megatron",
    phase="before_import_backend",
    description="Set CUDA_DEVICE_MAX_CONNECTIONS based on FSDP configuration",
    # No condition - applies to all versions
)
def set_cuda_device_max_connections(ctx: PatchContext):
    """Applies to all Megatron versions."""
    config = ctx.extra.get("config", {})
    use_fsdp = config.get("use_torch_fsdp2", False) or config.get("use_custom_fsdp", False)
    cuda_connections = "8" if use_fsdp else "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = cuda_connections
```

### Scenario 2: Model-Specific Patches

Use `model_name` instead of `backend_version`:

```python
@register_patch(
    "megatron.deepseek.moe_fix",
    backend="megatron",
    phase="before_train",
    description="Fix MoE load balancing for DeepSeek models",
    condition=lambda ctx: ctx.model_name and "deepseek" in ctx.model_name.lower(),
)
def fix_deepseek_moe(ctx: PatchContext):
    """Applies to DeepSeek models regardless of Megatron version."""
    args = ctx.extra.get("args")
    if args and hasattr(args, "moe_aux_loss_coeff"):
        args.moe_aux_loss_coeff = 0.001
```

## 5. Best Practices

### ✅ DO:

1. **Use version conditions for compatibility fixes**
   ```python
   condition=lambda ctx: version_matches(ctx.backend_version, "0.7.*")
   ```

2. **Use version ranges for multi-version issues**
   ```python
   condition=lambda ctx: ctx.backend_version >= "0.8.0"
   ```

3. **Combine with other conditions**
   ```python
   condition=lambda ctx: (
       version_matches(ctx.backend_version, "0.8.*") and
       ctx.model_name == "llama3_70B"
   )
   ```

4. **Document why version matters**
   ```python
   """
   Only applies to Megatron 0.7.x.

   Issue: Argument parsing changed in 0.8.0
   Fix: Use old API for 0.7.x
   """
   ```

### ❌ DON'T:

1. **Don't use version for universal patches**
   ```python
   # Bad: This applies to all versions
   condition=lambda ctx: ctx.backend_version is not None
   ```

2. **Don't hardcode version strings**
   ```python
   # Bad: Hardcoded version
   if ctx.backend_version == "0.8.0":

   # Good: Use version_matches
   if version_matches(ctx.backend_version, "0.8.*"):
   ```

3. **Don't forget to handle None**
   ```python
   # Bad: Can raise AttributeError if backend_version is None
   condition=lambda ctx: ctx.backend_version >= "0.8.0"

   # Good: Check for None first
   condition=lambda ctx: ctx.backend_version and ctx.backend_version >= "0.8.0"
   ```

## 6. Summary

| Parameter | Purpose | When to Use | Example |
|-----------|---------|-------------|---------|
| `backend_version` | Track backend framework version | Version-specific compatibility, bug fixes, feature detection | `"0.8.0"`, `"0.9.1"`, `"commit:abc123"` |
| `primus_version` | Track Primus version | Deprecate old patches, feature migration, backward compatibility | `"1.5.0"`, `"2.0.0"` |

**Key Takeaway**: Use version parameters to make patches **conditional** based on the versions of the backend framework and Primus itself, enabling fine-grained control over when patches are applied.
