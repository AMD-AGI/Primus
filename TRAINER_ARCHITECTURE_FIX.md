# Megatron Trainer Architecture Fix

## Problem Summary

The `MegatronPretrainTrainer` had a fundamental architecture mismatch with how Megatron-LM expects to receive arguments and initialize its training environment.

### Original Issues

1. **Double Initialization Conflict**: The trainer was calling `initialize_megatron()` directly in `init()`, but Megatron's `pretrain()` function also calls `initialize_megatron()` internally, causing conflicts.

2. **Argument Injection Problem**: The pre-built `backend_args` (SimpleNamespace from `MegatronArgBuilder`) were not being properly injected into Megatron's argument parsing system.

3. **Redundant Conversion**: Converting `SimpleNamespace` → `sys.argv` → re-parsing loses type information and is inefficient.

## Solution Architecture

### Key Insight

**The adapter has already done all the work!**

`MegatronAdapter.convert_config()` produces a complete `SimpleNamespace` that:
- Contains ALL Megatron parameters
- Has merged: Megatron defaults + Model preset + User config + CLI overrides
- Is ready to use directly

**Therefore, the trainer should:**
1. **Directly inject** this prepared namespace into Megatron's global state
2. **Skip argument parsing** entirely (no sys.argv conversion needed)
3. **Let pretrain() use** the pre-set global args

### Flow Overview

```
train_launcher.py
  ↓
  1. setup_training_env()       ← Setup HF_HOME, TORCH_HOME, cache paths
  2. init_distributed_env()     ← Setup RANK, WORLD_SIZE, MASTER_ADDR/PORT
  3. PrimusConfig.from_file()   ← Load YAML config
  4. init_global_logger()       ← Setup global logger
  ↓
BackendRegistry.get_adapter("megatron")
  ↓
MegatronAdapter.create_trainer()
  ↓
  a. prepare_backend()
     - Apply patches (before_import_backend, after_import_backend)
     - Setup backend path
  ↓
  b. convert_config()            ← ★ KEY STEP ★
     ┌──────────────────────────────────────────────────────┐
     │ 1. Apply before_build_args patches                   │
     │ 2. MegatronArgBuilder:                               │
     │    - Load Megatron defaults (via argparse)           │
     │    - Merge user config (module_config.params)        │
     │    - Produce SimpleNamespace with ALL fields         │
     │ 3. Apply after_build_args patches                    │
     └──────────────────────────────────────────────────────┘
     → Returns: Complete SimpleNamespace (backend_args)
  ↓
  c. load_trainer_class()        ← Get MegatronPretrainTrainer class
  ↓
  d. TrainerClass(
       primus_config,
       module_config,
       backend_args              ← Already complete!
     )
  ↓
MegatronPretrainTrainer.__init__()
  - Store backend_args
  - Initialize BaseModule (gets distributed env from RuntimeContext)
  ↓
trainer.init()
  ┌──────────────────────────────────────────────────────┐
  │ Try to inject args directly into Megatron:           │
  │   from megatron.training import global_vars          │
  │   global_vars._set_args(self.backend_args)           │
  │                                                      │
  │ Fallback if direct injection fails:                  │
  │   Convert backend_args → sys.argv                    │
  └──────────────────────────────────────────────────────┘
  ↓
trainer.run()
  - Apply before_train patches
  - Get model_provider
  - Call Megatron's pretrain()
    └→ If args already set: use directly
       If not: pretrain() calls initialize_megatron()
```

### Key Design Decisions

#### 1. **Dual Strategy for Argument Injection**

We use **two complementary strategies** to inject the prepared arguments into Megatron:

**Strategy 1: Direct Injection (Fast, Clean)**
```python
from megatron.training import global_vars
global_vars._GLOBAL_ARGS = self.backend_args
# or
global_vars._set_args(self.backend_args)
```

**Strategy 2: Monkey Patch `parse_args` (Reliable, Universal)**
```python
import megatron.training.arguments as megatron_args
import megatron.training.initialize as megatron_init

def patched_parse_args(*args, **kwargs):
    return self.backend_args

megatron_args.parse_args = patched_parse_args
megatron_init.parse_args = patched_parse_args
```

**Why Both?**

| Strategy | Pros | Cons |
|----------|------|------|
| Direct Injection | - Fastest<br>- Least invasive<br>- No side effects | - Depends on `_GLOBAL_ARGS` existence<br>- Might be bypassed if `parse_args()` is called |
| Monkey Patch | - Works with all versions<br>- Intercepts all `parse_args()` calls<br>- Can add custom logic | - More invasive<br>- Affects any code calling `parse_args()` |

**Combined Approach = Best of Both Worlds**:
- Try direct injection first (preferred)
- Always apply monkey patch as backup (guarantees it works)
- Fallback to sys.argv if both fail (extreme edge case)

**Implementation:**
```python
def init(self):
    # Strategy 1: Direct (fast path)
    direct_success = self._try_direct_injection()

    # Strategy 2: Patch (reliable path)
    self._patch_parse_args()

    # Strategy 3: sys.argv (last resort, only if patch fails)
    # Handled in _patch_parse_args exception handler
```

#### 2. **Why This Works Better**

| Aspect | Old Approach (sys.argv) | New Approach (Direct Injection) |
|--------|------------------------|--------------------------------|
| **Type Safety** | Lost (everything becomes strings) | Preserved (int, float, bool, list) |
| **Performance** | Slower (parse twice) | Faster (parse once) |
| **Accuracy** | Potential conversion errors | Exact values preserved |
| **Simplicity** | Complex conversion logic | Simple assignment |
| **Compatibility** | Depends on Megatron's parser | Works with any Megatron version |

#### 3. **Configuration Merge Happens Once**

The beauty of this design: **All configuration merging happens in the adapter**

```python
# In MegatronAdapter.convert_config()
builder = MegatronArgBuilder()
builder.update(module_config.params)  # Already includes:
                                       # - Megatron defaults
                                       # - Model preset
                                       # - User config
                                       # - CLI overrides
megatron_args = builder.finalize()    # → Complete SimpleNamespace

# In MegatronPretrainTrainer.init()
# Just use it directly, no re-processing!
global_vars._set_args(megatron_args)
```

#### 4. **Separation of Concerns**

| Component | Responsibility |
|-----------|----------------|
| `train_launcher.py` | Global env setup (distributed, logging, cache paths) |
| `MegatronAdapter` | Backend-specific patches, **complete config→args conversion** |
| `MegatronArgBuilder` | Merge all config layers → Megatron defaults |
| `MegatronPretrainTrainer` | **Just inject args and delegate to Megatron** |
| `BaseModule` | Access global state (RuntimeContext) |

**Key Point**: The trainer is now **extremely simple** - it just passes the prepared args to Megatron.

## Configuration Merge Priority

The final arguments passed to Megatron follow this priority (highest to lowest):

1. **CLI overrides** (`--lr=0.001`)
2. **User YAML config** (`params.lr: 0.001`)
3. **Model preset** (`models/llama/3.1-8B.yaml`)
4. **Megatron defaults** (from `megatron.training.arguments`)

This is achieved through:
```
MegatronArgBuilder:
  1. Load Megatron defaults (argparse)
  2. Apply model preset (if specified)
  3. Apply user config params
  4. Apply CLI overrides
  → Produces SimpleNamespace
```

## Testing Considerations

### Unit Test Coverage

1. **`test_megatron_adapter.py`**
   - Test `convert_config()` produces correct SimpleNamespace
   - Verify all config layers are merged correctly
   - Check patch application at each phase

2. **`test_megatron_pretrain_trainer.py`** (TODO)
   - Test `init()` converts args to sys.argv correctly
   - Verify boolean flags handled properly
   - Test list/tuple arguments expanded correctly
   - Mock `pretrain()` to verify it receives correct model_provider

3. **Integration Tests** (TODO)
   - Full flow: config → adapter → trainer → pretrain
   - Test with actual Megatron-LM (if available in CI)

### Manual Testing

```bash
# Test with a minimal config
primus-cli train pretrain \
  --config examples/configs/minimal_megatron.yaml \
  --backend_path /path/to/megatron-lm \
  num_layers=12 \
  hidden_size=768
```

## Potential Issues & Future Work

### 1. **sys.argv Pollution**

**Issue**: We permanently modify `sys.argv` in `init()`.

**Impact**: If the trainer is reused or other code reads `sys.argv`, they'll see Megatron arguments.

**Future Fix**: Consider using a context manager or restoring `sys.argv` after `pretrain()` returns (though training typically exits the process).

### 2. **Megatron Version Compatibility**

**Issue**: Different Megatron versions may have different argument names or requirements.

**Mitigation**: The Patch System can handle this via version-specific patches in `before_build_args` phase.

### 3. **Complex Argument Types**

**Issue**: Some Megatron arguments may have complex types (dicts, nested structures) that don't map cleanly to command-line format.

**Future Enhancement**: Add special handling in `init()` for complex types, or use Megatron's `extra_args_provider` mechanism.

### 4. **Distributed Initialization**

**Issue**: We initialize distributed environment in `train_launcher.py` via `init_distributed_env()`, but Megatron also has its own distributed init in `initialize_megatron()`.

**Current State**: Primus sets env vars (`MASTER_ADDR`, `RANK`, etc.) which Megatron then reads. This works but creates implicit coupling.

**Future Consideration**: Document the exact initialization order and any potential conflicts.

## Related Files

- `primus/core/launcher/train_launcher.py` - Entry point and global setup
- `primus/backends/megatron/adapters/megatron_adapter.py` - Adapter implementation
- `primus/backends/megatron/builders/argument_builder.py` - Config → Args conversion
- `primus/backends/megatron/trainers/megatron_pretrain_trainer.py` - Trainer implementation
- `primus/core/trainer/base_module.py` - Base class with global state access
- `primus/core/runtime/context.py` - Global runtime context singleton

## Summary

The fix establishes a clean separation between:
1. **Primus's configuration management** (YAML + CLI → unified config)
2. **Megatron's argument system** (command-line parsing)

By converting our prepared arguments back to command-line format and letting Megatron's `pretrain()` handle initialization naturally, we avoid conflicts and maintain compatibility across Megatron versions.
