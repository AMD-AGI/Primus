# Primus Trainer Architecture

## Design Pattern: Template Method

The Primus trainer architecture uses the **Template Method Pattern** to provide a consistent training workflow across **all backends** (Megatron, TorchTitan, JAX/Maxtext, etc.) while allowing flexibility in backend-specific implementations.

## Architecture Overview

```
BaseModule (abstract)
    ↓
BaseTrainer (universal template method implementation)
    ↓
MegatronBaseTrainer, TorchtitanBaseTrainer, MaxtextBaseTrainer (backend-specific)
    ↓
MegatronPretrainTrainer, TorchtitanPretrainTrainer, ... (task-specific)
```

### Key Design Principle

**Separation of Concerns:**
- `BaseTrainer`: Universal training workflow and patch management (backend-agnostic)
- `MegatronBaseTrainer`: Megatron-specific initialization and patch application
- `MegatronPretrainTrainer`: Pretrain-specific training logic

## Key Components

### 1. BaseTrainer (Universal Base Class)

**Location:** `primus/core/trainer/base_trainer.py`

**Responsibilities:**
- Defines universal training workflow (backend-agnostic)
- Manages patch execution lifecycle
- Provides template methods for all backends

**Template Method: `run()`**

```python
def run(self):
    # Phase 1: Apply before_train patches
    patch_count = self.apply_patches(phase="before_train")

    # Phase 2: Execute training (subclass implementation)
    self.run_train()  # ← Implemented by task-specific trainer

    # Phase 3: Apply after_train patches
    # patch_count = self.apply_patches(phase="after_train")
```

**Abstract Methods:**
```python
@abstractmethod
def run_train(self):
    """Task-specific training logic."""
    raise NotImplementedError()

@abstractmethod
def apply_patches(self, phase: str) -> int:
    """Backend-specific patch application."""
    raise NotImplementedError()
```

### 2. MegatronBaseTrainer (Backend-Specific Base Class)

**Location:** `primus/backends/megatron/trainers/megatron_base_trainer.py`

**Responsibilities:**
- Megatron-specific initialization (parse_args patching, ROCm patches)
- Implements `apply_patches()` for Megatron
- Common Megatron patterns

**Implementation:**
```python
class MegatronBaseTrainer(BaseTrainer):
    def apply_patches(self, phase: str) -> int:
        """Apply Megatron-specific patches."""
        return apply_megatron_patches(
            backend_version=self._detect_version(),
            model_name=self.model_name,
            phase=phase,
            extra={"args": self.backend_args},
        )

    # Megatron-specific methods
    def _patch_parse_args(self): ...
    def _detect_version(self): ...
```

### 3. Task-Specific Trainers (Concrete Implementations)

Each trainer only needs to implement `run_train()` with the specific training logic.

**Example: MegatronPretrainTrainer**

```python
class MegatronPretrainTrainer(MegatronBaseTrainer):
    def run_train(self):
        """Pretrain-specific logic."""
        # Import training components
        from megatron.training import pretrain
        from pretrain_gpt import forward_step, train_valid_test_datasets_provider

        # Execute training
        pretrain(train_valid_test_datasets_provider, model_provider, ...)
```

## Benefits

### ✅ Universal Design
- **Works across all backends**: Megatron, TorchTitan, JAX/Maxtext, etc.
- **Consistent workflow**: All backends follow the same pattern
- **No code duplication**: Patch management written once in `BaseTrainer`

### ✅ Separation of Concerns
- **BaseTrainer**: Universal workflow and patch lifecycle (backend-agnostic)
- **MegatronBaseTrainer**: Megatron-specific initialization and patch application
- **MegatronPretrainTrainer**: Pretrain-specific training logic

### ✅ DRY (Don't Repeat Yourself)
- Universal patch workflow in `BaseTrainer`
- Backend-specific logic in backend base trainers
- Task-specific logic in concrete trainers

### ✅ Consistency
- All backends and all tasks follow the same workflow
- Impossible to forget patch application
- Standardized training lifecycle

### ✅ Extensibility
- **New backend**: Inherit `BaseTrainer` + implement `apply_patches()`
- **New task**: Inherit backend base trainer + implement `run_train()`
- No need to understand or copy patch management code

## How to Add Support for a New Backend

### Step 1: Create Backend Base Trainer

```python
# primus/backends/torchtitan/trainers/torchtitan_base_trainer.py
from primus.core.trainer.base_trainer import BaseTrainer
from primus.backends.torchtitan.patches import apply_torchtitan_patches

class TorchtitanBaseTrainer(BaseTrainer):
    """Base trainer for all TorchTitan tasks."""

    def apply_patches(self, phase: str) -> int:
        """Apply TorchTitan-specific patches."""
        return apply_torchtitan_patches(
            backend_version=self._detect_version(),
            model_name=self.model_name,
            phase=phase,
            extra={"config": self.backend_args},
        )

    def _detect_version(self) -> str:
        """Detect TorchTitan version."""
        # TorchTitan-specific version detection
        return "0.1.0"
```

### Step 2: Create Task-Specific Trainer

```python
# primus/backends/torchtitan/trainers/torchtitan_pretrain_trainer.py
class TorchtitanPretrainTrainer(TorchtitanBaseTrainer):
    """Trainer for TorchTitan pre-training."""

    def run_train(self):
        """TorchTitan pretrain-specific logic."""
        from torchtitan.train import train
        train(self.backend_args)
```

### Step 3: Create Patch System (if needed)

```python
# primus/backends/torchtitan/patches/__init__.py
from primus.core.patches import run_patches

def apply_torchtitan_patches(*, backend_version, model_name, phase, extra):
    """Apply TorchTitan-specific patches."""
    return run_patches(
        backend="torchtitan",
        phase=phase,
        backend_version=backend_version,
        model_name=model_name,
        extra=extra,
    )
```

## How to Add a New Task for Existing Backend

### Example: Adding SFT for Megatron

```python
from primus.backends.megatron.trainers.megatron_base_trainer import MegatronBaseTrainer

class MegatronSFTTrainer(MegatronBaseTrainer):
    """Trainer for Supervised Fine-Tuning."""

    def run_train(self):
        """Implement SFT-specific training logic."""
        from megatron.training import pretrain
        # ... SFT-specific imports and setup

        # Execute training
        pretrain(...)
```

### Step 2: Register Trainer (Optional)

If using a trainer registry, register your new trainer:

```python
from primus.core.backend.backend_registry import BackendRegistry

BackendRegistry.register_trainer_class("megatron_sft", MegatronSFTTrainer)
```

### Step 3: Use the Trainer

```python
trainer = MegatronSFTTrainer(primus_config, module_config, backend_args)
trainer.setup()  # Optional
trainer.init()   # Optional
trainer.run()    # Executes: patches → run_train() → cleanup
```

## Universal Training Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  BaseTrainer.run() [Universal Template]                     │
│  Works for ALL backends: Megatron, TorchTitan, Maxtext, ... │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Phase 1: Apply before_train patches                        │
│  └─ Call: self.apply_patches("before_train")              │
│     ├─ MegatronBaseTrainer → apply_megatron_patches()     │
│     ├─ TorchtitanBaseTrainer → apply_torchtitan_patches() │
│     └─ MaxtextBaseTrainer → apply_maxtext_patches()       │
│                                                             │
│  Phase 2: Execute training                                  │
│  └─ Call: self.run_train()                                 │
│     ├─ MegatronPretrainTrainer → megatron pretrain        │
│     ├─ TorchtitanPretrainTrainer → torchtitan train       │
│     └─ MaxtextPretrainTrainer → maxtext train             │
│                                                             │
│  Phase 3: Apply after_train patches                         │
│  └─ Call: self.apply_patches("after_train")               │
│     └─ (Optional cleanup patches)                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Migration Guide

### Before (Old Design)

```python
class MegatronPretrainTrainer:
    def run(self):
        # ❌ Manually apply patches
        apply_megatron_patches(phase="before_train", ...)

        # Training logic
        pretrain(...)
```

**Problems:**
- Every trainer needs to remember to call patches
- Code duplication across trainers
- Easy to forget or misplace patch calls

### After (New Design)

```python
class MegatronPretrainTrainer(MegatronBaseTrainer):
    def run_train(self):
        # ✅ Just implement training logic
        # Patches are automatically handled by base class
        pretrain(...)
```

**Benefits:**
- Base class handles all patch management
- Trainers focus on training logic
- Consistent workflow across all trainers

## FAQ

### Q: Can I override `run()` in my trainer?

**A:** Not recommended. The `run()` method in `MegatronBaseTrainer` is the template that ensures patches are applied correctly. Instead, implement `run_train()` for your training logic.

### Q: What if I need custom pre-training setup?

**A:** Use the `setup()` or `init()` methods:

```python
class MyTrainer(MegatronBaseTrainer):
    def init(self):
        # Custom initialization
        self.my_custom_setup()

    def run_train(self):
        # Training logic
        pass
```

### Q: Can I add custom patches?

**A:** Yes! Use the patch system:

```python
from primus.core.patches import register_patch

@register_patch(
    "megatron.my_custom_patch",
    backend="megatron",
    phase="before_train",
)
def my_patch(ctx):
    # Custom patch logic
    pass
```

The base class will automatically apply all registered patches.

## See Also

- [Patch System Documentation](../../core/patches/README.md)
- [Backend Adapter Pattern](../../core/backend/README.md)
- [Template Method Pattern](https://refactoring.guru/design-patterns/template-method)
