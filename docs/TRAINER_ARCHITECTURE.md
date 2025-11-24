# Primus Trainer Architecture

## Overview

Primus uses a **three-layer hierarchy** based on the **Template Method Pattern** to provide a universal training framework that works across all backends (Megatron, TorchTitan, JAX/Maxtext, etc.).

## Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1: BaseModule (ABC)                              │
│  - Core trainer interface                               │
│  - Runtime context management                           │
│  - Distributed environment access                       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 2: BaseTrainer (Universal Template)              │
│  - Training workflow template (run method)              │
│  - Patch lifecycle management                           │
│  - Backend-agnostic training orchestration              │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 3: Backend-Specific Base Trainers                │
│  - MegatronBaseTrainer: Megatron initialization         │
│  - TorchtitanBaseTrainer: TorchTitan initialization     │
│  - MaxtextBaseTrainer: JAX/Maxtext initialization       │
│  - Backend-specific patch application                   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 4: Task-Specific Trainers                        │
│  - MegatronPretrainTrainer, MegatronSFTTrainer, ...     │
│  - TorchtitanPretrainTrainer, TorchtitanSFTTrainer, ... │
│  - Task-specific training logic                         │
└─────────────────────────────────────────────────────────┘
```

## Layer Details

### Layer 1: BaseModule (Abstract Base)

**Location:** `primus/core/trainer/base_module.py`

**Responsibilities:**
- Define core trainer interface (`init()`, `setup()`, `run()`)
- Provide access to runtime context (distributed env, logging)
- Configuration management

**Key Features:**
- Access to `RuntimeContext` (rank, world_size, etc.)
- Distributed environment validation
- Configuration properties

### Layer 2: BaseTrainer (Universal Template)

**Location:** `primus/core/trainer/base_trainer.py`

**Responsibilities:**
- Implement universal training workflow
- Manage patch lifecycle
- Provide template methods for all backends

**Template Method Pattern:**

```python
class BaseTrainer(BaseModule):
    def run(self):
        """Universal training workflow."""
        # Phase 1: Apply before_train patches
        patch_count = self.apply_patches(phase="before_train")

        # Phase 2: Execute training
        self.run_train()  # ← Implemented by task-specific trainer

        # Phase 3: Apply after_train patches (optional)
        # patch_count = self.apply_patches(phase="after_train")

    @abstractmethod
    def run_train(self):
        """Task-specific training logic."""
        raise NotImplementedError()

    @abstractmethod
    def apply_patches(self, phase: str) -> int:
        """Backend-specific patch application."""
        raise NotImplementedError()
```

**Key Principles:**
- **Backend-agnostic**: Works with any backend
- **Consistent workflow**: All trainers follow same pattern
- **DRY**: Patch management written once

### Layer 3: Backend-Specific Base Trainers

#### MegatronBaseTrainer

**Location:** `primus/backends/megatron/trainers/megatron_base_trainer.py`

**Responsibilities:**
- Megatron-specific initialization
- Parse args patching (argument injection)
- ROCm compatibility patches
- Implement `apply_patches()` for Megatron

**Implementation:**

```python
class MegatronBaseTrainer(BaseTrainer):
    def __init__(self, primus_config, module_config, backend_args):
        super().__init__(...)
        self._patch_parse_args()  # Inject args into Megatron
        self._patch_megatron_runtime_hooks()  # ROCm patches

    def apply_patches(self, phase: str) -> int:
        """Apply Megatron-specific patches."""
        return apply_megatron_patches(
            backend_version=self._detect_version(),
            model_name=self.model_name,
            phase=phase,
            extra={"args": self.backend_args},
        )
```

#### TorchtitanBaseTrainer (Example for Future)

**Location:** `primus/backends/torchtitan/trainers/torchtitan_base_trainer.py`

```python
class TorchtitanBaseTrainer(BaseTrainer):
    def apply_patches(self, phase: str) -> int:
        """Apply TorchTitan-specific patches."""
        return apply_torchtitan_patches(
            backend_version=self._detect_version(),
            model_name=self.model_name,
            phase=phase,
            extra={"config": self.backend_args},
        )
```

### Layer 4: Task-Specific Trainers

**Examples:**
- `MegatronPretrainTrainer`: Megatron pre-training
- `MegatronSFTTrainer`: Megatron supervised fine-tuning
- `TorchtitanPretrainTrainer`: TorchTitan pre-training

**Responsibilities:**
- Implement `run_train()` with task-specific logic
- Import and configure task-specific components
- Execute the actual training loop

**Example:**

```python
class MegatronPretrainTrainer(MegatronBaseTrainer):
    def run_train(self):
        """Pretrain-specific logic."""
        from megatron.training import pretrain
        from pretrain_gpt import forward_step, train_valid_test_datasets_provider

        # Configure and execute
        pretrain(train_valid_test_datasets_provider, model_provider, ...)
```

## Workflow Diagram

```
User calls trainer.run()
         ↓
┌────────────────────────────────────────────────────────┐
│ BaseTrainer.run() [Universal Template]                 │
├────────────────────────────────────────────────────────┤
│                                                        │
│  1. self.apply_patches("before_train")                │
│     ↓                                                  │
│     MegatronBaseTrainer.apply_patches()               │
│     └─ apply_megatron_patches(...)                    │
│        ├─ Environment patches                         │
│        ├─ Model-specific patches                      │
│        └─ Version-specific patches                    │
│                                                        │
│  2. self.run_train()                                  │
│     ↓                                                  │
│     MegatronPretrainTrainer.run_train()              │
│     └─ Execute Megatron pretrain                      │
│        ├─ Import training components                  │
│        ├─ Configure training                          │
│        └─ Run training loop                           │
│                                                        │
│  3. self.apply_patches("after_train") [optional]     │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## Benefits

### 🎯 Universal Design

- **Single workflow**: All backends use the same pattern
- **Consistent behavior**: Patches always applied in correct order
- **No special cases**: Every trainer follows the template

### 🔧 Separation of Concerns

| Layer | Responsibility |
|-------|----------------|
| `BaseTrainer` | Universal workflow, patch lifecycle |
| `MegatronBaseTrainer` | Megatron initialization, Megatron patches |
| `MegatronPretrainTrainer` | Pretrain-specific training logic |

### 📦 DRY (Don't Repeat Yourself)

- Patch management: **Once** in `BaseTrainer`
- Backend initialization: **Once** per backend base trainer
- Training logic: **Once** per task trainer

### ✨ Extensibility

**Adding a new backend:**
1. Create `{Backend}BaseTrainer(BaseTrainer)`
2. Implement `apply_patches()`
3. Add backend-specific initialization

**Adding a new task:**
1. Create `{Backend}{Task}Trainer({Backend}BaseTrainer)`
2. Implement `run_train()`
3. Done!

## Migration from Old Design

### Before (Backend-Specific Patch Management)

```python
# ❌ MegatronBaseTrainer had patch management built-in
class MegatronBaseTrainer(BaseModule):
    def run(self):
        # Megatron-specific patch management
        apply_megatron_patches(...)
        self.run_train()
```

**Problems:**
- Patch logic duplicated for each backend
- TorchTitan, Maxtext would need their own implementations
- Inconsistent across backends

### After (Universal Patch Management)

```python
# ✅ BaseTrainer provides universal patch management
class BaseTrainer(BaseModule):
    def run(self):
        # Universal workflow
        self.apply_patches("before_train")  # ← Backend implements this
        self.run_train()                     # ← Task implements this

class MegatronBaseTrainer(BaseTrainer):
    def apply_patches(self, phase):
        return apply_megatron_patches(...)  # ← Megatron-specific
```

**Benefits:**
- Single workflow implementation
- Easy to add new backends
- Consistent across all backends

## Code Examples

### Complete Example: Adding JAX/Maxtext Support

```python
# 1. Create backend base trainer
# primus/backends/maxtext/trainers/maxtext_base_trainer.py
from primus.core.trainer.base_trainer import BaseTrainer

class MaxtextBaseTrainer(BaseTrainer):
    """Base trainer for JAX/Maxtext."""

    def apply_patches(self, phase: str) -> int:
        from primus.backends.maxtext.patches import apply_maxtext_patches
        return apply_maxtext_patches(
            backend_version=self._detect_version(),
            model_name=self.model_name,
            phase=phase,
            extra={"config": self.backend_args},
        )

    def _detect_version(self) -> str:
        import maxtext
        return maxtext.__version__

# 2. Create task trainer
# primus/backends/maxtext/trainers/maxtext_pretrain_trainer.py
class MaxtextPretrainTrainer(MaxtextBaseTrainer):
    """Pretrain trainer for JAX/Maxtext."""

    def run_train(self):
        from maxtext.train import train
        train(self.backend_args)

# 3. Use it
trainer = MaxtextPretrainTrainer(primus_config, module_config, backend_args)
trainer.run()  # ← Uses universal workflow with Maxtext patches!
```

## See Also

- [Patch System Documentation](../primus/core/patches/README.md)
- [Backend Adapter Pattern](../primus/core/backend/README.md)
- [Megatron Trainer README](../primus/backends/megatron/trainers/README.md)
- [Template Method Pattern](https://refactoring.guru/design-patterns/template-method)
