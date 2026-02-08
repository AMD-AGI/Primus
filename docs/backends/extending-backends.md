# Backend Extension Guide

This guide explains how to add a **new training backend** to Primus using the
current runtime architecture:

- `BackendAdapter` – integrates a backend framework
- `BackendRegistry` – discovers and instantiates adapters
- `BaseTrainer` – defines the minimal training lifecycle that all backends follow
- `PrimusRuntime` – orchestrates config loading, env setup, patches, adapter & trainer

The examples below use a minimal **`dummy`** backend, but the pattern is the
same for real backends (Megatron, TorchTitan, MaxText, etc.).

---

## 1. What actually happens when you run Primus?

When you run:

```bash
primus train pretrain --config <exp.yaml>
```

the runtime (`PrimusRuntime`) does roughly:

1. Load `PrimusConfig` and the selected `module_config`
2. Apply CLI overrides to `module_config.params`
3. Initialize environment (HF, logging, distributed env, data dir)
4. Resolve backend adapter via `BackendRegistry.get_adapter(framework)`
5. Call `adapter.setup_backend_path(...)` to put the backend on `sys.path`
6. Call `adapter.prepare_backend(module_config)` (usually runs backend setup hooks)
7. Build backend args:

   ```python
   backend_args = adapter.convert_config(module_config.params)
   # run "build_args" patches and merge back into module_config.params
   ```

8. Detect backend version (for patches) via `adapter.detect_backend_version()`
9. Load and construct the trainer:

   ```python
   TrainerClass = adapter.load_trainer_class(stage=module_config.params.stage or "pretrain")
   trainer = TrainerClass(backend_args=backend_args)
   ```

10. Execute the trainer lifecycle (with patches around it):

    ```python
    # PrimusRuntime:
    run_patches(phase="setup",        backend_args=backend_args)
    trainer.setup()

    trainer.init()

    run_patches(phase="before_train", backend_args=backend_args)
    trainer.train()
    run_patches(phase="after_train",  backend_args=backend_args)

    trainer.cleanup()
    ```

So a complete backend must provide:

- An **adapter** subclassing `BackendAdapter`
- A **trainer** subclassing `BaseTrainer` and implementing `setup/init/train/cleanup`
- A small `primus.backends.<name>.__init__` that calls `BackendRegistry.register_adapter(...)`

---

## 2. Minimal backend layout

Create a new backend folder under `primus/backends/`:

```text
primus/backends/dummy/
├── __init__.py
├── dummy_adapter.py
└── dummy_pretrain_trainer.py
```

This mirrors the pattern used by existing backends (e.g. Megatron, TorchTitan).

---

## 3. Implement the Adapter (`BackendAdapter`)

**File**: `primus/backends/dummy/dummy_adapter.py`

```python
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

from primus.core.backend.backend_adapter import BackendAdapter
from primus.core.backend.backend_registry import BackendRegistry
from primus.modules.module_utils import log_rank_0


class DummyAdapter(BackendAdapter):
    """Minimal adapter for a 'dummy' backend."""

    def __init__(self, framework: str = "dummy"):
        super().__init__(framework)

    def setup_backend_path(self, backend_path=None) -> str:
        """
        Dummy backend lives inside the Primus tree (no third_party submodule),
        so we don't need to modify sys.path or resolve any external path here.

        For real backends that live under third_party/<backend>, you can rely on
        the default implementation in BackendAdapter instead.
        """
        log_rank_0("[Primus:DummyAdapter] setup_backend_path: no-op for in-tree dummy backend")
        return ""

    def prepare_backend(self, config: Any):
        """Run any one-time backend setup (paths, env, hooks)."""
        # Example: run any registered setup hooks for this backend
        BackendRegistry.run_setup(self.framework)
        log_rank_0("[Primus:DummyAdapter] Backend prepared")

    def convert_config(self, config: Any) -> Dict[str, Any]:
        """
        Convert Primus ModuleConfig → backend-specific args.

        For a real backend you would build a structured args object. Here we
        just wrap `config.params` in a SimpleNamespace.
        """
        params = getattr(config, "params", {}) or {}
        backend_args = SimpleNamespace(**params)
        log_rank_0("[Primus:DummyAdapter] Converted Primus params -> dummy backend_args")
        return backend_args

    def load_trainer_class(self, stage: str = "pretrain"):
        """Return the Trainer class for the specified training stage."""
        from primus.backends.dummy.dummy_pretrain_trainer import DummyPretrainTrainer

        log_rank_0("[Primus:DummyAdapter] Loaded trainer class: DummyPretrainTrainer")
        return DummyPretrainTrainer

    def detect_backend_version(self) -> str:
        """Return a version string used by the patch system."""
        return "dummy-0.0.1"
```

Key points:

- Since the dummy backend is implemented directly under `primus.backends.dummy`
  (not in `third_party/`), it overrides `setup_backend_path()` as a **no-op**
  so that the default third_party path resolution is skipped.
- `prepare_backend()` is called **after** setup patches and before config
  conversion. Use it for imports, extra `sys.path` tweaks, or framework init.
- `convert_config()` returns whatever your trainer expects as `backend_args`.
- `load_trainer_class()` imports and returns `DummyPretrainTrainer` directly
  (similar to `MegatronAdapter`), without going through a registry lookup.
- `detect_backend_version()` is used to key version-specific patches.

---

## 4. Implement a runnable trainer (`BaseTrainer`)

**File**: `primus/backends/dummy/dummy_pretrain_trainer.py`

```python
from typing import Any

from primus.core.trainer.base_trainer import BaseTrainer
from primus.modules.module_utils import log_rank_0


class DummyPretrainTrainer(BaseTrainer):
    """Minimal runnable trainer for the dummy backend."""

    def __init__(self, backend_args: Any):
        # BaseTrainer stores backend_args and reads torchrun env (rank, world_size, etc.)
        super().__init__(backend_args=backend_args)
        self._initialized = False

    def setup(self):
        # Optional pre-init setup (e.g., logging, sanity checks)
        log_rank_0(f"[DummyPretrainTrainer] setup() on rank={self.rank}")

    def init(self):
        # Build your model / optimizer / dataloader here in a real backend.
        log_rank_0("[DummyPretrainTrainer] init()")
        self._initialized = True

    def train(self):
        if not self._initialized:
            raise RuntimeError("DummyPretrainTrainer.init() must be called before train().")

        log_rank_0("[DummyPretrainTrainer] train()")
        # Real training loop would go here.
        log_rank_0("[DummyPretrainTrainer] training finished successfully.")

    def cleanup(self, on_error: bool = False):
        # Optional cleanup logic (close files, finalize logging, etc.)
        status = "error" if on_error else "success"
        log_rank_0(f"[DummyPretrainTrainer] cleanup(on_error={status})")
```

Why this matches the core architecture:

- `BaseTrainer.__init__` reads distributed env from `get_torchrun_env()`.
- `PrimusRuntime` drives the lifecycle: `setup → init → train → cleanup`
  and runs patch phases around these steps.
- Your trainer only needs to implement `setup/init/train/cleanup` using
  `backend_args` and the resolved env info.

---

## 5. Register the adapter in BackendRegistry

**File**: `primus/backends/dummy/__init__.py`

```python
from primus.backends.dummy.dummy_adapter import DummyAdapter
from primus.core.backend.backend_registry import BackendRegistry


# Register adapter (backend name → adapter class)
BackendRegistry.register_adapter("dummy", DummyAdapter)
```

At runtime, when `framework: dummy` is requested:

- `BackendRegistry.get_adapter("dummy")` lazily imports `primus.backends.dummy`
  (this file), which calls `register_adapter("dummy", DummyAdapter)`.
- The adapter instance is created and used by `PrimusRuntime` to:
  - set up backend path
  - run setup hooks
  - build `backend_args`
  - load and construct the trainer

---

## 6. Minimal config example

Create an experiment YAML (simplified; full example in Section 7):

```yaml
modules:
  pre_trainer:
    framework: dummy
    config: dummy_trainer.yaml
    model: dummy_8B.yaml
```

Run:

```bash
primus train pretrain --config examples/dummy/configs/dummy_8B-pretrain.yaml
```

You should see logs similar to:

- `[Primus:dummy] sys.path.insert → .../third_party/dummy`
- `[Primus:dummy] Backend prepared`
- `[DummyPretrainTrainer] setup()`
- `[DummyPretrainTrainer] init()`
- `[DummyPretrainTrainer] train()`

---

## 7. Example end-to-end YAML configs

This example mirrors the Megatron pattern:

- The **top-level experiment config** lives under `examples/<backend>/configs/...`
- The **module config** is resolved from `primus/configs/modules/{framework}/`
- The **model config** is resolved from `primus/configs/models/{framework}/`

### 7.1 Top-level experiment config

**File 1**: `examples/dummy/configs/dummy_8B-pretrain.yaml`

```yaml
work_group: ${PRIMUS_TEAM:local}
user_name: ${PRIMUS_USER:local}
exp_name: ${PRIMUS_EXP_NAME:dummy_8B-pretrain}
workspace: ${PRIMUS_WORKSPACE:./output}

modules:
  pre_trainer:
    framework: dummy
    config: dummy_trainer.yaml

    # model to run
    model: dummy_8B.yaml

    overrides:
      # log / debug
      stderr_sink_level: DEBUG

      # example training overrides (merged into module params)
      train_iters: 100
      global_batch_size: 32
      micro_batch_size: 4
      seq_length: 1024
      hello: world
```

### 7.2 Module-level trainer config

**File 2**: `primus/configs/modules/dummy/dummy_trainer.yaml`

```yaml
extends:
  - trainer_base.yaml        # optional, if you have a shared base; otherwise omit

train_iters: 1000
global_batch_size: 16
micro_batch_size: 1
seq_length: 512

log_interval: 1
save_interval: 100
```

This file defines **default training hyperparameters** for the `dummy` backend.
Fields under `modules.pre_trainer.overrides` in the top-level config will be
deep-merged on top of these defaults.

### 7.3 Model-level config

**File 3**: `primus/configs/models/dummy/dummy_8B.yaml`

```yaml
extends: []

model_name: dummy_8B
vocab_size: 32000
hidden_size: 4096
num_layers: 32
num_attention_heads: 32
```

This file plays the same role as Megatron model configs under
`primus/configs/models/megatron/`. It is loaded via:

- `modules.pre_trainer.model: dummy_8B.yaml`
- resolved as `primus/configs/models/{framework}/dummy_8B.yaml`

### 7.4 Running the example

Run:

```bash
primus train pretrain --config examples/dummy/configs/dummy_8B-pretrain.yaml
```

Primus will:

- load `examples/dummy/configs/dummy_8B-pretrain.yaml`
- resolve `dummy_trainer.config` → `primus/configs/modules/dummy/dummy_trainer.yaml`
- resolve `dummy_trainer.model`  → `primus/configs/models/dummy/dummy_8B.yaml`
- build `module_config.params` from these sources + `overrides`
- call `DummyAdapter.convert_config(params)` to build `backend_args`
- construct `DummyPretrainTrainer(backend_args=...)`
- and execute `setup → init → train → cleanup`.

---

## 8. Checklist for a complete backend

Use this as a quick checklist when adding a new backend:

- [ ] Adapter subclass of `BackendAdapter` implements:
      - `load_trainer_class(stage: str)`
      - `convert_config(params)`
      - `detect_backend_version()`
      - (optionally) overrides `prepare_backend()` / `third_party_dir_name`
- [ ] Trainer subclass of `BaseTrainer` implements:
      - `setup()`, `init()`, `train()`, and optional `cleanup(on_error: bool)`
- [ ] `BackendRegistry.register_adapter(backend, AdapterClass)` is called
      in `primus.backends.<backend>.__init__`
- [ ] At least one unit test is added under `tests/unit_tests/backends/`

Once these are in place, your backend is fully integrated into the Primus
runtime and follows the same lifecycle and patch phases as the built-in
backends.

