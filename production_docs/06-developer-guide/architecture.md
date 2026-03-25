# Architecture Overview

This document describes how the Primus training framework is structured: CLI and configuration, the core runtime orchestrator, backend adapters, trainer lifecycle, and the patch system.

## 1. System Overview

Primus is organized into three conceptual layers: runtime launch (how processes and GPUs are started), hooks and patches (environment and in-process adjustments), and task execution (CLI subcommands that drive training and utilities).

```
┌─────────────────────────────────────────────────────┐
│              Runtime Layer (runner/)                  │
│         direct | container | slurm                   │
│    GPU detection, env setup, distributed launch      │
├─────────────────────────────────────────────────────┤
│           Hook / Patch System                        │
│    runner/helpers/hooks/ | primus/core/patches/       │
│    Pre/post processing, runtime monkey-patches       │
├─────────────────────────────────────────────────────┤
│           Task Execution Layer                       │
│        primus/cli/subcommands/                       │
│    train | benchmark | preflight | projection        │
└─────────────────────────────────────────────────────┘
```

The repository also provides shell entrypoints under `runner/` (for example `primus-cli-direct.sh`, `primus-cli-container.sh`, `primus-cli-slurm.sh`) that prepare the environment and invoke the Python CLI.

## 2. CLI and Plugin System

- **Entry point:** `primus/cli/main.py` is the unified CLI entry. It discovers subcommand modules under `primus/cli/subcommands/` with `pkgutil.walk_packages`, skipping modules whose leaf name starts with `_`.
- **Registration contract:** Each subcommand module exposes `register_subcommand(subparsers)` and must return the configured parser. The parser must call `set_defaults(func=run)` so `main()` can dispatch to the handler.
- **Parsing:** The CLI uses the standard library `argparse` only (no Click or Typer).
- **Unknown arguments:** `main()` calls `parse_known_args()`. For selected subcommands (`train`, `projection`, `preflight`), trailing tokens are passed through to the handler as overrides; for other commands, unknown arguments are rejected.

## 3. Configuration Pipeline

Configuration flows from experiment YAML to a resolved structure consumed by the runtime.

1. **CLI** parses `--config` / `--exp` (required for train flows) pointing at an experiment YAML file.
2. **`load_primus_config()`** (used by `PrimusRuntime`) delegates to **`PrimusParser.parse()`** in `primus/core/launcher/parser.py`. The parser loads the experiment file via **`yaml_utils.parse_yaml_to_namespace()`**, which uses **`primus/core/config/yaml_loader.parse_yaml()`** for `${VAR}` / `${VAR:default}` substitution and `extends:` inheritance with deep merge.
3. **Per trainer module** (names containing `trainer`, for example `pre_trainer`):
   - **`PresetLoader.load()`** loads the module preset from `primus/configs/modules/<framework>/<config>.yaml`.
   - **`PresetLoader.load()`** loads the model preset from `primus/configs/models/<framework>/<model>.yaml`.
   - Each preset is loaded through the same YAML pipeline (env substitution and `extends:` chains).
4. **`parse_platform()`** merges platform settings from `primus/configs/platforms/` (defaulting to `platform_azure.yaml` when the experiment omits `platform`).
5. **CLI overrides:** For `primus train`, `main()` passes `unknown_args` into the train handler. `PrimusRuntime` parses them with `parse_cli_overrides()` and **deep-merges** them into `module_config.params`.
6. **Result:** A resolved configuration where each module exposes a **`params`** namespace (`SimpleNamespace`) for training parameters, produced by `_normalize_module_for_runtime()` in `primus/core/config/primus_config.py`.

The object returned from `load_primus_config()` is a lightweight `SimpleNamespace` (not `PrimusConfig`), with `modules` as a **list** of module configs, each tagged with a `.name` field.

## 4. Core Runtime (PrimusRuntime)

`primus/core/runtime/train_runtime.py` defines **`PrimusRuntime`**, the main orchestrator for the new core training path. Execution for a single module follows this flow:

1. **`load_primus_config()`** loads and validates the experiment; **`get_module_config()`** selects the requested module (for example `pre_trainer` or `post_trainer`).
2. **`_apply_overrides()`** merges CLI overrides into `module_config.params`.
3. **`_initialize_environment()`** ensures the data directory exists and calls **`setup_training_env()`** (Hugging Face cache and related setup).
4. **`_initialize_distributed_context()`** reads torchrun-style rank and master information via **`get_torchrun_env()`**.
5. **`_initialize_logging()`** initializes worker logging.
6. **`BackendRegistry.get_adapter(framework)`** resolves the **`BackendAdapter`** (lazy-importing `primus.backends.<name>` if needed).
7. **`adapter.setup_backend_path()`** inserts the backend tree on `sys.path` (CLI `--backend_path`, then `BACKEND_PATH`, then `third_party/<dir>` under the repo root).
8. **`adapter.prepare_backend()`** runs backend setup hooks (via **`BackendRegistry.run_setup()`** by default).
9. **`adapter.convert_config(module_config.params)`** produces **`backend_args`** for the trainer.
10. **`run_patches(phase="build_args", ...)`** runs registered patches; backend version detection runs when patches first need it (**`adapter.detect_backend_version()`** via **`_get_backend_version()`**).
11. **`merge_namespace()`** merges `backend_args` into `module_config.params` (backend wins on conflicts); **`adapter.load_trainer_class(stage)`** resolves the trainer class (default stage `pretrain`).
12. **`TrainerClass(backend_args=backend_args)`** constructs the trainer.
13. **`run_patches(phase="setup")`** then **`trainer.setup()`**.
14. **`trainer.init()`**.
15. **`run_patches(phase="before_train")`** then **`trainer.train()`** then **`run_patches(phase="after_train")`** then **`trainer.cleanup()`**.
16. On failure, **`_safe_cleanup()`** calls **`trainer.cleanup(on_error=True)`** when possible.

## 5. Backend System

- **`BackendAdapter`** (`primus/core/backend/backend_adapter.py`) is the abstract integration surface. Subclasses implement **`convert_config()`**, **`load_trainer_class()`**, and **`detect_backend_version()`**. Shared behavior includes **`setup_backend_path()`** and a default **`prepare_backend()`** that runs registered setup hooks.
- **`BackendRegistry`** (`primus/core/backend/backend_registry.py`) maps backend names to adapter classes, supports **lazy import** of `primus.backends.<backend>`, and stores optional **setup hooks** per backend.
- **Registered adapters** (via each backend package’s `__init__.py` calling **`BackendRegistry.register_adapter()`**): **`megatron`**, **`torchtitan`**, **`maxtext`**, **`megatron_bridge`**, **`hummingbirdxt`**.
- Backend code lives under **`primus/backends/<name>/`**. Importing the package registers the adapter and any trainers or hooks that package defines.

## 6. Trainer Lifecycle

- **`BaseTrainer`** (`primus/core/trainer/base_trainer.py`) defines the abstract lifecycle: **`setup()`**, **`init()`**, **`train()`**, and **`cleanup(on_error=False)`**. The constructor stores **`backend_args`** and reads distributed settings from **`get_torchrun_env()`**.
- Concrete trainers (for example Megatron or TorchTitan pretrain classes) subclass **`BaseTrainer`** and implement the abstract methods.
- **`PrimusRuntime`** drives **`setup` → `init` → `train` → `cleanup`**, with patch phases inserted around **`setup`**, **`train`**, and **`cleanup`** as described above.

## 7. Patch System

- **`PatchRegistry`** (`primus/core/patches/patch_registry.py`) stores **`FunctionPatch`** objects keyed by backend and phase, with wildcard buckets (`None`) for patches that apply broadly.
- The **`@register_patch`** decorator registers a patch with metadata (priority, optional version patterns, tags).
- **`run_patches()`** (`primus/core/patches/patch_runner.py`) collects applicable patches, filters by **`PatchContext`**, sorts by **priority**, and runs handlers. It accepts an optional **`enabled_ids`** list; if omitted, behavior is controlled by **`PRIMUS_PATCHES`**:
  - unset or **`all`**: all patches
  - **`none`**: disable all
  - comma-separated IDs: only those patches
- **Phases** used by the core runtime include **`build_args`**, **`setup`**, **`before_train`**, and **`after_train`**.
- Patch implementations are typically colocated with backends under **`primus/backends/<backend>/patches/`**.

## 8. Legacy Runtime

Setting **`PRIMUS_TRAIN_RUNTIME=legacy`** selects the legacy pretrain path for **`primus train pretrain`**. That path uses **`primus/pretrain.py`** (**`launch_pretrain_from_cli()`**) and the **`primus/modules/`** stack (**`BaseModule`**-style composition).

If **`PRIMUS_TRAIN_RUNTIME`** is unset or invalid, the implementation **defaults to `core`** for pretrain (the **`PrimusRuntime`** path). The docstring on **`_resolve_pretrain_runtime()`** mentions a possible framework-specific default; the current code applies **`core`** uniformly when the environment variable is not set to `legacy` or `core`.

## 9. Key Source Files

| Path | Role |
|------|------|
| `primus/cli/main.py` | CLI entry, subcommand discovery, dispatch |
| `primus/cli/subcommands/train.py` | `train` subcommand; chooses core vs legacy pretrain; `posttrain` via `PrimusRuntime` |
| `primus/core/launcher/parser.py` | **`PrimusParser`**: experiment, platform, and module preset loading |
| `primus/core/config/preset_loader.py` | **`PresetLoader`**: load framework presets from `primus/configs/` |
| `primus/core/config/yaml_loader.py` | YAML load with env substitution and `extends` |
| `primus/core/config/primus_config.py` | **`load_primus_config()`**, **`get_module_config()`**, module normalization |
| `primus/core/runtime/train_runtime.py` | **`PrimusRuntime`**, **`TrainContext`** |
| `primus/core/backend/backend_adapter.py` | **`BackendAdapter`** ABC |
| `primus/core/backend/backend_registry.py` | **`BackendRegistry`** |
| `primus/core/trainer/base_trainer.py` | **`BaseTrainer`** ABC |
| `primus/core/patches/patch_registry.py` | **`PatchRegistry`**, **`@register_patch`** |
| `primus/core/patches/patch_runner.py` | **`run_patches()`**, **`PRIMUS_PATCHES`** parsing |
| `runner/primus-cli-*.sh` | Shell wrappers for direct, container, and Slurm launch |

For day-to-day contribution workflows (style, tests, CI), see [Contributing Guide](contributing.md) and [Testing Guide](testing.md).
