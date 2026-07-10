# Configuration System

Primus experiments are described in YAML. The loader resolves **environment variables**, **`extends:` inheritance**, and **module/model/platform presets** before training starts. This document focuses on the Python configuration pipeline (`primus/core/config/` and `primus/core/launcher/parser.py`).

**Related documentation**

| Topic | Location |
| --- | --- |
| CLI launcher and `--config` | [CLI Reference](cli-reference.md) |
| Backend parameter references | [Megatron parameters](../03-configuration-reference/megatron-parameters.md), [TorchTitan parameters](../03-configuration-reference/torchtitan-parameters.md), [MaxText parameters](../03-configuration-reference/maxtext-parameters.md) |

---

## Overview: three-layer YAML

A typical experiment ties together:

1. **Experiment YAML** — your run: identity, workspace, and a `modules` section naming framework presets plus overrides.
2. **Module preset** — training defaults for a backend (optimizer, schedule, parallelism hooks) under `primus/configs/modules/<framework>/`.
3. **Model preset** — architecture and tokenizer metadata under `primus/configs/models/<framework>/`.

All of these are **deep-merged** (see `primus/core/config/yaml_loader.py` and `primus/core/config/merge_utils.py`). A **platform preset** (`primus/configs/platforms/`) maps distributed environment variable names and logging defaults; if omitted, the parser injects `platform_azure.yaml` (see `PrimusParser.parse_platform` in `primus/core/launcher/parser.py`).

---

## Experiment config structure

```yaml
work_group: ${PRIMUS_TEAM:amd}
user_name: ${PRIMUS_USER:root}
exp_name: ${PRIMUS_EXP_NAME:my_experiment}
workspace: ${PRIMUS_WORKSPACE:./output}

modules:
  pre_trainer:
    framework: megatron          # backend name (megatron, torchtitan, maxtext, megatron_bridge, …)
    config: pre_trainer.yaml     # module preset file under primus/configs/modules/<framework>/
    model: llama3_8B.yaml        # model preset file under primus/configs/models/<framework>/
    overrides:                   # training overrides (deep-merged last)
      train_iters: 50
      micro_batch_size: 4
```

Required top-level keys are validated in `PrimusParser.parse_meta_info`: `work_group`, `user_name`, `exp_name`, `workspace`.

---

## Module presets

- **Location:** `primus/configs/modules/<framework>/` (for example `primus/configs/modules/megatron/pre_trainer.yaml`).
- **Purpose:** Default training behavior (iterations, batching, optimizer, logging, parallelism-related flags for that backend).
- **Inheritance:** Use `extends:` to compose files in the same directory (or relative paths). Multiple entries merge in order; the **current file wins** on conflicts (`_apply_extends` in `yaml_loader.py`).

Example chain (excerpt): `pre_trainer.yaml` extends `trainer_base.yaml`, which extends `../module_base.yaml` and other shared fragments (`primus/configs/modules/megatron/trainer_base.yaml`).

---

## Model presets

- **Location:** `primus/configs/models/<framework>/` (for example `primus/configs/models/megatron/llama3_8B.yaml`).
- **Purpose:** Architecture dimensions, tokenizer identifiers, and other model metadata.
- **Inheritance:** Same `extends:` mechanism as modules (for example `llama3_8B.yaml` → `llama3_base.yaml` → …).

---

## Platform presets

- **Location:** `primus/configs/platforms/` (for example `primus/configs/platforms/platform_azure.yaml`).
- **Purpose:** Names of environment variables used for distributed launch (`NNODES`, `NODE_RANK`, `MASTER_ADDR`, …) and defaults such as `master_sink_level` and `workspace`.
- **Default:** If the experiment omits `platform`, the parser sets `config: platform_azure.yaml` (`primus/core/launcher/parser.py`).

---

## Environment variable substitution

`primus/core/config/yaml_loader.py` expands:

| Pattern | Behavior |
| --- | --- |
| `${VAR}` | **Required.** Raises if `VAR` is unset. |
| `${VAR:default}` | Uses `default` when `VAR` is unset. |

After substitution, purely numeric strings may be converted to `int` or `float`.

---

## `extends:` inheritance

For each YAML file:

1. Each path in `extends:` is resolved **relative to the directory of the current file**.
2. Presets are loaded recursively (each may have its own `extends:`).
3. Merge order: earlier presets in the list are merged first; **later presets override earlier ones**; the **current file overrides all** (`_apply_extends`).

`PresetLoader.load` (`primus/core/config/preset_loader.py`) resolves `primus/configs/<config_type>/<framework>/<name>.yaml` and runs the same `parse_yaml` pipeline.

---

## CLI overrides (training)

After the main arguments are parsed, unknown tokens are interpreted as **key=value overrides** and deep-merged into the active training module namespace (`module_cfg.params`). The core runtime applies them in `PrimusRuntime._apply_overrides` (`primus/core/runtime/train_runtime.py`) using `parse_cli_overrides` (`primus/core/utils/arg_utils.py`) followed by `deep_merge`. Both `key=value` and `--key value` forms are accepted. Unknown keys are merged in (not rejected) and forwarded to the backend; the stricter key-existence check in `parse_args` / `_check_keys_exist` (`primus/core/launcher/parser.py`) belongs to a legacy path that the `train` subcommand does not exercise.

Example (conceptual):

```bash
./runner/primus-cli direct -- train pretrain --config exp.yaml \
  --train_iters 100 --micro_batch_size 2
```

---

## Merge priority (training config)

The effective ordering of training parameters is:

1. **CLI overrides** (key=value after the main `train` arguments) — highest. Applied last by the runtime (`PrimusRuntime._apply_overrides`), after preset and experiment merging.
2. **`modules.pre_trainer.overrides`** in the experiment YAML (applied in `PrimusParser.parse_trainer_module`).
3. **Module preset with model preset additions**: the module preset is loaded first, then the model preset is merged in with `allow_override=False`, so duplicate top-level module keys are preserved while non-duplicate model keys are added (`merge_namespace` in `parse_trainer_module`).
4. **Preset chains** via `extends:` inside those files — base layers first, specialized layers later, file body last.

A concise mental model:

**CLI overrides > experiment `overrides` > module preset with model preset additions > each preset's own `extends:` chain > shared bases such as `module_base.yaml`.**

---

## Config resolution walkthrough: `llama2_7B-BF16-pretrain.yaml`

Example experiment: `examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml`.

1. **Load experiment** — `parse_yaml` reads the file; `${PRIMUS_*:…}` placeholders resolve.
2. **Meta** — `work_group`, `user_name`, `exp_name`, `workspace` are checked.
3. **Platform** — If not present, default `platform_azure.yaml` loads from `primus/configs/platforms/`.
4. **Module preset** — `PresetLoader.load("pre_trainer.yaml", "megatron", "modules")` loads `primus/configs/modules/megatron/pre_trainer.yaml` and applies its `extends:` chain (for example `trainer_base.yaml` → …).
5. **Model preset** — `PresetLoader.load("llama2_7B.yaml", "megatron", "models")` loads `primus/configs/models/megatron/llama2_7B.yaml` (which extends `llama2_base.yaml` → `llama_base.yaml` → …).
6. **Merge** — Module and model namespaces are merged for `pre_trainer`.
7. **Experiment overrides** — Keys under `modules.pre_trainer.overrides` in the example file (for example `mock_data: true`, parallelism, LR) are applied on top.
8. **CLI overrides** — Any key=value pairs from the command line are merged last.

---

## How to write a new config for a new model

1. **Add or reuse a model preset** under `primus/configs/models/<framework>/`, using `extends:` from the closest existing architecture (for example copy `llama3_8B.yaml` and adjust hidden size, layers, tokenizer).
2. **Point an experiment YAML at it** — set `modules.pre_trainer.framework`, `config: <module_preset>.yaml`, and `model: <your_model>.yaml`.
3. **Set overrides** in the experiment file for run-specific values (batch sizes, paths, `mock_data`, parallelism). Prefer small experiment files that reference presets instead of duplicating hundreds of keys.
4. **Validate** with `--dry-run` and by tracing the referenced experiment, module, and model presets (see below).
5. **Optional:** add an example under `examples/<backend>/configs/<platform>/` for others to copy.

---

## Debugging config issues

| Technique | What it does |
| --- | --- |
| `--export_config PATH` | Parsed by the training config parser, but the default core `PrimusRuntime` path does not currently write the resolved YAML. Treat this as legacy/future functionality unless your deployment implements it. |
| `./runner/primus-cli --dry-run …` | Shows the launcher command without executing (shell layer). |
| `--debug` | Enables verbose logging for launcher and Python (`PRIMUS_LOG_LEVEL=DEBUG`). |
| Inspect presets | Open the resolved `extends:` chain under `primus/configs/modules/` and `primus/configs/models/` for the framework you use. |

If `${VAR}` substitution fails, set the variable or switch to `${VAR:default}` in the YAML.

---

## Cross-references

- Default launcher YAML: `runner/.primus.yaml`
- YAML loader (env + extends): `primus/core/config/yaml_loader.py`
- Parser and merge: `primus/core/launcher/parser.py`
- Preset paths: `primus/core/config/preset_loader.py`
