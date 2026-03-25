# Adding Model Configurations

This guide explains how to add **model configuration YAML** for each Primus training backend. Model presets live under `primus/configs/models/<framework>/` and are referenced from **experiment** YAML under `examples/<framework>/configs/...`. Backend-specific parameter references:

- [Megatron parameters](../03-configuration-reference/megatron-parameters.md)
- [TorchTitan parameters](../03-configuration-reference/torchtitan-parameters.md)
- [MaxText parameters](../03-configuration-reference/maxtext-parameters.md)
- [Megatron Bridge parameters](../03-configuration-reference/megatron-bridge-parameters.md)

---

## Overview: three-layer configuration

For each backend, Primus composes configuration in three layers:

1. **Experiment config** (entry point): `examples/<backend>/configs/<GPU SKU>/<name>.yaml` — selects `framework`, `config` (module preset), `model` (model preset), and `overrides`.
2. **Module config** (trainer defaults): `primus/configs/modules/<framework>/<module>.yaml` — training loop defaults, logging, optimizer blocks, and backend-specific knobs.
3. **Model config** (architecture and assets): `primus/configs/models/<framework>/<model>.yaml` — architecture fields and tokenizer or Hugging Face paths, shaped differently per backend (see sections below).

At runtime, `modules.<module>.model: <file>.yaml` resolves to `primus/configs/models/<framework>/<file>.yaml` and is merged into module parameters before the backend adapter converts them.

---

## Adding a Megatron model

### How Megatron configs are wired

1. **Experiment config** (entry point):

   ```yaml
   # examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
   modules:
     pre_trainer:
       framework: megatron
       config: pre_trainer.yaml           # module-level trainer config

       # model to run
       model: llama3.1_8B.yaml            # model config name
   ```

2. **Module config** (trainer-level defaults): `primus/configs/modules/megatron/pre_trainer.yaml` — extends shared bases and sets Megatron training defaults.

3. **Model config** (architecture + tokenizer):

   ```yaml
   # primus/configs/models/megatron/llama3.1_8B.yaml
   extends:
     - llama3_8B.yaml

   tokenizer_type: HuggingFaceTokenizer
   tokenizer_model: meta-llama/Llama-3.1-8B

   max_position_embeddings: 131072
   ```

At runtime, `modules.pre_trainer.model: llama3.1_8B.yaml` resolves to `primus/configs/models/megatron/llama3.1_8B.yaml`. The `extends` chain pulls in parent files (for example `llama3_8B.yaml` → `llama3_base.yaml` → `llama_base.yaml`).

### Files you typically add

| Artifact | Purpose |
| -------- | ------- |
| **Model preset** (required) | New YAML under `primus/configs/models/megatron/` — architecture, tokenizer, and optional `extends`. |
| **Experiment config** (required) | New or copied YAML under `examples/megatron/configs/MI300X/` or `MI355X/` — points `model:` at your preset and sets `overrides` (batch size, precision, parallelism, `mock_data`, and so on). |
| **Module preset** (optional) | Only if you need trainer defaults that differ from `pre_trainer.yaml` — new file under `primus/configs/modules/megatron/` and reference it as `config:` in the experiment. |

### Example: TinyLlama 1.1B from Hugging Face

Assume Hugging Face repo `TinyLlama/TinyLlama-1.1B-Chat-v1.0` is not yet represented in Primus. You can add a local model preset (not necessarily committed upstream) as follows.

**1. Decide the architecture**

Because TinyLlama is not shipped as a Megatron preset in Primus, you can:

- **Option A (recommended):** extend `language_model.yaml` and set all architecture fields explicitly.
- **Option B:** extend the closest existing model (for example a LLaMA-style preset) and override differing fields.

**2. Map Hugging Face `config.json` to Megatron keys**

Read from the Hugging Face model (typically `config.json` or the model card):

| Hugging Face / concept | Megatron YAML (typical keys) |
| ---------------------- | ---------------------------- |
| `hidden_size` | `hidden_size` |
| `intermediate_size` | `ffn_hidden_size` |
| `num_attention_heads` | `num_attention_heads` |
| `num_hidden_layers` | `num_layers` |
| `num_key_value_heads` | Use with `num_attention_heads` to set `num_query_groups` (often `num_attention_heads / num_key_value_heads`) |
| `max_position_embeddings` | `max_position_embeddings` |

**3. Create `tinyllama_1.1B.yaml`**

Path: `primus/configs/models/megatron/tinyllama_1.1B.yaml`

```yaml
extends:
  - language_model.yaml        # generic Megatron language model base

tokenizer_type: HuggingFaceTokenizer
tokenizer_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0

hidden_size: 2048
ffn_hidden_size: 5632                # intermediate_size in HF config.json
num_attention_heads: 32
num_layers: 22                       # num_hidden_layers in HF config.json
num_query_groups: 8                  # e.g. 32 / 4 if HF has 4 KV heads

max_position_embeddings: 2048
position_embedding_type: rope
```

**4. Point an experiment at the new model**

Copy an existing experiment (for example `examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml`) and set `model:` to your preset. Use **mock data** first for a quick sanity check:

```yaml
work_group: ${PRIMUS_TEAM:amd}
user_name: ${PRIMUS_USER:root}
exp_name: ${PRIMUS_EXP_NAME:tinyllama_1.1B-pretrain}
workspace: ${PRIMUS_WORKSPACE:./output}

modules:
  pre_trainer:
    framework: megatron
    config: pre_trainer.yaml

    model: tinyllama_1.1B.yaml
    overrides:
      save: null
      disable_last_saving: true
      stderr_sink_level: DEBUG

      mock_data: true
      train_iters: 50
      micro_batch_size: 2
      global_batch_size: 128

      seq_length: 2048
```

**5. Run verification**

```bash
./primus-cli direct -- \
  train pretrain \
  --config examples/megatron/configs/MI300X/tinyllama_1.1B-pretrain.yaml
```

Confirm in logs that `framework` is `megatron`, the resolved model file is `tinyllama_1.1B.yaml`, and the tokenizer matches your preset.

### Megatron checklist

- [ ] Choose an appropriate base under `primus/configs/models/megatron/` (`language_model.yaml` or a close LLaMA-style model).
- [ ] Set `tokenizer_type`, `tokenizer_model`, and architecture fields aligned with Hugging Face.
- [ ] Add or update an experiment YAML under `examples/megatron/configs/...` with `model: <your_model>.yaml`.
- [ ] Run `./primus-cli direct -- train pretrain --config ...` to validate resolution and a short run.

---

## Adding a TorchTitan model

### How TorchTitan configs are wired in Primus

1. **Experiment config**:

   ```yaml
   # examples/torchtitan/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
   modules:
     pre_trainer:
       framework: torchtitan
       config: pre_trainer.yaml

       model: llama3.1_8B.yaml
       overrides:
         training:
           local_batch_size: 4
           seq_len: 8192
           mock_data: false
           steps: 50
   ```

2. **Module config**: `primus/configs/modules/torchtitan/pre_trainer.yaml` — training defaults, quantization fragments, and TorchTitan-oriented structure.

3. **Model config** — `job` and `model` sections consumed by the TorchTitan launcher:

   ```yaml
   # primus/configs/models/torchtitan/llama3.1_8B.yaml
   job:
     dump_folder: "./outputs"
     description: "Llama 3.1 8B training"

   model:
     name: "llama3"
     flavor: "8B"
     hf_assets_path: "meta-llama/Llama-3.1-8B"
     converters:
       - primus_turbo
   ```

At runtime, `modules.pre_trainer.model: llama3.1_8B.yaml` resolves to `primus/configs/models/torchtitan/llama3.1_8B.yaml`. The launcher uses `job` and `model` to wire the PyTorch model and training loop.

### Mapping from Hugging Face to TorchTitan

You need:

- **Model family** (`model.name`): must match a family implemented in TorchTitan (for example `llama3`, `qwen3`, `deepseek_v3`).
- **Flavor** (`model.flavor`): a size key defined in TorchTitan code (for example `8B`, `70B`, `1.7b`) — see `third_party/torchtitan/torchtitan/models/<family>/`.
- **Hugging Face assets** (`model.hf_assets_path`): repository used to load weights and tokenizer.

**Important limitations**

- TorchTitan can only train models that are **implemented in the TorchTitan codebase**. The YAML under `primus/configs/models/torchtitan/` does **not** define new architectures; it selects and configures existing `*ModelArgs` entries.
- If a family or flavor is missing in TorchTitan, you cannot enable it with YAML alone — extend TorchTitan first, then add a Primus preset.

### Example: Qwen3 8B preset

TorchTitan may already define Qwen3 8B in code (for example in `third_party/torchtitan/torchtitan/models/qwen3/__init__.py`). Add a local preset:

**File:** `primus/configs/models/torchtitan/qwen3_8B.yaml`

```yaml
job:
  dump_folder: "./outputs"
  description: "Qwen3 8B training (TorchTitan)"

model:
  name: "qwen3"
  flavor: "8B"
  hf_assets_path: "Qwen/Qwen3-8B"
  converters:
    - primus_turbo
```

**Experiment snippet** (copy from an existing TorchTitan example and change `model:`):

```yaml
modules:
  pre_trainer:
    framework: torchtitan
    config: pre_trainer.yaml
    model: qwen3_8B.yaml
    overrides:
      training:
        local_batch_size: 4
        seq_len: 4096
        mock_data: true
        steps: 50
```

**Run:**

```bash
./primus-cli direct -- \
  train pretrain \
  --config examples/torchtitan/configs/MI300X/qwen3_8B-BF16-pretrain.yaml
```

(Use a path that matches the experiment file you create.)

### TorchTitan checklist

- [ ] Define `job` (for example `dump_folder`, `description`) and `model` (`name`, `flavor`, `hf_assets_path`, `converters`).
- [ ] Add an experiment under `examples/torchtitan/configs/...` referencing `model: <your_model>.yaml`.
- [ ] Run `./primus-cli direct -- train pretrain --config ...` for a short job.

---

## Adding a MaxText model

MaxText (JAX) model presets in Primus are intentionally thin: they set **`model_name`** and **`tokenizer_path`** (and extend `model_base.yaml`) so MaxText can load its own architecture tables when available.

**Typical model preset**

Path pattern: `primus/configs/models/maxtext/<name>.yaml`

```yaml
extends:
  - model_base.yaml

model_name: "llama3-8b"
tokenizer_path: "meta-llama/Meta-Llama-3-8B"
```

Comments in `primus/configs/models/maxtext/model_base.yaml` explain that architecture parameters are resolved from MaxText’s `configs/models/<model_name>.yml` when present, or from Primus overrides as appropriate.

**Experiment wiring**

Experiments reference the preset the same way as other backends, for example:

```yaml
modules:
  pre_trainer:
    framework: maxtext
    config: pre_trainer.yaml
    model: llama3_8B.yaml
```

**Supported architectures**

For the authoritative list of model names and architectures MaxText supports, see the [MaxText](https://github.com/google/maxtext) repository and upstream documentation. Primus examples under `examples/maxtext/configs/MI300X/` and `MI355X/` illustrate which presets are exercised in this tree.

---

## Adding a Megatron Bridge model (post-training)

Megatron Bridge model presets are small YAML files that select a **recipe**, **flavor**, and **Hugging Face path**, plus optional dataset blocks.

**Example preset** (`primus/configs/models/megatron_bridge/qwen3_8b.yaml`):

```yaml
recipe: qwen.qwen3
flavor: qwen3_8b_finetune_config
hf_path: Qwen/Qwen3-8B

dataset:
  dataset_name: "rajpurkar/squad"
```

| Field | Role |
| ----- | ---- |
| `recipe` | Logical recipe module (for example `qwen.qwen3`, `llama.llama3`). |
| `flavor` | Named configuration within the recipe (for example `qwen3_8b_finetune_config`). |
| `hf_path` | Hugging Face model id for weights and tokenizer. |

**Experiment** (pattern from `examples/megatron_bridge/configs/`):

```yaml
modules:
  post_trainer:
    framework: megatron_bridge
    config: sft_trainer.yaml
    model: qwen3_8b.yaml
    overrides:
      precision_config: bf16_mixed
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
```

See [Megatron Bridge parameters](../03-configuration-reference/megatron-bridge-parameters.md) for the full override surface.

---

## Testing new models

Use a staged approach so failures are easy to localize.

| Stage | Goal | Typical settings |
| ----- | ---- | ---------------- |
| **Mock or synthetic data** | Validate config resolution, tokenizer, and a few steps without real datasets. | Megatron: `mock_data: true`. TorchTitan: `training.mock_data: true`. MaxText: `dataset_type: "synthetic"` in overrides where applicable. Keep `train_iters` / `steps` small. |
| **Single-GPU** | Confirm numerics and memory before scaling. | Set tensor and pipeline parallelism to 1 in overrides; use one process / one device per your launcher docs. |
| **Multi-GPU** | Match production parallelism. | Set `tensor_model_parallel_size`, `pipeline_model_parallel_size`, expert / context parallel sizes, or MaxText `ici_*` / `dcn_*` fields as required by the model size and hardware. |

Cross-check [Parallelism configuration](../04-technical-guides/parallelism-configuration.md) and [Model support matrix](./model-support-matrix.md) when moving from single-device to multi-device runs.
