## Adding a New TorchTitan Model Config

This guide shows how to add a **new TorchTitan model config** to Primus.

As a concrete example, we’ll add **Qwen3 8B** as a new TorchTitan model,
based on the builtin `Qwen3ModelArgs("8B", ...)` definition in:

```text
third_party/torchtitan/torchtitan/models/qwen3/__init__.py
```

You can replace the names/paths with your actual model.

---

## 1. How TorchTitan configs are wired in Primus

For TorchTitan, there are three layers of YAML involved:

1. **Experiment config** (entry point):

   ```yaml
   # examples/torchtitan/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
   work_group: ${PRIMUS_TEAM:amd}
   user_name: ${PRIMUS_USER:root}
   exp_name: ${PRIMUS_EXP_NAME:llama3.1_8B-pretrain}
   workspace: ./output

   modules:
     pre_trainer:
       framework: torchtitan
       config: pre_trainer.yaml

       # model to run
       model: llama3.1_8B.yaml
       overrides:
         sink_level: null
         file_sink_level: DEBUG
         stderr_sink_level: INFO

         metrics:
           log_freq: 1
           enable_wandb: false

         lr_scheduler:
           warmup_steps: 10

         training:
           local_batch_size: 4
           seq_len: 8192
           mock_data: false
           steps: 50
   ```

2. **Module config** (trainer-level defaults):

   ```yaml
   # primus/configs/modules/torchtitan/pre_trainer.yaml
   extends:
     - ../module_base.yaml
     - quantize.yaml

   training:
     mock_data: true
     dataset: c4
     seq_len: 2048
     steps: 10000
   # ...
   ```

3. **Model config** (TorchTitan model definition):

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

At runtime:

- `modules.pre_trainer.model: llama3.1_8B.yaml` is resolved as  
  `primus/configs/models/torchtitan/llama3.1_8B.yaml`.
- The TorchTitan launcher reads the `job` and `model` sections and wires up
  the actual PyTorch model + training loop.

Your new model config will plug into this same chain.

---

## 2. Decide the model you want to add

Pick a model you want TorchTitan to train, for example:

- Some small LLaMA-like model on Hugging Face
- A Qwen-style model
- Your in‑house Hugging Face‑compatible checkpoint

You mainly need to know:

- The **model name / family** (e.g. `llama3`, `qwen3`, …)
- The **size / flavor** you want to expose (e.g. `8B`, `1.7b`)
- The **Hugging Face assets path** (e.g. `Qwen/Qwen3-8B`)

For Qwen3 8B specifically, TorchTitan already defines the architecture in code:

```python
# third_party/torchtitan/torchtitan/models/qwen3/__init__.py
"8B": Qwen3ModelArgs(
    vocab_size=151936,
    max_seq_len=4096,
    head_dim=128,
    dim=4096,
    n_layers=36,
    n_heads=32,
    n_kv_heads=8,
    qk_norm=True,
    hidden_dim=12288,
    rope_theta=1000000,
),
```

TorchTitan’s model config is intentionally light‑weight:
most architecture details live in the underlying TorchTitan implementation.

> **Important**
>
> - In Primus, **TorchTitan can only train models that are implemented inside
>   the TorchTitan codebase itself** (like `llama3`, `qwen3`, `deepseek_v3`, …).
> - The YAML under `primus/configs/models/torchtitan/` does **not** define
>   new architectures; it only selects and configures models that already exist
>   in TorchTitan’s Python code.
> - If a model family / size is not defined in TorchTitan (no corresponding
>   `*ModelArgs` entry), you cannot enable it just by adding a new YAML file
>   in Primus—you would first need TorchTitan to add support for that model.

---

## 3. Create `qwen3_8B.yaml` under TorchTitan models

Create a new file (locally) under:

```text
primus/configs/models/torchtitan/qwen3_8B.yaml
```

with a minimal template like:

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

Field meanings:

- **job.dump_folder**:
  - Where TorchTitan will dump logs / checkpoints for this job.
- **job.description**:
  - Free‑form description shown in logs/metadata.
- **model.name**:
  - Logical model family name understood by TorchTitan
    (compare with existing configs like `llama3`, `qwen3`, …).
- **model.flavor**:
  - Size / variant name (e.g. `8B`, `70B`, `1.7b`, …).
- **model.hf_assets_path**:
  - Hugging Face repo path used by TorchTitan to load weights / tokenizer.
- **model.converters**:
  - Extra TorchTitan converters; `primus_turbo` is the default used in
    the existing Primus examples.

> Note: This file is **not** added to the repo in this guide. It is meant as a
> local example you can create in your workspace.

---

## 4. Point a TorchTitan experiment config at your new model

Pick an existing TorchTitan experiment config (or create one) under
`examples/torchtitan/configs/...` and change only the `model` field.

For example, starting from
`examples/torchtitan/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml`, you can
copy it and create a new experiment config like this:

```yaml
work_group: ${PRIMUS_TEAM:amd}
user_name: ${PRIMUS_USER:root}
exp_name: ${PRIMUS_EXP_NAME:qwen3_8B-pretrain}
workspace: ./output

modules:
  pre_trainer:
    framework: torchtitan
    config: pre_trainer.yaml

    # model to run
    model: qwen3_8B.yaml
    overrides:
      sink_level: null
      file_sink_level: DEBUG
      stderr_sink_level: INFO

      metrics:
        log_freq: 1
        enable_wandb: false

      lr_scheduler:
        warmup_steps: 10

      training:
        local_batch_size: 4
        seq_len: 4096
        mock_data: true
        steps: 50
```

Notes:

- `framework: torchtitan` selects the TorchTitan backend.
- `config: pre_trainer.yaml` reuses the common TorchTitan trainer defaults
  from `primus/configs/modules/torchtitan/pre_trainer.yaml`.
- `model: qwen3_8B.yaml` wires this experiment to your new
  TorchTitan model config.
- The `overrides` section can be copied from an existing example and adjusted
  (batch size, sequence length, steps, logging, etc.).

---

## 5. Run a quick verification run with primus-cli

Use `primus-cli direct` to validate that the config chain resolves correctly
and the TorchTitan training loop can start:

```bash
./primus-cli direct -- \
  train pretrain \
  --config examples/torchtitan/configs/MI300X/qwen3_8B-BF16-pretrain.yaml
```

You should see in the printed configuration and logs that:

- The framework is `torchtitan`.
- The model config being used is `qwen3_8B.yaml`.
- The TorchTitan job / model fields match your new file
  (`job.description`, `model.hf_assets_path`, etc.).

If this runs a few steps without errors, your new TorchTitan model config is
hooked up correctly.

---

## 6. Checklist for adding a TorchTitan model

- [ ] Create `primus/configs/models/torchtitan/<your_model>.yaml`:
      - Define a `job` section (dump folder, description).
      - Define a `model` section (`name`, `flavor`, `hf_assets_path`, `converters`).
- [ ] Create an experiment YAML under `examples/torchtitan/configs/...`
      that references `model: <your_model>.yaml`.
- [ ] Run `./primus-cli direct -- train pretrain --config ...`
      to validate the config chain and run a short training job.

Once these steps are done, your new TorchTitan model config behaves like any
other TorchTitan model in Primus and can be used in experiments, sweeps,
and CI.

