# Adding a New Megatron Model Config

This guide walks through how to add a **new Megatron model config** to Primus
for a real small model on Hugging Face that Primus does **not** configure yet.

As a concrete example, we will use:

- Hugging Face repo: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- New Primus model config: `tinyllama_1.1B.yaml`

The goal is that you can follow this doc step by step and end up with a config
that works end‑to‑end with `primus-cli direct`.

---

## 1. How Megatron model configs are wired

For Megatron, there are three layers of YAML config involved:

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

2. **Module config** (trainer-level defaults):

   ```yaml
   # primus/configs/modules/megatron/pre_trainer.yaml
   extends:
     - trainer_base.yaml
   # ...
   ```

3. **Model config** (architecture + tokenizer):

   ```yaml
   # primus/configs/models/megatron/llama3.1_8B.yaml
   extends:
     - llama3_8B.yaml

   tokenizer_type: HuggingFaceTokenizer
   tokenizer_model: meta-llama/Llama-3.1-8B

   max_position_embeddings: 131072
   ```

At runtime:

- `modules.pre_trainer.model: llama3.1_8B.yaml` is resolved as  
  `primus/configs/models/megatron/llama3.1_8B.yaml`.
- `extends: - llama3_8B.yaml` further chains into
  `primus/configs/models/megatron/llama3_8B.yaml`, which in turn extends
  `llama3_base.yaml` and so on.

Your new model config will fit into the same resolution chain.

---

## 2. Decide the architecture for your new model

Because `TinyLlama/TinyLlama-1.1B-Chat-v1.0` is **not** shipped in Primus,
there is no existing Megatron model YAML to extend from. You have two options:

- **Option A (recommended)**: extend from a generic base like
  `language_model.yaml` and explicitly set all TinyLlama architecture fields.
- **Option B**: extend from the closest existing model and override fields.

For clarity, this guide uses **Option A**.

You will need to look up the following values from the TinyLlama config on
Hugging Face (usually in `config.json` or the model card):

- `hidden_size`
- `ffn_hidden_size`
- `num_attention_heads`
- `num_layers`
- `num_query_groups` (if applicable)
- `max_position_embeddings`

You will copy those values into the new Megatron YAML.

---

## 3. Create `tinyllama_1.1B.yaml` under Megatron models

Create a new file (locally) under:

```text
primus/configs/models/megatron/tinyllama_1.1B.yaml
```

with the following template:

```yaml
extends:
  - language_model.yaml        # generic Megatron language model base

tokenizer_type: HuggingFaceTokenizer
tokenizer_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0

hidden_size: 2048
ffn_hidden_size: 5632                # intermediate_size in HF config.json
num_attention_heads: 32
num_layers: 22                       # num_hidden_layers in HF config.json
num_query_groups: 8                  # num_attention_heads / num_key_value_heads = 32 / 4

max_position_embeddings: 2048
position_embedding_type: rope
```

Field meanings:

- **extends**:
  - Re-uses generic language-model defaults.
  - If you prefer, you can instead extend from the closest existing model
    (e.g. `llama3_8B.yaml`) and then override all differing fields.
- **tokenizer_type**:
  - Case where you want to use a Hugging Face tokenizer implementation.
- **tokenizer_model**:
  - Hugging Face model repository to load tokenizer from.
  - Example: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` or your own fine-tuned fork.
> Note: This file is **not** added to the repo in this PR. It is meant as a
> local example you can create in your workspace.

---

## 4. Point an experiment config at your new model

Pick an existing Megatron experiment config (or create one) under
`examples/megatron/configs/...` and change only the `model` field.

For example, starting from
`examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml`, you can
create a new experiment config like this:

```yaml
work_group: ${PRIMUS_TEAM:amd}
user_name: ${PRIMUS_USER:root}
exp_name: ${PRIMUS_EXP_NAME:tinyllama_1.1B-pretrain}
workspace: ${PRIMUS_WORKSPACE:./output}

modules:
  pre_trainer:
    framework: megatron
    config: pre_trainer.yaml

    # model to run
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

All other overrides (batch size, seq length, LR, etc.) can stay the same
initially; you can tune them later for your new model.

---

## 5. Run a quick verification run with primus-cli

Use `primus-cli direct` to validate that the config chain resolves correctly
and the training loop can start.

```bash
./primus-cli direct -- \
  train pretrain \
  --config examples/megatron/configs/MI300X/tinyllama_1.1B-pretrain.yaml
```

You should see in the printed configuration and logs that:

- The framework is `megatron`.
- The model config being used is `tinyllama_1.1B.yaml`.
- The tokenizer matches your new file.

---

## 6. Checklist for adding a Megatron model

- [ ] Choose an appropriate base model under `primus/configs/models/megatron/`
      (e.g. `language_model.yaml`, or a close existing LLaMA-style model).
- [ ] Create `primus/configs/models/megatron/tinyllama_1.1B.yaml`:
      - `extends: - <base_model>.yaml`
      - set `tokenizer_type`, `tokenizer_model`
      - copy TinyLlama architecture fields (`hidden_size`, `num_layers`, etc.)
- [ ] Update / create an experiment YAML under `examples/megatron/configs/...`
      to reference `model: tinyllama_1.1B.yaml`.
- [ ] Run `./primus-cli direct -- train pretrain --config ...`
      to validate the config chain and run a short training job.

Once these steps are done, your new `tinyllama_1.1B` config behaves like any other
Megatron model in Primus and can be used in experiments, sweeps, and CI.

