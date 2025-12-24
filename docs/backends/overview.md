## Backend Patch Notes Overview

Primus integrates several large-model backends (Megatron-LM, TorchTitan, JAX MaxText, …) and applies a lightweight patch layer to keep configuration flags consistent with the Primus CLI.
This area of the docs captures those backend-specific switches so they live alongside the rest of the documentation (instead of the historical `primus/README_patch.md` file).

### How to read these docs
- Start with the **Base Module Parameters** table below – every backend module inherits these knobs.
- Jump to the backend-specific page for details on extra CLI/config options and links to the patched source files.
- When editing configs or CLI presets, cross-reference the [Primus CLI Guide](../cli/PRIMUS-CLI-GUIDE.md#configuration) so command examples and backend parameters stay in sync.

### Supported Models

This section lists, at a high level, the model families Primus currently targets on each backend.
For more details and configuration examples, refer to the backend-specific patch notes linked below.

#### Megatron-LM

- **LLaMA family**: LLaMA2, LLaMA3, LLaMA3.1, LLaMA3.3, LLaMA4 (various sizes from 7B up to 405B+)
- **DeepSeek family**: DeepSeek-V2 (lite/base/full) and DeepSeek-V3
- **MoE / Mixtral**: Mixtral-8x7B / 8x22B, large MoE configs (515B, 1T, 2T, 4T) and DeepSeek-style MoE
- **Qwen family**: Qwen2.5 (7B/72B) and Qwen3 (8B/30B/235B variants)
- **Other GPT-style models**: Grok1/2, GPT-OSS 20B and generic `language_model.yaml`

#### TorchTitan

- **LLaMA family**: LLaMA3, LLaMA3.1, LLaMA3.3 (including FP8 variants)
- **DeepSeek family**: DeepSeek-V3 (16B and 671B, FP8 and BF16 configs)
- **Qwen family**: Qwen3 small/medium models (0.6B, 1.7B, 32B)

#### JAX MaxText

- **LLaMA family**: LLaMA2 (7B/70B), LLaMA3 (8B/70B), LLaMA3.3 (70B)
- **DeepSeek family**: DeepSeek-V2 16B
- **MoE / Mixtral**: Mixtral-8x7B
- **Other models**: Grok1 and additional MaxText-supported transformers (see MaxText docs for the full list)

---

### Base Module Parameters
All modules inherit the options defined in [`primus/configs/modules/module_base.yaml`](https://github.com/AMD-AGI/Primus/blob/main/primus/configs/modules/module_base.yaml):

| Argument Name       | Default Value | Description                                                                                |
| ------------------- | ------------- | ------------------------------------------------------------------------------------------ |
| `trainable`         | `false`       | Whether the module participates in training.                                              |
| `sink_level`        | `null`        | Global sink level for logging. Overrides `file_sink_level` and `stderr_sink_level` if set. |
| `file_sink_level`   | `DEBUG`       | Logging level for file sink (log files).                                                  |
| `stderr_sink_level` | `INFO`        | Logging level for stderr/console output.                                                  |

### Backend Index
- [Megatron-LM Patch Notes](./megatron/patch-notes.md)
- [TorchTitan Patch Notes](./torchtitan/patch-notes.md)
- [JAX MaxText Patch Notes](./maxtext/patch-notes.md)
