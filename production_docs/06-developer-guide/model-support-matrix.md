# Model Support Matrix

This document summarizes which model families Primus targets per backend, lists **checked-in** model presets and example experiment YAML under the repository, and distinguishes **curated examples** from **theoretical** support (a preset or upstream stack may exist without a matching `examples/` entry).

For how to add presets, see [Adding model configurations](./adding-models.md). Backend parameter references: [Megatron](../03-configuration-reference/megatron-parameters.md), [TorchTitan](../03-configuration-reference/torchtitan-parameters.md), [MaxText](../03-configuration-reference/maxtext-parameters.md), [Megatron Bridge](../03-configuration-reference/megatron-bridge-parameters.md).

---

## Overview: supported model families (high level)

The following aligns with the backend overview and the configs present in this tree.

| Backend | Model families (documentation / stack scope) |
| ------- | ---------------------------------------------- |
| **Megatron-LM** | LLaMA2 / LLaMA3 / LLaMA3.1 / LLaMA3.3 / LLaMA4 (sizes from small to 405B+), DeepSeek-V2 (including lite) and DeepSeek-V3, Mixtral MoE and large MoE recipe YAML, Qwen2.5 and Qwen3 MoE, Grok, GPT-OSS, Zebra LLaMA, Mamba, and generic `language_model.yaml` bases. |
| **TorchTitan** | LLaMA3 family (including 3.1), DeepSeek-V3 (16B and 671B recipes in examples), Qwen3 (0.6B–32B in examples). Additional presets exist under `primus/configs/models/torchtitan/` without a matching example (see table). |
| **MaxText (JAX)** | LLaMA2 / LLaMA3 / LLaMA3.3, DeepSeek-V2 16B, Mixtral-8x7B, Grok1, Qwen3 14B / 30B-A3B (per presets and examples). Broader coverage may exist in upstream MaxText; see [MaxText](https://github.com/google/maxtext). |
| **Megatron Bridge** | Post-training recipes for Qwen3 (8B / 32B) and LLaMA 3.1 70B (SFT / LoRA examples). |

**Interpretation:** “Supported” in upstream code can exceed what this repository ships as YAML. Rows below reference **files that exist** under `primus/configs/models/` and `examples/`.

---

## Megatron model configs

Model presets live in `primus/configs/models/megatron/`. Example experiments that reference those presets appear under `examples/megatron/configs/MI300X/` and `examples/megatron/configs/MI355X/`.

| Model name (file) | Preset path | Role | Example experiment dirs | Precision in examples |
| ----------------- | ----------- | ---- | ----------------------- | ---------------------- |
| `deepseek_v2.yaml` | `primus/configs/models/megatron/deepseek_v2.yaml` | Dense model preset | MI300X, MI355X | BF16, FP8 |
| `deepseek_v2_base.yaml` | `primus/configs/models/megatron/deepseek_v2_base.yaml` | Base fragment (`extends` only) | — | — |
| `deepseek_v2_lite.yaml` | `primus/configs/models/megatron/deepseek_v2_lite.yaml` | Model preset | MI300X, MI355X | BF16, FP8 |
| `deepseek_v3.yaml` | `primus/configs/models/megatron/deepseek_v3.yaml` | MoE model preset | MI300X, MI355X | BF16, FP8 |
| `deepseek_v3_base.yaml` | `primus/configs/models/megatron/deepseek_v3_base.yaml` | Base fragment | — | — |
| `glm4_7.yaml` | `primus/configs/models/megatron/glm4_7.yaml` | Model preset | No curated example in this repo | — |
| `gpt_oss_20B.yaml` | `primus/configs/models/megatron/gpt_oss_20B.yaml` | Model preset | MI300X | BF16, FP8 |
| `grok1.yaml` | `primus/configs/models/megatron/grok1.yaml` | Model preset | MI300X, MI355X | BF16, FP8 |
| `grok2.yaml` | `primus/configs/models/megatron/grok2.yaml` | Model preset | MI300X, MI355X | BF16, FP8 |
| `grok_base.yaml` | `primus/configs/models/megatron/grok_base.yaml` | Base fragment | — | — |
| `hybrid_model_base.yaml` | `primus/configs/models/megatron/hybrid_model_base.yaml` | Base fragment | — | — |
| `kimi_k2.yaml` | `primus/configs/models/megatron/kimi_k2.yaml` | Model preset | No curated example in this repo | — |
| `language_model.yaml` | `primus/configs/models/megatron/language_model.yaml` | Generic Megatron LM defaults | Used via `extends` | — |
| `llama2_7B.yaml` | `primus/configs/models/megatron/llama2_7B.yaml` | Model preset | MI300X, MI355X | BF16, FP8 |
| `llama2_70B.yaml` | `primus/configs/models/megatron/llama2_70B.yaml` | Model preset | MI300X, MI355X | BF16, FP8 |
| `llama2_base.yaml` | `primus/configs/models/megatron/llama2_base.yaml` | Base fragment | — | — |
| `llama3_8B.yaml` | `primus/configs/models/megatron/llama3_8B.yaml` | Model preset | MI300X, MI355X | BF16, FP8 |
| `llama3_70B.yaml` | `primus/configs/models/megatron/llama3_70B.yaml` | Model preset | MI300X, MI355X | BF16, FP8 |
| `llama3_base.yaml` | `primus/configs/models/megatron/llama3_base.yaml` | Base fragment | — | — |
| `llama3.1_8B.yaml` | `primus/configs/models/megatron/llama3.1_8B.yaml` | Model preset | MI300X, MI355X | BF16, FP8 |
| `llama3.1_70B.yaml` | `primus/configs/models/megatron/llama3.1_70B.yaml` | Model preset | MI300X, MI355X | BF16, FP8 |
| `llama3.1_405B.yaml` | `primus/configs/models/megatron/llama3.1_405B.yaml` | Model preset | MI300X, MI355X | BF16, FP8 |
| `llama3.2_1B.yaml` | `primus/configs/models/megatron/llama3.2_1B.yaml` | Model preset | MI300X, MI355X | BF16, FP8 |
| `llama3.2_3B.yaml` | `primus/configs/models/megatron/llama3.2_3B.yaml` | Model preset | MI300X, MI355X | BF16, FP8 |
| `llama3.3_70B.yaml` | `primus/configs/models/megatron/llama3.3_70B.yaml` | Model preset | MI300X, MI355X | BF16, FP8 |
| `llama4_17B128E.yaml` | `primus/configs/models/megatron/llama4_17B128E.yaml` | MoE model preset | MI300X, MI355X | BF16, FP8 |
| `llama4_17B16E.yaml` | `primus/configs/models/megatron/llama4_17B16E.yaml` | MoE model preset | MI300X, MI355X | BF16, FP8 |
| `llama4_base.yaml` | `primus/configs/models/megatron/llama4_base.yaml` | Base fragment | — | — |
| `mamba_370M.yaml` | `primus/configs/models/megatron/mamba_370M.yaml` | Model preset | MI300X | Set in experiment overrides |
| `mamba_base.yaml` | `primus/configs/models/megatron/mamba_base.yaml` | Base fragment | — | — |
| `mixtral_8x7B_v0.1.yaml` | `primus/configs/models/megatron/mixtral_8x7B_v0.1.yaml` | MoE model preset | MI300X, MI355X | BF16, FP8 |
| `mixtral_8x22B_v0.1.yaml` | `primus/configs/models/megatron/mixtral_8x22B_v0.1.yaml` | MoE model preset | MI300X, MI355X | BF16, FP8 |
| `mixtral_base.yaml` | `primus/configs/models/megatron/mixtral_base.yaml` | Base fragment | — | — |
| `moe_515B.yaml` | `primus/configs/models/megatron/moe_515B.yaml` | Large MoE template | No curated example in this repo | — |
| `moe_1T.yaml` | `primus/configs/models/megatron/moe_1T.yaml` | Large MoE template | No curated example in this repo | — |
| `moe_2T.yaml` | `primus/configs/models/megatron/moe_2T.yaml` | Large MoE template | No curated example in this repo | — |
| `moe_4T.yaml` | `primus/configs/models/megatron/moe_4T.yaml` | Large MoE template | No curated example in this repo | — |
| `moe_proxy_single_node.yaml` | `primus/configs/models/megatron/moe_proxy_single_node.yaml` | MoE proxy / test template | No curated example in this repo | — |
| `primus_megatron_model.yaml` | `primus/configs/models/megatron/primus_megatron_model.yaml` | Primus Megatron root defaults | Used via `extends` | — |
| `qwen2.5_7B.yaml` | `primus/configs/models/megatron/qwen2.5_7B.yaml` | Model preset | MI300X, MI355X | BF16, FP8 |
| `qwen2.5_72B.yaml` | `primus/configs/models/megatron/qwen2.5_72B.yaml` | Model preset | MI300X, MI355X | BF16, FP8 |
| `qwen2.5_base.yaml` | `primus/configs/models/megatron/qwen2.5_base.yaml` | Base fragment | — | — |
| `qwen3_8B.yaml` | `primus/configs/models/megatron/qwen3_8B.yaml` | Model preset | MI355X (`qwen3_8B-BF16-pretrain.yaml`, `qwen3_8B-FP8-pretrain.yaml`); not under MI300X in this repo | BF16, FP8 |
| `qwen3_30B_A3B.yaml` | `primus/configs/models/megatron/qwen3_30B_A3B.yaml` | MoE model preset | MI300X, MI355X | BF16, FP8 |
| `qwen3_235B_A22B.yaml` | `primus/configs/models/megatron/qwen3_235B_A22B.yaml` | MoE model preset | MI300X, MI355X | BF16, FP8 |
| `zebra_llama_1B.yaml` | `primus/configs/models/megatron/zebra_llama_1B.yaml` | Model preset | MI300X, MI355X | Set in experiment overrides |
| `zebra_llama_3B.yaml` | `primus/configs/models/megatron/zebra_llama_3B.yaml` | Model preset | MI300X, MI355X | Set in experiment overrides |
| `zebra_llama_8B.yaml` | `primus/configs/models/megatron/zebra_llama_8B.yaml` | Model preset | MI300X, MI355X | Set in experiment overrides |

**Parallelism:** Tensor, pipeline, and expert parallel sizes are **not** fixed in model presets; they are set in experiment `overrides` (for example `tensor_model_parallel_size`, `pipeline_model_parallel_size`, `expert_model_parallel_size`). MoE presets such as `qwen3_235B_A22B.yaml` typically require non-default expert parallelism in real runs—see the matching experiment YAML.

---

## TorchTitan model configs

Presets: `primus/configs/models/torchtitan/`. Examples: `examples/torchtitan/configs/MI300X/` and `MI355X/`.

| Model name (file) | Preset path | Example experiment dirs | Precision in examples |
| ----------------- | ----------- | ------------------------- | ---------------------- |
| `deepseek_v3_16b.yaml` | `primus/configs/models/torchtitan/deepseek_v3_16b.yaml` | MI300X, MI355X | BF16 |
| `deepseek_v3_16b-fp8.yaml` | `primus/configs/models/torchtitan/deepseek_v3_16b-fp8.yaml` | MI300X, MI355X | FP8 |
| `deepseek_v3_671b.yaml` | `primus/configs/models/torchtitan/deepseek_v3_671b.yaml` | MI300X, MI355X | (see experiment) |
| `deepseek_v3_671b-fp8.yaml` | `primus/configs/models/torchtitan/deepseek_v3_671b-fp8.yaml` | Preset only; stock examples use `deepseek_v3_671b.yaml` | — |
| `llama3_8B.yaml` | `primus/configs/models/torchtitan/llama3_8B.yaml` | No example in this repo | — |
| `llama3_8B-fp8.yaml` | `primus/configs/models/torchtitan/llama3_8B-fp8.yaml` | No example in this repo | — |
| `llama3_70B.yaml` | `primus/configs/models/torchtitan/llama3_70B.yaml` | No example in this repo | — |
| `llama3_70B-fp8.yaml` | `primus/configs/models/torchtitan/llama3_70B-fp8.yaml` | No example in this repo | — |
| `llama3.1_8B.yaml` | `primus/configs/models/torchtitan/llama3.1_8B.yaml` | MI300X, MI355X | BF16 |
| `llama3.1_8B-fp8.yaml` | `primus/configs/models/torchtitan/llama3.1_8B-fp8.yaml` | MI300X, MI355X | FP8 |
| `llama3.1_70B.yaml` | `primus/configs/models/torchtitan/llama3.1_70B.yaml` | MI300X, MI355X | BF16 |
| `llama3.1_70B-fp8.yaml` | `primus/configs/models/torchtitan/llama3.1_70B-fp8.yaml` | MI300X, MI355X | FP8 |
| `llama3.1_405B.yaml` | `primus/configs/models/torchtitan/llama3.1_405B.yaml` | MI300X, MI355X | BF16 |
| `llama3.1_405B-fp8.yaml` | `primus/configs/models/torchtitan/llama3.1_405B-fp8.yaml` | MI300X, MI355X | FP8 |
| `llama3.2_1B.yaml` | `primus/configs/models/torchtitan/llama3.2_1B.yaml` | No example in this repo | — |
| `llama3.3_70B.yaml` | `primus/configs/models/torchtitan/llama3.3_70B.yaml` | No example in this repo | — |
| `llama3.3_70B-fp8.yaml` | `primus/configs/models/torchtitan/llama3.3_70B-fp8.yaml` | No example in this repo | — |
| `qwen3_0.6b.yaml` | `primus/configs/models/torchtitan/qwen3_0.6b.yaml` | MI300X, MI355X | (see experiment) |
| `qwen3_1.7b.yaml` | `primus/configs/models/torchtitan/qwen3_1.7b.yaml` | MI300X, MI355X | (see experiment) |
| `qwen3_32b.yaml` | `primus/configs/models/torchtitan/qwen3_32b.yaml` | MI300X, MI355X | (see experiment) |

**Parallelism:** Controlled by TorchTitan launch configuration and Primus module overrides (see TorchTitan patch notes and [TorchTitan parameters](../03-configuration-reference/torchtitan-parameters.md)); not embedded in the small `job` / `model` preset alone.

---

## MaxText model configs

Presets: `primus/configs/models/maxtext/`. Examples: `examples/maxtext/configs/MI300X/` and `examples/maxtext/configs/MI355X/`.

| Model name (file) | Preset path | Example experiment dirs |
| ----------------- | ----------- | ------------------------ |
| `deepseek_v2_16B.yaml` | `primus/configs/models/maxtext/deepseek_v2_16B.yaml` | MI300X, MI355X |
| `grok1.yaml` | `primus/configs/models/maxtext/grok1.yaml` | MI300X |
| `llama2_7B.yaml` | `primus/configs/models/maxtext/llama2_7B.yaml` | MI300X, MI355X |
| `llama2_70B.yaml` | `primus/configs/models/maxtext/llama2_70B.yaml` | MI300X, MI355X |
| `llama3_8B.yaml` | `primus/configs/models/maxtext/llama3_8B.yaml` | MI300X, MI355X |
| `llama3_70B.yaml` | `primus/configs/models/maxtext/llama3_70B.yaml` | MI300X, MI355X |
| `llama3.1_405B.yaml` | `primus/configs/models/maxtext/llama3.1_405B.yaml` | MI355X |
| `llama3.3_70B.yaml` | `primus/configs/models/maxtext/llama3.3_70B.yaml` | MI300X, MI355X |
| `mixtral_8x7B.yaml` | `primus/configs/models/maxtext/mixtral_8x7B.yaml` | MI300X, MI355X |
| `qwen3_14B.yaml` | `primus/configs/models/maxtext/qwen3_14B.yaml` | MI300X, MI355X |
| `qwen3_30B_A3B.yaml` | `primus/configs/models/maxtext/qwen3_30B_A3B.yaml` | MI300X, MI355X |
| `model_base.yaml` | `primus/configs/models/maxtext/model_base.yaml` | Extended by other presets (not a standalone run) |

**Parallelism:** JAX / MaxText sharding is configured in experiment overrides (for example `ici_fsdp_parallelism`, `ici_data_parallelism`, `dcn_*` in sample experiments). See [MaxText parameters](../03-configuration-reference/maxtext-parameters.md).

---

## Megatron Bridge model configs (post-training)

Presets: `primus/configs/models/megatron_bridge/`. Examples: `examples/megatron_bridge/configs/MI300X/` and `examples/megatron_bridge/configs/MI355X/`.

| Model name (file) | Preset path | Recipe / flavor (from preset) | Example experiment dirs |
| ----------------- | ----------- | ----------------------------- | ------------------------ |
| `qwen3_8b.yaml` | `primus/configs/models/megatron_bridge/qwen3_8b.yaml` | `qwen.qwen3` / `qwen3_8b_finetune_config` | MI355X |
| `qwen3_32b.yaml` | `primus/configs/models/megatron_bridge/qwen3_32b.yaml` | `qwen.qwen3` / `qwen3_32b_finetune_config` | MI300X, MI355X |
| `llama31_70b.yaml` | `primus/configs/models/megatron_bridge/llama31_70b.yaml` | `llama.llama3` / `llama31_70b_finetune_config` | MI355X |

Example filenames include `*_sft_posttrain.yaml` and `*_lora_posttrain.yaml`; precision such as `bf16_mixed` is set in experiment `overrides`.

---

## Hardware compatibility (example directories)

Curated example layouts under `examples/` use GPU SKU subdirectories. As of this document:

| GPU SKU | `examples/megatron/configs/` | `examples/torchtitan/configs/` | `examples/maxtext/configs/` | `examples/megatron_bridge/configs/` |
| ------- | ---------------------------- | ------------------------------ | ---------------------------- | ----------------------------------- |
| **MI300X** | Yes | Yes | Yes | Yes |
| **MI355X** | Yes | Yes | Yes | Yes |
| **MI325X** | **No** `MI325X/` directory in this repository | **No** | **No** | **No** |

Megatron and TorchTitan ship parallel MI300X and MI355X experiment sets for many of the same model names. MaxText includes additional MI355X-only examples (for example `llama3.1_405B-pretrain.yaml`). Megatron Bridge MI300X examples target **Qwen3 32B**; **Qwen3 8B** and **LLaMA 3.1 70B** examples appear under MI355X.

This does **not** imply MI325X is unsupported by Primus in general—only that this tree does not currently provide `examples/*/configs/MI325X/` paths to copy from.

---

## Model architecture reference (Megatron presets)

Values below come from `primus/configs/models/megatron/` presets (merged through `extends`). **Vocabulary size** is usually defined by the tokenizer / Hugging Face config, not duplicated in every YAML; **context** is `max_position_embeddings` where set in the chain. Use this table as a quick reference for common sizes—not an exhaustive spec of every parameter.

| Model family | Example preset | Hidden size | Layers | Attention heads | KV heads (GQA) | Max position (context) |
| ------------ | -------------- | ----------- | ------ | ----------------- | ---------------- | ------------------------ |
| LLaMA 2 7B | `llama2_7B.yaml` | 4096 | 32 | 32 | 32 (no GQA) | From `llama2_base` / tokenizer |
| LLaMA 3 8B | `llama3_8B.yaml` | 4096 | 32 | 32 | 8 | 8192 (`llama3_base`) |
| LLaMA 3 70B | `llama3_70B.yaml` | 8192 | 80 | 64 | 8 | 8192 |
| LLaMA 3.1 405B | `llama3.1_405B.yaml` | 16384 | 126 | 128 | 8 | 8192 |
| Qwen3 8B | `qwen3_8B.yaml` | 4096 | 36 | 32 | 8 | 131072 (`qwen2.5_base` chain) |
| Mixtral 8x7B | `mixtral_8x7B_v0.1.yaml` | 4096 | 32 | 32 | — | 4096 |
| DeepSeek-V3 (MoE) | `deepseek_v3.yaml` | 7168 | 61 | 128 (MLA) | — | See preset / HF |
| Mamba 370M | `mamba_370M.yaml` | (Mamba stack) | — | — | — | — |

For MoE and hybrid architectures (LLaMA 4, Qwen3-MoE, large `moe_*.yaml` templates), refer to the full YAML and upstream model cards; headline dimensions alone do not capture expert layout or MLA.
