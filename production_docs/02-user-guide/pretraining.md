# Pretraining Workflows

Primus is a YAML-driven training stack for AMD GPUs. You select a **backend** (Megatron-LM, TorchTitan, JAX MaxText, Megatron Bridge), point `train pretrain` at an **experiment YAML**, and launch with the unified CLI (`runner/primus-cli`) in **direct**, **container**, or **Slurm** mode. See [CLI Reference](cli-reference.md) and [Configuration System](configuration-system.md).

**Related documentation**

| Topic | Location |
| --- | --- |
| Launcher usage | [CLI Reference](cli-reference.md) |
| YAML merge rules | [Configuration System](configuration-system.md) |
| Backend knobs | [Megatron parameters](../03-configuration-reference/megatron-parameters.md), [TorchTitan parameters](../03-configuration-reference/torchtitan-parameters.md), [MaxText parameters](../03-configuration-reference/maxtext-parameters.md) |

---

## Overview

| Backend | Framework | Typical use |
| --- | --- | --- |
| Megatron-LM | `framework: megatron` | Large-scale transformer pretraining with Megatron-style parallelism (TP/PP/EP). |
| TorchTitan | `framework: torchtitan` | PyTorch-native scaled training (FSDP / tensor / pipeline / expert parallelism per config). |
| MaxText (JAX) | `framework: maxtext` | JAX/MaxText single- and multi-node runs; parallelism via MaxText `ici_*` / `dcn_*` settings. |
| Megatron Bridge | `framework: megatron_bridge` | Bridge-oriented workflows (configure like other backends; see parameter reference). |

---

## Megatron-LM pretraining

### Quick start (container mode)

From the repository root, with Docker or Podman available:

```bash
./runner/primus-cli container -- train pretrain \
  --config examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml
```

This uses the default image from `runner/.primus.yaml` (`rocm/primus:v26.1` unless overridden). The project tree is mounted into the container automatically by `runner/primus-cli-container.sh`.

### Example walkthrough: `llama2_7B-BF16-pretrain.yaml`

Path: `examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml`

| Section | Role |
| --- | --- |
| `work_group`, `user_name`, `exp_name`, `workspace` | Run identity and output root (supports `${VAR:default}` substitution). |
| `modules.pre_trainer.framework` | `megatron` selects Megatron-LM integration. |
| `config: pre_trainer.yaml` | Module preset under `primus/configs/modules/megatron/`. |
| `model: llama2_7B.yaml` | Model preset under `primus/configs/models/megatron/` (extends `llama2_base.yaml` → …). |
| `overrides` | Experiment-specific training knobs: iterations, batching, LR, **parallelism** (`tensor_model_parallel_size`, `pipeline_model_parallel_size`, `expert_model_parallel_size`), data paths, checkpoints, Primus Turbo flags, etc. |

The sample sets `mock_data: true` and `train_data_path: null` so you can validate the stack without real corpora.

### Example configs under `examples/megatron/configs/MI300X/`

The following files ship in the repository (sorted by name). Parallelism columns are taken from `tensor_model_parallel_size` / `pipeline_model_parallel_size` / `expert_model_parallel_size` in each file (literals or `${PRIMUS_TP:…}` defaults).

| Config | TP | PP | EP |
| --- | --- | --- | --- |
| `deepseek_v2-BF16-pretrain.yaml` | `${PRIMUS_TP:1}` | `${PRIMUS_PP:4}` | `${PRIMUS_EP:8}` |
| `deepseek_v2-FP8-pretrain.yaml` | `${PRIMUS_TP:1}` | `${PRIMUS_PP:4}` | `${PRIMUS_EP:8}` |
| `deepseek_v2_lite-BF16-pretrain.yaml` | `${PRIMUS_TP:1}` | `${PRIMUS_PP:1}` | `${PRIMUS_EP:8}` |
| `deepseek_v2_lite-FP8-pretrain.yaml` | `${PRIMUS_TP:1}` | `${PRIMUS_PP:1}` | `${PRIMUS_EP:8}` |
| `deepseek_v3-BF16-pretrain.yaml` | `${PRIMUS_TP:1}` | `${PRIMUS_PP:1}` | `${PRIMUS_EP:8}` |
| `deepseek_v3-FP8-pretrain.yaml` | `${PRIMUS_TP:1}` | `${PRIMUS_PP:1}` | `${PRIMUS_EP:8}` |
| `gpt_oss_20B-BF16-pretrain.yaml` | `${PRIMUS_TP:1}` | `${PRIMUS_PP:1}` | `${PRIMUS_EP:8}` |
| `gpt_oss_20B-FP8-pretrain.yaml` | `${PRIMUS_TP:1}` | `${PRIMUS_PP:1}` | `${PRIMUS_EP:8}` |
| `grok1-BF16-pretrain.yaml` | `1` | `4` | `8` |
| `grok1-FP8-pretrain.yaml` | `1` | `4` | `8` |
| `grok2-BF16-pretrain.yaml` | `1` | `4` | `8` |
| `grok2-FP8-pretrain.yaml` | `1` | `4` | `8` |
| `llama2_70B-BF16-pretrain.yaml` | `1` | `1` | `1` |
| `llama2_70B-FP8-pretrain.yaml` | `1` | `1` | `1` |
| `llama2_7B-BF16-pretrain.yaml` | `1` | `1` | `1` |
| `llama2_7B-FP8-pretrain.yaml` | `1` | `1` | `1` |
| `llama3.1_405B-BF16-pretrain.yaml` | `8` | `8` | `1` |
| `llama3.1_405B-FP8-pretrain.yaml` | `8` | `8` | `1` |
| `llama3.1_70B-BF16-pretrain.yaml` | `1` | `1` | `1` |
| `llama3.1_70B-FP8-pretrain.yaml` | `1` | `1` | `1` |
| `llama3.1_8B-BF16-pretrain.yaml` | `1` | `1` | `1` |
| `llama3.1_8B-FP8-pretrain.yaml` | `1` | `1` | `1` |
| `llama3.2_1B-BF16-pretrain.yaml` | `1` | `1` | `1` |
| `llama3.2_1B-FP8-pretrain.yaml` | `1` | `1` | `1` |
| `llama3.2_3B-BF16-pretrain.yaml` | `1` | `1` | `1` |
| `llama3.2_3B-FP8-pretrain.yaml` | `1` | `1` | `1` |
| `llama3.3_70B-BF16-pretrain.yaml` | `1` | `1` | `1` |
| `llama3.3_70B-FP8-pretrain.yaml` | `1` | `1` | `1` |
| `llama3_70B-BF16-pretrain.yaml` | `1` | `1` | `1` |
| `llama3_70B-FP8-pretrain.yaml` | `1` | `1` | `1` |
| `llama3_8B-BF16-pretrain.yaml` | `1` | `1` | `1` |
| `llama3_8B-FP8-pretrain.yaml` | `1` | `1` | `1` |
| `llama4_17B128E-BF16-pretrain.yaml` | `${PRIMUS_TP:1}` | `${PRIMUS_PP:1}` | `${PRIMUS_EP:8}` |
| `llama4_17B128E-FP8-pretrain.yaml` | `${PRIMUS_TP:1}` | `${PRIMUS_PP:1}` | `${PRIMUS_EP:8}` |
| `llama4_17B16E-BF16-pretrain.yaml` | `${PRIMUS_TP:1}` | `${PRIMUS_PP:1}` | `${PRIMUS_EP:8}` |
| `llama4_17B16E-FP8-pretrain.yaml` | `${PRIMUS_TP:1}` | `${PRIMUS_PP:1}` | `${PRIMUS_EP:8}` |
| `mamba_370M-pretrain.yaml` | `1` | `1` | `1` |
| `mixtral_8x22B_v0.1-BF16-pretrain.yaml` | `1` | `4` | `8` |
| `mixtral_8x22B_v0.1-FP8-pretrain.yaml` | `1` | `4` | `8` |
| `mixtral_8x7B_v0.1-BF16-pretrain.yaml` | `1` | `1` | `8` |
| `mixtral_8x7B_v0.1-FP8-pretrain.yaml` | `1` | `1` | `8` |
| `qwen2.5_72B-BF16-pretrain.yaml` | `1` | `1` | `1` |
| `qwen2.5_72B-FP8-pretrain.yaml` | `1` | `1` | `1` |
| `qwen2.5_7B-BF16-pretrain.yaml` | `1` | `1` | `1` |
| `qwen2.5_7B-FP8-pretrain.yaml` | `1` | `1` | `1` |
| `qwen3_235B_A22B-BF16-pretrain.yaml` | `1` | `1` | `8` |
| `qwen3_235B_A22B-FP8-pretrain.yaml` | `1` | `1` | `8` |
| `qwen3_30B_A3B-BF16-pretrain.yaml` | `${PRIMUS_TP:1}` | `${PRIMUS_PP:1}` | `${PRIMUS_EP:8}` |
| `qwen3_30B_A3B-FP8-pretrain.yaml` | `${PRIMUS_TP:1}` | `${PRIMUS_PP:1}` | `${PRIMUS_EP:8}` |
| `zebra_llama_1B-pretrain.yaml` | `1` | `1` | `1` |
| `zebra_llama_3B-pretrain.yaml` | `1` | `1` | `1` |
| `zebra_llama_8B-pretrain.yaml` | `1` | `1` | `1` |

### Mock data versus real data

- **Mock data:** set `mock_data: true` and leave `train_data_path` / `valid_data_path` empty (as in `llama2_7B-BF16-pretrain.yaml`).  
- **Real data:** set `mock_data: false` and populate Megatron-compatible data paths (and tokenizer assets) in `overrides`. Use paths visible inside your container mounts.

### Multi-node training with Slurm

```bash
./runner/primus-cli slurm srun -N 4 -p <partition> -- container -- train pretrain \
  --config examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml
```

`runner/primus-cli-slurm-entry.sh` derives `MASTER_ADDR`, `NNODES`, and `NODE_RANK` from Slurm and forwards them into the container. Align `tensor_model_parallel_size`, `pipeline_model_parallel_size`, and `expert_model_parallel_size` with your cluster width and job size.

---

## TorchTitan pretraining

### Quick start

```bash
./runner/primus-cli container -- train pretrain \
  --config examples/torchtitan/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
```

### Example walkthrough

Path: `examples/torchtitan/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml`

| Section | Role |
| --- | --- |
| `framework: torchtitan` | Selects the TorchTitan integration. |
| `config: pre_trainer.yaml` | Module preset under `primus/configs/modules/torchtitan/`. |
| `model: llama3.1_8B.yaml` | Model preset under `primus/configs/models/torchtitan/`. |
| `overrides.training`, `lr_scheduler`, `activation_checkpoint`, `primus_turbo` | Run-specific batching, steps, checkpointing, and Turbo options. |

Some experiments omit an explicit `parallelism:` block; in that case defaults come from the **module and model presets** (`primus/configs/modules/torchtitan/pre_trainer.yaml` and the chosen model YAML). Other examples (for example DeepSeek and Qwen) set `parallelism:` inline with `tensor_parallel_degree`, `pipeline_parallel_degree`, `expert_parallel_degree`, etc.

### Example configs under `examples/torchtitan/configs/MI300X/`

| File |
| --- |
| `deepseek_v3_16b-BF16-pretrain.yaml` |
| `deepseek_v3_16b-FP8-pretrain.yaml` |
| `deepseek_v3_671b-pretrain.yaml` |
| `llama3.1_405B-BF16-pretrain.yaml` |
| `llama3.1_405B-FP8-pretrain.yaml` |
| `llama3.1_70B-BF16-pretrain.yaml` |
| `llama3.1_70B-FP8-pretrain.yaml` |
| `llama3.1_8B-BF16-pretrain.yaml` |
| `llama3.1_8B-FP8-pretrain.yaml` |
| `qwen3_0.6B-pretrain.yaml` |
| `qwen3_1.7B-pretrain.yaml` |
| `qwen3_32B-pretrain.yaml` |

---

## MaxText (JAX) pretraining

### Quick start

```bash
./runner/primus-cli container -- train pretrain \
  --config examples/maxtext/configs/MI300X/llama2_7B-pretrain.yaml
```

### JAX-specific requirements

Install JAX/MaxText dependencies from the repository root:

```bash
pip install -r requirements-jax.txt
```

### Example configs under `examples/maxtext/configs/MI300X/`

| File | Key parallelism (`ici_*` intra-node, `dcn_*` inter-node) |
| --- | --- |
| `deepseek_v2_16B-pretrain.yaml` | `ici_fsdp_parallelism: 1`, `ici_data_parallelism: 1`, `dcn_fsdp_parallelism: 1`, `dcn_data_parallelism: -1` |
| `grok1-pretrain.yaml` | `ici_fsdp_parallelism: 1`, `ici_data_parallelism: 1`, `dcn_fsdp_parallelism: 1`, `dcn_data_parallelism: -1` |
| `llama2_70B-pretrain.yaml` | `ici_fsdp_parallelism: 8`, `ici_data_parallelism: 1`, `dcn_fsdp_parallelism: 1`, `dcn_data_parallelism: -1` |
| `llama2_7B-pretrain.yaml` | `ici_fsdp_parallelism: 8`, `ici_data_parallelism: 1`, `dcn_fsdp_parallelism: 1`, `dcn_data_parallelism: -1` |
| `llama3.3_70B-pretrain.yaml` | `ici_fsdp_parallelism: 8`, `ici_data_parallelism: 1`, `dcn_fsdp_parallelism: 1`, `dcn_data_parallelism: -1` |
| `llama3_70B-pretrain.yaml` | `ici_fsdp_parallelism: 8`, `ici_data_parallelism: 1`, `dcn_fsdp_parallelism: 1`, `dcn_data_parallelism: -1` |
| `llama3_8B-pretrain.yaml` | `ici_fsdp_parallelism: 8`, `ici_data_parallelism: 1`, `dcn_fsdp_parallelism: 1`, `dcn_data_parallelism: -1` |
| `mixtral_8x7B-pretrain.yaml` | `ici_fsdp_parallelism: 1`, `ici_data_parallelism: 1`, `dcn_fsdp_parallelism: 1`, `dcn_data_parallelism: -1` |
| `qwen3_14B-pretrain.yaml` | `ici_fsdp_parallelism: 8`, `ici_data_parallelism: 1`, `dcn_fsdp_parallelism: 1`, `dcn_data_parallelism: -1` |
| `qwen3_30B_A3B-pretrain.yaml` | `ici_fsdp_parallelism: 1`, `ici_data_parallelism: 1`, `dcn_fsdp_parallelism: 1`, `dcn_data_parallelism: -1` |

The `llama2_7B-pretrain.yaml` example also sets `dataset_type: "synthetic"` and `hf_access_token: ${HF_TOKEN:""}` for gated Hugging Face assets when you switch to real data.

---

## Common patterns

### Testing with mock data

Set `mock_data: true` (Megatron/TorchTitan) or synthetic dataset settings (MaxText) to validate configs and infrastructure without I/O-heavy datasets.

### Real training data

- Megatron: configure `train_data_path` / `valid_data_path` and tokenizer assets in `overrides` once `mock_data` is false.  
- Ensure host paths are mounted in **container** mode (`--volume` or `container.options.volume` in YAML).  
- TorchTitan / MaxText: follow backend-specific dataset fields in the experiment `overrides` and presets.

### Scaling from single-node to multi-node

- Use **Slurm** mode for allocation; keep **container** entry if you want the same image on every node.  
- Set environment variables consistently (`NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `GPUS_PER_NODE`); the Slurm entry script injects them when using `primus-cli slurm`.  
- Increase parallelism fields (Megatron TP/PP/EP; TorchTitan `parallelism`; MaxText `ici_*` / `dcn_*`) to match topology.

### Hugging Face token for gated models

Export `HF_TOKEN` on the host before launching **container** mode; `runner/.primus.yaml` lists `HF_TOKEN` under `container.options.env` so it can be forwarded into the container. MaxText configs may reference `${HF_TOKEN:""}` directly.

### HipBLASLt autotuning (three stages)

Controlled with `PRIMUS_HIPBLASLT_TUNING_STAGE` (see `examples/README.md`):

| Stage | Purpose |
| --- | --- |
| 1 | Dump GEMM shapes seen during training (reduce `train_iters` for faster collection). |
| 2 | Tune kernels from dumped shapes (offline tooling under `examples/offline_tune`). |
| 3 | Train using tuned kernel artifacts from `./output/tune_hipblaslt/...`. |

Example (from in-repo docs):

```bash
export PRIMUS_HIPBLASLT_TUNING_STAGE=1
export EXP=examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml
NNODES=1 bash ./examples/run_slurm_pretrain.sh
```

---

## Supported models (summary)

The tables in the Megatron, TorchTitan, and MaxText sections list **every** `examples/**/configs/MI300X/*.yaml` file in this repository for those backends. Use them as the authoritative list of shipped example configs.

| Backend | Example region | Parallelism vocabulary |
| --- | --- | --- |
| Megatron-LM | `examples/megatron/configs/MI300X/` | `tensor_model_parallel_size`, `pipeline_model_parallel_size`, `expert_model_parallel_size` (and env-driven `${PRIMUS_TP:…}` variants). |
| TorchTitan | `examples/torchtitan/configs/MI300X/` | `parallelism.*` (e.g. `tensor_parallel_degree`, `pipeline_parallel_degree`, `expert_parallel_degree`, FSDP shard settings). |
| MaxText | `examples/maxtext/configs/MI300X/` | `ici_fsdp_parallelism`, `ici_data_parallelism`, `dcn_fsdp_parallelism`, `dcn_data_parallelism`. |

For scripting patterns that predate `primus-cli`, the repository still documents `examples/run_local_pretrain.sh` and `examples/run_slurm_pretrain.sh` in `examples/README.md`; equivalent launches are shown above using `./runner/primus-cli`.
