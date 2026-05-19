# Zebra-Llama: Hybrid Recurrent-Attention Models on AMD GPUs

Zebra-Llama is a family of hybrid models that combine **recurrent layers** (Mamba SSM, KDA, or GDN) with **Multi-Latent Attention (MLA)** and **SwiGLU MLP** layers. These models are designed to achieve competitive quality with sub-quadratic inference cost.

This guide covers the complete workflow: environment setup, data preparation, pretraining, checkpoint conversion, and evaluation.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Available Configurations](#available-configurations)
- [Prerequisites](#prerequisites)
- [Step 1: Environment Setup](#step-1-environment-setup)
- [Step 2: Dataset Preparation](#step-2-dataset-preparation)
- [Step 3: Pretraining](#step-3-pretraining)
  - [Single-Node (Local / Docker)](#single-node-local--docker)
  - [Multi-Node (Slurm)](#multi-node-slurm)
  - [Mock Data (Smoke Test)](#mock-data-smoke-test)
- [Step 4: Checkpoint Conversion to HuggingFace](#step-4-checkpoint-conversion-to-huggingface)
- [Step 5: Evaluation with lm-eval-harness](#step-5-evaluation-with-lm-eval-harness)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)

---

## Architecture Overview

Zebra-Llama interleaves three types of layers in a repeating pattern:

```
[Attention] [MLP] [Recurrent] [MLP] [Recurrent] [MLP] ... [Attention] [MLP] ...
```

- **Recurrent layers** — one of Mamba SSM, Kimi Delta Attention (KDA), or Gated Delta Net (GDN)
- **Attention layers** — Multi-Latent Attention with YaRN rotary embeddings and LoRA-compressed KV
- **MLP layers** — SwiGLU feed-forward with Transformer Engine fused norms

The `hybrid_attention_ratio` parameter controls what fraction of recurrent+attention layer pairs use attention (default 0.25 = 1 attention layer per 3 recurrent layers). Setting it to `0.0` yields a pure recurrent model (all KDA or GDN), while `1.0` yields a pure MLA attention model.

---

## Available Configurations

### Pretrain Configs (`examples/megatron/configs/MI300X/`)

| Config | Model | Recurrent Type | Seq Length | Params | Tokenizer |
|--------|-------|---------------|------------|--------|-----------|
| `zebra_llama_1B-pretrain.yaml` | 1B (Mamba+MLA) | Mamba SSM | 2048 | ~1B | `meta-llama/Llama-3.2-1B` |
| `zebra_llama_1B_kda-pretrain.yaml` | 1B (KDA+MLA) | Kimi Delta Attention | 8192 | ~1B | `meta-llama/Llama-3.2-1B` |
| `zebra_llama_1B_kda_pure-pretrain.yaml` | 1B (pure KDA) | Kimi Delta Attention | 2048 | ~1.2B | `meta-llama/Llama-3.2-1B` |
| `zebra_llama_1B_gdn-pretrain.yaml` | 1B (GDN only) | Gated Delta Net | 8192 | ~1B | `fla-hub/gla-1.3B-100B` |
| `zebra_llama_1B_gdn_pure-pretrain.yaml` | 1B (pure GDN) | Gated Delta Net | 2048 | ~1.2B | `meta-llama/Llama-3.2-1B` |
| `zebra_llama_3B-pretrain.yaml` | 3B (Mamba+MLA) | Mamba SSM | 8192 | ~3B | `meta-llama/Llama-3.2-3B` |
| `zebra_llama_8B-pretrain.yaml` | 8B (Mamba+MLA) | Mamba SSM | 8192 | ~8B | `meta-llama/Llama-3.1-8B` |

### Model Configs (`primus/configs/models/megatron/`)

| Config | Layers | Hidden | FFN | Attention Ratio | Attention Type |
|--------|--------|--------|-----|----------------|----------------|
| `zebra_llama_1B.yaml` | 32 | 2048 | 8192 | 0.25 | MLA |
| `zebra_llama_1B_kda_pure.yaml` | 32 | 2048 | 8192 | 0.0 (pure KDA) | None |
| `zebra_llama_1B_gdn.yaml` | 32 | 2048 | 8192 | 0.0 (pure GDN) | None |
| `zebra_llama_1B_gdn_pure.yaml` | 32 (16 GDN+16 MLP) | 2048 | 8192 | 0.0 (pure GDN) | None |
| `zebra_llama_3B.yaml` | 56 | 3072 | 8192 | 0.25 | MLA |
| `zebra_llama_8B.yaml` | 64 | 4096 | 14436 | 0.25 | MLA |

> **Note on pure KDA**: The `zebra_llama_1B_kda_pure` config matches FLA's `kda_1B_pure.json`
> architecture (16 KDA layers, `head_dim=32` for keys, `head_dim=64` for values, tied
> embeddings, `norm_eps=1e-6`). It uses the FLA Triton kernel (`use_fla_triton_kda: true`)
> for fused forward+backward during training.

> **Note on pure GDN**: The `zebra_llama_1B_gdn_pure` config matches FLA's
> `gated_deltanet_1B_pure.json` architecture (16 GDN + 16 MLP layers, `num_heads=8`,
> `num_v_heads=16`, short convolution with kernel size 4, tied embeddings). This config
> has been validated end-to-end against FLA on MI300X — the training loss curves match
> within ~1% across 76K steps on FineWeb-Edu 10BT. See
> [Step 4](#step-4-checkpoint-conversion-to-huggingface) for conversion to FLA's HuggingFace
> format.

---

## Prerequisites

- **Hardware**: AMD Instinct MI300X (or compatible ROCm GPUs)
- **Software**: ROCm drivers >= 7.0, Docker >= 24.0
- **HuggingFace Token**: Required for gated tokenizers (`HF_TOKEN`)
- **Disk Space**: ~50 GB for FineWeb-Edu 10BT tokenized data

---

## Step 1: Environment Setup

### 1.1 Pull the Docker Image

```bash
docker pull docker.io/rocm/primus:v25.10
```

### 1.2 Clone the Repository

```bash
git clone --recurse-submodules https://github.com/AMD-AIG-AIMA/Primus.git
cd Primus
```

### 1.3 Start a Development Container

```bash
# Quick start (mounts Primus into /workspace/Primus)
bash tools/docker/start_container.sh
```

This creates a persistent container named `dev_primus_<user>`. You can customize it with environment variables:

```bash
DOCKER_IMAGE=docker.io/rocm/primus:v25.10 \
DATA_PATH=/path/to/data \
bash tools/docker/start_container.sh
```

Then exec into the container:

```bash
docker exec -it dev_primus_$(whoami) bash
cd /workspace/Primus
```

### 1.4 Install Python Dependencies (inside container)

```bash
pip install -r requirements.txt
```

For GDN models (required for the Triton kernel and FLA model classes):

```bash
pip install flash-linear-attention
```

For evaluation, also install:

```bash
pip install lm-eval
```

---

## Step 2: Dataset Preparation

Zebra-Llama uses the [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset, preprocessed into Megatron binary format.

### 2.1 Set Up Environment

```bash
export HF_TOKEN="hf_your_token_here"
export PYTHONPATH="$(pwd)/third_party/Megatron-LM:${PYTHONPATH}"
```

### 2.2 Run Data Preparation

```bash
python examples/megatron/prepare_fineweb_edu.py \
    --primus-path . \
    --data-path ./data \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model meta-llama/Llama-3.2-1B \
    --sample-size 10BT
```

This will:
1. Download the FineWeb-Edu 10BT dataset from HuggingFace
2. Tokenize it into Megatron binary format (`.bin` + `.idx` files)
3. Output files to `./data/fineweb-edu-10BT/HuggingFaceTokenizer/`

Available sample sizes: `10BT`, `100BT`, `350BT`

The script uses all available CPU cores by default. To limit parallelism, add `--workers N`.

> **Note**: The context length (sequence length) is not set during data prep. It is configured at training time via `seq_length` in your pretrain YAML.

### 2.3 Using a Different Tokenizer

For the GDN config which uses `fla-hub/gla-1.3B-100B`:

```bash
python examples/megatron/prepare_fineweb_edu.py \
    --primus-path . \
    --data-path ./data \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model fla-hub/gla-1.3B-100B \
    --sample-size 10BT
```

### 2.4 FLA-Aligned Data for Pure GDN (Recommended)

When training a pure GDN model for comparison against FLA, it is critical that both frameworks see **identical tokens in the same order**. The standard Megatron data pipeline produces different token ordering than FLA's `preprocess.py` (which shuffles with `seed=42` and concatenates into fixed-length chunks without EOD tokens). This difference alone can cause persistent training loss divergence.

To ensure exact alignment:

**Step 1: Preprocess with FLA** (if not already done):

```bash
cd /path/to/flash-linear-attention/legacy/training
python preprocess.py \
    --dataset HuggingFaceFW/fineweb-edu \
    --name sample-10BT \
    --tokenizer meta-llama/Llama-3.2-1B \
    --seq_len 2048 --num_proc 64
```

This produces an Arrow dataset under `data/HuggingFaceFW/fineweb-edu/sample-10BT/train/`.

**Step 2: Convert FLA's Arrow data to Megatron binary format**:

```bash
python convert_fla_to_megatron.py
```

> **Note**: Edit the `FLA_DATA` and `OUT_PREFIX` paths at the top of `convert_fla_to_megatron.py` to match your environment before running.

This reads the Arrow shard files directly with PyArrow and produces Megatron-compatible `.bin` + `.idx` files. Each FLA 2048-token sequence becomes one Megatron "document". The script verifies token-level consistency after writing.

**Step 3: Point the pretrain config at the converted data**:

```yaml
train_data_path: >
  /path/to/data/fla_aligned/fla_fineweb_edu_10BT_text_sentence
mock_data: false
```

### 2.5 Update Data Paths in Config

After preparation (standard or FLA-aligned), update the `train_data_path` in your pretrain config YAML to point to the generated files:

```yaml
# Standard Megatron data prep (multiple shards)
train_data_path: >
  /path/to/data/fineweb-edu-10BT/HuggingFaceTokenizer/fineweb_edu_10BT_0_text_sentence
  /path/to/data/fineweb-edu-10BT/HuggingFaceTokenizer/fineweb_edu_10BT_1_text_sentence
  /path/to/data/fineweb-edu-10BT/HuggingFaceTokenizer/fineweb_edu_10BT_2_text_sentence
  /path/to/data/fineweb-edu-10BT/HuggingFaceTokenizer/fineweb_edu_10BT_3_text_sentence
mock_data: false
```

---

## Step 3: Pretraining

### Single-Node (Local / Docker)

Launch training inside a Docker container on a single node:

```bash
# Zebra-Llama 1B with KDA (Kimi Delta Attention)
EXP=examples/megatron/configs/MI300X/zebra_llama_1B_kda-pretrain.yaml \
DATA_PATH=./data \
GPUS_PER_NODE=8 \
HF_TOKEN=$HF_TOKEN \
bash examples/run_local_pretrain.sh
```

Other model variants:

```bash
# Zebra-Llama 1B with Mamba SSM
EXP=examples/megatron/configs/MI300X/zebra_llama_1B-pretrain.yaml \
bash examples/run_local_pretrain.sh

# Zebra-Llama 1B with pure KDA (no attention layers)
EXP=examples/megatron/configs/MI300X/zebra_llama_1B_kda_pure-pretrain.yaml \
bash examples/run_local_pretrain.sh

# Zebra-Llama 1B with GDN (pure recurrent, no attention)
EXP=examples/megatron/configs/MI300X/zebra_llama_1B_gdn-pretrain.yaml \
bash examples/run_local_pretrain.sh

# Zebra-Llama 1B pure GDN (FLA-validated, 4-GPU)
EXP=examples/megatron/configs/MI300X/zebra_llama_1B_gdn_pure-pretrain.yaml \
GPUS_PER_NODE=4 \
bash examples/run_local_pretrain.sh

# Zebra-Llama 3B
EXP=examples/megatron/configs/MI300X/zebra_llama_3B-pretrain.yaml \
bash examples/run_local_pretrain.sh

# Zebra-Llama 8B
EXP=examples/megatron/configs/MI300X/zebra_llama_8B-pretrain.yaml \
bash examples/run_local_pretrain.sh
```

### Multi-Node (Slurm)

For multi-node training on a Slurm cluster:

```bash
EXP=examples/megatron/configs/MI300X/zebra_llama_1B_kda-pretrain.yaml \
DATA_PATH=/shared/data \
NNODES=2 \
bash examples/run_slurm_pretrain.sh
```

Ensure the `global_batch_size` in your config is divisible by `micro_batch_size * GPUS_PER_NODE * NNODES`.

### If Already Inside a Container

If you are already inside a Docker container or on a bare-metal node with the environment set up:

```bash
EXP=examples/megatron/configs/MI300X/zebra_llama_1B_kda-pretrain.yaml \
bash examples/run_pretrain.sh
```

### Mock Data (Smoke Test)

To quickly verify the model runs without real data, the 3B and 8B configs come with `mock_data: true` by default. For the 1B configs, you can override:

```bash
EXP=examples/megatron/configs/MI300X/zebra_llama_1B_kda-pretrain.yaml \
bash examples/run_local_pretrain.sh \
  --mock_data true --train_iters 10
```

### Key Training Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|---------------|
| `train_iters` | Total training iterations | 38147 (1B KDA pure), 400000 (1B Mamba) |
| `micro_batch_size` | Per-GPU batch size | 4 (KDA/GDN), 16 (Mamba 1B) |
| `global_batch_size` | Total batch size across all GPUs | micro_batch_size * num_gpus |
| `seq_length` | Sequence length | 2048, 4096, 8192 |
| `lr` | Peak learning rate | 2.0e-4 |
| `save_interval` | Checkpoint save frequency | 1000 |
| `auto_continue_train` | Auto-resume from last checkpoint on crash | `true` / `false` |
| `hybrid_attention_ratio` | Fraction of attention layers (0.0 = pure recurrent) | 0.0, 0.25 |

---

## Step 4: Checkpoint Conversion to HuggingFace

Convert a Megatron checkpoint to HuggingFace format for inference and evaluation.

### 4.1 Convert Checkpoint

#### Pure GDN Models

Pure GDN models use a dedicated converter that maps Primus's fused projections to FLA's native `GatedDeltaNetForCausalLM` format:

```bash
python tools/convert_gdn_to_fla_hf.py \
    --checkpoint-path output/amd/root/zebra_llama_1B_gdn_pure-pretrain/checkpoints/iter_0076294 \
    --output-dir output/gdn_pure_1B_fla_hf \
    --config /path/to/gated_deltanet_1B_pure.json
```

This handles:
- Splitting the fused `in_proj` (3104 → q/k/v/gate/beta/alpha projections)
- Splitting the fused `conv1d` (q/k/v convolutions)
- Splitting the fused SwiGLU `fc1` (gate_proj + up_proj)
- Mapping alternating GDN/MLP sublayers to combined FLA layers
- Handling tied embeddings

After conversion, verify with the sanity check:

```bash
python tools/verify_gdn_conversion.py --model-path output/gdn_pure_1B_fla_hf
```

Expected output: Loss ~2-4, top prediction for "The capital of France is" should be "Paris".

#### KDA / Hybrid Models

The general converter auto-detects architecture from the checkpoint's saved arguments:

```bash
# KDA+MLA hybrid model
python tools/convert_zebra_llama_to_hf.py \
    --checkpoint-path output/zebra_llama_1B_kda-pretrain/iter_0028000 \
    --output-dir output/zebra_llama_1B_kda_hf_iter_0028000

# Pure KDA model
python tools/convert_zebra_llama_to_hf.py \
    --checkpoint-path output/zebra_llama_1B_kda_pure-pretrain/iter_0038000 \
    --output-dir output/zebra_llama_1B_kda_pure_hf
```

The converter will:
- Read the Megatron checkpoint and training arguments
- Auto-detect architecture parameters (`hybrid_attention_ratio`, `kda_num_heads`, `q_lora_rank`, etc.)
- Remap parameter names from Megatron conventions to HuggingFace conventions
- Save `pytorch_model.bin`, `config.json`, and a model card `README.md` in the output directory
- Copy `modeling_zebra_llama.py` into the output directory for `trust_remote_code` loading

### 4.2 Verify Conversion

The script prints a summary of missing, extra, and shape-mismatched keys. A successful conversion shows:

```
0 missing, 0 extra, 0 shape mismatches
```

### 4.3 Supported Architectures

| Architecture | `hybrid_attention_ratio` | Layer pattern |
|---|---|---|
| Pure KDA | `0.0` | All KDA + MLP |
| KDA + MLA hybrid | `0.0 < r < 1.0` | Mix of KDA and MLA + MLP |
| Pure MLA | `1.0` | All MLA + MLP |
| Pure GDN | `0.0` (with GDN spec) | All GDN + MLP |
| Mamba + MLA hybrid | `0.0 < r < 1.0` (with Mamba spec) | Mix of Mamba and MLA + MLP |

---

## Step 5: Evaluation with lm-eval-harness

### 5.1 Pure GDN Models (FLA format)

Pure GDN models use a dedicated eval wrapper (`tools/eval_gdn_lm_eval.py`) that pre-registers FLA's `GatedDeltaNetForCausalLM` with transformers' `AutoModel` and patches compatibility issues with transformers >= 4.55:

```bash
python tools/eval_gdn_lm_eval.py \
    --model hf \
    --model_args pretrained=output/gdn_pure_1B_fla_hf,trust_remote_code=True,tokenizer=meta-llama/Llama-3.2-1B \
    --tasks arc_easy,arc_challenge,hellaswag,mmlu,openbookqa,piqa,race,winogrande \
    --batch_size auto \
    --output_path eval_results/gdn_pure_1B
```

> **Note**: Do not use `lm_eval --model hf` directly — it will fail because `AutoConfig` does not recognize `gated_deltanet` without FLA being imported first. The wrapper handles this. The `tokenizer=meta-llama/Llama-3.2-1B` argument is required since the converted model directory does not contain tokenizer files.

### 5.2 KDA / Hybrid Models (Zebra-Llama format)

KDA and hybrid models use the custom `ZebraLlamaForCausalLM` architecture, which requires a dedicated lm-eval wrapper:

```bash
python3 tools/lm_harness_eval.py --model zebra_llama \
    --model_args pretrained=output/zebra_llama_1B_kda_pure_hf,dtype=bfloat16 \
    --tasks arc_easy,arc_challenge,hellaswag,mmlu,openbookqa,piqa,race,winogrande \
    --batch_size auto
```

### 5.3 Using the Eval Shell Script (KDA/Hybrid)

```bash
bash tools/eval_zebra_llama_lm_eval.sh \
    --checkpoint output/zebra_llama_1B_kda_pure_hf \
    --tasks arc_easy,arc_challenge,hellaswag,mmlu,openbookqa,piqa,race,winogrande \
    --batch-size auto \
    --dtype bfloat16 \
    --output eval_results/zebra_llama_1B_kda_pure
```

> **Important**: The eval script internally invokes `python3 tools/lm_harness_eval.py --model zebra_llama` (not `lm_eval --model hf`). This ensures the custom model architecture is properly registered.

### 5.4 Available Benchmarks

| Task | Description | Metric |
|------|-------------|--------|
| `arc_easy` | ARC Easy (science QA) | acc, acc_norm |
| `arc_challenge` | ARC Challenge (harder science QA) | acc, acc_norm |
| `hellaswag` | HellaSwag (commonsense NLI) | acc, acc_norm |
| `mmlu` | MMLU (57 subject knowledge benchmark) | acc |
| `openbookqa` | OpenBookQA | acc, acc_norm |
| `piqa` | PIQA (physical intuition QA) | acc, acc_norm |
| `race` | RACE (reading comprehension) | acc |
| `winogrande` | Winogrande (coreference resolution) | acc |

### 5.5 Memory Considerations

The pure-PyTorch KDA chunked attention is memory-intensive. If you encounter OOM errors:

- Use `--batch_size auto` to let lm-eval find the largest fitting batch size
- Reduce `max_length` (e.g., `max_length=1024` in `--model_args`)
- Reduce `--batch_size` to 1

---

## Configuration Reference

### Hybrid Layer Specs

The `spec` field in the pretrain config selects the layer arrangement:

| Spec | Description |
|------|-------------|
| `hybrid_stack_spec` | Mamba SSM + MLA hybrid |
| `kda_hybrid_stack_spec` | KDA + MLA hybrid (or pure KDA with `hybrid_attention_ratio: 0.0`) |
| `gdn_hybrid_stack_spec` | GDN + MLA hybrid (or pure GDN with `hybrid_attention_ratio: 0.0`) |

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DOCKER_IMAGE` | Docker image for training | `docker.io/rocm/primus:v25.10` |
| `EXP` | Path to experiment config YAML | `examples/megatron/exp_pretrain.yaml` |
| `DATA_PATH` | Path to dataset directory | `./data` |
| `HF_TOKEN` | HuggingFace API token | (required for gated models) |
| `WANDB_API_KEY` | Weights & Biases API key | (optional) |
| `GPUS_PER_NODE` | Number of GPUs per node | 8 |
| `NNODES` | Number of nodes | 1 |
| `MASTER_ADDR` | Master node address | `localhost` |
| `MASTER_PORT` | Master node port | `1234` |

---

## Troubleshooting

### OOM During Training

- Reduce `micro_batch_size` or `seq_length`
- Enable activation checkpointing: add `recompute_granularity: selective` to the config

### OOM During Evaluation

- Use `--batch_size 1` or `--batch_size auto`
- Add `max_length=1024` to `--model_args`

### `ModuleNotFoundError: No module named 'megatron'`

Set the Python path before running data preparation:

```bash
export PYTHONPATH="$(pwd)/third_party/Megatron-LM:${PYTHONPATH}"
```

### Checkpoint Conversion Shape Mismatches

Ensure the `modeling_zebra_llama.py` model definition matches the architecture of your checkpoint (Mamba vs KDA vs GDN). The converter auto-detects architecture from checkpoint args, but the HF model code in `tools/modeling_zebra_llama.py` must support the target architecture. Common causes of shape mismatches:

- Mismatched `hybrid_attention_ratio` between config and checkpoint
- Incorrect `kda_num_heads` or head dimension settings
- Using a `modeling_zebra_llama.py` that doesn't support the checkpoint's attention type

### `ValueError: model type 'zebra_llama' not recognized`

This occurs when using `lm_eval --model hf` directly instead of the custom wrapper. Always use:

```bash
python3 tools/lm_harness_eval.py --model zebra_llama ...
```

Or the eval shell script, which handles this automatically.

### Truncation Warnings During Eval

Messages like `Combined length of context and continuation exceeds model's maximum length` mean some eval samples are being truncated. This has minimal impact on most benchmarks but can affect long-context tasks like RACE. To avoid truncation, increase `max_length` in `--model_args`.

### NCCL / RCCL Timeout During Training

On MI300X, intermittent RCCL hangs can occur (typically during checkpoint saves). Mitigations:

- Set `auto_continue_train: true` in the pretrain config to auto-resume from the last checkpoint
- Increase the heartbeat timeout: `export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200`

---

## File Reference

```
Primus/
├── examples/megatron/
│   ├── configs/MI300X/
│   │   ├── zebra_llama_1B-pretrain.yaml        # 1B Mamba+MLA
│   │   ├── zebra_llama_1B_kda-pretrain.yaml     # 1B KDA+MLA hybrid
│   │   ├── zebra_llama_1B_kda_pure-pretrain.yaml # 1B pure KDA
│   │   ├── zebra_llama_1B_gdn-pretrain.yaml     # 1B GDN
│   │   ├── zebra_llama_1B_gdn_pure-pretrain.yaml # 1B pure GDN (FLA-validated)
│   │   ├── zebra_llama_3B-pretrain.yaml         # 3B Mamba+MLA
│   │   └── zebra_llama_8B-pretrain.yaml         # 8B Mamba+MLA
│   ├── prepare_fineweb_edu.py                   # Data preparation script
│   ├── prepare_fineweb_edu.sh                   # Data prep shell wrapper
│   └── preprocess_data.py                       # Megatron tokenizer
├── primus/configs/models/megatron/
│   ├── zebra_llama_1B.yaml                      # 1B model architecture
│   ├── zebra_llama_1B_kda_pure.yaml             # 1B pure KDA architecture
│   ├── zebra_llama_1B_gdn.yaml                  # 1B GDN architecture
│   ├── zebra_llama_1B_gdn_pure.yaml             # 1B pure GDN (FLA-validated)
│   ├── zebra_llama_3B.yaml                      # 3B model architecture
│   └── zebra_llama_8B.yaml                      # 8B model architecture
├── tools/
│   ├── convert_zebra_llama_to_hf.py             # Megatron → HF converter (KDA/hybrid)
│   ├── convert_gdn_to_fla_hf.py                 # Megatron → FLA HF converter (pure GDN)
│   ├── verify_gdn_conversion.py                 # Post-conversion sanity check (pure GDN)
│   ├── eval_gdn_lm_eval.py                      # lm-eval wrapper for GDN (registers FLA)
│   ├── convert_zebra_llama_to_hf.sh             # Converter shell wrapper
│   ├── modeling_zebra_llama.py                  # HF model definition (KDA/hybrid)
│   ├── lm_harness_eval.py                       # lm-eval wrapper
│   ├── eval_zebra_llama_lm_eval.sh              # Eval shell wrapper
│   ├── run_zebra_eval.sh                        # Quick eval script
│   ├── chat_zebra_llama.py                      # Interactive chat
│   └── docker/start_container.sh                # Dev container launcher
├── convert_fla_to_megatron.py                    # FLA Arrow → Megatron binary converter
├── examples/
│   ├── run_local_pretrain.sh                    # Single-node Docker launcher
│   ├── run_slurm_pretrain.sh                    # Slurm launcher
│   └── run_pretrain.sh                          # Core training entrypoint
└── requirements.txt                             # Python dependencies
```
