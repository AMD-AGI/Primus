# 🧠 Pretraining with Primus

This guide demonstrates how to perform pretraining using **Megatron**/**torchtitan** within the **Primus** framework.
It supports both **single-node** and **multi-node** training, and includes optional **HipBLASLt auto-tuning** for optimal AMD GPU performance.

---

## 📚 Table of Contents

- [🧠 Pretraining with Primus](#-pretraining-with-primus)
  - [📚 Table of Contents](#-table-of-contents)
  - [⚙️ Supported Backends](#️-supported-backends)
  - [🖥️ Single Node Training](#️-single-node-training)
    - [Setup Docker](#setup-docker)
    - [Setup Primus](#setup-primus)
    - [Run Pretraining](#run-pretraining)
      - [🚀 Quick Start Mode](#-quick-start-mode)
      - [🧑‍🔧 Interactive Mode](#-interactive-mode)
  - [🌐 Multi-node Training](#-multi-node-training)
  - [🔧 HipblasLT Auto Tuning](#-hipblaslt-auto-tuning)
    - [Stage 1: Dump GEMM Shape](#stage-1-dump-gemm-shape)
    - [Stage 2: Tune GEMM Kernel](#stage-2-tune-gemm-kernel)
    - [Stage 3: Train with Tuned Kernel](#stage-3-train-with-tuned-kernel)
  - [✅ Supported Models](#-supported-models)
    - [🏃‍♂️ How to Run a Supported Model](#️-how-to-run-a-supported-model)

---

## ⚙️ Supported Backends

Primus supports multiple backends.

| Backend        | Description                                                  |
| -------------- | ------------------------------------------------------------ |
| Megatron       | Open-source framework for large-scale transformer training   |
| TorchTitan     | PyTorch-compatible framework developed for training at scale |
| NeMo AutoModel | NVIDIA-NeMo AutoModel (diffusion: Wan 2.2 T2V); `third_party/Automodel` submodule, installed editable on first run |


## 🖥️ Single Node Training

### Setup Docker
We recommend using the official [rocm/megatron-lm Docker image](https://hub.docker.com/r/rocm/megatron-lm) to ensure a stable and compatible training environment. Use the following commands to pull and launch the container:

```bash
# Pull the latest Docker image
docker pull docker.io/rocm/primus:v26.5

```

---

### Setup Primus
Clone the repository and install dependencies:

```bash
# Clone with submodules
cd /workspace
git clone --recurse-submodules git@github.com:AMD-AGI/Primus.git

# Or initialize submodules if already cloned
git submodule update --init --recursive

cd Primus

# Install Python dependencies
pip install -r requirements.txt

# Set up pre-commit hooks
pre-commit install
```

---

### Run Pretraining

`./runner/primus-cli` is the single entry point for all training. Pick a mode based on
where you want the process to run, then pass the Primus command after `--`:

| Mode | When to use | Command |
| ---- | ----------- | ------- |
| `container` | Single node, launch from the host; the CLI starts the Docker/Podman container for you | `./runner/primus-cli container -- train pretrain --config <exp.yaml>` |
| `slurm` | Multi-node via SLURM (`srun` or `sbatch`) | `./runner/primus-cli slurm srun -N <N> -- container -- train pretrain --config <exp.yaml>` |
| `direct` | You are already inside a container (or on a prepared bare-metal host) | `./runner/primus-cli direct -- train pretrain --config <exp.yaml>` |

Notes on the argument shape:

- `--` separates launcher options from the Primus Python CLI. Everything after the last
  `--` (e.g. `--num_layers 4`) is forwarded to Primus as a config override.
- Use `train posttrain` instead of `train pretrain` for SFT / post-training leaf configs.
- Environment defaults (NCCL/RCCL, ROCm, persistent JIT caches, `DATA_PATH`, `HF_HOME`)
  come from `runner/helpers/envs/base_env.sh` plus the per-GPU file
  `runner/helpers/envs/<GPU_MODEL>.sh`; setup steps such as dependency installs, AINIC
  enablement and dataset preparation run as hooks under `runner/helpers/hooks/`.
- `runner/.primus.yaml` holds the defaults for each mode (container image, devices, and
  the list of environment variables forwarded into the container).

The packaged launchers under `examples/customer_package/` and `examples/moe_package/` share
`runner/helpers/launch/slurm_pretrain.sh`, which turns the `EXP` / `NNODES` / `DATA_PATH`
environment contract into the `primus-cli slurm srun ... -- container -- train pretrain`
command above. It is an internal helper, not an entry point: launch training with
`primus-cli` directly.

#### 🚀 Quick Start Mode

Use this mode for **rapid iteration or validation** of a model config.
You do not need to enter the Docker container. Just set the config and run.

```bash
# Example for megatron llama3.1_8B
export EXP=examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
bash ./runner/primus-cli container -- train pretrain --config "$EXP"

# Custom leaf config (SFT, mlperf, bridge_aligned, etc.)
export EXP=examples/megatron/configs/MI355X/llama3_8B-BF16-sft.yaml
bash ./runner/primus-cli container -- train posttrain --config "$EXP"

# examples for torchtitan llama3.1_8B
export EXP=examples/torchtitan/configs/MI300X/llama3.1_8B-pretrain.yaml
bash ./runner/primus-cli container -- train pretrain --config "$EXP"
```

---

#### 🧑‍🔧 Interactive Mode

This mode is recommended for **development, debugging**, or running **custom workflows**.
You will manually enter the container and execute training inside.

```bash
# Launch the container
bash tools/docker/start_container.sh

# Access the container
docker exec -it dev_primus bash

# install required packages
cd Primus && pip install -r requirements.txt

# Example for megatron llama3.1_8B
export EXP=examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
bash ./runner/primus-cli direct -- train pretrain --config "$EXP"

# examples for torchtitan llama3.1_8B
export EXP=examples/torchtitan/configs/MI300X/llama3.1_8B-pretrain.yaml
bash ./runner/primus-cli direct -- train pretrain --config "$EXP"

```

---

## 🌐 Multi-node Training

Multi-node training is launched via **SLURM**.
Specify the number of nodes and the model config:

```bash
export DOCKER_IMAGE="docker.io/rocm/primus:v26.5"
export NNODES=8

# Example for megatron llama3.1_8B
export EXP=examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
bash ./runner/primus-cli slurm srun -N "$NNODES" -- container -- train pretrain --config "$EXP"

# examples for torchtitan llama3.1_8b
export EXP=examples/torchtitan/configs/MI300X/llama3.1_8B-pretrain.yaml
bash ./runner/primus-cli slurm srun -N "$NNODES" -- container -- train pretrain --config "$EXP"
```

## 🔧 HipblasLT Auto Tuning

HipblasLT tuning is divided into three stages, selected with `PRIMUS_HIPBLASLT_TUNING_STAGE`.
The stage is only honored when the master switch `PRIMUS_HIPBLASLT_TUNING=1` is also set and
deterministic mode is off (`PRIMUS_DETERMINISTIC != 1`); otherwise the
`runner/helpers/hooks/train/pretrain/prepare_experiment.sh` hook skips tuning entirely.

```bash
# master switch: tuning is off unless this is 1
export PRIMUS_HIPBLASLT_TUNING=1
# default 0 means no tuning
export PRIMUS_HIPBLASLT_TUNING_STAGE=${PRIMUS_HIPBLASLT_TUNING_STAGE:-0}
```

---

### Stage 1: Dump GEMM Shape
In this stage, GEMM shapes used during training are collected.
It is recommended to reduce `train_iters` for faster shape generation.

```bash
# Output will be stored to:
# ./output/tune_hipblaslt/${PRIMUS_MODEL}/gemm_shape

export PRIMUS_HIPBLASLT_TUNING_STAGE=1
export EXP=examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml
bash ./runner/primus-cli slurm srun -N 1 -- container -- train pretrain --config "$EXP"
```

---

### Stage 2: Tune GEMM Kernel

This stage performs kernel tuning based on the dumped GEMM shapes using the [offline_tune tool](https://github.com/AMD-AGI/Primus/tree/main/examples/offline_tune).
It typically takes 10–30 minutes depending on model size and shape complexity.


```bash
# Output will be stored to:
# ./output/tune_hipblaslt/${PRIMUS_MODEL}/gemm_tune/tune_hipblas_gemm_results.txt

export PRIMUS_HIPBLASLT_TUNING_STAGE=2
export EXP=examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml
bash ./runner/primus-cli slurm srun -N 1 -- container -- train pretrain --config "$EXP"
```

---

### Stage 3: Train with Tuned Kernel

In this final stage, the tuned kernel is loaded for efficient training:

```bash
export PRIMUS_HIPBLASLT_TUNING_STAGE=3
export EXP=examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml
bash ./runner/primus-cli slurm srun -N 1 -- container -- train pretrain --config "$EXP"
```

## ✅ Supported Models

The following models are supported out of the box via provided configuration files:

| Model            | Huggingface Config | Megatron Config | TorchTitan Config |
| ---------------- | ------------------ | --------------- | ----------------- |
| llama2_7B        | [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)         | [llama2_7B-BF16-pretrain.yaml](https://github.com/AMD-AGI/Primus/blob/main/examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml)               | |
| llama2_70B       | [meta-llama/Llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b-hf)       | [llama2_70B-BF16-pretrain.yaml](https://github.com/AMD-AGI/Primus/blob/main/examples/megatron/configs/MI300X/llama2_70B-BF16-pretrain.yaml)             | |
| llama3_8B        | [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)     | [llama3_8B-BF16-pretrain.yaml](https://github.com/AMD-AGI/Primus/blob/main/examples/megatron/configs/MI300X/llama3_8B-BF16-pretrain.yaml)               | |
| llama3_70B       | [meta-llama/Meta-Llama-3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B)   | [llama3_70B-BF16-pretrain.yaml](https://github.com/AMD-AGI/Primus/blob/main/examples/megatron/configs/MI300X/llama3_70B-BF16-pretrain.yaml)             | |
| llama3.1_8B      | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)           | [llama3.1_8B-BF16-pretrain.yaml](https://github.com/AMD-AGI/Primus/blob/main/examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml)           | [llama3.1_8B-BF16-pretrain.yaml](https://github.com/AMD-AGI/Primus/blob/main/examples/torchtitan/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml)|
| llama3.1_70B     | [meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B)         | [llama3.1_70B-BF16-pretrain.yaml](https://github.com/AMD-AGI/Primus/blob/main/examples/megatron/configs/MI300X/llama3.1_70B-BF16-pretrain.yaml)         | [llama3.1_70B-BF16-pretrain.yaml](https://github.com/AMD-AGI/Primus/blob/main/examples/torchtitan/configs/MI300X/llama3.1_70B-BF16-pretrain.yaml)|
| llama3.1_405B     | [meta-llama/Llama-3.1-405B](https://huggingface.co/meta-llama/Llama-3.1-405B)         | [llama3.1_405B-BF16-pretrain.yaml](https://github.com/AMD-AGI/Primus/blob/main/examples/megatron/configs/MI300X/llama3.1_405B-BF16-pretrain.yaml)         | [llama3.1_405B-BF16-pretrain.yaml](https://github.com/AMD-AGI/Primus/blob/main/examples/torchtitan/configs/MI300X/llama3.1_405B-BF16-pretrain.yaml)|
| deepseek_v2_lite | [deepseek-ai/DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) | [deepseek_v2_lite-BF16-pretrain.yaml](https://github.com/AMD-AGI/Primus/blob/main/examples/megatron/configs/MI300X/deepseek_v2_lite-BF16-pretrain.yaml) | |
| deepseek_v2      | [deepseek-ai/DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2)           | [deepseek_v2-BF16-pretrain.yaml](https://github.com/AMD-AGI/Primus/blob/main/examples/megatron/configs/MI300X/deepseek_v2-BF16-pretrain.yaml)           | |
| deepseek_v3      | [deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)           | [deepseek_v3-BF16-pretrain.yaml](https://github.com/AMD-AGI/Primus/blob/main/examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml)           | |
| Mixtral-8x7B-v0.1 | [mistralai/Mixtral-8x7B-v0.1 ](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)           | [mixtral_8x7B_v0.1-BF16-pretrain.yaml](https://github.com/AMD-AGI/Primus/blob/main/examples/megatron/configs/MI300X/mixtral_8x7B_v0.1-BF16-pretrain.yaml)           | |
| Mixtral-8x22B-v0.1 | [mistralai/Mixtral-8x22B-v0.1 ](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1)           | [mixtral_8x22B_v0.1-BF16-pretrain.yaml](https://github.com/AMD-AGI/Primus/blob/main/examples/megatron/configs/MI300X/mixtral_8x22B_v0.1-BF16-pretrain.yaml)           | |

### Diffusion Models

- **Flux** - Flow-based diffusion model for text-to-image generation
  - Training guide: [examples/megatron/diffusion/README.md](megatron/diffusion/README.md) (Flux 535M and 12B)
  - Architecture & developer docs: [docs/04-technical-guides/diffusion-models/README.md](../docs/04-technical-guides/diffusion-models/README.md)
  - FP8 training: [docs/04-technical-guides/diffusion-models/fp8_training.md](../docs/04-technical-guides/diffusion-models/fp8_training.md)

---

### 🏃‍♂️ How to Run a Supported Model

Use the following command pattern to start training with a selected model configuration:

```bash
export EXP=examples/megatron/configs/MI300X/<model_config>
bash ./runner/primus-cli container -- train pretrain --config "$EXP"
```

For example, to run the llama3.1_8B model quickly:

```bash
export EXP=examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
bash ./runner/primus-cli container -- train pretrain --config "$EXP"

export EXP=examples/torchtitan/configs/MI300X/llama3.1_8B-pretrain.yaml
bash ./runner/primus-cli container -- train pretrain --config "$EXP"
```


For multi-node training via SLURM, use:

```bash
export NNODES=8

# run megatron
export EXP=examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
bash ./runner/primus-cli slurm srun -N "$NNODES" -- container -- train pretrain --config "$EXP"

# run torchtitan
export EXP=examples/torchtitan/configs/MI300X/llama3.1_8B-pretrain.yaml
bash ./runner/primus-cli slurm srun -N "$NNODES" -- container -- train pretrain --config "$EXP"
```
