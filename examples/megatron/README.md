# Megatron Training Example

This example demonstrates how to perform pretraining using Megatron within the Primus framework. It supports both single-node and multi-node training and includes features like HipblasLT auto-tuning for optimal performance.


## 📚 Table of Contents
- [Megatron Training Example](#megatron-training-example)
  - [📚 Table of Contents](#-table-of-contents)
  - [🖥️ Single Node Training](#️-single-node-training)
    - [Setup Docker](#setup-docker)
    - [Setup Primus](#setup-primus)
    - [Run Pretraining](#run-pretraining)
  - [🌐 Multi-node Training](#-multi-node-training)
  - [🔧 HipblasLT Auto Tuning](#-hipblaslt-auto-tuning)
    - [Stage 1: Dump GEMM Shape](#stage-1-dump-gemm-shape)
    - [Stage 2: Tune GEMM Kernel](#stage-2-tune-gemm-kernel)
    - [Stage 3: Train with Tuned Kernel](#stage-3-train-with-tuned-kernel)


## 🖥️ Single Node Training

### Setup Docker
We recommend using the official [rocm/megatron-lm Docker image](https://hub.docker.com/r/rocm/megatron-lm) to ensure a stable and compatible training environment. Use the following commands to pull and launch the container:

```bash
# Pull the latest Docker image
docker pull docker.io/rocm/megatron-lm:latest

```

### Setup Primus
Clone the repository and install dependencies:

```bash
# Clone with submodules
cd /workspace
git clone --recurse-submodules git@github.com:AMD-AIG-AIMA/Primus.git

# Or initialize submodules if already cloned
git submodule update --init --recursive

cd Primus

# Install Python dependencies
pip install -r requirements.txt

# Set up pre-commit hooks
pre-commit install
```

### Run Pretraining
Use the `run_pretrain.sh` script to start training. The model config should match the YAML filename under `primus/configs/models/megatron` (excluding the `.yaml` extension):

#### Auto
```bash
# Example for llama2_7B
MODEL_CONFIG=llama2_7B ./examples/megatron/run_local_pretrain.sh

# Example for deepseek_v2_lite
MODEL_CONFIG=deepseek_v2_lite ./examples/megatron/run_local_pretrain.sh

```

#### Manual
```bash
# Launch the container
bash tools/docker/start_container.sh

# Access the container
docker exec -it dev_primus bash

# Example for llama2_7B
MODEL_CONFIG=llama2_7B ./examples/megatron/run_pretrain.sh

```


## 🌐 Multi-node Training
Multi-node training is launched via SLURM. Specify the number of nodes and model config:

```bash
export DOCKER_IMAGE="docker.io/rocm/megatron-lm:latest"
NUM_NODES=8 MODEL_CONFIG=llama2_7B ./examples/megatron/run_slurm_pretrain.sh
```

## 🔧 HipblasLT Auto Tuning
HipblasLT tuning is divided into three stages and controlled via the environment variable `PRIMUS_HIPBLASLT_TUNING_STAGE`:

```bash
# default 0 means no tuning
export PRIMUS_HIPBLASLT_TUNING_STAGE=${PRIMUS_HIPBLASLT_TUNING_STAGE:-0}
```

### Stage 1: Dump GEMM Shape
In this stage, GEMM shapes used during training are collected. It is recommended to reduce `train_iters` for faster shape generation. The output will be stored in:

```./output/tune_hipblaslt/${MODEL_CONFIG}/gemm_shape```

```bash
PRIMUS_HIPBLASLT_TUNING_STAGE=1 NUM_NODES=8 MODEL_CONFIG=deepseek_v2_lite bash ./examples/megatron/run_slurm_pretrain.sh
```

### Stage 2: Tune GEMM Kernel
This stage performs kernel tuning based on the dumped GEMM shapes using the [offline_tune tool](https://github.com/AMD-AIG-AIMA/Primus/tree/main/examples/offline_tune). It typically takes 10–30 minutes depending on model size and shape complexity. Output is saved to:

```./output/tune_hipblaslt/${MODEL_CONFIG}/gemm_tune/tune_hipblas_gemm_results.txt```

```bash
PRIMUS_HIPBLASLT_TUNING_STAGE=2 NUM_NODES=1 MODEL_CONFIG=deepseek_v2_lite bash ./examples/megatron/run_slurm_pretrain.sh
```

### Stage 3: Train with Tuned Kernel
In this final stage, the tuned kernel is loaded for efficient training:

```bash
PRIMUS_HIPBLASLT_TUNING_STAGE=3 NUM_NODES=1 MODEL_CONFIG=deepseek_v2_lite bash ./examples/megatron/run_slurm_pretrain.sh
```
