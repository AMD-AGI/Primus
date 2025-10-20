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
  - [☸️ Kubernetes Training Management (`run_k8s_pretrain.sh`)](#️-kubernetes-training-management-run_k8s_pretrainsh)
    - [Requirements](#requirements)
    - [Usage](#usage)
    - [⚙️ Commands](#️-commands)
    - [⚙️ Create Command Options](#️-create-command-options)
    - [Example](#example)

---

## ⚙️ Supported Backends

Primus supports multiple backends.

| Backend    | Description                                                  |
| ---------- | ------------------------------------------------------------ |
| Megatron   | Open-source framework for large-scale transformer training   |
| TorchTitan | PyTorch-compatible framework developed for training at scale |


## 🖥️ Single Node Training

### Setup Docker
We recommend using the official [rocm/megatron-lm Docker image](https://hub.docker.com/r/rocm/megatron-lm) to ensure a stable and compatible training environment. Use the following commands to pull and launch the container:

```bash
# Pull the latest Docker image
docker pull docker.io/rocm/primus:v25.9_gfx942

```

---

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

---

### Run Pretraining
Use the `run_pretrain.sh` script to start training.

#### 🚀 Quick Start Mode

Use this mode for **rapid iteration or validation** of a model config.
You do not need to enter the Docker container. Just set the config and run.

```bash
# Example for megatron llama3.1_8B
EXP=examples/megatron/configs/MI300X/llama3.1_8B-pretrain.yaml bash ./examples/run_local_pretrain.sh

# examples for torchtitan llama3.1_8B
EXP=examples/torchtitan/mi300/llama3.1_8B-pretrain.yaml bash ./examples/run_local_pretrain.sh
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
EXP=examples/megatron/configs/MI300X/llama3.1_8B-pretrain.yaml bash ./examples/run_pretrain.sh

# examples for torchtitan llama3.1_8B
EXP=examples/torchtitan/mi300/llama3.1_8B-pretrain.yaml bash ./examples/run_pretrain.sh

```

---

## 🌐 Multi-node Training

Multi-node training is launched via **SLURM**.
Specify the number of nodes and the model config:

```bash
export DOCKER_IMAGE="docker.io/rocm/primus:v25.9_gfx942"
export NNODES=8

# Example for megatron llama3.1_8B
EXP=examples/megatron/configs/MI300X/llama3.1_8B-pretrain.yaml bash ./examples/run_slurm_pretrain.sh

# examples for torchtitan llama3.1_8b
EXP=examples/torchtitan/mi300/llama3.1_8B-pretrain.yaml bash ./examples/run_slurm_pretrain.sh
```

## 🔧 HipblasLT Auto Tuning

HipblasLT tuning is divided into three stages and controlled via the environment variable `PRIMUS_HIPBLASLT_TUNING_STAGE`:

```bash
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
export EXP=examples/megatron/configs/MI300X/llama2_7B-pretrain.yaml
NNODES=1 bash ./examples/run_slurm_pretrain.sh
```

---

### Stage 2: Tune GEMM Kernel

This stage performs kernel tuning based on the dumped GEMM shapes using the [offline_tune tool](https://github.com/AMD-AIG-AIMA/Primus/tree/main/examples/offline_tune).
It typically takes 10–30 minutes depending on model size and shape complexity.


```bash
# Output will be stored to:
# ./output/tune_hipblaslt/${PRIMUS_MODEL}/gemm_tune/tune_hipblas_gemm_results.txt

export PRIMUS_HIPBLASLT_TUNING_STAGE=2
export EXP=examples/megatron/configs/MI300X/llama2_7B-pretrain.yaml
NNODES=1 bash ./examples/run_slurm_pretrain.sh
```

---

### Stage 3: Train with Tuned Kernel

In this final stage, the tuned kernel is loaded for efficient training:

```bash
export PRIMUS_HIPBLASLT_TUNING_STAGE=3
export EXP=examples/megatron/configs/MI300X/llama2_7B-pretrain.yaml
NNODES=1 bash ./examples/run_slurm_pretrain.sh
```

## ✅ Supported Models

The following models are supported out of the box via provided configuration files:

| Model            | Huggingface Config | Megatron Config | TorchTitan Config |
| ---------------- | ------------------ | --------------- | ----------------- |
| llama2_7B        | [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)         | [llama2_7B-pretrain.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/examples/megatron/configs/MI300X/llama2_7B-pretrain.yaml)               | |
| llama2_70B       | [meta-llama/Llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b-hf)       | [llama2_70B-pretrain.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/examples/megatron/configs/MI300X/llama2_70B-pretrain.yaml)             | |
| llama3_8B        | [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)     | [llama3_8B-pretrain.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/examples/megatron/configs/MI300X/llama3_8B-pretrain.yaml)               | |
| llama3_70B       | [meta-llama/Meta-Llama-3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B)   | [llama3_70B-pretrain.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/examples/megatron/configs/MI300X/llama3_70B-pretrain.yaml)             | |
| llama3.1_8B      | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)           | [llama3.1_8B-pretrain.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/examples/megatron/configs/MI300X/llama3.1_8B-pretrain.yaml)           | [llama3.1_8B-pretrain.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/examples/torchtitan/mi300/llama3.1_8B-pretrain.yaml)|
| llama3.1_70B     | [meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B)         | [llama3.1_70B-pretrain.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/examples/megatron/configs/MI300X/llama3.1_70B-pretrain.yaml)         | [llama3.1_70B-pretrain.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/examples/torchtitan/mi300/llama3.1_70B-pretrain.yaml)|
| llama3.1_405B     | [meta-llama/Llama-3.1-405B](https://huggingface.co/meta-llama/Llama-3.1-405B)         | [llama3.1_405B-pretrain.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/examples/megatron/configs/MI300X/llama3.1_405B-pretrain.yaml)         | [llama3.1_405B-pretrain.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/examples/torchtitan/mi300/llama3.1_405B-pretrain.yaml)|
| deepseek_v2_lite | [deepseek-ai/DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) | [deepseek_v2_lite-pretrain.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/examples/megatron/configs/MI300X/deepseek_v2_lite-pretrain.yaml) | |
| deepseek_v2      | [deepseek-ai/DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2)           | [deepseek_v2-pretrain.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/examples/megatron/configs/MI300X/deepseek_v2-pretrain.yaml)           | |
| deepseek_v3      | [deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)           | [deepseek_v3-pretrain.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/examples/megatron/configs/MI300X/deepseek_v3-pretrain.yaml)           | |
| Mixtral-8x7B-v0.1 | [mistralai/Mixtral-8x7B-v0.1 ](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)           | [mixtral_8x7B_v0.1-pretrain.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/examples/megatron/configs/MI300X/mixtral_8x7B_v0.1-pretrain.yaml)           | |
| Mixtral-8x22B-v0.1 | [mistralai/Mixtral-8x22B-v0.1 ](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1)           | [mixtral_8x22B_v0.1-pretrain.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/examples/megatron/configs/MI300X/mixtral_8x22B_v0.1-pretrain.yaml)           | |

---

### 🏃‍♂️ How to Run a Supported Model

Use the following command pattern to start training with a selected model configuration:

```bash
EXP=examples/megatron/configs/MI300X/<model_config> bash ./examples/run_local_pretrain.sh
```

For example, to run the llama3.1_8B model quickly:

```bash
EXP=examples/megatron/configs/MI300X/llama3.1_8B-pretrain.yaml bash ./examples/run_local_pretrain.sh

EXP=examples/torchtitan/mi300/llama3.1_8B-pretrain.yaml bash ./examples/run_local_pretrain.sh
```


For multi-node training via SLURM, use:

```bash
export NNODES=8

#run megatron
EXP=examples/megatron/configs/MI300X/llama3.1_8B-pretrain.yaml bash ./examples/run_slurm_pretrain.sh

# run torchtitan
EXP=examples/torchtitan/mi300/llama3.1_8B-pretrain.yaml bash ./examples/run_slurm_pretrain.sh
```

## ☸️ Kubernetes Training Management (`run_k8s_pretrain.sh`)

The `run_k8s_pretrain.sh` script provides convenient CLI commands to manage training workloads on a Kubernetes cluster via a REST API. It supports creating, querying, deleting training jobs, and listing cluster nodes, facilitating flexible workload control for distributed training with Primus or similar frameworks.

### Requirements

- `jq` installed (for JSON processing)
- Access to Kubernetes API endpoint URL

### Usage

```bash
./run_k8s_pretrain.sh --url <api_base_url> <command> [options]

```



### ⚙️ Commands

Primus provides several command-line interfaces to manage training workloads and cluster resources. Below are the commonly used commands:

| Command | Description                    |
| ------- | ------------------------------|
| create  | Create a new training workload |
| get     | Retrieve workload details      |
| delete  | Delete an existing workload    |
| list    | List all current workloads     |
| nodes   | List all nodes in the cluster  |

Use these commands to interact with Primus for workload scheduling and resource management.


---

### ⚙️ Create Command Options

When using the `create` command to start a new training workload, the following options are supported:

| Option       | Description                                          | Default                                  |
| ------------ | ---------------------------------------------------- | ---------------------------------------- |
| `--replica`    | Number of replicas (instances)                       | 1                                        |
| `--cpu`        | Number of CPUs                                       | 96                                       |
| `--gpu`        | Number of GPUs                                       | 8                                        |
| `--exp`        | Path to experiment (training config) file (required) | —                                        |
| `--data_path`  | Path to training data                                | —                                        |
| `--image`      | Docker image to use                                  | `docker.io/rocm/primus:v25.9_gfx942` |
| `--hf_token`   | HuggingFace token                                    | Read from env var `HF_TOKEN`             |
| `--workspace`  | Workspace name                                       | `primus-safe-pretrain`                   |
| `--nodelist`   | Comma-separated list of node hostnames to run on     | —                                        |

### Example

Create a training workload with 2 replicas and custom config:


```bash
bash examples/run_k8s_pretrain.sh --url http://api.example.com create --replica 2 --cpu 96 --gpu 4 \
  --exp examples/megatron/configs/MI300X/llama2_7B-pretrain.yaml --data_path /mnt/data/train \
  --image docker.io/custom/image:latest --hf_token myhf_token --workspace team-dev

#result:
{
  "workloadId": "abc123"
}

```

Get workload details:

```bash
bash examples/run_k8s_pretrain.sh --url http://api.example.com get --workload-id abc123

```

Delete a workload:

```bash
bash examples/run_k8s_pretrain.sh --url http://api.example.com delete --workload-id abc123

```

List all workloads:

```bash
bash examples/run_k8s_pretrain.sh --url http://api.example.com list

```

List all cluster nodes:

```bash
bash examples/run_k8s_pretrain.sh --url http://api.example.com nodes

```
