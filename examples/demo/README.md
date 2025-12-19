# Primus Quick Start Guide

---

## 1. Setup Primus

```bash
# Enter a working directory on the shared filesystem (e.g., NFS), which can be accessed by multiple nodes
cd /nfs/workspace

# Clone Primus repository
git clone -b dev/tas/20251218 https://github.com/AMD-AGI/Primus.git

# Initialize submodules if already cloned
git submodule update --init --recursive
```

---

## 2. Single Node Training

### 2.1 Setup Docker

We recommend using the official [rocm/primus](https://hub.docker.com/r/rocm/primus/tags) docker image to ensure a stable and compatible training environment.

```bash
export DOCKER_IMAGE="docker.io/rocm/primus:v25.10"

# Create a container named dev_primus
cd /nfs/workspace/Primus && bash ./tools/docker/start_container.sh
```

### 2.2 Run Pretraining

Use the `run_pretrain.sh` script to start training.

```bash
# Access the container dev_primus
docker exec -it dev_primus bash

# Inside the container dev_primus
cd /nfs/workspace/Primus

# Set your HuggingFace token
export HF_TOKEN=${HF_TOKEN}

# Example for Llama3.1 70B FSDP2 Training
# Logs: ./output/amd/root/llama3_70B-pretrain
EXP=examples/megatron/configs/MI300X/llama3.1_70B-pretrain.yaml bash ./examples/run_pretrain.sh \
    --micro_batch_size 2 --global_batch_size 16
```

---

## 3. Launch Multi-Node Training from Platform

> **Note:**
> On training platforms, the system will automatically schedule multiple worker nodes, create containers from the user-provided **Docker image**, and run the user-supplied **training command** inside those containers.

### 3.1 Docker Image

We recommend using the official [rocm/primus](https://hub.docker.com/r/rocm/primus/tags) docker image to ensure a stable and compatible training environment.

```text
Docker Image: docker.io/rocm/primus:v25.10
```

### 3.2 4-Nodes Llama3.1 70B FSDP2 Training

```bash
# Note: Please set the distributed variables provided by your platform
export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}
export NNODES=${NNODES}
export NODE_RANK=${NODE_RANK}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# Set your HuggingFace token and WandB key
export HF_TOKEN=${HF_TOKEN}
export WANDB_API_KEY=${WANDB_API_KEY}

# Change directory to your Primus
cd /nfs/workspace/Primus

# Example for Llama3.1 70B FSDP2 Training
# Logs: ./output/amd/root/llama3_70B-pretrain
EXP=examples/megatron/configs/MI300X/llama3.1_70B-pretrain.yaml bash ./examples/run_pretrain.sh \
    --train_iters 100 \
    --micro_batch_size 2 \
    --global_batch_size 64 \
    --disable_wandb False
```

### 3.3 4-Nodes DeepSeek-V2-Lite MoE Training

```bash
# Note: Please set the distributed variables provided by your platform
export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}
export NNODES=${NNODES}
export NODE_RANK=${NODE_RANK}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# Set your HuggingFace token and WandB key
export HF_TOKEN=${HF_TOKEN}
export WANDB_API_KEY=${WANDB_API_KEY}

# Change directory to your Primus
cd /nfs/workspace/Primus

export ENABLE_NUMA_BINDING=1
export HSA_KERNARG_POOL_SIZE=12582912

# Example for DeepSeek-V2-Lite MoE Training
# Logs: ./output/amd/root/deepseek_v2_lite-pretrain
EXP=examples/megatron/configs/MI300X/deepseek_v2_lite-pretrain.yaml bash ./examples/run_pretrain.sh \
    --train_iters 300 \
    --micro_batch_size 4 \
    --global_batch_size 512 \
    --disable_wandb False \
    --manual_gc True
```

---
