

---

## 1. Setup Primus
Clone the repository and install dependencies:

```bash
git clone -b dev/tas/20251218 https://github.com/AMD-AGI/Primus.git

# initialize submodules if already cloned
git submodule update --init --recursive
```

---

## 2. Single Node Training

### 2.1 Setup Docker
We recommend using the official [rocm/primus](https://hub.docker.com/r/rocm/primus/tags) docker image to ensure a stable and compatible training environment.

```bash
export DOCKER_IMAGE="docker.io/rocm/primus:v25.10"

# create a container named dev_primus
cd Primus && bash ./tools/docker/start_container.sh

```

---

## 2.2 Run Pretraining
Use the `run_pretrain.sh` script to start training.

```bash
# Access the container dev_primus
docker exec -it dev_primus bash

# inside the container dev_primus
cd /workspace/Primus

# set your huggingface token
export HF_TOKEN=${HF_TOKEN}

# Example for llama3.1_70B FSDP2 Training
# logs: ./output/amd/root/llama3_70B-pretrain
EXP=examples/megatron/configs/MI300X/llama3.1_70B-pretrain.yaml bash ./examples/run_pretrain.sh --train_iters 10

```
