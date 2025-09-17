# Primus

**Primus** is a flexible and high-performance training framework designed for large-scale foundation model training and inference. It is designed to support **pretraining**, **posttraining**, and **reinforcement learning** workflows, and is compatible with multiple backends including [Megatron](https://github.com/NVIDIA/Megatron-LM) and ROCm-optimized components.

---

## 🆕 What's New
- **[2025/06/18]** Added TorchTitan backend support.
- **[2025/05/16]** Added benchmark suite for performance evaluation across models and hardware.
- **[2025/04/18]** Added [Preflight](./tools/preflight/README.md) cluster sanity checker to verify environment readiness.
- **[2025/04/14]** Integrated HipblasLT autotuning for optimized GPU kernel performance.
- **[2025/04/09]** Extended support for LLaMA2, LLaMA3, DeepSeek-V2/V3 models in [Megatron model configs](https://github.com/AMD-AIG-AIMA/Primus/tree/main/primus/configs/models/megatron).
- **[2025/03/04]** Released Megatron trainer module for flexible and efficient large model training.

---


## 🚀 Setup & Deployment

Primus leverages AMD’s ROCm Docker images to provide a consistent, ready-to-run environment optimized for AMD GPUs. This eliminates manual dependency and environment configuration.

### Prerequisites

- AMD ROCm drivers (version ≥ 6.0 recommended)
- Docker (version ≥ 24.0) with ROCm support
- ROCm-compatible AMD GPUs (e.g., Instinct MI300 series)
- Proper permissions for Docker and GPU device access


### Quick Start with AMD ROCm Docker Image: Megatron Pretraining

1. Pull the latest Docker image

    ```bash
    docker pull docker.io/rocm/megatron-lm:v25.7_py310

    ```

2. Clone the repository:

    ```bash
    git clone --recurse-submodules https://github.com/AMD-AIG-AIMA/Primus.git

    ```

3. Run Pretraining

    ```bash
    cd Primus && pip install -r requirements.txt
    EXP=examples/megatron/configs/llama2_7B-pretrain.yaml bash ./examples/run_local_pretrain.sh

    ```

For more detailed usage instructions, configuration options, and examples, please refer to the [examples/README.md](./examples/README.md).

---

## 📝 TODOs

- [ ] Support for Primus-RL (training/inference modules for RLHF, OnlineDPO, GRPO, etc.)
- [ ] Add support for more model architectures and backends
