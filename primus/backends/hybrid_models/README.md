### Quick Start with AMD ROCm Docker Image: Hybrid Models Pretraining

1. Pull Pytorch Docker image

    ```bash
    docker pull docker.io/rocm/pytorch-training:v25.4
    ```
2. Run Hybrid models with config file and backend specification

    ```bash
    HF_TOKEN=<hf_token> DOCKER_IMAGE=rocm/pytorch-training:v25.4 \
    EXP=examples/hybrid_models/zebra_llama/configs/llama3.2_1B-pretrain.yaml \
    BACKEND_PATH=third_party/AMD-Hybrid-Models/Zebra-Llama \
    bash ./examples/run_local_pretrain.sh
    ```
