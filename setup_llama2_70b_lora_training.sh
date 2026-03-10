#!/bin/bash
# Script to setup and start llama2_70b_lora training
# This script automates all the steps required to start training

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="/data/mlperf_llama2/data"
SEQ_LENGTH=8192
HF_TOKEN="${HF_TOKEN:-}"
if [ -z "$HF_TOKEN" ]; then
    echo -e "${RED}[ERROR]${NC} HF_TOKEN is not set or is empty. Please set your Hugging Face token in the environment or in this script."
    exit 1
fi
WANDB_API_KEY="${WANDB_API_KEY:-}"
if [ -z "$WANDB_API_KEY" ]; then
    echo -e "${RED}[WARNING]${NC} WANDB_API_KEY is not set or is empty."
fi

# Docker configuration (for start_container.sh)
export DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/mad-private:primus_ci_1146b7a_20260225}"
export DATA_PATH="${DATA_DIR}"
export MOUNT_DATA_PATH="${MOUNT_DATA_PATH:-/data}"
export CONTAINER_NAME="${CONTAINER_NAME:-primus_llama2_lora}"

# Step 1: Update submodules
echo_info "Step 1: Updating submodules..."
git submodule update --init --recursive
echo_info "Submodules updated successfully"

# Step 2: Start Primus Docker container with data mount
echo_info "Step 2: Starting Primus Docker container..."
echo_info "Data directory: ${DATA_DIR}"
echo_info "Container name: ${CONTAINER_NAME}"
echo_info "Docker image: ${DOCKER_IMAGE}"

# Check if data directory exists
if [ ! -d "${DATA_DIR}" ]; then
    echo_error "Data directory ${DATA_DIR} does not exist!"
    echo_error "Please create it or update DATA_DIR variable in this script."
    exit 1
fi

# Check if container is already running
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo_info "Container ${CONTAINER_NAME} is already running."
elif docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo_info "Container ${CONTAINER_NAME} exists but is stopped. Starting it..."
    docker start "${CONTAINER_NAME}"
else
    echo_info "Creating and starting new container using start_container.sh..."
    bash "${SCRIPT_DIR}/tools/docker/start_container.sh"
    echo_info "Container started successfully"
fi

# Step 3: Download dataset and create tokenizer metadata
echo_info "Step 3: Download dataset and create tokenizer metadata..."
docker exec "${CONTAINER_NAME}" bash -c "
    cd /workspace/Primus
    if [ ! -f ${MOUNT_DATA_PATH}/train.npy ]; then
        python download_dataset.py --data_dir ${MOUNT_DATA_PATH}
        python convert_dataset.py --data_dir ${MOUNT_DATA_PATH}
    else
        echo 'Dataset already exists: ${MOUNT_DATA_PATH}/train.npy'
    fi

    if [ ! -f ${MOUNT_DATA_PATH}/packed_metadata.jsonl ]; then
        python3 create_metadata.py ${SEQ_LENGTH} ${MOUNT_DATA_PATH}/packed_metadata.jsonl
    else
        echo 'Metadata already exists: ${MOUNT_DATA_PATH}/packed_metadata.jsonl'
    fi
"

# Step 4: Install rocmProfileData
echo_info "Step 4: Installing rocmProfileData..."
docker exec "${CONTAINER_NAME}" bash -c "
    sudo apt-get update && \
    sudo apt-get install -y uuid-runtime sqlite3 libsqlite3-dev libfmt-dev

    if [ ! -d /tmp/rocmProfileData ]; then
        cd /tmp
        git clone https://github.com/ROCmSoftwarePlatform/rocmProfileData
        cd rocmProfileData
        # rpd stable commit
        git checkout 13e7600ca75b34078927bbc5a9f9191882e7048e

        # Build and install rocpd_python
        cd rocpd_python
        python3 setup.py bdist_wheel
        pip install dist/*.whl
        cd ..

        # Build and install rpd_tracer
        cd rpd_tracer
        python3 setup.py bdist_wheel
        pip3 install dist/*.whl
        cd ..

        # Fix Makefiles
        sed -i 's/pip install --user \./pip install ./g' rocpd_python/Makefile
        sed -i 's/pip install --user \./pip install ./g' rpd_tracer/Makefile

        # Install
        make && make install
        echo 'rocmProfileData installed successfully'
    else
        echo 'rocmProfileData already installed'
    fi
"
# Step 5: Patch Megatron-Bridge for llama2_70b_lora training. Fix consumed samples in Megtron-Bridge.
echo_info "Step 5: Patching Megatron-Bridge for llama2_70b_lora training..."
docker exec "${CONTAINER_NAME}" bash -c "
    cd /workspace/Primus/third_party/Megatron-Bridge
    git apply /workspace/Primus/megatron_bridge_validation_consumed_samples.patch || true"

# Step 5b: Patch Megatron-Bridge for deterministic evaluation. Resets validation data iterator
# before each evaluation so every eval pass sees the exact same data in the same order.
echo_info "Step 5b: Patching Megatron-Bridge for deterministic evaluation..."
docker exec "${CONTAINER_NAME}" bash -c "
    cd /workspace/Primus/third_party/Megatron-Bridge
    git apply /workspace/Primus/megatron_bridge_deterministic_eval.patch || true"

# Step 6: Start training
echo_info "Step 6: Starting llama2_70b_lora training..."
echo_info "Training configuration: examples/megatron_bridge/configs/MI355X/llama2_70b_lora_posttrain.yaml"
docker exec -it "${CONTAINER_NAME}" bash -c "
    cd /workspace/Primus && \
    HF_TOKEN=${HF_TOKEN} \
    WANDB_API_KEY=${WANDB_API_KEY} \
    MEGATRON_BRIDGE_LOGGING_LEVEL=10 \
    NVTE_DEBUG=1 \
    NVTE_DEBUG_LEVEL=2 \
    NVTE_FUSED_ATTN_LOG_CONFIG=1 \
    NVTE_RS_STRIDED_ATOMIC=2 \
    NVTE_FP8_DPA_BWD=1
    NVTE_FUSED_ATTN=1 && \
    NVTE_FUSED_ATTN_CK=1 && \
    NVTE_FUSED_ATTN_AOTRITON=1 && \
    NVTE_USE_HIPBLASLT=1 && \
    NVTE_USE_CAST_TRANSPOSE_TRITON=1 && \
    NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=0 && \
    NVTE_CK_USES_BWD_V3=1 && \
    NVTE_CK_USES_FWD_V3=1 && \
    NVTE_CK_IS_V3_ATOMIC_FP32=0 && \
    ./runner/primus-cli direct train posttrain --config examples/megatron_bridge/configs/MI355X/llama2_70b_lora_posttrain.yaml"

    #TODO: TE_SWIGLU
