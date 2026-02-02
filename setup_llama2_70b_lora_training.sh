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
DATA_DIR=""${MOUNT_DATA_PATH}"/mlperf_llama2"${MOUNT_DATA_PATH}""
SEQ_LENGTH=8192
HF_TOKEN="${HF_TOKEN:-}"
if [ -z "$HF_TOKEN" ]; then
    echo -e "${RED}[ERROR]${NC} HF_TOKEN is not set or is empty. Please set your Hugging Face token in the environment or in this script."
    exit 1
fi

# Docker configuration (for start_container.sh)
export DOCKER_IMAGE="${DOCKER_IMAGE:-docker.io/rocm/primus:v25.10}"
export DATA_PATH="${DATA_DIR}"
export MOUNT_DATA_PATH="${MOUNT_DATA_PATH}"
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

    if [ ! -f data/packed_metadata.jsonl ]; then
        python3 create_metadata.py ${SEQ_LENGTH} data/packed_metadata.jsonl
    else
        echo 'Metadata already exists: data/packed_metadata.jsonl'
    fi
"

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


# Step 4: Install rocmProfileData
echo_info "Step 4: Installing rocmProfileData..."
docker exec "${CONTAINER_NAME}" bash -c "
    if [ ! -d /tmp/rocmProfileData ]; then
        cd /tmp
        git clone https://github.com/ROCmSoftwarePlatform/rocmProfileData
        cd rocmProfileData

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

# Step 5: Start training
echo_info "Step 5: Starting llama2_70b_lora training..."
echo_info "Training configuration: examples/megatron_bridge/configs/MI355X/llama2_70b_lora_posttrain.yaml"
docker exec -it "${CONTAINER_NAME}" bash -c "cd /workspace/Primus && HF_TOKEN=${HF_TOKEN} ./runner/primus-cli direct train posttrain --config examples/megatron_bridge/configs/MI355X/llama2_70b_lora_posttrain.yaml"
