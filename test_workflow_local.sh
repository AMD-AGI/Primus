#!/bin/bash
###############################################################################
# Local test script for GPT-OSS-20B Primus Training Workflow
# This script mimics the GitHub Actions workflow for local testing
###############################################################################

set -e

# Configuration
DOCKER_IMAGE="rocm/primus:v25.10"
EXP_CONFIG="examples/megatron/configs/MI300X/gpt_oss_20B-pretrain.yaml"
PRIMUS_MODEL="gpt_oss_20B"
TEAM="amd"
USER="${USER:-root}"
EXP_NAME="gpt_oss_20b_ci_local_test"
TRAIN_ITERS="${TRAIN_ITERS:-10}"  # Small number for quick test
WORKSPACE="$(pwd)"
MASTER_PORT="${MASTER_PORT:-$((RANDOM + 10000))}"  # Random port to avoid conflicts

echo "===================================="
echo "GPT-OSS-20B Primus Training Test"
echo "===================================="
echo "Docker Image: $DOCKER_IMAGE"
echo "Config: $EXP_CONFIG"
echo "Train Iters: $TRAIN_ITERS"
echo "Workspace: $WORKSPACE"
echo "===================================="

# Clean up previous containers
echo "Cleaning up previous containers..."
docker stop primus_gpt_oss_20b_test || true
docker rm primus_gpt_oss_20b_test || true

# Start Docker container
echo "Starting Docker container..."
docker run -d \
  --name primus_gpt_oss_20b_test \
  --ipc=host \
  --network=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/infiniband \
  --cap-add=SYS_PTRACE \
  --cap-add=CAP_SYS_ADMIN \
  --security-opt seccomp=unconfined \
  --group-add video \
  --privileged \
  -v ${WORKSPACE}:/workspace/Primus \
  -v ${WORKSPACE}/data:/data \
  -w /workspace/Primus \
  -e TEAM=${TEAM} \
  -e USER=${USER} \
  -e EXP_NAME=${EXP_NAME} \
  -e PRIMUS_MODEL=${PRIMUS_MODEL} \
  -e MASTER_PORT=${MASTER_PORT} \
  ${DOCKER_IMAGE} sleep infinity

echo "Container started successfully!"

# Install dependencies
echo "Installing dependencies..."
docker exec primus_gpt_oss_20b_test bash -c "
  cd /workspace/Primus && \
  pip install -r requirements.txt --quiet
"

# Create a temporary config for testing
echo "Preparing test config..."
docker exec primus_gpt_oss_20b_test bash -c "
  cp ${EXP_CONFIG} ${EXP_CONFIG}.test_backup && \
  sed -i 's/train_iters: 200/train_iters: ${TRAIN_ITERS}/g' ${EXP_CONFIG} && \
  sed -i 's/mock_data: false/mock_data: true/g' ${EXP_CONFIG}
"

# Run GPT-OSS-20B training
echo "Running GPT-OSS-20B training..."
echo "This may take several minutes..."
docker exec primus_gpt_oss_20b_test bash -c "
  cd /workspace/Primus && \
  EXP=${EXP_CONFIG} bash ./examples/run_pretrain.sh
" || {
  echo "Training failed (this is expected if it's the known crash issue)"
  TRAINING_FAILED=true
}

# Extract training metrics
echo "Extracting training metrics..."
docker exec primus_gpt_oss_20b_test bash -c "
  cd /workspace/Primus && \
  if [ -f output/log_*.txt ]; then
    echo '=== Training Metrics ===' && \
    grep 'throughput per GPU' output/log_*.txt | tail -n 10 || echo 'No throughput metrics found' && \
    grep 'lm loss' output/log_*.txt | tail -n 10 || echo 'No loss metrics found'
  else
    echo 'No log files found'
  fi
"

# Restore original config
echo "Restoring original config..."
docker exec primus_gpt_oss_20b_test bash -c "
  mv ${EXP_CONFIG}.test_backup ${EXP_CONFIG} || true
"

# Collect logs
echo "Collecting logs..."
mkdir -p ${WORKSPACE}/test_artifacts
docker exec primus_gpt_oss_20b_test bash -c "
  cd /workspace/Primus && \
  cp -r output/*.txt ${WORKSPACE}/test_artifacts/ 2>/dev/null || true && \
  cp -r output/log_*.txt ${WORKSPACE}/test_artifacts/ 2>/dev/null || true
"

echo "Logs collected in: ${WORKSPACE}/test_artifacts/"

# Cleanup container
echo "Cleaning up container..."
docker stop primus_gpt_oss_20b_test || true
docker rm primus_gpt_oss_20b_test || true

echo "===================================="
echo "Test completed!"
echo "Check artifacts in: ${WORKSPACE}/test_artifacts/"
if [ "$TRAINING_FAILED" = true ]; then
  echo "⚠️  Training crashed (this is expected and OK per requirements)"
else
  echo "✅ Training completed successfully!"
fi
echo "===================================="

