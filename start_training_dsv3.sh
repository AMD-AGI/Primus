#!/bin/bash
# Start training with W&B loss logging on rank-0

echo "=== Starting DeepSeek V3 Training ==="
echo "This script will:"
echo "1. Allocate 8 nodes using SLURM"
echo "2. Run training with W&B logging enabled on rank-0"
echo ""

# Load Docker image if not already present
if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -q 'primus_kernel_benchmark:backup'; then
    echo "Loading Docker image..."
    docker load -i /data/john/primus_kernel_benchmark_backup.tar
else
    echo "Docker image already loaded, skipping."
fi

# Clean old output (optional - comment out if you want to keep old runs)
echo "Cleaning old output directory..."
sudo rm -rf /data/john/Primus/output/amd/root/deepseek_v3-pretrain/* 2>/dev/null

# Allocate nodes and run training
echo "Allocating 8 nodes (excluding GPU-20,GPU-73) and starting training..."
salloc -N 8 \
    --exclude=GPU-73 \
    --ntasks-per-node=1 \
    --cpus-per-task=128 \
    --exclusive --mem=0 \
    --job-name=qyy_test \
    --time=12:00:00 \
    bash -c 'cd /data/john/Primus && bash run_dsv3.sh'

echo "Training completed or allocation ended."
