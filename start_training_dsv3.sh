#!/bin/bash
# Start training with W&B loss logging on rank-0

export NNODES=24 # modify the number of nodes here

echo "=== Starting DeepSeek V3 Training ==="
echo "This script will:"
echo "1. Allocate ${NNODES} nodes using SLURM"
echo "2. Run training with W&B logging enabled on rank-0"
echo ""

# Clean old output (optional - comment out if you want to keep old runs)
echo "Cleaning old output directory..."
rm -rf output/amd/root/deepseek_v3-pretrain/* 2>/dev/null

# Allocate nodes and run training
echo "Allo  cating ${NNODES} nodes and starting training..."
salloc -N ${NNODES} \
    --ntasks-per-node=1 \
    --cpus-per-task=128 \
    --exclusive --mem=0 \
    --job-name=dsv3_test \
    --time=100-00:00:00 \
    --partition=amd-slc \
    bash -c '
        echo "Loading Docker image on all nodes..."
        srun --ntasks-per-node=1 bash -c "
            if ! docker images --format \"{{.Repository}}:{{.Tag}}\" | grep -q \"john132/tas:primus-25.9-ainic-56\"; then
                if [ -f /shared/primus-25.9-ainic-56.tar ]; then
                    echo \"[\$(hostname)] Loading Docker image from tar...\"
                    docker load -i /shared/primus-25.9-ainic-56.tar
                else
                    echo \"[\$(hostname)] Tar file not found, pulling Docker image...\"
                    docker pull john132/tas:primus-25.9-ainic-56
                fi
            else
                echo \"[\$(hostname)] Docker image already loaded, skipping.\"
            fi
        "
        echo "Docker image loaded on all nodes. Starting training..."
        bash run_dsv3.sh
    '

echo "Training completed or allocation ended."

