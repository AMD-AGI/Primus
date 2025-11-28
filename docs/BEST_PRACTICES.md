# Primus CLI Best Practices Guide

## Overview

This guide provides best practices, patterns, and recommendations for using Primus CLI effectively in production environments.

---

## Table of Contents

1. [Configuration Management](#configuration-management)
2. [Workflow Organization](#workflow-organization)
3. [Resource Management](#resource-management)
4. [Error Handling](#error-handling)
5. [Security](#security)
6. [Performance](#performance)
7. [Collaboration](#collaboration)
8. [Production Deployment](#production-deployment)

---

## Configuration Management

### ✅ DO: Use Three-Tier Configuration

**Global Config** (`~/.primusrc`) - Personal preferences
```bash
# Your personal settings
PRIMUS_PATHS_PRIMUS_PATH="/home/you/workspace/Primus"
```

**Project Config** (`.primus.yaml`) - Team shared
```yaml
# Commit to git
container:
  image: "rocm/primus:v25.9_gfx942"
distributed:
  gpus_per_node: 8
```

**CLI Args** - One-off overrides
```bash
primus-cli --debug container --cpus 64 -- train
```

### ❌ DON'T: Hardcode Everything in Scripts

**Bad**:
```bash
#!/bin/bash
primus-cli container \
  --image rocm/primus:v25.9_gfx942 \
  --cpus 32 \
  --memory 256G \
  --mount /data:/data \
  --mount /output:/output \
  -- train pretrain --config exp.yaml
```

**Good**:
```yaml
# .primus.yaml
container:
  image: "rocm/primus:v25.9_gfx942"
  cpus: 32
  memory: "256G"
```
```bash
#!/bin/bash
primus-cli container -- train pretrain --config exp.yaml
```

### ✅ DO: Document Configuration Choices

```yaml
# .primus.yaml
# Project: GPT-3 Training
# Team: AI Research
# Last Updated: 2025-11-06

container:
  image: "rocm/primus:v25.9_gfx942"  # Stable release, validated
  cpus: 32                            # Optimal for 8 GPUs
  memory: "256G"                      # Required for large models

distributed:
  gpus_per_node: 8                    # MI300X nodes have 8 GPUs
  master_port: 1234                   # Avoid conflicts with other jobs
```

### ✅ DO: Version Control Project Configs

```bash
# Track changes to project configuration
git add .primus.yaml configs/
git commit -m "Update Primus config for MI325X nodes"

# Tag stable configurations
git tag -a "config-v1.0" -m "Validated configuration for production"
```

### ❌ DON'T: Commit Personal Configs

```bash
# .gitignore
.primusrc         # Personal config
*.local.yaml      # Local overrides
.env.local        # Local environment
```

---

## Workflow Organization

### ✅ DO: Use Standard Project Structure

```
project/
├── .primus.yaml           # Primus CLI config
├── configs/               # Training configs
│   ├── base.yaml         # Base configuration
│   ├── dev.yaml          # Development settings
│   ├── prod.yaml         # Production settings
│   └── experiments/      # Experimental configs
├── data/                  # Data directory
│   ├── raw/              # Raw data
│   └── processed/        # Preprocessed data
├── logs/                  # Training logs
├── output/                # Model outputs
│   ├── checkpoints/      # Model checkpoints
│   └── metrics/          # Training metrics
├── scripts/               # Utility scripts
│   ├── preprocess.sh     # Data preprocessing
│   └── evaluate.sh       # Model evaluation
└── README.md              # Project documentation
```

### ✅ DO: Create Named Configurations

```bash
# configs/dev.yaml - Fast iteration
model:
  layers: 12
  hidden_size: 768
training:
  max_steps: 1000
  eval_interval: 100

# configs/prod.yaml - Full training
model:
  layers: 96
  hidden_size: 12288
training:
  max_steps: 100000
  eval_interval: 1000
```

```bash
# Development
primus-cli --config dev.yaml direct -- train --config configs/dev.yaml

# Production
primus-cli --config prod.yaml slurm srun -N 32 -- train --config configs/prod.yaml
```

### ✅ DO: Use Hooks for Repetitive Tasks

```bash
# runner/hooks/train/pretrain/01-setup.sh
#!/bin/bash
echo "[Hook] Setting up training environment..."

# Create output directories
mkdir -p /output/checkpoints /output/metrics /logs

# Download tokenizer if needed
if [[ ! -f /data/tokenizer.model ]]; then
    wget -q https://example.com/tokenizer.model -O /data/tokenizer.model
fi

# Set distributed training env vars
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

### ❌ DON'T: Mix Environment Setup with Training Code

**Bad**: Put everything in training script
```python
# train.py
import os
os.makedirs("/output/checkpoints", exist_ok=True)
download_tokenizer()  # Should be in hook
setup_distributed()    # Should be in launcher
```

**Good**: Separate concerns
```bash
# Hooks handle setup
primus-cli direct -- train pretrain  # train.py just does training
```

---

## Resource Management

### ✅ DO: Set Appropriate Resource Limits

```yaml
# .primus.yaml
container:
  cpus: 32        # 4 CPUs per GPU (8 GPUs)
  memory: "256G"  # 32GB per GPU
  shm_size: "16G" # For large batches
```

### ✅ DO: Monitor Resource Usage

```bash
# During training
watch -n 1 'rocm-smi && echo "---" && docker stats'

# After training
docker stats --no-stream

# Slurm jobs
sstat -j <JOBID> --format=JobID,MaxRSS,MaxVMSize,AveCPU
```

### ❌ DON'T: Request More Resources Than Needed

**Bad**:
```yaml
container:
  cpus: 128      # Only using 32
  memory: "1T"   # Only need 256G
```

This wastes resources and may block other users.

### ✅ DO: Use Node-Local Storage for Temporary Files

```bash
# Use node's /tmp for temporary data
primus-cli container \
  --mount /tmp:/tmp \
  -- train pretrain --temp-dir /tmp/training
```

### ✅ DO: Clean Up After Jobs

```bash
# Create cleanup hook
# runner/hooks/train/pretrain/99-cleanup.sh
#!/bin/bash
echo "[Hook] Cleaning up temporary files..."
rm -rf /tmp/training-*
rm -rf /tmp/cache-*
```

---

## Error Handling

### ✅ DO: Use Debug Mode for Troubleshooting

```bash
# Enable debug mode (bash -x for verbose execution)
primus-cli --debug direct -- train pretrain --config exp.yaml

# Dry-run to see commands
primus-cli --dry-run container -- train pretrain

# Verbose Slurm output
primus-cli slurm srun -N 4 -v -- train pretrain
```

### ✅ DO: Capture Logs Properly

```bash
# Save logs with timestamps
LOG_DIR=logs/$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"

primus-cli container -- train pretrain --log-dir "$LOG_DIR" 2>&1 | tee "$LOG_DIR/output.log"
```

### ✅ DO: Check Prerequisites

```bash
# Create a preflight check script
# scripts/check_setup.sh
#!/bin/bash
set -e

echo "Checking Primus CLI setup..."

# Check CLI is installed
command -v primus-cli || { echo "ERROR: primus-cli not found"; exit 1; }

# Check configuration
primus-cli --show-config

# Check data exists
[[ -d /data ]] || { echo "ERROR: /data not found"; exit 1; }

# Check GPU
rocm-smi || { echo "ERROR: No AMD GPUs detected"; exit 1; }

echo "✅ All checks passed"
```

```bash
# Run before training
./scripts/check_setup.sh && primus-cli container -- train pretrain
```

### ❌ DON'T: Ignore Error Messages

```bash
# Bad: Ignore errors
primus-cli container -- train pretrain || true

# Good: Handle errors
if ! primus-cli container -- train pretrain; then
    echo "Training failed, check logs/"
    exit 1
fi
```

---

## Security

### ✅ DO: Use Read-Only Mounts When Possible

```bash
# Data should not be modified during training
primus-cli container \
  --mount /data:/data:ro \
  --mount /output:/output \
  -- train pretrain
```

### ✅ DO: Run Containers as Non-Root User

```yaml
# .primus.yaml
container:
  user: "1000:1000"  # Your UID:GID
```

Or:
```bash
primus-cli container --user $(id -u):$(id -g) -- train
```

### ❌ DON'T: Store Secrets in Configuration Files

**Bad**:
```yaml
# .primus.yaml - DO NOT DO THIS
paths:
  data_path: "s3://bucket?access_key=AKIAIOSFODNN7EXAMPLE"
```

**Good**: Use environment variables
```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
primus-cli container -- train pretrain
```

### ✅ DO: Limit Container Capabilities

The default `primus-cli-container.sh` uses minimal required capabilities. Don't add unnecessary privileges.

```bash
# Don't do this unless required
docker run --privileged  # ❌

# Use specific capabilities
docker run --cap-add=SYS_PTRACE  # ✅ (already in primus-cli)
```

---

## Performance

### ✅ DO: Use Configuration Files (Faster Startup)

**Slow**: Parse arguments every time
```bash
primus-cli container --image ... --cpus ... --memory ... -- train
```

**Fast**: Load from config once
```bash
primus-cli container -- train
```

### ✅ DO: Use Direct Mode for Single-Node

**Slower**: Container overhead (~100-200ms)
```bash
primus-cli container -- benchmark gemm
```

**Faster**: Native execution
```bash
primus-cli direct -- benchmark gemm
```

### ❌ DON'T: Use Debug Mode in Production

```bash
# Development (with debug)
primus-cli --debug direct -- train

# Production (without debug)
primus-cli direct -- train
```

---

## Collaboration

### ✅ DO: Document Team Workflows

```markdown
# README.md

## Training Workflow

### Prerequisites
```bash
# Install Primus CLI
export PATH="/shared/tools/Primus-CLI/runner:$PATH"
```

### Run Training
```bash
# 1. Activate environment
source activate primus

# 2. Run training
primus-cli slurm srun -N 4 -p gpu -- train pretrain --config configs/base.yaml
```

### Monitor Progress
```bash
# Check job status
squeue -u $USER

# View logs
tail -f slurm-<JOBID>.out
```
```

### ✅ DO: Share Project Configuration

```yaml
# .primus.yaml (committed to git)
# Shared by entire team

distributed:
  gpus_per_node: 8

slurm:
  partition: "gpu"
  time_limit: "24:00:00"

paths:
  data_path: "/shared/data/gpt3"
  output_path: "/shared/output/$USER"
```

### ✅ DO: Use Consistent Naming

```bash
# Job names
primus-cli slurm sbatch --job-name "${USER}-gpt3-$(date +%Y%m%d)" -- train

# Container names
primus-cli container --name "${USER}-dev" -- train

# Log directories
LOG_DIR="logs/${USER}/$(date +%Y%m%d_%H%M%S)"
```

### ✅ DO: Create Team Templates

```bash
# templates/train_template.sh
#!/bin/bash
# Template for team training jobs

set -euo pipefail

# Configuration
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-"my-experiment"}

# Run training
primus-cli slurm srun \
    -N ${NNODES:-4} \
    --job-name "${USER}-${EXPERIMENT_NAME}" \
    -- train pretrain --config configs/${EXPERIMENT_NAME}.yaml
```

---

## Production Deployment

### ✅ DO: Test in Stages

```bash
# Stage 1: Local testing
primus-cli direct -- train pretrain --config test_tiny.yaml

# Stage 2: Single-node validation
primus-cli container -- train pretrain --config test_small.yaml

# Stage 3: Small-scale distributed
primus-cli slurm srun -N 2 -- train pretrain --config test_medium.yaml

# Stage 4: Full production
primus-cli slurm sbatch -N 32 -- train pretrain --config production.yaml
```

### ✅ DO: Use Checkpointing

```yaml
# configs/prod.yaml
training:
  checkpoint_interval: 1000
  checkpoint_dir: "/output/checkpoints"
  resume_from_checkpoint: true
```

### ✅ DO: Monitor Jobs

```bash
# Create monitoring script
# scripts/monitor.sh
#!/bin/bash

JOBID=$1

while true; do
    clear
    echo "=== Job $JOBID Status ==="
    squeue -j "$JOBID"
    echo ""
    echo "=== Resource Usage ==="
    sstat -j "$JOBID" --format=JobID,MaxRSS,AveCPU
    echo ""
    echo "=== Recent Logs ==="
    tail -20 "slurm-${JOBID}.out"
    sleep 30
done
```

```bash
# Launch and monitor
JOBID=$(primus-cli slurm sbatch -N 32 -- train pretrain | grep -oP '\d+')
./scripts/monitor.sh "$JOBID"
```

### ✅ DO: Plan for Failures

```bash
# Wrapper with error handling
#!/bin/bash
set -eo pipefail

MAX_RETRIES=3
RETRY=0

while [ $RETRY -lt $MAX_RETRIES ]; do
    if primus-cli slurm srun -N 32 -- train pretrain --resume; then
        echo "Training completed successfully"
        exit 0
    else
        RETRY=$((RETRY + 1))
        echo "Training failed, retry $RETRY/$MAX_RETRIES"
        sleep 60
    fi
done

echo "Training failed after $MAX_RETRIES attempts"
exit 1
```

### ✅ DO: Archive Results

```bash
# After training
#!/bin/bash
EXPERIMENT=gpt3-baseline
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_DIR=/shared/archives/${EXPERIMENT}_${TIMESTAMP}

mkdir -p "$ARCHIVE_DIR"

# Archive outputs
cp -r output/checkpoints "$ARCHIVE_DIR/"
cp -r logs "$ARCHIVE_DIR/"
cp configs/*.yaml "$ARCHIVE_DIR/"
cp .primus.yaml "$ARCHIVE_DIR/"

# Create metadata
cat > "$ARCHIVE_DIR/metadata.txt" << EOF
Experiment: $EXPERIMENT
Date: $TIMESTAMP
User: $USER
Nodes: 32
GPUs: 256
Config: $(cat .primus.yaml | base64)
EOF

echo "Archived to: $ARCHIVE_DIR"
```

---

## Summary Checklist

### Configuration
- [ ] Use `.primus.yaml` for project settings
- [ ] Use `~/.primusrc` for personal settings
- [ ] Document configuration choices
- [ ] Version control project configs
- [ ] Don't commit personal configs

### Organization
- [ ] Use standard project structure
- [ ] Create named configurations (dev/prod)
- [ ] Use hooks for repetitive setup
- [ ] Separate concerns (setup vs training)

### Resources
- [ ] Set appropriate resource limits
- [ ] Monitor resource usage
- [ ] Don't request more than needed
- [ ] Clean up temporary files

### Error Handling
- [ ] Use debug mode for troubleshooting
- [ ] Capture logs properly
- [ ] Check prerequisites before running
- [ ] Handle errors gracefully

### Security
- [ ] Use read-only mounts when possible
- [ ] Run as non-root user
- [ ] Don't store secrets in configs
- [ ] Use minimal container capabilities

### Performance
- [ ] Use configuration files
- [ ] Use direct mode for single-node
- [ ] Disable unnecessary logging in production
- [ ] No debug mode in production

### Collaboration
- [ ] Document workflows
- [ ] Share project configuration
- [ ] Use consistent naming
- [ ] Create team templates

### Production
- [ ] Test in stages
- [ ] Use checkpointing
- [ ] Monitor jobs
- [ ] Plan for failures
- [ ] Archive results

---

**Version**: 1.2.0
**Last Updated**: November 6, 2025
