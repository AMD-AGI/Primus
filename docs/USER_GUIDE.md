# Primus CLI User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Usage Modes](#usage-modes)
5. [Configuration](#configuration)
6. [Common Workflows](#common-workflows)
7. [Advanced Features](#advanced-features)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

---

## Introduction

**Primus CLI** is a unified launcher for distributed AI training and benchmarking on AMD GPUs. It supports three execution modes:

- **Slurm Mode**: Distributed training across HPC clusters
- **Container Mode**: Isolated training in Docker/Podman containers
- **Direct Mode**: Native execution on the host or inside existing containers

### Key Features

✅ **Unified Interface** - One CLI for all workflows
✅ **Configuration Files** - Save default settings
✅ **GPU Optimization** - Automatic AMD GPU tuning
✅ **Distributed Training** - Multi-node support via Slurm
✅ **Container Management** - Automated Docker/Podman workflows
✅ **Extensible** - Hooks and patches for customization

---

## Installation

### Prerequisites

- **OS**: Linux (tested on Ubuntu 20.04+, RHEL 8+)
- **Shell**: Bash 4.0+
- **Optional**: Docker/Podman, Slurm

### Install Primus CLI

```bash
# Clone the repository
git clone https://github.com/your-org/Primus-CLI.git
cd Primus-CLI

# Add to PATH
export PATH="$PWD/runner:$PATH"

# Or create symlink
sudo ln -s "$PWD/runner/primus-cli" /usr/local/bin/primus-cli

# Verify installation
primus-cli --version
```

### Verify Setup

```bash
# Check CLI works
primus-cli --help

# Check configuration
primus-cli --show-config
```

---

## Quick Start

### 1. Direct Mode (Simplest)

Run training directly on the host:

```bash
primus-cli direct -- train pretrain --config my_config.yaml
```

### 2. Container Mode (Isolated)

Run training in a container:

```bash
primus-cli container -- benchmark gemm -M 4096 -N 4096 -K 4096
```

### 3. Slurm Mode (Distributed)

Run distributed training on 4 nodes:

```bash
primus-cli slurm srun -N 4 -- train pretrain --config my_config.yaml
```

---

## Usage Modes

### Direct Mode

**When to use**: Local development, debugging, single-node training

**Syntax**:
```bash
primus-cli direct [OPTIONS] -- [TRAINING_ARGS]
```

**Examples**:
```bash
# Basic training
primus-cli direct -- train pretrain --config exp.yaml

# With custom data path
primus-cli direct --data-path /data -- train pretrain --config exp.yaml

# Benchmark GEMM
primus-cli direct -- benchmark gemm -M 8192 -N 8192 -K 8192
```

**Options**:
- `--data-path PATH` - Data directory
- `--output-path PATH` - Output directory
- `--debug` - Enable debug mode (verbose bash execution)

---

### Container Mode

**When to use**: Reproducible environments, isolation, testing

**Syntax**:
```bash
primus-cli container [OPTIONS] -- [TRAINING_ARGS]
```

**Examples**:
```bash
# Basic container run
primus-cli container -- train pretrain --config exp.yaml

# Custom image
primus-cli container --image rocm/primus:v25.9_gfx942 -- train pretrain

# Mount data directory
primus-cli container --mount /host/data:/data -- train pretrain --data-path /data

# Resource limits
primus-cli container --cpus 32 --memory 256G --gpus 8 -- train pretrain

# Use local Primus code
primus-cli container --primus-path ~/workspace/Primus -- train pretrain
```

**Options**:
- `--image IMAGE` - Docker image to use
- `--mount HOST[:CONTAINER]` - Mount directories (repeatable)
- `--primus-path PATH` - Use local Primus source
- `--cpus N` - CPU limit
- `--memory SIZE` - Memory limit (e.g., 128G)
- `--shm-size SIZE` - Shared memory size
- `--gpus N` - GPU count
- `--user UID:GID` - Run as specific user
- `--name NAME` - Container name
- `--clean` - Remove containers before launch

---

### Slurm Mode

**When to use**: Multi-node distributed training on HPC clusters

**Syntax**:
```bash
primus-cli slurm <srun|sbatch> [SLURM_ARGS] -- [TRAINING_ARGS]
```

**Examples**:
```bash
# Interactive 4-node training (srun)
primus-cli slurm srun -N 4 -p gpu -- train pretrain --config exp.yaml

# Batch job (sbatch)
primus-cli slurm sbatch -N 8 -p AIG_Model --time 24:00:00 -- train pretrain

# With custom partition
primus-cli slurm srun -N 2 -p debug -- benchmark gemm

# Override GPUs per node
primus-cli slurm srun -N 4 --gpus-per-node 4 -- train pretrain
```

**Common Slurm Arguments**:
- `-N, --nodes N` - Number of nodes
- `-p, --partition NAME` - Partition name
- `--time HH:MM:SS` - Time limit
- `--gpus-per-node N` - GPUs per node
- `--job-name NAME` - Job name

---

## Configuration

### Configuration Files

Primus CLI supports three configuration levels:

1. **Global**: `~/.primusrc` (personal defaults)
2. **Project**: `.primus.yaml` (team settings)
3. **CLI**: Command-line arguments (overrides)

**Priority**: CLI > Project > Global > Defaults

### Example: Global Configuration

**File**: `~/.primusrc`

```bash
# Personal preferences
PRIMUS_PATHS_PRIMUS_PATH="/home/myuser/workspace/Primus"
```

### Example: Project Configuration

**File**: `.primus.yaml` (in project root)

```yaml
# Team shared settings
distributed:
  gpus_per_node: 8
  master_port: 1234

container:
  image: "rocm/primus:v25.9_gfx942"
  cpus: 32
  memory: "256G"

paths:
  log_path: "logs"
  data_path: "/data"
```

### View Current Configuration

```bash
primus-cli --show-config
```

### Load Custom Configuration

```bash
primus-cli --config my-config.yaml container -- train
```

See [Configuration Guide](CONFIGURATION_GUIDE.md) for details.

---

## Common Workflows

### Workflow 1: Single-Node Training

```bash
# 1. Prepare configuration
cat > exp.yaml << EOF
model:
  type: llama
  hidden_size: 4096
training:
  batch_size: 32
  learning_rate: 0.0001
EOF

# 2. Run training
primus-cli direct -- train pretrain --config exp.yaml --data-path /data
```

### Workflow 2: Multi-Node Distributed Training

```bash
# 1. Create project config
cat > .primus.yaml << EOF
distributed:
  gpus_per_node: 8
slurm:
  partition: "gpu"
EOF

# 2. Launch 4-node training
primus-cli slurm srun -N 4 -- train pretrain --config exp.yaml
```

### Workflow 3: Container-Based Development

```bash
# 1. Set up container config
cat > .primus.yaml << EOF
container:
  image: "rocm/primus:dev"
  cpus: 16
  memory: "128G"
EOF

# 2. Mount your code for development
primus-cli container \
  --primus-path ~/workspace/Primus \
  --mount ~/data:/data \
  -- train pretrain --config exp.yaml
```

### Workflow 4: Benchmarking

```bash
# GEMM benchmark
primus-cli direct -- benchmark gemm -M 8192 -N 8192 -K 8192

# Attention benchmark
primus-cli direct -- benchmark attention --seq-len 2048 --heads 32

# Multi-node benchmark
primus-cli slurm srun -N 4 -- benchmark all-reduce --size 1G
```

### Workflow 5: Debug Mode

```bash
# Enable debug mode (bash -x for verbose execution)
primus-cli --debug direct -- train pretrain --config exp.yaml

# Dry-run mode (see commands without execution)
primus-cli --dry-run container -- train pretrain --config exp.yaml
```

---

## Advanced Features

### Hooks

Hooks allow custom scripts to run before training.

**Location**: `runner/hooks/<command>/<subcommand>/`

**Example**: Create a pre-training hook

```bash
# Create hook directory
mkdir -p runner/hooks/train/pretrain/

# Create hook script
cat > runner/hooks/train/pretrain/01-download-data.sh << 'EOF'
#!/bin/bash
echo "Downloading training data..."
wget -q https://example.com/data.tar.gz -O /tmp/data.tar.gz
tar xzf /tmp/data.tar.gz -C /data/
EOF

chmod +x runner/hooks/train/pretrain/01-download-data.sh

# Run training (hook executes automatically)
primus-cli direct -- train pretrain --config exp.yaml
```

### Patches

Patches modify the environment before execution.

**Example**: Apply custom optimization patch

```bash
primus-cli direct --patch my-optimization.sh -- train pretrain
```

### Environment Variables

Control behavior via environment variables:

```bash
# Set GPU type
export GPU_TYPE=MI300X

# Disable XNACK
export HSA_XNACK=0

# Run with custom env
primus-cli direct -- train pretrain
```

### Custom Primus Path

Use your development version of Primus:

```bash
# In container
primus-cli container --primus-path ~/workspace/Primus -- train pretrain

# Direct mode
export PRIMUS_PATH=~/workspace/Primus
primus-cli direct -- train pretrain
```

---

## Troubleshooting

### Problem: "Command not found: primus-cli"

**Solution**:
```bash
# Add to PATH
export PATH="/path/to/Primus-CLI/runner:$PATH"

# Or use full path
/path/to/Primus-CLI/runner/primus-cli --version
```

### Problem: Configuration not loading

**Solution**:
```bash
# Check config file exists
ls -la ~/.primusrc .primus.yaml

# View what's loaded
primus-cli --show-config

# Load specific config
primus-cli --config my-config.yaml --show-config
```

### Problem: Container fails to start

**Solution**:
```bash
# Check Docker is running
docker ps

# Pull image manually
docker pull rocm/primus:v25.9_gfx942

# Check permissions
docker run --rm rocm/primus:v25.9_gfx942 whoami
```

### Problem: Slurm job fails

**Solution**:
```bash
# Check Slurm status
sinfo

# View job status
squeue -u $USER

# Check job output
cat slurm-<JOBID>.out

# Enable debug mode
primus-cli --debug slurm srun -N 4 -- train pretrain
```

### Problem: Training hangs

**Solution**:
```bash
# Check GPU status
rocm-smi

# Check processes
ps aux | grep python

# Enable debug mode
primus-cli --debug direct -- train pretrain

# Check for deadlocks (distributed)
export NCCL_DEBUG=INFO
primus-cli direct -- train pretrain
```

### Getting Help

```bash
# General help
primus-cli --help

# Mode-specific help
primus-cli slurm --help
primus-cli container --help
primus-cli direct --help

# Show configuration
primus-cli --show-config

# Enable debug mode
primus-cli --debug <mode> -- <args>
```

---

## Best Practices

### 1. Use Configuration Files

**❌ Bad**: Repeat arguments every time
```bash
primus-cli container --image rocm/primus:v25.9 --cpus 32 --memory 256G -- train
```

**✅ Good**: Use project config
```yaml
# .primus.yaml
container:
  image: "rocm/primus:v25.9"
  cpus: 32
  memory: "256G"
```
```bash
primus-cli container -- train
```

### 2. Version Control Your Configs

```bash
# Commit project config
git add .primus.yaml
git commit -m "Add Primus CLI config"

# Don't commit personal config
echo ".primusrc" >> .gitignore
```

### 3. Use Descriptive Job Names

```bash
# Slurm jobs
primus-cli slurm sbatch -N 8 --job-name "gpt3-pretrain-8nodes" -- train pretrain

# Container names
primus-cli container --name "myuser-dev-container" -- train
```

### 4. Mount Data Read-Only When Possible

```bash
# Prevent accidental modifications
primus-cli container --mount /data:/data:ro -- train pretrain --data-path /data
```

### 5. Use Debug Mode for Development

```bash
# Development
primus-cli --debug --dry-run direct -- train pretrain

# Production
primus-cli direct -- train pretrain
```

### 6. Organize Outputs

```bash
# Project structure
project/
  ├── .primus.yaml        # Project config
  ├── configs/            # Training configs
  │   ├── exp1.yaml
  │   └── exp2.yaml
  ├── data/               # Data directory
  ├── logs/               # Training logs
  └── output/             # Model checkpoints

# Run with organized paths
primus-cli container \
  --mount $PWD/data:/data \
  --mount $PWD/logs:/logs \
  --mount $PWD/output:/output \
  -- train pretrain --config configs/exp1.yaml
```

### 7. Use Hooks for Repetitive Setup

```bash
# Instead of manual steps every time
# Create hooks/train/pretrain/01-setup.sh
#!/bin/bash
mkdir -p /output/checkpoints
export TORCH_DISTRIBUTED_DEBUG=INFO
```

### 8. Test with Small Configs First

```bash
# 1. Test with minimal config
primus-cli direct -- train pretrain --config test_small.yaml

# 2. Scale up gradually
primus-cli slurm srun -N 1 -- train pretrain --config test_1node.yaml

# 3. Full scale
primus-cli slurm srun -N 32 -- train pretrain --config production.yaml
```

### 9. Monitor Resource Usage

```bash
# Watch GPU usage
watch -n 1 rocm-smi

# Monitor container resources
docker stats

# Slurm job stats
sstat -j <JOBID>
```

### 10. Document Your Workflows

```bash
# Add README to your project
cat > README.md << 'EOF'
# My Training Project

## Setup
```bash
cp examples/primus.yaml.example .primus.yaml
# Edit .primus.yaml with your settings
```

## Run Training
```bash
primus-cli slurm srun -N 4 -- train pretrain --config configs/gpt.yaml
```
EOF
```

---

## Next Steps

- **Configuration**: Read [Configuration Guide](CONFIGURATION_GUIDE.md)
- **Reference**: See [Quick Reference](../runner/QUICK_REFERENCE.md)
- **Examples**: Check [examples/](../examples/) directory
- **Testing**: Review [Testing Guide](../tests/cli/TESTING_GUIDE.md)

---

## Support

- **Documentation**: `docs/` directory
- **Examples**: `examples/` directory
- **Issues**: GitHub Issues
- **Community**: [Your community channel]

---

**Version**: 1.2.0
**Last Updated**: November 6, 2025
**License**: [Your License]
