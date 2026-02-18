# Primus CLI User Guide

> Unified Command-Line Interface for AMD GPU Large Model Workflows

---

## 📋 Table of Contents

- [Quick Start](#quick-start) - 5-minute onboarding guide
- [Execution Modes](#execution-modes) - Direct / Container / Slurm
- [Configuration Files](#configuration-files) - YAML configuration explained
- [Usage Examples](#usage-examples) - Real-world case collection
- [Global Options](#global-options) - Common command parameters
- [Complete Call Logic](#complete-call-logic) - Internal execution flow (Advanced)
- [Best Practices](#best-practices) - Recommended workflows
- [Troubleshooting](#troubleshooting) - Common issue resolution

---

## Quick Start

### Basic Syntax

```bash
./primus-cli [global-options] <mode> [mode-args] -- [Primus commands and args]
```

### Core Concepts

- **Global Options**: Options applicable to all modes (e.g., `--debug`, `--config`)
- **Mode**: Execution environment (`slurm` / `container` / `direct`)
- **Separator `--`**: Required, used to separate mode parameters from Primus commands
- **Primus Commands**: Python CLI commands (`train`, `benchmark`, `preflight`, etc.)

### Your First Command

```bash
# Run GEMM benchmark directly on current host
./primus-cli direct -- benchmark gemm --M 4096 --N 4096 --K 4096
```

---

## Execution Modes

Primus CLI supports three execution modes, each suitable for different scenarios.

### 1. 🖥️ Direct Mode

**Use Case**: Execute directly on current host or within an existing container

**Features**:
- Simplest execution method
- Suitable for single-node training or debugging
- Runs directly in current environment with no extra overhead

**Syntax**:
```bash
./primus-cli direct [options] -- <Primus-command>
```

**Examples**:
```bash
# Basic training
./primus-cli direct -- train pretrain --config config.yaml

# GEMM benchmark
./primus-cli direct -- benchmark gemm --M 4096 --N 4096 --K 4096

# Environment check (info only)
./primus-cli direct -- preflight --host --gpu --network
```

**Suitable for**:
- ✅ Local development and debugging
- ✅ Single-node training
- ✅ Quick experiments
- ✅ Running within existing containers

---

### 2. 🐳 Container Mode

**Use Case**: Execute in Docker/Podman containers

**Features**:
- Provides isolated runtime environment
- Auto-mounts necessary devices and directories
- Supports custom images and resource limits
- Suitable for tasks requiring specific environments

**Syntax**:
```bash
./primus-cli container [container-options] -- <Primus-command>
```

**Common Options**:
| Option | Description | Example |
|--------|-------------|---------|
| `--image IMAGE` | Specify container image | `--image rocm/primus:v26.1` |
| `--volume PATH[:PATH]` | Mount directory | `--volume /data:/data` |
| `--cpus N` | Limit CPU cores | `--cpus 16` |
| `--memory SIZE` | Limit memory size | `--memory 128G` |
| `--clean` | Clean all containers before starting | `--clean` |

**Examples**:
```bash
# Run training with default image
./primus-cli container -- train pretrain --config config.yaml

# Specify image and mount data directory
./primus-cli container --image rocm/primus:latest \
  --volume /mnt/data:/data \
  -- train pretrain --config /data/config.yaml

# Set resource limits
./primus-cli container --cpus 32 --memory 256G \
  -- benchmark gemm --M 8192 --N 8192 --K 8192

# Mount local Primus code for development
./primus-cli container --volume ~/workspace/Primus:/workspace/Primus \
  -- train pretrain
```

**Suitable for**:
- ✅ Requiring specific dependency environments
- ✅ Environment isolation and reproducibility
- ✅ Developing and testing different versions
- ✅ CI/CD pipelines

---

### 3. 🖧 Slurm Mode

**Use Case**: Execute distributed tasks on Slurm clusters

**Features**:
- Supports multi-node distributed training
- Auto-handles node allocation and task scheduling
- Supports `srun` (interactive) and `sbatch` (batch)
- Full Slurm parameter support

**Syntax**:
```bash
./primus-cli slurm [srun|sbatch] [Slurm-params] -- <Primus-command>
```

**Common Slurm Parameters**:
| Parameter | Short | Description | Example |
|-----------|-------|-------------|---------|
| `--nodes` | `-N` | Number of nodes | `-N 4` |
| `--partition` | `-p` | Partition | `-p gpu` |
| `--time` | `-t` | Time limit | `-t 4:00:00` |
| `--output` | `-o` | Output log file | `-o job.log` |
| `--job-name` | `-J` | Job name | `-J train_job` |

**Examples**:
```bash
# Run training on 4 nodes using srun (interactive)
./primus-cli slurm srun -N 4 -p gpu -- train pretrain --config config.yaml

# Submit batch job using sbatch
./primus-cli slurm sbatch -N 8 -p AIG_Model -t 8:00:00 -o train.log \
  -- train pretrain --config deepseek_v2.yaml

# Run distributed GEMM benchmark
./primus-cli slurm srun -N 2 -- benchmark gemm --M 16384 --N 16384 --K 16384

# Multi-node environment check (info only)
# this will generate a fast info report of the host, GPU, and network
./primus-cli slurm srun -N 4 -- preflight --host --gpu --network

# this will generate a full preflight report of the host, GPU, and network, as well as the performance tests
./primus-cli slurm srun -N 4 -- preflight --report-file-name preflight-report-4N

# if you are using AINIC in your cluster, use the appropriate configuration file
# for preflight test, set docker image to rocm/primus:v26.1 in the configuration file
./primus-cli --config runner/use_ainic.yaml slurm srun -N 2 -- preflight --report-file-name preflight-report-2N
```

**Suitable for**:
- ✅ Multi-node distributed training
- ✅ Large-scale model training
- ✅ Requiring job scheduling and resource management
- ✅ Production environment training tasks

---

## Configuration Files

Primus CLI supports YAML format configuration files to preset various options.

### Configuration File Locations

Configuration files are loaded in the following priority order:

1. **Command-line specified**: `--config /path/to/config.yaml` (Highest priority)
2. **System default**: `runner/.primus.yaml`
3. **User config**: `~/.primus.yaml` (Lowest priority)

### Configuration File Structure

```yaml
# Global settings
main:
  debug: false
  dry_run: false

# Slurm configuration
slurm:
  nodes: 2
  time: "4:00:00"
  partition: "gpu"
  gpus_per_node: 8

# Container configuration
container:
  image: "rocm/primus:v26.1"
  options:
    cpus: "32"
    memory: "256G"
    ipc: "host"
    network: "host"

    # GPU devices (do not modify)
    devices:
      - "/dev/kfd"
      - "/dev/dri"
      - "/dev/infiniband"

    # Permissions
    capabilities:
      - "SYS_PTRACE"
      - "CAP_SYS_ADMIN"

    # Volume mounts
    volume:
      - "/data:/data"
      - "/model_weights:/weights:ro"

# Direct mode configuration
direct:
  gpus_per_node: 8
  master_port: 1234
  numa: "auto"
```

### Using Configuration Files

```bash
# Use project config file
./primus-cli --config .primus.yaml slurm srun -N 4 -- train pretrain

# Use custom user config
./primus-cli --config ~/my-config.yaml container -- benchmark gemm

# Config file + command-line args (command-line has higher priority)
./primus-cli --config prod.yaml slurm srun -N 8 -- train pretrain
```

### Using AINIC Configuration File

If you are using AINIC in your cluster, you can use `runner/use_ainic.yaml` configuration file to configure the AINIC environment. In this file, we have already set the environment variables for AINIC: `USING_AINIC=1`, `NCCL_PXN_DISABLE=0`, and `NCCL_IB_GID_INDEX=1`. You can modify the `NCCL_IB_GID_INDEX` value based on your AINIC settings. Also, you can modify the `image` value to the appropriate Docker image you are using. 

Here is an example of using the AINIC configuration file to run a training job: 
```bash
./primus-cli --config runner/use_ainic.yaml slurm srun -N 2 -- train pretrain --config examples/maxtext/configs/MI355X/llama2_7B-pretrain.yaml
```

### Configuration Priority

**Priority Order** (high to low):
```
Command-line args > Specified config file > System default config > User config
```

**Example**:
```bash
# Config file sets nodes=2, command-line specifies -N 4
# Final result uses 4 nodes (command-line takes priority)
./primus-cli --config .primus.yaml slurm srun -N 4 -- train pretrain
```

---

## Usage Examples

### Training Tasks

#### Single-Node Training (Direct)
```bash
# Basic training
./primus-cli direct -- train pretrain --config config.yaml

# With debug mode
./primus-cli --debug direct -- train pretrain --config config.yaml
```

#### Single-Node Training (Container)
```bash
# Run in container
./primus-cli container --volume /data:/data \
  -- train pretrain --config /data/config.yaml

# Custom resource limits
./primus-cli container --cpus 64 --memory 512G \
  -- train pretrain --config config.yaml
```

#### Multi-Node Training (Slurm)
```bash
# 4-node distributed training
./primus-cli slurm srun -N 4 -p gpu -- train pretrain --config config.yaml

# Submit batch job
./primus-cli slurm sbatch -N 8 -p AIG_Model -t 12:00:00 \
  -o train_%j.log -e train_%j.err \
  -- train pretrain --config deepseek_v2.yaml
```

### Benchmark Tasks

#### GEMM Benchmark
```bash
# Single-node GEMM
./primus-cli direct -- benchmark gemm --M 4096 --N 4096 --K 4096

# Run in container
./primus-cli container -- benchmark gemm --M 8192 --N 8192 --K 8192

# Multi-node GEMM
./primus-cli slurm srun -N 2 -- benchmark gemm --M 16384 --N 16384 --K 16384
```

#### Other Benchmarks
```bash
# All-reduce benchmark
./primus-cli slurm srun -N 4 -- benchmark allreduce --size 1GB

# End-to-end training performance
./primus-cli slurm srun -N 8 -- benchmark e2e --model llama2-7b
```

### Environment Check (Preflight)

```bash
# GPU info only
./primus-cli direct -- preflight --gpu

# Network info only
./primus-cli slurm srun -N 4 -- preflight --network

# Info only (host + GPU + network)
./primus-cli slurm srun -N 4 -- preflight --host --gpu --network

# Full preflight (all info + perf tests)
./primus-cli slurm srun -N 4 -- preflight

# Perf tests only
./primus-cli slurm srun -N 4 -- preflight --perf-test
```

> Tip: `preflight --host --gpu --network` is intended to be fast (info collection only).
> For full diagnostics (info + compute + intra/inter-node comm benchmarks), run `./primus-cli preflight`
> or use `--perf-test` for benchmarks only.

### Combined Usage

```bash
# Config file + debug mode + dry-run
./primus-cli --config prod.yaml --debug --dry-run \
  slurm srun -N 4 -- train pretrain

# Container + custom image + multiple mount points
./primus-cli container \
  --image rocm/primus:dev \
  --volume /data:/data \
  --volume /models:/models:ro \
  --volume /output:/output \
  -- train pretrain --config /data/config.yaml

# Slurm + config file + resource limits
./primus-cli --config cluster.yaml slurm sbatch \
  -N 16 -p bigmem --exclusive \
  -- train pretrain --config llama3-70b.yaml
```

---

## Global Options

Global options apply to all execution modes (Direct, Container, Slurm) and are specified before the mode name.

### Available Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config FILE` | Specify config file path | `runner/.primus.yaml` |
| `--debug` | Enable debug mode (verbose logging) | Off |
| `--dry-run` | Show command that would be executed without running | Off |
| `--version` | Show version info and exit | - |
| `-h, --help` | Show help info and exit | - |

### Detailed Description

#### `--config FILE`
Specify a custom configuration file, overriding the default config.

```bash
# Use production environment config
./primus-cli --config configs/prod.yaml slurm srun -N 4 -- train pretrain

# Use relative path
./primus-cli --config ./my-config.yaml direct -- benchmark gemm

# Use absolute path
./primus-cli --config /shared/configs/cluster.yaml container -- train pretrain
```

**Config Priority**: `--config` specified file > System default `runner/.primus.yaml` > User config `~/.primus.yaml`

#### `--debug`
Enable debug mode, output detailed execution logs including:
- Configuration loading process
- Environment variable setup
- Command building steps
- Internal function calls

```bash
# Debug Slurm job
./primus-cli --debug slurm srun -N 2 -- train pretrain --config config.yaml

# Debug container startup
./primus-cli --debug container --image rocm/primus:dev -- benchmark gemm

# Debug config loading
./primus-cli --debug --config test.yaml direct -- preflight --gpu
```

**Environment Variable**: `--debug` sets `PRIMUS_LOG_LEVEL=DEBUG`

#### `--dry-run`
Dry-run mode, shows the complete execution command without actually running it. Useful for:
- Verifying config correctness
- Viewing final command
- Debugging parameter passing
- CI/CD pipeline testing

```bash
# View how Slurm job would be submitted
./primus-cli --dry-run slurm sbatch -N 8 -p gpu -- train pretrain

# View container startup command
./primus-cli --dry-run container --volume /data:/data -- benchmark gemm

# View distributed training command
./primus-cli --dry-run direct -- train pretrain --config config.yaml
```

**Output Format**:
```
==========================================
  [DRY RUN] Slurm Configuration
==========================================
Launch Command: srun
SLURM Flags:
  -N 8
  -p gpu
  -t 4:00:00
Entry Script: primus-cli-slurm-entry.sh
==========================================
```

#### `--version`
Show Primus CLI version info and exit.

```bash
./primus-cli --version
# Output: Primus CLI v1.0.0
```

#### `-h, --help`
Show help info, can be used at different levels:

```bash
# Main entry help
./primus-cli --help

# Mode-specific help
./primus-cli direct --help
./primus-cli container --help
./primus-cli slurm --help

# Primus Python CLI help
./primus-cli direct -- --help
./primus-cli direct -- train --help
./primus-cli direct -- benchmark --help
```

### Combined Usage Examples

#### Debug + Config File
```bash
./primus-cli --config dev.yaml --debug direct -- train pretrain
```

#### Dry-run + Custom Config
```bash
./primus-cli --config prod.yaml --dry-run slurm srun -N 4 -- train pretrain
```

#### Multi-level Debugging
```bash
# View complete command building and execution process
./primus-cli --debug --dry-run slurm sbatch -N 8 -- container --debug -- train pretrain
```

### Global Options Scope

```
./primus-cli [global-options] <mode> [mode-args] -- [Primus commands]
           ↑
           └─ Affects entire execution flow
              • Main entry (./primus-cli)
              • Mode scripts (primus-cli-*.sh)
              • Final execution (primus-cli-direct.sh)
```

**Notes**:
- Global options must be specified before the mode name
- `--debug` is passed to all subscripts
- `--dry-run` intercepts execution at the mode script level
- `--config` configuration takes effect in all stages

---

## Complete Call Logic

> 📌 **Tip**: This section is for advanced users, explaining Primus CLI's internal execution flow in detail. If you're a beginner, you can skip to [Best Practices](#best-practices).

### Execution Flow Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│   User Command                                                         │
│   ./primus-cli [global-options] <mode> [mode-args] -- [Primus-cmd]       │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │
                                  ↓
┌────────────────────────────────────────────────────────────────────────┐
│   1. primus-cli (Main Entry)                                           │
│   • Parse global options (--config, --debug, --dry-run)                │
│   • Load config files (.primus.yaml)                                   │
│   • Extract main.* configuration                                       │
│   • Set debug mode and log level                                       │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │
               ┌──────────────────┼──────────────────┐
               │                  │                  │
               ↓                  ↓                  ↓
      ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
      │    Direct    │   │  Container   │   │    Slurm     │
      └───────┬──────┘   └───────┬──────┘   └───────┬──────┘
              │                  │                  │
              │                  │                  │
              ↓                  ↓                  ↓
┌────────────────────────────────────────────────────────────────────────┐
│   2. Mode-Specific Scripts (primus-cli-*.sh)                           │
│   • Load container/slurm/direct config                                 │
│   • Parse mode-specific parameters                                     │
│   • Prepare execution environment                                      │
│     - Slurm: Build srun/sbatch command                                 │
│     - Container: Start container                                       │
│     - Direct: Load GPU environment                                     │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │
                                  ↓
┌────────────────────────────────────────────────────────────────────────┐
│   3. primus-cli-direct.sh (Final Exec Layer)                           │
│   • Load environment (primus-env.sh)                                   │
│     - base_env.sh (base environment)                                   │
│     - detect_gpu.sh (GPU detection)                                    │
│     - GPU-specific env (MI300X.sh, etc.)                               │
│   • Execute Hooks (execute_hooks.sh)                                   │
│   • Apply Patches (execute_patches.sh)                                 │
│   • Set distributed environment variables                              │
│     - MASTER_ADDR, NODE_RANK                                           │
│     - NCCL/RCCL configuration                                          │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │
                                  ↓
┌────────────────────────────────────────────────────────────────────────┐
│   4. Primus Python CLI (primus/cli/main.py)                            │
│   • Parse Primus commands (train/benchmark)                            │
│   • Load subcommand plugins                                            │
│   • Execute specific tasks                                             │
│     - train: Start training (Megatron/etc.)                            │
│     - benchmark: Performance testing                                   │
│     - preflight: Environment check                                     │
└────────────────────────────────────────────────────────────────────────┘
```

### Example Call Breakdown

A complete Slurm multi-node containerized training command:

```bash
./primus-cli --config prod.yaml --debug \
  slurm srun -N 4 -- container --image rocm/megatron-lm:v25.8_py310 \
  -- train pretrain --config deepseek_v2.yaml
```

**Execution Steps Explained**:

```
Step 1: ./primus-cli main entry
  ├─ Parse global options: --config prod.yaml, --debug
  ├─ Load config: prod.yaml + .primus.yaml
  ├─ Extract main.debug=true
  ├─ Set PRIMUS_LOG_LEVEL=DEBUG
  └─ Identify mode: slurm

Step 2: primus-cli-slurm.sh
  ├─ Load slurm.* config (nodes, time, partition, etc.)
  ├─ Parse Slurm params: srun -N 4
  ├─ Merge config and CLI params (CLI takes priority)
  │   Config: nodes=2, time=4:00:00, partition=gpu
  │   CLI: -N 4
  │   Result: nodes=4, time=4:00:00, partition=gpu
  ├─ Build SLURM_FLAGS: [-N 4 -p gpu -t 4:00:00]
  └─ Generate command: srun -N 4 -p gpu -t 4:00:00 \
                primus-cli-slurm-entry.sh -- \
                --config prod.yaml --debug \
                container --image rocm/megatron-lm:v25.8_py310 \
                -- train pretrain --config deepseek_v2.yaml

Step 3: primus-cli-slurm-entry.sh (runs on each Slurm node)
  ├─ Set node environment variables
  │   NODE_RANK, MASTER_ADDR, WORLD_SIZE
  └─ Call: primus-cli-container.sh --config prod.yaml --debug \
            --image rocm/megatron-lm:v25.8_py310 \
            -- train pretrain --config deepseek_v2.yaml

Step 4: primus-cli-container.sh (on each node)
  ├─ Load container.* config (image, devices, mounts, etc.)
  ├─ Parse container params: --image rocm/megatron-lm:v25.8_py310
  ├─ Merge config and CLI params
  │   Config: image=rocm/primus:v26.1
  │   CLI: --image rocm/megatron-lm:v25.8_py310
  │   Result: image=rocm/megatron-lm:v25.8_py310
  ├─ Build container options
  │   --device /dev/kfd, /dev/dri, /dev/infiniband
  │   --cap-add SYS_PTRACE, CAP_SYS_ADMIN
  │   --volume (mount data and code)
  │   --cpus, --memory (resource limits)
  └─ Start container: docker/podman run --rm \
                --device /dev/kfd --device /dev/dri \
                --volume $PWD:/workspace/Primus \
                --env NODE_RANK=$NODE_RANK \
                --env MASTER_ADDR=$MASTER_ADDR \
                rocm/megatron-lm:v25.8_py310 \
                /bin/bash -c "cd /workspace/Primus && \
                  ./primus-cli --config prod.yaml --debug direct \
                  -- train pretrain --config deepseek_v2.yaml"

Step 5: primus-cli-direct.sh (runs inside container)
  ├─ Load environment scripts
  │   ├─ base_env.sh (common environment)
  │   ├─ detect_gpu.sh (detected MI300X)
  │   └─ MI300X.sh (GPU-specific config)
  ├─ Execute Hooks
  │   └─ execute_hooks "train" "pretrain"
  │       ├─ hooks/train/pretrain/01_prepare.sh
  │       └─ hooks/train/pretrain/02_preprocess_data.sh
  ├─ Apply Patches
  │   └─ execute_patches (if configured)
  ├─ Set distributed environment
  │   MASTER_ADDR=node-0, NODE_RANK=0..3
  │   NCCL_SOCKET_IFNAME, NCCL_IB_HCA
  └─ Launch: torchrun --nproc_per_node=8 \
            --nnodes=4 --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            primus/cli/main.py train pretrain \
            --config deepseek_v2.yaml

Step 6: primus/cli/main.py (Python CLI, inside container)
  ├─ Parse command: train pretrain
  ├─ Load subcommand plugin: primus/cli/subcommands/train.py
  ├─ Execute training
  │   ├─ Load config: deepseek_v2.yaml
  │   ├─ Initialize Megatron
  │   ├─ Set up model, data, optimizer
  │   └─ Start distributed training
  └─ Output logs and metrics
```

**Multi-Layer Nesting Explanation**:
- **Slurm Layer**: Handles multi-node resource allocation and task scheduling
- **Container Layer**: Provides isolated runtime environment on each node
- **Direct Layer**: Executes actual training task inside container

This three-layer architecture achieves:
- ✅ Multi-node distributed training (Slurm)
- ✅ Environment consistency and isolation (Container)
- ✅ GPU auto-configuration and optimization (Direct)

### Configuration Priority Flow

Configuration system uses layered merging strategy, priority from low to high:

```
┌──────────────────────────────────────────┐
│    Configuration Sources (low to high)   │
└──────────────────────────────────────── ─┘
        ↓
┌──────────────────────────────────────────┐
│  1. User Global Config (~/.primus.yaml)  │
│     Priority: ★☆☆☆                     │
│     Usage: Personal default config       │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│  2. System Default (runner/.primus.yaml) │
│     Priority: ★★☆☆                     │
│     Usage: Project default config        │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│  3. Specified Config File (--config)     │
│     Priority: ★★★☆                     │
│     Usage: Environment-specific config   │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│  4. Command-line Arguments               │
│     Priority: ★★★★ (Highest)           │
│     Usage: Temporary override            │
└──────────────────────────────────────────┘
```

> 📖 Reference system default config file: [`.primus.yaml`](.primus.yaml)

**Config Merge Example**:

```bash
# ~/.primus.yaml:     nodes=1
# .primus.yaml:       nodes=2, time=2:00:00
# --config prod.yaml: nodes=4, time=4:00:00, partition=gpu
# CLI: -N 8

Final result:
  nodes=8           (from CLI - highest priority)
  time=4:00:00      (from prod.yaml)
  partition=gpu     (from prod.yaml)
```

### Three Modes Call Differences

| Component | Direct Mode | Container Mode | Slurm Mode |
|-----------|-------------|----------------|------------|
| **Entry Script** | primus-cli-direct.sh | primus-cli-container.sh | primus-cli-slurm.sh |
| **Environment Prep** | Load local GPU env | Start container + mount + devices | Allocate nodes + network config |
| **Execution Location** | Current host | Inside container | Slurm-allocated nodes |
| **Final Call** | Direct torchrun execution | Execute direct.sh in container | Each node executes slurm-entry.sh → direct.sh |
| **Distributed Support** | Single-node multi-GPU | Single-node multi-GPU | Multi-node multi-GPU |
| **Use Case** | Dev debugging | Environment isolation | Production training |

### GPU Environment Auto-Configuration

Primus CLI automatically detects GPU model and loads optimized configuration:

```
detect_gpu.sh (detect GPU model)
      ↓
  MI300X / MI250X / MI210 / ...
      ↓
source ${GPU_MODEL}.sh (load GPU-specific config)
      ↓
  • HSA_* environment variables
  • NCCL/RCCL optimization parameters
  • GPU-specific runtime config
```

**Supported GPU Config Files**:
- `MI300X.sh` - AMD Instinct MI300X
- `MI250X.sh` - AMD Instinct MI250X
- `MI210.sh` - AMD Instinct MI210
- More...

If no corresponding GPU config is found, the system falls back to `base_env.sh`.

---

## Best Practices

### 1. Use Config Files to Manage Environments

Create different config files for different environments:

```bash
configs/
├── dev.yaml          # Development environment
├── test.yaml         # Testing environment
└── prod.yaml         # Production environment
```

### 2. Make Good Use of Debug Mode

```bash
# First use dry-run to see what will be executed
./primus-cli --dry-run slurm srun -N 4 -- train pretrain

# After confirming, use debug mode for detailed tracking
./primus-cli --debug slurm srun -N 4 -- train pretrain
```

### 3. Container Development Workflow

```bash
# Local dev: mount local code
./primus-cli container \
  --volume ~/workspace/Primus:/workspace/Primus \
  -- train pretrain --config config.yaml

# Testing: use staging image
./primus-cli container --image rocm/primus:staging \
  -- benchmark gemm

# Production: use release image
./primus-cli container --image rocm/primus:v1.0.0 \
  -- train pretrain
```

### 4. Slurm Job Management

```bash
# Interactive dev and debugging
./primus-cli slurm srun -N 1 -- train pretrain --debug

# Production training: batch mode
./primus-cli slurm sbatch -N 8 -t 24:00:00 \
  -o logs/train_%j.log \
  -- train pretrain --config production.yaml
```

### 5. Log Management

```bash
# Slurm auto-manages logs
./primus-cli slurm sbatch -N 4 \
  -o logs/stdout_%j.log \
  -e logs/stderr_%j.log \
  -- train pretrain

# Container log redirect
./primus-cli container -- train pretrain 2>&1 | tee train.log
```

---

## Troubleshooting

### Common Issues

#### 1. "Unknown or unsupported mode"

**Cause**: Incorrect mode name or missing script file

**Solution**:
```bash
# Check available modes
ls runner/primus-cli-*.sh

# Ensure correct mode name is used
./primus-cli slurm ...    # ✓ Correct
./primus-cli Slurm ...    # ✗ Wrong (case-sensitive)
```

#### 2. "Config file not found"

**Cause**: Incorrect config file path

**Solution**:
```bash
# Use absolute path
./primus-cli --config /full/path/to/config.yaml ...

# Or relative to current directory
./primus-cli --config ./configs/dev.yaml ...
```

#### 3. Container Startup Failure

**Cause**: Docker/Podman not installed or insufficient permissions

**Solution**:
```bash
# Check container runtime
which docker || which podman

# Check permissions
docker ps
podman ps

# Use dry-run to view command
./primus-cli --dry-run container -- train pretrain
```

#### 4. Slurm Job Submission Failure

**Cause**: Incorrect Slurm parameters or insufficient resources

**Solution**:
```bash
# Check available partitions
sinfo

# Check queue status
squeue

# Use dry-run to check command
./primus-cli --dry-run slurm srun -N 4 -- train pretrain
```

#### 5. "Failed to load library"

**Cause**: Missing library files

**Solution**:
```bash
# Check library files
ls runner/lib/

# Ensure necessary files exist
# - lib/common.sh
# - lib/config.sh
```

### Debugging Tips

#### Enable Verbose Logging
```bash
# Method 1: Use --debug
./primus-cli --debug direct -- train pretrain

# Method 2: Set environment variable
export PRIMUS_LOG_LEVEL=DEBUG
./primus-cli direct -- train pretrain
```

#### Use Dry-run
```bash
# View complete command without executing
./primus-cli --dry-run slurm srun -N 4 -- train pretrain
```

#### Check Config Loading
```bash
# Use debug mode to view loaded config
./primus-cli --debug --config .primus.yaml direct -- train pretrain
```

### Getting Help

```bash
# Main entry help
./primus-cli --help

# Mode-specific help
./primus-cli slurm --help
./primus-cli container --help
./primus-cli direct --help

# Primus Python CLI help
./primus-cli direct -- --help
./primus-cli direct -- train --help
./primus-cli direct -- benchmark --help
```

---

## Reference Resources

### Related Documentation
- [CLI Architecture](./CLI-ARCHITECTURE.md) - Primus CLI architecture deep dive
- [Main Documentation](../README.md) - Complete Primus documentation index
- [.primus.yaml](../../runner/.primus.yaml) - Default configuration example

### Exit Code Convention

| Exit Code | Meaning | Example |
|-----------|---------|---------|
| 0 | Success | Normal execution completed |
| 1 | Library or dependency failure | Missing config library file |
| 2 | Invalid argument or config | Config file doesn't exist |
| 3 | Runtime execution failure | Training process failed |

---

## Version Information

- **Current Version**: 1.0.0
- **Last Updated**: 2025-11-10

---

**Happy Training! 🚀**
