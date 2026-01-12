# Primus Runner - CLI Quick Start Guide

This document provides a quick start guide for using **Primus CLI commands directly** (no wrapper scripts), focusing on:

- **Direct mode**: single-node training
- **Slurm mode**: multi-node distributed training

## 📚 Overview

Primus CLI provides two primary training modes:

| Mode | Command | Use Case | Rating |
|------|---------|----------|--------|
| Direct | `primus-cli direct` | Single-node training on local host | ⭐⭐⭐ |
| Slurm | `primus-cli slurm` | Multi-node distributed training on a cluster | ⭐⭐⭐ |

> **Note**: For containerized environments, use `primus-cli container` (or `slurm ... -- container ...` in Slurm mode).

---

## 1️⃣ Direct Mode - Single Node Training

**Best for**: Development, testing, single-GPU/multi-GPU training on a single machine.

### Basic Usage

```bash
# From anywhere:
cd /path/to/Primus

# Use default config (Llama3.1 8B BF16)
bash runner/primus-cli direct \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
```

### Default Configuration

- **Config**: `examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml`
- **Override**: Set `EXP` environment variable

### CLI Feature Scenarios

The examples below focus on **Primus CLI features and runner behavior** (not training/hyperparameter tuning).

#### Scenario 1: Print the command without running (`--dry-run`)

Use this to validate parsing, config loading, and the final launcher command.

```bash
bash runner/primus-cli direct \
  --dry-run \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
```

#### Scenario 2: Pass environment variables via runner (`--env`)

```bash
bash runner/primus-cli direct \
  --env NCCL_DEBUG=INFO \
  --env PRIMUS_LOG_LEVEL=DEBUG \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
```

#### Scenario 3: Save logs to a file (`--log_file`)

```bash
bash runner/primus-cli direct \
  --log_file /tmp/primus_direct.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
```

#### Scenario 4: Enable NUMA binding (`--numa`)

```bash
bash runner/primus-cli direct \
  --numa \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
```

#### Scenario 5: Apply patch scripts before running (`--patch`)

Patch scripts are executed in order before launching the Python entrypoint.

```bash
bash runner/primus-cli direct \
  --patch runner/helpers/patches/00_hello_world.sh \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
```

#### Scenario 6: Run a custom Python entrypoint (`--script`) / single-process mode (`--single`)

Useful for debugging or for backends that require a non-`torchrun` launcher.

```bash
# Single process (python3), custom script, plus script args after '--'
bash runner/primus-cli direct \
  --single \
  --script runner/helpers/examples/hello_world.py \
  -- --arg1 val1
```

---

## 2️⃣ Slurm Mode - Multi-Node Launcher

**Best for**: Launching Primus jobs on a Slurm cluster (multi-node / multi-GPU).

### Basic Usage

The Slurm wrapper uses a **two-level `--` separator**:
- **Before the first `--`**: passed to Slurm (`srun|sbatch` and all Slurm flags).
- **After the first `--`**: Primus slurm entry (`container | direct`).
- **After the second `--`** (optional): arguments to the Primus CLI entry (for example, `primus-cli container ...` arguments).

```bash
cd /path/to/Primus

# Minimal multi-node training launch (example: 4 nodes)
bash runner/primus-cli slurm -N 2 \
  --nodelist "node[01-04]" \
-- train pretrain --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
```

### CLI Feature Scenarios

The examples below focus on **primus-cli slurm launcher features** (not training workflows or hyperparameter tuning).

#### Scenario 1: Dry-run the Slurm launch (global `--dry-run`)

```bash
bash runner/primus-cli --dry-run slurm \
  -N 2 \
  --nodelist "node[01-02]" \
-- train pretrain --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
```

#### Scenario 2: Common Slurm flags (partition/reservation/account/qos/constraint)

```bash
bash runner/primus-cli slurm \
  -N 4 \
  -p AIG_Model \
  --reservation my_reservation \
  --account my_account \
  --qos my_qos \
  --constraint "mi300x" \
  --exclusive \
-- train pretrain --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
```

#### Scenario 3: Run inside a container on Slurm (`-- container`)

This is the most common pattern when you need a fixed software stack.

```bash
bash runner/primus-cli slurm \
  -N 4 \
  --nodelist "node[01-04]" \
-- --image "rocm/primus:v25.10" \
-- --env NCCL_DEBUG=INFO \
-- train pretrain --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
```

---

## 🔧 Tips

```bash
# Dry-run is your best friend for validating quoting and the final Slurm command:
bash runner/primus-cli slurm -N 2 --nodelist "node[01-02]" --dry-run -- train pretrain --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
```

---


## 📚 References

- [CLI User Guide](../docs/cli/PRIMUS-CLI-GUIDE.md)
- [Configuration Examples](../examples/)
- [Troubleshooting Guide](../docs/troubleshooting.md)

---

**Last Updated**: 2026-01-12
**Version**: v1.2
