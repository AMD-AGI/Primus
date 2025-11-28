# Primus CLI Configuration Guide

## Overview

Primus CLI supports configuration files to avoid repetitive command-line arguments and set project or global defaults.

### Configuration Priority

Settings are applied in this order (highest to lowest priority):

```
Command-line arguments > Project config > Global config > Defaults
```

Example:
- **Defaults**: `gpus_per_node: 8`
- **Global config** (`~/.primusrc`): `gpus_per_node: 16`
- **Project config** (`.primus.yaml`): `gpus_per_node: 32`
- **CLI**: `--gpus 64`

**Result**: 64 (CLI wins)

---

## Configuration File Locations

### 1. Global Configuration
**File**: `~/.primusrc` (Shell format)

Used for personal preferences across all projects.

**Example**:
```bash
# ~/.primusrc
PRIMUS_DISTRIBUTED_GPUS_PER_NODE="8"
PRIMUS_CONTAINER_IMAGE="rocm/primus:v25.9_gfx942"
PRIMUS_PATHS_PRIMUS_PATH="/home/user/workspace/Primus"
```

### 2. Project Configuration
**File**: `.primus.yaml` (YAML format)

Place in your project root for team-shared settings.

**Example**:
```yaml
# .primus.yaml
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

### 3. Custom Configuration
**CLI**: `--config <file>`

Load any configuration file explicitly.

```bash
primus-cli --config my-config.yaml container -- train
```

---

## Configuration Format

### Shell Format (.primusrc)

```bash
# Naming convention: PRIMUS_<SECTION>_<KEY>="value"

# Distributed settings
PRIMUS_DISTRIBUTED_GPUS_PER_NODE="8"
PRIMUS_DISTRIBUTED_MASTER_PORT="1234"

# Container settings
PRIMUS_CONTAINER_IMAGE="rocm/primus:latest"
PRIMUS_CONTAINER_CPUS="16"
PRIMUS_CONTAINER_MEMORY="128G"

# Path settings
PRIMUS_PATHS_LOG_PATH="logs"
```

### YAML Format (.primus.yaml)

```yaml
# Hierarchical structure

global:
  debug: false         # Enable debug mode
  dry_run: false       # Dry-run mode

distributed:
  gpus_per_node: 8     # GPUs per node
  master_port: 1234    # Master port
  nnodes: 1            # Number of nodes

container:
  image: "rocm/primus:v25.9_gfx942"
  cpus: 16             # CPU limit
  memory: "128G"       # Memory limit
  shm_size: "16G"      # Shared memory size
  user: "1000:1000"    # Run as user

slurm:
  partition: "AIG_Model"
  default_nodes: 4

paths:
  log_path: "logs"
  data_path: "/data"
  output_path: "/output"
```

---

## Configuration Options

### Global Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `global.debug` | boolean | false | Enable debug mode (bash -x) |
| `global.dry_run` | boolean | false | Dry-run mode |

### Distributed Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `distributed.gpus_per_node` | int | 8 | GPUs per node |
| `distributed.master_port` | int | 1234 | Master port |
| `distributed.nnodes` | int | 1 | Number of nodes |
| `distributed.master_addr` | string | localhost | Master address |

### Container Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `container.image` | string | rocm/primus:v25.9_gfx942 | Docker image |
| `container.cpus` | int | - | CPU limit |
| `container.memory` | string | - | Memory limit (e.g., "128G") |
| `container.shm_size` | string | - | Shared memory size |
| `container.user` | string | - | Run as user (UID:GID) |

### Path Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `paths.log_path` | string | logs | Log directory |
| `paths.data_path` | string | - | Data directory |
| `paths.output_path` | string | - | Output directory |
| `paths.primus_path` | string | - | Primus source path |

---

## Usage Examples

### Example 1: Personal Defaults

**File**: `~/.primusrc`
```bash
# My personal preferences
PRIMUS_PATHS_PRIMUS_PATH="/home/myuser/workspace/Primus"
```

**Usage**:
```bash
# Automatically uses my Primus path
primus-cli direct -- train pretrain --config exp.yaml
```

### Example 2: Team Project Configuration

**File**: `.primus.yaml` (in project root)
```yaml
# Team-shared configuration
container:
  image: "rocm/primus:v25.9_gfx942"
  cpus: 32
  memory: "256G"

distributed:
  gpus_per_node: 8

paths:
  data_path: "/mnt/shared/data"
  output_path: "/mnt/shared/output"
```

**Usage**:
```bash
# All team members use the same settings
primus-cli container -- train pretrain --config exp.yaml
```

### Example 3: Override Configuration

**Command**:
```bash
# Project config has cpus: 32, but override to 64
primus-cli container --cpus 64 -- train pretrain --config exp.yaml
```

### Example 4: Multiple Environments

**dev.yaml**:
```yaml
container:
  image: "rocm/primus:dev"
  cpus: 8
  memory: "64G"
```

**prod.yaml**:
```yaml
container:
  image: "rocm/primus:v25.9_gfx942"
  cpus: 64
  memory: "512G"
```

**Usage**:
```bash
# Development
primus-cli --config dev.yaml container -- train

# Production
primus-cli --config prod.yaml container -- train
```

---

## CLI Options

### View Current Configuration

```bash
primus-cli --show-config
```

**Output**:
```
========== Current Configuration ==========
Global:
  dry_run: 0
  debug: 0

Distributed:
  gpus_per_node: 8
  master_port: 1234
  nnodes: 1

Container:
  image: rocm/primus:v25.9_gfx942
  cpus: <not set>
  memory: <not set>

Paths:
  log_path: logs
==========================================
```

### Load Specific Configuration

```bash
primus-cli --config <file> <mode> -- [args]
```

**Example**:
```bash
primus-cli --config production.yaml container -- train pretrain --config exp.yaml
```

---

## Best Practices

### 1. Use Global Config for Personal Preferences

**~/.primusrc**:
```bash
# Your personal defaults
PRIMUS_PATHS_PRIMUS_PATH="/home/you/workspace/Primus"
```

### 2. Use Project Config for Team Settings

**.primus.yaml** (commit to git):
```yaml
# Team-shared defaults
container:
  image: "rocm/primus:v25.9_gfx942"
distributed:
  gpus_per_node: 8
```

### 3. Use CLI for One-off Changes

```bash
# Override just for this run
primus-cli --debug container --cpus 64 -- train
```

### 4. Document Your Configuration

```yaml
# .primus.yaml

# Project: GPT-3 Training
# Team: AI Research
# Last updated: 2025-11-06

container:
  image: "rocm/primus:v25.9_gfx942"  # Stable release
  cpus: 32                            # Per-node CPU limit
  memory: "256G"                      # Required for large models
```

### 5. Version Control Project Config

```bash
# Add to git
git add .primus.yaml
git commit -m "Add project configuration"
```

**Do NOT commit** personal configs (`~/.primusrc`).

---

## Troubleshooting

### Configuration Not Loading

**Problem**: Configuration file not found or ignored

**Solution**:
1. Check file location:
   ```bash
   ls -la ~/.primusrc
   ls -la .primus.yaml
   ```

2. Check file format:
   ```bash
   # YAML must be valid
   cat .primus.yaml | python3 -c "import yaml, sys; yaml.safe_load(sys.stdin)"
   ```

3. Enable debug mode:
   ```bash
   primus-cli --debug --show-config
   ```

### Configuration Override Not Working

**Problem**: CLI arguments don't override config

**Solution**: Check argument syntax:
```bash
# Correct (global options before mode)
primus-cli --debug container -- train

# Incorrect (global options after mode)
primus-cli container --debug -- train
```

Global options must come **before** the mode.

### Example Configuration Files

See example configurations:
- `examples/primusrc.example` - Shell format example
- `examples/primus.yaml.example` - YAML format example

Copy and customize:
```bash
# Global config
cp examples/primusrc.example ~/.primusrc
nano ~/.primusrc

# Project config
cp examples/primus.yaml.example .primus.yaml
nano .primus.yaml
```

---

## Summary

- **Three levels**: Global (`~/.primusrc`), Project (`.primus.yaml`), CLI args
- **Priority**: CLI > Project > Global > Defaults
- **Formats**: Shell (.primusrc) or YAML (.primus.yaml)
- **View**: `primus-cli --show-config`
- **Load**: `primus-cli --config <file>`
- **Best Practice**: Global for personal, Project for team, CLI for overrides

---

For more information, see:
- [Quick Reference](../runner/QUICK_REFERENCE.md)
- [Example Configurations](../examples/)
- [Week 3 Summary](../OPTIMIZATION_WEEK3_SUMMARY.md)
