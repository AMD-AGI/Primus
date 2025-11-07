# Primus CLI Quick Reference Guide

## 🚀 Quick Start

### Using the New Library Functions

#### 1. **Add to Your Script**
```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load common library
source "$SCRIPT_DIR/lib/common.sh"

# Load validation library (optional)
source "$SCRIPT_DIR/lib/validation.sh"
```

#### 2. **Use Logging Functions**
```bash
LOG_INFO "Starting process..."
LOG_DEBUG "Debug information"
LOG_WARN "Warning message"
LOG_ERROR "Error message"
LOG_SUCCESS "Success!"

# Only log on rank 0 (distributed jobs)
LOG_INFO_RANK0 "Master node message"
```

#### 3. **Validate Parameters**
```bash
# Validate all distributed training parameters
validate_distributed_params

# Validate specific parameters
validate_gpus_per_node
validate_nnodes
validate_file_readable "config.yaml"
```

#### 4. **Error Handling**
```bash
# Require command
require_command "python3" "Install Python 3.8+"

# Require file/directory
require_file "config.yaml"
require_dir "/data"

# Run command with error checking
run_cmd "python train.py"

# Exit with error
die "Fatal error occurred"
```

---

## 📋 Common Use Cases

### Use Case 1: Script with Logging
```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

export NODE_RANK=0

LOG_INFO "Starting training..."
require_command "python3"
run_cmd "python3 train.py"
LOG_SUCCESS "Training completed!"
```

### Use Case 2: Distributed Training Setup
```bash
#!/bin/bash
source "$SCRIPT_DIR/lib/common.sh"
source "$SCRIPT_DIR/lib/validation.sh"

# Set parameters
export NNODES=4
export NODE_RANK=0
export GPUS_PER_NODE=8
export MASTER_ADDR="node1"
export MASTER_PORT=1234

# Validate
validate_distributed_params

# Log configuration
log_exported_vars "Distributed Setup" \
    NNODES NODE_RANK GPUS_PER_NODE MASTER_ADDR MASTER_PORT
```

### Use Case 3: Path Validation
```bash
#!/bin/bash
source "$SCRIPT_DIR/lib/common.sh"
source "$SCRIPT_DIR/lib/validation.sh"

CONFIG_FILE="config.yaml"
OUTPUT_DIR="/output"

# Validate
validate_file_readable "$CONFIG_FILE" "configuration file"
validate_dir_writable "$OUTPUT_DIR" "output directory"

LOG_SUCCESS "All paths validated"
```

### Use Case 4: Container Runtime Detection
```bash
#!/bin/bash
source "$SCRIPT_DIR/lib/common.sh"
source "$SCRIPT_DIR/lib/validation.sh"

# Detect and validate container runtime
if validate_container_runtime; then
    LOG_INFO "Using container runtime: $CONTAINER_RUNTIME"
else
    die "No container runtime found"
fi
```

---

## 🎨 Customization

### Environment Variables

#### Logging Control
```bash
# Set log level (DEBUG, INFO, WARN, ERROR)
export PRIMUS_LOG_LEVEL=DEBUG

# Disable timestamps
export PRIMUS_LOG_TIMESTAMP=0

# Disable colors
export PRIMUS_LOG_COLOR=0
```

#### Example with Custom Logging
```bash
export PRIMUS_LOG_LEVEL=DEBUG
export NODE_RANK=0

source "$SCRIPT_DIR/lib/common.sh"

LOG_DEBUG "This will be shown because log level is DEBUG"
LOG_INFO "This will be shown"
```

---

## 🔧 Validation Functions Reference

### Distributed Training
```bash
validate_distributed_params     # Validate all distributed parameters
validate_nnodes                 # Validate NNODES (must be > 0)
validate_node_rank              # Validate NODE_RANK (0 to NNODES-1)
validate_gpus_per_node          # Validate GPUS_PER_NODE (1-8)
validate_master_addr            # Validate MASTER_ADDR
validate_master_port            # Validate MASTER_PORT (1024-65535)
```

### Numeric
```bash
validate_integer "123" "value_name"
validate_integer_range "5" 1 10 "value_name"
validate_positive_integer "5" "value_name"
```

### Paths
```bash
validate_file_readable "file.txt" "input file"
validate_dir_readable "/data" "data directory"
validate_dir_writable "/output" "output directory"
validate_absolute_path "/abs/path" "path"
```

### Container & Slurm
```bash
validate_container_runtime      # Detect docker/podman
validate_docker_image "image:tag"
validate_slurm_env             # Validate Slurm environment
validate_slurm_nodes           # Validate node count consistency
```

---

## 📊 Logging Functions Reference

### Basic Logging
```bash
LOG_DEBUG "message"      # Debug level (only if PRIMUS_LOG_LEVEL=DEBUG)
LOG_INFO "message"       # Info level
LOG_WARN "message"       # Warning (stderr)
LOG_ERROR "message"      # Error (stderr)
LOG_SUCCESS "message"    # Success
```

### Rank-Aware Logging (for distributed jobs)
```bash
LOG_INFO_RANK0 "message"      # Only log on NODE_RANK=0
LOG_DEBUG_RANK0 "message"     # Only debug on rank 0
LOG_SUCCESS_RANK0 "message"   # Only success on rank 0
```

### Variable Logging
```bash
log_exported_vars "Title" VAR1 VAR2 VAR3

# Example:
export NNODES=4
export GPUS_PER_NODE=8
log_exported_vars "Cluster Config" NNODES GPUS_PER_NODE

# Output:
# ========== Cluster Config ==========
#     NNODES=4
#     GPUS_PER_NODE=8
```

---

## 🛠️ Utility Functions Reference

### Error Handling
```bash
die "error message"                           # Exit with error
require_command "cmd" "hint"                 # Require command exists
require_file "file"                          # Require file exists
require_dir "dir"                            # Require directory exists
run_cmd "command"                            # Run command with error check
run_cmd_capture "command"                    # Run and capture output
```

### Path Utilities
```bash
get_absolute_path "path"                     # Get absolute path
get_script_dir                               # Get current script directory
ensure_dir "dir"                             # Create directory if needed
cleanup_temp "path"                          # Clean up temp files
```

### Environment Utilities
```bash
export_and_log "KEY" "value"                 # Export and log variable
set_default "KEY" "default_value"            # Set default if not set
load_env_file ".env"                         # Load environment file
```

### System Utilities
```bash
get_cpu_count                                # Get number of CPUs
get_memory_gb                                # Get memory in GB
is_container                                 # Check if in container
is_slurm_job                                 # Check if in Slurm job
print_system_info                            # Print system information
```

### Cleanup Hooks
```bash
register_cleanup_hook "command"              # Register cleanup function

# Example:
TEMP_DIR=$(mktemp -d)
register_cleanup_hook "rm -rf $TEMP_DIR"
# Cleanup runs automatically on exit
```

---

## 📚 Examples by Scenario

### Scenario: Distributed Training Script
```bash
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"
source "$SCRIPT_DIR/lib/validation.sh"

# Validate environment
validate_distributed_params

# Log configuration
LOG_INFO_RANK0 "Starting distributed training"
log_exported_vars "Training Configuration" \
    NNODES NODE_RANK GPUS_PER_NODE MASTER_ADDR MASTER_PORT

# Run training
require_command "torchrun"
run_cmd "torchrun --nproc_per_node=$GPUS_PER_NODE train.py"

LOG_SUCCESS_RANK0 "Training completed successfully"
```

### Scenario: Container Launch Script
```bash
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"
source "$SCRIPT_DIR/lib/validation.sh"

# Validate container runtime
validate_container_runtime
LOG_INFO "Using $CONTAINER_RUNTIME"

# Validate mounts
MOUNT_DIR="/data"
validate_dir_readable "$MOUNT_DIR" "data directory"

# Launch container
IMAGE="rocm/primus:latest"
LOG_INFO "Launching container: $IMAGE"
run_cmd "$CONTAINER_RUNTIME run -v $MOUNT_DIR:$MOUNT_DIR $IMAGE"

LOG_SUCCESS "Container completed"
```

### Scenario: Slurm Job Script
```bash
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"
source "$SCRIPT_DIR/lib/validation.sh"

# Validate Slurm environment
validate_slurm_env

# Extract Slurm info
export NNODES=${SLURM_NNODES}
export NODE_RANK=${SLURM_NODEID}

# Validate and log
validate_distributed_params
LOG_INFO "Slurm Job ID: ${SLURM_JOB_ID}"
LOG_INFO "Node: $NODE_RANK / $NNODES"

# Run job
run_cmd "./train.sh"

LOG_SUCCESS "Job completed"
```

---

## 🎯 Best Practices

1. **Always source common.sh first**
   ```bash
   source "$SCRIPT_DIR/lib/common.sh"
   source "$SCRIPT_DIR/lib/validation.sh"  # After common.sh
   ```

2. **Set NODE_RANK for proper logging**
   ```bash
   export NODE_RANK=${NODE_RANK:-0}
   ```

3. **Use validation functions early**
   ```bash
   # Validate at the start
   validate_distributed_params
   # Then proceed with logic
   ```

4. **Register cleanup hooks**
   ```bash
   TEMP_DIR=$(mktemp -d)
   register_cleanup_hook "rm -rf $TEMP_DIR"
   ```

5. **Use LOG_*_RANK0 for distributed jobs**
   ```bash
   # Avoid log spam from all ranks
   LOG_INFO_RANK0 "Master node message"
   ```

6. **Check for library availability (for compatibility)**
   ```bash
   if type LOG_INFO &>/dev/null; then
       LOG_INFO "Using new library"
   else
       echo "Fallback to old method"
   fi
   ```

---

## 🐛 Troubleshooting

### Issue: Library not found
```bash
# Solution: Check path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Script dir: $SCRIPT_DIR"
ls -la "$SCRIPT_DIR/lib/"
```

### Issue: Functions not available
```bash
# Solution: Check if sourced correctly
type LOG_INFO  # Should show "LOG_INFO is a function"
```

### Issue: Validation fails
```bash
# Solution: Check environment variables
echo "NNODES=$NNODES"
echo "NODE_RANK=$NODE_RANK"
echo "GPUS_PER_NODE=$GPUS_PER_NODE"
```

### Issue: Logging not showing
```bash
# Solution: Check log level
export PRIMUS_LOG_LEVEL=DEBUG
export NODE_RANK=0
```

---

## 📞 Getting Help

- **Documentation**: `runner/lib/README.md`
- **Examples**: `runner/lib/test_*.sh`
- **Summary**: `OPTIMIZATION_WEEK1_SUMMARY.md`

---

**Happy coding! 🚀**
