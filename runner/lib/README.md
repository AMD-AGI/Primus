# Primus CLI Library

This directory contains shared libraries for Primus CLI runner scripts.

## Files

### `common.sh`
Common utility functions used across all Primus CLI scripts.

**Features:**
- **Logging**: Structured logging with levels (DEBUG, INFO, WARN, ERROR)
- **Error Handling**: Functions for graceful error handling and cleanup
- **Path Utilities**: Path resolution, directory creation, file validation
- **Environment Utilities**: Environment variable management
- **System Utilities**: CPU/memory detection, container detection, Slurm detection
- **Color Support**: Automatic color detection for better readability

**Usage:**
```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

# Now you can use logging functions
LOG_INFO "Starting process..."
LOG_ERROR "Something went wrong!"
```

**Environment Variables:**
- `PRIMUS_LOG_LEVEL`: Set log level (DEBUG, INFO, WARN, ERROR) [default: INFO]
- `PRIMUS_LOG_TIMESTAMP`: Enable timestamps (0=off, 1=on) [default: 1]
- `PRIMUS_LOG_COLOR`: Enable colors (0=off, 1=on) [default: 1]

**Key Functions:**
- `LOG_INFO()`, `LOG_WARN()`, `LOG_ERROR()`, `LOG_DEBUG()`, `LOG_SUCCESS()`
- `LOG_INFO_RANK0()` - Only logs on rank 0 for distributed jobs
- `die()` - Exit with error message
- `require_command()`, `require_file()`, `require_dir()` - Validate requirements
- `run_cmd()` - Run command with error checking
- `ensure_dir()` - Create directory if it doesn't exist
- `load_env_file()` - Load .env file
- `register_cleanup_hook()` - Register cleanup function to run on exit

### `validation.sh`
Parameter validation functions for Primus CLI.

**Features:**
- **Numeric Validation**: Integer, range, positive number validation
- **Distributed Training**: Validate NNODES, NODE_RANK, GPUS_PER_NODE, etc.
- **Path Validation**: File existence, readability, writability checks
- **Container Validation**: Docker/Podman runtime validation
- **Slurm Validation**: Slurm environment validation

**Usage:**
```bash
#!/bin/bash
source "$SCRIPT_DIR/lib/common.sh"
source "$SCRIPT_DIR/lib/validation.sh"

# Validate distributed training parameters
validate_distributed_params

# Validate specific parameters
validate_gpus_per_node
validate_nnodes
validate_node_rank

# Validate paths
validate_file_readable "config.yaml" "configuration file"
validate_dir_writable "/output" "output directory"
```

**Key Functions:**
- `validate_integer()`, `validate_integer_range()`, `validate_positive_integer()`
- `validate_distributed_params()` - Validate all distributed parameters
- `validate_gpus_per_node()`, `validate_nnodes()`, `validate_node_rank()`
- `validate_master_addr()`, `validate_master_port()`
- `validate_file_readable()`, `validate_dir_readable()`, `validate_dir_writable()`
- `validate_container_runtime()` - Check Docker/Podman availability
- `validate_slurm_env()` - Validate Slurm environment

## Examples

### Example 1: Basic Logging
```bash
#!/bin/bash
source "$(dirname "$0")/lib/common.sh"

LOG_INFO "Starting training..."
LOG_DEBUG "Debug information"
LOG_WARN "Warning: using default configuration"
LOG_ERROR "Error: file not found"
LOG_SUCCESS "Training completed!"
```

### Example 2: Error Handling
```bash
#!/bin/bash
source "$(dirname "$0")/lib/common.sh"

# Exit if command not found
require_command "python3" "Install Python 3.8 or later"

# Validate file exists
require_file "config.yaml"

# Run command with error checking
run_cmd "python3 train.py --config config.yaml"
```

### Example 3: Parameter Validation
```bash
#!/bin/bash
source "$(dirname "$0")/lib/common.sh"
source "$(dirname "$0")/lib/validation.sh"

# Set environment variables
export NNODES=4
export NODE_RANK=0
export GPUS_PER_NODE=8

# Validate all distributed parameters
validate_distributed_params

LOG_SUCCESS "All parameters validated successfully"
```

### Example 4: Cleanup Hooks
```bash
#!/bin/bash
source "$(dirname "$0")/lib/common.sh"

# Create temporary directory
TEMP_DIR=$(mktemp -d)

# Register cleanup hook
register_cleanup_hook "rm -rf $TEMP_DIR"

# Use temporary directory
LOG_INFO "Using temporary directory: $TEMP_DIR"

# Cleanup will be called automatically on exit
```

## Integration with Existing Scripts

The library is designed to be backward compatible. All scripts check if functions are available before using them:

```bash
if type LOG_INFO &>/dev/null; then
    LOG_INFO "Using new logging"
else
    echo "Fallback to old logging"
fi
```

This ensures that scripts work even if the library is not available.

## Migration Guide

To migrate existing scripts to use the library:

1. **Add library source at the top:**
   ```bash
   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
   if [[ -f "$SCRIPT_DIR/lib/common.sh" ]]; then
       source "$SCRIPT_DIR/lib/common.sh"
   fi
   ```

2. **Replace echo with LOG_* functions:**
   ```bash
   # Before:
   echo "[INFO] Starting..."

   # After:
   LOG_INFO "Starting..."
   ```

3. **Add parameter validation:**
   ```bash
   source "$SCRIPT_DIR/lib/validation.sh"
   validate_distributed_params
   ```

4. **Use error handling functions:**
   ```bash
   # Before:
   if [[ ! -f "$file" ]]; then
       echo "Error: file not found" >&2
       exit 1
   fi

   # After:
   require_file "$file"
   ```

## Testing

Run the test script to verify library functionality:
```bash
bash runner/lib/test_common.sh
```

## Version History

### v1.0.0 (Week 1 - Current)
- Initial release
- Added `common.sh` with logging, error handling, and utilities
- Added `validation.sh` with parameter validation
- Integrated with main scripts (primus-cli, base_env.sh, primus-cli-slurm-entry.sh)
- Optimized GPU-specific configuration files (MI300X, MI325X, MI355X)

## Future Enhancements

- Add performance profiling utilities
- Add metrics collection functions
- Add distributed job coordination helpers
- Add container orchestration helpers
