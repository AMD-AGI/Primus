#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Validation functions library for Primus CLI
# Source this file in other scripts: source "${SCRIPT_DIR}/lib/validation.sh"
#

# Requires common.sh to be sourced first
if [[ -z "${__PRIMUS_COMMON_SOURCED:-}" ]]; then
    echo "[ERROR] validation.sh requires common.sh to be sourced first" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Guard: avoid duplicate sourcing
# ---------------------------------------------------------------------------
if [[ -n "${__PRIMUS_VALIDATION_SOURCED:-}" ]]; then
  return 0
fi
export __PRIMUS_VALIDATION_SOURCED=1

# ---------------------------------------------------------------------------
# Numeric Validation
# ---------------------------------------------------------------------------

# Validate integer
validate_integer() {
    local value="$1"
    local name="${2:-value}"

    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        die "Invalid $name: '$value' (must be a positive integer)"
    fi
}

# Validate integer in range
validate_integer_range() {
    local value="$1"
    local min="$2"
    local max="$3"
    local name="${4:-value}"

    validate_integer "$value" "$name"

    if [[ "$value" -lt "$min" ]] || [[ "$value" -gt "$max" ]]; then
        die "Invalid $name: $value (must be between $min and $max)"
    fi
}

# Validate positive integer
validate_positive_integer() {
    local value="$1"
    local name="${2:-value}"

    validate_integer "$value" "$name"

    if [[ "$value" -le 0 ]]; then
        die "Invalid $name: $value (must be greater than 0)"
    fi
}

# ---------------------------------------------------------------------------
# Distributed Training Parameter Validation
# ---------------------------------------------------------------------------

# Validate GPUS_PER_NODE
validate_gpus_per_node() {
    local gpus="${GPUS_PER_NODE:-}"

    if [[ -z "$gpus" ]]; then
        LOG_WARN "GPUS_PER_NODE not set, using default: 8"
        export GPUS_PER_NODE=8
        return 0
    fi

    validate_integer_range "$gpus" 1 8 "GPUS_PER_NODE"

    LOG_DEBUG "Validated GPUS_PER_NODE: $gpus"
}

# Validate NNODES
validate_nnodes() {
    local nnodes="${NNODES:-}"

    if [[ -z "$nnodes" ]]; then
        LOG_WARN "NNODES not set, using default: 1"
        export NNODES=1
        return 0
    fi

    validate_positive_integer "$nnodes" "NNODES"

    LOG_DEBUG "Validated NNODES: $nnodes"
}

# Validate NODE_RANK
validate_node_rank() {
    local node_rank="${NODE_RANK:-}"
    local nnodes="${NNODES:-1}"

    if [[ -z "$node_rank" ]]; then
        LOG_WARN "NODE_RANK not set, using default: 0"
        export NODE_RANK=0
        return 0
    fi

    validate_integer "$node_rank" "NODE_RANK"

    if [[ "$node_rank" -ge "$nnodes" ]]; then
        die "Invalid NODE_RANK: $node_rank (must be less than NNODES: $nnodes)"
    fi

    LOG_DEBUG "Validated NODE_RANK: $node_rank"
}

# Validate MASTER_ADDR
validate_master_addr() {
    local master_addr="${MASTER_ADDR:-}"

    if [[ -z "$master_addr" ]]; then
        LOG_WARN "MASTER_ADDR not set, using default: localhost"
        export MASTER_ADDR="localhost"
        return 0
    fi

    # Basic validation: not empty and doesn't contain invalid characters
    if [[ ! "$master_addr" =~ ^[a-zA-Z0-9._-]+$ ]]; then
        die "Invalid MASTER_ADDR: $master_addr (contains invalid characters)"
    fi

    LOG_DEBUG "Validated MASTER_ADDR: $master_addr"
}

# Validate MASTER_PORT
validate_master_port() {
    local master_port="${MASTER_PORT:-}"

    if [[ -z "$master_port" ]]; then
        LOG_WARN "MASTER_PORT not set, using default: 1234"
        export MASTER_PORT=1234
        return 0
    fi

    validate_integer_range "$master_port" 1024 65535 "MASTER_PORT"

    LOG_DEBUG "Validated MASTER_PORT: $master_port"
}

# Validate all distributed training parameters
validate_distributed_params() {
    LOG_DEBUG_RANK0 "Validating distributed training parameters..."

    validate_nnodes
    validate_node_rank
    validate_gpus_per_node
    validate_master_addr
    validate_master_port

    LOG_DEBUG_RANK0 "All distributed parameters validated successfully"
}

# ---------------------------------------------------------------------------
# Path Validation
# ---------------------------------------------------------------------------

# Validate file exists and is readable
validate_file_readable() {
    local file="$1"
    local name="${2:-file}"

    if [[ ! -f "$file" ]]; then
        die "Invalid $name: '$file' (file does not exist)"
    fi

    if [[ ! -r "$file" ]]; then
        die "Invalid $name: '$file' (file is not readable)"
    fi

    LOG_DEBUG "Validated $name: $file"
}

# Validate directory exists and is readable
validate_dir_readable() {
    local dir="$1"
    local name="${2:-directory}"

    if [[ ! -d "$dir" ]]; then
        die "Invalid $name: '$dir' (directory does not exist)"
    fi

    if [[ ! -r "$dir" ]]; then
        die "Invalid $name: '$dir' (directory is not readable)"
    fi

    LOG_DEBUG "Validated $name: $dir"
}

# Validate directory exists and is writable
validate_dir_writable() {
    local dir="$1"
    local name="${2:-directory}"

    if [[ ! -d "$dir" ]]; then
        die "Invalid $name: '$dir' (directory does not exist)"
    fi

    if [[ ! -w "$dir" ]]; then
        die "Invalid $name: '$dir' (directory is not writable)"
    fi

    LOG_DEBUG "Validated $name: $dir (writable)"
}

# Validate path is absolute
validate_absolute_path() {
    local path="$1"
    local name="${2:-path}"

    if [[ "$path" != /* ]]; then
        die "Invalid $name: '$path' (must be an absolute path)"
    fi

    LOG_DEBUG "Validated $name: $path (absolute)"
}

# ---------------------------------------------------------------------------
# Docker/Container Validation
# ---------------------------------------------------------------------------

# Validate Docker/Podman is available
validate_container_runtime() {
    if command -v podman >/dev/null 2>&1; then
        export CONTAINER_RUNTIME="podman"
        LOG_DEBUG "Container runtime: podman"
        return 0
    elif command -v docker >/dev/null 2>&1; then
        export CONTAINER_RUNTIME="docker"
        LOG_DEBUG "Container runtime: docker"
        return 0
    else
        die "No container runtime found (docker or podman required)"
    fi
}

# Validate Docker image exists
validate_docker_image() {
    local image="$1"
    local runtime="${CONTAINER_RUNTIME:-docker}"

    if ! $runtime images --format '{{.Repository}}:{{.Tag}}' | grep -q "^${image}$"; then
        LOG_WARN "Docker image not found locally: $image"
        LOG_INFO "Will attempt to pull on first use..."
    else
        LOG_DEBUG "Docker image found: $image"
    fi
}

# Validate mount path format
validate_mount_path() {
    local mount="$1"

    # Check if it's in HOST:CONTAINER format
    if [[ "$mount" == *:* ]]; then
        local host_path="${mount%%:*}"
        local container_path="${mount#*:}"

        if [[ ! -d "$host_path" ]]; then
            die "Invalid mount: host path does not exist: $host_path"
        fi

        if [[ "$container_path" != /* ]]; then
            die "Invalid mount: container path must be absolute: $container_path"
        fi
    else
        # Single path format
        if [[ ! -d "$mount" ]]; then
            die "Invalid mount: path does not exist: $mount"
        fi
    fi

    LOG_DEBUG "Validated mount: $mount"
}

# Validate volume format (supports newline-separated list)
# Format: /host:/container[:options] or /path or named_volume:/container[:options]
validate_volume_format() {
    local volumes="$1"
    local error_prefix="${2:-[validation]}"

    # Skip if empty or empty array marker
    [[ -z "$volumes" || "$volumes" == "[]" ]] && return 0

    local validation_failed=0
    while IFS= read -r volume_entry; do
        [[ -n "$volume_entry" ]] || continue

        # Volume format: /host:/container[:options] or /path or named_volume:/container[:options]
        if [[ "$volume_entry" == *:* ]]; then
            IFS=':' read -r src dst opts <<< "$volume_entry"

            # Check that source is not empty
            if [[ -z "$src" ]]; then
                LOG_ERROR "$error_prefix Invalid volume format: $volume_entry"
                LOG_ERROR "$error_prefix Source path cannot be empty"
                validation_failed=1
                continue
            fi

            # Count colons to distinguish /host: from /host:/dst
            local colon_count
            colon_count=$(echo "$volume_entry" | grep -o ":" | wc -l)
            if [[ $colon_count -ge 1 && -z "$dst" ]]; then
                LOG_ERROR "$error_prefix Invalid volume format: $volume_entry"
                LOG_ERROR "$error_prefix Destination path cannot be empty when colon is present"
                validation_failed=1
                continue
            fi
        else
            # Single path format (e.g., /workspace)
            opts=""
        fi

        # If options specified, validate they are valid
        if [[ -n "$opts" ]]; then
            IFS=',' read -ra opt_array <<< "$opts"
            for opt in "${opt_array[@]}"; do
                if ! [[ "$opt" =~ ^(ro|rw|z|Z|shared|slave|private|delegated|cached|consistent)$ ]]; then
                    LOG_ERROR "$error_prefix Invalid volume option: $opt in $volume_entry"
                    LOG_ERROR "$error_prefix Valid options: ro, rw, z, Z, shared, slave, private, delegated, cached, consistent"
                    validation_failed=1
                fi
            done
        fi

        LOG_DEBUG "Volume validated: $volume_entry"
    done <<< "$volumes"

    if [[ $validation_failed -eq 1 ]]; then
        die "$error_prefix Volume format validation failed"
    fi

    LOG_DEBUG "All volumes validated"
}

# Validate device paths exist on host (supports newline-separated list)
validate_device_paths() {
    local devices="$1"
    local error_prefix="${2:-[validation]}"
    local missing_error_msg="${3:-}"
    local validation_error_msg="${4:-}"

    # First check if devices are configured
    if [[ -z "$devices" || "$devices" == "[]" ]]; then
        if [[ -n "$missing_error_msg" ]]; then
            die "$missing_error_msg"
        else
            die "$error_prefix No GPU devices configured. Add via CLI (--device /dev/kfd --device /dev/dri) or config file"
        fi
    fi

    # Validate each device path exists
    local validation_failed=0
    while IFS= read -r device; do
        [[ -n "$device" ]] || continue
        if [[ ! -e "$device" ]]; then
            LOG_ERROR "$error_prefix Device does not exist on host: $device"
            validation_failed=1
        else
            LOG_DEBUG "Device validated: $device"
        fi
    done <<< "$devices"

    if [[ $validation_failed -eq 1 ]]; then
        if [[ -n "$validation_error_msg" ]]; then
            die "$validation_error_msg"
        else
            die "$error_prefix Device validation failed. Ensure ROCm drivers are installed. Check: ls -la /dev/kfd /dev/dri"
        fi
    fi

    LOG_DEBUG "All device paths validated"
}

# Validate memory format (e.g., 256G, 1024M)
validate_memory_format() {
    local memory="$1"
    local param_name="${2:-memory}"
    local error_msg="${3:-}"

    # Skip if empty or multiline
    [[ -z "$memory" || "$memory" == *$'\n'* ]] && return 0

    if ! [[ "$memory" =~ ^[0-9]+[bkmgBKMG]?$ ]]; then
        if [[ -n "$error_msg" ]]; then
            die "$error_msg"
        else
            die "Invalid $param_name format: $memory. Expected: <number>[b|k|m|g] (e.g., 256G, 1024M)"
        fi
    fi

    LOG_DEBUG "Validated $param_name: $memory"
}

# Validate CPUs format (e.g., 32, 16.5)
validate_cpus_format() {
    local cpus="$1"
    local param_name="${2:-cpus}"
    local error_msg="${3:-}"

    # Skip if empty or multiline
    [[ -z "$cpus" || "$cpus" == *$'\n'* ]] && return 0

    if ! [[ "$cpus" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        if [[ -n "$error_msg" ]]; then
            die "$error_msg"
        else
            die "Invalid $param_name format: $cpus. Expected: <number> or <number>.<decimal> (e.g., 32, 16.5)"
        fi
    fi

    LOG_DEBUG "Validated $param_name: $cpus"
}

# ---------------------------------------------------------------------------
# Slurm Validation
# ---------------------------------------------------------------------------

# Validate Slurm environment
validate_slurm_env() {
    if [[ -z "${SLURM_JOB_ID:-}" ]] && [[ -z "${SLURM_JOBID:-}" ]]; then
        die "Not running in a Slurm job (SLURM_JOB_ID not set)"
    fi

    if [[ -z "${SLURM_NODELIST:-}" ]]; then
        die "SLURM_NODELIST not set"
    fi

    LOG_DEBUG "Validated Slurm environment"
}

# Validate Slurm node count matches NNODES
validate_slurm_nodes() {
    local slurm_nnodes="${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-}}"
    local nnodes="${NNODES:-}"

    if [[ -n "$nnodes" ]] && [[ -n "$slurm_nnodes" ]]; then
        if [[ "$nnodes" != "$slurm_nnodes" ]]; then
            LOG_WARN "NNODES ($nnodes) doesn't match SLURM_NNODES ($slurm_nnodes)"
            LOG_INFO "Using SLURM_NNODES: $slurm_nnodes"
            export NNODES="$slurm_nnodes"
        fi
    fi
}

# ---------------------------------------------------------------------------
# Environment Variable Validation
# ---------------------------------------------------------------------------

# Validate required environment variable is set
validate_env_var() {
    local var_name="$1"
    local hint="${2:-}"

    if [[ -z "${!var_name:-}" ]]; then
        local msg="Required environment variable not set: $var_name"
        if [[ -n "$hint" ]]; then
            msg="$msg. $hint"
        fi
        die "$msg"
    fi

    LOG_DEBUG "Validated env var: $var_name=${!var_name}"
}

# Validate environment variable is one of allowed values
validate_env_var_choices() {
    local var_name="$1"
    shift
    local allowed_values=("$@")
    local value="${!var_name:-}"

    if [[ -z "$value" ]]; then
        die "Required environment variable not set: $var_name"
    fi

    local found=0
    for allowed in "${allowed_values[@]}"; do
        if [[ "$value" == "$allowed" ]]; then
            found=1
            break
        fi
    done

    if [[ "$found" == "0" ]]; then
        die "Invalid $var_name: $value (must be one of: ${allowed_values[*]})"
    fi

    LOG_DEBUG "Validated $var_name: $value"
}

# ---------------------------------------------------------------------------
# Configuration Validation
# ---------------------------------------------------------------------------

# Validate required config parameter is set
validate_config_param() {
    local param_value="$1"
    local param_name="$2"
    local error_msg="${3:-Missing required parameter: $param_name}"

    if [[ -z "$param_value" ]]; then
        die "$error_msg"
    fi

    LOG_DEBUG "Validated config param: $param_name"
}

# Validate environment variable format (supports newline-separated list)
# Supports two formats:
#   1. KEY=VALUE - Set environment variable to specific value
#   2. KEY       - Pass through host environment variable (for containers)
validate_env_format() {
    local env_vars="$1"
    local error_prefix="${2:-[validation]}"

    # Skip if empty or empty array marker
    [[ -z "$env_vars" || "$env_vars" == "[]" ]] && return 0

    local validation_failed=0
    while IFS= read -r env_entry; do
        [[ -n "$env_entry" ]] || continue

        # Accept two formats:
        # 1. KEY=VALUE (set value)
        # 2. KEY (pass through from host)
        # Valid environment variable names: start with letter or underscore, followed by letters, digits, or underscores
        if [[ "$env_entry" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || [[ "$env_entry" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
            # Valid format
            continue
        else
            LOG_ERROR "$error_prefix Invalid env format: $env_entry"
            LOG_ERROR "$error_prefix Expected format:"
            LOG_ERROR "$error_prefix   - KEY=VALUE (e.g., NCCL_DEBUG=INFO)"
            LOG_ERROR "$error_prefix   - KEY (pass through from host, e.g., CUDA_VISIBLE_DEVICES)"
            validation_failed=1
        fi
    done <<< "$env_vars"

    if [[ $validation_failed -eq 1 ]]; then
        die "$error_prefix Environment variable validation failed"
    fi

    LOG_DEBUG "All environment variables validated"
}

# Validate array config parameter is not empty
validate_config_array() {
    local param_value="$1"
    local param_name="$2"
    local error_msg="${3:-Missing required array parameter: $param_name}"

    if [[ -z "$param_value" || "$param_value" == "[]" ]]; then
        die "$error_msg"
    fi

    LOG_DEBUG "Validated config array: $param_name"
}

# Validate positional arguments are provided
validate_positional_args() {
    local -n args_array="$1"
    local error_msg="${2:-Missing required arguments}"

    if [[ ${#args_array[@]} -eq 0 ]]; then
        die "$error_msg"
    fi

    LOG_DEBUG "Validated positional args: ${#args_array[@]} arguments"
}

# ---------------------------------------------------------------------------
# Script Validation
# ---------------------------------------------------------------------------

# Validate Python script exists
validate_python_script() {
    local script="$1"

    validate_file_readable "$script" "Python script"

    if [[ "$script" != *.py ]]; then
        LOG_WARN "Script doesn't have .py extension: $script"
    fi
}

# Validate bash script exists and is executable
validate_bash_script() {
    local script="$1"

    validate_file_readable "$script" "Bash script"

    if [[ ! -x "$script" ]]; then
        LOG_WARN "Script is not executable: $script"
    fi
}

# ---------------------------------------------------------------------------
# Export all functions
# ---------------------------------------------------------------------------
export -f validate_integer validate_integer_range validate_positive_integer
export -f validate_gpus_per_node validate_nnodes validate_node_rank
export -f validate_master_addr validate_master_port validate_distributed_params
export -f validate_file_readable validate_dir_readable validate_dir_writable validate_absolute_path
export -f validate_container_runtime validate_docker_image validate_mount_path validate_volume_format validate_device_paths
export -f validate_memory_format validate_cpus_format
export -f validate_slurm_env validate_slurm_nodes
export -f validate_env_var validate_env_var_choices
export -f validate_config_param validate_config_array validate_positional_args validate_env_format
export -f validate_python_script validate_bash_script

LOG_DEBUG_RANK0 "Primus validation library loaded successfully"
