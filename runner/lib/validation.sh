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
    LOG_INFO_RANK0 "Validating distributed training parameters..."

    validate_nnodes
    validate_node_rank
    validate_gpus_per_node
    validate_master_addr
    validate_master_port

    LOG_SUCCESS_RANK0 "All distributed parameters validated successfully"
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
export -f validate_container_runtime validate_docker_image validate_mount_path
export -f validate_slurm_env validate_slurm_nodes
export -f validate_env_var validate_env_var_choices
export -f validate_python_script validate_bash_script

LOG_DEBUG_RANK0 "Primus validation library loaded successfully"
