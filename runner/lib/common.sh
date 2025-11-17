#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Common functions library for Primus CLI
# Source this file in other scripts: source "${SCRIPT_DIR}/lib/common.sh"
#

# ---------------------------------------------------------------------------
# Guard: avoid duplicate sourcing
# ---------------------------------------------------------------------------
if [[ -n "${__PRIMUS_COMMON_SOURCED:-}" ]]; then
  return 0
fi
export __PRIMUS_COMMON_SOURCED=1

# ---------------------------------------------------------------------------
# Configuration and State
# ---------------------------------------------------------------------------
export PRIMUS_LOG_LEVEL="${PRIMUS_LOG_LEVEL:-INFO}"  # DEBUG, INFO, WARN, ERROR
export PRIMUS_LOG_TIMESTAMP="${PRIMUS_LOG_TIMESTAMP:-1}"  # 0=off, 1=on
export PRIMUS_LOG_COLOR="${PRIMUS_LOG_COLOR:-1}"  # 0=off, 1=on (auto-detect TTY)

# Auto-detect color support
if [[ "$PRIMUS_LOG_COLOR" == "1" ]] && [[ ! -t 1 ]]; then
    PRIMUS_LOG_COLOR=0
fi

# Color definitions (only if enabled)
if [[ "$PRIMUS_LOG_COLOR" == "1" ]]; then
    export COLOR_RESET='\033[0m'
    export COLOR_RED='\033[0;31m'
    export COLOR_GREEN='\033[0;32m'
    export COLOR_YELLOW='\033[0;33m'
    export COLOR_BLUE='\033[0;34m'
    export COLOR_CYAN='\033[0;36m'
    export COLOR_GRAY='\033[0;90m'
else
    export COLOR_RESET=''
    export COLOR_RED=''
    export COLOR_GREEN=''
    export COLOR_YELLOW=''
    export COLOR_BLUE=''
    export COLOR_CYAN=''
    export COLOR_GRAY=''
fi

# ---------------------------------------------------------------------------
# Logging Functions
# ---------------------------------------------------------------------------

# Get timestamp for logging
_get_timestamp() {
    if [[ "$PRIMUS_LOG_TIMESTAMP" == "1" ]]; then
        date +'%Y-%m-%d %H:%M:%S'
    fi
}

# Get node identifier for logging
_get_node_id() {
    local node_rank="${NODE_RANK:-?}"
    local hostname="${HOSTNAME:-$(hostname)}"
    echo "NODE-${node_rank}(${hostname})"
}

# Internal logging function
_log() {
    local level="$1"
    local color="$2"
    shift 2
    local message="$*"

    local timestamp=""
    if [[ "$PRIMUS_LOG_TIMESTAMP" == "1" ]]; then
        timestamp="[$(date +'%Y-%m-%d %H:%M:%S')] "
    fi

    local node_id
    node_id="[$(_get_node_id)]"

    if [[ -n "$message" ]]; then
        echo -e "${color}${timestamp}${node_id} [${level}]${COLOR_RESET} ${message}"
    else
        echo ""
    fi
}

# Log level functions
LOG_DEBUG() {
    if [[ "$PRIMUS_LOG_LEVEL" == "DEBUG" ]]; then
        _log "DEBUG" "$COLOR_GRAY" "$@"
    fi
}

LOG_INFO() {
    case "$PRIMUS_LOG_LEVEL" in
        DEBUG|INFO)
            _log "INFO" "$COLOR_BLUE" "$@"
            ;;
    esac
}

LOG_WARN() {
    case "$PRIMUS_LOG_LEVEL" in
        DEBUG|INFO|WARN)
            _log "WARN" "$COLOR_YELLOW" "$@" >&2
            ;;
    esac
}

LOG_ERROR() {
    _log "ERROR" "$COLOR_RED" "$@" >&2
}

LOG_SUCCESS() {
    case "$PRIMUS_LOG_LEVEL" in
        DEBUG|INFO)
            _log "SUCCESS" "$COLOR_GREEN" "$@"
            ;;
    esac
}

# Log only on rank 0
LOG_INFO_RANK0() {
    if [[ "${NODE_RANK:-0}" == "0" ]]; then
        LOG_INFO "$@"
    fi
}

LOG_DEBUG_RANK0() {
    if [[ "${NODE_RANK:-0}" == "0" ]]; then
        LOG_DEBUG "$@"
    fi
}

LOG_SUCCESS_RANK0() {
    if [[ "${NODE_RANK:-0}" == "0" ]]; then
        LOG_SUCCESS "$@"
    fi
}

LOG_ERROR_RANK0() {
    if [[ "${NODE_RANK:-0}" == "0" ]]; then
        LOG_ERROR "$@"
    fi
}

# ---------------------------------------------------------------------------
# Simple Print Functions (without timestamps and prefixes)
# ---------------------------------------------------------------------------

# Simple print functions - no timestamp, no node info, no level prefix
# Useful for formatted output like tables, config sections, etc.

PRINT_DEBUG() {
    if [[ "$PRIMUS_LOG_LEVEL" == "DEBUG" ]]; then
        echo "$*"
    fi
}

PRINT_INFO() {
    case "$PRIMUS_LOG_LEVEL" in
        DEBUG|INFO)
            echo "$*"
            ;;
    esac
}

PRINT_WARN() {
    case "$PRIMUS_LOG_LEVEL" in
        DEBUG|INFO|WARN)
            echo "$*" >&2
            ;;
    esac
}

PRINT_ERROR() {
    echo "$*" >&2
}

PRINT_SUCCESS() {
    case "$PRIMUS_LOG_LEVEL" in
        DEBUG|INFO)
            echo "$*"
            ;;
    esac
}

# Simple print only on rank 0
PRINT_INFO_RANK0() {
    if [[ "${NODE_RANK:-0}" == "0" ]]; then
        PRINT_INFO "$@"
    fi
}

PRINT_DEBUG_RANK0() {
    if [[ "${NODE_RANK:-0}" == "0" ]]; then
        PRINT_DEBUG "$@"
    fi
}

PRINT_SUCCESS_RANK0() {
    if [[ "${NODE_RANK:-0}" == "0" ]]; then
        PRINT_SUCCESS "$@"
    fi
}

# Log exported environment variables in a formatted way
log_exported_vars() {
    local title="$1"
    shift
    LOG_INFO_RANK0 "========== $title =========="
    for var in "$@"; do
        LOG_INFO_RANK0 "    $var=${!var:-<unset>}"
    done
}

# Print a section header (for formatted output)
print_section() {
    local title="$1"
    PRINT_INFO_RANK0 ""
    PRINT_INFO_RANK0 "=========================================="
    PRINT_INFO_RANK0 "  $title"
    PRINT_INFO_RANK0 "=========================================="
}

# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------

# Exit with error message
die() {
    LOG_ERROR "$@"
    exit 1
}

# Check if command exists
require_command() {
    local cmd="$1"
    local hint="${2:-}"

    LOG_DEBUG "Checking for required command: $cmd"

    if ! command -v "$cmd" &> /dev/null; then
        local msg="Required command not found: $cmd"
        if [[ -n "$hint" ]]; then
            msg="$msg. $hint"
        fi
        die "$msg"
    fi

    LOG_DEBUG "Required command found: $cmd"
}

# Check if file exists
require_file() {
    local file="$1"
    local msg="${2:-File not found: $file}"

    if [[ ! -f "$file" ]]; then
        die "$msg"
    fi
}

# Check if directory exists
require_dir() {
    local dir="$1"
    local msg="${2:-Directory not found: $dir}"

    if [[ ! -d "$dir" ]]; then
        die "$msg"
    fi
}

# Run command with error checking
run_cmd() {
    local cmd="$*"
    LOG_DEBUG "Executing: $cmd"

    if ! eval "$cmd"; then
        local exit_code=$?
        die "Command failed (exit code: $exit_code): $cmd"
    fi
}

# Run command and capture output
run_cmd_capture() {
    local cmd="$*"
    LOG_DEBUG "Executing: $cmd"

    local output
    if ! output=$(eval "$cmd" 2>&1); then
        local exit_code=$?
        LOG_ERROR "Command failed (exit code: $exit_code): $cmd"
        LOG_ERROR "Output: $output"
        return $exit_code
    fi
    echo "$output"
}

# ---------------------------------------------------------------------------
# Path and File Utilities
# ---------------------------------------------------------------------------

# Get absolute path (handles symlinks)
get_absolute_path() {
    local path="$1"
    local result
    result=$(realpath -m "$path" 2>/dev/null || readlink -f "$path" 2>/dev/null || echo "$path")
    echo "$result"
}

# Get script directory (handles symlinks)
get_script_dir() {
    local script="${BASH_SOURCE[1]:-$0}"
    cd "$(dirname "$(realpath "$script")")" && pwd
}

# Ensure directory exists, create if needed
ensure_dir() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        LOG_DEBUG "Creating directory: $dir"
        mkdir -p "$dir" || die "Failed to create directory: $dir"
    fi
}

# Clean up temporary files/directories
cleanup_temp() {
    local path="$1"
    if [[ -n "$path" ]] && [[ -e "$path" ]]; then
        LOG_DEBUG "Cleaning up: $path"
        rm -rf "$path" 2>/dev/null || LOG_WARN "Failed to clean up: $path"
    fi
}

# ---------------------------------------------------------------------------
# Environment Variable Utilities
# ---------------------------------------------------------------------------

# Export variable and log it (rank 0 only)
export_and_log() {
    local key="$1"
    local value="$2"
    export "$key"="$value"
    LOG_DEBUG_RANK0 "Exported: $key=$value"
}

# Set default value for variable if not already set
set_default() {
    local key="$1"
    local default_value="$2"

    if [[ -z "${!key:-}" ]]; then
        export "$key"="$default_value"
        LOG_DEBUG "Set default: $key=$default_value"
    fi
}

# Load environment file (.env format)
load_env_file() {
    local env_file="$1"

    if [[ ! -f "$env_file" ]]; then
        LOG_WARN "Environment file not found: $env_file"
        return 1
    fi

    LOG_INFO "Loading environment file: $env_file"
    LOG_DEBUG "Parsing environment file: $env_file"

    local count=0
    # Read file line by line, skip comments and empty lines
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip comments and empty lines
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// /}" ]] && continue

        # Parse KEY=VALUE
        if [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
            local key="${BASH_REMATCH[1]}"
            local value="${BASH_REMATCH[2]}"
            # Remove quotes if present
            value="${value%\"}"
            value="${value#\"}"
            value="${value%\'}"
            value="${value#\'}"
            export "$key"="$value"
            ((count++))
            LOG_DEBUG "Loaded: $key=$value"
        fi
    done < "$env_file"

    LOG_DEBUG "Loaded $count environment variables from: $env_file"
}

# ---------------------------------------------------------------------------
# String Utilities
# ---------------------------------------------------------------------------

# Check if string contains substring
contains() {
    local string="$1"
    local substring="$2"
    [[ "$string" == *"$substring"* ]]
}

# Trim whitespace from string
trim() {
    local var="$*"
    var="${var#"${var%%[![:space:]]*}"}"
    var="${var%"${var##*[![:space:]]}"}"
    echo "$var"
}

# Join array elements with delimiter
join_by() {
    local delimiter="$1"
    shift
    local first="$1"
    shift
    printf %s "$first" "${@/#/$delimiter}"
}

# ---------------------------------------------------------------------------
# Process and System Utilities
# ---------------------------------------------------------------------------

# Get number of available CPU cores
get_cpu_count() {
    nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "1"
}

# Get total memory in GB
get_memory_gb() {
    local mem_kb
    mem_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    echo $((mem_kb / 1024 / 1024))
}

# Check if running in container
is_container() {
    [[ -f /.dockerenv ]] || [[ -f /run/.containerenv ]] || grep -q docker /proc/1/cgroup 2>/dev/null
}

# Check if running in Slurm job
is_slurm_job() {
    [[ -n "${SLURM_JOB_ID:-}" ]] || [[ -n "${SLURM_JOBID:-}" ]]
}

# ---------------------------------------------------------------------------
# Argument Parsing Utilities
# ---------------------------------------------------------------------------

# Parse KEY=VALUE argument
parse_key_value() {
    local arg="$1"
    if [[ "$arg" == *=* ]]; then
        local key="${arg%%=*}"
        local value="${arg#*=}"
        echo "$key" "$value"
        return 0
    else
        return 1
    fi
}

# Check if argument is a flag (starts with -)
is_flag() {
    [[ "$1" == -* ]]
}

# ---------------------------------------------------------------------------
# Version and Information
# ---------------------------------------------------------------------------

# Get Primus CLI version
get_primus_version() {
    local version_file
    version_file="$(get_script_dir)/../VERSION"
    if [[ -f "$version_file" ]]; then
        cat "$version_file"
    else
        echo "unknown"
    fi
}

# Print system information
print_system_info() {
    PRINT_DEBUG "Gathering system information..."
    PRINT_INFO_RANK0 "========== System Information =========="
    PRINT_INFO_RANK0 "    Hostname: $(hostname)"
    PRINT_INFO_RANK0 "    Kernel: $(uname -r)"
    PRINT_INFO_RANK0 "    CPUs: $(get_cpu_count)"
    PRINT_INFO_RANK0 "    Memory: $(get_memory_gb) GB"
    PRINT_INFO_RANK0 "    Container: $(is_container && echo 'Yes' || echo 'No')"
    PRINT_INFO_RANK0 "    Slurm Job: $(is_slurm_job && echo 'Yes' || echo 'No')"
    if command -v rocm-smi &>/dev/null; then
        local gpu_count
        gpu_count=$(rocm-smi --showid | grep -c "GPU" || echo "0")
        PRINT_INFO_RANK0 "    GPUs: $gpu_count"
        PRINT_DEBUG "ROCm SMI available, GPU count: $gpu_count"
    else
        PRINT_DEBUG "ROCm SMI not available"
    fi
}

# ---------------------------------------------------------------------------
# Trap and Cleanup
# ---------------------------------------------------------------------------

# Cleanup function to be called on exit
PRIMUS_CLEANUP_HOOKS=()

# Register cleanup hook
register_cleanup_hook() {
    PRIMUS_CLEANUP_HOOKS+=("$1")
    LOG_DEBUG "Registered cleanup hook: $1"
}

# Execute all cleanup hooks
run_cleanup_hooks() {
    local exit_code=$?
    LOG_DEBUG "Running ${#PRIMUS_CLEANUP_HOOKS[@]} cleanup hooks (exit code: $exit_code)"
    for hook in "${PRIMUS_CLEANUP_HOOKS[@]}"; do
        LOG_DEBUG "Running cleanup hook: $hook"
        eval "$hook" || LOG_WARN "Cleanup hook failed: $hook"
    done
    exit $exit_code
}

# Set trap for cleanup on exit
trap run_cleanup_hooks EXIT INT TERM

# ---------------------------------------------------------------------------
# Export all functions
# ---------------------------------------------------------------------------
export -f _get_timestamp _get_node_id _log
export -f LOG_DEBUG LOG_INFO LOG_WARN LOG_ERROR LOG_SUCCESS
export -f LOG_INFO_RANK0 LOG_DEBUG_RANK0 LOG_SUCCESS_RANK0 LOG_ERROR_RANK0
export -f PRINT_DEBUG PRINT_INFO PRINT_WARN PRINT_ERROR PRINT_SUCCESS
export -f PRINT_INFO_RANK0 PRINT_DEBUG_RANK0 PRINT_SUCCESS_RANK0
export -f log_exported_vars print_section
export -f die require_command require_file require_dir run_cmd run_cmd_capture
export -f get_absolute_path get_script_dir ensure_dir cleanup_temp
export -f export_and_log set_default load_env_file
export -f contains trim join_by
export -f get_cpu_count get_memory_gb is_container is_slurm_job
export -f parse_key_value is_flag
export -f get_primus_version print_system_info
export -f register_cleanup_hook run_cleanup_hooks

LOG_DEBUG_RANK0 "Primus common library loaded successfully"
