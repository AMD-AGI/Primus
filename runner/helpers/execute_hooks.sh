#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Execute hooks based on command group and name
# Usage: execute_hooks <hook_group> <hook_name> [args...]
#

# Requires common.sh to be sourced
if [[ -z "${__PRIMUS_COMMON_SOURCED:-}" ]]; then
    # Fallback logging if common.sh not loaded
    LOG_INFO_RANK0() {
        if [ "${NODE_RANK:-0}" -eq 0 ]; then
            echo "[INFO] $*"
        fi
    }
    LOG_ERROR_RANK0() {
        if [ "${NODE_RANK:-0}" -eq 0 ]; then
            echo "[ERROR] $*" >&2
        fi
    }
    LOG_WARN() {
        if [ "${NODE_RANK:-0}" -eq 0 ]; then
            echo "[WARN] $*" >&2
        fi
    }
fi

# Execute hooks for a given command
# Args:
#   $1: hook_group (e.g., "train", "benchmark")
#   $2: hook_name (e.g., "pretrain", "gemm")
#   $@: Additional arguments to pass to hooks
execute_hooks() {
    if [[ $# -lt 2 ]]; then
        LOG_INFO_RANK0 "[Hooks] No hook target specified (need group and name)"
        return 0
    fi

    local hook_group="$1"
    local hook_name="$2"
    shift 2
    local hook_args=("$@")

    # Determine script directory
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    local hook_dir="${script_dir}/hooks/${hook_group}/${hook_name}"

    if [[ ! -d "$hook_dir" ]]; then
        LOG_INFO_RANK0 "[Hooks] No hook directory for [$hook_group/$hook_name]"
        return 0
    fi

    LOG_INFO_RANK0 "[Hooks] Detected hooks directory: $hook_dir"

    # Find all hook files (*.sh and *.py)
    local hook_files=()
    mapfile -t hook_files < <(find "$hook_dir" -maxdepth 1 -type f \( -name "*.sh" -o -name "*.py" \) | sort)

    if [[ ${#hook_files[@]} -eq 0 ]]; then
        LOG_INFO_RANK0 "[Hooks] No hook files found in $hook_dir"
        return 0
    fi

    # Execute each hook file
    for hook_file in "${hook_files[@]}"; do
        LOG_INFO_RANK0 "[Hooks] Executing hook: $hook_file ${hook_args[*]}"

        start_time=$(date +%s)

        if [[ "$hook_file" == *.sh ]]; then
            if ! bash "$hook_file" "${hook_args[@]}"; then
                LOG_ERROR_RANK0 "[Hooks] Hook failed: $hook_file (exit code: $?)"
                return 1
            fi
        elif [[ "$hook_file" == *.py ]]; then
            if ! python3 "$hook_file" "${hook_args[@]}"; then
                LOG_ERROR_RANK0 "[Hooks] Hook failed: $hook_file (exit code: $?)"
                return 1
            fi
        else
            LOG_WARN "[Hooks] Skipping unknown hook type: $hook_file"
        fi

        duration=$(( $(date +%s) - start_time ))
        LOG_INFO_RANK0 "[Hooks] Hook $hook_file finished in ${duration}s"
    done

    LOG_INFO_RANK0 "[Hooks] All hooks executed successfully"
    return 0
}

# If called directly (not sourced), execute the function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Always source common.sh when called directly, as functions are not inherited by subshells
    # Unset the guard variable to force re-sourcing in this new shell instance
    unset __PRIMUS_COMMON_SOURCED
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/../lib/common.sh" || {
        echo "[ERROR] Failed to load common library" >&2
        exit 1
    }
    execute_hooks "$@"
fi
