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

# Global array to collect extra Primus CLI arguments emitted by hooks via
# the "extra.*=value" protocol. Each match is converted to a pair:
#   --<name> <value>
HOOK_EXTRA_PRIMUS_ARGS=()

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

    local global_hook_dir="${script_dir}/hooks"
    local hook_dir="${script_dir}/hooks/${hook_group}/${hook_name}"

    _run_hook_dir() {
        local dir="$1"
        shift
        local args=("$@")

        if [[ ! -d "$dir" ]]; then
            return 0
        fi

        LOG_INFO_RANK0 "[Hooks] Detected hooks directory: $dir"

        # Find all hook files (*.sh and *.py) in this directory only.
        local hook_files=()
        mapfile -t hook_files < <(find "$dir" -maxdepth 1 -type f \( -name "*.sh" -o -name "*.py" \) | sort)

        if [[ ${#hook_files[@]} -eq 0 ]]; then
            LOG_INFO_RANK0 "[Hooks] No hook files found in $dir"
            return 0
        fi

        # Execute each hook file
        for hook_file in "${hook_files[@]}"; do
            LOG_INFO_RANK0 "[Hooks] Executing hook: $hook_file ${args[*]}"

            start_time=$(date +%s)

            local hook_output
            local exit_code

            if [[ "$hook_file" == *.sh ]]; then
                hook_output="$(bash "$hook_file" "${args[@]}" 2>&1)"
                exit_code=$?
            elif [[ "$hook_file" == *.py ]]; then
                hook_output="$(python3 "$hook_file" "${args[@]}" 2>&1)"
                exit_code=$?
            else
                LOG_WARN "[Hooks] Skipping unknown hook type: $hook_file"
                continue
            fi

            # Re-echo hook output so users see logs as usual.
            if [[ -n "$hook_output" ]]; then
                printf '%s\n' "$hook_output"
            fi

            # Parse hook output for extra.* and env.* key=value pairs.
            while IFS= read -r line; do
                [[ -z "$line" ]] && continue

                if [[ "$line" =~ ^extra\.([A-Za-z_][A-Za-z0-9_.]*[A-Za-z0-9_])=(.*)$ ]]; then
                    local name="${BASH_REMATCH[1]}"
                    local value="${BASH_REMATCH[2]}"
                    HOOK_EXTRA_PRIMUS_ARGS+=("--${name}" "${value}")
                    LOG_INFO_RANK0 "[Hooks] extra arg from hook: --${name} ${value}"
                elif [[ "$line" =~ ^env\.([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
                    local env_name="${BASH_REMATCH[1]}"
                    local env_value="${BASH_REMATCH[2]}"
                    export "${env_name}"="${env_value}"
                    LOG_INFO_RANK0 "[Hooks] exported env from hook: ${env_name}=${env_value}"
                fi
            done <<< "$hook_output"

            if [[ $exit_code -ne 0 ]]; then
                LOG_ERROR_RANK0 "[Hooks] Hook failed: $hook_file (exit code: $exit_code)"
                return 1
            fi

            duration=$(( $(date +%s) - start_time ))
            LOG_INFO_RANK0 "[Hooks] Hook $hook_file finished in ${duration}s"
        done
        return 0
    }

    # 1) Run system/global hooks (runner/helpers/hooks/*.sh|*.py)
    if ! _run_hook_dir "$global_hook_dir" "${hook_args[@]}"; then
        return 1
    fi

    # 2) Run command-specific hooks (runner/helpers/hooks/<group>/<name>/*.sh|*.py)
    if [[ ! -d "$hook_dir" ]]; then
        LOG_INFO_RANK0 "[Hooks] No hook directory for [$hook_group/$hook_name]"
        LOG_INFO_RANK0 "[Hooks] All hooks executed successfully"
        return 0
    fi
    if ! _run_hook_dir "$hook_dir" "${hook_args[@]}"; then
        return 1
    fi

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
