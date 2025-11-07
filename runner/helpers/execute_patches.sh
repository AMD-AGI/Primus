#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Execute patch scripts
# Usage: execute_patches <patch_script1> [patch_script2] ...
#

# Requires common.sh to be sourced
if [[ -z "${__PRIMUS_COMMON_SOURCED:-}" ]]; then
    # Fallback logging if common.sh not loaded
    LOG_INFO_RANK0() {
        if [ "${NODE_RANK:-0}" -eq 0 ]; then
            echo "[INFO] $*"
        fi
    }
    LOG_WARN() {
        echo "[WARN] $*" >&2
    }
    LOG_ERROR() {
        echo "[ERROR] $*" >&2
    }
fi

# Execute multiple patch scripts
# Args:
#   $@: Patch script paths
execute_patches() {
    if [[ $# -eq 0 ]]; then
        LOG_INFO_RANK0 "[Execute Patches] No patch scripts specified"
        return 0
    fi

    local patch_scripts=("$@")

    LOG_INFO_RANK0 "[Execute Patches] Detected patch scripts: ${patch_scripts[*]}"

    for patch in "${patch_scripts[@]}"; do
        if [[ ! -f "$patch" ]]; then
            LOG_WARN "[Execute Patches] Patch script not found: $patch (skipping)"
            continue
        fi

        if [[ ! -r "$patch" ]]; then
            LOG_WARN "[Execute Patches] Patch script not readable: $patch (skipping)"
            continue
        fi

        LOG_INFO_RANK0 "[Execute Patches] Running patch: bash $patch"

        if ! bash "$patch"; then
            LOG_ERROR "[Execute Patches] Patch script failed: $patch (exit code: $?)"
            return 1
        fi
    done

    LOG_INFO_RANK0 "[Execute Patches] All patch scripts executed successfully"
    return 0
}

# If called directly (not sourced), execute the function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    execute_patches "$@"
fi
