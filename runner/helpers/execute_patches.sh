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
# Exit codes from patch scripts:
#   0   - Success, continue to next patch
#   2   - Skip this patch (not an error)
#   other - Failure, stop execution
#
# Example patch script with conditional skip:
#   #!/bin/bash
#   # Check if patch is needed
#   if [[ -f /tmp/already_patched ]]; then
#       echo "Patch already applied, skipping"
#       exit 2  # Skip this patch
#   fi
#
#   # Apply patch
#   echo "Applying patch..."
#   # ... patch logic ...
#   exit 0  # Success
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

        bash "$patch"
        local exit_code=$?

        if [[ $exit_code -eq 0 ]]; then
            LOG_INFO_RANK0 "[Execute Patches] Patch completed successfully: $patch"
        elif [[ $exit_code -eq 2 ]]; then
            LOG_INFO_RANK0 "[Execute Patches] Patch skipped (exit code 2): $patch"
        else
            LOG_ERROR "[Execute Patches] Patch script failed: $patch (exit code: $exit_code)"
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
