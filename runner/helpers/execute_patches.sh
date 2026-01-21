#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Execute patch scripts
# Usage:
#   execute_patches <patch_script1> [patch_script2] ...
#
#   Or pass a single "patch list" string with one patch per line:
#     execute_patches "$PATCH_LIST"
#   (This is convenient when patches are stored in config as a newline-separated list.)
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
            echo "[INFO] $*" >&2
        fi
    }
    LOG_ERROR_RANK0() {
        if [ "${NODE_RANK:-0}" -eq 0 ]; then
            echo "[ERROR] $*" >&2
        fi
    }
    LOG_SUCCESS_RANK0() {
        if [ "${NODE_RANK:-0}" -eq 0 ]; then
            echo "[SUCCESS] $*"
        fi
    }
fi

# Global array to collect extra Primus CLI arguments emitted by patches via
# the "extra.*=value" protocol (same as hooks). Each match is converted to
# a pair: --<name> <value>.
PATCH_EXTRA_PRIMUS_ARGS=()

# Execute multiple patch scripts
# Args:
#   $@: Patch script paths
execute_patches() {
    if [[ $# -eq 0 ]]; then
        LOG_INFO_RANK0 "[Execute Patches] No patch scripts specified"
        return 0
    fi

    # Support both:
    # - multiple patch script args
    # - a single newline-separated list (or any arg containing newlines)
    local patch_scripts=()
    local arg
    for arg in "$@"; do
        while IFS= read -r patch_entry; do
            [[ -n "$patch_entry" ]] || continue
            patch_scripts+=("$patch_entry")
        done <<< "$arg"
    done

    if [[ ${#patch_scripts[@]} -eq 0 ]]; then
        LOG_INFO_RANK0 "[Execute Patches] No patch scripts specified"
        return 0
    fi

    LOG_INFO_RANK0 "[Execute Patches] Detected patch scripts: ${patch_scripts[*]}"

    for patch in "${patch_scripts[@]}"; do
        if [[ ! -f "$patch" ]]; then
            LOG_ERROR_RANK0 "[Execute Patches] Patch script not found: $patch"
            return 1
        fi

        if [[ ! -r "$patch" ]]; then
            LOG_ERROR_RANK0 "[Execute Patches] Patch script not readable: $patch"
            return 1
        fi

        LOG_INFO_RANK0 "[Execute Patches] Running patch: bash $patch"

        # Run the patch script in a child shell and capture its output so that we can
        # process special lines (e.g., env.* and extra.*) while still displaying
        # the output in real-time.
        local patch_output
        patch_output="$(bash "$patch" 2>&1 | tee /dev/stderr)"
        local exit_code=${PIPESTATUS[0]}

        # Allow patches to:
        #   1) Export environment variables back to the caller by printing lines:
        #        env.VAR_NAME=VALUE
        #      These will be exported into the current shell (e.g., primus-cli-direct.sh).
        #   2) Provide extra Primus CLI arguments using the same "extra.*" protocol
        #      as hooks, for example:
        #        extra.model.hf_assets_path=/path
        #      which becomes:
        #        --model.hf_assets_path /path
        while IFS= read -r line; do
            [[ -z "$line" ]] && continue

            if [[ "$line" =~ ^env\.([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
                local env_key="${BASH_REMATCH[1]}"
                local env_value="${BASH_REMATCH[2]}"
                export "$env_key"="$env_value"
                LOG_INFO_RANK0 "[Execute Patches] Exported from patch (env.*): $env_key=$env_value"
            elif [[ "$line" =~ ^extra\.([A-Za-z_][A-Za-z0-9_.]*[A-Za-z0-9_])=(.*)$ ]]; then
                local name="${BASH_REMATCH[1]}"
                local value="${BASH_REMATCH[2]}"
                PATCH_EXTRA_PRIMUS_ARGS+=("--${name}" "${value}")
                LOG_INFO_RANK0 "[Execute Patches] extra arg from patch: --${name} ${value}"
            fi
        done <<< "$patch_output"

        if [[ $exit_code -eq 0 ]]; then
            LOG_INFO_RANK0 "[Execute Patches] Patch completed successfully: $patch"
        elif [[ $exit_code -eq 2 ]]; then
            LOG_INFO_RANK0 "[Execute Patches] Patch skipped (exit code 2): $patch"
        else
            LOG_ERROR_RANK0 "[Execute Patches] Patch script failed: $patch (exit code: $exit_code)"
            return 1
        fi
    done

    LOG_SUCCESS_RANK0 "[Execute Patches] All patch scripts executed successfully"
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
    execute_patches "$@"
fi
