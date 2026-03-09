#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
set -euo pipefail

# Framework Dispatcher for post-training hooks
# This script detects the framework from config and dispatches to framework-specific hooks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../../../../lib/common.sh" || {
    echo "[ERROR] Failed to load common library" >&2
    exit 1
}

LOG_INFO_RANK0 "[+] Dispatching to framework-specific hooks..."

# Find --config argument from command line
CONFIG_FILE=""
for ((i=1; i<=$#; i++)); do
    if [[ "${!i}" == "--config" ]]; then
        j=$((i+1))
        CONFIG_FILE="${!j}"
        break
    fi
done

if [[ -z "$CONFIG_FILE" ]]; then
    # ANSI color codes
    YELLOW='\033[1;33m'
    RESET='\033[0m'
    echo -e "${YELLOW}[WARNING] No --config argument found, skipping framework dispatch${RESET}"
    exit 0
fi

# Determine script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "$PRIMUS_ROOT"

# Convert CONFIG_FILE to absolute path
if [[ ! "$CONFIG_FILE" = /* ]]; then
    CONFIG_FILE="${PRIMUS_ROOT}/${CONFIG_FILE#./}"
fi

LOG_INFO_RANK0 "CONFIG_FILE: ${CONFIG_FILE}"

# Extract framework from config
FRAMEWORK="$(python3 -c "
import sys

# Debug output to stderr so it doesn't interfere with stdout capture
print(f'[DEBUG] CONFIG_FILE: ${CONFIG_FILE}', file=sys.stderr)

sys.path.insert(0, '${PRIMUS_ROOT}')
from pathlib import Path
from primus.core.config.primus_config import load_primus_config, get_module_config

# Load config using the same method as train_runtime.py
cfg = load_primus_config(Path('${CONFIG_FILE}'), None)
print(f'[DEBUG] cfg type: {type(cfg)}', file=sys.stderr)

# Get post_trainer module config
post_trainer = get_module_config(cfg, 'post_trainer')
print(f'[DEBUG] post_trainer type: {type(post_trainer)}', file=sys.stderr)

if post_trainer is None:
    print('[DEBUG] post_trainer is None', file=sys.stderr)
    sys.exit(1)

if not hasattr(post_trainer, 'framework'):
    print('[DEBUG] post_trainer has no framework attribute', file=sys.stderr)
    sys.exit(1)

print(f'[DEBUG] post_trainer.framework: {post_trainer.framework}', file=sys.stderr)

print(post_trainer.framework)
" 2> >(tee /dev/stderr >&2) | tail -n 1 | tr -d '\r' || true)"

if [[ -z "$FRAMEWORK" ]]; then
    LOG_ERROR_RANK0 "[WARNING] No framework found in config"
    LOG_ERROR_RANK0 "[WARNING] Skipping framework-specific hooks"
    exit 0
fi

# Determine framework hook directory
FRAMEWORK_HOOK_DIR="${SCRIPT_DIR}/${FRAMEWORK}"

if [[ ! -d "$FRAMEWORK_HOOK_DIR" ]]; then
    LOG_ERROR_RANK0 "[WARNING] Framework directory not found: ${FRAMEWORK_HOOK_DIR}"
    LOG_ERROR_RANK0 "[WARNING] Skipping framework-specific hooks"
    exit 0
fi

LOG_INFO_RANK0 "Framework: ${FRAMEWORK}"
LOG_INFO_RANK0 "Framework hook directory: ${FRAMEWORK_HOOK_DIR}"

# Find all hook files (*.sh and *.py) in framework directory
HOOK_FILES=()
mapfile -t HOOK_FILES < <(find "$FRAMEWORK_HOOK_DIR" -maxdepth 1 -type f \( -name "*.sh" -o -name "*.py" \) | sort)

if [[ ${#HOOK_FILES[@]} -eq 0 ]]; then
    LOG_ERROR_RANK0 "No hook files found in ${FRAMEWORK_HOOK_DIR}"
    exit 0
fi

LOG_INFO_RANK0 "Found ${#HOOK_FILES[@]} hook file(s) to execute"

# Execute each hook file
for hook_file in "${HOOK_FILES[@]}"; do
    LOG_INFO_RANK0 "[+] Executing framework hook: $(basename "$hook_file")"

    start_time=$(date +%s)

    hook_output=""
    exit_code=0

    if [[ "$hook_file" == *.sh ]]; then
        # Execute bash script and capture output
        hook_output="$(bash "$hook_file" "$@" 2>&1 | tee /dev/stderr)"
        exit_code=${PIPESTATUS[0]}
    elif [[ "$hook_file" == *.py ]]; then
        # Execute python script and capture output
        hook_output="$(python3 "$hook_file" "$@" 2>&1 | tee /dev/stderr)"
        exit_code=${PIPESTATUS[0]}
    else
        LOG_ERROR_RANK0 "[WARNING] Skipping unknown hook type: $hook_file"
        continue
    fi

    # Parse hook output for extra.* lines (for passing args to main process)
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue

        if [[ "$line" =~ ^extra\.([A-Za-z_][A-Za-z0-9_.]*[A-Za-z0-9_])=(.*)$ ]]; then
            # Forward extra.* lines to stdout so parent can capture them
            echo "$line"
        elif [[ "$line" =~ ^env\.([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
            # Forward env.* lines to stdout so parent can capture them
            echo "$line"
        fi
    done <<< "$hook_output"

    if [[ $exit_code -ne 0 ]]; then
        LOG_ERROR_RANK0 "[ERROR] Framework hook failed: $hook_file (exit code: $exit_code)"
        exit 1
    fi

    duration=$(( $(date +%s) - start_time ))
    LOG_SUCCESS_RANK0 "Framework hook $(basename "$hook_file") finished in ${duration}s"
done

LOG_SUCCESS_RANK0 "All framework-specific hooks executed successfully"
