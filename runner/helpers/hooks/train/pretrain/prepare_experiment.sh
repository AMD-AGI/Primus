#!/bin/bash
###############################################################################
# Primus Pre-Train Backend Dispatcher
#
# This hook is executed by execute_hooks.sh before training. It:
#   1. Finds --config / --data_path / --patch_args / --backend_path from args
#   2. Detects the framework from the pre_trainer module in the config
#   3. Dispatches to runner/helpers/hooks/train/pretrain/<framework>/prepare.py
#      with a stable CLI:
#         --config <exp.yaml>
#         --data_path <data_root>
#         --primus_path <primus_root>
#         --patch_args <patch_args.txt>
#         [--backend_path <path>] [extra args...]
#
# Any extra.* / env.* lines printed by the framework-specific prepare.py are
# forwarded back to primus-cli direct by execute_hooks.sh.
###############################################################################
set -euo pipefail

# First two args are the hook group/name injected by execute_hooks.sh (e.g. train pretrain)
HOOK_GROUP="$1"
HOOK_NAME="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

CONFIG_FILE=""
DATA_PATH="./data/"
PATCH_ARGS="/tmp/primus_patch_args.txt"
BACKEND_PATH=""
EXTRA_ARGS=()

# Parse CLI args (after hook group/name)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --patch_args)
            PATCH_ARGS="$2"
            shift 2
            ;;
        --backend_path)
            BACKEND_PATH="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$CONFIG_FILE" ]]; then
    echo "[WARNING] prepare_experiment.sh: no --config provided, skipping framework dispatch" >&2
    exit 0
fi

# Normalize CONFIG_FILE to absolute path
cd "$PRIMUS_ROOT"
if [[ ! "$CONFIG_FILE" = /* ]]; then
    CONFIG_FILE="${PRIMUS_ROOT}/${CONFIG_FILE#./}"
fi

# Ensure patch args file exists (truncate if already there)
PATCH_DIR="$(dirname "$PATCH_ARGS")"
mkdir -p "$PATCH_DIR"
: > "$PATCH_ARGS"
echo "[INFO] prepare_experiment.sh: patch args file: $PATCH_ARGS" >&2

# Extract framework from config (pre_trainer module), mirroring dispatch_framework.sh
FRAMEWORK="$(python3 -c "
import sys
from pathlib import Path

print(f'[DEBUG] CONFIG_FILE: ${CONFIG_FILE}', file=sys.stderr)

sys.path.insert(0, '${PRIMUS_ROOT}')
from primus.core.config.primus_config import load_primus_config, get_module_config

cfg = load_primus_config(Path('${CONFIG_FILE}'), None)
print(f'[DEBUG] cfg type: {type(cfg)}', file=sys.stderr)

pre_trainer = get_module_config(cfg, 'pre_trainer')
print(f'[DEBUG] pre_trainer type: {type(pre_trainer)}', file=sys.stderr)

if pre_trainer is None:
    print('[DEBUG] pre_trainer is None', file=sys.stderr)
    sys.exit(1)

if not hasattr(pre_trainer, 'framework'):
    print('[DEBUG] pre_trainer has no framework attribute', file=sys.stderr)
    sys.exit(1)

print(f'[DEBUG] pre_trainer.framework: {pre_trainer.framework}', file=sys.stderr)

print(pre_trainer.framework)
" 2> >(tee /dev/stderr >&2) | tail -n 1 | tr -d '\r' || true)"

if [[ -z "$FRAMEWORK" ]]; then
    echo "[WARNING] prepare_experiment.sh: no framework found in pre_trainer, skipping dispatch" >&2
    exit 0
fi

# Normalize framework aliases (e.g. light-megatron → megatron)
case "$FRAMEWORK" in
    light-megatron) FRAMEWORK_DIR="megatron" ;;
    *)              FRAMEWORK_DIR="$FRAMEWORK" ;;
esac

FRAMEWORK_SCRIPT="${SCRIPT_DIR}/${FRAMEWORK_DIR}/prepare.py"

if [[ ! -f "$FRAMEWORK_SCRIPT" ]]; then
    echo "[WARNING] prepare_experiment.sh: backend prepare script not found: $FRAMEWORK_SCRIPT" >&2
    exit 0
fi

echo "[INFO] prepare_experiment.sh: framework=$FRAMEWORK script=$FRAMEWORK_SCRIPT" >&2

CMD=(python3 "$FRAMEWORK_SCRIPT"
     --config "$CONFIG_FILE"
     --data_path "$DATA_PATH"
     --primus_path "$PRIMUS_ROOT"
     --patch_args "$PATCH_ARGS")

if [[ -n "$BACKEND_PATH" ]]; then
    CMD+=(--backend_path "$BACKEND_PATH")
fi

CMD+=("${EXTRA_ARGS[@]}")

exec "${CMD[@]}"

