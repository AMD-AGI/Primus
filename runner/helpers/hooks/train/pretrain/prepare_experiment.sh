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
BACKEND_OVERRIDE=""
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
        --backend)
            BACKEND_OVERRIDE="$2"
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

# Route priority:
#   1) --backend (explicit override)
#   2) pre_trainer.framework from --config
FRAMEWORK_FROM_CONFIG="$(python3 -c "
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

if [[ -n "$BACKEND_OVERRIDE" ]]; then
    FRAMEWORK="$BACKEND_OVERRIDE"
    if [[ -n "$FRAMEWORK_FROM_CONFIG" ]]; then
        if [[ "$BACKEND_OVERRIDE" != "$FRAMEWORK_FROM_CONFIG" ]]; then
            echo "[ERROR] prepare_experiment.sh: --backend=$BACKEND_OVERRIDE conflicts with framework=$FRAMEWORK_FROM_CONFIG from config" >&2
            exit 1
        fi
    fi
else
    FRAMEWORK="$FRAMEWORK_FROM_CONFIG"
fi

if [[ -z "$FRAMEWORK" ]]; then
    echo "[WARNING] prepare_experiment.sh: no framework found (provide --backend or --config with pre_trainer.framework), skipping dispatch" >&2
    exit 0
fi

# No legacy alias light-megatron mapping
FRAMEWORK_DIR="$FRAMEWORK"

FRAMEWORK_SCRIPT="${SCRIPT_DIR}/${FRAMEWORK_DIR}/prepare.py"
FRAMEWORK_HOOK_DIR="${SCRIPT_DIR}/${FRAMEWORK_DIR}"

# Execute framework-local shell hooks in lexical order before prepare.py.
# This is used for per-framework dependency installation
if [[ -d "$FRAMEWORK_HOOK_DIR" ]]; then
    mapfile -t SH_HOOK_FILES < <(find "$FRAMEWORK_HOOK_DIR" -maxdepth 1 -type f -name "*.sh" | sort)
    HOOK_ARGS=(--config "$CONFIG_FILE"
               --data_path "$DATA_PATH"
               --primus_path "$PRIMUS_ROOT"
               --patch_args "$PATCH_ARGS")
    if [[ -n "$BACKEND_PATH" ]]; then
        HOOK_ARGS+=(--backend_path "$BACKEND_PATH")
    fi
    if [[ -n "$BACKEND_OVERRIDE" ]]; then
        HOOK_ARGS+=(--backend "$BACKEND_OVERRIDE")
    fi
    HOOK_ARGS+=("${EXTRA_ARGS[@]}")
    exit_code=0
    set +e
    for hook_file in "${SH_HOOK_FILES[@]}"; do
        echo "[INFO] prepare_experiment.sh: executing shell hook $(basename "$hook_file")" >&2
        bash "$hook_file" "${HOOK_ARGS[@]}"
        exit_code=$?
        if [[ $exit_code -ne 0 ]]; then
            echo "[ERROR] prepare_experiment.sh: shell hook failed: $hook_file (exit code: $exit_code)" >&2
            exit $exit_code
        fi
    done
    set -e
fi

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
if [[ -n "$BACKEND_OVERRIDE" ]]; then
    CMD+=(--backend "$BACKEND_OVERRIDE")
fi

CMD+=("${EXTRA_ARGS[@]}")

# env sync for Megatron pretrain:
#   - PRIMUS_HIPBLASLT_TUNING
#   - PRIMUS_HIPBLASLT_TUNING_STAGE
#   - TE_HIPBLASLT_TUNING_*
#   - HIPBLASLT_*
#
# Emit env.* lines for execute_hooks.sh to export.
emit_env() {
    local k="$1"
    local v="${2:-}"
    echo "env.${k}=${v}"
}

if [[ "$FRAMEWORK_DIR" == "megatron" ]]; then
    # Deterministic mode: keep old behavior (disable hipblaslt tuning counters)
    if [[ "${PRIMUS_DETERMINISTIC:-0}" == "1" ]]; then
        emit_env "TE_HIPBLASLT_TUNING_RUN_COUNT" "0"
        emit_env "TE_HIPBLASLT_TUNING_ALGO_COUNT" "0"
    elif [[ "${PRIMUS_HIPBLASLT_TUNING:-0}" == "1" ]]; then
        STAGE="${PRIMUS_HIPBLASLT_TUNING_STAGE:-0}"

        # Try to derive a stable model tag for tune output dir
        MODEL_TAG="${PRIMUS_MODEL:-$(basename "${CONFIG_FILE}" .yaml)}"
        TUNE_ROOT="${PRIMUS_ROOT}/output/tune_hipblaslt/${MODEL_TAG}"
        RESULT_FILE="tune_hipblas_gemm_results.txt"
        NODE_RANK_VAL="${NODE_RANK:-0}"
        NUM_DEVICES="${GPUS_PER_NODE:-8}"

        case "$STAGE" in
            0)
                emit_env "TE_HIPBLASLT_TUNING_RUN_COUNT" "${TE_HIPBLASLT_TUNING_RUN_COUNT:-10}"
                emit_env "TE_HIPBLASLT_TUNING_ALGO_COUNT" "${TE_HIPBLASLT_TUNING_ALGO_COUNT:-50}"
                ;;
            1)
                mkdir -p "${TUNE_ROOT}/gemm_shape"
                emit_env "HIPBLASLT_LOG_MASK" "${HIPBLASLT_LOG_MASK:-32}"
                emit_env "HIPBLASLT_LOG_FILE" "${HIPBLASLT_LOG_FILE:-${TUNE_ROOT}/gemm_shape/dump_hipblaslt_gemm_shape_${NODE_RANK_VAL}.txt}"
                # Explicitly clear override in stage-1
                emit_env "HIPBLASLT_TUNING_OVERRIDE_FILE" ""
                ;;
            2)
                mkdir -p "${TUNE_ROOT}/gemm_tune"
                # Usually stage-2 is single-node. Guard by rank for safety.
                if [[ "${NODE_RANK_VAL}" == "0" ]]; then
                    python "${PRIMUS_ROOT}/examples/offline_tune/offline_tune_gemm.py" \
                        --dump-shape-path-or-file "${TUNE_ROOT}/gemm_shape" \
                        --tune-result-path "${TUNE_ROOT}/gemm_tune/${RESULT_FILE}" \
                        --num-devices "${NUM_DEVICES}"
                fi
                # Ask direct launcher to skip main training launch after tuning.
                emit_env "PRIMUS_SKIP_MAIN_LAUNCH" "1"
                ;;
            3)
                TUNE_FILE="${HIPBLASLT_TUNING_OVERRIDE_FILE:-${TUNE_ROOT}/gemm_tune/${RESULT_FILE}}"
                if [[ ! -f "$TUNE_FILE" ]]; then
                    echo "[ERROR] prepare_experiment.sh: Missing tuning result file: $TUNE_FILE" >&2
                    exit 1
                fi
                emit_env "HIPBLASLT_TUNING_OVERRIDE_FILE" "$TUNE_FILE"
                ;;
            *)
                echo "[ERROR] prepare_experiment.sh: Invalid PRIMUS_HIPBLASLT_TUNING_STAGE=$STAGE (expected 0/1/2/3)" >&2
                exit 1
                ;;
        esac
    else
        # Default: tuning disabled
        emit_env "TE_HIPBLASLT_TUNING_RUN_COUNT" "0"
        emit_env "TE_HIPBLASLT_TUNING_ALGO_COUNT" "0"
    fi
fi

exec "${CMD[@]}"

