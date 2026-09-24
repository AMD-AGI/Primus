#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# End-to-end convergence test driver: build a real dataset, lint the config,
# train, then plot the loss curve.
#
#   tools/convergence_test/run_convergence_test.sh --model llama3.2_1B
#   tools/convergence_test/run_convergence_test.sh --model mixtral_8x7B_v0.1 \
#       --probe 20 --budget-hours 3
#
# Run --help for the full option list.
#
###############################################################################

set -euo pipefail

TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_PATH="$(cd "${TOOL_DIR}/../.." && pwd)"

CONFIG=""
MODEL=""
TRAIN_ITERS=""
PROBE=""
BUDGET_HOURS=""
DETERMINISTIC=0
SKIP_PREPARE=0
PREPARE_ONLY=0
PLOT_ONLY=0
STRICT=0
IMAGE=""
DATA_DIR=""
EXTRA=()

usage() {
cat <<EOF
Usage: $(basename "$0") [options] [-- extra primus-cli args]

Selecting what to run:
  --model <name>        Use a bundled config from ${TOOL_DIR}/configs
                        (llama3.2_1B, mixtral_8x7B_v0.1, ...)
  --config <file>       Use an arbitrary Primus experiment YAML
  --train-iters <N>     Override train_iters (also fixes up lr_decay_iters)
  --probe <N>           Run only N iterations to measure throughput, then stop
  --budget-hours <H>    With --probe, report the train_iters that fit in H hours

Dataset:
  --data-dir <path>     Dataset directory (default: \$DATA_PATH/convergence/<tag>)
  --skip-prepare        Assume the dataset already exists
  --prepare-only        Build the dataset and exit

Other:
  --deterministic       Export PRIMUS_DETERMINISTIC=1 for a repeatable run
  --image <image>       Docker image (default: runner/.primus.yaml)
  --strict              Fail on lint warnings as well as errors
  --plot-only           Re-plot the last run of an existing experiment
  -h, --help            This message

Anything after -- is forwarded to primus-cli.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)          MODEL="$2"; shift 2;;
        --config)         CONFIG="$2"; shift 2;;
        --train-iters)    TRAIN_ITERS="$2"; shift 2;;
        --probe)          PROBE="$2"; shift 2;;
        --budget-hours)   BUDGET_HOURS="$2"; shift 2;;
        --data-dir)       DATA_DIR="$2"; shift 2;;
        --skip-prepare)   SKIP_PREPARE=1; shift;;
        --prepare-only)   PREPARE_ONLY=1; shift;;
        --deterministic)  DETERMINISTIC=1; shift;;
        --image)          IMAGE="$2"; shift 2;;
        --strict)         STRICT=1; shift;;
        --plot-only)      PLOT_ONLY=1; shift;;
        -h|--help)        usage; exit 0;;
        --)               shift; EXTRA=("$@"); break;;
        *)                echo "unknown option: $1" >&2; usage; exit 1;;
    esac
done

if [[ -n "${MODEL}" && -z "${CONFIG}" ]]; then
    CONFIG="${TOOL_DIR}/configs/${MODEL}-convergence.yaml"
fi
if [[ -z "${CONFIG}" ]]; then
    echo "error: one of --model or --config is required" >&2
    echo "available bundled models:" >&2
    for f in "${TOOL_DIR}"/configs/*-convergence.yaml; do
        [[ -e "$f" ]] && echo "  $(basename "${f%-convergence.yaml}")" >&2
    done
    exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "error: config not found: ${CONFIG}" >&2
    exit 1
fi

cd "${PRIMUS_PATH}"

# ---------------------------------------------------------------------------
# Resolve the fields we need out of the fully merged config (module preset +
# model preset + overrides), exactly as the trainer would see them.
# ---------------------------------------------------------------------------
PLAN=$(CONFIG="${CONFIG}" python3 - <<'PY'
import os, sys
from types import SimpleNamespace
sys.path.insert(0, os.getcwd())
from primus.core.launcher.parser import PrimusParser

import os.path

cfg_path = os.environ["CONFIG"]
exp = PrimusParser().parse(SimpleNamespace(config=cfg_path))
c = exp.get_module_config("pre_trainer")
print(f"EXP_DIR={exp.exp_root_path}")
print(f"EXP_NAME={os.path.basename(exp.exp_root_path.rstrip('/'))}")
print(f"MODEL_PRESET={getattr(c, 'model', '')}")
print(f"GBS={getattr(c, 'global_batch_size', 0)}")
print(f"SEQ={getattr(c, 'seq_length', 0)}")
print(f"ITERS={getattr(c, 'train_iters', 0)}")
PY
)
eval "${PLAN}"

ITERS="${TRAIN_ITERS:-${ITERS}}"
[[ -n "${PROBE}" ]] && ITERS="${PROBE}"

echo "[convergence] config       : ${CONFIG}"
echo "[convergence] model preset : ${MODEL_PRESET}"
echo "[convergence] experiment   : ${EXP_DIR}"
echo "[convergence] iterations   : ${ITERS} (gbs ${GBS} x seq ${SEQ})"

# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------
if [[ -z "${DATA_DIR}" ]]; then
    DATA_DIR=$(MODEL_PRESET="${MODEL_PRESET}" python3 - <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "tools", "convergence_test"))
from prepare_dataset import resolve_tokenizer_from_model
from pathlib import Path
_, name = resolve_tokenizer_from_model(os.environ["MODEL_PRESET"])
root = Path(os.environ.get("DATA_PATH", Path.cwd() / "data"))
print(root / "convergence" / f"fineweb-edu-{name.split('/')[-1].lower()}")
PY
)
    # Defence in depth: only ever trust the final line as the path.
    DATA_DIR=$(printf '%s\n' "${DATA_DIR}" | tail -n 1)
fi
export PRIMUS_CONVERGENCE_DATA="${DATA_DIR}"
echo "[convergence] dataset      : ${DATA_DIR}"

if [[ "${PLOT_ONLY}" -eq 0 && "${SKIP_PREPARE}" -eq 0 ]]; then
    # Build ~10% more tokens than the run consumes so nothing repeats.
    TARGET_TOKENS=$(python3 -c "print(f'{${ITERS}*${GBS}*${SEQ}*1.1:.0f}')")
    echo "[convergence] ensuring ${TARGET_TOKENS} training tokens exist"
    python3 "${TOOL_DIR}/prepare_dataset.py" \
        --model "${MODEL_PRESET}" \
        --out-dir "${DATA_DIR}" \
        --target-tokens "${TARGET_TOKENS}"
fi
[[ "${PREPARE_ONLY}" -eq 1 ]] && exit 0

# ---------------------------------------------------------------------------
# 2. Lint
# ---------------------------------------------------------------------------
if [[ "${PLOT_ONLY}" -eq 0 ]]; then
    LINT_ARGS=(--config "${CONFIG}")
    [[ "${STRICT}" -eq 1 ]] && LINT_ARGS+=(--strict)
    if ! python3 "${TOOL_DIR}/check_config.py" "${LINT_ARGS[@]}"; then
        echo "[convergence] lint failed; fix the errors above or re-run with the fixes applied" >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------
LOG_DIR="${PRIMUS_PATH}/output/convergence"
mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/${EXP_NAME}_$(date +%Y%m%d-%H%M%S).log"

if [[ "${PLOT_ONLY}" -eq 0 ]]; then
    CLI=(./primus-cli container --env "PRIMUS_CONVERGENCE_DATA=${DATA_DIR}")
    [[ -n "${IMAGE}" ]] && CLI+=(--image "${IMAGE}")
    CLI+=(-- train pretrain --config "${CONFIG}")
    [[ -n "${TRAIN_ITERS}" ]] && CLI+=(--train_iters "${TRAIN_ITERS}" --lr_decay_iters "${TRAIN_ITERS}")
    if [[ -n "${PROBE}" ]]; then
        # A probe only measures speed: skip validation, and log every iteration
        # so there are enough timing samples to take a median from.
        CLI+=(--train_iters "${PROBE}" --eval_iters 0 --eval_interval 100000000 --log_interval 1)
    fi
    CLI+=("${EXTRA[@]+"${EXTRA[@]}"}")

    if [[ "${DETERMINISTIC}" -eq 1 ]]; then
        export PRIMUS_DETERMINISTIC=1
        echo "[convergence] PRIMUS_DETERMINISTIC=1 (Ring all-reduce, no rocBLAS atomics, no autotune)"
    fi

    # A leftover container holds the torchrun master port and the next launch
    # dies with EADDRINUSE, which is not obvious from the traceback.
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^primus-training'; then
        echo "[convergence] error: a primus-training container is already running:" >&2
        docker ps --filter name=primus-training --format '  {{.Names}} ({{.Status}})' >&2
        echo "[convergence] remove it first: docker rm -f \$(docker ps -q -f name=primus-training)" >&2
        exit 1
    fi

    echo "[convergence] launching: ${CLI[*]}"
    echo "[convergence] console log: ${RUN_LOG}"
    START=$(date +%s)
    "${CLI[@]}" 2>&1 | tee "${RUN_LOG}"
    WALL_CLOCK_MIN=$(python3 -c "print(f'{($(date +%s) - ${START})/60:.1f}')")
    echo "[convergence] wall clock: ${WALL_CLOCK_MIN} min"
fi

# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------
if [[ ! -d "${EXP_DIR}" ]]; then
    echo "[convergence] no experiment directory at ${EXP_DIR}; skipping plot" >&2
    exit 0
fi

PLOT_ARGS=("${EXP_DIR}" --out "${LOG_DIR}/${EXP_NAME}" --seq-length "${SEQ}" --detailed)
[[ -f "${DATA_DIR}/dataset_info.json" ]] && PLOT_ARGS+=(--dataset-info "${DATA_DIR}/dataset_info.json")
[[ -n "${BUDGET_HOURS}" ]] && PLOT_ARGS+=(--budget-hours "${BUDGET_HOURS}")
[[ -n "${WALL_CLOCK_MIN:-}" ]] && PLOT_ARGS+=(--wall-clock-min "${WALL_CLOCK_MIN}")
python3 "${TOOL_DIR}/plot_loss.py" "${PLOT_ARGS[@]}"
