#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# End-to-end convergence test driver: build a real dataset, lint the config,
# train, then plot the loss curve. Works for the Megatron and MaxText backends;
# the backend comes from the config's `framework:`.
#
#   tools/convergence_test/run_convergence_test.sh --model megatron/llama3.2_1B
#   tools/convergence_test/run_convergence_test.sh --model maxtext/mixtral_8x7B \
#       --probe 20 --budget-hours 3
#
# Run --help for the full option list.
#
###############################################################################

# The whole script is one compound command, which bash parses completely before
# running any of it. Bash otherwise reads a script as it executes, so editing or
# `git pull`ing this file during a multi-hour run would change what the running
# instance does next -- up to launching the training a second time.
{
set -euo pipefail

TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_PATH="$(cd "${TOOL_DIR}/../.." && pwd)"
CALLER_PWD="$(pwd)"
ORIG_ARGS=("$@")

# runner/.primus.yaml's default image is the PyTorch one; MaxText needs JAX.
MAXTEXT_DEFAULT_IMAGE="rocm/jax-training:maxtext-v26.7"

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
SOURCE="fineweb-edu"
TEXT_FIELD=""
TOKENIZER=""
OUTPUT_DIR=""
BASELINE=""
TOLERANCE=""
ENVS=()
EXTRA=()

usage() {
cat <<EOF
Usage: $(basename "$0") [options] [-- extra primus-cli args]

Selecting what to run:
  --model <name>        A bundled config: <backend>/<model>, or just <model>
                        when only one backend has it (see --list)
  --config <file>       Use an arbitrary Primus experiment YAML
  --list                List the bundled configs
  --train-iters <N>     Override the run length (and the LR schedule with it)
  --probe <N>           Run only N iterations to measure throughput, then stop
  --budget-hours <H>    With --probe, report the iterations that fit in H hours

Dataset:
  --source <corpus>     fineweb-edu (default), c4 (allenai/c4 en), wikitext103;
                        any Hugging Face dataset as <owner>/<name>[:<subset>];
                        or your own .jsonl/.json/.parquet/.txt files as a path,
                        directory or glob starting with /, ./ or ../
  --text-field <name>   Column that holds the document text (default: text)
  --data-dir <path>     Dataset directory (default: \$DATA_PATH/convergence/<tag>)
  --tokenizer <repo>    Build the dataset with this tokenizer instead of the
                        model preset's (e.g. an ungated mirror)
  --skip-prepare        Assume the dataset already exists
  --prepare-only        Build the dataset and exit

Results:
  --output-dir <dir>    Write everything under <dir>: the console log, loss
                        curve and CSV, and the trainer's own logs (default:
                        output/convergence and output/ in the Primus checkout)
  --baseline <run>      Compare against a reference run (log, experiment
                        directory, or a CSV this tool wrote); exit 3 on FAIL
  --tolerance <loss>    Largest loss difference that passes (default 0.05)

Other:
  --deterministic       Export PRIMUS_DETERMINISTIC=1 for a repeatable run
  --image <image>       Docker image (default: runner/.primus.yaml for Megatron,
                        ${MAXTEXT_DEFAULT_IMAGE} for MaxText)
  --env <KEY=VALUE>     Set an environment variable in the container (repeatable)
  --strict              Fail on lint warnings as well as errors
  --plot-only           Re-plot the latest run of this config (same --source
                        and --output-dir as the run)
  -h, --help            This message

Anything after -- is forwarded to primus-cli as training overrides.
EOF
}

list_configs() {
    for f in "${TOOL_DIR}"/configs/*/*-convergence.yaml; do
        [[ -e "$f" ]] || continue
        local backend name
        backend="$(basename "$(dirname "$f")")"
        name="$(basename "${f%-convergence.yaml}")"
        echo "  ${backend}/${name}"
    done
}

resolve_model() {
    local name="$1" matches=() f
    if [[ "${name}" == */* ]]; then
        [[ -f "${TOOL_DIR}/configs/${name}-convergence.yaml" ]] && matches+=("${TOOL_DIR}/configs/${name}-convergence.yaml")
    else
        for f in "${TOOL_DIR}"/configs/*/"${name}"-convergence.yaml; do
            [[ -e "$f" ]] && matches+=("$f")
        done
    fi
    if [[ ${#matches[@]} -eq 1 ]]; then
        echo "${matches[0]}"
        return
    fi
    if [[ ${#matches[@]} -eq 0 ]]; then
        echo "error: no bundled config for --model ${name}; available:" >&2
    else
        echo "error: --model ${name} is ambiguous; prefix the backend:" >&2
    fi
    list_configs >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)          MODEL="$2"; shift 2;;
        --config)         CONFIG="$2"; shift 2;;
        --list)           list_configs; exit 0;;
        --train-iters)    TRAIN_ITERS="$2"; shift 2;;
        --probe)          PROBE="$2"; shift 2;;
        --budget-hours)   BUDGET_HOURS="$2"; shift 2;;
        --data-dir)       DATA_DIR="$2"; shift 2;;
        --source)         SOURCE="$2"; shift 2;;
        --text-field)     TEXT_FIELD="$2"; shift 2;;
        --tokenizer)      TOKENIZER="$2"; shift 2;;
        --output-dir)     OUTPUT_DIR="$2"; shift 2;;
        --skip-prepare)   SKIP_PREPARE=1; shift;;
        --prepare-only)   PREPARE_ONLY=1; shift;;
        --baseline)       BASELINE="$2"; shift 2;;
        --tolerance)      TOLERANCE="$2"; shift 2;;
        --deterministic)  DETERMINISTIC=1; shift;;
        --image)          IMAGE="$2"; shift 2;;
        --env)            ENVS+=("$2"); shift 2;;
        --strict)         STRICT=1; shift;;
        --plot-only)      PLOT_ONLY=1; shift;;
        -h|--help)        usage; exit 0;;
        --)               shift; EXTRA=("$@"); break;;
        *)                echo "unknown option: $1" >&2; usage; exit 1;;
    esac
done

if [[ -n "${MODEL}" && -z "${CONFIG}" ]]; then
    CONFIG="$(resolve_model "${MODEL}")"
fi
if [[ -z "${CONFIG}" ]]; then
    echo "error: one of --model or --config is required; bundled configs:" >&2
    list_configs >&2
    exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "error: config not found: ${CONFIG}" >&2
    exit 1
fi
CONFIG="$(cd "$(dirname "${CONFIG}")" && pwd)/$(basename "${CONFIG}")"
for kv in "${ENVS[@]+"${ENVS[@]}"}"; do
    if [[ ! "${kv}" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
        echo "error: --env expects KEY=VALUE, got: ${kv}" >&2
        exit 1
    fi
done

# Paths on the command line are relative to where the caller stands; everything
# below runs from the Primus root.
abspath() { [[ "$1" = /* ]] && echo "$1" || echo "${CALLER_PWD}/$1"; }
[[ -n "${DATA_DIR}" ]] && DATA_DIR="$(abspath "${DATA_DIR}")"
[[ -n "${BASELINE}" ]] && BASELINE="$(abspath "${BASELINE}")"
[[ -n "${OUTPUT_DIR}" ]] && OUTPUT_DIR="$(abspath "${OUTPUT_DIR}")"
# A tokenizer is a Hub repo unless it names a local directory.
[[ -n "${TOKENIZER}" && -d "${CALLER_PWD}/${TOKENIZER}" ]] && TOKENIZER="$(abspath "${TOKENIZER}")"
# A local corpus is written as a path; a Hub dataset never starts with a dot.
case "${SOURCE}" in .|..|./*|../*) SOURCE="${CALLER_PWD}/${SOURCE}";; esac
# primus-cli forwards PRIMUS_* variables into the container. Every bundled
# config sets workspace: ${PRIMUS_WORKSPACE:./output}.
[[ "${DETERMINISTIC}" -eq 1 ]] && export PRIMUS_DETERMINISTIC=1
[[ -n "${OUTPUT_DIR}" ]] && export PRIMUS_WORKSPACE="${OUTPUT_DIR}"
# primus-cli lets an exported DOCKER_IMAGE override its --image.
[[ -n "${IMAGE}" ]] && export DOCKER_IMAGE="${IMAGE}"

cd "${PRIMUS_PATH}"

# ---------------------------------------------------------------------------
# Resolve the fields we need out of the fully merged config, exactly as the
# trainer will see them (for MaxText that includes its base.yml and model file).
# ---------------------------------------------------------------------------
PLAN_ARGS=(--config "${CONFIG}" --source "${SOURCE}")
[[ -n "${TOKENIZER}" ]] && PLAN_ARGS+=(--tokenizer "${TOKENIZER}")
PLAN=$(python3 "${TOOL_DIR}/resolve_config.py" "${PLAN_ARGS[@]}" "${EXTRA[@]+"${EXTRA[@]}"}")
# Values are shell-quoted by resolve_config.py; only evaluate KEY=VALUE lines.
eval "$(printf '%s\n' "${PLAN}" | grep -E '^[A-Z_]+=')"

# ---------------------------------------------------------------------------
# Host prerequisites. Training runs in the container image, but dataset
# preparation, linting and plotting all run here on the host, so fail now
# rather than three hours from now.
# ---------------------------------------------------------------------------
if ! FRAMEWORK="${FRAMEWORK}" PLOT_ONLY="${PLOT_ONLY}" SKIP_PREPARE="${SKIP_PREPARE}" python3 - <<'PY'
import importlib.util, os, sys
required = []
if os.environ["PLOT_ONLY"] == "0":
    required += ["yaml", "numpy", "transformers"]
    if os.environ["SKIP_PREPARE"] == "0":
        required += ["datasets", "pyarrow" if os.environ["FRAMEWORK"] == "maxtext" else "torch"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    print("error: the host python is missing: " + ", ".join(missing), file=sys.stderr)
    print("       pip install " + " ".join(missing), file=sys.stderr)
    sys.exit(1)
if importlib.util.find_spec("matplotlib") is None:
    print("[convergence] note: matplotlib missing, only CSV output will be produced",
          file=sys.stderr)
PY
then
    exit 1
fi

ITERS="${TRAIN_ITERS:-${ITERS}}"
# A probe plans the full run, so it builds the full run's dataset; a probe-sized
# one would be reused, too small, by the run that follows.
FULL_ITERS="${ITERS}"
[[ -n "${PROBE}" ]] && ITERS="${PROBE}"
STAMP="$(date +%Y%m%d-%H%M%S)"

if [[ -z "${IMAGE}" && -z "${DOCKER_IMAGE:-}" && "${FRAMEWORK}" == "maxtext" ]]; then
    IMAGE="${MAXTEXT_DEFAULT_IMAGE}"
fi

# Console log, loss curve and CSV of a run share one name: <exp>[-<corpus>]_<stamp>.
LOG_DIR="${OUTPUT_DIR:-${PRIMUS_PATH}/output/convergence}"
PLOT_LOG=""
if [[ "${PLOT_ONLY}" -eq 1 ]]; then
    # Re-plot the newest run of this config under its own name; timestamps sort.
    for f in "${LOG_DIR}/${EXP_NAME}${SOURCE_TAG}_"[0-9]*.log; do
        [[ -e "${f}" ]] && PLOT_LOG="${f}"
    done
    if [[ -n "${PLOT_LOG}" ]]; then
        STAMP="${PLOT_LOG##*_}"
        STAMP="${STAMP%.log}"
    fi
fi
RUN_TAG="${EXP_NAME}${SOURCE_TAG}_${STAMP}"
RUN_LOG="${LOG_DIR}/${RUN_TAG}.log"

# Every MaxText run gets its own run_name, so its TensorBoard file is not mixed
# with an earlier run's under base_output_directory/run_name.
if [[ "${FRAMEWORK}" == "maxtext" ]]; then
    RUN_NAME="${RUN_NAME:-${EXP_NAME}}${SOURCE_TAG}-${STAMP}"
    [[ -n "${OUTPUT_DIR}" ]] && BASE_OUTPUT_DIR="${OUTPUT_DIR}"
fi

echo "[convergence] config       : ${CONFIG}"
echo "[convergence] framework    : ${FRAMEWORK}"
echo "[convergence] model preset : ${MODEL_PRESET}"
echo "[convergence] iterations   : ${ITERS} (gbs ${GBS} x seq ${SEQ})"
echo "[convergence] image        : ${IMAGE:-${DOCKER_IMAGE:-runner/.primus.yaml default}}"
echo "[convergence] results      : ${LOG_DIR}/${RUN_TAG}.{log,csv,png}"
echo "[convergence] trainer logs : ${EXP_DIR}"
[[ "${FRAMEWORK}" == "maxtext" ]] && echo "[convergence] tensorboard  : ${BASE_OUTPUT_DIR}/${RUN_NAME}"

# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------
DATA_DIR="${DATA_DIR:-${DEFAULT_DATA_DIR}}"
export PRIMUS_CONVERGENCE_DATA="${DATA_DIR}"
echo "[convergence] dataset      : ${DATA_DIR} (${SOURCE})"

if [[ "${PLOT_ONLY}" -eq 0 && "${SKIP_PREPARE}" -eq 0 ]]; then
    # Build ~10% more tokens than the run consumes so nothing repeats.
    TARGET_TOKENS=$(python3 -c "print(f'{${FULL_ITERS}*${GBS}*${SEQ}*1.1:.0f}')")
    echo "[convergence] ensuring ${TARGET_TOKENS} training tokens exist"
    PREP=(python3 "${TOOL_DIR}/prepare_dataset.py"
        --format "${FRAMEWORK}"
        --model "${MODEL_PRESET}"
        --source "${SOURCE}"
        --out-dir "${DATA_DIR}"
        --target-tokens "${TARGET_TOKENS}")
    [[ -n "${TOKENIZER}" ]] && PREP+=(--tokenizer "${TOKENIZER}")
    [[ -n "${TEXT_FIELD}" ]] && PREP+=(--text-field "${TEXT_FIELD}")
    # MaxText rows are pre-chunked: grain rejects rows longer than max_target_length.
    [[ "${FRAMEWORK}" == "maxtext" ]] && PREP+=(--seq-length "${SEQ}")
    "${PREP[@]}"
fi
[[ "${PREPARE_ONLY}" -eq 1 ]] && exit 0

# ---------------------------------------------------------------------------
# Overrides for this run. The lint sees them too, so it checks the run that is
# actually launched rather than the config on disk.
# ---------------------------------------------------------------------------
OVERRIDES=()
if [[ "${FRAMEWORK}" == "maxtext" ]]; then
    OVERRIDES+=(--run_name "${RUN_NAME}")
    [[ -n "${OUTPUT_DIR}" ]] && OVERRIDES+=(--base_output_directory "${OUTPUT_DIR}")
    if [[ -n "${PROBE}" ]]; then
        # A probe measures speed, but MaxText also evaluates at step 0 whenever
        # eval_interval divides it, so a huge interval evaluates exactly once:
        # the eval data pipeline is exercised before a long run depends on it.
        # --probe wins over --train-iters; passing both would duplicate --steps.
        # MaxText scales warmup with the schedule length, so pin it to the full
        # run's: the probe then trains the first steps of the real run.
        OVERRIDES+=(--steps "${PROBE}" --eval_interval 100000000 --learning_rate_schedule_steps "${FULL_ITERS}")
    elif [[ -n "${TRAIN_ITERS}" ]]; then
        OVERRIDES+=(--steps "${TRAIN_ITERS}" --learning_rate_schedule_steps "${TRAIN_ITERS}")
    fi
else
    if [[ -n "${PROBE}" ]]; then
        # The first iterations of the full run's schedule, logged every iteration
        # so there are enough timing samples to take a median from. With no
        # evaluation inside the run, eval_iters 1 is Megatron's end-of-training
        # evaluation only: it exercises the validation data before a long run
        # depends on it.
        OVERRIDES+=(--train_iters "${PROBE}" --lr_decay_iters "${FULL_ITERS}" --log_interval 1
            --eval_iters 1 --eval_interval 100000000)
    elif [[ -n "${TRAIN_ITERS}" ]]; then
        OVERRIDES+=(--train_iters "${TRAIN_ITERS}" --lr_decay_iters "${TRAIN_ITERS}")
    fi
fi
OVERRIDES+=("${EXTRA[@]+"${EXTRA[@]}"}")

# ---------------------------------------------------------------------------
# 2. Lint
# ---------------------------------------------------------------------------
if [[ "${PLOT_ONLY}" -eq 0 ]]; then
    LINT_ARGS=(--config "${CONFIG}")
    [[ "${STRICT}" -eq 1 ]] && LINT_ARGS+=(--strict)
    [[ -n "${PROBE}" ]] && LINT_ARGS+=(--probe)
    if ! python3 "${TOOL_DIR}/check_config.py" "${LINT_ARGS[@]}" "${OVERRIDES[@]+"${OVERRIDES[@]}"}"; then
        echo "[convergence] lint failed; fix the errors above or re-run with the fixes applied" >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------
mkdir -p "${LOG_DIR}"

if [[ "${PLOT_ONLY}" -eq 0 ]]; then
    CLI=(./primus-cli container --env "PRIMUS_CONVERGENCE_DATA=${DATA_DIR}")
    # The container only mounts the Primus checkout; a dataset or an output
    # directory kept elsewhere (shared storage) is mounted at the same path.
    DATA_DIR_ABS="$(cd "${DATA_DIR}" && pwd)"
    [[ "${DATA_DIR_ABS}/" != "${PRIMUS_PATH}/"* ]] && CLI+=(--volume "${DATA_DIR_ABS}:${DATA_DIR_ABS}")
    if [[ -n "${OUTPUT_DIR}" && "${OUTPUT_DIR}/" != "${PRIMUS_PATH}/"* && "${OUTPUT_DIR}" != "${DATA_DIR_ABS}" ]]; then
        CLI+=(--volume "${OUTPUT_DIR}:${OUTPUT_DIR}")
    fi
    for kv in "${ENVS[@]+"${ENVS[@]}"}"; do
        CLI+=(--env "${kv}")
    done
    [[ -n "${IMAGE}" ]] && CLI+=(--image "${IMAGE}")
    CLI+=(-- train pretrain --config "${CONFIG}" "${OVERRIDES[@]+"${OVERRIDES[@]}"}")

    if [[ "${DETERMINISTIC}" -eq 1 ]]; then
        echo "[convergence] PRIMUS_DETERMINISTIC=1 (Ring all-reduce, deterministic TE, no rocBLAS atomics, no autotune)"
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
    # What produced this run, for whoever later uses its CSV as a baseline.
    cp "${CONFIG}" "${LOG_DIR}/${RUN_TAG}.yaml"
    BACKEND_DIR="${PRIMUS_PATH}/third_party/$([[ "${FRAMEWORK}" == "maxtext" ]] && echo maxtext || echo Megatron-LM)"
    {
        printf '[convergence] command      :'
        printf ' %q' "$0" "${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"}"
        echo
        echo "[convergence] primus       : $(git -C "${PRIMUS_PATH}" describe --always --dirty 2>/dev/null || echo unknown)"
        echo "[convergence] backend      : ${BACKEND_DIR#"${PRIMUS_PATH}/"} $(git -C "${BACKEND_DIR}" describe --always --dirty 2>/dev/null || echo unknown)"
        echo "[convergence] config       : ${CONFIG} (copied to ${RUN_TAG}.yaml)"
        echo "[convergence] dataset      : ${DATA_DIR} (${SOURCE})"
        echo "[convergence] overrides    : ${OVERRIDES[*]+"${OVERRIDES[*]}"}"
    } > "${RUN_LOG}"
    START=$(date +%s)
    "${CLI[@]}" 2>&1 | tee -a "${RUN_LOG}"
    WALL_CLOCK_MIN=$(python3 -c "print(f'{($(date +%s) - ${START})/60:.1f}')")
    echo "[convergence] wall clock: ${WALL_CLOCK_MIN} min"
    # MaxText turns a failing data iterator (or running out of data) into a
    # graceful stop and exits 0; the iteration count check below catches it.
    STOPPED=$(grep -m 1 -o "Training stopped: .*" "${RUN_LOG}" || true)
    [[ -n "${STOPPED}" ]] && echo "[convergence] MaxText reported: ${STOPPED}" >&2
fi

# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------
# A run is read from its own console log whenever the loss reaches the console:
# the experiment directory's logs also hold every earlier run of the config, and
# a run that trained nothing must not be scored on the one before it. Megatron's
# evaluations never reach the console; they are taken from the experiment log's
# copy of this same run.
PLOT_INPUT="${EXP_DIR}"
VALIDATION_FROM=""
if [[ "${LOSS_ON_CONSOLE}" -eq 1 && ( "${PLOT_ONLY}" -eq 0 || -n "${PLOT_LOG}" ) ]]; then
    PLOT_INPUT="${RUN_LOG}"
    [[ "${FRAMEWORK}" == "megatron" ]] && VALIDATION_FROM="${EXP_DIR}"
elif [[ "${PLOT_ONLY}" -eq 1 && ! -d "${EXP_DIR}" ]]; then
    echo "[convergence] error: no run to plot: no ${LOG_DIR}/${EXP_NAME}${SOURCE_TAG}_*.log and no ${EXP_DIR}" >&2
    exit 1
fi
PLOT_ARGS=("${PLOT_INPUT}" --out "${LOG_DIR}/${RUN_TAG}" --seq-length "${SEQ}" --global-batch-size "${GBS}" --detailed)
[[ "${PLOT_ONLY}" -eq 0 ]] && PLOT_ARGS+=(--expect-iters "${ITERS}")
if [[ "${FRAMEWORK}" == "maxtext" ]]; then
    # Loss to full precision, learning rate and grad norm only reach TensorBoard.
    TB_DIR="${BASE_OUTPUT_DIR}/${RUN_NAME}"
    if [[ "${PLOT_ONLY}" -eq 1 && -z "${PLOT_LOG}" ]]; then
        # Run names end in a sortable timestamp, so the last match is the latest
        # run; the digit keeps a default-corpus name from matching "-c4-" runs.
        TB_DIR=""
        for dir in "${BASE_OUTPUT_DIR}/${RUN_NAME%-"${STAMP}"}"-[0-9]*/; do
            [[ -d "${dir}" ]] && TB_DIR="${dir}"
        done
    fi
    [[ -n "${TB_DIR}" && -d "${TB_DIR}" ]] && PLOT_ARGS+=(--tensorboard "${TB_DIR}")
    # The model's vocab_size, not the tokenizer's, sets the step-1 loss.
    PLOT_ARGS+=(--vocab-size "${VOCAB}")
else
    [[ -n "${VALIDATION_FROM}" ]] && PLOT_ARGS+=(--validation-from "${VALIDATION_FROM}")
    [[ -f "${DATA_DIR}/dataset_info.json" ]] && PLOT_ARGS+=(--dataset-info "${DATA_DIR}/dataset_info.json")
fi
[[ -n "${BUDGET_HOURS}" ]] && PLOT_ARGS+=(--budget-hours "${BUDGET_HOURS}")
[[ -n "${WALL_CLOCK_MIN:-}" ]] && PLOT_ARGS+=(--wall-clock-min "${WALL_CLOCK_MIN}")
[[ -n "${BASELINE}" ]] && PLOT_ARGS+=(--baseline "${BASELINE}")
[[ -n "${TOLERANCE}" ]] && PLOT_ARGS+=(--tolerance "${TOLERANCE}")

# The plot itself is a post-processing convenience, and plot_loss.py skips it
# when matplotlib is missing. What must reach the caller is an incomplete run
# (exit 4), a baseline FAIL (exit 3), or a baseline that could not be checked.
set +e
python3 "${TOOL_DIR}/plot_loss.py" "${PLOT_ARGS[@]}"
PLOT_RC=$?
set -e
case "${PLOT_RC}" in
    0) ;;
    4)
        echo "[convergence] FAIL: training did not complete ${ITERS} iterations; see ${RUN_LOG}" >&2
        exit 4;;
    3)
        echo "[convergence] FAIL: the loss curve does not match the baseline ${BASELINE}" >&2
        exit 3;;
    *)
        # 2 is a plot that could not be drawn after every check passed; anything
        # else means the checks themselves did not run.
        if [[ -n "${BASELINE}" && "${PLOT_RC}" -ne 2 ]]; then
            echo "[convergence] FAIL: could not compare against the baseline ${BASELINE}" >&2
            exit 1
        fi
        echo "[convergence] plotting failed; the run itself is unaffected." >&2
        echo "[convergence] re-plot later with:" >&2
        echo "  python3 ${TOOL_DIR}/plot_loss.py ${PLOT_ARGS[*]}" >&2;;
esac
exit
}
