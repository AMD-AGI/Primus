#!/usr/bin/env bash
# Removed: set -e (allow script to continue on errors in benchmarks)

if [[ -z "${BASH_VERSION:-}" ]]; then
    echo "This script must be run with bash (not sh)." >&2
    exit 1
fi

# Set up trap to debug unexpected exits
trap 'echo "[DEBUG] Script exiting at line $LINENO with exit code $?"' EXIT

# ------------------------------------------
# Colors & Icons
# ------------------------------------------
BOLD="\033[1m"
DIM="\033[2m"
RESET="\033[0m"
GREEN="\033[32m"
YELLOW="\033[33m"
CYAN="\033[36m"
MAGENTA="\033[35m"
RED="\033[31m"

CHECK="${GREEN}✓${RESET}"
DOT="${YELLOW}●${RESET}"
STAR="${MAGENTA}★${RESET}"
ARROW="${CYAN}➜${RESET}"
INFO="${CYAN}ℹ${RESET}"

# ------------------------------------------
# Paths
# ------------------------------------------
PRIMUS_ROOT="/workspace/Primus"
MEGATRON_BASE_DIR="${PRIMUS_ROOT}/examples/megatron/configs"
MEGATRON_BRIDGE_BASE_DIR="${PRIMUS_ROOT}/examples/megatron_bridge/configs"
TORCHTITAN_BASE_DIR="${PRIMUS_ROOT}/examples/torchtitan/configs"
MLPERF_BASE_DIR="${PRIMUS_ROOT}/examples/mlperf"
PRIMUS_CLI="${PRIMUS_ROOT}/primus-cli"
VALID_DEVICES=(MI300X MI325X MI355X)
MLPERF_MODELS=(llama3.1_8b llama2_70b)

if [[ ! -f "$PRIMUS_CLI" && -f "${PRIMUS_ROOT}/runner/primus-cli" ]]; then
    PRIMUS_CLI="${PRIMUS_ROOT}/runner/primus-cli"
fi

# ------------------------------------------
# Helpers
# ------------------------------------------
install_vim_editor() {
    echo -e "${YELLOW}⚠ No editor found. Installing vim...${RESET}"

    if [[ $EUID -eq 0 ]]; then
        apt-get update && apt-get install -y vim
    elif command -v sudo &>/dev/null; then
        sudo apt-get update && sudo apt-get install -y vim
    else
        echo -e "${RED}✗ Cannot install vim: not root and sudo is unavailable.${RESET}"
        echo -e "   ${DOT} Install vim manually, then re-run config editing:"
        echo -e "   ${CYAN}apt-get update && apt-get install -y vim${RESET}"
        return 1
    fi
}

configure_git_safe_directories() {
    local repo_dir git_dir

    if ! command -v git &>/dev/null; then
        return 0
    fi

    if [[ -d "$PRIMUS_ROOT/.git" || -f "$PRIMUS_ROOT/.git" ]]; then
        git config --global --add safe.directory "$PRIMUS_ROOT" 2>/dev/null || true
    fi

    if [[ -d "$PRIMUS_ROOT/third_party" ]]; then
        while IFS= read -r -d '' git_dir; do
            repo_dir=$(dirname "$git_dir")
            git config --global --add safe.directory "$repo_dir" 2>/dev/null || true
        done < <(find "$PRIMUS_ROOT/third_party" -name .git -print0 2>/dev/null)
    fi
}

open_config_editor() {
    local config_file="$1"
    local candidate editor_bin editor_args editor_label

    for candidate in \
        "${EDITOR:-}" \
        "${VISUAL:-}" \
        "vim" \
        "vi" \
        "nano" \
        "emacs -nw" \
        "code --wait" \
        "cursor --wait"; do
        if [[ -z "$candidate" ]]; then
            continue
        fi

        editor_bin="${candidate%% *}"
        if ! command -v "$editor_bin" &>/dev/null; then
            continue
        fi

        editor_args="${candidate#"$editor_bin"}"
        editor_label="$candidate"
        echo -e "   ${DOT} Using editor: ${CYAN}$editor_label${RESET}"
        # shellcheck disable=SC2086
        "$editor_bin" $editor_args "$config_file"
        return 0
    done

    if install_vim_editor && command -v vim &>/dev/null; then
        echo -e "   ${DOT} Using editor: ${CYAN}vim${RESET}"
        vim "$config_file"
        return 0
    fi

    echo -e "${RED}✗ Failed to open an editor for:${RESET} ${CYAN}$config_file${RESET}"
    return 1
}

edit_config_interactively() {
    local cfg="$1"
    local model_name temp_edit_config

    model_name=$(basename "$cfg" .yaml)

    if [[ -n "${EDITED_CONFIGS[$cfg]:-}" && -f "${EDITED_CONFIGS[$cfg]}" ]]; then
        temp_edit_config="${EDITED_CONFIGS[$cfg]}"
    else
        temp_edit_config="/tmp/primus_edit_${model_name}_$$.yaml"
        cp "$cfg" "$temp_edit_config"
        EDITED_CONFIGS["$cfg"]="$temp_edit_config"
    fi

    echo -e "\n${STAR} ${BOLD}Opening config for editing: ${CYAN}$model_name${RESET}"
    echo -e "   ${DOT} Editing working copy: ${CYAN}$temp_edit_config${RESET}"
    echo -e "   ${DOT} Original config is unchanged: ${DIM}$cfg${RESET}"
    echo -e "   ${DOT} Edit the file, save, and close the editor to continue\n"

    open_config_editor "$temp_edit_config"

    echo -e " ${CHECK} ${GREEN}Working copy saved (original unchanged)${RESET}\n"
}

detect_train_suite() {
    local config_file="$1"
    local base_name

    if grep -qE '^[[:space:]]*post_trainer:' "$config_file" 2>/dev/null; then
        echo "posttrain"
        return
    fi
    if grep -qE '^[[:space:]]*pre_trainer:' "$config_file" 2>/dev/null; then
        echo "pretrain"
        return
    fi

    base_name=$(basename "$config_file" .yaml)
    if [[ "$base_name" == *posttrain* || "$base_name" == *-sft* || "$base_name" == *sft_* ]]; then
        echo "posttrain"
    else
        echo "pretrain"
    fi
}

next_run_number() {
    local model_name="$1"
    local prefix="${model_name}_${BACKEND}_${DEVICE}"
    local max_run=0
    local f bn run_n legacy_count=0

    shopt -s nullglob
    for f in "$LOG_DIR"/"${prefix}"_run*.log; do
        bn=$(basename "$f" .log)
        if [[ "$bn" =~ _run([0-9]+)$ ]]; then
            run_n="${BASH_REMATCH[1]}"
            if (( run_n > max_run )); then
                max_run=$run_n
            fi
        fi
    done

    for f in "$LOG_DIR"/"${prefix}"_*.log; do
        bn=$(basename "$f" .log)
        if [[ "$bn" =~ _run[0-9]+$ ]]; then
            continue
        fi
        legacy_count=$((legacy_count + 1))
    done
    shopt -u nullglob

    echo $((max_run + legacy_count + 1))
}

prepare_benchmark_artifacts() {
    local cfg_file="$1"
    local model_name run_num artifact_prefix

    PREP_CFG_FILE="$cfg_file"
    PREP_MODEL_NAME=$(basename "$cfg_file" .yaml)
    model_name="$PREP_MODEL_NAME"
    run_num=$(next_run_number "$model_name")
    PREP_RUN_LABEL="run${run_num}"
    artifact_prefix="${model_name}_${BACKEND}_${DEVICE}_${PREP_RUN_LABEL}"

    PREP_LOG_FILE="$LOG_DIR/${artifact_prefix}.log"
    PREP_WORKING_CONFIG="$LOG_DIR/${artifact_prefix}_working.yaml"

    if [[ -n "${EDITED_CONFIGS[$cfg_file]:-}" ]]; then
        cp "${EDITED_CONFIGS[$cfg_file]}" "$PREP_WORKING_CONFIG"
    else
        cp "$cfg_file" "$PREP_WORKING_CONFIG"
    fi
}

megatron_checkpoint_exists() {
    local ckpt_path="$1"

    [[ -n "$ckpt_path" && -d "$ckpt_path" ]] || return 1
    [[ -f "${ckpt_path}/latest_checkpointed_iteration.txt" ]] && return 0
    [[ -f "${ckpt_path}/latest_train_state.pt" ]] && return 0
    compgen -G "${ckpt_path}/iter_*" >/dev/null && return 0
    [[ -d "${ckpt_path}/release" ]] && return 0
    return 1
}

mlperf_pretrained_checkpoint_path() {
    local model_id="$1"

    case "$model_id" in
        llama2_70b)
            echo "/data/megatron_checkpoints/Llama-2-70b-hf"
            ;;
        *)
            echo ""
            ;;
    esac
}

resolve_hf_path_from_config() {
    local config_file="$1"

    grep -E '^[[:space:]]*hf_path:' "$config_file" 2>/dev/null \
        | head -1 \
        | sed -E 's/^[[:space:]]*hf_path:[[:space:]]*//' \
        | tr -d '"' \
        | tr -d "'"
}

resolve_pretrained_checkpoint_from_config() {
    local config_file="$1"
    local line ckpt_path hf_path

    line=$(grep -E '^[[:space:]]*pretrained_checkpoint:' "$config_file" 2>/dev/null | head -1 || true)
    if [[ -n "$line" ]]; then
        ckpt_path=$(printf '%s\n' "$line" \
            | sed -E 's/^[[:space:]]*pretrained_checkpoint:[[:space:]]*//' \
            | sed -E 's/^\$\{PRETRAINED_CHECKPOINT:(.*)\}$/\1/' \
            | tr -d '"' \
            | tr -d "'")
        if [[ -n "$ckpt_path" && "$ckpt_path" != "null" ]]; then
            echo "$ckpt_path"
            return 0
        fi
    fi

    hf_path="$(resolve_hf_path_from_config "$config_file")"
    if [[ -n "$hf_path" ]]; then
        echo "/data/megatron_checkpoints/$(basename "$hf_path")"
        return 0
    fi

    return 1
}

export_pretrained_checkpoint_env() {
    local ckpt_path="$1"
    local model_label="$2"

    [[ -n "$ckpt_path" ]] || return 1

    export PRETRAINED_CHECKPOINT="$ckpt_path"
    export MEGATRON_PATH="$ckpt_path"

    if megatron_checkpoint_exists "$ckpt_path"; then
        echo -e " ${CHECK} Megatron checkpoint found for ${CYAN}${model_label}${RESET} at ${CYAN}${ckpt_path}${RESET}; skipping download/conversion"
        return 0
    fi

    echo -e " ${DOT} Megatron checkpoint not found at ${CYAN}${ckpt_path}${RESET}; hooks will download/convert if needed"
    return 1
}

prepare_posttrain_checkpoint_env() {
    local config_file="$1"
    local model_label="$2"
    local ckpt_path

    if ! ckpt_path="$(resolve_pretrained_checkpoint_from_config "$config_file")"; then
        return 0
    fi

    export_pretrained_checkpoint_env "$ckpt_path" "$model_label"
}

execute_benchmark_run() {
    local cfg_file="$1"
    local working_config="$2"
    local log_file="$3"
    local model_name="$4"
    local current="$5"
    local total="$6"

    local run_exit_code=0
    local train_suite

    train_suite=$(detect_train_suite "$working_config")

    echo -e "${STAR} ${BOLD}Starting Benchmark ${current}/${total}...${RESET}"
    echo -e "   ${DOT} Model: ${CYAN}$model_name${RESET}"
    echo -e "   ${DOT} Backend: ${CYAN}$BACKEND${RESET}"
    echo -e "   ${DOT} Train suite: ${CYAN}$train_suite${RESET}"
    echo -e "   ${DOT} Device: ${CYAN}$DEVICE${RESET}"
    echo -e "   ${DOT} Config (working copy): ${YELLOW}$working_config${RESET}"
    echo -e "   ${DOT} Original (unchanged): ${DIM}$cfg_file${RESET}"
    echo -e "   ${DOT} Log: ${YELLOW}$log_file${RESET}\n"

    if [[ ! -f "$PRIMUS_CLI" ]]; then
        echo -e "${RED}✗ primus-cli not found at:${RESET} ${CYAN}$PRIMUS_CLI${RESET}"
        return 1
    fi

    if [[ "$train_suite" == "posttrain" ]]; then
        prepare_posttrain_checkpoint_env "$working_config" "$model_name"
    fi

    export EXP="$working_config" BACKEND HF_TOKEN
    echo -e " ${CHECK} Launching via primus-cli direct"
    echo -e "   ${DOT} ${CYAN}primus-cli direct --log_file $log_file -- train $train_suite --config $working_config${RESET}\n"

    echo -e " ${DOT} Changing to Primus root directory: ${CYAN}$PRIMUS_ROOT${RESET}"
    cd "$PRIMUS_ROOT" || return 1

    set +e
    bash "$PRIMUS_CLI" direct --log_file "$log_file" -- train "$train_suite" --config "$working_config"
    run_exit_code=$?
    set +e

    cd "$SCRIPT_DIR" || return 1

    echo
    echo -e "${GREEN}==========================================${RESET}"
    if [[ $run_exit_code -eq 0 ]]; then
        echo -e " ${BOLD}${GREEN}✓ Benchmark ${current}/${total} Completed Successfully!${RESET}"
    else
        echo -e " ${BOLD}${YELLOW}⚠ Benchmark ${current}/${total} Completed with Exit Code: $run_exit_code${RESET}"
    fi
    echo -e " Log saved at:"
    echo -e "   ${CYAN}$log_file${RESET}"
    echo -e "${GREEN}==========================================${RESET}"
    echo

    return "$run_exit_code"
}

resolve_mlperf_config_script() {
    local model_dir="$1"
    local candidate

    for candidate in \
        "${model_dir}/config_${DEVICE}_1x8x1.sh" \
        "${model_dir}/config_MI355X_1x8x1.sh"; do
        if [[ -f "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done

    return 1
}

MLPERF_LLAMA31_8B_ROOT="/data/mlperf_llama31_8b"
MLPERF_LLAMA31_8B_DATA="${MLPERF_LLAMA31_8B_ROOT}/data"
MLPERF_LLAMA31_8B_MODEL="${MLPERF_LLAMA31_8B_ROOT}/model"
MLPERF_LLAMA31_8B_DATA_URI="https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri"
MLPERF_LLAMA31_8B_MODEL_URI="https://training.mlcommons-storage.org/metadata/llama-3-1-8b-tokenizer.uri"
MLPERF_LLAMA31_8B_TRAIN_PREFIX="c4-train.en_6_text_document"
MLPERF_LLAMA31_8B_VALID_PREFIX="c4-validation-91205-samples.en_text_document"
MLPERF_LLAMA2_70B_PACKED_DATA_DIR="/data"
MLPERF_R2_DOWNLOADER_URL="https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh"
MLPERF_R2_DOWNLOADER_CACHE="${MLPERF_LLAMA31_8B_ROOT}/.cache/mlc-r2-downloader.sh"

fetch_url_to_file() {
    local url="$1"
    local dest="$2"

    if command -v curl &>/dev/null; then
        curl -fsSL "$url" -o "$dest"
        return $?
    fi

    if command -v wget &>/dev/null; then
        wget -qO "$dest" "$url"
        return $?
    fi

    if command -v python3 &>/dev/null; then
        python3 - "$url" "$dest" <<'PY'
import sys
import urllib.request

urllib.request.urlretrieve(sys.argv[1], sys.argv[2])
PY
        return $?
    fi

    return 1
}

ensure_mlperf_download_tools() {
    local missing=()
    local tool

    for tool in curl wget; do
        if ! command -v "$tool" &>/dev/null; then
            missing+=("$tool")
        fi
    done

    if [[ ${#missing[@]} -eq 0 ]]; then
        return 0
    fi

    echo -e " ${YELLOW}⚠ Missing MLPerf download tools: ${missing[*]}${RESET}"
    echo -e "   ${DOT} Attempting to install ${CYAN}curl${RESET} and ${CYAN}wget${RESET}..."

    if [[ $EUID -eq 0 ]]; then
        apt-get update && apt-get install -y curl wget
    elif command -v sudo &>/dev/null; then
        sudo apt-get update && sudo apt-get install -y curl wget
    else
        echo -e "${RED}✗ Cannot install curl/wget: not root and sudo is unavailable.${RESET}"
        echo -e "   ${DOT} Install manually: ${CYAN}apt-get update && apt-get install -y curl wget${RESET}"
        echo -e "   ${DOT} Or pre-download under ${CYAN}${MLPERF_LLAMA31_8B_ROOT}${RESET} and re-run"
        return 1
    fi

    for tool in curl wget; do
        if ! command -v "$tool" &>/dev/null; then
            echo -e "${RED}✗ ${tool} is still unavailable after install attempt${RESET}"
            return 1
        fi
    done

    echo -e " ${CHECK} MLPerf download tools ready: ${CYAN}curl${RESET}, ${CYAN}wget${RESET}"
    return 0
}

ensure_mlperf_r2_downloader() {
    if [[ -f "$MLPERF_R2_DOWNLOADER_CACHE" && -s "$MLPERF_R2_DOWNLOADER_CACHE" ]]; then
        echo "$MLPERF_R2_DOWNLOADER_CACHE"
        return 0
    fi

    mkdir -p "$(dirname "$MLPERF_R2_DOWNLOADER_CACHE")" || return 1

    if ! fetch_url_to_file "$MLPERF_R2_DOWNLOADER_URL" "$MLPERF_R2_DOWNLOADER_CACHE"; then
        echo -e "${RED}✗ Failed to fetch mlc-r2-downloader.sh${RESET}"
        echo -e "   ${DOT} Requires ${CYAN}curl${RESET} or ${CYAN}wget${RESET} (mlc-r2-downloader needs both)"
        echo -e "   ${DOT} Or pre-download dataset/model under ${CYAN}${MLPERF_LLAMA31_8B_ROOT}${RESET}"
        return 1
    fi

    chmod +x "$MLPERF_R2_DOWNLOADER_CACHE"
    echo "$MLPERF_R2_DOWNLOADER_CACHE"
}

run_mlperf_r2_downloader() {
    local downloader

    if ! downloader="$(ensure_mlperf_r2_downloader)"; then
        return 1
    fi

    bash "$downloader" "$@"
}

prepare_mlperf_artifacts() {
    local model_id="$1"
    local run_num artifact_prefix

    PREP_MLPERF_MODEL="$model_id"
    PREP_MLPERF_MODEL_DIR="${MLPERF_BASE_DIR}/${model_id}"
    PREP_MODEL_NAME="$model_id"
    run_num=$(next_run_number "$model_id")
    PREP_RUN_LABEL="run${run_num}"
    artifact_prefix="${model_id}_${BACKEND}_${DEVICE}_${PREP_RUN_LABEL}"
    PREP_LOG_FILE="$LOG_DIR/${artifact_prefix}.log"
}

mlperf_c4_dataset_exists_in_dir() {
    local dir="$1"

    [[ -n "$dir" && -d "$dir" ]] || return 1
    compgen -G "${dir}/${MLPERF_LLAMA31_8B_TRAIN_PREFIX}.*" >/dev/null || return 1
    compgen -G "${dir}/${MLPERF_LLAMA31_8B_VALID_PREFIX}.*" >/dev/null
}

resolve_mlperf_llama31_8b_dataset_dir() {
    local candidate

    for candidate in "$MLPERF_LLAMA31_8B_DATA" "/data"; do
        if mlperf_c4_dataset_exists_in_dir "$candidate"; then
            echo "$candidate"
            return 0
        fi
    done

    return 1
}

resolve_mlperf_llama31_8b_model_dir() {
    if [[ -d "$MLPERF_LLAMA31_8B_MODEL" && -n "$(ls -A "${MLPERF_LLAMA31_8B_MODEL}" 2>/dev/null)" ]]; then
        echo "$MLPERF_LLAMA31_8B_MODEL"
        return 0
    fi

    if [[ -d /model && -n "$(ls -A /model 2>/dev/null)" ]]; then
        echo "/model"
        return 0
    fi

    return 1
}

mlperf_llama31_8b_dataset_ready() {
    resolve_mlperf_llama31_8b_dataset_dir >/dev/null
}

mlperf_llama31_8b_model_ready() {
    resolve_mlperf_llama31_8b_model_dir >/dev/null
}

mlperf_llama2_70b_dataset_ready() {
    [[ -f "${MLPERF_LLAMA2_70B_PACKED_DATA_DIR}/train.npy" \
        && -f "${MLPERF_LLAMA2_70B_PACKED_DATA_DIR}/validation.npy" \
        && -f "${MLPERF_LLAMA2_70B_PACKED_DATA_DIR}/packed_metadata.jsonl" ]]
}

mlperf_logging_out_matches_model() {
    local model_id="$1"
    local logging_file="/results/mlperf_logging.out"

    [[ -f "$logging_file" ]] || return 1

    case "$model_id" in
        llama2_70b)
            grep -q "mlperf_sft.py" "$logging_file"
            ;;
        llama3.1_8b)
            grep -q "mlperf_pretrain_trainer.py" "$logging_file"
            ;;
        *)
            return 0
            ;;
    esac
}

prepare_mlperf_llama31_8b_dataset() {
    local existing_data_dir existing_model_dir

    echo -e " ${CHECK} mkdir -p ${CYAN}${MLPERF_LLAMA31_8B_ROOT}${RESET}"
    mkdir -p "$MLPERF_LLAMA31_8B_ROOT" || return 1

    if existing_data_dir="$(resolve_mlperf_llama31_8b_dataset_dir)"; then
        echo -e " ${CHECK} MLPerf llama3.1_8b dataset already present at ${CYAN}${existing_data_dir}${RESET}; skipping download"
    else
        cd "$MLPERF_LLAMA31_8B_ROOT" || return 1

        if ! ensure_mlperf_download_tools; then
            return 1
        fi

        echo -e " ${CHECK} Downloading MLPerf llama3.1_8b dataset into ${CYAN}${MLPERF_LLAMA31_8B_DATA}${RESET}"
        if ! run_mlperf_r2_downloader -d data "$MLPERF_LLAMA31_8B_DATA_URI"; then
            echo -e "${RED}✗ MLPerf llama3.1_8b dataset download failed${RESET}"
            return 1
        fi
    fi

    if existing_model_dir="$(resolve_mlperf_llama31_8b_model_dir)"; then
        echo -e " ${CHECK} MLPerf llama3.1_8b tokenizer already present at ${CYAN}${existing_model_dir}${RESET}; skipping download"
    else
        cd "$MLPERF_LLAMA31_8B_ROOT" || return 1

        if ! ensure_mlperf_download_tools; then
            return 1
        fi

        echo -e " ${CHECK} Downloading MLPerf llama3.1_8b tokenizer into ${CYAN}${MLPERF_LLAMA31_8B_MODEL}${RESET}"
        if ! run_mlperf_r2_downloader -d model "$MLPERF_LLAMA31_8B_MODEL_URI"; then
            echo -e "${RED}✗ MLPerf llama3.1_8b tokenizer download failed${RESET}"
            return 1
        fi
    fi

    if ! existing_data_dir="$(resolve_mlperf_llama31_8b_dataset_dir)"; then
        echo -e "${RED}✗ MLPerf llama3.1_8b dataset not found under ${MLPERF_LLAMA31_8B_DATA} or /data${RESET}"
        return 1
    fi

    if ! existing_model_dir="$(resolve_mlperf_llama31_8b_model_dir)"; then
        echo -e "${RED}✗ MLPerf llama3.1_8b tokenizer not found under ${MLPERF_LLAMA31_8B_MODEL} or /model${RESET}"
        return 1
    fi

    return 0
}

prepare_mlperf_llama31_8b_data_layout() {
    local item base target dataset_dir model_dir

    mkdir -p /data /results 2>/dev/null || true

    dataset_dir="$(resolve_mlperf_llama31_8b_dataset_dir || true)"
    model_dir="$(resolve_mlperf_llama31_8b_model_dir || true)"

    if [[ -n "$dataset_dir" && "$dataset_dir" != "/data" ]]; then
        shopt -s nullglob
        for item in "${dataset_dir}"/*; do
            base=$(basename "$item")
            target="/data/${base}"
            if [[ ! -e "$target" ]]; then
                ln -sf "$item" "$target"
            fi
        done
        shopt -u nullglob
    fi

    if [[ -n "$model_dir" && "$model_dir" != "/model" && ! -e /model ]]; then
        ln -sf "$model_dir" /model
    fi
}

apply_mlperf_runtime_tunables() {
    local model_id="$1"
    local tunables_script=""

    case "$model_id" in
        llama2_70b)
            tunables_script="/workspace/Primus/examples/mlperf/llama2_70b/runtime_tunables.sh"
            ;;
        llama3.1_8b)
            tunables_script="/workspace/Primus/examples/mlperf/llama3.1_8b/runtime_tunables.sh"
            ;;
        *)
            return 0
            ;;
    esac

    if [[ "${RUN_RUNTIME_TUNABLES:-1}" != "1" ]]; then
        echo -e " ${DOT} Skipping runtime tunables (${CYAN}RUN_RUNTIME_TUNABLES=${RUN_RUNTIME_TUNABLES}${RESET})"
        return 0
    fi

    if [[ ! -f "$tunables_script" ]]; then
        echo -e "${YELLOW}⚠ runtime_tunables.sh not found:${RESET} ${CYAN}${tunables_script}${RESET}"
        return 0
    fi

    echo -e " ${CHECK} bash ${CYAN}${tunables_script}${RESET}"
    bash "$tunables_script"
}

run_mlperf_llama31_8b_benchmark() {
    local model_dir="/workspace/Primus/examples/mlperf/llama3.1_8b"
    local config_script="${model_dir}/config_MI355X_1x8x1.sh"
    local run_script="${model_dir}/run_and_time.sh"

    if [[ ! -f "$config_script" ]]; then
        echo -e "${RED}✗ MLPerf config script not found:${RESET} ${CYAN}$config_script${RESET}"
        return 1
    fi

    if [[ ! -f "$run_script" ]]; then
        echo -e "${RED}✗ run_and_time.sh not found:${RESET} ${CYAN}$run_script${RESET}"
        return 1
    fi

    export PRIMUS_PATH="$PRIMUS_ROOT"
    export HF_TOKEN

    if ! prepare_mlperf_llama31_8b_dataset; then
        return 1
    fi

    prepare_mlperf_llama31_8b_data_layout

    apply_mlperf_runtime_tunables "llama3.1_8b"

    cd "$model_dir" || return 1

    echo -e " ${CHECK} source config_MI355X_1x8x1.sh"
    # shellcheck disable=SC1091
    source config_MI355X_1x8x1.sh

    echo -e " ${CHECK} bash run_and_time.sh\n"

    set +e
    bash run_and_time.sh
    local ret_code=$?
    set -e

    return "$ret_code"
}

execute_mlperf_benchmark_run() {
    local model_id="$1"
    local log_file="$2"
    local current="$3"
    local total="$4"

    local model_dir="$PREP_MLPERF_MODEL_DIR"
    local config_script run_script run_exit_code=0

    echo -e "${STAR} ${BOLD}Starting MLPerf Benchmark ${current}/${total}...${RESET}"
    echo -e "   ${DOT} Model: ${CYAN}$model_id${RESET}"
    echo -e "   ${DOT} Backend: ${CYAN}$BACKEND${RESET}"
    echo -e "   ${DOT} Device: ${CYAN}$DEVICE${RESET}"
    echo -e "   ${DOT} Model dir: ${YELLOW}$model_dir${RESET}"
    echo -e "   ${DOT} Log: ${YELLOW}$log_file${RESET}\n"

    if [[ "$model_id" == "llama2_70b" ]]; then
        config_script="/workspace/Primus/examples/mlperf/llama2_70b/config_MI355X_1x8x1.sh"
        run_script="/workspace/Primus/examples/mlperf/llama2_70b/run_and_time.sh"

        if [[ ! -f "$config_script" ]]; then
            echo -e "${RED}✗ MLPerf config script not found:${RESET} ${CYAN}$config_script${RESET}"
            return 1
        fi

        if [[ ! -f "$run_script" ]]; then
            echo -e "${RED}✗ run_and_time.sh not found:${RESET} ${CYAN}$run_script${RESET}"
            return 1
        fi

        echo -e " ${CHECK} export PACKED_DATA_DIR=/data"
        export PACKED_DATA_DIR=/data

        if mlperf_llama2_70b_dataset_ready; then
            echo -e " ${CHECK} MLPerf llama2_70b packed dataset already present under ${CYAN}${MLPERF_LLAMA2_70B_PACKED_DATA_DIR}${RESET}; skipping download"
        else
            echo -e " ${DOT} MLPerf llama2_70b packed dataset not found under ${CYAN}${MLPERF_LLAMA2_70B_PACKED_DATA_DIR}${RESET}; hooks will prepare if needed"
        fi

        ckpt_path="$(mlperf_pretrained_checkpoint_path "$model_id")"
        echo -e " ${CHECK} export PRETRAINED_CHECKPOINT=${ckpt_path}"
        export_pretrained_checkpoint_env "$ckpt_path" "$model_id"

        apply_mlperf_runtime_tunables "llama2_70b"

        mkdir -p /results 2>/dev/null || true

        echo -e " ${CHECK} bash ${CYAN}${run_script}${RESET}\n"

        set +e
        bash "$run_script" 2>&1 | tee "$log_file"
        run_exit_code=${PIPESTATUS[0]}
        set +e
    elif [[ "$model_id" == "llama3.1_8b" ]]; then
        set +e
        run_mlperf_llama31_8b_benchmark 2>&1 | tee "$log_file"
        run_exit_code=${PIPESTATUS[0]}
        set +e
    else
        if [[ ! -d "$model_dir" ]]; then
            echo -e "${RED}✗ MLPerf model directory not found:${RESET} ${CYAN}$model_dir${RESET}"
            return 1
        fi

        if ! config_script=$(resolve_mlperf_config_script "$model_dir"); then
            echo -e "${RED}✗ MLPerf config script not found for ${DEVICE} under:${RESET} ${CYAN}$model_dir${RESET}"
            return 1
        fi

        if [[ ! -f "${model_dir}/run_and_time.sh" ]]; then
            echo -e "${RED}✗ run_and_time.sh not found in:${RESET} ${CYAN}$model_dir${RESET}"
            return 1
        fi

        export PRIMUS_PATH="$PRIMUS_ROOT"
        export HF_TOKEN

        mkdir -p /results 2>/dev/null || true

        echo -e " ${CHECK} Sourcing MLPerf config: ${CYAN}$config_script${RESET}"
        # shellcheck disable=SC1090
        source "$config_script"

        echo -e " ${CHECK} Launching: ${CYAN}bash run_and_time.sh${RESET}"
        echo -e "   ${DOT} EXP=${CYAN}${EXP:-<unset>}${RESET}"
        echo -e "   ${DOT} DATA_PATH=${CYAN}${DATA_PATH:-<unset>}${RESET}\n"

        cd "$model_dir" || return 1

        set +e
        bash run_and_time.sh 2>&1 | tee "$log_file"
        run_exit_code=${PIPESTATUS[0]}
        set +e
    fi

    if mlperf_logging_out_matches_model "$model_id"; then
        {
            echo
            echo "===== MLLOG OUTPUT (/results/mlperf_logging.out) ====="
            cat /results/mlperf_logging.out
        } >> "$log_file"
    elif [[ -f /results/mlperf_logging.out ]]; then
        echo -e " ${DOT} Skipping stale /results/mlperf_logging.out (does not match ${CYAN}$model_id${RESET})"
    fi

    cd "$SCRIPT_DIR" || return 1

    echo
    echo -e "${GREEN}==========================================${RESET}"
    if [[ $run_exit_code -eq 0 ]]; then
        echo -e " ${BOLD}${GREEN}✓ MLPerf Benchmark ${current}/${total} Completed Successfully!${RESET}"
    else
        echo -e " ${BOLD}${YELLOW}⚠ MLPerf Benchmark ${current}/${total} Completed with Exit Code: $run_exit_code${RESET}"
    fi
    echo -e " Log saved at:"
    echo -e "   ${CYAN}$log_file${RESET}"
    echo -e "${GREEN}==========================================${RESET}"
    echo

    return "$run_exit_code"
}

generate_metrics_table() {
    local metrics_script="metrics.py"

    echo
    echo -e "${STAR} ${BOLD}Generating Metrics Table...${RESET}\n"

    if [[ -f "$SCRIPT_DIR/$metrics_script" && ( "$BACKEND" == "megatron" || "$BACKEND" == "megatron_bridge" || "$BACKEND" == "torchtitan" || "$BACKEND" == "mlperf" ) ]]; then
        echo -e " ${CHECK} Running: ${CYAN}python $metrics_script $BACKEND${RESET}\n"
        metrics_output=$(cd "$SCRIPT_DIR" && python "$metrics_script" "$BACKEND")
        metrics_status=$?
        printf '%s\n' "$metrics_output"
        echo
        if [[ $metrics_status -eq 0 ]]; then
            csv_path=$(printf '%s\n' "$metrics_output" | awk '/^  Latest:/{print $2}')
            echo -e " ${CHECK} ${GREEN}Metrics table generated successfully${RESET}"
            if [[ -n "$csv_path" ]]; then
                echo -e " ${DOT} CSV: ${CYAN}$csv_path${RESET}"
            fi
        else
            echo -e " ${RED}✗ Metrics generation failed${RESET}"
        fi
    else
        echo -e " ${RED}✗ Metrics script not found: ${metrics_script:-unknown}${RESET}"
    fi
}

# ------------------------------------------
# Banner
# ------------------------------------------
clear
echo -e "${MAGENTA}"
echo "██████╗ ██████╗ ██╗███╗   ███╗██╗   ██╗███████╗"
echo "██╔══██╗██╔══██╗██║████╗ ████║██║   ██║██╔════╝"
echo "██████╔╝██████╔╝██║██╔████╔██║██║   ██║███████╗"
echo "██╔═══╝ ██╔══██╗██║██║╚██╔╝██║██║   ██║╚════██║"
echo "██║     ██║  ██║██║██║ ╚═╝ ██║╚██████╔╝███████║"
echo "╚═╝     ╚═╝  ╚═╝╚═╝╚═╝     ╚═╝ ╚═════╝ ╚══════╝"
echo -e "${RESET}"
echo -e "           ${BOLD}${CYAN}Auto Benchmarking Tool${RESET}\n"

sleep 0.2

# ------------------------------------------
# 1. BACKEND SELECTION
# ------------------------------------------
echo -e "${STAR} ${BOLD}Choose Backend:${RESET}"
echo -e "  ${DOT} 1) megatron"
echo -e "  ${DOT} 2) torchtitan"
echo -e "  ${DOT} 3) megatron_bridge"
echo -e "  ${DOT} 4) mlperf ${DIM}(MI355X only; not supported on MI300X/MI325X)${RESET}"

echo -en " ${ARROW} Enter number or name: "
read -r BACKEND_IN

case "$BACKEND_IN" in
    1|megatron|MegaTron|MEGATRON)
        BACKEND="megatron"
        BACKEND_BASE_DIR="$MEGATRON_BASE_DIR"
        ;;
    2|torchtitan|TorchTitan|TORCHTITAN)
        BACKEND="torchtitan"
        BACKEND_BASE_DIR="$TORCHTITAN_BASE_DIR"
        ;;
    3|megatron_bridge|MegatronBridge|MEGATRON_BRIDGE|megatron-bridge)
        BACKEND="megatron_bridge"
        BACKEND_BASE_DIR="$MEGATRON_BRIDGE_BASE_DIR"
        ;;
    4|mlperf|MLPerf|MLPERF)
        BACKEND="mlperf"
        BACKEND_BASE_DIR="$MLPERF_BASE_DIR"
        ;;
    *)
        echo -e "${RED}✗ Invalid backend: $BACKEND_IN${RESET}"
        exit 1
        ;;
esac

echo -e " ${CHECK} Backend selected: ${GREEN}$BACKEND${RESET}\n"
sleep 0.2

# ------------------------------------------
# 2. DEVICE DETECTION
# ------------------------------------------
echo -e "${STAR} ${BOLD}Detecting Device...${RESET}"

ROCMINFO=""
for candidate in \
    "$(command -v rocminfo 2>/dev/null)" \
    "/opt/rocm/bin/rocminfo" \
    "${ROCM_PATH:+$ROCM_PATH/bin/rocminfo}" \
    /opt/rocm-*/bin/rocminfo; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
        ROCMINFO="$candidate"
        break
    fi
done

is_valid_device() {
    local candidate="$1"
    for dev in "${VALID_DEVICES[@]}"; do
        if [[ "$candidate" == "$dev" ]]; then
            return 0
        fi
    done
    return 1
}

if [[ -z "$ROCMINFO" ]]; then
    echo -e " ${YELLOW}⚠ rocminfo not found (checked PATH, /opt/rocm/bin, ROCM_PATH)${RESET}"
    DEVICE=""
else
    echo -e " ${DOT} Using rocminfo: ${CYAN}$ROCMINFO${RESET}"
    DEVICE=$("$ROCMINFO" 2>/dev/null | grep -oE 'MI3[0-9]{2}X' | head -n1)
    if [[ -z "$DEVICE" ]]; then
        DEVICE=$("$ROCMINFO" 2>/dev/null | grep "AMD Instinct" | head -n1 | awk '{print $5}')
    fi
    echo -e " ${DOT} Device found: ${CYAN}$DEVICE${RESET}"
fi

if ! is_valid_device "$DEVICE"; then
  if [[ -n "$ROCMINFO" ]]; then
    ARCH=$("$ROCMINFO" 2>/dev/null | grep -o 'gfx942\|gfx950' | head -n 1 | tr -d '[:space:]')
  else
    ARCH=""
  fi
  case "$ARCH" in
    "gfx950") DEVICE="MI355X" ;;
    # gfx942 is shared by MI300X and MI325X; require manual selection if marketing name is missing
    *) DEVICE="" ;;
  esac
fi

if [[ -z "$DEVICE" ]]; then
    echo -e "${RED}✗ Could not detect device automatically${RESET}"
    echo -e "${STAR} ${BOLD}Please select Device manually:${RESET}"
    echo -e "  ${DOT} 1) MI300X"
    echo -e "  ${DOT} 2) MI325X"
    echo -e "  ${DOT} 3) MI355X"

    echo -en " ${ARROW} Enter number or name: "
    read -r DEV_IN

    case "$DEV_IN" in
        1|MI300X|mi300x|Mi300x)
            DEVICE="MI300X"
            ;;
        2|MI325X|mi325x|Mi325x)
            DEVICE="MI325X"
            ;;
        3|MI355X|mi355x|Mi355x)
            DEVICE="MI355X"
            ;;
        *)
            echo -e "${RED}✗ Invalid device: $DEV_IN${RESET}"
            exit 1
            ;;
    esac
fi

echo -e " ${CHECK} GPU Device: ${GREEN}$DEVICE${RESET}\n"

if [[ "$BACKEND" == "mlperf" && ( "$DEVICE" == "MI300X" || "$DEVICE" == "MI325X" ) ]]; then
    echo -e "${RED}✗ MLPerf is not supported on ${DEVICE}.${RESET}"
    echo -e "   ${DOT} MLPerf benchmarks require ${CYAN}MI355X${RESET}."
    echo -e "   ${DOT} Use megatron, torchtitan, or megatron_bridge on MI300X/MI325X instead.\n"
    exit 1
fi

sleep 0.2

# ------------------------------------------
# 2.5. SET CONFIG DIRECTORY / MLPerf MODELS
# ------------------------------------------
if [[ "$BACKEND" == "mlperf" ]]; then
    echo -e " ${CHECK} MLPerf examples directory: ${CYAN}$MLPERF_BASE_DIR${RESET}"
    echo -e " ${CHECK} MLPerf supported device: ${CYAN}MI355X${RESET} (not supported on MI300X/MI325X)"
else
    CONFIG_DIR="${BACKEND_BASE_DIR}/${DEVICE}"
    echo -e " ${CHECK} Config directory set to: ${CYAN}$CONFIG_DIR${RESET}\n"
fi
sleep 0.2

# ------------------------------------------
# 3. MODEL SELECTION
# ------------------------------------------
SELECTED_CONFIGS=()
SELECTED_MLPERF_MODELS=()

if [[ "$BACKEND" == "mlperf" ]]; then
    echo -e "${STAR} ${BOLD}Available MLPerf Models:${RESET} (${CYAN}$BACKEND${RESET} / ${CYAN}$DEVICE${RESET})"
    i=1
    for model in "${MLPERF_MODELS[@]}"; do
        if [[ -d "${MLPERF_BASE_DIR}/${model}" ]]; then
            echo -e "  ${DOT} ${i}) ${model}"
            ((i++))
        fi
    done
    echo

    mapfile -t AVAILABLE_MLPERF_MODELS < <(
        for model in "${MLPERF_MODELS[@]}"; do
            if [[ -d "${MLPERF_BASE_DIR}/${model}" ]]; then
                echo "$model"
            fi
        done
    )

    if [[ ${#AVAILABLE_MLPERF_MODELS[@]} -eq 0 ]]; then
        echo -e "${RED}No MLPerf models found in $MLPERF_BASE_DIR${RESET}"
        exit 1
    fi

    echo -en " ${ARROW} Select model number(s) (comma-separated, range, or 'all'): "
    echo -e "${DIM}(Examples: 1 or 1,2 or all)${RESET}"
    echo -en " ${ARROW} "
    read -r CFG_NUM

    if [[ "$CFG_NUM" == "all" ]]; then
        SELECTED_MLPERF_MODELS=("${AVAILABLE_MLPERF_MODELS[@]}")
    elif [[ "$CFG_NUM" =~ ^[0-9]+-[0-9]+$ ]]; then
        START="${CFG_NUM%%-*}"
        END="${CFG_NUM##*-}"
        if [[ $START -lt 1 || $END -gt ${#AVAILABLE_MLPERF_MODELS[@]} || $START -gt $END ]]; then
            echo -e "${RED}✗ Invalid range: $START-$END${RESET}"
            exit 1
        fi
        for ((i=START; i<=END; i++)); do
            SELECTED_MLPERF_MODELS+=("${AVAILABLE_MLPERF_MODELS[$i-1]}")
        done
    else
        _saved_ifs=$IFS
        IFS=',' read -ra CFG_NUMS <<< "$CFG_NUM"
        IFS=$_saved_ifs
        for num in "${CFG_NUMS[@]}"; do
            num=$(echo "$num" | xargs)
            if [[ $num -ge 1 && $num -le ${#AVAILABLE_MLPERF_MODELS[@]} ]]; then
                SELECTED_MLPERF_MODELS+=("${AVAILABLE_MLPERF_MODELS[$num-1]}")
            else
                echo -e "${RED}✗ Invalid model number: $num${RESET}"
                exit 1
            fi
        done
    fi

    SELECTED_CONFIG_COUNT=${#SELECTED_MLPERF_MODELS[@]}
    echo -e " ${CHECK} Selected ${GREEN}${SELECTED_CONFIG_COUNT}${RESET} MLPerf model(s):"
    for model in "${SELECTED_MLPERF_MODELS[@]}"; do
        echo -e "    ${DOT} ${model}"
    done
    echo
else
    echo -e "${STAR} ${BOLD}Available Model Configs:${RESET} (${CYAN}$BACKEND${RESET} / ${CYAN}$DEVICE${RESET})"

    mapfile -t CONFIG_LIST < <(find "$CONFIG_DIR" -name "*.yaml" -type f | sort -u)

    if [[ ${#CONFIG_LIST[@]} -eq 0 ]]; then
        echo -e "${RED}No configs found in $CONFIG_DIR${RESET}"
        exit 1
    fi

    declare -A SEEN_MODELS
    UNIQUE_CONFIGS=()

    for cfg in "${CONFIG_LIST[@]}"; do
        model_name=$(basename "$cfg" .yaml)
        if [[ -z "${SEEN_MODELS[$model_name]}" ]]; then
            SEEN_MODELS[$model_name]=1
            UNIQUE_CONFIGS+=("$cfg")
        fi
    done

    i=1
    for cfg in "${UNIQUE_CONFIGS[@]}"; do
        echo -e "  ${DOT} ${i}) $(basename "$cfg")"
        ((i++))
    done
    echo

    CONFIG_LIST=("${UNIQUE_CONFIGS[@]}")

    echo -en " ${ARROW} Select config number(s) (comma-separated, range, or 'all'): "
    echo -e "${DIM}(Examples: 1,3,5 or 4-8 or all)${RESET}"
    echo -en " ${ARROW} "
    read -r CFG_NUM

    if [[ "$CFG_NUM" == "all" ]]; then
        SELECTED_CONFIGS=("${CONFIG_LIST[@]}")
    elif [[ "$CFG_NUM" =~ ^[0-9]+-[0-9]+$ ]]; then
        START="${CFG_NUM%%-*}"
        END="${CFG_NUM##*-}"

        if [[ $START -lt 1 || $END -gt ${#CONFIG_LIST[@]} || $START -gt $END ]]; then
            echo -e "${RED}✗ Invalid range: $START-$END${RESET}"
            exit 1
        fi

        for ((i=START; i<=END; i++)); do
            SELECTED_CONFIGS+=("${CONFIG_LIST[$i-1]}")
        done
    else
        _saved_ifs=$IFS
        IFS=',' read -ra CFG_NUMS <<< "$CFG_NUM"
        IFS=$_saved_ifs

        for num in "${CFG_NUMS[@]}"; do
            num=$(echo "$num" | xargs)

            if [[ $num -ge 1 && $num -le ${#CONFIG_LIST[@]} ]]; then
                SELECTED_CONFIGS+=("${CONFIG_LIST[$num-1]}")
            else
                echo -e "${RED}✗ Invalid config number: $num${RESET}"
                exit 1
            fi
        done
    fi

    SELECTED_CONFIG_COUNT=${#SELECTED_CONFIGS[@]}
    echo -e " ${CHECK} Selected ${GREEN}${SELECTED_CONFIG_COUNT}${RESET} configs:"
    for cfg in "${SELECTED_CONFIGS[@]}"; do
        echo -e "    ${DOT} $(basename "$cfg")"
    done
    echo
fi
sleep 0.2

if [[ "$BACKEND" != "mlperf" ]]; then
# ------------------------------------------
# 2.5. VIEW CONFIGURATION PARAMETERS
# ------------------------------------------
echo -e "${STAR} ${BOLD}View Configuration Parameters?${RESET}"
echo -en " ${ARROW} (y/n): "
read -r VIEW_PARAMS

if [[ "$VIEW_PARAMS" == "y" || "$VIEW_PARAMS" == "Y" ]]; then
    for cfg in "${SELECTED_CONFIGS[@]}"; do
        echo -e "\n${CYAN}${BOLD}Parameters in $(basename "$cfg"):${RESET}"
        echo -e "${DIM}-----------------------------------${RESET}"
        grep -v "^#" "$cfg" | grep -v "^$"
        echo -e "${DIM}-----------------------------------${RESET}"
    done
    echo
fi

# ------------------------------------------
# 2.6. EDIT CONFIG FILES (once, before all runs)
# ------------------------------------------
declare -A EDITED_CONFIGS

echo -e "${STAR} ${BOLD}Edit configuration files before running?${RESET}"
echo -e "   ${DOT} ${DIM}Edits use working copies; original repo configs are never modified.${RESET}"
echo -en " ${ARROW} (y/n): "
read -r EDIT_CONFIGS

if [[ "$EDIT_CONFIGS" == "y" || "$EDIT_CONFIGS" == "Y" ]]; then
    if [[ ${#SELECTED_CONFIGS[@]} -gt 1 ]]; then
        echo -e "\n${CYAN}${BOLD}Selected models:${RESET}"
        i=1
        for cfg in "${SELECTED_CONFIGS[@]}"; do
            echo -e "  ${DOT} ${i}) $(basename "$cfg")"
            ((i++))
        done
        echo

        echo -e " ${DOT} Enter model numbers to edit (comma-separated, or 'all'): "
        echo -en " ${ARROW} "
        read -r EDIT_SELECTION

        if [[ "$EDIT_SELECTION" == "all" ]]; then
            MODELS_TO_EDIT=("${!SELECTED_CONFIGS[@]}")
        else
            _saved_ifs=$IFS
            IFS=',' read -ra EDIT_NUMS <<< "$EDIT_SELECTION"
            IFS=$_saved_ifs
            MODELS_TO_EDIT=()
            for num in "${EDIT_NUMS[@]}"; do
                num=$(echo "$num" | xargs)
                if [[ $num -ge 1 && $num -le ${#SELECTED_CONFIGS[@]} ]]; then
                    MODELS_TO_EDIT+=($((num-1)))
                fi
            done
        fi

        for idx in "${MODELS_TO_EDIT[@]}"; do
            edit_config_interactively "${SELECTED_CONFIGS[$idx]}"
        done
    else
        edit_config_interactively "${SELECTED_CONFIGS[0]}"
    fi
fi
fi

# ------------------------------------------
# 4. ENVIRONMENT SETUP
# ------------------------------------------
echo -e "${STAR} ${BOLD}Setting up environment...${RESET}"

configure_git_safe_directories
echo -e " ${CHECK} Configured git safe directories for ${CYAN}$PRIMUS_ROOT${RESET}"

# Prompt for HuggingFace token
echo -en " ${ARROW} Enter HuggingFace Token: "
read -r -s HF_TOKEN
echo
export HF_TOKEN
echo -e " ${CHECK} HuggingFace token set\n"

if [[ "$BACKEND" == "mlperf" ]]; then
    export RUN_RUNTIME_TUNABLES="${RUN_RUNTIME_TUNABLES:-1}"
    echo -e " ${CHECK} MLPerf llama3.1_8b dataset root: ${CYAN}${MLPERF_LLAMA31_8B_ROOT}${RESET}"
    echo -e " ${CHECK} MLPerf llama2_70b uses fixed paths: ${CYAN}PACKED_DATA_DIR=/data${RESET}, ${CYAN}PRETRAINED_CHECKPOINT=/data/megatron_checkpoints/Llama-2-70b-hf${RESET}"
    echo -e " ${CHECK} MLPerf runtime tunables: ${CYAN}RUN_RUNTIME_TUNABLES=${RUN_RUNTIME_TUNABLES}${RESET} (set to 0 to skip)\n"
fi

sleep 0.2

# ------------------------------------------
# 5. RUN BENCHMARK(S)
# ------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="${SCRIPT_DIR}/results/logs_${BACKEND}"
mkdir -p "$LOG_DIR"

if [[ "$BACKEND" == "mlperf" ]]; then
    TOTAL_CONFIGS=${#SELECTED_MLPERF_MODELS[@]}
else
    TOTAL_CONFIGS=${#SELECTED_CONFIGS[@]}
fi
CURRENT=1

echo -e "${INFO} ${BOLD}Total configurations to run: ${TOTAL_CONFIGS}${RESET}"
echo -e "${INFO} ${BOLD}Configuration list:${RESET}"
if [[ "$BACKEND" == "mlperf" ]]; then
    for i in "${!SELECTED_MLPERF_MODELS[@]}"; do
        echo -e "   ${DOT} $((i+1)). ${SELECTED_MLPERF_MODELS[$i]}"
    done
else
    for i in "${!SELECTED_CONFIGS[@]}"; do
        echo -e "   ${DOT} $((i+1)). $(basename "${SELECTED_CONFIGS[$i]}")"
    done
fi
echo

if [[ "$BACKEND" == "mlperf" ]]; then
    for MLPERF_MODEL in "${SELECTED_MLPERF_MODELS[@]}"; do
        echo -e "\n${MAGENTA}${BOLD}╔════════════════════════════════════════════════════════════╗${RESET}"
        echo -e "${MAGENTA}${BOLD}║  LOOP ITERATION: ${CURRENT}/${TOTAL_CONFIGS}${RESET}"
        echo -e "${MAGENTA}${BOLD}║  MLPerf MODEL: ${MLPERF_MODEL}${RESET}"
        echo -e "${MAGENTA}${BOLD}╚════════════════════════════════════════════════════════════╝${RESET}\n"

        prepare_mlperf_artifacts "$MLPERF_MODEL"
        echo -e "   ${DOT} Run label: ${CYAN}$PREP_RUN_LABEL${RESET}"

        execute_mlperf_benchmark_run \
            "$MLPERF_MODEL" \
            "$PREP_LOG_FILE" \
            "$CURRENT" \
            "$TOTAL_CONFIGS" || true

        CURRENT=$((CURRENT + 1))

        if [[ $CURRENT -le $TOTAL_CONFIGS ]]; then
            echo -e "${YELLOW}Preparing next benchmark...${RESET}\n"
            echo -e "${INFO} ${BOLD}Next: Model ${CURRENT}/${TOTAL_CONFIGS}${RESET}\n"
            sleep 2
        fi
    done
else
for CFG_FILE in "${SELECTED_CONFIGS[@]}"; do
    echo -e "\n${MAGENTA}${BOLD}╔════════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${MAGENTA}${BOLD}║  LOOP ITERATION: ${CURRENT}/${TOTAL_CONFIGS}${RESET}"
    echo -e "${MAGENTA}${BOLD}║  CONFIG FILE: $(basename "$CFG_FILE")${RESET}"
    echo -e "${MAGENTA}${BOLD}╚════════════════════════════════════════════════════════════╝${RESET}\n"

    prepare_benchmark_artifacts "$CFG_FILE"

    if [[ -n "${EDITED_CONFIGS[$CFG_FILE]:-}" ]]; then
        echo -e "${INFO} ${BOLD}Using edited working copy for ${CYAN}$PREP_MODEL_NAME${RESET}"
    fi
    echo -e "   ${DOT} Run label: ${CYAN}$PREP_RUN_LABEL${RESET}"
    echo -e "   ${DOT} Working config: ${CYAN}$PREP_WORKING_CONFIG${RESET}"

    execute_benchmark_run \
        "$PREP_CFG_FILE" \
        "$PREP_WORKING_CONFIG" \
        "$PREP_LOG_FILE" \
        "$PREP_MODEL_NAME" \
        "$CURRENT" \
        "$TOTAL_CONFIGS" || true

    CURRENT=$((CURRENT + 1))

    if [[ $CURRENT -le $TOTAL_CONFIGS ]]; then
        echo -e "${YELLOW}Preparing next benchmark...${RESET}\n"
        echo -e "${INFO} ${BOLD}Next: Config ${CURRENT}/${TOTAL_CONFIGS}${RESET}\n"
        sleep 2
    fi
done
fi

echo
echo -e "${MAGENTA}${BOLD}=========================================${RESET}"
echo -e "${MAGENTA}${BOLD}  All ${TOTAL_CONFIGS} benchmarks completed!${RESET}"
echo -e "${MAGENTA}${BOLD}=========================================${RESET}"

# ------------------------------------------
# 6. GENERATE METRICS TABLE
# ------------------------------------------
generate_metrics_table
