#!/usr/bin/env bash
# Removed: set -e (allow script to continue on errors in benchmarks)

if [[ -z "${BASH_VERSION:-}" ]]; then
    echo "This script must be run with bash (not sh)." >&2
    exit 1
fi

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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_ROOT="${PRIMUS_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
MEGATRON_BASE_DIR="${PRIMUS_ROOT}/examples/megatron/configs"
MEGATRON_BRIDGE_BASE_DIR="${PRIMUS_ROOT}/examples/megatron_bridge/configs"
TORCHTITAN_BASE_DIR="${PRIMUS_ROOT}/examples/torchtitan/configs"
MAXTEXT_BASE_DIR="${PRIMUS_ROOT}/examples/maxtext/configs"
MAXDIFFUSION_BASE_DIR="${PRIMUS_ROOT}/examples/maxdiffusion/configs"
MLPERF_BASE_DIR="${PRIMUS_ROOT}/examples/mlperf"
PRIMUS_CLI="${PRIMUS_ROOT}/primus-cli"
VALID_DEVICES=(MI300X MI325X MI355X)
MLPERF_MODELS=(llama3.1_8b llama2_70b gpt_oss_20b flux1)

running_inside_container() {
    if [[ -f /.dockerenv || -f /run/.containerenv ]]; then
        return 0
    fi
    if [[ -r /proc/1/cgroup ]] && grep -Eq 'docker|containerd|kubepods|podman|lxc' /proc/1/cgroup; then
        return 0
    fi
    if [[ -r /proc/1/environ ]] && tr '\0' '\n' < /proc/1/environ | grep -q '^container='; then
        return 0
    fi
    return 1
}

if [[ -n "${PRIMUS_MODE:-}" ]]; then
    AUTO_BENCHMARK_MODE="${PRIMUS_MODE}"
elif running_inside_container; then
    AUTO_BENCHMARK_MODE="direct"
else
    AUTO_BENCHMARK_MODE="container"
fi

if [[ "$AUTO_BENCHMARK_MODE" != "direct" && "$AUTO_BENCHMARK_MODE" != "container" ]]; then
    echo "[ERROR] PRIMUS_MODE must be 'direct' or 'container', got: $AUTO_BENCHMARK_MODE" >&2
    exit 1
fi

# Bootstrap socket for this tool only. runner/helpers/envs/get_ip_interface.sh
# stays as it is for other Primus launches. Keep a name the user already
# exported when that interface is up here. Otherwise pick the host's default
# route, then any other up interface. Loopback and virtual devices such as
# docker0 are not used. GPU collectives still use NCCL_IB_HCA.
autobench_is_virtual_interface() {
    case "$1" in
        lo|docker*|veth*|virbr*|br-*|cni*|flannel*|kube*|podman*|tun*|tap*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

autobench_interface_is_usable() {
    local dev="$1"

    [[ -n "$dev" ]] || return 1
    autobench_is_virtual_interface "$dev" && return 1
    ip -4 -o addr show dev "$dev" scope global up 2>/dev/null | grep -q .
}

autobench_select_socket_interface() {
    local candidate="${1:-}"
    local default_dev dev

    if autobench_interface_is_usable "$candidate"; then
        printf '%s\n' "$candidate"
        return 0
    fi

    default_dev="$(ip -4 route show default 2>/dev/null | awk '/^default/ {for (i = 1; i <= NF; i++) if ($i == "dev") {print $(i + 1); exit}}')"
    if autobench_interface_is_usable "$default_dev"; then
        printf '%s\n' "$default_dev"
        return 0
    fi

    while read -r dev; do
        if autobench_interface_is_usable "$dev"; then
            printf '%s\n' "$dev"
            return 0
        fi
    done < <(ip -4 -o addr show scope global up 2>/dev/null | awk '{print $2}' | awk '!seen[$0]++')

    return 1
}

if NCCL_SOCKET_IFNAME="$(autobench_select_socket_interface "${NCCL_SOCKET_IFNAME:-}")"; then
    export NCCL_SOCKET_IFNAME
else
    unset NCCL_SOCKET_IFNAME
fi
if GLOO_SOCKET_IFNAME="$(autobench_select_socket_interface "${GLOO_SOCKET_IFNAME:-}")"; then
    export GLOO_SOCKET_IFNAME
else
    unset GLOO_SOCKET_IFNAME
fi

if [[ ! -f "$PRIMUS_CLI" && -f "${PRIMUS_ROOT}/runner/primus-cli" ]]; then
    PRIMUS_CLI="${PRIMUS_ROOT}/runner/primus-cli"
fi

if [[ ! -x "$PRIMUS_CLI" && ! -f "$PRIMUS_CLI" ]]; then
    echo "[ERROR] primus-cli not found at $PRIMUS_CLI" >&2
    exit 1
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

sanitize_docker_image() {
    local image="$1"

    image="${image#"${image%%[![:space:]]*}"}"
    image="${image%"${image##*[![:space:]]}"}"
    while [[ "$image" == :* ]]; do
        image="${image#:}"
        image="${image#"${image%%[![:space:]]*}"}"
    done
    printf '%s\n' "$image"
}

docker_image_is_valid() {
    local image="$1"

    [[ -n "$image" && "$image" != *" "* && "$image" != *$'\t'* ]] || return 1
    [[ "$image" =~ ^[A-Za-z0-9][A-Za-z0-9._/-]*(:[A-Za-z0-9_][A-Za-z0-9._-]*)?$ ]]
}

prompt_docker_image() {
    local answer=""

    if [[ -n "${DOCKER_IMAGE:-}" ]]; then
        answer="$(sanitize_docker_image "$DOCKER_IMAGE")"
    elif [[ "${AUTO_BENCHMARK_NONINTERACTIVE:-0}" == "1" ]]; then
        echo -e "${RED}✗ Docker image is required.${RESET}"
        exit 1
    else
        echo -en " ${ARROW} Enter Docker image: "
        read -r answer
        answer="$(sanitize_docker_image "$answer")"
    fi

    if [[ -z "$answer" ]]; then
        echo -e "${RED}✗ Docker image is required.${RESET}"
        exit 1
    fi
    if ! docker_image_is_valid "$answer"; then
        echo -e "${RED}✗ Docker image is not a valid reference: ${answer}${RESET}"
        exit 1
    fi
    DOCKER_IMAGE="$answer"
    export DOCKER_IMAGE
}

launcher_error_is_fatal() {
    local log_file="$1"

    [[ -f "$log_file" ]] || return 1
    grep -Eq 'invalid reference format|Cannot connect to the Docker daemon|permission denied while trying to connect to the Docker daemon' "$log_file"
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

resolve_release_config() {
    local base_dir="$1"
    local gpu="$2"
    local name="$3"
    local primary="${base_dir}/${gpu}/${name}"

    if [[ "$gpu" == "MI325X" && ! -f "$primary" ]]; then
        echo "${base_dir}/MI300X/${name}"
    else
        echo "$primary"
    fi
}

load_release_configs() {
    local gpu="$1"
    local backend="$2"
    local name
    local -a names=()

    # This curated release preset intentionally lives in the auto-benchmark
    # tool. It mirrors the Docker release coverage without invoking tools/perf.
    case "$backend" in
        megatron)
            names=(
                deepseek_v2_lite-BF16-pretrain.yaml
                gpt_oss_20B-BF16-pretrain.yaml
                gpt_oss_20B-FP8-pretrain.yaml
                llama3.1_8B-BF16-pretrain.yaml
                llama3.1_8B-FP8-pretrain.yaml
                llama3.1_70B-BF16-pretrain.yaml
                llama3.1_70B-FP8-pretrain.yaml
                llama3.3_70B-BF16-pretrain.yaml
                llama3.3_70B-FP8-pretrain.yaml
                mixtral_8x7B_v0.1-BF16-pretrain.yaml
                qwen2.5_7B-BF16-pretrain.yaml
                qwen2.5_7B-FP8-pretrain.yaml
                qwen2.5_72B-BF16-pretrain.yaml
                qwen3_30B_A3B-BF16-pretrain.yaml
                qwen3_30B_A3B-FP8-pretrain.yaml
            )
            if [[ "$gpu" == "MI355X" ]]; then
                names+=(
                    llama3.1_8B-MXFP8-pretrain.yaml
                    llama3.1_8B-MXFP4-pretrain.yaml
                )
            fi
            if [[ "$gpu" == "MI300X" || "$gpu" == "MI325X" || "$gpu" == "MI355X" ]]; then
                names+=(
                    mamba_370M-pretrain.yaml
                    zebra_llama_mamba_1B_BF16-pretrain.yaml
                    zebra_llama_mamba_3B_BF16-pretrain.yaml
                    zebra_llama_mamba_8B_BF16-pretrain.yaml
                    gdn_1B_BF16-pretrain.yaml
                    kda_1B_BF16-pretrain.yaml
                    zebra_llama_gdn_1B_BF16-pretrain.yaml
                    zebra_llama_kda_1B_BF16-pretrain.yaml
                )
            fi
            for name in "${names[@]}"; do
                resolve_release_config "$MEGATRON_BASE_DIR" "$gpu" "$name"
            done
            ;;
        torchtitan)
            for name in \
                deepseek_v3_16b-BF16-pretrain.yaml \
                llama3.1_8B-BF16-pretrain.yaml \
                llama3.1_8B-FP8-pretrain.yaml \
                llama3.1_70B-BF16-pretrain.yaml \
                llama3.1_70B-FP8-pretrain.yaml; do
                resolve_release_config "$TORCHTITAN_BASE_DIR" "$gpu" "$name"
            done
            ;;
        megatron_bridge)
            for name in qwen3_32b_lora_posttrain.yaml qwen3_32b_sft_posttrain.yaml; do
                resolve_release_config "$MEGATRON_BRIDGE_BASE_DIR" "$gpu" "$name"
            done
            ;;
        maxtext)
            # BF16 on MI300X, MI325X, and MI355X.
            # Llama 3 70B, Llama 3.3 70B, and Mixtral 8x22B stay BF16 on MI300X.
            local -a bf16_names=(
                llama2_7B
                llama2_70B
                llama3_8B
                llama3_70B
                deepseek_v2_16B
                mixtral_8x7B
                llama3.3_70B
                qwen3_14B
                qwen3_30B_A3B
                gemma4_26B
                gemma4_31B
                mixtral_8x22B
            )
            # Nanoo FP8 on MI300X and MI325X. No Llama 70B or Mixtral 8x22B.
            local -a nanoo_names=(
                llama2_7B
                llama2_70B
                llama3_8B
                deepseek_v2_16B
                mixtral_8x7B
                qwen3_14B
                qwen3_30B_A3B
                gemma4_26B
                gemma4_31B
            )
            # FP8 on MI355X, including both 70B Llama models. Mixtral 8x22B stays BF16.
            local -a fp8_names=(
                llama2_7B
                llama2_70B
                llama3_8B
                llama3_70B
                llama3.3_70B
                deepseek_v2_16B
                mixtral_8x7B
                qwen3_14B
                qwen3_30B_A3B
                gemma4_26B
                gemma4_31B
            )
            local quant_tag
            local -a quant_names=()
            for name in "${bf16_names[@]}"; do
                resolve_release_config "$MAXTEXT_BASE_DIR" "$gpu" "${name}-bf16-pretrain.yaml"
            done
            if [[ "$gpu" == "MI355X" ]]; then
                quant_tag="fp8"
                quant_names=("${fp8_names[@]}")
            else
                quant_tag="nanoo_fp8"
                quant_names=("${nanoo_names[@]}")
            fi
            for name in "${quant_names[@]}"; do
                resolve_release_config "$MAXTEXT_BASE_DIR" "$gpu" "${name}-${quant_tag}-pretrain.yaml"
            done
            ;;
        maxdiffusion)
            for name in flux_dev-pretrain.yaml wan2.1_1.3b-pretrain.yaml wan2.1_14b-pretrain.yaml; do
                resolve_release_config "$MAXDIFFUSION_BASE_DIR" "$gpu" "$name"
            done
            ;;
    esac
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

run_bounded() {
    if command -v timeout >/dev/null 2>&1; then
        timeout 20 "$@"
    else
        "$@"
    fi
}

gpu_tool_path() {
    local name="$1"
    local fallback="$2"

    if command -v "$name" >/dev/null 2>&1; then
        command -v "$name"
        return 0
    fi
    if [[ -x "$fallback" ]]; then
        printf '%s\n' "$fallback"
        return 0
    fi
    return 1
}

select_gpu_tool() {
    local bin

    if [[ "${GPU_SMI_PROBED:-0}" == "1" ]]; then
        [[ -n "${GPU_SMI_BIN:-}" ]]
        return
    fi

    GPU_SMI_PROBED=1
    GPU_SMI_BIN=""
    GPU_SMI_KIND=""

    if bin="$(gpu_tool_path amd-smi /opt/rocm/bin/amd-smi)" \
        && run_bounded "$bin" version >/dev/null 2>&1; then
        GPU_SMI_BIN="$bin"
        GPU_SMI_KIND="amd-smi"
        return 0
    fi
    if bin="$(gpu_tool_path rocm-smi /opt/rocm/bin/rocm-smi)" \
        && run_bounded "$bin" --showdriverversion >/dev/null 2>&1; then
        GPU_SMI_BIN="$bin"
        GPU_SMI_KIND="rocm-smi"
        return 0
    fi
    return 1
}

append_snapshot_command() {
    local label="$1"
    shift

    echo "----- ${label} -----"
    if ! run_bounded "$@" 2>&1; then
        echo "(unavailable)"
    fi
    echo
}

capture_system_snapshot() {
    local dest="${SCRIPT_DIR}/results/system_snapshot.txt"
    local git_commit git_branch git_dirty="unknown"

    mkdir -p "${SCRIPT_DIR}/results"
    SUITE_SNAPSHOT_FILE="$dest"

    git_commit="$(git -C "$PRIMUS_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    git_branch="$(git -C "$PRIMUS_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    if git -C "$PRIMUS_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git_dirty="$(git -C "$PRIMUS_ROOT" status --porcelain 2>/dev/null | wc -l | tr -d '[:space:]')"
    fi
    SUITE_GIT_COMMIT="$git_commit"
    collect_setup_identity

    {
        echo "captured_at=$(date -Iseconds)"
        echo "hostname=$(hostname 2>/dev/null || echo unknown)"
        echo "kernel=$(uname -srm 2>/dev/null || echo unknown)"
        echo "git_commit=${git_commit}"
        echo "primus_commit=${SUITE_GIT_COMMIT}"
        echo "submodule_pins=${SUITE_SUBMODULE_PINS}"
        echo "gpu=${SUITE_GPU}"
        echo "rocm_version=${SUITE_ROCM}"
        echo "framework_versions=${SUITE_FRAMEWORK}"
        echo "image_digest=${SUITE_IMAGE_DIGEST}"
        echo "manifest=${MANIFEST_PATH}"
        echo "git_branch=${git_branch}"
        echo "git_dirty_files=${git_dirty}"
        echo "device=${DEVICE:-}"
        echo "launch_mode=${AUTO_BENCHMARK_MODE}"
        echo "docker_image=${DOCKER_IMAGE:-}"
        echo "docker_release=${DOCKER_RELEASE:-}"
        echo "inside_container=$(running_inside_container && echo yes || echo no)"
        echo "slurm_job_id=${SLURM_JOB_ID:-}"
        echo "slurm_nodelist=${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-}}"
        echo "hf_token_set=$([[ -n "${HF_TOKEN:-}" ]] && echo yes || echo no)"
        echo "nccl_socket_ifname=${NCCL_SOCKET_IFNAME:-}"
        echo "gloo_socket_ifname=${GLOO_SOCKET_IFNAME:-}"
        echo "nccl_ib_hca=${NCCL_IB_HCA:-}"
        echo

        echo "----- rocm version -----"
        echo "${SUITE_ROCM:-unavailable}"
        echo

        if select_gpu_tool; then
            echo "gpu_tool=${GPU_SMI_KIND}"
            echo
            if [[ "$GPU_SMI_KIND" == "amd-smi" ]]; then
                append_snapshot_command "gpu firmware" "$GPU_SMI_BIN" version
                append_snapshot_command "gpu board" "$GPU_SMI_BIN" static --vbios --board --asic --driver
            else
                append_snapshot_command "gpu firmware" "$GPU_SMI_BIN" --showdriverversion --showvbios --showproductname
            fi
        else
            echo "----- gpu firmware -----"
            echo "(amd-smi and rocm-smi unavailable)"
            echo
        fi

        if [[ "$AUTO_BENCHMARK_MODE" == "container" && -n "${DOCKER_IMAGE:-}" ]]; then
            # The variables in this snippet intentionally expand in the child shell.
            # shellcheck disable=SC2016
            append_snapshot_command "container image digest" bash -c '
                image="$1"
                if command -v docker >/dev/null 2>&1; then
                    cli=docker
                elif command -v podman >/dev/null 2>&1; then
                    cli=podman
                else
                    echo "no container cli"
                    exit 1
                fi
                "$cli" image inspect --format "id={{.Id}}{{println}}digests={{json .RepoDigests}}" "$image"
            ' bash "$DOCKER_IMAGE"
        fi
    } > "$dest"

    echo -e " ${CHECK} System snapshot: ${CYAN}${dest}${RESET}"
    echo -e " ${CHECK} Run manifest: ${CYAN}${MANIFEST_PATH}${RESET}"
}

write_reproduce_script() {
    local dest="$1"
    local exit_code="$2"
    local kind="$3"
    local detail="$4"
    local q_root q_cli q_image q_config q_nccl q_gloo model_dir config_path

    q_root="$(printf '%q' "$PRIMUS_ROOT")"
    q_cli="$(printf '%q' "$PRIMUS_CLI")"
    q_image="$(printf '%q' "${DOCKER_IMAGE:-}")"
    q_config="$(printf '%q' "${PREP_WORKING_CONFIG:-}")"
    q_nccl="$(printf '%q' "${NCCL_SOCKET_IFNAME:-}")"
    q_gloo="$(printf '%q' "${GLOO_SOCKET_IFNAME:-}")"

    {
        echo "#!/usr/bin/env bash"
        echo "# Reproduce one Primus auto-benchmark result."
        echo "# Model: ${PREP_MODEL_NAME:-unknown}"
        echo "# Backend: ${BACKEND:-unknown}  Device: ${DEVICE:-unknown}"
        echo "# Git commit: ${SUITE_GIT_COMMIT:-unknown}"
        echo "# Original exit code: ${exit_code}"
        echo "# Export HF_TOKEN before running. The token is not stored in this file."
        echo "set -euo pipefail"
        echo "cd ${q_root}"
        if [[ -n "${NCCL_SOCKET_IFNAME:-}" ]]; then
            echo "export NCCL_SOCKET_IFNAME=${q_nccl}"
        fi
        if [[ -n "${GLOO_SOCKET_IFNAME:-}" ]]; then
            echo "export GLOO_SOCKET_IFNAME=${q_gloo}"
        fi
        # HF_TOKEN is intentionally evaluated when the generated script runs.
        # shellcheck disable=SC2016
        echo 'if [[ -z "${HF_TOKEN:-}" ]]; then'
        echo '    echo "Export HF_TOKEN before reproducing this run." >&2'
        echo "    exit 1"
        echo "fi"
        echo "export HF_TOKEN"

        if [[ "$kind" == "primus" ]]; then
            if [[ "$AUTO_BENCHMARK_MODE" == "container" ]]; then
                echo "export DOCKER_IMAGE=${q_image}"
                echo "exec bash ${q_cli} container -- train ${detail} --config ${q_config}"
            else
                echo "exec bash ${q_cli} direct -- train ${detail} --config ${q_config}"
            fi
        else
            case "$detail" in
                llama2_70b)
                    echo "export PACKED_DATA_DIR=/data"
                    echo "export PRETRAINED_CHECKPOINT=$(printf '%q' "$(mlperf_pretrained_checkpoint_path llama2_70b)")"
                    echo "export PRIMUS_PATH=${q_root}"
                    echo "exec bash $(printf '%q' "${PRIMUS_ROOT}/examples/mlperf/llama2_70b/run_and_time.sh")"
                    ;;
                llama3.1_8b|gpt_oss_20b)
                    model_dir="${PRIMUS_ROOT}/examples/mlperf/${detail}"
                    echo "export PRIMUS_PATH=${q_root}"
                    echo "cd $(printf '%q' "$model_dir")"
                    echo "source ./config_MI355X_1x8x1.sh"
                    echo "exec bash ./run_and_time.sh"
                    ;;
                flux1)
                    config_path="${PRIMUS_ROOT}/examples/mlperf/flux1/flux.1_schnell_t2i-pretrain.yaml"
                    echo "export CONFIG=$(printf '%q' "$config_path")"
                    echo "export PRIMUS_WORKSPACE=$(printf '%q' "${MLPERF_FLUX_OUTPUT_ROOT}/primus_workspace")"
                    echo "export DATASET_PATH=$(printf '%q' "${MLPERF_FLUX_ROOT}/cc12m_preprocessed")"
                    echo "export EVAL_DATASET_PATH=$(printf '%q' "${MLPERF_FLUX_ROOT}/coco_preprocessed")"
                    echo "export EMPTY_ENCODINGS_PATH=$(printf '%q' "${MLPERF_FLUX_ROOT}/empty_encodings")"
                    echo "export OUTPUT_DIR=$(printf '%q' "${MLPERF_FLUX_OUTPUT_ROOT}/flux_mlperf")"
                    echo "export FLUX_FLOAT8_RECIPE=$(printf '%q' "${FLUX_FLOAT8_RECIPE:-tensorwise}")"
                    echo "export ATTENTION_BACKEND=$(printf '%q' "${ATTENTION_BACKEND:-flash_attn_aiter}")"
                    echo "export LOCAL_BATCH_SIZE=$(printf '%q' "${LOCAL_BATCH_SIZE:-64}")"
                    echo "export MAX_STEPS=$(printf '%q' "${MAX_STEPS:-30000}")"
                    echo "export GPUS_PER_NODE=$(printf '%q' "${GPUS_PER_NODE:-8}")"
                    echo "exec bash ${q_cli} direct -- train pretrain --config $(printf '%q' "$config_path")"
                    ;;
                *)
                    model_dir="${PRIMUS_ROOT}/examples/mlperf/${detail}"
                    echo "export PRIMUS_PATH=${q_root}"
                    echo "cd $(printf '%q' "$model_dir")"
                    echo "exec bash ./run_and_time.sh"
                    ;;
            esac
        fi
    } > "$dest"
    chmod +x "$dest"
}

capture_gpu_state() {
    local dest="$1"

    {
        echo "captured_at=$(date -Iseconds)"
        if ! select_gpu_tool; then
            echo "(amd-smi and rocm-smi unavailable)"
        elif [[ "$GPU_SMI_KIND" == "amd-smi" ]]; then
            echo "gpu_tool=amd-smi"
            echo "----- metric -----"
            run_bounded "$GPU_SMI_BIN" metric || echo "(amd-smi metric failed)"
            echo
            echo "----- process -----"
            run_bounded "$GPU_SMI_BIN" process || echo "(amd-smi process failed)"
        else
            echo "gpu_tool=rocm-smi"
            echo "----- summary -----"
            run_bounded "$GPU_SMI_BIN" || echo "(rocm-smi failed)"
            echo
            echo "----- memory -----"
            run_bounded "$GPU_SMI_BIN" --showmeminfo vram || echo "(rocm-smi memory failed)"
        fi
    } > "$dest"
}

write_failure_bundle() {
    local stem="$1"
    local snapshot="$2"
    local reproduce="$3"
    local gpu_state archive tmp_dir base

    gpu_state="${stem}_gpu_state.txt"
    archive="${stem}_failure.tar.gz"
    capture_gpu_state "$gpu_state"

    tmp_dir="$(mktemp -d)"
    for base in "$PREP_LOG_FILE" "${PREP_WORKING_CONFIG:-}" "$snapshot" "$reproduce" "$gpu_state"; do
        if [[ -n "$base" && -f "$base" ]]; then
            cp "$base" "${tmp_dir}/$(basename "$base")"
        fi
    done
    if tar -czf "$archive" -C "$tmp_dir" .; then
        echo -e " ${DOT} Failure bundle: ${CYAN}${archive}${RESET}"
        echo "FAILURE_BUNDLE=${archive}" >> "$PREP_LOG_FILE"
    else
        echo -e "${YELLOW}⚠ Could not write failure bundle for ${PREP_MODEL_NAME:-this run}${RESET}"
    fi
    rm -rf "$tmp_dir"
}

one_line() {
    printf '%s' "$1" | tr '\n\r' '  ' | sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//'
}

collect_setup_identity() {
    local cli probe_out rocm_line framework_line gpu_name

    SUITE_SUBMODULE_PINS="$(
        git -C "$PRIMUS_ROOT" submodule status --recursive 2>/dev/null | awk '
            {
                commit=$1
                gsub(/^[+-]/, "", commit)
                if (commit != "" && $2 != "") printf "%s@%s; ", $2, commit
            }
        '
    )"
    SUITE_SUBMODULE_PINS="$(one_line "${SUITE_SUBMODULE_PINS:-none}")"
    [[ -n "$SUITE_SUBMODULE_PINS" ]] || SUITE_SUBMODULE_PINS="none"

    gpu_name="$(rocminfo 2>/dev/null | grep -m1 -oE 'MI3[0-9]{2}X' || true)"
    SUITE_GPU="$(one_line "${gpu_name:-${DEVICE:-unknown}}")"

    SUITE_ROCM="unavailable"
    SUITE_FRAMEWORK="unavailable"
    SUITE_IMAGE_DIGEST="${DOCKER_IMAGE:-none}"

    cli=""
    if command -v docker >/dev/null 2>&1; then
        cli="docker"
    elif command -v podman >/dev/null 2>&1; then
        cli="podman"
    fi

    if [[ -n "$cli" && -n "${DOCKER_IMAGE:-}" ]]; then
        SUITE_IMAGE_DIGEST="$(one_line "$("$cli" image inspect --format '{{if .RepoDigests}}{{index .RepoDigests 0}}{{else}}{{.Id}}{{end}}' "$DOCKER_IMAGE" 2>/dev/null || echo "$DOCKER_IMAGE")")"
    fi

    # Both modes read the "ROCm version" field from amd-smi inside the image
    # that trains. On the host, launch that image first. A shell that is already
    # inside the container runs amd-smi in the current environment. Older images
    # without that line still use /opt/rocm/.info/version. The host install is not used.
    local probe_script probe_cmd
    probe_script="$(mktemp)"
    cat > "$probe_script" << 'EOF'
#!/usr/bin/env bash
rocm=""
for bin in amd-smi /opt/rocm/bin/amd-smi; do
    if [[ "$bin" == /* ]]; then
        [[ -x "$bin" ]] || continue
    elif ! command -v "$bin" >/dev/null 2>&1; then
        continue
    fi
    rocm="$("$bin" version 2>/dev/null | sed -n 's/.*ROCm version[[:space:]]*:[[:space:]]*//p' | head -n 1 | awk '{print $1}' | tr -d '\r|')"
    [[ -n "$rocm" ]] && break
done
if [[ -z "$rocm" && -f /opt/rocm/.info/version ]]; then
    rocm="$(head -n 1 /opt/rocm/.info/version)"
elif [[ -z "$rocm" && -f /opt/rocm/share/rocm/.info/version ]]; then
    rocm="$(head -n 1 /opt/rocm/share/rocm/.info/version)"
fi
printf '%s\n' "${rocm:-unavailable}"
PYTHONUNBUFFERED=1 python3 - << 'PY'
import importlib
parts = []
for name in ("torch", "jax", "transformers"):
    try:
        mod = importlib.import_module(name)
        parts.append(name + "=" + str(getattr(mod, "__version__", "unknown")))
    except Exception:
        pass
print(" ".join(parts) if parts else "unavailable", flush=True)
PY
EOF
    if ! running_inside_container && [[ -n "$cli" && -n "${DOCKER_IMAGE:-}" ]]; then
        probe_cmd=("$cli" run --rm)
        [[ -e /dev/kfd ]] && probe_cmd+=(--device=/dev/kfd)
        [[ -d /dev/dri ]] && probe_cmd+=(--device=/dev/dri)
        [[ -e /dev/infiniband ]] && probe_cmd+=(--device=/dev/infiniband)
        probe_cmd+=(-v "${probe_script}:/tmp/primus_probe.sh:ro" --entrypoint bash "$DOCKER_IMAGE" /tmp/primus_probe.sh)
        if command -v timeout >/dev/null 2>&1; then
            probe_out="$(timeout 180 "${probe_cmd[@]}" 2>/dev/null || true)"
        else
            probe_out="$("${probe_cmd[@]}" 2>/dev/null || true)"
        fi
    else
        probe_out="$(bash "$probe_script" 2>/dev/null || true)"
    fi
    rm -f "$probe_script"
    rocm_line="$(printf '%s\n' "$probe_out" | sed -n '1p')"
    framework_line="$(printf '%s\n' "$probe_out" | sed -n '2p')"
    [[ -n "$rocm_line" ]] && SUITE_ROCM="$(one_line "$rocm_line")"
    [[ -n "$framework_line" ]] && SUITE_FRAMEWORK="$(one_line "$framework_line")"

    [[ -n "$SUITE_FRAMEWORK" ]] || SUITE_FRAMEWORK="unavailable"
    export SUITE_GIT_COMMIT SUITE_SUBMODULE_PINS SUITE_GPU SUITE_ROCM SUITE_FRAMEWORK SUITE_IMAGE_DIGEST

    if [[ -z "${MANIFEST_PATH:-}" ]]; then
        local label stamp
        label="${DOCKER_RELEASE:-${BACKEND:-run}}"
        stamp="$(date +%Y%m%d_%H%M%S)"
        MANIFEST_PATH="${SCRIPT_DIR}/results/manifest_${label}_${DEVICE}_${stamp}.json"
        export MANIFEST_PATH
    fi
}

append_environment_to_log() {
    local log_file="$1"
    local runtime_seconds="$2"

    {
        echo
        echo "===== AUTO BENCHMARK ENVIRONMENT ====="
        echo "image_digest=${SUITE_IMAGE_DIGEST:-}"
        echo "primus_commit=${SUITE_GIT_COMMIT:-}"
        echo "submodule_pins=${SUITE_SUBMODULE_PINS:-}"
        echo "gpu=${SUITE_GPU:-}"
        echo "rocm_version=${SUITE_ROCM:-}"
        echo "framework_versions=${SUITE_FRAMEWORK:-}"
        echo "runtime_seconds=${runtime_seconds}"
    } >> "$log_file"
}

update_run_manifest() {
    local exit_code="$1"
    local runtime_seconds="$2"
    local status="$3"
    local failure_bundle="$4"

    [[ -n "${MANIFEST_PATH:-}" ]] || return 0
    MANIFEST_EXIT_CODE="$exit_code" \
    MANIFEST_RUNTIME="$runtime_seconds" \
    MANIFEST_STATUS="$status" \
    MANIFEST_FAILURE="$failure_bundle" \
    MANIFEST_BACKEND="${BACKEND:-}" \
    MANIFEST_MODEL="${PREP_MODEL_NAME:-}" \
    MANIFEST_RUN="${PREP_RUN_LABEL:-}" \
    MANIFEST_LOG="${PREP_LOG_FILE:-}" \
    MANIFEST_CONFIG="${PREP_WORKING_CONFIG:-}" \
    python3 - << 'PY' || echo -e "${YELLOW}⚠ Could not update the run manifest${RESET}"
import json, os
from datetime import datetime, timezone

path = os.environ.get("MANIFEST_PATH", "")
if not path:
    raise SystemExit(0)
os.makedirs(os.path.dirname(path), exist_ok=True)
try:
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
except (OSError, json.JSONDecodeError):
    manifest = {}

manifest.setdefault("suite", os.environ.get("DOCKER_RELEASE") or os.environ.get("AUTO_BENCHMARK_SUITE") or os.environ.get("BACKEND") or "")
manifest.setdefault("device", os.environ.get("DEVICE", ""))
manifest.setdefault("started_at", datetime.now(timezone.utc).isoformat())
manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
manifest["environment"] = {
    "docker_image": os.environ.get("DOCKER_IMAGE", ""),
    "image_digest": os.environ.get("SUITE_IMAGE_DIGEST", ""),
    "primus_commit": os.environ.get("SUITE_GIT_COMMIT", ""),
    "submodule_pins": os.environ.get("SUITE_SUBMODULE_PINS", ""),
    "gpu": os.environ.get("SUITE_GPU", ""),
    "rocm_version": os.environ.get("SUITE_ROCM", ""),
    "framework_versions": os.environ.get("SUITE_FRAMEWORK", ""),
    "hostname": os.uname().nodename,
    "nccl_socket_ifname": os.environ.get("NCCL_SOCKET_IFNAME", ""),
    "gloo_socket_ifname": os.environ.get("GLOO_SOCKET_IFNAME", ""),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
}
log_path = os.environ.get("MANIFEST_LOG", "")
stem = log_path[:-4] if log_path.endswith(".log") else log_path
failure = os.environ.get("MANIFEST_FAILURE", "")
entry = {
    "backend": os.environ.get("MANIFEST_BACKEND", ""),
    "model": os.environ.get("MANIFEST_MODEL", ""),
    "run": os.environ.get("MANIFEST_RUN", ""),
    "status": os.environ.get("MANIFEST_STATUS", ""),
    "exit_code": int(os.environ.get("MANIFEST_EXIT_CODE") or 0),
    "runtime_seconds": int(os.environ.get("MANIFEST_RUNTIME") or 0),
    "artifacts": {
        "log": log_path,
        "working_config": os.environ.get("MANIFEST_CONFIG", ""),
        "snapshot": f"{stem}_snapshot.txt" if stem else "",
        "reproduce": f"{stem}_reproduce.sh" if stem else "",
        "failure_bundle": failure if failure and os.path.isfile(failure) else "",
    },
}
models = manifest.setdefault("models", [])
models.append(entry)
with open(path, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)
    handle.write("\n")
PY
}

record_run_result() {
    local exit_code="$1"
    local kind="$2"
    local detail="$3"
    local runtime_seconds="${4:-0}"
    local stem snapshot reproduce status failure_bundle=""

    if [[ -z "${PREP_LOG_FILE:-}" ]]; then
        return 0
    fi
    if [[ ! -f "${SUITE_SNAPSHOT_FILE:-}" ]]; then
        capture_system_snapshot || true
    fi

    stem="${PREP_LOG_FILE%.log}"
    snapshot="${stem}_snapshot.txt"
    reproduce="${stem}_reproduce.sh"
    cp "$SUITE_SNAPSHOT_FILE" "$snapshot"
    {
        echo
        echo "----- this run -----"
        echo "backend=${BACKEND:-}"
        echo "device=${DEVICE:-}"
        echo "model=${PREP_MODEL_NAME:-}"
        echo "run=${PREP_RUN_LABEL:-}"
        echo "exit_code=${exit_code}"
        echo "log=${PREP_LOG_FILE}"
        echo "config=${PREP_WORKING_CONFIG:-}"
        echo "original_config=${PREP_CFG_FILE:-}"
    } >> "$snapshot"

    write_reproduce_script "$reproduce" "$exit_code" "$kind" "$detail"
    if [[ "$exit_code" -eq 0 ]]; then
        status="ok"
    else
        status="failed"
        failure_bundle="${stem}_failure.tar.gz"
        write_failure_bundle "$stem" "$snapshot" "$reproduce"
    fi
    append_environment_to_log "$PREP_LOG_FILE" "$runtime_seconds"
    {
        echo
        echo "===== AUTO BENCHMARK RECORD ====="
        echo "SNAPSHOT=${snapshot}"
        echo "REPRODUCE=${reproduce}"
        echo "MANIFEST=${MANIFEST_PATH:-}"
    } >> "$PREP_LOG_FILE"

    update_run_manifest "$exit_code" "$runtime_seconds" "$status" "$failure_bundle"
    echo -e " ${DOT} Snapshot: ${CYAN}${snapshot}${RESET}"
    echo -e " ${DOT} Reproduce: ${CYAN}${reproduce}${RESET}"
    echo -e " ${DOT} Manifest: ${CYAN}${MANIFEST_PATH:-}${RESET}"
}

execute_benchmark_run() {
    local cfg_file="$1"
    local working_config="$2"
    local log_file="$3"
    local model_name="$4"
    local current="$5"
    local total="$6"

    local run_exit_code=0
    local train_suite started=$SECONDS

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

    export EXP="$working_config" BACKEND HF_TOKEN DOCKER_IMAGE
    echo -e " ${CHECK} Launching via primus-cli ${AUTO_BENCHMARK_MODE}"

    echo -e " ${DOT} Changing to Primus root directory: ${CYAN}$PRIMUS_ROOT${RESET}"
    cd "$PRIMUS_ROOT" || return 1

    set +e
    if [[ "$AUTO_BENCHMARK_MODE" == "container" ]]; then
        echo -e "   ${DOT} ${CYAN}DOCKER_IMAGE=$DOCKER_IMAGE primus-cli container -- train $train_suite --config $working_config${RESET}\n"
        bash "$PRIMUS_CLI" container -- train "$train_suite" --config "$working_config" 2>&1 | tee "$log_file"
        run_exit_code=${PIPESTATUS[0]}
    else
        echo -e "   ${DOT} ${CYAN}primus-cli direct --log_file $log_file -- train $train_suite --config $working_config${RESET}\n"
        bash "$PRIMUS_CLI" direct --log_file "$log_file" -- train "$train_suite" --config "$working_config"
        run_exit_code=$?
    fi
    set +e

    echo "AUTO_BENCHMARK_EXIT_CODE=$run_exit_code" >> "$log_file"

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

    record_run_result "$run_exit_code" primus "$train_suite" "$((SECONDS - started))"
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
MLPERF_GPT_OSS_20B_ROOT="/data/gpt_oss_20b"
MLPERF_GPT_OSS_20B_DATA="${MLPERF_GPT_OSS_20B_ROOT}/data"
MLPERF_GPT_OSS_20B_MODEL="${MLPERF_GPT_OSS_20B_ROOT}/model"
MLPERF_FLUX_ROOT="/data/mlperf_flux1"
MLPERF_FLUX_OUTPUT_ROOT="${MLPERF_FLUX_ROOT}/output"
MLPERF_FLUX_CC12M_URI="https://training.mlcommons-storage.org/metadata/flux-1-cc12m-preprocessed.uri"
MLPERF_FLUX_COCO_URI="https://training.mlcommons-storage.org/metadata/flux-1-coco-preprocessed.uri"
MLPERF_FLUX_EMPTY_URI="https://training.mlcommons-storage.org/metadata/flux-1-empty-encodings.uri"
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
        gpt_oss_20b)
            grep -q "gpt_oss" "$logging_file"
            ;;
        flux1)
            grep -Eq "flux|FLUX" "$logging_file"
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
            tunables_script="${PRIMUS_ROOT}/examples/mlperf/llama2_70b/runtime_tunables.sh"
            ;;
        llama3.1_8b)
            tunables_script="${PRIMUS_ROOT}/examples/mlperf/llama3.1_8b/runtime_tunables.sh"
            ;;
        gpt_oss_20b)
            tunables_script="${PRIMUS_ROOT}/examples/mlperf/gpt_oss_20b/runtime_tunables.sh"
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
    local model_dir="${PRIMUS_ROOT}/examples/mlperf/llama3.1_8b"
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

resolve_mlperf_gpt_oss_20b_dataset_dir() {
    local candidate

    for candidate in "$MLPERF_GPT_OSS_20B_DATA" "$MLPERF_LLAMA31_8B_DATA" "/data"; do
        if mlperf_c4_dataset_exists_in_dir "$candidate"; then
            echo "$candidate"
            return 0
        fi
    done

    return 1
}

resolve_mlperf_gpt_oss_20b_model_dir() {
    if [[ -d "$MLPERF_GPT_OSS_20B_MODEL" && -n "$(ls -A "${MLPERF_GPT_OSS_20B_MODEL}" 2>/dev/null)" ]]; then
        echo "$MLPERF_GPT_OSS_20B_MODEL"
        return 0
    fi

    resolve_mlperf_llama31_8b_model_dir
}

prepare_mlperf_gpt_oss_20b_dataset() {
    local existing_data_dir existing_model_dir

    echo -e " ${CHECK} mkdir -p ${CYAN}${MLPERF_GPT_OSS_20B_ROOT}${RESET}"
    mkdir -p "$MLPERF_GPT_OSS_20B_ROOT" || return 1

    if existing_data_dir="$(resolve_mlperf_gpt_oss_20b_dataset_dir)"; then
        echo -e " ${CHECK} MLPerf gpt_oss_20b C4 dataset already present at ${CYAN}${existing_data_dir}${RESET}; skipping download"
    else
        cd "$MLPERF_GPT_OSS_20B_ROOT" || return 1

        if ! ensure_mlperf_download_tools; then
            return 1
        fi

        echo -e " ${CHECK} Downloading MLPerf gpt_oss_20b C4 dataset into ${CYAN}${MLPERF_GPT_OSS_20B_DATA}${RESET}"
        if ! run_mlperf_r2_downloader -d data "$MLPERF_LLAMA31_8B_DATA_URI"; then
            echo -e "${RED}✗ MLPerf gpt_oss_20b dataset download failed${RESET}"
            return 1
        fi
    fi

    if existing_model_dir="$(resolve_mlperf_gpt_oss_20b_model_dir)"; then
        echo -e " ${CHECK} MLPerf gpt_oss_20b tokenizer already present at ${CYAN}${existing_model_dir}${RESET}; skipping download"
    else
        cd "$MLPERF_GPT_OSS_20B_ROOT" || return 1

        if ! ensure_mlperf_download_tools; then
            return 1
        fi

        echo -e " ${CHECK} Downloading MLPerf gpt_oss_20b tokenizer into ${CYAN}${MLPERF_GPT_OSS_20B_MODEL}${RESET}"
        if ! run_mlperf_r2_downloader -d model "$MLPERF_LLAMA31_8B_MODEL_URI"; then
            echo -e "${RED}✗ MLPerf gpt_oss_20b tokenizer download failed${RESET}"
            return 1
        fi
    fi

    if ! existing_data_dir="$(resolve_mlperf_gpt_oss_20b_dataset_dir)"; then
        echo -e "${RED}✗ MLPerf gpt_oss_20b dataset not found under ${MLPERF_GPT_OSS_20B_DATA} or /data${RESET}"
        return 1
    fi

    if ! existing_model_dir="$(resolve_mlperf_gpt_oss_20b_model_dir)"; then
        echo -e "${RED}✗ MLPerf gpt_oss_20b tokenizer not found under ${MLPERF_GPT_OSS_20B_MODEL} or /model${RESET}"
        return 1
    fi

    return 0
}

prepare_mlperf_gpt_oss_20b_data_layout() {
    local item base target dataset_dir model_dir

    mkdir -p /data /results 2>/dev/null || true

    dataset_dir="$(resolve_mlperf_gpt_oss_20b_dataset_dir || true)"
    model_dir="$(resolve_mlperf_gpt_oss_20b_model_dir || true)"

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

run_mlperf_gpt_oss_20b_benchmark() {
    local model_dir="${PRIMUS_ROOT}/examples/mlperf/gpt_oss_20b"
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

    if ! prepare_mlperf_gpt_oss_20b_dataset; then
        return 1
    fi

    prepare_mlperf_gpt_oss_20b_data_layout
    apply_mlperf_runtime_tunables "gpt_oss_20b"

    cd "$model_dir" || return 1

    echo -e " ${CHECK} source config_MI355X_1x8x1.sh"
    # shellcheck disable=SC1091
    source config_MI355X_1x8x1.sh

    echo -e " ${CHECK} bash run_and_time.sh"
    echo -e "   ${DOT} EXP=${CYAN}${EXP:-<unset>}${RESET}"
    echo -e "   ${DOT} DATA_PATH=${CYAN}${DATA_PATH:-<unset>}${RESET}\n"

    set +e
    bash run_and_time.sh
    local ret_code=$?
    set +e

    return "$ret_code"
}

mlperf_flux_dataset_ready() {
    [[ -d "${MLPERF_FLUX_ROOT}/cc12m_preprocessed" \
        && -d "${MLPERF_FLUX_ROOT}/coco_preprocessed" \
        && -d "${MLPERF_FLUX_ROOT}/empty_encodings" ]]
}

prepare_mlperf_flux_dataset() {
    local uri dest_name

    echo -e " ${CHECK} mkdir -p ${CYAN}${MLPERF_FLUX_ROOT}${RESET}"
    mkdir -p "$MLPERF_FLUX_ROOT" || return 1

    if mlperf_flux_dataset_ready; then
        echo -e " ${CHECK} MLPerf flux1 datasets already present under ${CYAN}${MLPERF_FLUX_ROOT}${RESET}; skipping download"
        return 0
    fi

    cd "$MLPERF_FLUX_ROOT" || return 1

    if ! ensure_mlperf_download_tools; then
        return 1
    fi

    for uri in "$MLPERF_FLUX_CC12M_URI" "$MLPERF_FLUX_COCO_URI" "$MLPERF_FLUX_EMPTY_URI"; do
        case "$uri" in
            *cc12m*) dest_name="cc12m_preprocessed" ;;
            *coco*) dest_name="coco_preprocessed" ;;
            *) dest_name="empty_encodings" ;;
        esac

        if [[ -d "${MLPERF_FLUX_ROOT}/${dest_name}" ]]; then
            echo -e " ${CHECK} MLPerf flux1 ${dest_name} already present; skipping download"
            continue
        fi

        echo -e " ${CHECK} Downloading MLPerf flux1 ${dest_name} into ${CYAN}${MLPERF_FLUX_ROOT}${RESET}"
        if ! run_mlperf_r2_downloader "$uri"; then
            echo -e "${RED}✗ MLPerf flux1 ${dest_name} download failed${RESET}"
            return 1
        fi
    done

    if ! mlperf_flux_dataset_ready; then
        echo -e "${RED}✗ MLPerf flux1 data root is missing cc12m_preprocessed, coco_preprocessed, or empty_encodings under ${MLPERF_FLUX_ROOT}${RESET}"
        return 1
    fi

    return 0
}

run_mlperf_flux1_benchmark() {
    local config_path="${PRIMUS_ROOT}/examples/mlperf/flux1/flux.1_schnell_t2i-pretrain.yaml"
    local output_dir="${MLPERF_FLUX_OUTPUT_ROOT}/flux_mlperf"

    if [[ ! -f "$config_path" ]]; then
        echo -e "${RED}✗ MLPerf flux config not found:${RESET} ${CYAN}$config_path${RESET}"
        return 1
    fi

    if [[ ! -f "$PRIMUS_CLI" ]]; then
        echo -e "${RED}✗ primus-cli not found at:${RESET} ${CYAN}$PRIMUS_CLI${RESET}"
        return 1
    fi

    export HF_TOKEN

    if ! prepare_mlperf_flux_dataset; then
        return 1
    fi

    mkdir -p "$output_dir" || return 1

    export CONFIG="$config_path"
    export PRIMUS_WORKSPACE="${MLPERF_FLUX_OUTPUT_ROOT}/primus_workspace"
    export DATASET_PATH="${MLPERF_FLUX_ROOT}/cc12m_preprocessed"
    export EVAL_DATASET_PATH="${MLPERF_FLUX_ROOT}/coco_preprocessed"
    export EMPTY_ENCODINGS_PATH="${MLPERF_FLUX_ROOT}/empty_encodings"
    export OUTPUT_DIR="$output_dir"
    export FLUX_FLOAT8_RECIPE="${FLUX_FLOAT8_RECIPE:-tensorwise}"
    export ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash_attn_aiter}"
    export LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-64}"
    export MAX_STEPS="${MAX_STEPS:-30000}"
    export GRADIENT_CHECKPOINTING_RATIO="${GRADIENT_CHECKPOINTING_RATIO:-0.25}"
    export COMPILE_TRANSFORMER_BLOCKS="${COMPILE_TRANSFORMER_BLOCKS:-true}"
    export FSDP2_RESHARD_AFTER_FORWARD="${FSDP2_RESHARD_AFTER_FORWARD:-false}"
    export FSDP2_REDUCE_DTYPE="${FSDP2_REDUCE_DTYPE:-fp32}"
    export SAVE_STEPS="${SAVE_STEPS:-100}"
    export SAVE_STRATEGY="${SAVE_STRATEGY:-dtcp_full}"
    export CHECKPOINT_KEEP_LATEST="${CHECKPOINT_KEEP_LATEST:-3}"
    export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-latest}"
    export MLPERF_ENABLE="${MLPERF_ENABLE:-true}"
    export MLPERF_CLEAR_CACHES="${MLPERF_CLEAR_CACHES:-true}"
    export MLLOG_OUTPUT_FILE="${MLLOG_OUTPUT_FILE:-${output_dir}/mlperf_compliance.log}"
    export TARGET_ACCURACY="${TARGET_ACCURACY:-0.586}"
    export VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-262144}"
    export SEED="${SEED:-10007}"
    export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

    if [[ "${MLPERF_CLEAR_CACHES}" == "true" ]]; then
        sync || true
        if [[ -w /proc/sys/vm/drop_caches ]]; then
            echo 3 > /proc/sys/vm/drop_caches || true
        fi
    fi

    cd "$PRIMUS_ROOT" || return 1

    echo -e " ${CHECK} primus-cli direct -- train pretrain --config ${CYAN}${CONFIG}${RESET}"
    echo -e "   ${DOT} DATASET_PATH=${CYAN}${DATASET_PATH}${RESET}"
    echo -e "   ${DOT} OUTPUT_DIR=${CYAN}${OUTPUT_DIR}${RESET}\n"

    set +e
    bash "$PRIMUS_CLI" direct -- train pretrain --config "$CONFIG"
    local ret_code=$?
    set +e

    return "$ret_code"
}

execute_mlperf_benchmark_run() {
    local model_id="$1"
    local log_file="$2"
    local current="$3"
    local total="$4"

    local model_dir="$PREP_MLPERF_MODEL_DIR"
    local config_script run_script run_exit_code=0 started=$SECONDS

    echo -e "${STAR} ${BOLD}Starting MLPerf Benchmark ${current}/${total}...${RESET}"
    echo -e "   ${DOT} Model: ${CYAN}$model_id${RESET}"
    echo -e "   ${DOT} Backend: ${CYAN}$BACKEND${RESET}"
    echo -e "   ${DOT} Device: ${CYAN}$DEVICE${RESET}"
    echo -e "   ${DOT} Model dir: ${YELLOW}$model_dir${RESET}"
    echo -e "   ${DOT} Log: ${YELLOW}$log_file${RESET}\n"

    if [[ "$model_id" == "llama2_70b" ]]; then
        config_script="${PRIMUS_ROOT}/examples/mlperf/llama2_70b/config_MI355X_1x8x1.sh"
        run_script="${PRIMUS_ROOT}/examples/mlperf/llama2_70b/run_and_time.sh"

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
    elif [[ "$model_id" == "gpt_oss_20b" ]]; then
        set +e
        run_mlperf_gpt_oss_20b_benchmark 2>&1 | tee "$log_file"
        run_exit_code=${PIPESTATUS[0]}
        set +e
    elif [[ "$model_id" == "flux1" ]]; then
        set +e
        run_mlperf_flux1_benchmark 2>&1 | tee "$log_file"
        run_exit_code=${PIPESTATUS[0]}
        set +e
        if [[ -n "${MLLOG_OUTPUT_FILE:-}" && -f "$MLLOG_OUTPUT_FILE" ]]; then
            {
                echo
                echo "===== MLLOG OUTPUT (${MLLOG_OUTPUT_FILE}) ====="
                cat "$MLLOG_OUTPUT_FILE"
            } >> "$log_file"
        fi
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

    echo "AUTO_BENCHMARK_EXIT_CODE=$run_exit_code" >> "$log_file"

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

    record_run_result "$run_exit_code" mlperf "$model_id" "$((SECONDS - started))"
    return "$run_exit_code"
}

set_backend_globals() {
    BACKEND="$1"
    case "$BACKEND" in
        megatron) BACKEND_BASE_DIR="$MEGATRON_BASE_DIR" ;;
        torchtitan) BACKEND_BASE_DIR="$TORCHTITAN_BASE_DIR" ;;
        megatron_bridge) BACKEND_BASE_DIR="$MEGATRON_BRIDGE_BASE_DIR" ;;
        mlperf) BACKEND_BASE_DIR="$MLPERF_BASE_DIR" ;;
        maxtext) BACKEND_BASE_DIR="$MAXTEXT_BASE_DIR" ;;
        maxdiffusion) BACKEND_BASE_DIR="$MAXDIFFUSION_BASE_DIR" ;;
        *)
            echo -e "${RED}✗ Unknown backend: $BACKEND${RESET}" >&2
            exit 1
            ;;
    esac
}

collect_release_items() {
    local backend cfg
    local -a configs=()

    if [[ ${#SUITE_BACKENDS[@]} -eq 0 ]]; then
        echo -e "${RED}✗ No backends are selected for this release suite.${RESET}" >&2
        exit 1
    fi

    RUN_ITEMS=()
    for backend in "${SUITE_BACKENDS[@]}"; do
        mapfile -t configs < <(load_release_configs "$DEVICE" "$backend")
        if [[ ${#configs[@]} -eq 0 ]]; then
            echo -e "${RED}✗ No built-in release configs are defined for ${DEVICE}/${backend}${RESET}" >&2
            exit 1
        fi
        echo -e " ${CHECK} ${CYAN}${backend}${RESET}: ${GREEN}${#configs[@]}${RESET} release configs"
        for cfg in "${configs[@]}"; do
            if [[ ! -f "$cfg" ]]; then
                echo -e "${RED}✗ Built-in release config does not exist: $cfg${RESET}" >&2
                exit 1
            fi
            echo -e "    ${STAR} $(basename "$cfg") ${DIM}[${backend}]${RESET}"
            RUN_ITEMS+=("${backend}|${cfg}")
        done
    done
    echo
}

save_run_state() {
    local state_file="$1"
    local edit_dir orig copied dest

    mkdir -p "$(dirname "$state_file")/edits"
    edit_dir="$(dirname "$state_file")/edits"
    {
        echo "suite=${DOCKER_RELEASE:-}"
        echo "device=${DEVICE}"
        echo "backend=${BACKEND:-}"
        echo "---items---"
        if [[ ${#RUN_ITEMS[@]} -gt 0 ]]; then
            printf '%s\n' "${RUN_ITEMS[@]}"
        fi
        echo "---mlperf---"
        if [[ ${#SELECTED_MLPERF_MODELS[@]} -gt 0 ]]; then
            printf '%s\n' "${SELECTED_MLPERF_MODELS[@]}"
        fi
        echo "---edits---"
    } > "$state_file"

    if declare -p EDITED_CONFIGS >/dev/null 2>&1; then
        for orig in "${!EDITED_CONFIGS[@]}"; do
            copied="${EDITED_CONFIGS[$orig]}"
            if [[ -f "$copied" ]]; then
                dest="${edit_dir}/$(basename "$orig")"
                cp "$copied" "$dest"
                printf '%s=%s\n' "$orig" "$dest" >> "$state_file"
            fi
        done
    fi
}

load_run_state() {
    local state_file="$1"
    local section="header" line key value

    DOCKER_RELEASE=""
    DEVICE=""
    BACKEND=""
    RUN_ITEMS=()
    SELECTED_CONFIGS=()
    SELECTED_MLPERF_MODELS=()
    declare -gA EDITED_CONFIGS=()

    while IFS= read -r line || [[ -n "$line" ]]; do
        case "$line" in
            ---items---) section="items"; continue ;;
            ---mlperf---) section="mlperf"; continue ;;
            ---edits---) section="edits"; continue ;;
        esac
        case "$section" in
            header)
                key="${line%%=*}"
                value="${line#*=}"
                case "$key" in
                    suite) DOCKER_RELEASE="$value" ;;
                    device) DEVICE="$value" ;;
                    backend) BACKEND="$value" ;;
                esac
                ;;
            items)
                [[ -n "$line" ]] && RUN_ITEMS+=("$line")
                ;;
            mlperf)
                [[ -n "$line" ]] && SELECTED_MLPERF_MODELS+=("$line")
                ;;
            edits)
                EDITED_CONFIGS["${line%%=*}"]="${line#*=}"
                ;;
        esac
    done < "$state_file"

    if [[ -z "$DEVICE" ]]; then
        echo -e "${RED}✗ Run state is missing a device: $state_file${RESET}" >&2
        exit 1
    fi
}

submit_saved_run() {
    local time_limit gpus account script_dir script output_pattern state_file label stamp

    if ! command -v sbatch >/dev/null 2>&1; then
        echo -e "${RED}✗ sbatch was not found, and this machine has no local GPU.${RESET}"
        return 1
    fi
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo -e "${RED}✗ HF_TOKEN is required before submitting the batch job.${RESET}"
        return 1
    fi

    time_limit="${SBATCH_TIME:-24:00:00}"
    gpus="${SBATCH_GPUS_PER_NODE:-8}"
    account="${SBATCH_ACCOUNT:-}"
    label="${DOCKER_RELEASE:-$BACKEND}"
    stamp="$(date +%Y%m%d_%H%M%S)"
    script_dir="${SCRIPT_DIR}/results/slurm"
    mkdir -p "$script_dir"
    state_file="${script_dir}/autobench_${DEVICE}_${label}_${stamp}.state"
    script="${script_dir}/autobench_${DEVICE}_${label}_${stamp}.sbatch"
    output_pattern="${script_dir}/%j_${DEVICE}_${label}.out"
    save_run_state "$state_file"

    cat > "$script" <<EOF
#!/bin/bash
#SBATCH --job-name=primus-autobench-${DEVICE}-${label}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=${gpus}
#SBATCH --time=${time_limit}
#SBATCH --chdir=${PRIMUS_ROOT}
#SBATCH --output=${output_pattern}
#SBATCH --open-mode=append
EOF
    if [[ -n "${SBATCH_PARTITION:-}" ]]; then
        echo "#SBATCH --partition=${SBATCH_PARTITION}" >> "$script"
    fi
    if [[ -n "$account" ]]; then
        echo "#SBATCH --account=${account}" >> "$script"
    fi
    cat >> "$script" <<EOF

set -uo pipefail
cd "${PRIMUS_ROOT}"
export AUTO_BENCHMARK_NONINTERACTIVE=1
export AUTO_BENCHMARK_DETACHED=1
export AUTO_BENCHMARK_STATE_FILE="${state_file}"
export AUTO_BENCHMARK_DEVICE="${DEVICE}"
export AUTO_BENCHMARK_SUITE="${DOCKER_RELEASE:-}"
export PRIMUS_ROOT="${PRIMUS_ROOT}"
exec bash "${SCRIPT_DIR}/run_primus_autobenchmark.sh"
EOF
    chmod +x "$script"

    echo -e " ${CHECK} Queued script: ${CYAN}$script${RESET}"
    echo -e "   ${DOT} Wall time ${CYAN}${time_limit}${RESET}, GPUs per node ${CYAN}${gpus}${RESET}."
    if [[ -z "${DOCKER_IMAGE:-}" ]]; then
        echo -e "   ${DOT} The allocated node is outside a container, so it needs a Docker image."
        prompt_docker_image
    fi
    echo -e "   ${DOT} The job inherits ${CYAN}DOCKER_IMAGE${RESET}."
    echo -e "   ${DOT} On the allocated node, a shell already inside a container uses direct mode."
    echo -e "   ${DOT} HF_TOKEN is inherited from this shell and is not written into the script."
    echo -e "   ${DOT} Closing this connection will not stop the queued run."
    if [[ "${SBATCH_DRY_RUN:-0}" == "1" ]]; then
        echo -e " ${CHECK} SBATCH_DRY_RUN=1; not submitted."
        return 0
    fi

    sbatch --export=ALL "$script"
}

detach_local_run() {
    local session_dir session_log state_file label stamp child_pid tail_pid

    label="${DOCKER_RELEASE:-$BACKEND}"
    stamp="$(date +%Y%m%d_%H%M%S)"
    session_dir="${SCRIPT_DIR}/results/sessions"
    mkdir -p "$session_dir"
    state_file="${session_dir}/autobench_${DEVICE}_${label}_${stamp}.state"
    session_log="${session_dir}/autobench_${DEVICE}_${label}_${stamp}.log"
    save_run_state "$state_file"
    touch "$session_log"

    export AUTO_BENCHMARK_STATE_FILE="$state_file"
    export AUTO_BENCHMARK_DETACHED=1
    export AUTO_BENCHMARK_NONINTERACTIVE=1
    export HF_TOKEN DOCKER_IMAGE PRIMUS_ROOT

    echo -e " ${CHECK} Starting the run in its own session."
    echo -e "   ${DOT} An SSH disconnect will not stop it."
    echo -e "   ${DOT} Session log: ${CYAN}${session_log}${RESET}"

    if command -v setsid >/dev/null 2>&1; then
        setsid nohup bash "${SCRIPT_DIR}/run_primus_autobenchmark.sh" >>"$session_log" 2>&1 </dev/null &
    else
        nohup bash "${SCRIPT_DIR}/run_primus_autobenchmark.sh" >>"$session_log" 2>&1 </dev/null &
    fi
    child_pid=$!
    echo -e "   ${DOT} PID: ${CYAN}${child_pid}${RESET}"

    tail -n +1 -f --pid "$child_pid" "$session_log" &
    tail_pid=$!
    trap 'echo; echo -e " ${CHECK} PID ${child_pid} is still running."; echo -e "   ${DOT} Log: ${session_log}"; kill "$tail_pid" 2>/dev/null || true; exit 0' HUP INT TERM
    wait "$child_pid"
    local status=$?
    kill "$tail_pid" 2>/dev/null || true
    wait "$tail_pid" 2>/dev/null || true
    exit "$status"
}

current_container_id() {
    local cgroup_info id

    cgroup_info="$(cat /proc/self/cgroup /proc/1/cgroup 2>/dev/null || true)"
    id="$(printf '%s\n' "$cgroup_info" | grep -oE '[0-9a-f]{12,64}' | head -n 1 || true)"
    if [[ -z "$id" && -f /.dockerenv ]]; then
        id="$(cat /etc/hostname 2>/dev/null || true)"
    fi
    printf '%s\n' "$id"
}

container_cli() {
    local runtime="$1"
    shift

    if "$runtime" "$@"; then
        return 0
    fi
    if command -v sudo >/dev/null 2>&1; then
        sudo -n "$runtime" "$@"
        return $?
    fi
    return 1
}

stop_other_containers() {
    local runtime id name self_id stopped=0

    self_id="$(current_container_id)"
    for runtime in docker podman; do
        if ! command -v "$runtime" >/dev/null 2>&1; then
            continue
        fi
        while read -r id name; do
            [[ -z "$id" ]] && continue
            if [[ -n "$self_id" && ( "$id" == "$self_id" || "$self_id" == "$id"* || "$id" == "$self_id"* ) ]]; then
                echo -e "   ${DOT} Leaving the current container ${CYAN}${name:-$id}${RESET} running"
                continue
            fi
            echo -e "   ${DOT} Stopping ${runtime} container ${CYAN}${name:-$id}${RESET} (${id})"
            if container_cli "$runtime" stop -t 20 "$id" >/dev/null; then
                stopped=$((stopped + 1))
            else
                echo -e "   ${YELLOW}⚠ Could not stop ${runtime} container ${id}${RESET}"
            fi
        done < <(container_cli "$runtime" ps --format '{{.ID}} {{.Names}}' 2>/dev/null || true)
    done
    echo -e " ${CHECK} Stopped ${GREEN}${stopped}${RESET} other container(s)"
}

pid_is_protected() {
    local pid="$1"
    local cursor="$$"

    while [[ -n "$cursor" && "$cursor" != "0" && "$cursor" != "1" ]]; do
        if [[ "$pid" == "$cursor" ]]; then
            return 0
        fi
        cursor="$(ps -o ppid= -p "$cursor" 2>/dev/null | tr -d '[:space:]')"
    done
    return 1
}

collect_gpu_pids() {
    local device
    local -a devices=()

    {
        if [[ -e /dev/kfd ]]; then
            devices+=("/dev/kfd")
        fi
        for device in /dev/dri/renderD* /dev/dri/card*; do
            if [[ -e "$device" ]]; then
                devices+=("$device")
            fi
        done
        if [[ ${#devices[@]} -gt 0 ]] && command -v fuser >/dev/null 2>&1; then
            fuser "${devices[@]}" 2>/dev/null || true
        fi
        if command -v amd-smi >/dev/null 2>&1; then
            amd-smi process 2>/dev/null | grep -oE 'PID[^0-9]*[0-9]+' || true
        fi
        if command -v rocm-smi >/dev/null 2>&1; then
            rocm-smi --showpids 2>/dev/null | grep -oE 'PID[^0-9]*[0-9]+|[Pp]rocess ID[^0-9]*[0-9]+' || true
        fi
    } | awk '{
        for (i = 1; i <= NF; i++) {
            digit = $i
            gsub(/[^0-9]/, "", digit)
            if (digit ~ /^[0-9]+$/ && digit > 1) {
                print digit
            }
        }
    }' | sort -u
}

gpu_holders_remain() {
    local pid
    while read -r pid; do
        [[ -z "$pid" ]] && continue
        if [[ -d "/proc/${pid}" ]] && ! pid_is_protected "$pid"; then
            return 0
        fi
    done < <(collect_gpu_pids)
    return 1
}

stop_gpu_processes() {
    local pid signal stopped=0
    local -a victims=()

    while read -r pid; do
        [[ -z "$pid" || ! -d "/proc/${pid}" ]] && continue
        if pid_is_protected "$pid"; then
            continue
        fi
        victims+=("$pid")
    done < <(collect_gpu_pids)

    if [[ ${#victims[@]} -eq 0 ]]; then
        echo -e " ${CHECK} No other process is holding GPU memory"
        return 0
    fi

    for signal in TERM KILL; do
        for pid in "${victims[@]}"; do
            if [[ ! -d "/proc/${pid}" ]]; then
                continue
            fi
            echo -e "   ${DOT} Sending ${signal} to GPU process ${CYAN}${pid}${RESET} ($(ps -o comm= -p "$pid" 2>/dev/null | tr -d '[:space:]'))"
            kill "-${signal}" "$pid" 2>/dev/null || true
            stopped=$((stopped + 1))
        done
        sleep 2
    done
    echo -e " ${CHECK} Cleared ${GREEN}${#victims[@]}${RESET} GPU process(es)"
}

prepare_exclusive_gpu() {
    echo
    echo -e "${STAR} ${BOLD}Freeing the GPUs before the run${RESET}"
    echo -e "   ${DOT} Other running containers are stopped, and processes holding GPU memory are cleared."
    stop_other_containers
    stop_gpu_processes
    if gpu_holders_remain; then
        echo -e "${RED}✗ GPU memory is still held by another process. The benchmark was not started.${RESET}"
        while read -r pid; do
            if [[ -n "$pid" && -d "/proc/${pid}" ]] && ! pid_is_protected "$pid"; then
                echo -e "   ${DOT} PID ${CYAN}${pid}${RESET} ($(ps -o comm= -p "$pid" 2>/dev/null | tr -d '[:space:]'))"
            fi
        done < <(collect_gpu_pids)
        exit 1
    fi
    echo -e " ${CHECK} GPU memory is clear"
    echo
}

launch_without_disconnect() {
    if [[ "${AUTO_BENCHMARK_DETACHED:-0}" == "1" || "${AUTO_BENCHMARK_NONINTERACTIVE:-0}" == "1" ]]; then
        trap '' HUP
        return 0
    fi

    if [[ "${LOCAL_GPU_AVAILABLE:-0}" != "1" && -z "${SLURM_JOB_ID:-}" ]]; then
        echo -e "${STAR} ${BOLD}No local GPU. Submitting this run to Slurm.${RESET}"
        submit_saved_run
        exit $?
    fi

    detach_local_run
}

generate_metrics_table() {
    local metrics_script="metrics.py"

    echo
    echo -e "${STAR} ${BOLD}Generating Metrics Table...${RESET}\n"

    if [[ -f "$SCRIPT_DIR/$metrics_script" && ( "$BACKEND" == "megatron" || "$BACKEND" == "megatron_bridge" || "$BACKEND" == "torchtitan" || "$BACKEND" == "maxtext" || "$BACKEND" == "maxdiffusion" || "$BACKEND" == "mlperf" ) ]]; then
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

run_config_group() {
    local TOTAL_CONFIGS CURRENT CFG_FILE

    TOTAL_CONFIGS=${#SELECTED_CONFIGS[@]}
    CURRENT=1
    LOG_DIR="${SCRIPT_DIR}/results/logs_${BACKEND}"
    mkdir -p "$LOG_DIR"

    echo -e "${INFO} ${BOLD}Total configurations to run: ${TOTAL_CONFIGS}${RESET} ${DIM}(${BACKEND})${RESET}"
    echo -e "${INFO} ${BOLD}Configuration list:${RESET}"
    for i in "${!SELECTED_CONFIGS[@]}"; do
        echo -e "   ${DOT} $((i+1)). $(basename "${SELECTED_CONFIGS[$i]}")"
    done
    echo

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

        if launcher_error_is_fatal "$PREP_LOG_FILE"; then
            echo -e "${RED}✗ Docker could not start ${DOCKER_IMAGE}. The remaining models were not launched.${RESET}"
            exit 1
        fi

        CURRENT=$((CURRENT + 1))

        if [[ $CURRENT -le $TOTAL_CONFIGS ]]; then
            echo -e "${YELLOW}Preparing next benchmark...${RESET}\n"
            echo -e "${INFO} ${BOLD}Next: Config ${CURRENT}/${TOTAL_CONFIGS}${RESET}\n"
            sleep 2
        fi
    done
}

run_mlperf_group() {
    local TOTAL_CONFIGS CURRENT MLPERF_MODEL

    BACKEND="mlperf"
    TOTAL_CONFIGS=${#SELECTED_MLPERF_MODELS[@]}
    CURRENT=1
    LOG_DIR="${SCRIPT_DIR}/results/logs_${BACKEND}"
    mkdir -p "$LOG_DIR"
    export RUN_RUNTIME_TUNABLES="${RUN_RUNTIME_TUNABLES:-1}"

    echo -e "${INFO} ${BOLD}Total configurations to run: ${TOTAL_CONFIGS}${RESET}"
    echo -e "${INFO} ${BOLD}Configuration list:${RESET}"
    for i in "${!SELECTED_MLPERF_MODELS[@]}"; do
        echo -e "   ${DOT} $((i+1)). ${SELECTED_MLPERF_MODELS[$i]}"
    done
    echo

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
}

flush_release_group() {
    if [[ -z "${current_backend:-}" || ${#group[@]} -eq 0 ]]; then
        return 0
    fi
    set_backend_globals "$current_backend"
    SELECTED_CONFIGS=("${group[@]}")
    echo
    echo -e "${MAGENTA}${BOLD}=========================================${RESET}"
    echo -e "${MAGENTA}${BOLD}  ${current_backend} (${#group[@]} configs)${RESET}"
    echo -e "${MAGENTA}${BOLD}=========================================${RESET}"
    run_config_group
    generate_metrics_table
    group=()
}

run_benchmarks() {
    local item backend cfg current_backend=""
    local -a group=()

    launch_without_disconnect

    configure_git_safe_directories
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo -e "${RED}✗ HF_TOKEN is not set.${RESET}"
        exit 1
    fi
    export HF_TOKEN

    echo -e "${STAR} ${BOLD}Running on ${CYAN}${DEVICE}${RESET}"
    if [[ -n "${DOCKER_RELEASE:-}" ]]; then
        echo -e " ${CHECK} Docker release: ${CYAN}${DOCKER_RELEASE}${RESET}"
    fi
    if [[ "$AUTO_BENCHMARK_MODE" == "container" ]]; then
        prompt_docker_image
        echo -e " ${CHECK} Docker image: ${CYAN}$DOCKER_IMAGE${RESET}"
    fi
    prepare_exclusive_gpu
    capture_system_snapshot

    if [[ ${#SELECTED_MLPERF_MODELS[@]} -gt 0 ]]; then
        run_mlperf_group
        generate_metrics_table
    fi

    for item in "${RUN_ITEMS[@]}"; do
        backend="${item%%|*}"
        cfg="${item#*|}"
        if [[ -n "$current_backend" && "$backend" != "$current_backend" ]]; then
            flush_release_group
        fi
        current_backend="$backend"
        group+=("$cfg")
    done
    flush_release_group

    if [[ ${#SELECTED_MLPERF_MODELS[@]} -eq 0 && ${#RUN_ITEMS[@]} -eq 0 ]]; then
        echo -e "${RED}✗ Nothing selected to run.${RESET}"
        exit 1
    fi

    echo
    echo -e "${MAGENTA}${BOLD}=========================================${RESET}"
    echo -e "${MAGENTA}${BOLD}  Run finished${RESET}"
    echo -e "${MAGENTA}${BOLD}=========================================${RESET}"
}

# ------------------------------------------
# Banner
# ------------------------------------------
main() {
trap 'echo "[DEBUG] Script exiting at line $LINENO with exit code $?"' EXIT
if [[ -n "${AUTO_BENCHMARK_STATE_FILE:-}" ]]; then
    if [[ ! -f "$AUTO_BENCHMARK_STATE_FILE" ]]; then
        echo -e "${RED}✗ Run state file not found: ${AUTO_BENCHMARK_STATE_FILE}${RESET}" >&2
        exit 1
    fi
    load_run_state "$AUTO_BENCHMARK_STATE_FILE"
    run_benchmarks
    return
fi
if [[ "${AUTO_BENCHMARK_NONINTERACTIVE:-0}" != "1" ]]; then
    clear
fi
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
# 1. WORKFLOW / BACKEND SELECTION
# ------------------------------------------
DOCKER_RELEASE=""
SUITE_BACKENDS=()

if [[ -n "${AUTO_BENCHMARK_SUITE:-}" ]]; then
    WORKFLOW="release"
    RELEASE_IN="${AUTO_BENCHMARK_SUITE}"
elif [[ "${AUTO_BENCHMARK_NONINTERACTIVE:-0}" == "1" && -n "${AUTO_BENCHMARK_RELEASE_BACKEND:-}" ]]; then
    WORKFLOW="release"
    RELEASE_IN="backend"
elif [[ "${AUTO_BENCHMARK_NONINTERACTIVE:-0}" == "1" ]]; then
    WORKFLOW="interactive"
    BACKEND_IN="${AUTO_BENCHMARK_BACKEND:?AUTO_BENCHMARK_BACKEND is required in batch mode}"
else
    echo -e "${STAR} ${BOLD}What do you want to run?${RESET}"
    echo -e "  ${DOT} 1) Docker release suite ${DIM}(automatic model selection)${RESET}"
    echo -e "  ${DOT} 2) Interactive benchmarking ${DIM}(choose backend and models)${RESET}"
    echo -en " ${ARROW} Enter number or name: "
    read -r WORKFLOW_IN

    case "$WORKFLOW_IN" in
        1|release|docker|docker-release|docker_release)
            WORKFLOW="release"
            ;;
        2|interactive|benchmark|manual)
            WORKFLOW="interactive"
            ;;
        *)
            echo -e "${RED}✗ Invalid workflow: $WORKFLOW_IN${RESET}"
            exit 1
            ;;
    esac
fi

if [[ "$WORKFLOW" == "release" ]]; then
    if [[ "${RELEASE_IN:-}" == "backend" ]]; then
        DOCKER_RELEASE=""
        SUITE_BACKENDS=("${AUTO_BENCHMARK_RELEASE_BACKEND}")
        set_backend_globals "${AUTO_BENCHMARK_RELEASE_BACKEND}"
    elif [[ -z "${RELEASE_IN:-}" ]]; then
        echo
        echo -e "${STAR} ${BOLD}Choose Docker Release Suite:${RESET}"
        echo -e "  ${DOT} 1) Primus ${DIM}(megatron, megatron_bridge, torchtitan)${RESET}"
        echo -e "  ${DOT} 2) JAX ${DIM}(maxtext, maxdiffusion)${RESET}"
        echo -en " ${ARROW} Enter number or name: "
        read -r RELEASE_IN
    fi

    if [[ "${RELEASE_IN:-}" != "backend" ]]; then
    case "$RELEASE_IN" in
        1|primus|docker-primus|docker_primus|release-primus)
            DOCKER_RELEASE="primus"
            SUITE_BACKENDS=(megatron megatron_bridge torchtitan)
            set_backend_globals "megatron"
            ;;
        2|jax|docker-jax|docker_jax|release-jax|maxtext-release)
            DOCKER_RELEASE="jax"
            SUITE_BACKENDS=(maxtext maxdiffusion)
            set_backend_globals "maxtext"
            ;;
        *)
            echo -e "${RED}✗ Invalid Docker release suite: $RELEASE_IN${RESET}"
            exit 1
            ;;
    esac
    fi

    echo -e " ${CHECK} Docker release selected: ${GREEN}${DOCKER_RELEASE:-${AUTO_BENCHMARK_RELEASE_BACKEND}}${RESET} (${CYAN}${SUITE_BACKENDS[*]}${RESET})\n"
else
    if [[ -z "${BACKEND_IN:-}" ]]; then
        echo
        echo -e "${STAR} ${BOLD}Choose Backend:${RESET}"
        echo -e "  ${DOT} 1) megatron"
        echo -e "  ${DOT} 2) torchtitan"
        echo -e "  ${DOT} 3) megatron_bridge"
        echo -e "  ${DOT} 4) mlperf ${DIM}(MI355X only; not supported on MI300X/MI325X)${RESET}"
        echo -e "  ${DOT} 5) maxtext"
        echo -e "  ${DOT} 6) maxdiffusion ${DIM}(MI300X/MI355X configs; MI325X uses MI300X)${RESET}"
        echo -en " ${ARROW} Enter number or name: "
        read -r BACKEND_IN
    fi

    case "$BACKEND_IN" in
        1|megatron|MegaTron|MEGATRON) set_backend_globals "megatron" ;;
        2|torchtitan|TorchTitan|TORCHTITAN) set_backend_globals "torchtitan" ;;
        3|megatron_bridge|MegatronBridge|MEGATRON_BRIDGE|megatron-bridge)
            set_backend_globals "megatron_bridge"
            ;;
        4|mlperf|MLPerf|MLPERF) set_backend_globals "mlperf" ;;
        5|maxtext|MaxText|MAXTEXT) set_backend_globals "maxtext" ;;
        6|maxdiffusion|MaxDiffusion|MAXDIFFUSION|max-diffusion)
            set_backend_globals "maxdiffusion"
            ;;
        *)
            echo -e "${RED}✗ Invalid backend: $BACKEND_IN${RESET}"
            exit 1
            ;;
    esac

    echo -e " ${CHECK} Backend selected: ${GREEN}$BACKEND${RESET}\n"
fi
sleep 0.2

# ------------------------------------------
# 2. DEVICE DETECTION
# ------------------------------------------
echo -e "${STAR} ${BOLD}Detecting Device...${RESET}"

LOCAL_GPU_AVAILABLE=0
if [[ "${AUTO_BENCHMARK_NONINTERACTIVE:-0}" == "1" ]]; then
    DEVICE="${AUTO_BENCHMARK_DEVICE:?AUTO_BENCHMARK_DEVICE is required in batch mode}"
else

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

if is_valid_device "$DEVICE"; then
    LOCAL_GPU_AVAILABLE=1
fi

if ! is_valid_device "$DEVICE"; then
  if [[ -n "$ROCMINFO" ]]; then
    ARCH=$("$ROCMINFO" 2>/dev/null | grep -o 'gfx942\|gfx950' | head -n 1 | tr -d '[:space:]')
  else
    ARCH=""
  fi
  case "$ARCH" in
    "gfx950")
        DEVICE="MI355X"
        LOCAL_GPU_AVAILABLE=1
        ;;
    # gfx942 is shared by MI300X and MI325X; require manual selection if marketing name is missing
    "gfx942")
        LOCAL_GPU_AVAILABLE=1
        DEVICE=""
        ;;
    *) DEVICE="" ;;
  esac
fi

if [[ -z "$DEVICE" ]]; then
    echo -e "${RED}✗ Could not detect a local GPU. This is expected on a login node.${RESET}"
    if [[ -n "${DOCKER_RELEASE:-}" ]]; then
        echo -e "   ${DOT} Choose the target GPU. This release suite is submitted to Slurm after the token prompt."
    else
        echo -e "   ${DOT} Choose the target GPU. A run started here is submitted to Slurm when this machine has no GPU."
    fi
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
    if [[ "$DEVICE" == "MI325X" && ! -d "$CONFIG_DIR" ]]; then
        CONFIG_DIR="${BACKEND_BASE_DIR}/MI300X"
        echo -e " ${CHECK} MI325X config directory is not present. Using ${CYAN}MI300X${RESET} configs."
    fi
    echo -e " ${CHECK} Config directory set to: ${CYAN}$CONFIG_DIR${RESET}\n"
fi
sleep 0.2

# ------------------------------------------
# 3. MODEL SELECTION
# ------------------------------------------
SELECTED_CONFIGS=()
SELECTED_MLPERF_MODELS=()
RUN_ITEMS=()
USE_RELEASE_SUITE="n"

if [[ -n "${DOCKER_RELEASE:-}" || "${AUTO_BENCHMARK_RELEASE_SUITE:-0}" == "1" ]]; then
    if [[ -z "${DOCKER_RELEASE:-}" ]]; then
        SUITE_BACKENDS=("$BACKEND")
    fi
    USE_RELEASE_SUITE="y"
    if [[ "$DOCKER_RELEASE" == "primus" ]]; then
        echo -e "${STAR} ${BOLD}Docker release Primus${RESET} on ${CYAN}${DEVICE}${RESET}"
        echo -e "   ${DOT} Backends: ${CYAN}megatron, megatron_bridge, torchtitan${RESET}"
    elif [[ "$DOCKER_RELEASE" == "jax" ]]; then
        echo -e "${STAR} ${BOLD}Docker release JAX${RESET} on ${CYAN}${DEVICE}${RESET}"
        echo -e "   ${DOT} Backends: ${CYAN}maxtext, maxdiffusion${RESET}"
    else
        echo -e "${STAR} ${BOLD}Docker release ${BACKEND}${RESET} on ${CYAN}${DEVICE}${RESET}"
    fi
    echo -e "   ${DOT} Model, view, and edit prompts are skipped."
    collect_release_items
elif [[ "$BACKEND" == "mlperf" ]]; then
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
    for cfg in "${SELECTED_CONFIGS[@]}"; do
        RUN_ITEMS+=("${BACKEND}|${cfg}")
    done
fi
sleep 0.2

declare -A EDITED_CONFIGS

if [[ "$BACKEND" != "mlperf" && "$USE_RELEASE_SUITE" != "y" && "$USE_RELEASE_SUITE" != "Y" ]]; then
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
if [[ "$BACKEND" == "mlperf" ]]; then
    echo -e " ${CHECK} MLPerf launch mode: ${CYAN}module-native run script${RESET}"
else
    echo -e " ${CHECK} Primus launch mode: ${CYAN}$AUTO_BENCHMARK_MODE${RESET}"
    if [[ "$AUTO_BENCHMARK_MODE" == "container" ]]; then
        prompt_docker_image
        echo -e " ${CHECK} Docker image: ${CYAN}$DOCKER_IMAGE${RESET}"
    else
        echo -e " ${CHECK} This shell is inside a container, so training uses direct mode."
    fi
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
    echo -e " ${CHECK} Using HuggingFace token from the environment\n"
elif [[ "${AUTO_BENCHMARK_NONINTERACTIVE:-0}" == "1" ]]; then
    echo -e "${RED}✗ HF_TOKEN is not set in this batch job.${RESET}"
    exit 1
else
    echo -en " ${ARROW} Enter HuggingFace Token: "
    read -r -s HF_TOKEN
    echo
    export HF_TOKEN
    echo -e " ${CHECK} HuggingFace token set\n"
fi

if [[ "$BACKEND" == "mlperf" ]]; then
    export RUN_RUNTIME_TUNABLES="${RUN_RUNTIME_TUNABLES:-1}"
    echo -e " ${CHECK} MLPerf llama3.1_8b dataset root: ${CYAN}${MLPERF_LLAMA31_8B_ROOT}${RESET}"
    echo -e " ${CHECK} MLPerf llama2_70b uses fixed paths: ${CYAN}PACKED_DATA_DIR=/data${RESET}, ${CYAN}PRETRAINED_CHECKPOINT=/data/megatron_checkpoints/Llama-2-70b-hf${RESET}"
    echo -e " ${CHECK} MLPerf gpt_oss_20b dataset root: ${CYAN}${MLPERF_GPT_OSS_20B_ROOT}${RESET}"
    echo -e " ${CHECK} MLPerf flux1 dataset root: ${CYAN}${MLPERF_FLUX_ROOT}${RESET}"
    echo -e " ${CHECK} MLPerf runtime tunables: ${CYAN}RUN_RUNTIME_TUNABLES=${RUN_RUNTIME_TUNABLES}${RESET} (set to 0 to skip)\n"
fi

sleep 0.2

# ------------------------------------------
# 5. RUN BENCHMARK(S)
# ------------------------------------------
run_benchmarks
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
