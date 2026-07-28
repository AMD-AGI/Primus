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
TORCHTITAN_BASE_DIR="${PRIMUS_ROOT}/examples/torchtitan/configs"
RUN_SCRIPT="${PRIMUS_ROOT}/examples/run_pretrain.sh"
VALID_DEVICES=(MI300X MI325X MI355X)

# Check if run_pretrain.sh exists, otherwise try run_pretrain_1.sh
if [[ ! -f "$RUN_SCRIPT" && -f "${PRIMUS_ROOT}/examples/run_pretrain_1.sh" ]]; then
    RUN_SCRIPT="${PRIMUS_ROOT}/examples/run_pretrain_1.sh"
    echo "[DEBUG] Using run_pretrain_1.sh instead"
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

    if [[ -n "${EDITED_CONFIGS[$cfg_file]:-}" ]]; then
        PREP_WORKING_CONFIG="${EDITED_CONFIGS[$cfg_file]}"
    else
        PREP_WORKING_CONFIG="$cfg_file"
    fi

    if [[ ${#PARAM_OVERRIDES[@]} -gt 0 ]]; then
        PREP_WORKING_CONFIG="$LOG_DIR/${artifact_prefix}_override.yaml"
        cp "${EDITED_CONFIGS[$cfg_file]:-$cfg_file}" "$PREP_WORKING_CONFIG"
        for KEY in "${!PARAM_OVERRIDES[@]}"; do
            sed -i "s|^\([[:space:]]*${KEY}:[[:space:]]*\).*|\1${PARAM_OVERRIDES[$KEY]}|g" "$PREP_WORKING_CONFIG"
        done
    elif [[ -n "${EDITED_CONFIGS[$cfg_file]:-}" ]]; then
        PREP_WORKING_CONFIG="$LOG_DIR/${artifact_prefix}_edited.yaml"
        cp "${EDITED_CONFIGS[$cfg_file]}" "$PREP_WORKING_CONFIG"
    fi
}

execute_benchmark_run() {
    local cfg_file="$1"
    local working_config="$2"
    local log_file="$3"
    local model_name="$4"
    local current="$5"
    local total="$6"

    local original_config_backup=""
    local run_exit_code=0

    echo -e "${STAR} ${BOLD}Starting Benchmark ${current}/${total}...${RESET}"
    echo -e "   ${DOT} Model: ${CYAN}$model_name${RESET}"
    echo -e "   ${DOT} Backend: ${CYAN}$BACKEND${RESET}"
    echo -e "   ${DOT} Device: ${CYAN}$DEVICE${RESET}"
    echo -e "   ${DOT} Config: ${YELLOW}$working_config${RESET}"
    echo -e "   ${DOT} Log: ${YELLOW}$log_file${RESET}\n"

    if [[ "$working_config" != "$cfg_file" ]]; then
        original_config_backup="${cfg_file}.backup_$$"
        cp "$cfg_file" "$original_config_backup"
        cp "$working_config" "$cfg_file"
        echo -e " ${CHECK} Copied edited/overridden config to: ${CYAN}$cfg_file${RESET}"
    fi

    EXP="${BACKEND_BASE_DIR}/${DEVICE}/$(basename "$cfg_file")"
    export EXP
    echo -e " ${CHECK} EXP set to: ${CYAN}$EXP${RESET}\n"

    echo -e " ${DOT} Changing to Primus root directory: ${CYAN}$PRIMUS_ROOT${RESET}"
    cd "$PRIMUS_ROOT" || return 1

    set +e
    bash "$RUN_SCRIPT" 2>&1 | tee "$log_file" || true
    run_exit_code=$?
    set +e

    cd "$SCRIPT_DIR" || return 1

    if [[ -n "$original_config_backup" && -f "$original_config_backup" ]]; then
        mv "$original_config_backup" "$cfg_file"
        echo -e " ${CHECK} Restored original config file"
    fi

    echo
    echo -e "${GREEN}==========================================${RESET}"
    if [[ $run_exit_code -eq 0 ]]; then
        echo -e " ${BOLD}${GREEN}✓ Benchmark ${current}/${total} Completed Successfully!${RESET}"
    else
        echo -e " ${BOLD}${YELLOW}⚠ Benchmark ${current}/${total} Completed with Exit Code: $run_exit_code${RESET}"
    fi
    echo -e " Log saved at:"
    echo -e "   ${CYAN}$log_file${RESET}"
    if [[ ${#PARAM_OVERRIDES[@]} -gt 0 ]]; then
        echo -e " Override config saved at:"
        echo -e "   ${CYAN}$working_config${RESET}"
    fi
    echo -e "${GREEN}==========================================${RESET}"
    echo

    return "$run_exit_code"
}

generate_metrics_table() {
    local metrics_script="metrics.py"

    echo
    echo -e "${STAR} ${BOLD}Generating Metrics Table...${RESET}\n"

    if [[ -f "$SCRIPT_DIR/$metrics_script" && ( "$BACKEND" == "megatron" || "$BACKEND" == "torchtitan" ) ]]; then
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
sleep 0.2

# ------------------------------------------
# 2.5. SET DEVICE-SPECIFIC CONFIG DIRECTORY
# ------------------------------------------
CONFIG_DIR="${BACKEND_BASE_DIR}/${DEVICE}"
echo -e " ${CHECK} Config directory set to: ${CYAN}$CONFIG_DIR${RESET}\n"
sleep 0.2

# ------------------------------------------
# 3. MODEL CONFIG SELECTION
# ------------------------------------------
echo -e "${STAR} ${BOLD}Available Model Configs:${RESET} (${CYAN}$BACKEND${RESET} / ${CYAN}$DEVICE${RESET})"

# Use find and sort -u to get unique files
mapfile -t CONFIG_LIST < <(find "$CONFIG_DIR" -name "*.yaml" -type f | sort -u)

if [[ ${#CONFIG_LIST[@]} -eq 0 ]]; then
    echo -e "${RED}No configs found in $CONFIG_DIR${RESET}"
    exit 1
fi

# Store unique model names to avoid duplicates
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

# Parse input into array
SELECTED_CONFIGS=()

if [[ "$CFG_NUM" == "all" ]]; then
    # Select all configs
    SELECTED_CONFIGS=("${CONFIG_LIST[@]}")
elif [[ "$CFG_NUM" =~ ^[0-9]+-[0-9]+$ ]]; then
    # Handle range input like 4-8
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
    # Handle comma-separated input
    _saved_ifs=$IFS
    IFS=',' read -ra CFG_NUMS <<< "$CFG_NUM"
    IFS=$_saved_ifs

    for num in "${CFG_NUMS[@]}"; do
        # Trim whitespace
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
sleep 0.2

# ------------------------------------------
# 2.5. VIEW & OVERRIDE PARAMETERS
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
# 2.6. EDIT CONFIG FILES
# ------------------------------------------
declare -A EDITED_CONFIGS

if [[ ${#SELECTED_CONFIGS[@]} -gt 1 ]]; then
    echo -e "${STAR} ${BOLD}Edit any configuration files before running?${RESET}"
    echo -en " ${ARROW} (y/n): "
    read -r EDIT_CONFIGS

    if [[ "$EDIT_CONFIGS" == "y" || "$EDIT_CONFIGS" == "Y" ]]; then
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

        # Edit selected files one by one
        for idx in "${MODELS_TO_EDIT[@]}"; do
            cfg="${SELECTED_CONFIGS[$idx]}"
            model_name=$(basename "$cfg" .yaml)

            echo -e "\n${STAR} ${BOLD}Opening config for editing: ${CYAN}$model_name${RESET}"
            echo -e "   ${DOT} Edit the file, save, and close the editor to continue\n"

            # Create a temporary working copy
            TEMP_EDIT_CONFIG="/tmp/primus_edit_${model_name}_$$.yaml"
            cp "$cfg" "$TEMP_EDIT_CONFIG"

            open_config_editor "$TEMP_EDIT_CONFIG"

            # Store the edited config
            EDITED_CONFIGS["$cfg"]="$TEMP_EDIT_CONFIG"
            echo -e " ${CHECK} ${GREEN}Config edited and saved${RESET}\n"
        done
    fi
elif [[ ${#SELECTED_CONFIGS[@]} -eq 1 ]]; then
    echo -e "\n${STAR} ${BOLD}Edit configuration file before running?${RESET}"
    echo -en " ${ARROW} (y/n): "
    read -r EDIT_SINGLE

    if [[ "$EDIT_SINGLE" == "y" || "$EDIT_SINGLE" == "Y" ]]; then
        cfg="${SELECTED_CONFIGS[0]}"
        model_name=$(basename "$cfg" .yaml)

        echo -e "\n${STAR} ${BOLD}Opening config for editing: ${CYAN}$model_name${RESET}"
        echo -e "   ${DOT} Edit the file, save, and close the editor to continue\n"

        # Create a temporary working copy
        TEMP_EDIT_CONFIG="/tmp/primus_edit_${model_name}_$$.yaml"
        cp "$cfg" "$TEMP_EDIT_CONFIG"

        open_config_editor "$TEMP_EDIT_CONFIG"

        # Store the edited config
        EDITED_CONFIGS["$cfg"]="$TEMP_EDIT_CONFIG"
        echo -e " ${CHECK} ${GREEN}Config edited and saved${RESET}\n"
    fi
fi

# Initialize associative array for parameter overrides
declare -A PARAM_OVERRIDES

echo -e "\n${STAR} ${BOLD}Override any parameters?${RESET}"
echo -e "  ${DIM}(Format: key=value, e.g., batch_size=32)${RESET}"
echo -en " ${ARROW} (y/n): "
read -r OVERRIDE_PARAMS

if [[ "$OVERRIDE_PARAMS" == "y" || "$OVERRIDE_PARAMS" == "Y" ]]; then
    echo -e " ${DOT} Enter overrides one per line. Press Enter on empty line to finish."
    while true; do
        echo -en " ${ARROW} Override (or press Enter to finish): "
        read -r OVERRIDE_LINE

        if [[ -z "$OVERRIDE_LINE" ]]; then
            break
        fi

        # Parse key=value
        if [[ "$OVERRIDE_LINE" =~ ^([^=]+)=(.+)$ ]]; then
            KEY="${BASH_REMATCH[1]}"
            VALUE="${BASH_REMATCH[2]}"
            PARAM_OVERRIDES["$KEY"]="$VALUE"
            echo -e " ${CHECK} Will override: ${CYAN}$KEY${RESET} = ${GREEN}$VALUE${RESET}"
        else
            echo -e " ${RED}Invalid format. Use: key=value${RESET}"
        fi
    done

    if [[ ${#PARAM_OVERRIDES[@]} -gt 0 ]]; then
        PARAM_OVERRIDE_COUNT=${#PARAM_OVERRIDES[@]}
        echo -e "\n ${CHECK} ${GREEN}${PARAM_OVERRIDE_COUNT}${RESET} parameters will be overridden\n"
    fi
fi

sleep 0.2

# ------------------------------------------
# 4. DEVICE-SPECIFIC ENVIRONMENT VARIABLES
# ------------------------------------------
declare -a DEVICE_ENV_VARS

echo -e "${STAR} ${BOLD}Add device-specific environment variables for ${DEVICE}?${RESET}"
echo -e "  ${DIM}(e.g., HSA_OVERRIDE_GFX_VERSION=11.0.0)${RESET}"
echo -en " ${ARROW} (y/n): "
read -r ADD_ENV_VARS

if [[ "$ADD_ENV_VARS" == "y" || "$ADD_ENV_VARS" == "Y" ]]; then
    echo -e " ${DOT} Enter environment variables one per line. Press Enter on empty line to finish."
    while true; do
        echo -en " ${ARROW} Variable (or press Enter to finish): "
        read -r ENV_LINE

        if [[ -z "$ENV_LINE" ]]; then
            break
        fi

        # Parse VAR=value
        if [[ "$ENV_LINE" =~ ^([^=]+)=(.*)$ ]]; then
            VAR_NAME="${BASH_REMATCH[1]}"
            VAR_VALUE="${BASH_REMATCH[2]}"
            DEVICE_ENV_VARS+=("$VAR_NAME=$VAR_VALUE")
            echo -e " ${CHECK} Will set: ${CYAN}${VAR_NAME}${RESET}=${GREEN}${VAR_VALUE}${RESET}"
        else
            echo -e " ${RED}Invalid format. Use: VAR_NAME=value${RESET}"
        fi
    done

    if [[ ${#DEVICE_ENV_VARS[@]} -gt 0 ]]; then
        DEVICE_ENV_VAR_COUNT=${#DEVICE_ENV_VARS[@]}
        echo -e "\n ${CHECK} ${GREEN}${DEVICE_ENV_VAR_COUNT}${RESET} environment variables will be set\n"
    fi
fi

sleep 0.2

# ------------------------------------------
# 5. ENVIRONMENT SETUP
# ------------------------------------------
echo -e "${STAR} ${BOLD}Setting up environment...${RESET}"

# Set HSA environment variable
export HSA_NO_SCRATCH_RECLAIM=1
echo -e " ${CHECK} Set ${CYAN}HSA_NO_SCRATCH_RECLAIM=1${RESET}"

# Apply device-specific environment variables
if [[ ${#DEVICE_ENV_VARS[@]} -gt 0 ]]; then
    for ENV_VAR in "${DEVICE_ENV_VARS[@]}"; do
        eval export "$ENV_VAR"
        echo -e " ${CHECK} Set ${CYAN}$ENV_VAR${RESET}"
    done
fi

# Prompt for HuggingFace token
echo -en " ${ARROW} Enter HuggingFace Token: "
read -r -s HF_TOKEN
echo
export HF_TOKEN
echo -e " ${CHECK} HuggingFace token set\n"

sleep 0.2

# ------------------------------------------
# 6. RUN BENCHMARK(S)
# ------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="${SCRIPT_DIR}/results/logs_${BACKEND}"
mkdir -p "$LOG_DIR"

TOTAL_CONFIGS=${#SELECTED_CONFIGS[@]}
CURRENT=1

echo -e "${INFO} ${BOLD}Total configurations to run: ${TOTAL_CONFIGS}${RESET}"
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

    if [[ -n "${EDITED_CONFIGS[$CFG_FILE]:-}" && ${#PARAM_OVERRIDES[@]} -eq 0 ]]; then
        echo -e "${INFO} ${BOLD}Using edited config for ${CYAN}$PREP_MODEL_NAME${RESET}"
    fi
    echo -e "   ${DOT} Run label: ${CYAN}$PREP_RUN_LABEL${RESET}"
    if [[ ${#PARAM_OVERRIDES[@]} -gt 0 ]]; then
        echo -e "${STAR} ${BOLD}Applying parameter overrides...${RESET}"
        for KEY in "${!PARAM_OVERRIDES[@]}"; do
            echo -e "   ${DOT} ${CYAN}$KEY${RESET}: ${PARAM_OVERRIDES[$KEY]}"
        done
        echo -e " ${CHECK} Override config saved: ${YELLOW}$PREP_WORKING_CONFIG${RESET}\n"
    fi

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

echo
echo -e "${MAGENTA}${BOLD}=========================================${RESET}"
echo -e "${MAGENTA}${BOLD}  All ${TOTAL_CONFIGS} benchmarks completed!${RESET}"
echo -e "${MAGENTA}${BOLD}=========================================${RESET}"

# ------------------------------------------
# 7. GENERATE METRICS TABLE
# ------------------------------------------
generate_metrics_table
