#!/usr/bin/env bash
# Removed: set -e (allow script to continue on errors in benchmarks)

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

# Check if run_pretrain.sh exists, otherwise try run_pretrain_1.sh
if [[ ! -f "$RUN_SCRIPT" && -f "${PRIMUS_ROOT}/examples/run_pretrain_1.sh" ]]; then
    RUN_SCRIPT="${PRIMUS_ROOT}/examples/run_pretrain_1.sh"
    echo "[DEBUG] Using run_pretrain_1.sh instead"
fi

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
read BACKEND_IN

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

DEVICE=$(/opt/rocm/bin/rocminfo | grep "AMD Instinct" | head -n1 | awk '{print $5}')
echo -e " ${DOT} Device found: ${CYAN}$DEVICE${RESET}"

if [[ -z "$DEVICE" || "$DEVICE" != "MI300X" && "$DEVICE" != "MI355X" ]]; then
  ARCH=$(/opt/rocm/bin/rocminfo | grep -o 'gfx942\|gfx950' | head -n 1 | tr -d '[:space:]')
  case "$ARCH" in
    "gfx942") DEVICE="MI300X" ;;
    "gfx950") DEVICE="MI355X" ;;
    *) DEVICE="" ;;
  esac
fi

if [[ -z "$DEVICE" ]]; then
    echo -e "${RED}✗ Could not detect device automatically${RESET}"
    echo -e "${STAR} ${BOLD}Please select Device manually:${RESET}"
    echo -e "  ${DOT} 1) MI300X"
    echo -e "  ${DOT} 2) MI355X"
    
    echo -en " ${ARROW} Enter number or name: "
    read DEV_IN
    
    case "$DEV_IN" in
        1|MI300X|mi300x|Mi300x)
            DEVICE="MI300X"
            ;;
        2|MI355X|mi355x|Mi355x)
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
CONFIG_LIST=($(find "$CONFIG_DIR" -name "*.yaml" -type f | sort -u))

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
    echo -e "  ${DOT} ${i}) $(basename $cfg)"
    ((i++))
done
echo

CONFIG_LIST=("${UNIQUE_CONFIGS[@]}")

echo -en " ${ARROW} Select config number(s) (comma-separated, range, or 'all'): "
echo -e "${DIM}(Examples: 1,3,5 or 4-8 or all)${RESET}"
echo -en " ${ARROW} "
read CFG_NUM

# Parse input into array
SELECTED_CONFIGS=()

if [[ "$CFG_NUM" == "all" ]]; then
    # Select all configs
    SELECTED_CONFIGS=("${CONFIG_LIST[@]}")
elif [[ "$CFG_NUM" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    # Handle range input like 4-8
    START="${BASH_REMATCH[1]}"
    END="${BASH_REMATCH[2]}"
    
    if [[ $START -lt 1 || $END -gt ${#CONFIG_LIST[@]} || $START -gt $END ]]; then
        echo -e "${RED}✗ Invalid range: $START-$END${RESET}"
        exit 1
    fi
    
    for ((i=START; i<=END; i++)); do
        SELECTED_CONFIGS+=("${CONFIG_LIST[$i-1]}")
    done
else
    # Handle comma-separated input
    IFS=',' read -ra CFG_NUMS <<< "$CFG_NUM"
    
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

echo -e " ${CHECK} Selected ${GREEN}${#SELECTED_CONFIGS[@]}${RESET} config(s):"
for cfg in "${SELECTED_CONFIGS[@]}"; do
    echo -e "    ${DOT} $(basename $cfg)"
done
echo
sleep 0.2

# ------------------------------------------
# 2.5. VIEW & OVERRIDE PARAMETERS
# ------------------------------------------
echo -e "${STAR} ${BOLD}View Configuration Parameters?${RESET}"
echo -en " ${ARROW} (y/n): "
read VIEW_PARAMS

if [[ "$VIEW_PARAMS" == "y" || "$VIEW_PARAMS" == "Y" ]]; then
    for cfg in "${SELECTED_CONFIGS[@]}"; do
        echo -e "\n${CYAN}${BOLD}Parameters in $(basename $cfg):${RESET}"
        echo -e "${DIM}-----------------------------------${RESET}"
        cat "$cfg" | grep -v "^#" | grep -v "^$"
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
    read EDIT_CONFIGS
    
    if [[ "$EDIT_CONFIGS" == "y" || "$EDIT_CONFIGS" == "Y" ]]; then
        echo -e "\n${CYAN}${BOLD}Selected models:${RESET}"
        i=1
        for cfg in "${SELECTED_CONFIGS[@]}"; do
            echo -e "  ${DOT} ${i}) $(basename $cfg)"
            ((i++))
        done
        echo
        
        echo -e " ${DOT} Enter model numbers to edit (comma-separated, or 'all'): "
        echo -en " ${ARROW} "
        read EDIT_SELECTION
        
        if [[ "$EDIT_SELECTION" == "all" ]]; then
            MODELS_TO_EDIT=("${!SELECTED_CONFIGS[@]}")
        else
            IFS=',' read -ra EDIT_NUMS <<< "$EDIT_SELECTION"
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
            
            # Try to find an editor
            if command -v nano &> /dev/null; then
                nano "$TEMP_EDIT_CONFIG"
            elif command -v vim &> /dev/null; then
                vim "$TEMP_EDIT_CONFIG"
            elif command -v vi &> /dev/null; then
                vi "$TEMP_EDIT_CONFIG"
            elif command -v code &> /dev/null; then
                code --wait "$TEMP_EDIT_CONFIG"
            else
                ${EDITOR:-vi} "$TEMP_EDIT_CONFIG"
            fi
            
            # Store the edited config
            EDITED_CONFIGS["$cfg"]="$TEMP_EDIT_CONFIG"
            echo -e " ${CHECK} ${GREEN}Config edited and saved${RESET}\n"
        done
    fi
elif [[ ${#SELECTED_CONFIGS[@]} -eq 1 ]]; then
    echo -e "\n${STAR} ${BOLD}Edit configuration file before running?${RESET}"
    echo -en " ${ARROW} (y/n): "
    read EDIT_SINGLE
    
    if [[ "$EDIT_SINGLE" == "y" || "$EDIT_SINGLE" == "Y" ]]; then
        cfg="${SELECTED_CONFIGS[0]}"
        model_name=$(basename "$cfg" .yaml)
        
        echo -e "\n${STAR} ${BOLD}Opening config for editing: ${CYAN}$model_name${RESET}"
        echo -e "   ${DOT} Edit the file, save, and close the editor to continue\n"
        
        # Create a temporary working copy
        TEMP_EDIT_CONFIG="/tmp/primus_edit_${model_name}_$$.yaml"
        cp "$cfg" "$TEMP_EDIT_CONFIG"
        
        # Try to find an editor
        if command -v nano &> /dev/null; then
            nano "$TEMP_EDIT_CONFIG"
        elif command -v vim &> /dev/null; then
            vim "$TEMP_EDIT_CONFIG"
        elif command -v vi &> /dev/null; then
            vi "$TEMP_EDIT_CONFIG"
        elif command -v code &> /dev/null; then
            code --wait "$TEMP_EDIT_CONFIG"
        else
            ${EDITOR:-vi} "$TEMP_EDIT_CONFIG"
        fi
        
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
read OVERRIDE_PARAMS

if [[ "$OVERRIDE_PARAMS" == "y" || "$OVERRIDE_PARAMS" == "Y" ]]; then
    echo -e " ${DOT} Enter overrides one per line. Press Enter on empty line to finish."
    while true; do
        echo -en " ${ARROW} Override (or press Enter to finish): "
        read OVERRIDE_LINE
        
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
        echo -e "\n ${CHECK} ${GREEN}${#PARAM_OVERRIDES[@]}${RESET} parameter(s) will be overridden\n"
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
read ADD_ENV_VARS

if [[ "$ADD_ENV_VARS" == "y" || "$ADD_ENV_VARS" == "Y" ]]; then
    echo -e " ${DOT} Enter environment variables one per line. Press Enter on empty line to finish."
    while true; do
        echo -en " ${ARROW} Variable (or press Enter to finish): "
        read ENV_LINE
        
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
        echo -e "\n ${CHECK} ${GREEN}${#DEVICE_ENV_VARS[@]}${RESET} environment variable(s) will be set\n"
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
        export "$ENV_VAR"
        echo -e " ${CHECK} Set ${CYAN}$ENV_VAR${RESET}"
    done
fi

# Prompt for HuggingFace token
echo -en " ${ARROW} Enter HuggingFace Token: "
read -s HF_TOKEN
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
    
    # Extract full filename without extension to preserve all details
    CONFIG_FILENAME=$(basename "$CFG_FILE" .yaml)
    MODEL_NAME="$CONFIG_FILENAME"
    TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

    # Use the full config filename in the log name to preserve all details
    LOG_FILE="$LOG_DIR/${CONFIG_FILENAME}_${BACKEND}_${DEVICE}_${TIMESTAMP}.log"

    # Use edited config if available, otherwise use original
    if [[ -n "${EDITED_CONFIGS[$CFG_FILE]}" ]]; then
        WORKING_CONFIG="${EDITED_CONFIGS[$CFG_FILE]}"
        echo -e "${INFO} ${BOLD}Using edited config for ${CYAN}$MODEL_NAME${RESET}"
    else
        WORKING_CONFIG="$CFG_FILE"
    fi
    
    # Apply parameter overrides if any
    if [[ ${#PARAM_OVERRIDES[@]} -gt 0 ]]; then
        OVERRIDE_CONFIG="$LOG_DIR/${MODEL_NAME}_${BACKEND}_${DEVICE}_${TIMESTAMP}_override.yaml"
        cp "$WORKING_CONFIG" "$OVERRIDE_CONFIG"
        
        echo -e "${STAR} ${BOLD}Applying parameter overrides...${RESET}"
        for KEY in "${!PARAM_OVERRIDES[@]}"; do
            VALUE="${PARAM_OVERRIDES[$KEY]}"
            # Use sed to replace the parameter value in YAML (handles both 'key: value' and 'key:value')
            sed -i "s|^\([[:space:]]*${KEY}:[[:space:]]*\).*|\1${VALUE}|g" "$OVERRIDE_CONFIG"
            echo -e "   ${DOT} ${CYAN}$KEY${RESET}: $VALUE"
        done
        WORKING_CONFIG="$OVERRIDE_CONFIG"
        echo -e " ${CHECK} Override config saved: ${YELLOW}$OVERRIDE_CONFIG${RESET}\n"
    elif [[ -n "${EDITED_CONFIGS[$CFG_FILE]}" ]]; then
        # Save edited config to logs directory
        SAVED_CONFIG="$LOG_DIR/${MODEL_NAME}_${BACKEND}_${DEVICE}_${TIMESTAMP}_edited.yaml"
        cp "$WORKING_CONFIG" "$SAVED_CONFIG"
        WORKING_CONFIG="$SAVED_CONFIG"
    fi

    echo -e "${STAR} ${BOLD}Starting Benchmark ${CURRENT}/${TOTAL_CONFIGS}...${RESET}"
    echo -e "   ${DOT} Model: ${CYAN}$MODEL_NAME${RESET}"
    echo -e "   ${DOT} Backend: ${CYAN}$BACKEND${RESET}"
    echo -e "   ${DOT} Device: ${CYAN}$DEVICE${RESET}"
    echo -e "   ${DOT} Config: ${YELLOW}$WORKING_CONFIG${RESET}"
    echo -e "   ${DOT} Log: ${YELLOW}$LOG_FILE${RESET}\n"

    # Set EXP to the working config (edited or overridden version if available)
    # For edited/overridden configs, we need to copy them to the expected location
    if [[ "$WORKING_CONFIG" != "$CFG_FILE" ]]; then
        # Config was edited or overridden, copy it to the original location temporarily
        ORIGINAL_CONFIG_BACKUP="${CFG_FILE}.backup_$$"
        cp "$CFG_FILE" "$ORIGINAL_CONFIG_BACKUP"
        cp "$WORKING_CONFIG" "$CFG_FILE"
        echo -e " ${CHECK} Copied edited/overridden config to: ${CYAN}$CFG_FILE${RESET}"
    fi
    
    # Set EXP to the device-specific config path (now contains edited content if applicable)
    EXP_CONFIG_PATH="${BACKEND_BASE_DIR}/${DEVICE}/$(basename "$CFG_FILE")"
    export EXP="$EXP_CONFIG_PATH"
    echo -e " ${CHECK} EXP set to: ${CYAN}$EXP${RESET}\n"

    # Change to Primus root directory before running the script
    echo -e " ${DOT} Changing to Primus root directory: ${CYAN}$PRIMUS_ROOT${RESET}"
    cd "$PRIMUS_ROOT"

    # Run the script and capture exit code, but don't stop on failure
    # Use 'set +e' locally to ensure we continue even on errors
    set +e
    bash $RUN_SCRIPT 2>&1 | tee "$LOG_FILE" || true
    RUN_EXIT_CODE=$?
    set -e

    # Return to script directory
    cd "$SCRIPT_DIR"

    # Restore original config if it was temporarily replaced
    if [[ -n "$ORIGINAL_CONFIG_BACKUP" && -f "$ORIGINAL_CONFIG_BACKUP" ]]; then
        mv "$ORIGINAL_CONFIG_BACKUP" "$CFG_FILE"
        echo -e " ${CHECK} Restored original config file"
        unset ORIGINAL_CONFIG_BACKUP
    fi

    echo
    echo -e "${GREEN}==========================================${RESET}"
    if [[ $RUN_EXIT_CODE -eq 0 ]]; then
        echo -e " ${BOLD}${GREEN}✓ Benchmark ${CURRENT}/${TOTAL_CONFIGS} Completed Successfully!${RESET}"
    else
        echo -e " ${BOLD}${YELLOW}⚠ Benchmark ${CURRENT}/${TOTAL_CONFIGS} Completed with Exit Code: $RUN_EXIT_CODE${RESET}"
    fi
    echo -e " Log saved at:"
    echo -e "   ${CYAN}$LOG_FILE${RESET}"
    if [[ ${#PARAM_OVERRIDES[@]} -gt 0 ]]; then
        echo -e " Override config saved at:"
        echo -e "   ${CYAN}$OVERRIDE_CONFIG${RESET}"
    fi
    echo -e "${GREEN}==========================================${RESET}"
    echo

    CURRENT=$((CURRENT + 1))

    # Add a short delay between runs
    if [[ $CURRENT -le $TOTAL_CONFIGS ]]; then
        echo -e "${YELLOW}Preparing next benchmark...${RESET}\n"
        echo -e "${INFO} ${BOLD}Next: Config ${CURRENT}/${TOTAL_CONFIGS}${RESET}\n"
        sleep 2
    fi
done

echo
echo -e "${MAGENTA}${BOLD}=========================================${RESET}"
echo -e "${MAGENTA}${BOLD}  All ${TOTAL_CONFIGS} Benchmark(s) Completed!${RESET}"
echo -e "${MAGENTA}${BOLD}=========================================${RESET}"

# ------------------------------------------
# 7. GENERATE METRICS TABLE
# ------------------------------------------
echo
echo -e "${STAR} ${BOLD}Generating Metrics Table...${RESET}\n"

if [[ "$BACKEND" == "megatron" ]]; then
    METRICS_SCRIPT="metrics_megatron.py"
elif [[ "$BACKEND" == "torchtitan" ]]; then
    METRICS_SCRIPT="metrics_torchtitan.py"
fi

if [[ -f "$METRICS_SCRIPT" ]]; then
    echo -e " ${CHECK} Running: ${CYAN}python $METRICS_SCRIPT${RESET}\n"
    python "$METRICS_SCRIPT"
    echo
    echo -e " ${CHECK} ${GREEN}Metrics table generated successfully${RESET}"
else
    echo -e " ${RED}✗ Metrics script not found: $METRICS_SCRIPT${RESET}"
fi
