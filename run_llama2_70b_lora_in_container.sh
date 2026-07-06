#!/bin/bash
# Run Llama 2 70B LoRA MLPerf (Megatron-Bridge, MI355X) via primus-cli.
# Patches, dataset, and model download are handled by posttrain hooks automatically.
#
# Usage (inside container):
#   export HF_TOKEN="hf_..."
#   bash /workspace/Primus/run_llama2_70b_lora_in_container.sh
#
# Optional overrides:
#   PACKED_DATA_DIR=/path/to/data
#   SKIP_PATCHES=1
#   SKIP_TRAIN=1
#   MLLOG_VERBOSE_LOGS=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PRIMUS_ROOT="${PRIMUS_ROOT:-${SCRIPT_DIR}}"

if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
    export SKIP_PATCHES="${SKIP_PATCHES:-0}"
    exec bash "${PRIMUS_ROOT}/examples/megatron_bridge/llama2_70b_lora/apply_patches.sh"
fi

exec bash "${PRIMUS_ROOT}/examples/megatron_bridge/llama2_70b_lora/run_mlperf_cli.sh" "$@"
