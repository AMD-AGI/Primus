#!/bin/bash
# Run Llama 2 70B LoRA MLPerf (Megatron-Bridge, MI355X) via primus-cli.
# Dataset and model download are handled by posttrain hooks automatically.
# MLPerf overrides are applied at runtime from primus/backends/megatron_bridge/patches/mlperf_llama2_70b/.
#
# Usage (inside container):
#   export HF_TOKEN="hf_..."
#   bash /workspace/Primus/examples/mlperf/llama2_70b/run_in_container.sh
#
# Optional overrides:
#   PACKED_DATA_DIR=/path/to/data
#   MLLOG_VERBOSE_LOGS=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PRIMUS_ROOT="${PRIMUS_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

exec bash "${SCRIPT_DIR}/run_and_time.sh" "$@"
