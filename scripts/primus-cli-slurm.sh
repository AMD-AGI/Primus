#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -euo pipefail

print_usage() {
cat <<'EOF'
Primus Slurm Launcher

Usage:
    primus-cli slurm [srun|sbatch] [SLURM_FLAGS...] -- <entry> [ENTRY_ARGS...] -- [PRIMUS_ARGS...]

Description:
    Launch distributed Primus jobs via Slurm.
    - Everything before the first '--' is passed to Slurm (srun/sbatch and flags).
    - <entry> specifies Primus execution mode: container | direct | preflight (see below).
    - The second '--' (if any) separates Primus entry args from Primus CLI arguments.

Examples:
    # Launch 4 nodes using srun and container mode
    primus-cli slurm srun -N 4 -p AIG_Model -- container -- train pretrain --config exp.yaml

    # # Launch with sbatch, log to file, run benchmark
    primus-cli slurm sbatch --output=run.log -N 2 -- container -- benchmark gemm -M 4096 -N 4096 -K 4096

    # Run preflight environment check across 4 nodes
    primus-cli slurm srun -N 4 -- preflight

Notes:
    - [srun|sbatch] is optional; defaults to srun if not specified.
    - All SLURM_FLAGS before '--' are passed directly to Slurm (supports both --flag=value and --flag value).
    - Everything after the first '--' is passed to Primus entry (e.g. container, direct, etc.), and then to Primus CLI.
    - For unsupported or extra Slurm options, just pass them after '--' (they'll be ignored by this wrapper).

Debug:
    - Collected SLURM flags and primus arguments will be printed before launch.

EOF
}

# Show help if requested or if no args are given
if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
    print_usage
    exit 0
fi

# 1. Detect srun/sbatch mode
LAUNCH_CMD="srun"   # Default launcher
if [[ "${1:-}" == "sbatch" || "${1:-}" == "srun" ]]; then
    LAUNCH_CMD="$1"
    shift
fi

# 2. Collect SLURM_FLAGS before '--'
SLURM_FLAGS=()
while [[ $# -gt 0 && "$1" != "--" ]]; do
    SLURM_FLAGS+=("$1")
    shift
done

# Skip '--'
if [[ "$#" -gt 0 && "$1" == "--" ]]; then
    shift
fi

# 3. Check for primus-run args
if [[ $# -eq 0 ]]; then
    echo "[primus-cli-slurm][ERROR] Missing Primus entry (container|direct|preflight)" >&2
    print_usage >&2
    exit 2
fi

# 4. Logging and launch
SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
ENTRY="$SCRIPT_DIR/primus-cli-slurm-entry.sh"
echo "[primus-cli-slurm] Executing: $LAUNCH_CMD ${SLURM_FLAGS[*]} $ENTRY $*"
exec "$LAUNCH_CMD" "${SLURM_FLAGS[@]}" "$ENTRY" "$@"
