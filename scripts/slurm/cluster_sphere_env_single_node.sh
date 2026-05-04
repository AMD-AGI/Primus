#!/usr/bin/env bash
###############################################################################
# Cluster Sphere — RDMA env recommender on one Slurm node (no torchrun).
# Run via: srun/salloc with -N1, or sbatch. Set PRIMUS_ROOT to Primus repo root if unset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_ROOT="${PRIMUS_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export PYTHONPATH="${PRIMUS_ROOT}${PYTHONPATH:+:${PYTHONPATH:-}}"

OUT="${CLUSTER_SPHERE_OUT:-${SLURM_SUBMIT_DIR:-.}/cluster_sphere_env_${SLURM_JOB_ID:-local}.md}"

python3 -m primus.tools.preflight.cluster_sphere env --markdown >"${OUT}"
echo "Wrote ${OUT}" >&2
