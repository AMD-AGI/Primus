#!/usr/bin/env bash
set -euo pipefail

DOCKER_IMAGE="rocm/primus:v26.2"
#DOCKER_IMAGE="docker.gpuperf:5000/aai_2026_training/rocm/primus_megatron:v25.11_gpt_oss_sink"
#DOCKER_IMAGE="docker.gpuperf:5000/gpuperf/primus:v26.1_sinkfa"
NNODES=2
PARTITION="Compute-DCPT"
SCRIPT="./run_slurm.sh"

NODELISTS=(
  "smci355-ccs-aus-n01-21,smci355-ccs-aus-n01-25"
  "smci355-ccs-aus-n01-33,smci355-ccs-aus-n02-21"
  "smci355-ccs-aus-n02-25,smci355-ccs-aus-n02-29"
  "smci355-ccs-aus-n03-25,smci355-ccs-aus-n03-33"
  "smci355-ccs-aus-n04-21,smci355-ccs-aus-n04-25"
  "smci355-ccs-aus-n04-29,smci355-ccs-aus-n04-33"
  "smci355-ccs-aus-n05-21,smci355-ccs-aus-n05-29"
  "smci355-ccs-aus-n05-33,smci355-ccs-aus-n06-25"
  "smci355-ccs-aus-n06-33,smci355-ccs-aus-n10-29"
)

echo "Submitting ${#NODELISTS[@]} jobs..."
export DOCKER_IMAGE NNODES
for nodelist in "${NODELISTS[@]}"; do
  echo ">> $nodelist"
  sbatch -N "${NNODES}" -w "$nodelist" -p "$PARTITION" "$SCRIPT"
done

echo "Done."
