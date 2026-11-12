#!/bin/bash
# Multi-node MLPerf Flux1 (Primus) launcher. Submit with: sbatch job.sh


#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
# #SBATCH --nodelist=<>             # uncomment to pin specific nodes
#SBATCH --time=60:00:00
#SBATCH --job-name=flux1_primus_mlperf
#SBATCH --output=flux1_primus_%j.out
#SBATCH --error=flux1_primus_%j.err
#SBATCH --partition=YOUR_PARTITION

set -x

# Fill these in before submitting.
export CONT="${CONT:?set CONT to the Primus flux1 image}"
export DATADIR="${DATADIR:-/path/to/dataset}"
export LOGDIR="${LOGDIR:-/path/to/results}"
: "${DGXSYSTEM:?source a config_MI355X_*.sh before sbatch}"
: "${NNODES:?source a config_MI355X_*.sh before sbatch}"
if [[ "${SLURM_JOB_NUM_NODES:-}" != "${NNODES}" ]]; then
    echo "config_${DGXSYSTEM}.sh needs ${NNODES} nodes but the job has ${SLURM_JOB_NUM_NODES:-?}; submit with: sbatch --nodes=\$NNODES job.sh" >&2
    exit 1
fi
export NEXP=${NEXP:-10}
export CHECK_COMPLIANCE=${CHECK_COMPLIANCE:-1}

export ROOT_SEED=${ROOT_SEED:-$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')}

# Stage the dataset locally if not already done
if [[ "${FLUX_STAGE_LOCAL:-1}" == 1 ]]; then
    if ! srun bash scripts/stage_dataset_local.sh; then
        echo "Node-local dataset staging failed; set FLUX_STAGE_LOCAL=0 to train from DATADIR." >&2
        exit 1
    fi
fi

srun bash run_with_docker_slurm.sh
