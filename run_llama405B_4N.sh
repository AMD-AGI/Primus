#!/bin/bash
export USING_AINIC=1
export REBUILD_PRIMUS_TURBO=1
# export NCCL_IB_HCA="rocep105s0,rocep121s0,rocep137s0,rocep153s0,rocep233s0,rocep249s0,rocep25s0,rocep9s0"
export NCCL_IB_HCA="ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7"
# export ANP_HOME_DIR="/shared/apps/ubuntu/rocm-7.0.1/amd-anp-1.1.0-5"
# export AINIC_LIB="/apps/gpuperf/ainic-driver-20251007/lib/"
# export RCCL_HOME_DIR="/shared/apps/ubuntu/rocm-7.0.1/rccl-drop-2025-08"
export RCCL_HOME_DIR=/apps/shared/ankit/rccl
export ANP_HOME_DIR=/apps/shared/ankit/amd-anp
export MPI_HOME_DIR=/apps/shared/ankit/mpi_5/ompi-5.0.8/install
# export NCCL_SOCKET_IFNAME="enp193s0f1np1"
# export GLOO_SOCKET_IFNAME="enp193s0f1np1"
export NCCL_SOCKET_IFNAME="enp81s0f1"
export GLOO_SOCKET_IFNAME="enp81s0f1"
export CLEAN_DOCKER_CONTAINER=1
export USE_ROCM_AITER_ROPE_BACKEND=0

export DOCKER_IMAGE=${DOCKER_IMAGE:="docker.io/rocm/pytorch-training-private:20250929_gfx950_25dot9_rc4"}

export CPUS_PER_TASK=128
export HSA_NO_SCRATCH_RECLAIM=0 
export NVTE_CK_USES_BWD_V3=1

export EXP="examples/torchtitan/configs/llama3.1_405B-FP8-pretrain_4N.yaml"
mkdir -p data
# the real number of nodes to run
export NNODES=4


export HF_TOKEN=${HF_TOKEN:="your_token"}


export PRIMUS_WORKSPACE=output/llama3-405B 
export PRIMUS_USER=john
export PRIMUS_GROUP="date-$(date +%Y%m%d-%H%M%S)"
export PRIMUS_EXP_NAME=llama3.1_405B-FP8-pretrain
mkdir -p $PRIMUS_WORKSPACE


LOG_DIR=./$PRIMUS_WORKSPACE/$PRIMUS_GROUP/$PRIMUS_USER/$CONFIG/
export DUMP_PP_DIR=$LOG_DIR/pp_dump/
mkdir -p $LOG_DIR
LOG_FILE=$LOG_DIR/training.log
echo $LOG_FILE

EXPORT_CONFIG=$LOG_DIR/config.yaml
bash ./examples/run_slurm_pretrain.sh   2>&1 | tee $LOG_FILE
