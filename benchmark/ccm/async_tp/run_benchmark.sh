#!/bin/bash
export NCCL_SOCKET_IFNAME=lo
# Setting
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-1234}
PRIMUS_ROOT_PATH="$(pwd)/../../.."
MEGATRON_PATH="$PRIMUS_ROOT_PATH/third_party/Megatron-LM"
export PYTHONPATH=$PRIMUS_ROOT_PATH:$MEGATRON_PATH:$PYTHONPATH

#########################################################################
# torchrun --master_addr "$MASTER_ADDR"   \
#         --master_port "$MASTER_PORT"    \
#         --nnodes=1                      \
#         --node_rank=0                   \
#         --nproc_per_node=4              \
# 	./benchmark_gemm_rs_overlap.py --model "Llama3.1_70B" --model-config-path ./model_configs.json --report-dir-path ./output --backend "megatron" --fp8

# torchrun --master_addr "$MASTER_ADDR"   \
#         --master_port "$MASTER_PORT"    \
#         --nnodes=1                      \
#         --node_rank=0                   \
#         --nproc_per_node=4              \
# 	./benchmark_ag_gemm_overlap.py --model "Llama3.1_70B" --model-config-path ./model_configs.json --report-dir-path ./output --backend "torchtitan" --fp8

torchrun --master_addr "$MASTER_ADDR"   \
        --master_port "$MASTER_PORT"    \
        --nnodes=1                      \
        --node_rank=0                   \
        --nproc_per_node=4              \
	./benchmark_ag_gemm_overlap.py --model "Llama3.1_70B" --model-config-path ./model_configs.json --report-dir-path ./output --backend "megatron" --fp8
