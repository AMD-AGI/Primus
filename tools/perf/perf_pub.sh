#!/bin/bash

gpu=mi325x # set to mi300x, mi325x, or mi355x; names the result directories only
version=v26.8
: "${HF_TOKEN:?export HF_TOKEN=hf_... before running this script}"
export CONFIG_FILE=tools/perf/perf_configs.yaml
export DOCKER_IMAGE=rocm/primus:$version   # required
export BACKEND=megatron #,torchtitan                        # required
export RESULT_DIR=~/primus-bench/${gpu}-primus-${version}-perf

# GPU is auto-detected from amd-smi / rocm-smi; set it to override.
# GPU=MI325X

# RCCL still needs a real NIC on a single node.
# NCCL_SOCKET_IFNAME and GLOO_SOCKET_IFNAME are auto-detected from get_ip_interface.sh,
# which maps the first IPv4 from hostname -I to an interface.
# On some nodes that address belongs to docker0.
# So, the following may need to be set manually.

export NCCL_SOCKET_IFNAME=eno0                            # e.g. eno0 / ibp...
export GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME

TRAIN_STEPS=20 NUM_REPS=3 bash tools/perf/run_batch.sh                              # single node

# run_batch.sh extract the results automatically
# python3 tools/perf/extract_results.py "$RESULT_DIR"       # logs -> CSV


export DOCKER_IMAGE=rocm/jax-training:maxtext-$version   # required
export BACKEND=maxtext                        # required
export RESULT_DIR=~/primus-bench/${gpu}-jax-${version}-perf

TRAIN_STEPS=20 NUM_REPS=3 bash tools/perf/run_batch.sh                              # single node

# run_batch.sh extract the results automatically
# python3 tools/perf/extract_results.py "$RESULT_DIR"       # logs -> CSV
