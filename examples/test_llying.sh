#!/bin/bash
export EXP=examples/maxtext/config/MI300X/llama3_8B-pretrain.yaml
export BACKEND=MaxText
NNODES=1 bash examples/run_slurm_pretrain.sh
