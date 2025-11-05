#!/bin/bash
export EXP=examples/maxtext/config/MI300X/llama2_7B-pretrain.yaml
export BACKEND=MaxText
NNODES=2 bash examples/run_slurm_pretrain.sh
