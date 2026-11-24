#!/bin/bash
# One 20-iteration Qwen3-30B-A3B FP8 run on top of the R3 stack (turbo-opt, fused qk-norm, 160 DeepEP CUs).
# Usage: e2e_run.sh NAME [extra primus args ...]   log: /tmp/q3_e2e_<NAME>.log
ulimit -c 0
E2E=/home/xiaompen/turbo-opt/agent/workspace/qwen3_fp8_tw_gfx950_20261010/e2e
CFG=examples/megatron/configs/MI355X/qwen3_30B_A3B-FP8-pretrain.yaml
name=$1
shift
echo "[e2e] $name start $(date +%T)"
(cd /home/xiaompen/Primus-qwen3-qknorm && bash runner/primus-cli direct --env "$E2E/env_turbo_opt_qknorm.sh" -- \
    train pretrain --config "$CFG" --train_iters 20 --use_turbo_fused_act_with_probs True \
    --turbo_deepep_num_cu 160 "$@") > "/tmp/q3_e2e_${name}.log" 2>&1
echo "[e2e] $name exit=$? $(date +%T)"
