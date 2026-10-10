#!/bin/bash
# After queue2: one profiled step (rank 0, iteration 10) of the R3 stack.
ulimit -c 0
E2E=/home/xiaompen/turbo-opt/agent/workspace/qwen3_fp8_tw_gfx950_20261010/e2e
while ! grep -q "^\[queue2\] done" /tmp/q3_e2e_queue2.log; do sleep 15; done
bash $E2E/e2e_run.sh p3_profile --profile True --use_pytorch_profiler True \
    --profile_step_start 10 --profile_step_end 11 --train_iters 12
echo "[queue3] done $(date +%T)"
