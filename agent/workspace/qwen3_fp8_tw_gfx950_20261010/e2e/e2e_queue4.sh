#!/bin/bash
# After queue3: DeepEP CU sweep on the R3 stack.
ulimit -c 0
E2E=/home/xiaompen/turbo-opt/agent/workspace/qwen3_fp8_tw_gfx950_20261010/e2e
while ! grep -q "^\[queue3\] done" /tmp/q3_e2e_queue3.log; do sleep 15; done
bash $E2E/e2e_run.sh c192 --turbo_deepep_num_cu 192
bash $E2E/e2e_run.sh c256 --turbo_deepep_num_cu 256
echo "[queue4] done $(date +%T)"
