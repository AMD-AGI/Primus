#!/bin/bash
# After the attribution chain: MegaMoE (mxfp8 / bf16 experts) and sync-free stage 2 on the R3 stack.
ulimit -c 0
E2E=/home/xiaompen/turbo-opt/agent/workspace/qwen3_fp8_tw_gfx950_20261010/e2e
while ! grep -q "^\[chain\] done" /tmp/q3_e2e_chain.log; do sleep 15; done
bash $E2E/e2e_run.sh m1_mega_mxfp8 --use_turbo_mega_moe True --turbo_mega_moe_precision mxfp8
bash $E2E/e2e_run.sh m2_mega_bf16 --use_turbo_mega_moe True --turbo_mega_moe_precision bf16
bash $E2E/e2e_run.sh s2_syncfree2 --turbo_sync_free_moe_stage 2
echo "[queue2] done $(date +%T)"
