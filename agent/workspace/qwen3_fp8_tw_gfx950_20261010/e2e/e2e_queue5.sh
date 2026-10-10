#!/bin/bash
# After queue4: fused grouped MLP (fc1 + GLU + fc2 in Turbo), then also without explicit padding.
ulimit -c 0
E2E=/home/xiaompen/turbo-opt/agent/workspace/qwen3_fp8_tw_gfx950_20261010/e2e
while ! grep -q "^\[queue4\] done" /tmp/q3_e2e_queue4.log; do sleep 15; done
bash $E2E/e2e_run.sh f1_fused_mlp --turbo_fused_grouped_gemm True
bash $E2E/e2e_run.sh f2_fused_mlp_nopad --turbo_fused_grouped_gemm True --turbo_grouped_gemm_without_padding True
echo "[queue5] done $(date +%T)"
