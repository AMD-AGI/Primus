#!/bin/bash
# r3_qknorm, then per-PR e2e attribution on top of turbo-base (1103b2df): base -> +quant -> +layout.
# turbo-opt (r1_turbo) is +layout plus the dispatcher permute change.
ulimit -c 0
E2E=/home/xiaompen/turbo-opt/agent/workspace/qwen3_fp8_tw_gfx950_20261010/e2e
TB=/home/xiaompen/turbo-base
G="git -c safe.directory=* -C $TB"

build() {
    (cd $TB && GPU_ARCHS=gfx950 MAX_JOBS=64 python setup.py build_ext --inplace) > "/tmp/turbo_base_incr_$1.log" 2>&1
    local rc=$?
    echo "[chain] build $1 exit=$rc $(date +%T)"
    return $rc
}

bash $E2E/e2e_ab.sh r3_qknorm rb_base
$G cherry-pick -n 1194eaa0 && build quant && bash $E2E/e2e_ab.sh b1_quant
$G cherry-pick -n 11b6d998 && build layout && bash $E2E/e2e_ab.sh b2_layout
chown -R 11768:11768 $TB /home/xiaompen/Primus/third_party/Turbo/.git/worktrees/turbo-base
echo "[chain] done $(date +%T)"
