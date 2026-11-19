#!/bin/bash
# Sequential 20-iteration Qwen3-30B-A3B FP8 tensorwise runs (even routing, MBS 8, GBS 512).
# Usage: e2e_ab.sh NAME [NAME ...]   (names below); logs: /tmp/q3_e2e_<name>.log
ulimit -c 0
E2E=/home/xiaompen/turbo-opt/agent/workspace/qwen3_fp8_tw_gfx950_20261010/e2e
CFG=examples/megatron/configs/MI355X/qwen3_30B_A3B-FP8-pretrain.yaml
BASE_ARGS=(--train_iters 20 --use_turbo_fused_act_with_probs True)

run() {
    local name=$1 repo=$2
    shift 2
    echo "[e2e] $name start $(date +%T)"
    (cd "$repo" && bash runner/primus-cli direct "$@" -- train pretrain --config "$CFG" "${BASE_ARGS[@]}" "${EXTRA[@]}") \
        > "/tmp/q3_e2e_${name}.log" 2>&1
    echo "[e2e] $name exit=$? $(date +%T)"
}

for name in "$@"; do
    EXTRA=()
    case $name in
        r0_base) run "$name" /home/xiaompen/Primus-qwen3 ;;
        rb_base | b1_quant | b2_layout) run "$name" /home/xiaompen/Primus-qwen3 --env "$E2E/env_turbo_base.sh" ;;
        r1_turbo) run "$name" /home/xiaompen/Primus-qwen3 --env "$E2E/env_turbo_opt.sh" ;;
        r2_cu160)
            EXTRA=(--turbo_deepep_num_cu 160)
            run "$name" /home/xiaompen/Primus-qwen3 --env "$E2E/env_turbo_opt.sh"
            ;;
        r3_qknorm)
            EXTRA=(--turbo_deepep_num_cu 160)
            run "$name" /home/xiaompen/Primus-qwen3-qknorm --env "$E2E/env_turbo_opt_qknorm.sh"
            ;;
        *) echo "unknown run $name" ;;
    esac
done
echo "[e2e] done"
