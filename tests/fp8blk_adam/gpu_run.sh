#!/bin/bash
# P3 fp8blk-in-Adam GPU plan (NOT run; GPU paused). Run steps one at a time from the host.
# Container (no GPU use at creation):
#   docker run -d --name yyq_opt3adamq --device /dev/kfd --device /dev/dri --group-add video --ipc host --network host \
#     --security-opt seccomp=unconfined --cap-add SYS_PTRACE --shm-size 32g \
#     -e TRITON_HIP_USE_EXPERT_SCHEDULING=0 -e PRIMUS_GPU_MODEL=MI455X -e HSA_COREDUMP_PATTERN=/dev/null \
#     -e PYTHONPATH=/workspace/Primus-Turbo:/workspace/Primus:/workspace/Primus/third_party/Megatron-LM \
#     -v $HOME/dsv41:/dsv41 -v $HOME/dsv41/work/opt3adamq/Primus:/workspace/Primus \
#     -v $HOME/dsv41/work/opt3adamq/Primus-Turbo:/workspace/Primus-Turbo \
#     -w /workspace/Primus --entrypoint sleep amdprimus/amdprimus:tas-20261109 infinity
set -u
C=yyq_opt3adamq
T=/workspace/Primus/tests/fp8blk_adam/test_fp8blk_adam_gpu.py
EXT=-eTORCH_EXTENSIONS_DIR=/dsv41/work/opt3adamq/torch_ext_gpu
step=${1:-dry}
case $step in
dry)   # 1. dry run: small shapes, bit-identity + bytes (<60 s incl. extension build cached beforehand)
  ~/dsv41/bin/yqgpu -g 0 -- docker exec $EXT $C timeout 60 python $T --dry ;;
check) # 2. full DSv4 shapes: Adam bit-identical to fused_adam_clip, W/W^T/scales == Turbo quant
  ~/dsv41/bin/yqgpu -g 0 -- docker exec $EXT $C timeout 300 python $T ;;
bench) # 3. kernel times from a rocprofv3 kernel trace (baseline chunk Adam + weight quant + W^T vs fused tile kernel)
  ~/dsv41/bin/yqgpu -g 0 -- docker exec $EXT $C timeout 300 rocprofv3 --kernel-trace --stats -d /dsv41/work/opt3adamq/prof/bench -o bench -- python $T --bench --iters 10
  echo "-> per-kernel stats under ~/dsv41/work/opt3adamq/prof/bench (fused_adam_clip_kernel, quant_fp8_blockwise_for_weight_kernel, copy, fused_adam_clip_fp8blk_kernel); append to ~/dsv41/work/gemmx/RUNS.md" ;;
e2e_verify) # 4. e2e 4 iters with cache verification (re-quantizes every cache hit, raises on any byte diff)
  ~/dsv41/bin/yqgpu -g 0 -- docker exec -e NAME=o3aq_verify -e ARM=merged -e PREC=BF16 -e PRIMUS_TURBO_FP8_BLOCK=1 \
    -e PRIMUS_FP8BLK_ADAM_QUANT=1 -e PRIMUS_TURBO_FP8_WCACHE_VERIFY=1 -e TRAIN_ITERS=4 -e RUN_TIMEOUT=330 $C bash /dsv41/work/g90host/run_arm2.sh ;;
e2e_ab) # 5. A/B (2+2 runs), eager; loss must be bit-identical (same weights, same FP8 bytes)
  for r in 1 2; do
    ~/dsv41/bin/yqgpu -g 0 -- docker exec -e NAME=o3aq_off_$r -e ARM=merged -e PREC=BF16 -e PRIMUS_TURBO_FP8_BLOCK=1 -e RUN_TIMEOUT=330 $C bash /dsv41/work/g90host/run_arm2.sh | head -2
    ~/dsv41/bin/yqgpu -g 0 -- docker exec -e NAME=o3aq_on_$r -e ARM=merged -e PREC=BF16 -e PRIMUS_TURBO_FP8_BLOCK=1 -e PRIMUS_FP8BLK_ADAM_QUANT=1 -e RUN_TIMEOUT=330 $C bash /dsv41/work/g90host/run_arm2.sh | head -2
  done ;;
e2e_prof) # 6. profile arm (kernel table: quant_fp8_blockwise_for_weight_kernel must be gone, 64 -> 0 calls)
  ~/dsv41/bin/yqgpu -g 0 -- docker exec -e NAME=o3aq_on_prof -e ARM=merged -e PREC=BF16 -e PRIMUS_TURBO_FP8_BLOCK=1 -e PRIMUS_FP8BLK_ADAM_QUANT=1 \
    -e PROFILE=True -e PRIMUS_PROFILE_LIGHT=1 -e RUN_TIMEOUT=420 $C bash /dsv41/work/g90host/run_arm2.sh | head -2
  docker exec $C bash -c 'python /dsv41/mb/analyze_trace.py "$(ls -t /workspace/Primus/output/amd/*/o3aq_on_prof/tensorboard/*.pt.trace.json* | head -1)" 1 40' ;;
esac
