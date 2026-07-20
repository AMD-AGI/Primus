#!/bin/bash
# Memory characterization sweep for 300M GDN pure on the current docker image.
# Runs a few iters at several micro_batch_sizes (grad-accum=1) and records the
# peak per-GPU memory, to separate fixed overhead from per-sample memory.
set -u
cd /home/vanbhati@amd.com/Primus
RESULTS=/tmp/memsweep_results.txt
: > "$RESULTS"

for MBS in 1 8 24 48 64; do
  GBS=$((MBS * 8))
  LOG="/tmp/memsweep_mbs${MBS}.log"
  echo "=== Running mbs=$MBS gbs=$GBS ==="
  env MASTER_ADDR=127.0.0.1 MASTER_PORT=1234 \
      GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      SWEEP_MBS=$MBS SWEEP_GBS=$GBS \
      EXP=examples/megatron/configs/MI300X/zebra_llama_300M_gdn_pure-memsweep.yaml \
      bash examples/run_pretrain.sh > "$LOG" 2>&1

  CLEAN=$(sed -E "s/\x1b\[[0-9;]*m//g" "$LOG")
  # Per-rank ROCm memory used (max across the printed iters), parsed from the
  # throughput patch line: "total_r_used: [a, b, ...]"
  RUSED=$(echo "$CLEAN" | grep -aoE "total_r_used: \[[^]]*\]" | tail -1)
  ITERMEM=$(echo "$CLEAN" | grep -aoE "rocm max mem usage/usage_ratio: [0-9.]+GB" | tail -1 | grep -oE "[0-9.]+GB")
  OOM=$(echo "$CLEAN" | grep -aciE "OutOfMemory")
  if [ "$OOM" -gt 0 ]; then
    echo "mbs=$MBS gbs=$GBS  RESULT=OOM" | tee -a "$RESULTS"
  else
    echo "mbs=$MBS gbs=$GBS  itermax=${ITERMEM:-NA}  ${RUSED:-NA}" | tee -a "$RESULTS"
  fi
done

echo "=== SWEEP DONE ==="
cat "$RESULTS"
