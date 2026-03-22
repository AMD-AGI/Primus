#!/usr/bin/env bash
# =============================================================================
# DSv2-Lite 4-node batch node validation
#
# Groups all idle nodes into 4-node batches, submits all jobs in parallel,
# and reports which nodes can produce training iteration metrics.
#
# A group is HEALTHY if its log shows >= 3 iterations with
# "throughput per GPU" output (same as log.benchmark reference).
#
# Usage:
#   cd /shared_aig/xiaoming/Primus-moe
#   bash benchmark/moe/validate_nodes_dsv2lite.sh            # run all
#   bash benchmark/moe/validate_nodes_dsv2lite.sh --dry-run  # print groups only
# =============================================================================
set -euo pipefail

WORKSPACE="/shared_aig/xiaoming/Primus-moe"
cd "$WORKSPACE"

MODE="${1:-}"

# ── Job config ────────────────────────────────────────────────────────────────
IMAGE="docker.io/tasimage/primus:pr-609-ainic"
PARTITION="amd-aig"
SLURM_TIME_VAL="0:25:00"    # 25 min timeout – enough for 5 iters
TRAIN_ITERS=5
MBS=4
# With PP=1, EP=8, 4 nodes (32 GPUs): DP = 32/(1×8×1) = 4
# GBS = MBS × DP × grad_acc = 4 × 4 × 2 = 32
GBS=32
PRIMUS_EP_VAL=8   # EP spans all 4 nodes → exercises inter-node AINIC A2A
PRIMUS_PP_VAL=1
GROUP_SIZE=4
MIN_PASS_ITERS=3  # iterations needed to call a group HEALTHY

# ── NCCL / AINIC env (same as production scripts) ─────────────────────────────
export USING_AINIC=1
export NCCL_IB_HCA="ionic_0:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1,ionic_9:1"
export GLOO_SOCKET_IFNAME=ens9np0
export NCCL_SOCKET_IFNAME=ens9np0
export TURBO_DEEPEEP=True
export LEGACY_GG=True
export TURBO_GROUPED_MLP=False
export TURBO_ATTENTION=False
export APPLY_ROPE_FUSION=True
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1
export GPU_MAX_HW_QUEUES=4
export PRIMUS_TURBO_DEEPEP_TIMEOUT=300
export ENABLE_NUMA_BINDING=1
export HSA_KERNARG_POOL_SIZE=12582912
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SLURM_PARTITION="$PARTITION"
export SLURM_TIME="$SLURM_TIME_VAL"

# ── Idle nodes from sinfo (captured 2026-03-21) ───────────────────────────────
# uswslocpm2m-106-[2165,2171,2177,2180,2182-2184,2186,2189-2190,2194-2198,2201,
#   2205,2207,2210,2212,2214-2215,2218,2222,2224,2226-2230,2233-2234,2236,
#   2238-2241,2243-2247,2250,2252-2253,2255-2256,2260,2264,2266-2268,2270,
#   2272,2274,2279-2280,2283-2284,2300-2304,2306-2307,2310]  (67 total)
ALL_NODES=(
  2165 2171 2177 2180
  2182 2183 2184 2186
  2189 2190 2194 2195
  2196 2197 2198 2201
  2205 2207 2210 2212
  2214 2215 2218 2222
  2224 2226 2227 2228
  2229 2230 2233 2234
  2236 2238 2239 2240
  2241 2243 2244 2245
  2246 2247 2250 2252
  2253 2255 2256 2260
  2264 2266 2267 2268
  2270 2272 2274 2279
  2280 2283 2284 2300
  2301 2302 2303 2304
  # leftover (3 nodes, skipped): 2306 2307 2310
)

N_NODES=${#ALL_NODES[@]}   # 64  (16 groups × 4)
N_GROUPS=$((N_NODES / GROUP_SIZE))  # 16

# ── Print header ─────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════╗"
echo "║    DSv2-Lite 4-node Batch Node Validation            ║"
echo "╚══════════════════════════════════════════════════════╝"
printf "%-18s %s\n" "Nodes to test:"  "$N_NODES (16 groups × 4; nodes 2306,2307,2310 leftover, skipped)"
printf "%-18s %s\n" "TRAIN_ITERS:"    "$TRAIN_ITERS"
printf "%-18s %s\n" "MBS/GBS:"        "$MBS / $GBS"
printf "%-18s %s\n" "EP / PP:"        "$PRIMUS_EP_VAL / $PRIMUS_PP_VAL"
printf "%-18s %s\n" "Pass threshold:" ">= $MIN_PASS_ITERS iterations with throughput metric"
echo ""

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="$WORKSPACE/output/amd/node-validate/dsv2lite_$RUN_TAG"
mkdir -p "$REPORT_DIR"
printf "%-18s %s\n" "Output dir:" "$REPORT_DIR"
echo ""

# ── Build groups ──────────────────────────────────────────────────────────────
PIDS=()
GROUP_LOGS=()
GROUP_NODES_LIST=()

echo "Groups:"
for i in $(seq 0 $((N_GROUPS - 1))); do
  start=$((i * GROUP_SIZE))
  N0="${ALL_NODES[$start]}"
  N1="${ALL_NODES[$((start+1))]}"
  N2="${ALL_NODES[$((start+2))]}"
  N3="${ALL_NODES[$((start+3))]}"

  NODELIST="uswslocpm2m-106-[$N0,$N1,$N2,$N3]"
  LABEL="$(printf 'g%02d' $i)"
  EXP_NAME="val-${LABEL}-n${N0}_${N1}_${N2}_${N3}"
  LOGFILE="$REPORT_DIR/${EXP_NAME}.log"

  GROUP_LOGS+=("$LOGFILE")
  GROUP_NODES_LIST+=("$N0 $N1 $N2 $N3")

  printf "  Group %02d : [%s,%s,%s,%s]  → %s\n" "$i" "$N0" "$N1" "$N2" "$N3" "$LOGFILE"

  if [[ "$MODE" == "--dry-run" ]]; then
    continue
  fi

  # ── Submit one group in background ─────────────────────────────────────────
  (
    export NNODES=4
    export SLURM_NODELIST="$NODELIST"
    export PRIMUS_TEAM="amd"
    export PRIMUS_USER="node-validate"
    export PRIMUS_EXP_NAME="$EXP_NAME"

    mkdir -p "$REPORT_DIR/$EXP_NAME"

    # 1500s = 25 min hard timeout
    timeout 1500 \
    ./primus-cli slurm -N 4 --nodelist "$NODELIST" \
       -- --image "$IMAGE" --clean \
      -- --numa -- train pretrain \
         --config "examples/megatron/configs/MI355X/deepseek_v2_lite-BF16-pretrain.yaml" \
         --train_iters          $TRAIN_ITERS \
         --micro_batch_size     $MBS \
         --global_batch_size    $GBS \
         --pipeline_model_parallel_size $PRIMUS_PP_VAL \
         --expert_model_parallel_size   $PRIMUS_EP_VAL \
         --use_turbo_deepep     True \
         --use_turbo_grouped_mlp False \
         --use_turbo_attention  False \
         --moe_use_legacy_grouped_gemm True \
         --apply_rope_fusion    True \
         --enable_experimental  True \
         --turbo_deepep_num_cu  64 \
         --manual_gc            True \
         --manual_gc_interval   1 \
         --disable_last_saving  True \
         --mock_data            True \
         --disable_wandb        True \
         --disable_tensorboard  True \
    2>&1 || true
  ) > "$LOGFILE" 2>&1 &

  PIDS+=($!)
done

if [[ "$MODE" == "--dry-run" ]]; then
  echo ""
  echo "Dry-run complete. $N_GROUPS groups listed. No jobs submitted."
  exit 0
fi

echo ""
echo "All $N_GROUPS jobs submitted in background."
echo "Waiting (up to 25 min each)... Ctrl+C to abort."
echo ""

# ── Progress monitor ──────────────────────────────────────────────────────────
(
  while true; do
    sleep 60
    DONE=0
    for pid in "${PIDS[@]}"; do
      kill -0 "$pid" 2>/dev/null || ((DONE++)) || true
    done
    echo "[$(date +%H:%M:%S)] $DONE/${#PIDS[@]} groups finished"
    for i in $(seq 0 $((N_GROUPS - 1))); do
      CNT=$(grep -c "throughput per GPU" "${GROUP_LOGS[$i]}" 2>/dev/null || echo 0)
      [[ "$CNT" -gt 0 ]] && \
        printf "  g%02d [%s]: %d iter(s)\n" "$i" "${GROUP_NODES_LIST[$i]// /,}" "$CNT"
    done
    [[ "$DONE" -eq "${#PIDS[@]}" ]] && break
  done
) &
MONITOR_PID=$!

trap 'echo ""; echo "Interrupted – killing jobs..."; kill "${PIDS[@]}" 2>/dev/null || true; kill $MONITOR_PID 2>/dev/null || true' INT

for pid in "${PIDS[@]}"; do
  wait "$pid" 2>/dev/null || true
done
kill "$MONITOR_PID" 2>/dev/null || true
trap - INT

# ── Analyze results ──────────────────────────────────────────────────────────
echo ""
echo "=== All groups finished. Analyzing logs... ==="
echo ""

HEALTHY_NODES=()
BROKEN_NODES=()
declare -a RESULTS_TABLE

for i in $(seq 0 $((N_GROUPS - 1))); do
  LOGFILE="${GROUP_LOGS[$i]}"
  NODE_ARR=(${GROUP_NODES_LIST[$i]})

  ITER_COUNT=$(grep -c "throughput per GPU" "$LOGFILE" 2>/dev/null || echo 0)

  if [[ "$ITER_COUNT" -ge "$MIN_PASS_ITERS" ]]; then
    LAST_LINE=$(grep "throughput per GPU" "$LOGFILE" 2>/dev/null | tail -1)
    TFLOP=$(echo "$LAST_LINE" | grep -oP "throughput per GPU \(TFLOP/s/GPU\): \K[0-9.]+" || echo "?")
    TOKENS=$(echo "$LAST_LINE" | grep -oP "tokens per GPU \(tokens/s/GPU\): \K[0-9.]+" || echo "?")
    LABEL="✅ PASS"
    DETAIL="${ITER_COUNT} iters | ${TFLOP} TFLOP/s | ${TOKENS} tok/s"
    HEALTHY_NODES+=("${NODE_ARR[@]}")
  else
    LABEL="❌ FAIL"
    ERR=$(grep -m1 -E \
      "(exitcode [^0]|error:|Error:|OOM|NCCL.*error|DeepEP error|Timeout|timeout after|Broken pipe|Segmentation)" \
      "$LOGFILE" 2>/dev/null | \
      grep -v "Warning\|warn\|injection\|mem_dump" | \
      sed 's/\x1b\[[0-9;]*m//g' | \
      cut -c1-90 || echo "no output / timed out")
    DETAIL="${ITER_COUNT} iter(s) | ${ERR}"
    BROKEN_NODES+=("${NODE_ARR[@]}")
  fi

  LINE="$(printf '%s Group %02d [%s,%s,%s,%s]: %s' \
    "$LABEL" "$i" "${NODE_ARR[0]}" "${NODE_ARR[1]}" "${NODE_ARR[2]}" "${NODE_ARR[3]}" "$DETAIL")"
  RESULTS_TABLE[$i]="$LINE"
  echo "$LINE"
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║                     SUMMARY                         ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "Healthy nodes (${#HEALTHY_NODES[@]}): ${HEALTHY_NODES[*]:-none}"
echo "Broken  nodes (${#BROKEN_NODES[@]}):  ${BROKEN_NODES[*]:-none}"

HEALTHY_COMPRESSED=""
if [[ ${#HEALTHY_NODES[@]} -gt 0 ]]; then
  HEALTHY_HOSTS=$(printf 'uswslocpm2m-106-%s,' "${HEALTHY_NODES[@]}" | sed 's/,$//')
  HEALTHY_COMPRESSED=$(scontrol show hostlist "$HEALTHY_HOSTS" 2>/dev/null || echo "$HEALTHY_HOSTS")
  echo ""
  echo "Healthy NODELIST (for use in SLURM scripts):"
  echo "  $HEALTHY_COMPRESSED"
fi

# ── Save report ───────────────────────────────────────────────────────────────
{
  echo "# DSv2-Lite Node Validation Report"
  echo "- Date     : $(date)"
  echo "- Image    : $IMAGE"
  echo "- Config   : deepseek_v2_lite-BF16-pretrain.yaml"
  echo "- Per-group: 4 nodes | PP=$PRIMUS_PP_VAL EP=$PRIMUS_EP_VAL MBS=$MBS GBS=$GBS iters=$TRAIN_ITERS"
  echo "- Groups   : $N_GROUPS tested"
  echo ""
  echo "## Results"
  for line in "${RESULTS_TABLE[@]}"; do echo "$line"; done
  echo ""
  echo "## Healthy nodes (${#HEALTHY_NODES[@]})"
  printf '%s\n' "${HEALTHY_NODES[@]:-}" | sort -n | sed 's/^/  uswslocpm2m-106-/'
  echo ""
  echo "## Broken nodes (${#BROKEN_NODES[@]})"
  printf '%s\n' "${BROKEN_NODES[@]:-}" | sort -n | sed 's/^/  uswslocpm2m-106-/'
  echo ""
  echo "## Healthy NODELIST"
  echo "  ${HEALTHY_COMPRESSED:-none}"
} | tee "$REPORT_DIR/summary.txt"

echo ""
echo "Report saved to : $REPORT_DIR/summary.txt"
echo "Per-job logs    : $REPORT_DIR/val-g*.log"
