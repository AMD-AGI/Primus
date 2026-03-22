#!/usr/bin/env bash
# Find healthy SLURM nodes for MoE overlap validation.
# Runs health checks in parallel batches and outputs two 8-node nodelists.
# Usage: bash find_healthy_nodes.sh [partition]
set -euo pipefail

PARTITION="${1:-amd-aig}"
WORKSPACE_DIR="/shared_aig/xiaoming/Primus-moe"
MAX_NODES_TO_CHECK=60
BATCH_SIZE=15
NEED_PASS=16
# Exclude known broken/reserved nodes
BAD_REGEX='uswslocpm2m-106-(2155|2164|2191)$'

REPORT_DIR="${WORKSPACE_DIR}/output/skills/slurm-health-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPORT_DIR"

health_ssh() {
  local node="$1"
  local out="$REPORT_DIR/${node}.line"
  result=$(ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$node" bash -s 2>/dev/null <<'REMOTE'
  docker info > /dev/null 2>&1; DOCKER_OK=$?
  WORKSPACE_DIR="/shared_aig/xiaoming/Primus-moe"
  [ -d "$WORKSPACE_DIR" ] && [ -r "$WORKSPACE_DIR" ] && WORKSPACE_OK=0 || WORKSPACE_OK=1
  QOS_OUTPUT=$(sudo nicctl show qos 2>&1); QOS_RC=$?
  QOS_HASH=""; QOS_REQUIRED_OK=1
  if [ $QOS_RC -eq 0 ]; then
    QOS_HASH=$(echo "$QOS_OUTPUT" | md5sum | awk "{print \$1}")
    echo "$QOS_OUTPUT" | grep -q "Classification type[[:space:]]*:[[:space:]]*DSCP" && \
    echo "$QOS_OUTPUT" | grep -q "DSCP[[:space:]]*:[[:space:]]*10[[:space:]]*==>[[:space:]]*priority[[:space:]]*:[[:space:]]*0" && \
    echo "$QOS_OUTPUT" | grep -q "PFC no-drop priorities[[:space:]]*:[[:space:]]*0" && QOS_REQUIRED_OK=0 || true
  fi
  DCQCN_OUTPUT=$(sudo nicctl show dcqcn 2>&1); DCQCN_RC=$?
  DCQCN_HASH=""
  [ $DCQCN_RC -eq 0 ] && DCQCN_HASH=$(echo "$DCQCN_OUTPUT" | md5sum | awk "{print \$1}")
  ERRORS=""
  [ $DOCKER_OK -ne 0 ] && ERRORS="${ERRORS}docker_fail; "
  [ $WORKSPACE_OK -ne 0 ] && ERRORS="${ERRORS}workspace_fail; "
  [ $QOS_RC -ne 0 ] && ERRORS="${ERRORS}nicctl_qos_fail; " || \
    [ $QOS_REQUIRED_OK -ne 0 ] && ERRORS="${ERRORS}QoS_misconfigured; " || true
  [ $DCQCN_RC -ne 0 ] && ERRORS="${ERRORS}nicctl_dcqcn_fail; "
  [ -z "$ERRORS" ] && echo "PASS||${QOS_HASH}|${DCQCN_HASH}" || echo "FAIL|${ERRORS}|${QOS_HASH}|${DCQCN_HASH}"
REMOTE
  ) || result="FAIL|ssh_failed|||"
  echo "${result:-FAIL|empty_response|||}" > "$out"
  echo "  $node: $(cat "$out")"
}

# Step 1: Get idle nodes
echo "=== Getting idle nodes from partition: $PARTITION ==="
mapfile -t ALL_NODES < <(
  sinfo -h -o "%P %T %N" | awk -v p="$PARTITION" '$2 == "idle" && $1 == p {print $3}' | \
  while read -r nl; do scontrol show hostnames "$nl"; done
)

# Filter bad nodes and deduplicate
declare -A SEEN=()
CANDIDATES=()
for n in "${ALL_NODES[@]}"; do
  [[ -z "$n" ]] && continue
  [[ "$n" =~ $BAD_REGEX ]] && continue
  [[ -n "${SEEN[$n]:-}" ]] && continue
  SEEN[$n]=1
  CANDIDATES+=("$n")
  (( ${#CANDIDATES[@]} >= MAX_NODES_TO_CHECK )) && break
done

echo "Total candidates to check: ${#CANDIDATES[@]} (max $MAX_NODES_TO_CHECK)"

# Step 2: Health check in parallel batches
PASS_LIST=()
FAIL_LIST=()
idx=0
total=${#CANDIDATES[@]}

while (( ${#PASS_LIST[@]} < NEED_PASS && idx < total )); do
  pids=()
  batch_start=$idx
  while (( (idx - batch_start) < BATCH_SIZE && idx < total )); do
    n="${CANDIDATES[idx]}"
    idx=$((idx + 1))
    health_ssh "$n" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done

  # Count results so far
  PASS_LIST=(); FAIL_LIST=()
  for i in $(seq 0 $((idx-1))); do
    n="${CANDIDATES[$i]}"
    f="$REPORT_DIR/${n}.line"
    [[ -f "$f" ]] || continue
    line="$(cat "$f")"
    if [[ "$line" == PASS* ]]; then
      PASS_LIST+=("$n")
    else
      reason="${line#FAIL|}"
      reason="${reason%%|*}"
      FAIL_LIST+=("$n|$reason")
    fi
  done
  echo "  Progress: checked $idx nodes → ${#PASS_LIST[@]} PASS, ${#FAIL_LIST[@]} FAIL"
done

echo ""
echo "=== RESULTS ==="
echo "Healthy nodes (${#PASS_LIST[@]}):"
printf '  %s\n' "${PASS_LIST[@]}"
echo ""
echo "Failing nodes (${#FAIL_LIST[@]}):"
printf '  %s\n' "${FAIL_LIST[@]}"

# Step 3: Cross-node NIC consistency check
echo ""
echo "=== NIC Configuration Consistency ==="
declare -A QOS_GROUPS=() DCQCN_GROUPS=()
for n in "${PASS_LIST[@]}"; do
  f="$REPORT_DIR/${n}.line"
  [[ -f "$f" ]] || continue
  IFS='|' read -r _st _err qh dh < "$f"
  QOS_GROUPS["$qh"]+=" $n"
  DCQCN_GROUPS["$dh"]+=" $n"
done

if (( ${#QOS_GROUPS[@]} > 1 )); then
  echo "WARNING: QoS configuration is INCONSISTENT across healthy nodes!"
  for h in "${!QOS_GROUPS[@]}"; do echo "  hash=$h:${QOS_GROUPS[$h]}"; done
else
  echo "QoS: consistent (${#QOS_GROUPS[@]} unique config across all PASS nodes)"
fi
if (( ${#DCQCN_GROUPS[@]} > 1 )); then
  echo "WARNING: DCQCN configuration is INCONSISTENT across healthy nodes!"
  for h in "${!DCQCN_GROUPS[@]}"; do echo "  hash=$h:${DCQCN_GROUPS[$h]}"; done
else
  echo "DCQCN: consistent (${#DCQCN_GROUPS[@]} unique config across all PASS nodes)"
fi

# Step 4: Output two 8-node nodelists
echo ""
echo "=== NODE LISTS FOR TRAINING ==="
if (( ${#PASS_LIST[@]} >= 8 )); then
  A=("${PASS_LIST[@]:0:8}")
  HOSTS_A=$(printf '%s,' "${A[@]}" | sed 's/,$//')
  NODELIST_A=$(scontrol show hostlist "$HOSTS_A" 2>/dev/null || echo "${A[*]}")
  echo "NODELIST_A (baseline):  $NODELIST_A"
else
  echo "NODELIST_A: INSUFFICIENT (need 8, have ${#PASS_LIST[@]})"
fi

if (( ${#PASS_LIST[@]} >= 16 )); then
  B=("${PASS_LIST[@]:8:8}")
  HOSTS_B=$(printf '%s,' "${B[@]}" | sed 's/,$//')
  NODELIST_B=$(scontrol show hostlist "$HOSTS_B" 2>/dev/null || echo "${B[*]}")
  echo "NODELIST_B (overlap):   $NODELIST_B"
else
  echo "NODELIST_B: INSUFFICIENT (need 16 total PASS, have ${#PASS_LIST[@]})"
fi

# Save report
{
  echo "# SLURM Idle Node Health Check Report"
  echo "- Date: $(date)"
  echo "- Partition: $PARTITION"
  echo "- Total candidates checked: $idx"
  echo "- Healthy: ${#PASS_LIST[@]}"
  echo "- Problematic: ${#FAIL_LIST[@]}"
  echo ""
  echo "## Healthy Nodes"
  echo "| Node | Status | Issue |"
  echo "|------|--------|-------|"
  for n in "${PASS_LIST[@]}"; do echo "| $n | PASS | - |"; done
  echo ""
  echo "## Problematic Nodes"
  echo "| Node | Status | Issue |"
  echo "|------|--------|-------|"
  for entry in "${FAIL_LIST[@]}"; do
    n="${entry%%|*}"; reason="${entry#*|}"
    echo "| $n | FAIL | $reason |"
  done
} > "$REPORT_DIR/report.md"

echo ""
echo "Full report saved to: $REPORT_DIR/report.md"
