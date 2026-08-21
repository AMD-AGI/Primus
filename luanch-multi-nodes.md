# Launch four-node FLUX training across allocations

Use this fallback only when the four nodes do not belong to one Slurm/Spur
allocation. A single `srun` cannot span multiple allocation job IDs, so launch
one node-local process through each allocation and point all four processes at
the same torchrun rendezvous.

Prefer one four-node allocation and
`examples/mlperf/flux1/run_with_docker_slurm.sh` whenever possible.

## Prerequisites

- All four nodes can read the same Primus checkout, datasets, and output path.
- `zirui3/primus-v26.3-flux:v0.4` is available on every node.
- Each selected node has eight free GPUs.
- The rendezvous port is unused and reachable between the nodes.
- Every `spur run` uses the job ID that owns its selected node.

The example below uses three nodes from one allocation and one node from
another. Replace job IDs, nodes, paths, and the master address with current
values.

## Create the per-node launcher

Run from a CRS login node:

```bash
REPO=/shared_nfs/zirui/code/primus-compile
ROOT=/shared_nfs/zirui/runs/flux-fp8-4n-$(date -u +%Y%m%dT%H%M%SZ)
MASTER_NODE=crsuse2-m2m-017
MASTER_ADDR=$(getent ahostsv4 "$MASTER_NODE" | awk 'NR == 1 {print $1}')
MASTER_PORT=29971

mkdir -p "$ROOT"
cat >"$ROOT/launch_node.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

rank=$1
node=$(hostname)
REPO=${REPO:?}
ROOT=${ROOT:?}

cd "$REPO"
exec env \
  DATA_ROOT=/shared_nfs/zirui/data \
  OUTPUT_ROOT="$ROOT" \
  DOCKER_IMAGE=zirui3/primus-v26.3-flux:v0.4 \
  NNODES=4 NODE_RANK="$rank" MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" \
  GPUS_PER_NODE=8 DP_REPLICATE=4 \
  LOCAL_BATCH_SIZE=32 GRADIENT_ACCUMULATION_STEPS=1 \
  MAX_STEPS=30000 LR=0.00025 WARMUP_STEPS=800 SEED=10007 \
  GRADIENT_CHECKPOINTING_RATIO=0 \
  FLUX_FLOAT8_RECIPE=tensorwise FLUX_FP8_GEMM_BACKEND=selective_flydsl \
  TORCHINDUCTOR_BENCHMARK_FUSION=1 PRIMUS_FLUX_AITER_ATOMIC_FP32=0 \
  PRIMUS_FLUX_REUSE_FP8_INPUT=1 \
  ATTENTION_BACKEND=flash_attn_aiter \
  COMPILE_TRANSFORMER_BLOCKS=true COMPILE_STRATEGY=per_block \
  COMPILE_FULLGRAPH=true COMPILE_DYNAMIC=false \
  FSDP2_RESHARD_AFTER_FORWARD=false FSDP2_REDUCE_DTYPE=bf16 \
  FLUX_PERFORMANCE_MODE=nemo_mlperf \
  TARGET_ACCURACY=0.586 VAL_CHECK_INTERVAL=262144 \
  CONTAINER_NAME="flux-fp8-4n-${rank}" \
  bash examples/mlperf/flux1/run_with_docker.sh
EOF
chmod +x "$ROOT/launch_node.sh"
export REPO ROOT MASTER_ADDR MASTER_PORT
```

`DATA_ROOT` must contain `cc12m_preprocessed`, `coco_preprocessed`, and
`empty_encodings`.

## Launch all ranks

The fields are `JOB_ID NODE RANK`. Start the four commands together so the
rendezvous does not time out:

```bash
launches=(
  "33795 crsuse2-m2m-017 0"
  "33795 crsuse2-m2m-102 1"
  "33795 crsuse2-m2m-255 2"
  "30487 crsuse2-m2m-042 3"
)

: >"$ROOT/launches.tsv"
for entry in "${launches[@]}"; do
  read -r job_id node rank <<<"$entry"
  nohup spur run --jobid="$job_id" --overlap -N1 -n1 --nodelist="$node" \
    env REPO="$REPO" ROOT="$ROOT" MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" \
    "$ROOT/launch_node.sh" "$rank" \
    >"$ROOT/${node}.launch.log" 2>&1 </dev/null &
  printf '%s\t%s\t%s\t%s\n' "$node" "$job_id" "$rank" "$!" | tee -a "$ROOT/launches.tsv"
done
```

## Monitor and clean up

```bash
while IFS=$'\t' read -r node job_id rank pid; do
  printf '%s rank=%s job=%s ' "$node" "$rank" "$job_id"
  ps -p "$pid" -o stat=,etime= || echo EXITED
done <"$ROOT/launches.tsv"

tail -f "$ROOT"/*.launch.log
```

To stop this run, remove only its named containers through their owning
allocations. Do not cancel shared allocations unless every user of those
allocations agrees.
