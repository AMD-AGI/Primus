# FLUX.1 Multi-Node Development Launch Guide (Internal)

This document is for repository developers and coding agents launching or debugging multi-node training. It is not intended as external documentation. See [README.md](README.md) for recipe, dataset, and configuration details.

## Launch entry point

Start from the repository root:

```bash
REPO=/shared_nfs/zirui/code/primus-flux-slim
DATA_ROOT=/shared_nfs/zirui/data
OUTPUT_ROOT=/shared_nfs/zirui/runs/flux-$(date -u +%Y%m%dT%H%M%SZ)
cd "$REPO"
```

The default `config_4n_gbs1024.sh` requires four nodes with eight GPUs per node. For a short smoke test, add:

```bash
MAX_STEPS=1 SAVE_STRATEGY=none MLPERF_CLEAR_CACHES=false
```

Launch modes:

| `LAUNCH_MODE` | Actual entry point inside the container | Use case |
|---|---|---|
| `native` (default) | `torchrun ... examples/diffusion/train_native.py` | Normal development and performance testing; shortest execution path |
| `primus` | `./primus-cli direct -- train pretrain ...` | Primus CLI integration validation; the CLI still uses `torchrun` internally |

Both modes use `run_with_docker.sh` and therefore share the same container mounts, RCCL/AINIC settings, and distributed environment. Do not start one Slurm task per GPU. Start one `run_with_docker.sh` task per node and let the node-local `torchrun` create eight workers.

## Scenario 1: One multi-node allocation

This is the preferred approach. One allocation must own all participating nodes. Run the following from a shell inside that allocation:

```bash
export DATA_ROOT OUTPUT_ROOT
export LAUNCH_MODE=native                 # or primus
bash examples/mlperf/flux1/run_with_docker_slurm.sh
```

`run_with_docker_slurm.sh` will:

1. Verify that the allocation node count matches `NNODES` from `FLUX_CONFIG`.
2. Select the rank 0 node from the allocation nodelist as `MASTER_ADDR`.
3. Use `srun`/`spur run` to start one `run_with_docker.sh` process per node.
4. Use `SLURM_NODEID` as each node's `NODE_RANK`.

To select a configuration or override launch settings, export or prefix them before the command:

```bash
LAUNCH_MODE=primus \
FLUX_CONFIG=config_4n_gbs1024.sh \
MASTER_PORT=29601 \
DATA_ROOT="$DATA_ROOT" OUTPUT_ROOT="$OUTPUT_ROOT" \
bash examples/mlperf/flux1/run_with_docker_slurm.sh
```

If `alloc` returned only a group job ID and the current shell is still on the login node, launch one step across the complete group. This still creates only one task per node, and `SLURM_NODEID` becomes `NODE_RANK`:

```bash
JOB_ID=<multi-node-job-id>
NNODES=4
NODELIST=$(squeue -j "$JOB_ID" -h -o %N)
MASTER_NODE=$(scontrol show hostnames "$NODELIST" | head -n1)
MASTER_ADDR=$(getent ahostsv4 "$MASTER_NODE" | awk 'NR == 1 {print $1}')
mkdir -p "$OUTPUT_ROOT"

nohup spur run --jobid="$JOB_ID" --overlap \
  -N"$NNODES" -n"$NNODES" --ntasks-per-node=1 \
  env DATA_ROOT="$DATA_ROOT" OUTPUT_ROOT="$OUTPUT_ROOT" \
    NNODES="$NNODES" MASTER_ADDR="$MASTER_ADDR" MASTER_PORT=29601 \
    LAUNCH_MODE=native \
    bash "$REPO/examples/mlperf/flux1/run_with_docker.sh" \
  >"$OUTPUT_ROOT/group.launch.log" 2>&1 </dev/null &
```

If no allocation exists yet, submit one directly:

```bash
mkdir -p "$OUTPUT_ROOT"
sbatch -A amd-spur -p amd-spur --qos=amd-spur-qos \
  -N4 --ntasks-per-node=1 --exclusive --gpus-per-node=8 -t 04:00:00 \
  --output="$OUTPUT_ROOT/slurm-%j.log" \
  --wrap="cd '$REPO' && DATA_ROOT='$DATA_ROOT' OUTPUT_ROOT='$OUTPUT_ROOT' \
    LAUNCH_MODE=native bash examples/mlperf/flux1/run_with_docker_slurm.sh"
```

## Scenario 2: Multiple single-node allocations

A single `srun` cannot span multiple job IDs. When only separate single-node allocations are available, start one node-local process through each allocation and point every process at the same torchrun rendezvous.

Requirements:

- Every node can access the same repository, datasets, and output directory.
- Each node has eight available GPUs and access to the same Docker image.
- Ranks are contiguous from `0` through `NNODES-1`.
- Every node uses exactly the same `NNODES`, `MASTER_ADDR`, and `MASTER_PORT`.
- `MASTER_ADDR` is a hostname or IP for the rank 0 node that all training nodes can reach.
- Each `spur run` uses the job ID that owns its selected node.

First, create the per-node launcher:

```bash
NNODES=4
MASTER_NODE=<rank-0-node>
MASTER_ADDR=$(getent ahostsv4 "$MASTER_NODE" | awk 'NR == 1 {print $1}')
MASTER_PORT=29601
LAUNCH_MODE=native                       # or primus

mkdir -p "$OUTPUT_ROOT"
cat >"$OUTPUT_ROOT/launch_node.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

rank=$1
cd "${REPO:?}"
exec env \
  NNODES="${NNODES:?}" NODE_RANK="$rank" \
  MASTER_ADDR="${MASTER_ADDR:?}" MASTER_PORT="${MASTER_PORT:?}" \
  LAUNCH_MODE="${LAUNCH_MODE:?}" \
  CONTAINER_NAME="flux-multinode-${rank}" \
  bash examples/mlperf/flux1/run_with_docker.sh
EOF
chmod +x "$OUTPUT_ROOT/launch_node.sh"
export REPO DATA_ROOT OUTPUT_ROOT NNODES MASTER_ADDR MASTER_PORT LAUNCH_MODE
```

Fill in each `JOB_ID NODE RANK` entry. Start all commands close together so that some ranks do not wait too long at the rendezvous:

```bash
launches=(
  "<job-id-0> <node-0> 0"
  "<job-id-1> <node-1> 1"
  "<job-id-2> <node-2> 2"
  "<job-id-3> <node-3> 3"
)

: >"$OUTPUT_ROOT/launches.tsv"
for entry in "${launches[@]}"; do
  read -r job_id node rank <<<"$entry"
  nohup spur run --jobid="$job_id" --overlap -N1 -n1 --nodelist="$node" \
    env REPO="$REPO" DATA_ROOT="$DATA_ROOT" OUTPUT_ROOT="$OUTPUT_ROOT" \
      NNODES="$NNODES" MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" \
      LAUNCH_MODE="$LAUNCH_MODE" \
      "$OUTPUT_ROOT/launch_node.sh" "$rank" \
    >"$OUTPUT_ROOT/${node}.launch.log" 2>&1 </dev/null &
  printf '%s\t%s\t%s\t%s\n' "$node" "$job_id" "$rank" "$!" \
    | tee -a "$OUTPUT_ROOT/launches.tsv"
done
```

Add experimental environment variables explicitly to the `spur run ... env` arguments above so that every node receives identical values. For example, place these next to `LAUNCH_MODE=...`:

```bash
FSDP2_HSDP_FP8_ALL_REDUCE=e4m3 \
FSDP2_HSDP_FP8_BLOCK_SIZE=1024 \
```

## How torchrun and primus-cli relate

`run_with_docker.sh` is the common entry point for both modes.

### Native torchrun

```bash
LAUNCH_MODE=native bash examples/mlperf/flux1/run_with_docker.sh
```

This is equivalent to the following command inside the container:

```bash
torchrun \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --nproc_per_node="$GPUS_PER_NODE" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  examples/diffusion/train_native.py \
  --config examples/mlperf/flux1/flux.1_schnell_t2i-native.yaml
```

### Primus CLI

```bash
LAUNCH_MODE=primus bash examples/mlperf/flux1/run_with_docker.sh
```

This is equivalent to the following command inside the container:

```bash
./primus-cli direct -- train pretrain \
  --config examples/mlperf/flux1/flux.1_schnell_t2i-pretrain.yaml
```

`primus-cli direct` reads the same `NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, and `GPUS_PER_NODE` values and then constructs the torchrun command. Do not change rank or rendezvous settings when switching launch modes.

## Monitoring and cleanup

```bash
while IFS=$'\t' read -r node job_id rank pid; do
  printf '%s rank=%s job=%s ' "$node" "$rank" "$job_id"
  ps -p "$pid" -o stat=,etime= || echo EXITED
done <"$OUTPUT_ROOT/launches.tsv"

tail -f "$OUTPUT_ROOT"/*.launch.log
```

Check the rank 0 log first for the configuration summary, the 32-rank DeviceMesh, and the first training step. Also search for `Traceback`, `RuntimeError`, `NCCL`, `GPU Hang`, and OOM errors.

To stop a run launched across separate allocations, remove only this run's named containers or steps. Do not directly `scancel` a shared allocation without confirmation. For example, run the following through each owning allocation:

```bash
spur run --jobid=<job-id> --overlap -N1 -n1 --nodelist=<node> \
  docker rm -f flux-multinode-<rank>
```

## Common failures

- `Invalid config ... expected GBS`: `NNODES` does not match the selected `FLUX_CONFIG`.
- Rendezvous timeout: verify that every rank is present and unique, the port is unused, and every node can reach `MASTER_ADDR:MASTER_PORT`.
- `ABI-4 libionic or /dev/infiniband is unavailable`: the node is missing the required RDMA device or libionic path before container startup.
- Docker name conflict: choose a unique `CONTAINER_NAME` for each run or remove the stale container from the previous run.
- Only rank 0 starts: do not pass nodes owned by different job IDs to one `srun`; use the per-node `spur run` procedure above.
