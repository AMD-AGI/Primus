# FLUX.1 多节点开发启动手册（内部）

本文件供本仓库开发者和 coding agent 启动、排查多节点训练使用，不作为对外文档。配方、数据和配置说明见 [README.md](README.md)。

## 启动入口

统一从仓库根目录启动：

```bash
REPO=/shared_nfs/zirui/code/primus-flux-slim
DATA_ROOT=/shared_nfs/zirui/data
OUTPUT_ROOT=/shared_nfs/zirui/runs/flux-$(date -u +%Y%m%dT%H%M%SZ)
cd "$REPO"
```

默认使用 `config_4n_gbs1024.sh`，要求 4 个节点、每节点 8 张 GPU。短 smoke test 可增加：

```bash
MAX_STEPS=1 SAVE_STRATEGY=none MLPERF_CLEAR_CACHES=false
```

启动模式：

| `LAUNCH_MODE` | 容器内实际入口 | 用途 |
|---|---|---|
| `native`（默认） | `torchrun ... examples/diffusion/train_native.py` | 日常开发和性能测试，路径最短 |
| `primus` | `./primus-cli direct -- train pretrain ...` | Primus CLI 集成验证；CLI 内部仍使用 `torchrun` |

两个模式都通过 `run_with_docker.sh` 获得相同的容器挂载、RCCL/AINIC 配置和分布式环境。不要在每张 GPU 上启动一个 Slurm task：每个节点只启动一个 `run_with_docker.sh`，由节点内的 `torchrun` 创建 8 个 worker。

## 场景一：一个 multi-node allocation

这是首选方式。一个 allocation 必须同时拥有全部节点；在该 allocation 的 shell 中执行：

```bash
export DATA_ROOT OUTPUT_ROOT
export LAUNCH_MODE=native                 # 或 primus
bash examples/mlperf/flux1/run_with_docker_slurm.sh
```

`run_with_docker_slurm.sh` 会：

1. 检查 allocation 节点数与 `FLUX_CONFIG` 中的 `NNODES` 一致；
2. 从 allocation nodelist 选择 rank 0 节点作为 `MASTER_ADDR`；
3. 用 `srun`/`spur run` 在每个节点启动一个 `run_with_docker.sh`；
4. 用 `SLURM_NODEID` 设置各节点的 `NODE_RANK`。

指定配置或启动参数时，在命令前导出或传入：

```bash
LAUNCH_MODE=primus \
FLUX_CONFIG=config_4n_gbs1024.sh \
MASTER_PORT=29601 \
DATA_ROOT="$DATA_ROOT" OUTPUT_ROOT="$OUTPUT_ROOT" \
bash examples/mlperf/flux1/run_with_docker_slurm.sh
```

如果 `alloc` 只返回了 group job ID、当前仍在 login node，可直接在整个 group 上启动一步。这里仍然只为每个节点创建一个 task，`SLURM_NODEID` 会成为 `NODE_RANK`：

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

如果还没有 allocation，可直接提交：

```bash
mkdir -p "$OUTPUT_ROOT"
sbatch -A amd-spur -p amd-spur --qos=amd-spur-qos \
  -N4 --ntasks-per-node=1 --exclusive --gpus-per-node=8 -t 04:00:00 \
  --output="$OUTPUT_ROOT/slurm-%j.log" \
  --wrap="cd '$REPO' && DATA_ROOT='$DATA_ROOT' OUTPUT_ROOT='$OUTPUT_ROOT' \
    LAUNCH_MODE=native bash examples/mlperf/flux1/run_with_docker_slurm.sh"
```

## 场景二：多个 single-node allocations

只有多个独立的 single-node allocation 时，不能用一个 `srun` 跨越多个 job ID。需要通过每个 allocation 各启动一个节点本地进程，并让它们连接到同一个 torchrun rendezvous。

要求：

- 所有节点可见同一份仓库、数据和输出目录；
- 每个节点有 8 张空闲 GPU，并可访问相同 Docker image；
- rank 必须是连续的 `0..NNODES-1`；
- 所有节点使用完全相同的 `NNODES`、`MASTER_ADDR` 和 `MASTER_PORT`；
- `MASTER_ADDR` 是 rank 0 节点可被其他训练节点访问的 hostname 或 IP；
- 每条 `spur run` 使用拥有对应节点的 job ID。

先准备节点 launcher：

```bash
NNODES=4
MASTER_NODE=<rank-0-node>
MASTER_ADDR=$(getent ahostsv4 "$MASTER_NODE" | awk 'NR == 1 {print $1}')
MASTER_PORT=29601
LAUNCH_MODE=native                       # 或 primus

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

填写 `JOB_ID NODE RANK`。所有命令应尽快一起启动，避免部分 rank 长时间等待 rendezvous：

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

若需要实验环境变量，在上面的 `spur run ... env` 参数中显式加入，确保每个节点一致。例如将以下两项放在 `LAUNCH_MODE=...` 旁边：

```bash
FSDP2_HSDP_FP8_ALL_REDUCE=e4m3 \
FSDP2_HSDP_FP8_BLOCK_SIZE=1024 \
```

## torchrun 与 primus-cli 的对应关系

`run_with_docker.sh` 是两种模式的共同入口。

### Native torchrun

```bash
LAUNCH_MODE=native bash examples/mlperf/flux1/run_with_docker.sh
```

容器内等价于：

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

容器内等价于：

```bash
./primus-cli direct -- train pretrain \
  --config examples/mlperf/flux1/flux.1_schnell_t2i-pretrain.yaml
```

`primus-cli direct` 读取相同的 `NNODES`、`NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT` 和 `GPUS_PER_NODE`，再构造 torchrun。切换模式时不要改 rank 或 rendezvous 设置。

## 监控和停止

```bash
while IFS=$'\t' read -r node job_id rank pid; do
  printf '%s rank=%s job=%s ' "$node" "$rank" "$job_id"
  ps -p "$pid" -o stat=,etime= || echo EXITED
done <"$OUTPUT_ROOT/launches.tsv"

tail -f "$OUTPUT_ROOT"/*.launch.log
```

优先检查 rank 0 日志中的配置摘要、32-rank DeviceMesh、首个训练 step，以及 `Traceback`、`RuntimeError`、`NCCL`、`GPU Hang` 或 OOM。

停止拆分 allocation 的运行时，只清理本次运行的命名容器或对应 step；共享 allocation 未经确认不要直接 `scancel`。例如在每个所属 allocation 上执行：

```bash
spur run --jobid=<job-id> --overlap -N1 -n1 --nodelist=<node> \
  docker rm -f flux-multinode-<rank>
```

## 常见错误

- `Invalid config ... expected GBS`：`NNODES` 与所选 `FLUX_CONFIG` 不一致。
- rendezvous timeout：检查 rank 是否完整且唯一、端口是否被占用、其他节点是否能访问 `MASTER_ADDR:MASTER_PORT`。
- `ABI-4 libionic or /dev/infiniband is unavailable`：节点容器启动前的 RDMA 设备或 libionic 路径不满足要求。
- Docker name conflict：为每次运行设置唯一的 `CONTAINER_NAME`，或清理上次残留容器。
- 只有 rank 0 启动：不要把不同 job ID 的节点交给单个 `srun`；使用上面的 per-node `spur run` 方法。
