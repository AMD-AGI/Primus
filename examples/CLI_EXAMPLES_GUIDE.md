# Primus CLI Examples - Quick Start Guide

本文档介绍如何使用 Primus CLI 示例脚本进行模型训练。

## 📚 概述

Primus 提供了三种训练模式的示例脚本：

| 脚本 | 模式 | 适用场景 |
|------|------|----------|
| `run_pretrain_cli.sh` | Direct | 直接在主机上运行，无容器开销 |
| `run_local_pretrain_cli.sh` | Container | 使用 Docker/Podman 容器，环境隔离 |
| `run_slurm_pretrain_cli.sh` | Slurm | 集群环境，多节点训练 |

---

## 1️⃣ Direct Mode - 直接模式

**适用场景**: 在已配置好的环境中快速测试和训练

### 使用方法

```bash
# 基本使用
EXP=examples/megatron/exp_pretrain.yaml bash examples/run_pretrain_cli.sh

# 或者先导出环境变量
export EXP=examples/megatron/exp_pretrain.yaml
bash examples/run_pretrain_cli.sh
```

### 必需参数

- `EXP`: 实验配置文件路径（必须存在）

### 示例

#### 场景 1: 快速测试（默认配置）

```bash
# Megatron 训练，使用默认配置
export EXP=examples/megatron/exp_pretrain.yaml
bash examples/run_pretrain_cli.sh
```

#### 场景 2: 使用自定义配置

```bash
# 自定义实验配置
export EXP=my_experiments/custom_config.yaml
bash examples/run_pretrain_cli.sh
```

#### 场景 3: 传递额外参数

```bash
# 传递自定义参数到训练命令
export EXP=examples/megatron/exp_pretrain.yaml
bash examples/run_pretrain_cli.sh \
  --checkpoint-interval 500 \
  --log-level DEBUG \
  --enable-profiling
```

#### 场景 4: 完整命令行调用（绕过脚本）

如果需要更多控制，可以直接调用 `primus-cli-direct.sh`:

```bash
# 直接模式 + NUMA 绑定 + 自定义日志
bash $PRIMUS_PATH/runner/primus-cli-direct.sh \
  --numa \
  --log_file /tmp/training.log \
  -- train pretrain \
  --config examples/megatron/exp_pretrain.yaml \
  --checkpoint-interval 1000
```

---

## 2️⃣ Container Mode - 容器模式

**适用场景**: 需要环境隔离，或使用特定的 Docker 镜像

### 使用方法

```bash
# 基本使用 (PyTorch)
bash examples/run_local_pretrain_cli.sh

# MaxText/JAX 训练
BACKEND=MaxText bash examples/run_local_pretrain_cli.sh
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `EXP` | `examples/megatron/exp_pretrain.yaml` | 实验配置文件 |
| `BACKEND` | (空，使用 PyTorch) | 设置为 `MaxText` 使用 JAX 镜像 |
| `DOCKER_IMAGE` | PyTorch: `docker.io/rocm/primus:v25.10`<br>MaxText: `docker.io/rocm/jax-training:maxtext-v25.9` | Docker 镜像 |
| `DATA_PATH` | `$(pwd)/data` | 数据目录（自动挂载到容器） |
| `MASTER_ADDR` | `localhost` | 主节点地址 |
| `MASTER_PORT` | `1234` | 主节点端口 |
| `NNODES` | `1` | 节点数量 |
| `NODE_RANK` | `0` | 当前节点编号 |
| `GPUS_PER_NODE` | `8` | 每节点 GPU 数量 |

### 示例

#### 场景 1: 基本使用（默认配置）

```bash
# PyTorch 训练，使用默认镜像和配置
bash examples/run_local_pretrain_cli.sh

# 指定配置文件
EXP=examples/megatron/exp_pretrain.yaml \
bash examples/run_local_pretrain_cli.sh
```

#### 场景 2: 使用自定义镜像

```bash
# 使用自定义 Docker 镜像
DOCKER_IMAGE=my-registry.com/custom-image:v1.0 \
DATA_PATH=/mnt/shared/datasets \
bash examples/run_local_pretrain_cli.sh
```

#### 场景 3: 添加环境变量（性能调优）

```bash
# 添加 NCCL 和 PyTorch 性能相关环境变量
bash $PRIMUS_PATH/runner/primus-cli container \
  --image docker.io/rocm/primus:v25.10 \
  --volume $(pwd)/data:/data \
  --env HSA_NO_SCRATCH_RECLAIM=1 \
  --env NVTE_CK_USES_BWD_V3=1 \
  --env GPU_MAX_HW_QUEUES=2 \
  --env NCCL_SOCKET_IFNAME=eth0 \
  --env TORCH_DISTRIBUTED_DEBUG=OFF \
-- train pretrain --config examples/megatron/exp_pretrain.yaml
```

**说明**:
- `HSA_NO_SCRATCH_RECLAIM=1`: ROCm 性能优化
- `NVTE_CK_USES_BWD_V3=1`: TransformerEngine 后向传播优化
- `GPU_MAX_HW_QUEUES=2`: GPU 硬件队列配置
- `NCCL_SOCKET_IFNAME=eth0`: 指定 NCCL 使用的网络接口

#### 场景 4: MaxText/JAX 训练 + 自定义配置

```bash
# MaxText 后端 + 自定义镜像 + 环境变量
bash $PRIMUS_PATH/runner/primus-cli container \
  --image my-registry.com/jax-maxtext:custom \
  --volume /data/tokenized:/workspace/data \
  --env JAX_PLATFORMS=rocm \
  --env XLA_FLAGS="--xla_gpu_enable_async_all_gather=true --xla_gpu_all_reduce_combine_threshold_bytes=134217728" \
  --env NVTE_FUSED_ATTN=1 \
-- train pretrain --config examples/maxtext/config.yaml
```

#### 场景 5: 多节点训练（本地多容器协同）

在多台物理机上分别运行：

```bash
# 主节点 (Node 0)
NNODES=2 \
NODE_RANK=0 \
MASTER_ADDR=192.168.1.100 \
MASTER_PORT=29500 \
GPUS_PER_NODE=8 \
bash examples/run_local_pretrain_cli.sh

# 从节点 (Node 1)
NNODES=2 \
NODE_RANK=1 \
MASTER_ADDR=192.168.1.100 \
MASTER_PORT=29500 \
GPUS_PER_NODE=8 \
bash examples/run_local_pretrain_cli.sh
```

#### 场景 6: 完整的镜像 + 卷挂载 + 环境变量示例

```bash
#!/bin/bash
# advanced_container_training.sh

export DOCKER_IMAGE=docker.io/rocm/primus:v25.10
export DATA_PATH=/shared/datasets/llama
export EXP=experiments/llama3_70b.yaml

bash $PRIMUS_PATH/runner/primus-cli container \
  --image $DOCKER_IMAGE \
  --volume $DATA_PATH:/data:ro \
  --volume $(pwd)/checkpoints:/checkpoints \
  --volume $(pwd)/output:/workspace/output \
  --env NCCL_DEBUG=WARN \
  --env TORCH_DISTRIBUTED_DEBUG=OFF \
  --env CUDA_DEVICE_MAX_CONNECTIONS=1 \
  --env HF_TOKEN \
  --env WANDB_API_KEY \
  --gpus all \
  --shm-size 16g \
-- train pretrain --config $EXP --checkpoint-dir /checkpoints
```

**配置说明**:
- `/data` 只读挂载 (`:ro`)
- `/checkpoints` 可读写挂载
- `HF_TOKEN` 和 `WANDB_API_KEY` 从主机环境传递
- `--gpus all`: 使用所有可用 GPU
- `--shm-size 16g`: 增加共享内存（大模型训练必需）

---

## 3️⃣ Slurm Mode - 集群模式

**适用场景**: 使用 Slurm 管理的集群环境，多节点分布式训练

### 使用方法

```bash
# 基本使用
bash examples/run_slurm_pretrain_cli.sh

# 指定节点数和节点列表
NNODES=4 NODES_LIST="node[01-04]" \
bash examples/run_slurm_pretrain_cli.sh
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `EXP` | `examples/megatron/exp_pretrain.yaml` | 实验配置文件 |
| `NNODES` | `1` | 使用的节点数量 |
| `NODES_LIST` | `node[02,03,10,14,15,34,38]` | Slurm 节点列表 |
| `MASTER_PORT` | `12345` | 主节点端口 |
| `LOG_DIR` | `./output` | 日志输出目录 |

### 示例

#### 场景 1: 单节点快速测试

```bash
# 使用默认配置，单节点训练
bash examples/run_slurm_pretrain_cli.sh
```

#### 场景 2: 标准多节点训练

```bash
# 4 节点训练
export NNODES=4
export NODES_LIST="node[01-04]"
export EXP=examples/megatron/exp_pretrain.yaml
bash examples/run_slurm_pretrain_cli.sh
```

#### 场景 3: 使用容器镜像 + 环境变量

```bash
# Slurm 模式下使用自定义 Docker 镜像和环境变量
bash $PRIMUS_PATH/runner/primus-cli slurm srun \
  -N 4 \
  --nodelist "node[01-04]" \
-- \
  --image docker.io/rocm/primus:v25.10 \
  --env NCCL_DEBUG=INFO \
  --env TORCH_DISTRIBUTED_DEBUG=DETAIL \
  --env CUDA_DEVICE_MAX_CONNECTIONS=1 \
-- train pretrain --config examples/megatron/exp_pretrain.yaml
```

**说明**:
- `--image`: 指定 Docker 镜像
- `--env`: 设置容器内环境变量（可多次使用）
- 第一个 `--` 分隔 Slurm 选项
- 第二个 `--` 分隔容器选项
- 最后是 Primus 训练命令

#### 场景 4: 大规模训练 + 完整配置

```bash
#!/bin/bash
# large_scale_training.sh - 大规模 Llama3 70B 训练

# 实验配置
export EXP=experiments/llama3_70b.yaml
export LOG_DIR=/shared/experiments/llama3_70b_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"

# 集群配置
export NNODES=32
export NODES_LIST="gpu[001-032]"
export MASTER_PORT=29500

# Slurm + 容器 + 环境变量 完整示例
bash $PRIMUS_PATH/runner/primus-cli slurm srun \
  -N $NNODES \
  --nodelist "$NODES_LIST" \
  --gpus-per-node=8 \
  --ntasks-per-node=8 \
-- \
  --image docker.io/rocm/primus:v25.10 \
  --volume /shared/datasets:/data \
  --volume /shared/checkpoints:/checkpoints \
  --env NCCL_DEBUG=INFO \
  --env NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3 \
  --env TORCH_DISTRIBUTED_DEBUG=OFF \
  --env CUDA_DEVICE_MAX_CONNECTIONS=1 \
  --env HF_TOKEN \
-- train pretrain --config $EXP 2>&1 | tee "$LOG_DIR/training.log"
```

**配置说明**:
- 32 节点，每节点 8 GPU (256 GPUs 总计)
- 挂载共享数据集和 checkpoint 目录
- 配置 NCCL 通信参数
- HF_TOKEN 从主机环境传递（用于下载模型）
- 日志同时输出到文件和终端

#### 场景 5: MaxText/JAX 训练

```bash
# MaxText 后端训练（使用 JAX 镜像）
export EXP=examples/maxtext/config.yaml
export NNODES=8
export NODES_LIST="gpu[01-08]"

bash $PRIMUS_PATH/runner/primus-cli slurm srun \
  -N $NNODES \
  --nodelist "$NODES_LIST" \
-- \
  --image docker.io/rocm/jax-training:maxtext-v25.9 \
  --env JAX_COORDINATOR_IP \
  --env JAX_COORDINATOR_PORT \
  --env XLA_FLAGS="--xla_gpu_enable_async_all_gather=true" \
  --env NVTE_FUSED_ATTN=1 \
-- train pretrain --config $EXP
```

#### 场景 6: 使用特定 GPU 节点 + 调试模式

```bash
# 指定高性能 GPU 节点 + 开启调试
LOG_DIR=/tmp/debug_run \
bash $PRIMUS_PATH/runner/primus-cli slurm srun \
  -N 2 \
  --nodelist "mi300x[01-02]" \
  --constraint="mi300x" \
-- \
  --env NCCL_DEBUG=INFO \
  --env TORCH_DISTRIBUTED_DEBUG=DETAIL \
  --env PRIMUS_LOG_LEVEL=DEBUG \
-- train pretrain \
  --config examples/megatron/exp_pretrain.yaml \
  --dry-run
```

#### 场景 7: 简化版本（使用脚本的默认行为）

如果你的需求简单，可以编辑 `run_slurm_pretrain_cli.sh` 脚本：

```bash
# 1. 编辑脚本，修改默认值
nano examples/run_slurm_pretrain_cli.sh

# 2. 修改这些行：
# export NNODES=${NNODES:-4}              # 默认 4 节点
# export NODES_LIST=${NODES_LIST:-"gpu[01-04]"}  # 你的节点列表

# 3. 添加自定义环境变量（在第 27 行附近）：
#    --env NCCL_DEBUG=INFO \
#    --env CUDA_DEVICE_MAX_CONNECTIONS=1 \

# 4. 直接运行
bash examples/run_slurm_pretrain_cli.sh
```

---

## 🔧 高级用法

### 传递额外参数

所有脚本都支持传递额外参数到 `primus train` 命令：

```bash
# Direct mode
bash examples/run_pretrain_cli.sh --extra-param value

# Container mode
bash examples/run_local_pretrain_cli.sh --debug --dry-run

# Slurm mode
bash examples/run_slurm_pretrain_cli.sh --checkpoint-interval 100
```

### 环境变量传递（Container Mode）

`run_local_pretrain_cli.sh` 支持传递环境变量到容器：

```bash
# 脚本中已包含的环境变量示例：
# --env HSA_NO_SCRATCH_RECLAIM     # 从主机传递
# --env NVTE_CK_USES_BWD_V3        # 从主机传递
# --env GPU_MAX_HW_QUEUES          # 从主机传递
# --env GLOO_SOCKET_IFNAME         # 从主机传递

# 在主机设置环境变量，容器会自动获取
export HSA_NO_SCRATCH_RECLAIM=1
export GPU_MAX_HW_QUEUES=2
bash examples/run_local_pretrain_cli.sh
```

---

## 📝 常见问题

### Q: 如何选择使用哪个脚本？

**A**:
- 🏃 **快速测试**: 使用 `run_pretrain_cli.sh`（直接模式）
- 🐳 **环境隔离**: 使用 `run_local_pretrain_cli.sh`（容器模式）
- 🖥️ **多节点训练**: 使用 `run_slurm_pretrain_cli.sh`（Slurm 模式）

### Q: 容器模式的数据路径如何设置？

**A**: 使用 `DATA_PATH` 环境变量，该路径会自动挂载到容器内：

```bash
DATA_PATH=/mnt/shared/datasets bash examples/run_local_pretrain_cli.sh
```

### Q: Slurm 模式如何查看日志？

**A**: 日志会保存到 `LOG_DIR/log_slurm_pretrain.txt`：

```bash
# 实时查看日志
tail -f ./output/log_slurm_pretrain.txt

# 或指定日志目录
LOG_DIR=/tmp/my_logs bash examples/run_slurm_pretrain_cli.sh
tail -f /tmp/my_logs/log_slurm_pretrain.txt
```

### Q: 如何验证脚本配置是否正确？

**A**: 使用 `--dry-run` 参数（Direct 和 Container 模式支持）：

```bash
# 验证配置但不实际执行
bash examples/run_pretrain_cli.sh --dry-run
bash examples/run_local_pretrain_cli.sh --dry-run
```

---

## 📚 参考资料

- [Primus CLI 完整文档](../runner/README.md)
- [配置文件示例](../examples/)
- [故障排查指南](../docs/troubleshooting.md)

---

## 🎯 快速参考

```bash
# ===== Direct Mode =====
EXP=config.yaml bash examples/run_pretrain_cli.sh

# ===== Container Mode (PyTorch) =====
bash examples/run_local_pretrain_cli.sh

# ===== Container Mode (MaxText) =====
BACKEND=MaxText bash examples/run_local_pretrain_cli.sh

# ===== Slurm Mode =====
NNODES=4 NODES_LIST="node[01-04]" bash examples/run_slurm_pretrain_cli.sh
```

---

**更新时间**: 2026-01-09
**版本**: v1.0
