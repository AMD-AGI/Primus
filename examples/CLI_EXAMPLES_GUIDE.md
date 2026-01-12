# Primus CLI Examples - Quick Start Guide

本文档介绍如何使用 Primus CLI 示例脚本进行模型训练。

## 📚 概述

Primus 提供了三种训练模式的示例脚本，**重点推荐使用 Direct 和 Slurm 模式**：

| 脚本 | 模式 | 适用场景 | 推荐度 |
|------|------|----------|--------|
| `run_pretrain_cli.sh` | Direct | 直接在主机上运行，无容器开销 | ⭐⭐⭐ |
| `run_slurm_pretrain_cli.sh` | Slurm | 集群环境，多节点训练 | ⭐⭐⭐ |
| `run_local_pretrain_cli.sh` | Container | 使用 Docker/Podman 容器，环境隔离 | ⭐ |

---

## 1️⃣ Direct Mode - 直接模式

**适用场景**: 在已配置好的环境中快速测试和训练 ⭐⭐⭐ **推荐**

### 使用方法

```bash
# 使用默认配置（Llama3.1 8B BF16）
bash examples/run_pretrain_cli.sh

# 使用自定义配置
export EXP=my_experiments/custom_config.yaml
bash examples/run_pretrain_cli.sh

# 传递额外参数
bash examples/run_pretrain_cli.sh \
  --train_iters 10 \
  --micro_batch_size 4 \
  --global_batch_size 128
```

> 💡 **提示**: `run_pretrain_cli.sh` 脚本中包含完整的使用示例和说明，查看脚本头部注释获取更多场景。

### 默认配置

- 默认使用 `examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml`
- 可通过 `EXP` 环境变量覆盖

---

## 2️⃣ Slurm Mode - 集群模式

**适用场景**: 使用 Slurm 管理的集群环境，多节点分布式训练 ⭐⭐⭐ **推荐**

### 使用方法

```bash
# 基本使用（单节点，默认配置）
bash examples/run_slurm_pretrain_cli.sh

# 指定节点数
NNODES=4 bash examples/run_slurm_pretrain_cli.sh

# 自定义配置文件
EXP=my_experiments/custom_config.yaml bash examples/run_slurm_pretrain_cli.sh
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `EXP` | `examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml` | 实验配置文件 |
| `NNODES` | `1` | 使用的节点数量 |
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
# 4 节点训练，指定节点列表
bash $PRIMUS_PATH/runner/primus-cli slurm srun \
  -N 4 \
  --nodelist "node[01-04]" \
-- train pretrain --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml

# 或使用环境变量
NNODES=4 EXP=my_config.yaml bash examples/run_slurm_pretrain_cli.sh
```

#### 场景 3: 传递额外训练参数

```bash
# 覆盖批次大小和迭代次数
bash examples/run_slurm_pretrain_cli.sh \
  --micro_batch_size 4 \
  --global_batch_size 128 \
  --train_iters 10
```

#### 场景 4: 使用容器镜像 + 清理旧容器

```bash
# Slurm 模式下使用自定义 Docker 镜像
bash $PRIMUS_PATH/runner/primus-cli slurm srun \
  -N 2 \
  --nodelist "node[01-02]" \
-- container \
  --image docker.io/rocm/primus:v25.10 \
  --clean \
-- train pretrain --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
```

**说明**:
- `container` 子命令用于指定容器相关选项
- `--image`: 指定 Docker 镜像
- `--clean`: 训练结束后清理容器
- 第一个 `--` 分隔 Slurm 选项
- 第二个 `--` 分隔容器选项和训练命令

#### 场景 5: 环境变量 + 容器模式

```bash
# 设置调试环境变量
bash $PRIMUS_PATH/runner/primus-cli slurm srun \
  -N 4 \
  --nodelist "node[01-04]" \
-- container \
  --image docker.io/rocm/primus:v25.10 \
-- \
  --env NCCL_DEBUG=INFO \
  --env TORCH_DISTRIBUTED_DEBUG=DETAIL \
-- train pretrain --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
```

**说明**:
- 第一个 `--` 后是 `container` 子命令和镜像选项
- 第二个 `--` 后是环境变量（`--env` 可多次使用）
- 第三个 `--` 后是训练命令

#### 场景 6: 大规模训练 + 完整配置

```bash
#!/bin/bash
# large_scale_training.sh - 大规模 Llama3 70B 训练

# 实验配置
export EXP=experiments/llama3_70b.yaml
export LOG_DIR=/shared/experiments/llama3_70b_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"

# 集群配置
export NNODES=32
export MASTER_PORT=29500

# Slurm + 容器 + 环境变量 完整示例
bash $PRIMUS_PATH/runner/primus-cli slurm srun \
  -N $NNODES \
  --nodelist "gpu[001-032]" \
  --gpus-per-node=8 \
  --ntasks-per-node=8 \
-- container \
  --image docker.io/rocm/primus:v25.10 \
  --volume /shared/datasets:/data \
  --volume /shared/checkpoints:/checkpoints \
-- \
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
- `HF_TOKEN` 从主机环境传递（用于下载模型）
- 日志同时输出到文件和终端

#### 场景 7: MaxText/JAX 训练

```bash
# MaxText 后端训练（使用 JAX 镜像）
export EXP=examples/maxtext/config.yaml
export NNODES=8

bash $PRIMUS_PATH/runner/primus-cli slurm srun \
  -N $NNODES \
  --nodelist "gpu[01-08]" \
-- container \
  --image docker.io/rocm/jax-training:maxtext-v25.9 \
-- \
  --env JAX_COORDINATOR_IP \
  --env JAX_COORDINATOR_PORT \
  --env XLA_FLAGS="--xla_gpu_enable_async_all_gather=true" \
  --env NVTE_FUSED_ATTN=1 \
-- train pretrain --config $EXP
```

#### 场景 8: 使用特定 GPU 节点 + 调试模式

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
  --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml \
  --dry-run
```

#### 场景 9: 简化版本（使用脚本的默认行为）

如果你的需求简单，可以编辑 `run_slurm_pretrain_cli.sh` 脚本：

```bash
# 1. 编辑脚本，修改默认值
nano examples/run_slurm_pretrain_cli.sh

# 2. 修改这些行：
# export NNODES=${NNODES:-4}              # 默认 4 节点
# export EXP=${EXP:-"your_default_config.yaml"}

# 3. 如需容器模式，将 Scenario 1 改为：
# bash $PRIMUS_PATH/runner/primus-cli slurm srun -N $NNODES \
# -- container \
#   --image docker.io/rocm/primus:v25.10 \
# -- \
#   --env NCCL_DEBUG=INFO \
# -- train pretrain --config $EXP $* 2>&1 | tee $LOG_FILE

# 4. 直接运行
bash examples/run_slurm_pretrain_cli.sh
```

---

## 🔧 高级用法

### 传递额外参数

所有脚本都支持传递额外参数到 `primus train` 命令：

```bash
# Direct mode - 覆盖批次大小和迭代次数
bash examples/run_pretrain_cli.sh \
  --train_iters 10 \
  --micro_batch_size 4 \
  --global_batch_size 128

# Slurm mode - 添加 checkpoint 间隔
bash examples/run_slurm_pretrain_cli.sh \
  --checkpoint-interval 100 \
  --log-level DEBUG
```

### 使用 NUMA 绑定优化性能（Direct Mode）

```bash
# 启用 NUMA 绑定
bash $PRIMUS_PATH/runner/primus-cli direct \
  --numa \
  -- train pretrain --config $EXP
```

### Slurm 高级选项

```bash
# 指定 GPU 类型和资源
bash $PRIMUS_PATH/runner/primus-cli slurm srun \
  -N 8 \
  --nodelist "gpu[01-08]" \
  --gpus-per-node=8 \
  --ntasks-per-node=8 \
  --constraint="mi300x" \
  --exclusive \
-- train pretrain --config $EXP
```

---

## 🐳 Container Mode (可选)

**注意**: 如果您已经有配置好的环境，推荐使用 Direct 或 Slurm 模式。Container Mode 主要用于环境隔离或特殊镜像需求。

### 基本使用

```bash
# PyTorch 训练
bash examples/run_local_pretrain_cli.sh

# MaxText/JAX 训练
BACKEND=MaxText bash examples/run_local_pretrain_cli.sh

# 自定义镜像
DOCKER_IMAGE=my-registry.com/custom:v1.0 \
bash examples/run_local_pretrain_cli.sh
```

### 详细配置

详细的容器配置选项请参考：`bash $PRIMUS_PATH/runner/primus-cli container --help`

---

## 📝 常见问题

### Q: 如何选择使用哪个脚本？

**A**:
- 🏃 **单机测试/开发**: 使用 `run_pretrain_cli.sh` (Direct Mode) - **推荐**
- 🖥️ **多节点训练**: 使用 `run_slurm_pretrain_cli.sh` (Slurm Mode) - **推荐**
- 🐳 **环境隔离/特殊镜像**: 使用 `run_local_pretrain_cli.sh` (Container Mode) - 可选

### Q: Direct Mode 如何设置实验配置？

**A**: 必须通过 `EXP` 环境变量指定配置文件：

```bash
# 方式 1: 导出环境变量
export EXP=examples/megatron/exp_pretrain.yaml
bash examples/run_pretrain_cli.sh

# 方式 2: 内联设置
EXP=config.yaml bash examples/run_pretrain_cli.sh
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

### Q: 如何在 Slurm 上使用特定的 GPU 节点？

**A**: 使用 `--nodelist` 选项：

```bash
# 方式 1: 使用环境变量（推荐简单场景）
NNODES=4 bash examples/run_slurm_pretrain_cli.sh

# 方式 2: 直接调用 CLI 指定节点列表
bash $PRIMUS_PATH/runner/primus-cli slurm srun \
  -N 4 \
  --nodelist "mi300x[01-04]" \
  --constraint="mi300x" \
-- train pretrain --config $EXP
```

### Q: 如何传递额外的训练参数？

**A**: 在脚本后面直接添加参数即可：

```bash
# Direct mode - 传递批次大小参数
bash examples/run_pretrain_cli.sh \
  --train_iters 10 \
  --micro_batch_size 4

# Slurm mode - 传递 checkpoint 参数
bash examples/run_slurm_pretrain_cli.sh \
  --checkpoint-interval 500 \
  --enable-profiling
```

### Q: 如何验证配置是否正确？

**A**: 使用 `--dry-run` 参数：

```bash
# Direct mode
bash examples/run_pretrain_cli.sh --dry-run

# Slurm mode
bash examples/run_slurm_pretrain_cli.sh --dry-run
```

---

## 📚 参考资料

- [Primus CLI 完整文档](../runner/README.md)
- [配置文件示例](../examples/)
- [故障排查指南](../docs/troubleshooting.md)

---

## 🎯 快速参考

```bash
# ===== Direct Mode (推荐用于单机测试) =====
# 使用默认配置
bash examples/run_pretrain_cli.sh

# 使用自定义配置
EXP=config.yaml bash examples/run_pretrain_cli.sh

# 传递额外参数
bash examples/run_pretrain_cli.sh --train_iters 10 --micro_batch_size 4

# ===== Slurm Mode (推荐用于多节点训练) =====
# 单节点
bash examples/run_slurm_pretrain_cli.sh

# 多节点
NNODES=4 bash examples/run_slurm_pretrain_cli.sh

# 指定节点列表
bash $PRIMUS_PATH/runner/primus-cli slurm srun \
  -N 4 --nodelist "node[01-04]" \
-- train pretrain --config $EXP

# ===== Container Mode (可选) =====
bash examples/run_local_pretrain_cli.sh
```

---

**更新时间**: 2026-01-12
**版本**: v1.1
