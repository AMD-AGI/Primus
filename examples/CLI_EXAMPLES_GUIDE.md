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

```bash
# Megatron 训练
export EXP=examples/megatron/exp_pretrain.yaml
bash examples/run_pretrain_cli.sh

# 自定义配置
export EXP=my_experiments/custom_config.yaml
bash examples/run_pretrain_cli.sh
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

#### 基本使用

```bash
# PyTorch 训练
bash examples/run_local_pretrain_cli.sh

# 指定配置文件
EXP=examples/megatron/exp_pretrain.yaml \
bash examples/run_local_pretrain_cli.sh
```

#### MaxText/JAX 训练

```bash
# 使用 MaxText 后端
BACKEND=MaxText \
EXP=examples/maxtext/exp_config.yaml \
bash examples/run_local_pretrain_cli.sh
```

#### 自定义镜像和数据路径

```bash
# 使用自定义 Docker 镜像
DOCKER_IMAGE=my-registry.com/custom-image:v1.0 \
DATA_PATH=/mnt/shared/datasets \
bash examples/run_local_pretrain_cli.sh
```

#### 多节点训练（本地多容器）

```bash
# Node 0
NNODES=2 NODE_RANK=0 MASTER_ADDR=192.168.1.100 \
bash examples/run_local_pretrain_cli.sh

# Node 1
NNODES=2 NODE_RANK=1 MASTER_ADDR=192.168.1.100 \
bash examples/run_local_pretrain_cli.sh
```

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

#### 单节点训练

```bash
# 使用默认配置
bash examples/run_slurm_pretrain_cli.sh
```

#### 多节点训练

```bash
# 4 节点训练
export NNODES=4
export NODES_LIST="node[01-04]"
export EXP=examples/megatron/exp_pretrain.yaml
bash examples/run_slurm_pretrain_cli.sh
```

#### 指定日志目录

```bash
# 自定义日志目录
LOG_DIR=/shared/experiments/run_001 \
NNODES=8 \
NODES_LIST="gpu[01-08]" \
bash examples/run_slurm_pretrain_cli.sh
```

#### 完整示例

```bash
#!/bin/bash
# my_training_job.sh

# 设置实验配置
export EXP=experiments/llama3_8b.yaml

# 集群配置
export NNODES=16
export NODES_LIST="gpu[001-016]"
export MASTER_PORT=29500

# 日志配置
export LOG_DIR=/shared/experiments/llama3_8b_$(date +%Y%m%d_%H%M%S)

# 提交训练任务
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
