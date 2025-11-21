# Primus CLI 使用指南

> 统一的 AMD GPU 大模型工作流命令行接口

[English](./PRIMUS-CLI-GUIDE.md) | **中文版**

---

## 📋 目录

- [快速开始](#快速开始) - 5 分钟上手指南
- [执行模式](#执行模式) - Direct / Container / Slurm
- [配置文件](#配置文件) - YAML 配置详解
- [使用示例](#使用示例) - 实战案例集合
- [全局选项](#全局选项) - 通用命令参数
- [完整调用逻辑](#完整调用逻辑) - 内部执行流程（高级）
- [最佳实践](#最佳实践) - 推荐工作流
- [故障排除](#故障排除) - 常见问题解决

---

## 快速开始

### 基本语法

```bash
primus-cli [全局选项] <模式> [模式参数] -- [Primus 命令和参数]
```

### 核心概念

- **全局选项**：适用于所有模式的选项（如 `--debug`, `--config`）
- **模式**：执行环境（`slurm` / `container` / `direct`）
- **分隔符 `--`**：必须使用，用于分隔模式参数和 Primus 命令
- **Primus 命令**：Python CLI 命令（`train`, `benchmark`, `preflight` 等）

### 第一个命令

```bash
# 在当前主机直接运行 GEMM benchmark
primus-cli direct -- benchmark gemm -M 4096 -N 4096 -K 4096
```

---

## 执行模式

Primus CLI 支持三种执行模式，每种适用于不同的场景。

### 1. 🖥️ Direct 模式

**适用场景**：在当前主机或已有容器内直接执行

**特点**：
- 最简单的执行方式
- 适合单节点训练或调试
- 在当前环境直接运行，无额外开销

**语法**：
```bash
primus-cli direct [选项] -- <Primus命令>
```

**示例**：
```bash
# 基本训练
primus-cli direct -- train pretrain --config config.yaml

# GEMM benchmark
primus-cli direct -- benchmark gemm -M 4096 -N 4096 -K 4096

# 环境检查
primus-cli direct -- preflight check --gpu
```

**适用于**：
- ✅ 本地开发和调试
- ✅ 单节点训练
- ✅ 快速实验
- ✅ 在已有容器内运行

---

### 2. 🐳 Container 模式

**适用场景**：在 Docker/Podman 容器中执行

**特点**：
- 提供隔离的运行环境
- 自动挂载必要的设备和目录
- 支持自定义镜像和资源限制
- 适合需要特定环境的任务

**语法**：
```bash
primus-cli container [容器选项] -- <Primus命令>
```

**常用选项**：
| 选项 | 描述 | 示例 |
|------|------|------|
| `--image IMAGE` | 指定容器镜像 | `--image rocm/primus:v25.9` |
| `--mount PATH[:PATH]` | 挂载目录 | `--mount /data:/data` |
| `--cpus N` | 限制 CPU 核心数 | `--cpus 16` |
| `--memory SIZE` | 限制内存大小 | `--memory 128G` |
| `--clean` | 启动前清理所有容器 | `--clean` |

**示例**：
```bash
# 使用默认镜像运行训练
primus-cli container -- train pretrain --config config.yaml

# 指定镜像和挂载数据目录
primus-cli container --image rocm/primus:latest \
  --mount /mnt/data:/data \
  -- train pretrain --config /data/config.yaml

# 设置资源限制
primus-cli container --cpus 32 --memory 256G \
  -- benchmark gemm -M 8192 -N 8192 -K 8192

# 挂载本地 Primus 代码用于开发
primus-cli container --mount ~/workspace/Primus:/workspace/Primus \
  -- train pretrain
```

**适用于**：
- ✅ 需要特定依赖环境
- ✅ 环境隔离和复现性
- ✅ 开发和测试不同版本
- ✅ CI/CD 流水线

---

### 3. 🖧 Slurm 模式

**适用场景**：在 Slurm 集群上执行分布式任务

**特点**：
- 支持多节点分布式训练
- 自动处理节点分配和任务调度
- 支持 `srun`（交互式）和 `sbatch`（批处理）
- 完整的 Slurm 参数支持

**语法**：
```bash
primus-cli slurm [srun|sbatch] [Slurm参数] -- <Primus命令>
```

**常用 Slurm 参数**：
| 参数 | 简写 | 描述 | 示例 |
|------|------|------|------|
| `--nodes` | `-N` | 节点数 | `-N 4` |
| `--partition` | `-p` | 分区 | `-p gpu` |
| `--time` | `-t` | 时间限制 | `-t 4:00:00` |
| `--output` | `-o` | 输出日志文件 | `-o job.log` |
| `--job-name` | `-J` | 作业名称 | `-J train_job` |

**示例**：
```bash
# 使用 srun 在 4 个节点上运行训练（交互式）
primus-cli slurm srun -N 4 -p gpu -- train pretrain --config config.yaml

# 使用 sbatch 提交批处理作业
primus-cli slurm sbatch -N 8 -p AIG_Model -t 8:00:00 -o train.log \
  -- train pretrain --config deepseek_v2.yaml

# 运行分布式 GEMM benchmark
primus-cli slurm srun -N 2 -- benchmark gemm -M 16384 -N 16384 -K 16384

# 多节点环境检查
primus-cli slurm srun -N 4 -- preflight check --network
```

**适用于**：
- ✅ 多节点分布式训练
- ✅ 大规模模型训练
- ✅ 需要作业调度和资源管理
- ✅ 生产环境训练任务

---

## 配置文件

Primus CLI 支持 YAML 格式的配置文件，可以预设各种选项。

### 配置文件位置

配置文件按以下优先级加载：

1. **命令行指定**：`--config /path/to/config.yaml`（最高优先级）
2. **系统默认**：`runner/.primus.yaml`
3. **用户配置**：`~/.primus.yaml`（最低优先级）

### 配置文件结构

```yaml
# 全局设置
main:
  debug: false
  dry_run: false

# Slurm 配置
slurm:
  nodes: 2
  time: "4:00:00"
  partition: "gpu"
  gpus_per_node: 8

# Container 配置
container:
  image: "rocm/primus:v25.9_gfx942"
  options:
    cpus: "32"
    memory: "256G"
    ipc: "host"
    network: "host"

    # GPU 设备（不要修改）
    devices:
      - "/dev/kfd"
      - "/dev/dri"
      - "/dev/infiniband"

    # 权限
    capabilities:
      - "SYS_PTRACE"
      - "CAP_SYS_ADMIN"

    # 挂载点
    mounts:
      - "/data:/data"
      - "/model_weights:/weights:ro"

# Direct 模式配置
direct:
  gpus_per_node: 8
  master_port: 1234
  numa: "auto"
```

### 使用配置文件

```bash
# 使用项目配置文件
primus-cli --config .primus.yaml slurm srun -N 4 -- train pretrain

# 使用用户自定义配置
primus-cli --config ~/my-config.yaml container -- benchmark gemm

# 配置文件 + 命令行参数（命令行参数优先级更高）
primus-cli --config prod.yaml slurm srun -N 8 -- train pretrain
```

### 配置优先级

**优先级顺序**（从高到低）：
```
命令行参数 > 指定配置文件 > 系统默认配置 > 用户配置
```

**示例**：
```bash
# 配置文件中设置 nodes=2，命令行指定 -N 4
# 最终使用 4 个节点（命令行优先）
primus-cli --config .primus.yaml slurm srun -N 4 -- train pretrain
```

---

## 使用示例

### 训练任务

#### 单节点训练（Direct）
```bash
# 基本训练
primus-cli direct -- train pretrain --config config.yaml

# 使用调试模式
primus-cli --debug direct -- train pretrain --config config.yaml
```

#### 单节点训练（Container）
```bash
# 使用容器运行
primus-cli container --mount /data:/data \
  -- train pretrain --config /data/config.yaml

# 自定义资源限制
primus-cli container --cpus 64 --memory 512G \
  -- train pretrain --config config.yaml
```

#### 多节点训练（Slurm）
```bash
# 4 节点分布式训练
primus-cli slurm srun -N 4 -p gpu -- train pretrain --config config.yaml

# 提交批处理作业
primus-cli slurm sbatch -N 8 -p AIG_Model -t 12:00:00 \
  -o train_%j.log -e train_%j.err \
  -- train pretrain --config deepseek_v2.yaml
```

### Benchmark 任务

#### GEMM Benchmark
```bash
# 单节点 GEMM
primus-cli direct -- benchmark gemm -M 4096 -N 4096 -K 4096

# 容器中运行
primus-cli container -- benchmark gemm -M 8192 -N 8192 -K 8192

# 多节点 GEMM
primus-cli slurm srun -N 2 -- benchmark gemm -M 16384 -N 16384 -K 16384
```

#### 其他 Benchmark
```bash
# All-reduce benchmark
primus-cli slurm srun -N 4 -- benchmark allreduce --size 1GB

# 端到端训练性能
primus-cli slurm srun -N 8 -- benchmark e2e --model llama2-7b
```

### 环境检查（Preflight）

```bash
# GPU 检查
primus-cli direct -- preflight check --gpu

# 网络检查
primus-cli slurm srun -N 4 -- preflight check --network

# 完整环境检查
primus-cli slurm srun -N 4 -- preflight check --all
```

### 组合使用

```bash
# 使用配置文件 + 调试模式 + 干跑
primus-cli --config prod.yaml --debug --dry-run \
  slurm srun -N 4 -- train pretrain

# 容器 + 自定义镜像 + 多挂载点
primus-cli container \
  --image rocm/primus:dev \
  --mount /data:/data \
  --mount /models:/models:ro \
  --mount /output:/output \
  -- train pretrain --config /data/config.yaml

# Slurm + 配置文件 + 资源限制
primus-cli --config cluster.yaml slurm sbatch \
  -N 16 -p bigmem --exclusive \
  -- train pretrain --config llama3-70b.yaml
```

---

## 全局选项

全局选项适用于所有执行模式（Direct、Container、Slurm），在模式名称之前指定。

### 可用选项

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `--config FILE` | 指定配置文件路径 | `runner/.primus.yaml` |
| `--debug` | 启用调试模式（详细日志） | 关闭 |
| `--dry-run` | 显示将要执行的命令但不实际运行 | 关闭 |
| `--version` | 显示版本信息并退出 | - |
| `-h, --help` | 显示帮助信息并退出 | - |

### 详细说明

#### `--config FILE`
指定自定义配置文件，覆盖默认配置。

```bash
# 使用生产环境配置
primus-cli --config configs/prod.yaml slurm srun -N 4 -- train pretrain

# 使用相对路径
primus-cli --config ./my-config.yaml direct -- benchmark gemm

# 使用绝对路径
primus-cli --config /shared/configs/cluster.yaml container -- train pretrain
```

**配置优先级**：`--config` 指定的文件 > 系统默认 `runner/.primus.yaml` > 用户配置 `~/.primus.yaml`

#### `--debug`
启用调试模式，输出详细的执行日志，包括：
- 配置加载过程
- 环境变量设置
- 命令构建步骤
- 内部函数调用

```bash
# 调试 Slurm 作业
primus-cli --debug slurm srun -N 2 -- train pretrain --config config.yaml

# 调试容器启动
primus-cli --debug container --image rocm/primus:dev -- benchmark gemm

# 调试配置加载
primus-cli --debug --config test.yaml direct -- preflight check
```

**环境变量**：`--debug` 会设置 `PRIMUS_LOG_LEVEL=DEBUG`

#### `--dry-run`
干跑模式，显示完整的执行命令但不实际运行，适用于：
- 验证配置正确性
- 查看最终命令
- 调试参数传递
- CI/CD 流水线测试

```bash
# 查看 Slurm 作业会如何提交
primus-cli --dry-run slurm sbatch -N 8 -p gpu -- train pretrain

# 查看容器启动命令
primus-cli --dry-run container --mount /data:/data -- benchmark gemm

# 查看分布式训练命令
primus-cli --dry-run direct -- train pretrain --config config.yaml
```

**输出格式**：
```
==========================================
  [DRY RUN] Slurm Configuration
==========================================
Launch Command: srun
SLURM Flags:
  -N 8
  -p gpu
  -t 4:00:00
Entry Script: primus-cli-slurm-entry.sh
==========================================
```

#### `--version`
显示 Primus CLI 版本信息并退出。

```bash
primus-cli --version
# 输出: Primus CLI v1.0.0
```

#### `-h, --help`
显示帮助信息，可以在不同层级使用：

```bash
# 主入口帮助
primus-cli --help

# 模式特定帮助
primus-cli direct --help
primus-cli container --help
primus-cli slurm --help

# Primus Python CLI 帮助
primus-cli direct -- --help
primus-cli direct -- train --help
primus-cli direct -- benchmark --help
```

### 组合使用示例

#### 调试 + 配置文件
```bash
primus-cli --config dev.yaml --debug direct -- train pretrain
```

#### 干跑 + 自定义配置
```bash
primus-cli --config prod.yaml --dry-run slurm srun -N 4 -- train pretrain
```

#### 多层级调试
```bash
# 查看完整的命令构建和执行过程
primus-cli --debug --dry-run slurm sbatch -N 8 -- container --debug -- train pretrain
```

### 全局选项的生效范围

```
primus-cli [全局选项] <模式> [模式参数] -- [Primus 命令]
           ↑
           └─ 影响整个执行流程
              • 主入口 (primus-cli)
              • 模式脚本 (primus-cli-*.sh)
              • 最终执行 (primus-cli-direct.sh)
```

**注意事项**：
- 全局选项必须在模式名称之前指定
- `--debug` 会传递到所有子脚本
- `--dry-run` 会在模式脚本层面拦截执行
- `--config` 的配置会在所有阶段生效

---

## 完整调用逻辑

> 📌 **提示**: 本节面向高级用户，详细解释 Primus CLI 的内部执行流程。如果你是新手，可以直接跳到[最佳实践](#最佳实践)。

### 执行流程图

```
┌───────────────────────────────────────────────────────────────────┐
│   用户命令                                                         │
│   primus-cli [全局选项] <模式> [模式参数] -- [Primus命令]           │
└────────────────────────────────┬──────────────────────────────────┘
                                 │
                                 ↓
┌───────────────────────────────────────────────────────────────────┐
│   1. primus-cli (主入口)                                           │
│   • 解析全局选项 (--config, --debug, --dry-run)                    │
│   • 加载配置文件 (.primus.yaml)                                    │
│   • 提取 main.* 配置                                               │
│   • 设置调试模式和日志级别                                          │
└────────────────────────────────┬──────────────────────────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ↓                  ↓                  ↓
     ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
     │    Direct    │   │  Container   │   │    Slurm     │
     └───────┬──────┘   └───────┬──────┘   └───────┬──────┘
             │                  │                  │
             │                  │                  │
             ↓                  ↓                  ↓
┌──────────────────────────────────────────────────────────────────┐
│   2. 模式特定脚本 (primus-cli-*.sh)                                │
│   • 加载 container/slurm/direct 配置                              │
│   • 解析模式特定参数                                               │
│   • 准备执行环境                                                   │
│     - Slurm: 构建 srun/sbatch 命令                                │
│     - Container: 启动容器                                         │
│     - Direct: 加载 GPU 环境                                       │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
                                 ↓
┌──────────────────────────────────────────────────────────────────┐
│   3. primus-cli-direct.sh (最终执行层)                            │
│   • 加载环境 (primus-env.sh)                                      │
│     - base_env.sh (基础环境)                                      │
│     - detect_gpu.sh (GPU 检测)                                   │
│     - GPU-specific env (MI300X.sh 等)                            │
│   • 执行 Hooks (execute_hooks.sh)                                │
│   • 应用 Patches (execute_patches.sh)                            │
│   • 设置分布式环境变量                                            │
│     - MASTER_ADDR, NODE_RANK                                     │
│     - NCCL/RCCL 配置                                             │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
                                 ↓
┌──────────────────────────────────────────────────────────────────┐
│   4. Primus Python CLI (primus/cli/main.py)                      │
│   • 解析 Primus 命令 (train/benchmark/等)                         │
│   • 加载子命令插件                                                │
│   • 执行具体任务                                                  │
│     - train: 启动训练 (Megatron/TorchTitan)                       │
│     - benchmark: 性能测试 (GEMM/RCCL)                             │
│     - preflight: 环境检查                                         │
└──────────────────────────────────────────────────────────────────┘
```

### 调用示例解析

以一个完整的 Slurm 多节点容器化训练命令为例：

```bash
primus-cli --config prod.yaml --debug \
  slurm srun -N 4 -- container --image rocm/megatron-lm:v25.8_py310 \
  -- train pretrain --config deepseek_v2.yaml
```

**执行步骤详解**：

```
第 1 步: primus-cli 主入口
  ├─ 解析全局选项: --config prod.yaml, --debug
  ├─ 加载配置: prod.yaml + .primus.yaml
  ├─ 提取 main.debug=true
  ├─ 设置 PRIMUS_LOG_LEVEL=DEBUG
  └─ 识别模式: slurm

第 2 步: primus-cli-slurm.sh
  ├─ 加载 slurm.* 配置 (nodes, time, partition 等)
  ├─ 解析 Slurm 参数: srun -N 4
  ├─ 合并配置和 CLI 参数 (CLI 优先)
  │   配置: nodes=2, time=4:00:00, partition=gpu
  │   CLI: -N 4
  │   结果: nodes=4, time=4:00:00, partition=gpu
  ├─ 构建 SLURM_FLAGS: [-N 4 -p gpu -t 4:00:00]
  └─ 生成命令: srun -N 4 -p gpu -t 4:00:00 \
                primus-cli-slurm-entry.sh -- \
                --config prod.yaml --debug \
                container --image rocm/megatron-lm:v25.8_py310 \
                -- train pretrain --config deepseek_v2.yaml

第 3 步: primus-cli-slurm-entry.sh (在每个 Slurm 节点上执行)
  ├─ 设置节点环境变量
  │   NODE_RANK, MASTER_ADDR, WORLD_SIZE
  └─ 调用: primus-cli-container.sh --config prod.yaml --debug \
            --image rocm/megatron-lm:v25.8_py310 \
            -- train pretrain --config deepseek_v2.yaml

第 4 步: primus-cli-container.sh (在每个节点上)
  ├─ 加载 container.* 配置 (image, devices, mounts 等)
  ├─ 解析容器参数: --image rocm/megatron-lm:v25.8_py310
  ├─ 合并配置和 CLI 参数
  │   配置: image=rocm/primus:v25.9
  │   CLI: --image rocm/megatron-lm:v25.8_py310
  │   结果: image=rocm/megatron-lm:v25.8_py310
  ├─ 构建容器选项
  │   --device /dev/kfd, /dev/dri, /dev/infiniband
  │   --cap-add SYS_PTRACE, CAP_SYS_ADMIN
  │   --volume (挂载数据和代码)
  │   --cpus, --memory (资源限制)
  └─ 启动容器: docker/podman run --rm \
                --device /dev/kfd --device /dev/dri \
                --volume $PWD:/workspace/Primus \
                --env NODE_RANK=$NODE_RANK \
                --env MASTER_ADDR=$MASTER_ADDR \
                rocm/megatron-lm:v25.8_py310 \
                /bin/bash -c "cd /workspace/Primus && \
                  bash runner/primus-cli-direct.sh \
                  --config prod.yaml --debug \
                  -- train pretrain --config deepseek_v2.yaml"

第 5 步: primus-cli-direct.sh (在容器内执行)
  ├─ 加载环境脚本
  │   ├─ base_env.sh (通用环境)
  │   ├─ detect_gpu.sh (检测到 MI300X)
  │   └─ MI300X.sh (GPU 专属配置)
  ├─ 执行 Hooks
  │   └─ execute_hooks "train" "pretrain"
  │       ├─ hooks/train/pretrain/01_prepare.sh
  │       └─ hooks/train/pretrain/02_preprocess_data.sh
  ├─ 应用 Patches
  │   └─ execute_patches (如果配置了)
  ├─ 设置分布式环境
  │   MASTER_ADDR=node-0, NODE_RANK=0..3
  │   NCCL_SOCKET_IFNAME, NCCL_IB_HCA
  └─ 启动: torchrun --nproc_per_node=8 \
            --nnodes=4 --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            primus/cli/main.py train pretrain \
            --config deepseek_v2.yaml

第 6 步: primus/cli/main.py (Python CLI，在容器内)
  ├─ 解析命令: train pretrain
  ├─ 加载子命令插件: primus/cli/subcommands/train.py
  ├─ 执行训练
  │   ├─ 加载配置: deepseek_v2.yaml
  │   ├─ 初始化 Megatron
  │   ├─ 设置模型、数据、优化器
  │   └─ 开始分布式训练
  └─ 输出日志和指标
```

**多层嵌套说明**：
- **Slurm 层**：负责多节点资源分配和任务调度
- **Container 层**：在每个节点上提供隔离的运行环境
- **Direct 层**：在容器内执行实际的训练任务

这种三层架构实现了：
- ✅ 多节点分布式训练（Slurm）
- ✅ 环境一致性和隔离（Container）
- ✅ GPU 自动配置和优化（Direct）

### 配置优先级流程

配置系统采用分层合并策略，优先级从低到高：

```
┌─────────────────────────────────────────┐
│        配置来源 (从低到高优先级)          │
└─────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────┐
│  1. 用户全局配置 (~/.primus.yaml)        │
│     优先级: ★☆☆☆                     │
│     用途: 个人默认配置                   │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  2. 系统默认配置 (runner/.primus.yaml)   │
│     优先级: ★★☆☆                     │
│     用途: 项目默认配置                   │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  3. 指定配置文件 (--config FILE)         │
│     优先级: ★★★☆                     │
│     用途: 环境特定配置                   │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  4. 命令行参数                           │
│     优先级: ★★★★ (最高)               │
│     用途: 临时覆盖                       │
└─────────────────────────────────────────┘
```

> 📖 参考系统默认配置文件：[`.primus.yaml`](.primus.yaml)

**配置合并示例**：

```bash
# ~/.primus.yaml:     nodes=1
# .primus.yaml:       nodes=2, time=2:00:00
# --config prod.yaml: nodes=4, time=4:00:00, partition=gpu
# CLI: -N 8

最终结果:
  nodes=8           (来自 CLI - 最高优先级)
  time=4:00:00      (来自 prod.yaml)
  partition=gpu     (来自 prod.yaml)
```

### 三种模式的调用差异

| 组件 | Direct 模式 | Container 模式 | Slurm 模式 |
|------|------------|----------------|-----------|
| **入口脚本** | primus-cli-direct.sh | primus-cli-container.sh | primus-cli-slurm.sh |
| **环境准备** | 加载本地 GPU 环境 | 启动容器 + 挂载 + 设备 | 分配节点 + 网络配置 |
| **执行位置** | 当前主机 | 容器内 | Slurm 分配的节点 |
| **最终调用** | 直接执行 torchrun | 容器内执行 direct.sh | 每节点执行 slurm-entry.sh → direct.sh |
| **分布式支持** | 单节点多卡 | 单节点多卡 | 多节点多卡 |
| **适用场景** | 开发调试 | 环境隔离 | 生产训练 |

### GPU 环境自动配置

Primus CLI 会自动检测 GPU 型号并加载优化配置：

```
detect_gpu.sh (检测 GPU 型号)
      ↓
  MI300X / MI250X / MI210 / ...
      ↓
source ${GPU_MODEL}.sh (加载 GPU 专属配置)
      ↓
  • HSA_* 环境变量
  • NCCL/RCCL 优化参数
  • GPU 特定的运行时配置
```

**支持的 GPU 配置文件**：
- `MI300X.sh` - AMD Instinct MI300X
- `MI250X.sh` - AMD Instinct MI250X
- `MI210.sh` - AMD Instinct MI210
- 更多...

如果没有找到对应的 GPU 配置，系统会使用 `base_env.sh` 作为后备。

---

## 最佳实践

### 1. 使用配置文件管理环境

为不同环境创建不同的配置文件：

```bash
configs/
├── dev.yaml          # 开发环境
├── test.yaml         # 测试环境
└── prod.yaml         # 生产环境
```

### 2. 善用调试模式

```bash
# 先用 dry-run 查看会执行什么
primus-cli --dry-run slurm srun -N 4 -- train pretrain

# 确认无误后用 debug 模式详细追踪
primus-cli --debug slurm srun -N 4 -- train pretrain
```

### 3. 容器开发工作流

```bash
# 本地开发：挂载本地代码
primus-cli container \
  --mount ~/workspace/Primus:/workspace/Primus \
  -- train pretrain --config config.yaml

# 测试：使用 staging 镜像
primus-cli container --image rocm/primus:staging \
  -- benchmark gemm

# 生产：使用 release 镜像
primus-cli container --image rocm/primus:v1.0.0 \
  -- train pretrain
```

### 4. Slurm 作业管理

```bash
# 交互式开发和调试
primus-cli slurm srun -N 1 -- train pretrain --debug

# 生产训练：批处理模式
primus-cli slurm sbatch -N 8 -t 24:00:00 \
  -o logs/train_%j.log \
  -- train pretrain --config production.yaml
```

### 5. 日志管理

```bash
# Slurm 自动管理日志
primus-cli slurm sbatch -N 4 \
  -o logs/stdout_%j.log \
  -e logs/stderr_%j.log \
  -- train pretrain

# Container 日志重定向
primus-cli container -- train pretrain 2>&1 | tee train.log
```

---

## 故障排除

### 常见问题

#### 1. "Unknown or unsupported mode"

**原因**：模式名称错误或脚本文件缺失

**解决**：
```bash
# 检查可用模式
ls runner/primus-cli-*.sh

# 确保使用正确的模式名
primus-cli slurm ...    # ✓ 正确
primus-cli Slurm ...    # ✗ 错误（大小写）
```

#### 2. "Config file not found"

**原因**：配置文件路径错误

**解决**：
```bash
# 使用绝对路径
primus-cli --config /full/path/to/config.yaml ...

# 或相对于当前目录
primus-cli --config ./configs/dev.yaml ...
```

#### 3. Container 启动失败

**原因**：Docker/Podman 未安装或权限不足

**解决**：
```bash
# 检查容器运行时
which docker || which podman

# 检查权限
docker ps
podman ps

# 使用 dry-run 查看命令
primus-cli --dry-run container -- train pretrain
```

#### 4. Slurm 作业提交失败

**原因**：Slurm 参数错误或资源不足

**解决**：
```bash
# 检查可用分区
sinfo

# 检查队列状态
squeue

# 使用 dry-run 检查命令
primus-cli --dry-run slurm srun -N 4 -- train pretrain
```

#### 5. "Failed to load library"

**原因**：依赖库文件缺失

**解决**：
```bash
# 检查库文件
ls runner/lib/

# 确保必要文件存在
# - lib/common.sh
# - lib/config.sh
```

### 调试技巧

#### 启用详细日志
```bash
# 方法 1：使用 --debug
primus-cli --debug direct -- train pretrain

# 方法 2：设置环境变量
export PRIMUS_LOG_LEVEL=DEBUG
primus-cli direct -- train pretrain
```

#### 使用 Dry-run
```bash
# 查看完整命令但不执行
primus-cli --dry-run slurm srun -N 4 -- train pretrain
```

#### 检查配置加载
```bash
# 使用 debug 模式查看加载的配置
primus-cli --debug --config .primus.yaml direct -- train pretrain
```

### 获取帮助

```bash
# 主入口帮助
primus-cli --help

# 模式特定帮助
primus-cli slurm --help
primus-cli container --help
primus-cli direct --help

# Primus Python CLI 帮助
primus-cli direct -- --help
primus-cli direct -- train --help
primus-cli direct -- benchmark --help
```

---

## 参考资源

### 相关文档
- [CLI 架构文档](./CLI-ARCHITECTURE_CN.md) - Primus CLI 架构深度解析
- [主文档索引](../README.md) - Primus 完整文档索引
- [.primus.yaml](../../runner/.primus.yaml) - 默认配置示例

### 退出码约定

| 退出码 | 含义 | 示例 |
|--------|------|------|
| 0 | 成功 | 正常执行完成 |
| 1 | 库或依赖失败 | 缺失配置库文件 |
| 2 | 无效参数或配置 | 配置文件不存在 |
| 3 | 运行时执行失败 | 训练过程失败 |

---

## 版本信息

- **当前版本**：1.0.0
- **最后更新**：2025-11-10

---

**Happy Training! 🚀**
