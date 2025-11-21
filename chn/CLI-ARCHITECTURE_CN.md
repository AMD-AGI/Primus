# 🚀 从混乱到秩序：打造 AMD GPU 大模型训练的统一入口

> **作者**: AMD AI Brain - 规模化训练 (TAS) 团队
> **发布日期**: 2025-11-10
> **标签**: `#ROCm` `#LLM训练` `#Primus` `#开发者工具` `#AMD-GPU`

[English](./README.md) | **中文版**

---

## 📖 故事的开始：训练流程的痛点

想象一下这样的场景：

你刚接手一个大模型训练项目，项目目录里散落着几十个 Bash 脚本：`setup_env_mi300x.sh`、`run_megatron_8node.sh`、`check_network.sh`、`preprocess_data_v2_final_really.sh`... 每个脚本都有自己的逻辑，相互之间耦合严重。

当你想在新的 GPU 集群上复现一次训练时，你需要：
1. 手动检测 GPU 型号，找到对应的环境配置脚本
2. 根据节点数量修改分布式参数
3. 确保数据预处理脚本在训练前执行
4. 手动设置十几个环境变量
5. 祈祷一切顺利运行...

**这就是我们在构建 Primus 平台时遇到的真实问题。**

在 AMD GPU 大模型训练生态中，我们面临着多个层次的复杂性：
- 🔧 **环境准备**：不同 GPU 型号（MI300X、MI250X）需要不同的 ROCm 配置
- 🔗 **网络拓扑**：RCCL/NCCL 环境、InfiniBand 配置千差万别
- 🎯 **框架调度**：Megatron、TorchTitan、JAX 等框架各有特点
- 🖥️ **执行环境**：本地开发、容器化、Slurm 集群三种场景
- 📊 **性能验证**：GEMM benchmark、通信性能测试
- 🛠️ **前后处理**：数据预处理、环境检查、热修复补丁

传统的做法是用大量 Bash 脚本来处理这些环节，但这带来了：
- ❌ 逻辑重复，难以维护
- ❌ 缺乏统一的错误处理
- ❌ 实验难以复现
- ❌ 新人上手门槛高

**Primus CLI 的诞生，就是为了解决这些痛点。**

---

## 💡 设计哲学：一条命令，搞定一切

我们的核心理念很简单：**一条命令，从环境配置到训练启动，全自动完成。**

```bash
# 就这么简单！
primus-cli direct -- train pretrain --config deepseek_v2.yaml
```

### 🏗️ 三层架构设计

Primus CLI 采用清晰的**三层结构 + 插件化体系**：

```
┌─────────────────────────────────────────────┐
│         Runtime Layer (运行时层)            │
│    direct | container | slurm               │
│  自动检测环境、配置 GPU、管理分布式           │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│      Hook/Patch System (扩展层)            │
│  数据预处理 | 环境检查 | 热修复补丁           │
│         可插拔的任务前后处理逻辑              │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│    Task Execution Layer (任务执行层)        │
│  train | benchmark | preflight | analyze   │
│       插件化的业务逻辑和任务执行              │
└─────────────────────────────────────────────┘
```

### 🎯 四大设计目标

| 目标 | 实现方式 | 用户收益 |
|------|---------|---------|
| **🔄 一致性** | 统一 CLI 入口，支持 Megatron/TorchTitan/JAX | 无需为不同框架学习不同命令 |
| **🔌 可扩展性** | 插件自动发现，动态注册新功能 | 添加新功能无需修改核心代码 |
| **🐛 可调试性** | Rank-aware 日志，详细的执行追踪 | 快速定位多节点训练问题 |
| **📦 可复现性** | 自动导出运行环境和配置 | 一键复现任何历史实验 |

---

## 🔍 深入了解：架构剖析

### ⚙️ 第一层：智能运行时抽象

不同的场景需要不同的运行环境，但用户不应该关心这些细节。Primus CLI 提供了三种无缝切换的运行模式：

| 🎭 模式 | 📝 使用场景 | 🎯 典型命令 |
|---------|------------|------------|
| **🖥️ Direct** | 本地开发、快速验证 | `primus-cli direct -- train pretrain` |
| **🐳 Container** | 环境隔离、依赖管理 | `primus-cli container --image rocm/megatron:v25.8 -- train pretrain` |
| **🖧 Slurm** | 多节点生产训练 | `primus-cli slurm srun -N 8 -- train pretrain` |

**关键亮点**：这三种模式共享相同的命令语法，只是在运行环境准备上有所不同：

```bash
# 本地测试
primus-cli direct -- benchmark gemm -M 4096

# 容器中测试（确保环境一致）
primus-cli container -- benchmark gemm -M 4096

# 生产环境（8节点集群）
primus-cli slurm srun -N 8 -- benchmark gemm -M 4096
```

**从开发到生产，只需要改变运行模式，命令和参数保持不变！**

---

### 🔁 第二层：Hook 与 Patch 系统

训练不只是运行一个 Python 脚本那么简单。你可能需要：
- 🗂️ 在训练前预处理数据集
- 🔍 检查 GPU 和网络环境
- 🩹 应用临时的 hotfix
- 📊 收集和上报指标

Primus CLI 的 Hook 系统让这一切自动化：

```bash
# 目录结构
runner/helpers/hooks/
└── train/
    └── pretrain/
        ├── 01_check_environment.sh
        ├── 02_prepare_dataset.sh
        └── 03_setup_monitoring.sh
```

当你运行 `primus-cli train pretrain` 时，这些 Hook 会按顺序自动执行。无需手动调用，无需修改训练代码。

**Patch 系统**则用于更灵活的场景：

```bash
primus-cli direct --patch fixes/workaround_rccl.sh \
  -- train pretrain --config config.yaml
```

这在你需要快速应用临时修复、或针对特定环境做调整时特别有用。

---

### 🧩 第三层：任务执行层

这一层负责执行具体的业务逻辑——训练、测试、环境检查等实际任务。还记得我们说要"零侵入扩展"吗？这是怎么做到的？

```bash
# 训练任务
primus-cli direct -- train pretrain --config deepseek_v2.yaml

# 性能测试任务
primus-cli direct -- benchmark gemm --dtype bf16 -M 8192

# 环境检查任务
primus-cli direct -- preflight check --gpu --network
```

每个任务都是一个独立的 Python 插件模块，通过装饰器自动注册：

```python
from primus.cli.registry import register_subcommand

@register_subcommand("train")
def run_train(args, unknown_args):
    # 你的训练业务逻辑
    ...
```

**想添加新任务？** 只需要在 `primus/cli/subcommands/` 目录下创建一个新插件文件，无需修改任何核心代码。比如你想添加一个拓扑分析任务：

```bash
# 新增文件: primus/cli/subcommands/analyze.py
# 就能直接使用
primus-cli direct -- analyze topology --visualize
```

这种插件化设计让 Primus CLI 能够快速响应新需求，保持核心稳定的同时不断扩展功能。

---

## 🌐 魔法背后：智能环境检测

这可能是 Primus CLI 最 "黑科技" 的部分了。

### 问题：不同 GPU 需要不同配置

AMD 的 GPU 家族很丰富：MI300X、MI250X、MI210... 每种 GPU 都有其最佳的 ROCm 配置、环境变量设置。传统做法是让用户手动选择配置，但这既容易出错又不够自动化。

### 解决方案：三步自动配置

**第 1 步：加载通用环境**

```bash
# base_env.sh 提供统一的日志、工具函数
source "${SCRIPT_DIR}/base_env.sh"
```

**第 2 步：自动检测 GPU**

```bash
# 智能检测当前节点的 GPU 型号
GPU_MODEL=$(bash "${SCRIPT_DIR}/detect_gpu.sh")
# 输出: MI300X
```

**第 3 步：加载 GPU 专属配置**

```bash
# 根据检测结果，自动加载最优配置
GPU_CONFIG_FILE="${SCRIPT_DIR}/${GPU_MODEL}.sh"
source "$GPU_CONFIG_FILE"  # 加载 MI300X.sh
```

现在，`MI300X.sh` 里可以包含这个 GPU 型号的所有最佳实践：
- ROCm 环境变量（`HSA_*`, `HIP_*`）
- RCCL 通信优化参数
- 内存管理策略
- 性能调优选项

**用户完全不需要关心这些细节，一切都是自动的。**

### 实战示例

```bash
# 在 MI300X 集群上
primus-cli direct -- train pretrain --config config.yaml
# 自动加载: MI300X.sh → 优化的 RCCL 配置

# 换到 MI250X 集群，同样的命令
primus-cli direct -- train pretrain --config config.yaml
# 自动加载: MI250X.sh → 适配的不同配置
```

**跨 GPU 迁移，零配置修改！**

---

## 🧪 科学实验的基石：可复现性

在机器学习研究中，可复现性至关重要。但现实是残酷的：

> *"这个实验我三个月前跑的，现在想复现一下... 咦，配置文件哪去了？环境变量是怎么设的来着？"*

这样的对话你是不是很熟悉？Primus CLI 用**自动化快照机制**彻底解决这个问题。

### 自动记录一切

每次训练启动时，Primus CLI 会自动保存完整的运行上下文：

```
output/exp_2025_11_10_134522/
├── env/
│   ├── primus_env_dump.txt      # 所有环境变量
│   ├── gpu_model.txt            # GPU 型号信息
│   ├── rocm_version.txt         # ROCm 版本
│   └── system_info.json         # 系统配置
├── config/
│   ├── primus_config.yaml       # Primus 配置
│   ├── model_config.yaml        # 模型配置
│   └── data_config.yaml         # 数据配置
├── logs/
│   └── launch.log               # 启动日志
└── metadata.json                # 运行元数据
```

### 一键复现

三个月后，当你想复现这个实验：

```bash
# 就这么简单！
primus-cli direct -- train pretrain --replay output/exp_2025_11_10_134522/
```

Primus CLI 会自动：
1. 恢复所有环境变量
2. 加载原始配置文件
3. 验证 GPU 和系统环境
4. 启动训练（如果环境兼容）

### 实战价值

| 场景 | 传统做法 | 使用 Primus CLI |
|------|---------|----------------|
| **🔁 性能对比** | 手动记录配置，容易遗漏 | 自动快照，精确复现 |
| **🐛 Bug 调试** | 难以在新环境复现问题 | `--replay` 一键复现 |
| **📊 A/B 测试** | 手动确保两次运行一致 | 自动保证配置一致性 |
| **🏆 论文实验** | 手写实验设置文档 | 一键导出完整环境 |
| **🔄 版本回归** | 依赖人工记录 | 自动化 CI 集成 |

---

## 📊 实战案例：从开发到生产

让我们通过一个真实场景，看看 Primus CLI 如何简化整个工作流。

### 场景：训练 DeepSeek-V2 模型

**第 1 步：本地开发与验证** 🖥️

```bash
# 在开发机上快速验证配置
primus-cli direct --debug -- train pretrain \
  --config configs/deepseek_v2_debug.yaml
```

几分钟内就能发现配置问题、数据格式错误等。

**第 2 步：容器环境测试** 🐳

```bash
# 确保在标准化环境中也能正常运行
primus-cli container \
  --image rocm/megatron-lm:v25.8_py310 \
  --mount /data:/data \
  -- train pretrain --config configs/deepseek_v2_debug.yaml
```

容器确保了环境一致性，避免"在我机器上能跑"的问题。

**第 3 步：小规模集群验证** 🖧

```bash
# 2 节点测试，验证分布式通信
primus-cli slurm srun -N 2 -p gpu-test \
  -- container --image rocm/megatron-lm:v25.8_py310 \
  -- train pretrain --config configs/deepseek_v2_small.yaml
```

**第 4 步：生产环境大规模训练** 🚀

```bash
# 64 节点、512 GPU 的生产训练
primus-cli slurm sbatch \
  -N 64 -p gpu-prod -t 72:00:00 \
  --job-name=deepseek-v2-prod \
  -o logs/train_%j.log \
  -- container --image rocm/megatron-lm:v25.8_py310 \
  -- train pretrain --config configs/deepseek_v2_prod.yaml
```

### 关键洞察

注意到了吗？**从开发到生产，核心命令结构保持不变**：
```
primus-cli <runtime> -- <subcommand> <args>
```

只是运行时环境（`direct` → `container` → `slurm`）在变化，其他一切都是统一的。

---

## 🎯 核心优势总结

经过前面的详细介绍，让我们总结一下 Primus CLI 带来的核心价值：

| 🎁 特性 | 💪 能力 | 🚀 价值 |
|--------|--------|--------|
| **统一入口** | 一条命令，适配所有场景 | 降低学习成本，提高开发效率 |
| **插件化** | 零侵入扩展新功能 | 快速响应新需求，保持系统稳定 |
| **智能环境** | 自动检测 GPU 并优化配置 | 跨平台迁移零成本 |
| **Hook 系统** | 自动化任务前后处理 | 解耦复杂流程，提高代码复用 |
| **可复现性** | 一键快照和恢复 | 科学实验的坚实基础 |
| **三层运行时** | Direct/Container/Slurm 无缝切换 | 从开发到生产的平滑路径 |
| **统一日志** | Rank-aware 的结构化日志 | 快速定位多节点问题 |

---

## 🛣️ 未来路线图

Primus CLI 仍在持续演进，我们的近期计划包括：

### 短期目标 (2025)
- 🎯 **Python Hook API**：支持 Python 脚本编写 Hook，提供更灵活的扩展能力
- 🎯 **智能 Preflight**：启动前自动检查 GPU 健康度、网络拓扑、InfiniBand 连通性
- 🎯 **配置模板系统**：内置常见模型的最佳实践配置模板
- 🎯 **性能分析报告**：自动生成训练性能分析报告，包含 GEMM、通信、I/O 瓶颈分析
- 🎯 **增强错误诊断**：智能错误分析和修复建议（如 RCCL hang、OOM 等常见问题）
- 🎯 **扩展框架支持**：完善对 TorchTitan、JAX/Flax 等更多训练框架的支持
- 🎯 **CI/CD 集成**：提供标准化的测试和验证流程，支持自动化回归测试

### 长期愿景
- 🌟 成为 **ROCm 生态的标准训练入口**

---

## 🎓 总结：一条命令的力量

回到文章开头的问题：如何让大模型训练从复杂走向简单？

Primus CLI 的答案是：**通过精心设计的抽象层，把复杂性隐藏在统一的接口之下。**

```bash
# 从这个简单的命令...
primus-cli direct -- train pretrain --config deepseek_v2.yaml

# ...到背后的：
# ✅ 自动 GPU 检测和配置
# ✅ 智能环境变量设置
# ✅ 数据预处理 Hook
# ✅ 分布式通信优化
# ✅ 日志收集和分析
# ✅ 实验快照和复现
# ✅ 错误处理和恢复
```

**这就是 Primus CLI：让复杂的事情变简单，让简单的事情变自动。**

---

## 📚 了解更多

- 📖 **用户指南**：[PRIMUS-CLI-GUIDE.md](./PRIMUS-CLI-GUIDE.md)
- 🔧 **快速开始**：`primus-cli --help`
- 💬 **问题反馈**：GitHub Issues
- 🌐 **ROCm 生态**：[rocm.github.io](https://rocm.github.io)

---

> *"The best interface is no interface."*
>
> 但在无法消除接口的地方，最好的接口是**统一、简洁、自动化**的。这就是 Primus CLI。

**Happy Training with AMD ROCm! 🎉**
