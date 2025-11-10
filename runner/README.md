# 🚀 Primus CLI：AMD 统一训练入口的核心控制层

> *“让 AMD GPU 的训练体验，从复杂脚本到统一入口。”*

---

## 一、背景与动机

在大型模型训练场景中，训练系统往往包含多个层次：
- 环境准备（GPU 检测、网络拓扑、RCCL/NCCL 环境配置）
- 框架调度（Megatron、TorchTitan、JAX 等）
- 任务启动（单节点、多节点、容器化、Slurm 作业）
- 性能验证与 Benchmark
- 训练前后的 Hook、Patch、Dataset 预处理

这些环节传统上由大量 Bash 脚本零散控制，逻辑重复且难以维护。
而 **Primus CLI** 的目标，是通过一个统一的入口层，将这些流程模块化、自动化、可配置化。

---

## 二、CLI 设计理念

Primus CLI 采用 **“三层结构 + 插件化体系”**：

```
Primus CLI
├── Command Layer        # 统一命令入口（train, benchmark, preflight）
├── Runtime Layer        # 环境与任务执行封装（direct, container, slurm）
└── Hook/Patch System    # 任务前后可插拔逻辑
```

| 维度 | 设计目标 |
|------|-----------|
| **一致性** | 不论 Megatron、TorchTitan 还是其他 backend，统一 CLI 入口 |
| **可扩展性** | Subcommand 插件自动发现，可动态注册新功能 |
| **可调试性** | 日志体系统一，支持 rank-aware 输出 |
| **可复现性** | 所有运行依赖由 Primus 环境层控制，可容器化、可 Slurm 化 |

---

## 三、核心结构

### 🧩 1. Subcommand 插件系统

CLI 主入口 `primus/cli/main.py` 自动扫描 `primus/cli/subcommands/` 下的模块：

```bash
primus-cli train pretrain --config configs/deepseek_v2.yaml
primus-cli benchmark gemm --dtype bf16
primus-cli preflight check --gpu
```

子命令通过注册表 (`primus/cli/registry.py`) 动态加载，实现可插拔扩展：

```python
from primus.cli.registry import register_subcommand

@register_subcommand("train")
def run_train(args, unknown_args):
    ...
```

这样添加一个新命令（如 `primus-cli analyze topology`）只需新增一个文件，而无需修改主入口逻辑。

---

### ⚙️ 2. Runtime 模式抽象

Primus CLI 支持三种执行模式：

| 模式 | 描述 | 典型入口 |
|------|------|-----------|
| **direct** | 直接在宿主环境中执行 | `primus-cli-direct.sh` |
| **container** | 在容器内执行（Docker/Podman） | `primus-cli-container.sh` |
| **slurm** | 在 Slurm 集群调度系统中执行 | `primus-cli-slurm.sh` |

三者共享统一的命令分发逻辑，区别仅在于：
- 运行环境的准备（容器、节点分配、网络配置）
- 环境变量传递（如 `MASTER_ADDR`, `NODE_RANK`, `HIP_VISIBLE_DEVICES`）
- 执行层入口（`primus-cli-entrypoint.sh`）

---

### 🔁 3. Hook 系统（execute_hooks.sh）

在训练流程中，CLI 会根据任务类型自动执行 Hook：

```
runner/helpers/hooks/train/pretrain/megatron/prepare.sh
runner/helpers/hooks/train/pretrain/megatron/preprocess_data.sh
```

执行逻辑：

```bash
execute_hooks "train" "pretrain" "$@"
```

它支持：
- 自动发现同目录下的脚本并按字典序执行；
- 统一日志与错误处理；
- 灵活注册 dataset preparation、env setup、sanity check 等任务。

---

## 四、🌐 环境自动检测与分层加载机制

Primus CLI 的环境加载由 `primus-env.sh` 实现，是整个 CLI 的 **基础支撑层**。

该脚本自动完成三件事：

### 1️⃣ 加载基础环境

```bash
source "${SCRIPT_DIR}/base_env.sh"
```

定义通用变量（`NODE_RANK`, `HOSTNAME`, `LOG_INFO_RANK0` 等）并配置常规 ROCm 环境。

### 2️⃣ 自动检测 GPU 型号

```bash
GPU_MODEL=$(bash "${SCRIPT_DIR}/detect_gpu.sh")
LOG_INFO_RANK0 "Detected GPU model: ${GPU_MODEL}"
```

支持自动识别 MI300X、MI355X 等 GPU 类型。

### 3️⃣ 按 GPU 型号加载特定配置

```bash
case "$GPU_MODEL" in
    *MI300*) GPU_CONFIG_FILE="${SCRIPT_DIR}/MI300X.sh" ;;
    *MI355*) GPU_CONFIG_FILE="${SCRIPT_DIR}/MI355X.sh" ;;
esac
source "$GPU_CONFIG_FILE"
```

> **👉 这一机制让 CLI 能自动匹配不同 GPU 架构的最佳配置，保证所有模块都在最优 ROCm 环境中运行。**

---

## 五、🧪 实验可复现性机制（Reproducibility）

Primus CLI 通过对运行时环境与配置文件的全量导出，实现“一键复现”。

### 1️⃣ 自动导出运行环境与配置

每次训练或 Benchmark 启动时，CLI 会将关键运行信息导出到 `output` 目录：

```
output/
├── env/
│   ├── primus_env_dump.txt
│   ├── gpu_model.txt
│   └── system_info.json
├── config/
│   └── primus_config.yaml
└── logs/
    └── launch.log
```

### 2️⃣ 一键复现机制

```bash
primus-cli train pretrain --replay output/exp_2025_11_07/
```

CLI 会自动加载环境与 config，保证运行一致性。

### 3️⃣ 使用场景

| 场景 | 说明 |
|------|------|
| 🔁 性能复现 | 比较不同版本模型在相同配置下的性能 |
| 🧩 问题定位 | 在其他集群快速重现 hang 或通信问题 |
| 🧱 回归测试 | 结合 CI 验证版本更新对训练影响 |
| 📦 模型归档 | 保存完整运行上下文，支持审计和迁移 |

---

## 六、Patch 系统（execute_patches.sh）

Patch 系统用于执行任务级动态修补脚本：

```bash
execute_patches runner/helpers/patches/train/pretrain/fix_env.sh                 runner/helpers/patches/train/pretrain/prepare_dataset.sh
```

每个 patch 独立执行，失败即中断。可用于 framework 兼容修补、性能优化等。

---

## 七、日志与调试体系

统一日志接口（由 `base_env.sh` / `common.sh` 提供）：

```bash
LOG_INFO_RANK0 "[Train] Starting pretrain job"
LOG_ERROR "[Patch] Missing dataset config"
```

输出样式示例：

```
[NODE-0(pdfc-aig-23)] [INFO] Preparing Megatron dataset
[NODE-1(pdfc-aig-24)] [ERROR] Tokenizer missing
```

---

## 八、典型执行流程

```bash
primus-cli train pretrain --config examples/megatron/configs/MI355X/deepseek_v2_lite.yaml
```

完整流程：

```
1. primus-cli-direct.sh               → 加载基础与 GPU 专属 env
2. execute_hooks train/pretrain       → 数据预处理与检查
3. execute_patches                    → 动态 patch 应用
4. export env + config to output      → 保存实验上下文
5. launch_megatron_trainer            → 启动分布式训练
```

---

## 九、CLI 架构优势

| 模块 | 优势 |
|------|------|
| Subcommand 插件 | 零侵入扩展 |
| Hook 系统 | 解耦预处理流程 |
| Patch 系统 | 支持 hotfix 与兼容调整 |
| 自动 Env 检测 | 自动适配 GPU 配置 |
| Reproducibility 导出 | 支持复现与审计 |
| 容器 / Slurm 支持 | 统一执行路径 |
| 多后端适配 | TorchTitan / Megatron / JAX 等统一接口 |

---

## 十、未来展望

- ✅ Hook/Patch 支持 Python API
- ✅ `primus plugin list` 插件管理
- ✅ Preflight 智能拓扑检查
- ✅ 异构后端（ROCm + CPU/FPGA）集成
- ✅ 可视化 Dashboard 展示训练状态、性能对比

---

## 十一、总结

Primus CLI 是 AMD 大模型训练平台的统一入口。它封装了调度、配置、日志、兼容性和复现性，为大规模训练提供了一条 **一致、可控、可扩展、可复现** 的路径。

> 🧠 一条命令，从环境到训练，全自动、可复现、可扩展 —— 这就是 Primus CLI。

---
