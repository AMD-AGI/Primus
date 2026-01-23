---
title: "Primus CLI：设计理念与优势"
date: "2026-01-23"
tags: ["Primus", "CLI", "ROCm", "大模型训练", "HPC", "Slurm", "工程效率"]
---

## 为什么需要一个统一的训练入口

大规模模型训练的难点，往往不在训练代码本身，而在“如何把一次实验稳定跑起来”：本机、容器、Slurm 集群之间的环境差异；分布式参数与拓扑差异；以及越来越多的“配套工作”（benchmarks、preflight 检查、诊断分析、热修复等）。

Primus CLI 的目标就是解决这个问题：提供一个**统一且一致的命令入口**，把训练、benchmark、环境检查等工作流收敛到一个结构化的 CLI 中，同时尽可能保证在不同运行环境下走**同一条执行路径**。

本文重点介绍 Primus CLI 的**设计理念**与**工程优势**。具体用法可参考 `docs/cli/README.md` 与完整指南 `docs/cli/PRIMUS-CLI-GUIDE.md`。

## 设计理念

### 1）统一入口，统一心智模型

传统大规模训练项目里，训练、benchmark、preflight 往往由不同脚本启动；参数风格不一致、环境变量设置方式各不相同，最终会带来较高的使用与维护成本。Primus CLI 用子命令体系把这些入口统一起来。

示例：

```bash
primus-cli direct -- train posttrain --config exp.yaml
primus-cli direct -- benchmark gemm -M 8192 -N 8192 -K 8192 --dtype bf16
primus-cli direct -- preflight --host --gpu --network
```

带来的收益：

- 命令结构清晰、易记
- 减少脚本间重复的“胶水逻辑”
- 新同学上手更快

### 2）跨环境保持同一路径（Preserved Execution Path）

Primus CLI 的核心目标之一是：在本机、容器、Slurm 等不同运行环境下，尽可能保持同一条任务执行链路，只把差异收敛到运行时准备层。

Primus 支持三种执行模式（见 `docs/cli/README.md`）：

- **Direct**：本机/快速验证/调试
- **Container**：隔离环境、保证依赖一致性
- **Slurm**：HPC 集群多节点调度与分布式执行

命令结构保持一致：

```bash
# 本机
primus-cli direct -- benchmark gemm -M 4096 -N 4096 -K 4096

# 容器
primus-cli container --image rocm/primus:v25.10 -- benchmark gemm -M 4096 -N 4096 -K 4096

# Slurm
primus-cli slurm srun -N 2 -- benchmark gemm -M 16384 -N 16384 -K 16384
```

带来的收益：

- 避免“本机脚本”和“集群脚本”分叉
- 调试更容易复现（入口一致、日志路径更统一）
- 更少环境污染与不确定性

### 3）模块化、可扩展

Primus CLI 的设计强调“核心稳定 + 能力可插拔”。新增任务（例如新的 benchmark suite、新的诊断工具、新的训练流程）应尽量以独立模块形式加入，而不是修改核心入口并复制/分叉已有逻辑。

### 4）Python 优先，同时 Slurm 友好

Primus 在编排层采用 Python（便于处理 YAML 配置、对接训练框架与工具链），同时对 Slurm 保持一等公民支持。在 `runner/` 下，运行时 launcher 把调度与环境准备的差异封装起来，让任务层只关心“做什么”。

## 架构如何映射到代码仓库

从工程结构上，Primus CLI 可以理解为三层：

### Runtime 层：direct / container / slurm

`runner/` 目录包含统一入口与不同运行时 launcher，例如：

- `runner/primus-cli`
- `runner/primus-cli-direct.sh`
- `runner/primus-cli-container.sh`
- `runner/primus-cli-slurm.sh`
- `runner/primus-cli-slurm-entry.sh`

这些脚本负责“在哪跑/怎么调度/怎么准备环境”，并把任务参数交给任务层执行。

### Hook / Patch 层：把前后置流程从训练代码剥离

大规模训练往往需要复杂的前后置步骤：安装依赖、准备 checkpoint、环境检查、热修复等。Primus 通过 hook/patch 机制把这类步骤做成可组合的流水线，从而减少对训练代码的侵入，也减少脚本复制粘贴。

### Task 层：train / benchmark / preflight / analyze

任务层负责具体的训练、benchmark、preflight、分析工具等逻辑；运行时层负责把任务以一致方式落到 direct/container/slurm 中执行。

## 工程优势总结

- **学习成本低**：统一入口覆盖多个工作流
- **可复现性更强**：跨环境尽量保持同一路径
- **更易调试**：减少分叉路径带来的日志/行为差异
- **减少胶水代码**：hook/patch 复用通用前后置步骤
- **扩展更安全**：新增能力不需要改核心入口

## 使用示例

### 训练

```bash
primus-cli direct -- train posttrain --config examples/megatron_bridge/configs/MI355X/qwen3_8b_sft_posttrain.yaml
```

### Benchmarks

```bash
primus-cli direct -- benchmark gemm -M 8192 -N 8192 -K 8192 --dtype bf16
primus-cli direct -- benchmark rccl --op allreduce --num-bytes 1048576
```

### Preflight 环境检查

```bash
primus-cli direct -- preflight --host --gpu --network
```

## Roadmap（方向性）

- 支持更多后端与任务类型
- 更统一的训练与微调任务面（pretrain / sft / rlhf 等）
- 诊断与自动调优（拓扑采集、通信调参、profiling 报告）
- 沉淀可复现的模型与集群“配方化”示例

## 结语

Primus CLI 试图把训练工程中最昂贵的不确定性（环境差异、入口分叉、脚本重复、难以复现）系统性地下沉到可维护的架构中，让用户用一个一致的命令结构完成从开发到生产、从单机到多节点的迁移与复现。

