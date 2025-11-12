# Primus CLI 快速入门指南

本指南帮助你快速上手使用优化后的 Primus CLI。

## 📦 安装

### 基本安装
```bash
# 安装 Primus（已包含 CLI）
pip install -e .
```

### 可选功能安装
```bash
# Shell 自动补全支持
pip install -r primus/cli/requirements-cli-optional.txt
```

---

## 🚀 快速开始

### 1. 查看可用命令
```bash
# 查看版本
primus --version

# 查看所有命令
primus --help

# 查看特定命令帮助
primus train --help
primus benchmark --help
primus preflight --help
```

### 2. 环境检查（推荐首次运行）
```bash
# 运行环境检查
primus preflight --check-all

# 输出示例：
# 🐍 Python Environment Check
# ----------------------------------------------------------------------
#   Python version: 3.10.12
# ✅   Python version is compatible (>= 3.8)
# ✅   torch is installed
# ✅   yaml is installed
# ...
```

### 3. 训练任务
```bash
# 基本训练
primus train pretrain --config config.yaml

# 详细输出（调试）
primus -v train pretrain --config config.yaml

# 使用配置文件
primus --config ~/.primus/config.yaml train pretrain
```

### 4. 性能基准测试
```bash
# GEMM 基准测试
primus benchmark gemm

# 详细输出
primus -v benchmark gemm --batch-size 32

# 静默模式（仅显示错误）
primus -q benchmark gemm
```

---

## 🎛️ 全局参数

所有命令都支持以下全局参数：

| 参数 | 简写 | 说明 |
|------|------|------|
| `--version` | - | 显示版本信息 |
| `--verbose` | `-v` | 详细输出（DEBUG 级别） |
| `--quiet` | `-q` | 静默模式（仅错误） |
| `--config PATH` | - | 使用指定配置文件 |
| `--completion SHELL` | - | 生成 Shell 补全脚本 |

---

## 🔧 常用命令示例

### 训练命令

```bash
# 使用 Megatron 训练
primus train pretrain \
    --config config.yaml \
    --backend megatron \
    --num-gpus 8

# 使用 TorchTitan 训练
primus train pretrain \
    --config config.yaml \
    --backend torchtitan

# 调试模式（显示完整堆栈）
PRIMUS_DEBUG=1 primus train pretrain --config config.yaml
```

### 基准测试命令

```bash
# GEMM 基准测试
primus benchmark gemm \
    --batch-size 32 \
    --seq-len 2048 \
    --warmup 10 \
    --iterations 100

# Dense GEMM 基准测试
primus benchmark gemm-dense \
    --model-size 7B

# DeepSeek GEMM 基准测试
primus benchmark gemm-deepseek \
    --batch-size 16
```

### 环境检查命令

```bash
# 检查所有项
primus preflight --check-all

# 仅检查 GPU
primus preflight --check-gpu

# 仅检查 Python 环境
primus preflight --check-python

# 仅检查 ROCM
primus preflight --check-rocm

# 详细输出
primus preflight --check-all -v
```

---

## ⚙️ 配置文件

### 配置文件位置

Primus CLI 会自动搜索以下位置的配置文件（按优先级排序）：

1. 命令行指定：`--config /path/to/config.yaml`
2. `~/.primus/config.yaml`
3. `~/.config/primus/config.yaml`
4. `.primus.yaml`（当前目录）

### 配置文件示例

创建 `~/.primus/config.yaml`：

```yaml
# 全局配置
verbose: false
quiet: false

# 训练配置
backend: megatron
num_gpus: 8
batch_size: 32

# 基准测试配置
warmup: 10
iterations: 100
```

### 使用配置文件

```bash
# 自动使用默认配置文件
primus train pretrain

# 使用指定配置文件
primus --config my-config.yaml train pretrain

# 命令行参数会覆盖配置文件
primus --config config.yaml train pretrain --batch-size 64
```

---

## 🌟 Shell 自动补全

### Bash

```bash
# 1. 安装 argcomplete
pip install argcomplete

# 2. 生成补全脚本
primus --completion bash > ~/.primus-completion.bash

# 3. 添加到 .bashrc
echo 'source ~/.primus-completion.bash' >> ~/.bashrc

# 4. 重新加载
source ~/.bashrc

# 5. 使用 Tab 补全
primus <TAB>
```

### Zsh

```bash
# 1. 安装 argcomplete
pip install argcomplete

# 2. 在 .zshrc 中启用 bashcompinit
echo 'autoload -U bashcompinit && bashcompinit' >> ~/.zshrc

# 3. 生成补全脚本
primus --completion zsh > ~/.primus-completion.zsh

# 4. 添加到 .zshrc
echo 'source ~/.primus-completion.zsh' >> ~/.zshrc

# 5. 重新加载
source ~/.zshrc
```

### Fish

```bash
# 1. 安装 argcomplete
pip install argcomplete

# 2. 生成补全脚本
primus --completion fish > ~/.config/fish/completions/primus.fish

# 3. Fish 会自动加载
```

---

## 🔍 调试和性能分析

### 调试模式

显示完整的错误堆栈和详细日志：

```bash
# 方式 1：环境变量
PRIMUS_DEBUG=1 primus train pretrain --config config.yaml

# 方式 2：结合 verbose
primus -v train pretrain --config config.yaml
```

### 性能分析模式

显示命令执行时间：

```bash
# 启用性能分析
PRIMUS_PROFILE=1 primus benchmark gemm

# 输出：
# ... 命令执行 ...
# ⏱️  Command completed in 45.32s
```

### 日志级别

```bash
# INFO 级别（默认）
primus train pretrain --config config.yaml

# DEBUG 级别（最详细）
primus -v train pretrain --config config.yaml
primus --verbose train pretrain --config config.yaml

# ERROR 级别（静默）
primus -q train pretrain --config config.yaml
primus --quiet train pretrain --config config.yaml
```

---

## 🎯 最佳实践

### 1. 首次使用建议

```bash
# Step 1: 检查环境
primus preflight --check-all

# Step 2: 查看帮助
primus --help
primus train --help

# Step 3: 创建配置文件
mkdir -p ~/.primus
cat > ~/.primus/config.yaml << EOF
verbose: false
backend: megatron
num_gpus: 8
EOF

# Step 4: 运行测试
primus -v benchmark gemm
```

### 2. 开发和调试

```bash
# 启用详细日志和调试模式
PRIMUS_DEBUG=1 primus -v train pretrain --config config.yaml

# 检查特定组件
primus preflight --check-gpu -v
```

### 3. 生产环境

```bash
# 使用静默模式，仅记录错误
primus -q train pretrain --config config.yaml 2>> error.log

# 或使用重定向
primus train pretrain --config config.yaml > output.log 2>&1
```

### 4. 性能基准测试

```bash
# 启用性能分析
PRIMUS_PROFILE=1 primus benchmark gemm > results.txt

# 多次运行取平均
for i in {1..5}; do
    PRIMUS_PROFILE=1 primus benchmark gemm
done
```

---

## 📚 命令参考

### Train 命令

```bash
primus train pretrain [OPTIONS]

Options:
  --config PATH          配置文件路径
  --backend {megatron,torchtitan}
                        训练后端
  --num-gpus INT        GPU 数量
  --batch-size INT      批次大小
  ... （更多选项见 --help）
```

### Benchmark 命令

```bash
primus benchmark {gemm,gemm-dense,gemm-deepseek} [OPTIONS]

Suites:
  gemm                  基础 GEMM 基准测试
  gemm-dense           Dense GEMM 基准测试
  gemm-deepseek        DeepSeek GEMM 基准测试

Options:
  --batch-size INT     批次大小
  --seq-len INT        序列长度
  --warmup INT         预热迭代次数
  --iterations INT     测试迭代次数
  ... （更多选项见 --help）
```

### Preflight 命令

```bash
primus preflight [OPTIONS]

Options:
  --check-all          运行所有检查
  --check-python       检查 Python 环境
  --check-gpu          检查 GPU 配置
  --check-rocm         检查 ROCM/HIP
  --check-network      检查网络配置
  --check-filesystem   检查文件系统
```

---

## ❓ 常见问题

### Q: 如何查看当前版本？
```bash
primus --version
```

### Q: 如何获得更详细的输出？
```bash
primus -v <command>
# 或
primus --verbose <command>
```

### Q: 如何只显示错误信息？
```bash
primus -q <command>
# 或
primus --quiet <command>
```

### Q: 配置文件不生效？
```bash
# 检查配置文件是否被正确加载
primus -v train pretrain --config ~/.primus/config.yaml

# 查看日志中的 "Loaded configuration from ..." 信息
```

### Q: 如何调试命令执行问题？
```bash
# 启用调试模式
PRIMUS_DEBUG=1 primus -v <command>
```

### Q: Shell 补全不工作？
```bash
# 确保 argcomplete 已安装
pip install argcomplete

# 重新生成补全脚本
primus --completion bash > ~/.primus-completion.bash

# 重新加载 shell 配置
source ~/.bashrc  # 或 ~/.zshrc
```

---

## 🔗 相关文档

- [CLI 优化总结](./CLI_OPTIMIZATION_SUMMARY.md) - 详细的优化说明
- [可选依赖](./requirements-cli-optional.txt) - 可选功能依赖

---

## 💡 提示

1. **使用 `--help` 探索命令**：每个命令和子命令都有详细的帮助信息
2. **善用配置文件**：避免重复输入常用参数
3. **启用 Shell 补全**：大幅提升命令输入效率
4. **运行 preflight**：确保环境配置正确
5. **使用 `-v` 调试**：遇到问题时获取详细信息

---

**Happy Training! 🚀**
