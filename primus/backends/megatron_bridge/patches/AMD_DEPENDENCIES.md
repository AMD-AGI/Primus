# Megatron-Bridge AMD GPU 依赖安装指南

从 `pyproject.toml` 提取的依赖，标注 AMD GPU 上的安装策略。

## ✅ 必需的通用依赖（AMD + NVIDIA 通用）

```bash
# Python 环境要求
# requires-python = ">=3.10"

# 核心依赖 - 直接安装
pip install "transformers<5.0.0"
pip install datasets
pip install accelerate
pip install "omegaconf>=2.3.0"
pip install "tensorboard>=2.19.0"
pip install typing-extensions
pip install rich
pip install "wandb>=0.19.10"
pip install "six>=1.17.0"
pip install "regex>=2024.11.6"
pip install "pyyaml>=6.0.2"
pip install "tqdm>=4.67.1"
pip install "hydra-core>1.3,<=1.3.2"

# 模型特定依赖
pip install qwen-vl-utils
pip install timm
pip install "open-clip-torch>=3.2.0"
```

## 🔧 需要特殊处理的依赖（AMD 版本）

### 1. Megatron-Core（AMD 定制版本）

```bash
# 原始依赖:
# megatron-core[dev,mlm]>=0.15.0a0,<0.17.0

# AMD 安装方式:
# 使用 third_party/Megatron-LM (AMD 适配版本)
cd ${PRIMUS_ROOT}/third_party/Megatron-LM
pip install -e ".[dev,mlm]"
```

**位置**: `third_party/Megatron-LM/`（已包含在 Primus 项目中）

### 2. Transformer-Engine（AMD ROCm 版本）

```bash
# 原始依赖:
# transformer-engine[pytorch]>=2.10.0a0,<2.12.0

# AMD 安装方式:
# 安装 ROCm 版本的 transformer-engine-torch
pip install transformer-engine-torch  # ROCm 特定版本
```

**注意**: 
- NVIDIA 版本: `transformer-engine`
- AMD 版本: `transformer-engine-torch`
- 功能相同，但针对不同的硬件后端

### 3. Flash Attention 相关（可选，性能优化）

```bash
# 原始依赖:
# flash-linear-attention

# AMD 安装方式:
# 检查是否有 ROCm 支持，或从源码编译
# 如果不支持，可以跳过（会使用 fallback 实现）
pip install flash-linear-attention  # 尝试安装，如果失败则跳过
```

### 4. Mamba 相关（可选，特定模型架构）

```bash
# 原始依赖:
# mamba-ssm
# causal-conv1d

# AMD 安装方式:
# 这些包主要用于 Mamba 架构模型
# 如果不使用 Mamba 模型，可以跳过
# pip install mamba-ssm        # 如果需要 Mamba 模型
# pip install causal-conv1d    # 如果需要 Mamba 模型
```

**注意**: Mamba 模型不是必需的，只有使用特定模型时才需要。

## ❌ NVIDIA 特定依赖（AMD 上跳过）

```bash
# ❌ 跳过 - NVIDIA 特定
# nvidia-resiliency-ext      # NVIDIA 容错扩展
# nvidia-modelopt[torch]     # NVIDIA 模型优化工具
# nv-grouped-gemm            # NVIDIA GEMM 优化
```

### nvidia-resiliency-ext

**原始依赖**:
```toml
nvidia-resiliency-ext
```

**AMD 处理**:
- ❌ 跳过安装
- 功能: 提供容错和检查点恢复
- AMD 替代: 使用 PyTorch 原生的分布式容错机制

### nvidia-modelopt

**原始依赖**:
```toml
nvidia-modelopt[torch]>=0.37.0
```

**AMD 处理**:
- ❌ 跳过安装（已验证有兼容性问题）
- 功能: 模型量化、剪枝、优化
- 问题: 依赖 `torch.onnx._type_utils`（新版 PyTorch 中已移除）
- AMD 替代: 使用 PyTorch 原生量化或其他工具

## 📦 完整的 AMD 安装脚本

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "================================================"
echo "Installing Megatron-Bridge Dependencies for AMD"
echo "================================================"

# 1. 基础依赖
echo "[1/6] Installing core dependencies..."
pip install -U pip setuptools wheel

pip install \
    "transformers<5.0.0" \
    datasets \
    accelerate \
    "omegaconf>=2.3.0" \
    "tensorboard>=2.19.0" \
    typing-extensions \
    rich \
    "wandb>=0.19.10" \
    "six>=1.17.0" \
    "regex>=2024.11.6" \
    "pyyaml>=6.0.2" \
    "tqdm>=4.67.1" \
    "hydra-core>1.3,<=1.3.2"

# 2. 模型特定依赖
echo "[2/6] Installing model-specific dependencies..."
pip install \
    qwen-vl-utils \
    timm \
    "open-clip-torch>=3.2.0"

# 3. Transformer-Engine (AMD ROCm 版本)
echo "[3/6] Installing Transformer-Engine for ROCm..."
pip install transformer-engine-torch

# 4. Megatron-Core (AMD 适配版本)
echo "[4/6] Installing Megatron-Core (AMD version)..."
cd "${PRIMUS_ROOT}/third_party/Megatron-LM"
pip install -e ".[dev,mlm]"

# 5. 可选：Flash Attention（如果支持）
echo "[5/6] Installing optional dependencies..."
pip install flash-linear-attention || echo "[WARNING] flash-linear-attention not available, using fallback"

# 6. Megatron-Bridge (从源码安装)
echo "[6/6] Installing Megatron-Bridge..."
cd "${PRIMUS_ROOT}/third_party/Megatron-Bridge"
pip install -e .

echo "================================================"
echo "✅ Installation complete!"
echo "================================================"
```

## 🔍 依赖对比表

| 依赖包 | NVIDIA 版本 | AMD 版本 | 状态 | 说明 |
|--------|------------|----------|------|------|
| **核心框架** |
| PyTorch | `torch` | `torch+rocm` | ✅ 已安装 | AMD 使用 ROCm 版本 |
| Transformers | `<5.0.0` | `<5.0.0` | ✅ 通用 | 无差异 |
| Megatron-Core | `>=0.15.0` | AMD 分支 | ✅ 适配 | 使用 AMD 定制版本 |
| **加速库** |
| Transformer-Engine | `transformer-engine` | `transformer-engine-torch` | ✅ 适配 | ROCm 专用版本 |
| Flash Attention | `flash-linear-attention` | 同左 | ⚠️ 可选 | 可能需要源码编译 |
| Mamba | `mamba-ssm` | 同左 | ⚠️ 可选 | 仅特定模型需要 |
| **NVIDIA 特定** |
| nvidia-resiliency-ext | ✅ 安装 | ❌ 跳过 | 🔄 替代 | 使用 PyTorch 原生功能 |
| nvidia-modelopt | ✅ 安装 | ❌ 跳过 | ❌ 不兼容 | 有 ONNX API 问题 |
| nv-grouped-gemm | ✅ 安装 | ❌ 跳过 | 🔄 替代 | ROCm 有其他优化 |
| **工具库** |
| omegaconf | `>=2.3.0` | 同左 | ✅ 通用 | 配置管理 |
| hydra-core | `>1.3,<=1.3.2` | 同左 | ✅ 通用 | 配置管理 |
| wandb | `>=0.19.10` | 同左 | ✅ 通用 | 实验跟踪 |
| tensorboard | `>=2.19.0` | 同左 | ✅ 通用 | 可视化 |

## 📝 注意事项

### 1. ROCm 版本兼容性

确保你的 PyTorch 是为 ROCm 编译的版本：

```bash
python -c "import torch; print(torch.version.hip)"  # 应该输出 ROCm 版本号
```

### 2. 环境变量

AMD GPU 可能需要设置特定的环境变量：

```bash
export HSA_NO_SCRATCH_RECLAIM=1
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
```

### 3. 不支持的功能

以下功能在 AMD 上可能不可用或有限制：

- ❌ NVIDIA-specific quantization (nvidia-modelopt)
- ⚠️ 某些自定义 CUDA kernels 可能需要替换为 ROCm 版本
- ⚠️ Flash Attention 可能有性能差异

### 4. 性能优化

AMD GPU 上的最佳实践：

1. **使用 transformer-engine-torch**: 提供 ROCm 优化的算子
2. **启用 Flash Attention**: 如果可用，会显著提升性能
3. **调整 batch size**: AMD GPU 内存特性可能需要不同的配置
4. **使用 BF16**: AMD GPUs 通常对 BF16 有良好支持

## 🚀 快速开始

最小化安装（仅核心功能）：

```bash
# 1. 核心依赖
pip install transformers datasets accelerate omegaconf tensorboard

# 2. AMD Transformer-Engine
pip install transformer-engine-torch

# 3. Megatron-Core (AMD)
cd third_party/Megatron-LM && pip install -e .

# 4. Megatron-Bridge
cd third_party/Megatron-Bridge && pip install -e .
```

完整安装（包含所有功能）：

```bash
# 运行完整安装脚本
bash primus/backends/megatron_bridge/patches/install_amd_deps.sh
```

## 🐛 故障排除

### 问题 1: `ImportError: cannot import name '_type_utils' from 'torch.onnx'`

**原因**: nvidia-modelopt 不兼容  
**解决**: 跳过 nvidia-modelopt 安装（已处理）

### 问题 2: `ImportError: cannot import name 'Glm4vMoeForConditionalGeneration'`

**原因**: Transformers 版本不支持某些模型  
**解决**: 应用 GLM-4V import patch（可选模型）

### 问题 3: Transformer-Engine 编译失败

**原因**: ROCm 环境配置问题  
**解决**: 
```bash
# 确保 ROCm 正确安装
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
pip install transformer-engine-torch --no-cache-dir
```

## 📚 参考资料

- [Megatron-Bridge GitHub](https://github.com/NVIDIA/Megatron-Bridge)
- [Transformer-Engine ROCm](https://github.com/ROCm/TransformerEngine)
- [PyTorch ROCm](https://pytorch.org/get-started/locally/)
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
