# GPU 型号检测和配置系统

## 📋 系统概述

Primus 现在支持基于 GPU 型号的自动配置加载系统。系统会自动检测 AMD GPU 型号（如 MI300X、MI250X），并加载对应的优化配置。

---

## 🏗️ 系统架构

```
用户启动 Primus
    ↓
primus-cli-entrypoint.sh
    ↓
source primus-env.sh
    ↓
调用 detect_gpu_model.sh
    ↓
rocm-smi --showproductname
    ↓
提取型号（MI300X, MI250X, etc.）
    ↓
加载 envs/${GPU_MODEL}.sh
    ↓
应用 GPU 特定优化
    ↓
继续执行训练/基准测试
```

---

## 📁 文件结构

```
runner/helpers/
├── primus-env.sh              # 主环境配置脚本（已修改）
├── detect_gpu_model.sh        # GPU 型号检测脚本（新增）
└── envs/                      # GPU 配置目录（新增）
    ├── README.md              # 配置说明文档
    ├── MI300X.sh              # MI300X 专用配置
    ├── MI300A.sh              # MI300A 专用配置
    ├── MI250X.sh              # MI250X 专用配置
    └── default.sh             # 默认配置（降级选项）
```

---

## 🔧 核心组件

### 1. GPU 型号检测脚本

**文件**: `detect_gpu_model.sh`

**功能**:
- 调用 `rocm-smi --showproductname` 获取 GPU 信息
- 解析输出提取型号（MI300X, MI250X 等）
- 返回标准化的型号名称

**示例**:
```bash
$ bash detect_gpu_model.sh
MI300X
```

### 2. 主环境配置脚本

**文件**: `primus-env.sh`（已修改）

**新增逻辑**:
```bash
# 1. 检测 GPU 型号（如果未手动指定）
if [[ -z "$GPU_MODEL" ]]; then
    GPU_MODEL=$(bash "$SCRIPT_DIR/helpers/detect_gpu_model.sh")
fi

# 2. 确定配置文件路径
if [[ -f "$ENV_CONFIG_DIR/${GPU_MODEL}.sh" ]]; then
    GPU_CONFIG_FILE="$ENV_CONFIG_DIR/${GPU_MODEL}.sh"
elif [[ "$GPU_MODEL" =~ ^MI300 ]]; then
    GPU_CONFIG_FILE="$ENV_CONFIG_DIR/MI300X.sh"
# ... 其他匹配规则 ...
else
    GPU_CONFIG_FILE="$ENV_CONFIG_DIR/default.sh"
fi

# 3. 加载配置
source "$GPU_CONFIG_FILE"
```

### 3. GPU 特定配置文件

**目录**: `envs/`

每个配置文件包含该 GPU 型号的优化设置：
- HSA（Heterogeneous System Architecture）设置
- RCCL（ROCm Communication Collectives Library）参数
- Transformer Engine 优化
- 内存管理配置

---

## 🎯 支持的 GPU 型号

| GPU 型号 | 配置文件 | 架构 | 内存 | 关键特性 |
|---------|---------|------|------|---------|
| MI300X | `MI300X.sh` | CDNA 3 | 192GB HBM3 | FP8, 高内存 |
| MI300A | `MI300A.sh` | CDNA 3 | 128GB 统一 | APU, XNACK |
| MI250X | `MI250X.sh` | CDNA 2 | 128GB HBM2e | 双 GCD |
| 未知/其他 | `default.sh` | - | - | 保守设置 |

---

## 🚀 使用方式

### 自动模式（推荐）

系统会自动检测并加载配置：

```bash
# 直接启动，自动检测
./runner/primus-cli-entrypoint.sh -- train pretrain --config config.yaml

# 输出示例：
# [NODE-0(hostname)] Detected GPU model: MI300X
# [NODE-0(hostname)] Loading GPU configuration: .../envs/MI300X.sh
# [NODE-0(hostname)] ========== MI300X-specific optimizations ==========
```

### 手动指定模式

如果自动检测失败或需要测试特定配置：

```bash
# 方式 1: 环境变量
export GPU_MODEL=MI300X
./runner/primus-cli-entrypoint.sh -- train pretrain

# 方式 2: 内联指定
GPU_MODEL=MI250X ./runner/primus-cli-entrypoint.sh -- benchmark gemm

# 方式 3: 在 entrypoint.sh 中使用 --env
./runner/primus-cli-entrypoint.sh --env GPU_MODEL=MI300X -- train pretrain
```

---

## 📊 配置差异示例

### MI300X vs MI250X 关键配置对比

```bash
# MI300X (高内存, CDNA 3)
HSA_XNACK=0                                    # 禁用
GPU_MAX_HEAP_SIZE=100                          # 100% 内存
NVTE_USE_CAST_TRANSPOSE_TRITON=1               # Triton 优化
PATCH_TE_FLASH_ATTN=0                          # 无需补丁

# MI250X (双 GCD, CDNA 2)
HSA_XNACK=0                                    # 禁用
GPU_MAX_HEAP_SIZE=90                           # 保守 90%
NVTE_USE_CAST_TRANSPOSE_TRITON=0               # 禁用
NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=1   # 优化版本
PATCH_TE_FLASH_ATTN=1                          # 需要补丁
```

---

## ➕ 添加新 GPU 型号

### 快速添加流程

```bash
# 1. 创建配置文件
cd runner/helpers/envs
cp MI300X.sh MI355.sh

# 2. 编辑配置
vim MI355.sh
# 根据 MI355 特性调整参数

# 3. 测试
GPU_MODEL=MI355 ./runner/primus-cli-entrypoint.sh -- preflight --check-all

# 4. 验证日志
# 应该看到: "Loading GPU configuration: .../envs/MI355.sh"
```

### 配置文件模板

```bash
#!/bin/bash
###############################################################################
# AMD MI{XXX} GPU-specific optimizations
###############################################################################

LOG_INFO_RANK0 "Loading MI{XXX}-specific optimizations..."

# ----------------- GPU 特定设置 -----------------
export HSA_ENABLE_SDMA=1
export HSA_NO_SCRATCH_RECLAIM=0
export HSA_XNACK=0  # 根据 GPU 特性设置

# RCCL 设置
export RCCL_MSCCL_ENABLE=0
export RCCL_MSCCLPP_THRESHOLD=$((1*1024*1024*1024))

# 性能调优
export GPU_MAX_HW_QUEUES=2
export GPU_MAX_HEAP_SIZE=100  # 根据实际内存调整

# Transformer Engine
export NVTE_USE_CAST_TRANSPOSE_TRITON=1  # 根据架构调整
export NVTE_CK_USES_BWD_V3=0

# 记录配置
log_exported_vars "MI{XXX}-specific optimizations" \
    HSA_ENABLE_SDMA HSA_XNACK GPU_MAX_HEAP_SIZE \
    NVTE_USE_CAST_TRANSPOSE_TRITON
```

---

## 🔍 调试和验证

### 1. 验证 GPU 检测

```bash
# 查看 rocm-smi 输出
rocm-smi --showproductname

# 运行检测脚本
bash runner/helpers/detect_gpu_model.sh

# 预期输出: MI300X
```

### 2. 验证配置加载

```bash
# 设置为 rank 0 以看到日志
export NODE_RANK=0

# 加载环境
source runner/helpers/primus-env.sh

# 查看日志输出
# [NODE-0(hostname)] Detected GPU model: MI300X
# [NODE-0(hostname)] Loading GPU configuration: .../envs/MI300X.sh
```

### 3. 验证环境变量

```bash
# 加载环境后检查
source runner/helpers/primus-env.sh
env | grep -E "HSA_|RCCL_|NVTE_|GPU_MAX"

# 预期输出:
# HSA_ENABLE_SDMA=1
# HSA_NO_SCRATCH_RECLAIM=0
# RCCL_MSCCL_ENABLE=0
# ...
```

---

## 🐛 常见问题

### Q1: 检测到错误的 GPU 型号

**问题**: 系统检测到 `MI300` 但实际是 `MI300A`

**解决**:
```bash
# 手动指定准确型号
export GPU_MODEL=MI300A
./runner/primus-cli-entrypoint.sh -- train pretrain
```

### Q2: 找不到配置文件

**错误信息**:
```
[ERROR] GPU configuration file not found: .../envs/MI355.sh
```

**解决**:
```bash
# 选项 1: 创建配置文件
cp runner/helpers/envs/MI300X.sh runner/helpers/envs/MI355.sh

# 选项 2: 使用已有配置
export GPU_MODEL=MI300X

# 选项 3: 使用默认配置
export GPU_MODEL=default
```

### Q3: rocm-smi 不可用

**问题**: `rocm-smi: command not found`

**解决**:
```bash
# 检查 ROCm 安装
which rocm-smi

# 添加到 PATH
export PATH=/opt/rocm/bin:$PATH

# 或手动指定型号
export GPU_MODEL=MI300X
```

### Q4: 配置不生效

**检查步骤**:
```bash
# 1. 确认文件存在
ls -la runner/helpers/envs/

# 2. 确认文件可执行
chmod +x runner/helpers/envs/*.sh

# 3. 手动加载测试
source runner/helpers/envs/MI300X.sh
echo $HSA_ENABLE_SDMA  # 应该输出 1

# 4. 检查日志
export NODE_RANK=0
source runner/helpers/primus-env.sh 2>&1 | grep -A 10 "Loading GPU"
```

---

## 🎨 自定义配置示例

### 场景 1: 为特定工作负载调优

```bash
# 创建自定义配置
cp envs/MI300X.sh envs/MI300X-largemodel.sh

# 编辑配置
# envs/MI300X-largemodel.sh
export GPU_MAX_HEAP_SIZE=95  # 为大模型保留更多内存
export HSA_NO_SCRATCH_RECLAIM=1  # 启用回收以支持更大的模型

# 使用自定义配置
GPU_MODEL=MI300X-largemodel ./runner/primus-cli-entrypoint.sh -- train pretrain
```

### 场景 2: 调试配置

```bash
# 创建调试配置
cp envs/MI300X.sh envs/MI300X-debug.sh

# 启用所有调试选项
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2
export NCCL_DEBUG=INFO
export HSA_ENABLE_DEBUG=1

# 使用调试配置
GPU_MODEL=MI300X-debug ./runner/primus-cli-entrypoint.sh -- train pretrain
```

---

## 📈 性能调优建议

### 1. 内存管理

```bash
# 大模型: 使用更保守的内存设置
GPU_MAX_HEAP_SIZE=85

# 小模型: 可以使用全部内存
GPU_MAX_HEAP_SIZE=100
```

### 2. 通信优化

```bash
# InfiniBand 环境
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1

# Ethernet 环境
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
```

### 3. 数值精度

```bash
# FP8 训练 (MI300X)
export NVTE_USE_CAST_TRANSPOSE_TRITON=1

# FP16/BF16 训练
export NVTE_USE_CAST_TRANSPOSE_TRITON=0
```

---

## 🔄 升级和迁移

### 从旧配置迁移

如果你之前在 `primus-env.sh` 中有自定义配置：

```bash
# 1. 识别你的 GPU 型号
rocm-smi --showproductname

# 2. 编辑对应的配置文件
vim runner/helpers/envs/MI300X.sh

# 3. 添加你的自定义设置
# 保持现有的优化，添加你的自定义变量

# 4. 测试新配置
GPU_MODEL=MI300X ./runner/primus-cli-entrypoint.sh -- preflight --check-all
```

---

## 📚 相关文档

- [envs/README.md](envs/README.md) - 详细的配置说明
- [AMD ROCm 文档](https://rocm.docs.amd.com/)
- [RCCL 调优指南](https://github.com/ROCmSoftwarePlatform/rccl)

---

## 🎯 最佳实践总结

1. ✅ **使用自动检测**: 大多数情况下让系统自动检测即可
2. ✅ **测试配置**: 添加新配置后运行 `preflight --check-all`
3. ✅ **记录性能**: 为每个配置记录性能基线
4. ✅ **版本控制**: 在配置文件中记录版本和测试信息
5. ✅ **逐步调优**: 从默认配置开始，逐步优化
6. ✅ **文档更新**: 添加新型号时更新 README

---

**创建日期**: 2025-11-05
**版本**: 1.0
**维护者**: Primus Team
